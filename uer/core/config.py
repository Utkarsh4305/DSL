"""
UER Configuration Management

Production-ready configuration loading with validation, caching, and provider-aware defaults.
Handles the issues with numeric type conversion and configuration flexibility.
"""

import yaml
import json
from typing import Dict, Any, Union, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class UERConfig:
    """
    Enhanced UER configuration with validation and provider-aware settings.

    Handles the critical issues:
    - Proper numeric type conversion from YAML
    - Provider-specific defaults
    - Validation rules
    - Backward compatibility
    """

    def __init__(self, spec: Dict[str, Any]):
        """
        Initialize UER configuration with enhanced validation.

        Args:
            spec: UER specification dictionary

        Raises:
            ValueError: If specification is invalid or missing required fields
        """
        self.spec = self._validate_and_normalize_spec(spec)
        self._cache_numeric_values()

    def _validate_and_normalize_spec(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize specification with intelligent error handling."""
        # Required fields with informative error messages
        required_fields = {
            'uer_version': 'UER specification version (e.g., "0.2.0")',
            'vector_dimension': 'Embedding dimension as integer (e.g., 768)',
            'normalization_rules': 'Normalization configuration with method and parameters',
            'metric': 'Similarity metric (cosine, dot, euclidean)',
            'dtype': 'Data type (float32, float16, int8, etc.)',
            'geometry': 'Embedding geometry (spherical, euclidean, hyperbolic)'
        }

        missing_fields = []
        invalid_fields = []

        for field, description in required_fields.items():
            if field not in spec:
                missing_fields.append(f"{field}: {description}")
            else:
                # Validate field types where possible
                if field == 'uer_version' and not isinstance(spec[field], str):
                    invalid_fields.append(f"{field} must be a string, got {type(spec[field])}")
                elif field == 'vector_dimension' and not isinstance(spec[field], int):
                    invalid_fields.append(f"{field} must be an integer, got {type(spec[field])}")

        if missing_fields:
            raise ValueError(
                "UER specification missing required fields:\n" +
                "\n".join(f"  - {field}" for field in missing_fields) +
                "\n\nSuggestion: Add these fields to your YAML specification file."
            )

        if invalid_fields:
            raise ValueError(
                "UER specification has invalid field types:\n" +
                "\n".join(f"  - {field}" for field in invalid_fields)
            )

        # Normalize and validate optional sections
        normalized = spec.copy()

        # Ensure normalization rules have required structure
        norm_rules = spec.get('normalization_rules', {})
        if isinstance(norm_rules, dict):
            normalized['normalization_rules'] = self._normalize_normalization_rules(norm_rules)
        else:
            raise ValueError("normalization_rules must be a dictionary with 'method' key")

        # Ensure validation rules exist
        if 'validation_rules' not in normalized:
            normalized['validation_rules'] = self._get_default_validation_rules()

        # Validate alignment metadata
        if 'alignment_metadata' not in normalized:
            normalized['alignment_metadata'] = {
                'mapping_type': 'identity',
                'alignment_version': spec.get('uer_version', '0.0.0'),
                'mapping_file': None
            }

        return normalized

    def _normalize_normalization_rules(self, rules: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize normalization rules with proper type conversion."""
        normalized = rules.copy()

        # Ensure method is valid
        method = normalized.get('method', 'l2')
        valid_methods = ['l2', 'l1', 'none', 'max']

        if method not in valid_methods:
            logger.warning(f"Unknown normalization method '{method}', defaulting to 'l2'")
            method = 'l2'
            normalized['method'] = method

        # Convert numeric fields properly (fix YAML string issues)
        if 'epsilon' in normalized:
            try:
                normalized['epsilon'] = float(normalized['epsilon'])
            except (ValueError, TypeError):
                logger.warning(f"Invalid epsilon value {normalized['epsilon']}, using 1e-12")
                normalized['epsilon'] = 1e-12

        # Set defaults based on method
        if method == 'l2' and 'epsilon' not in normalized:
            normalized['epsilon'] = 1e-12

        return normalized

    def _get_default_validation_rules(self) -> Dict[str, Any]:
        """Get sensible default validation rules."""
        return {
            'dimension_tolerance': 0.0,
            'norm_tolerance': 1e-6,
            'dtype_strict': True,
            'nan_check': True,
            'inf_check': True,
            'anisotropy_check': False,  # New: anisotropy validation
            'distribution_check': False,  # New: mean/std sanity checks
            'semantic_preservation_check': False,  # New: k-NN preservation
            'auto_repair': False  # New: attempt automatic fixes
        }

    def _cache_numeric_values(self) -> None:
        """Pre-cache frequently used numeric values to avoid repeated conversions."""
        # Convert numeric strings from YAML to proper types
        self.vector_dim = int(self.spec['vector_dimension'])
        self.norm_eps = float(self.spec.get('normalization_rules', {}).get('epsilon', 1e-12))
        self.norm_tolerance = float(self.spec.get('validation_rules', {}).get('norm_tolerance', 1e-6))
        self.dim_tolerance = float(self.spec.get('validation_rules', {}).get('dimension_tolerance', 0.0))

    def get_target_dtype(self):
        """Get the target numpy dtype for embeddings."""
        import numpy as np

        dtype_map = {
            'float32': np.float32,
            'float16': np.float16,
            'bf16': np.dtype('float16'),  # Placeholder for bfloat16
            'float64': np.float64,
            'int8': np.int8,
            'int16': np.int16,
            'int32': np.int32,
            'uint8': np.uint8
        }

        dtype_str = self.spec.get('dtype', 'float32')
        if dtype_str in dtype_map:
            return dtype_map[dtype_str]
        else:
            logger.warning(f"Unknown dtype '{dtype_str}', using float32")
            return np.float32

    def to_dict(self) -> Dict[str, Any]:
        """Return the normalized specification as a dictionary."""
        return self.spec.copy()

    def get_provider_suggestions(self) -> List[str]:
        """Suggest which providers might work well with this configuration."""
        suggestions = []

        # Suggest based on dimension
        if self.vector_dim == 768:
            suggestions.extend(['BERT-base', 'RoBERTa-base'])
        elif self.vector_dim == 1536:
            suggestions.extend(['OpenAI Ada-002', 'SBERT-large'])
        elif self.vector_dim == 4096:
            suggestions.extend(['OpenAI Text-Embedding-3-Large'])

        if self.spec.get('normalization_rules', {}).get('method') == 'l2':
            suggestions.extend(['Most sentence transformers'])

        return list(set(suggestions))  # Remove duplicates


class UERSpecLoader:
    """Enhanced specification loader with caching and error recovery."""

    def __init__(self):
        self._spec_cache = {}

    def load_spec(self, spec_file: Union[str, Path]) -> UERConfig:
        """
        Load a UER specification from YAML or JSON file.

        Args:
            spec_file: Path to specification file

        Returns:
            UERConfig instance with validated specification

        Raises:
            FileNotFoundError: If spec file doesn't exist
            ValueError: If spec format is invalid
        """
        spec_path = Path(spec_file)

        if not spec_path.exists():
            # Suggest common locations
            suggestions = [
                "specs/uer_v0.1.yaml",
                "specs/uer_v0.2.yaml",
                "uer_spec.yaml"
            ]
            raise FileNotFoundError(
                f"UER spec file not found: {spec_path}\n" +
                "Suggestion: Check these common locations:\n" +
                "\n".join(f"  - {loc}" for loc in suggestions)
            )

        # Check cache
        cache_key = str(spec_path.resolve())
        if cache_key in self._spec_cache:
            return self._spec_cache[cache_key]

        # Load based on extension
        try:
            if spec_path.suffix.lower() in ['.yaml', '.yml']:
                with open(spec_path, 'r', encoding='utf-8') as f:
                    spec = yaml.safe_load(f)
            elif spec_path.suffix.lower() == '.json':
                with open(spec_path, 'r', encoding='utf-8') as f:
                    spec = json.load(f)
            else:
                raise ValueError(f"Unsupported spec file format: {spec_path.suffix}")

            if spec is None:
                raise ValueError(f"Spec file is empty or invalid: {spec_path}")

            # Create UERConfig (this validates the spec)
            config = UERConfig(spec)

            # Cache the config
            self._spec_cache[cache_key] = config

            logger.info(f"Successfully loaded UER spec v{config.spec['uer_version']} from {spec_path}")
            return config

        except Exception as e:
            raise ValueError(
                f"Failed to load UER specification from {spec_path}: {e}\n" +
                "Suggestion: Ensure the file is valid YAML/JSON with all required fields.\n" +
                "See examples in specs/ directory."
            ) from e

    def clear_cache(self) -> None:
        """Clear the specification cache."""
        self._spec_cache.clear()


def load_uer_spec(spec_file: Union[str, Path]) -> UERConfig:
    """
    Convenience function to load UER spec.

    Args:
        spec_file: Path to UER specification file

    Returns:
        UERConfig instance
    """
    loader = UERSpecLoader()
    return loader.load_spec(spec_file)
