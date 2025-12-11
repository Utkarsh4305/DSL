"""
UER Specification Loader Module

Loads and parses UER specification files (YAML/JSON).
Provides validation for specification format with v0.2 strict validation.
"""

import yaml
import json
from typing import Dict, Any, Union, Optional
from pathlib import Path
import logging

from .spec_validator import SpecValidator, SpecMissingFieldError, SpecTypeError, SpecBoundsError, SpecEnumError
from .spec_validator import AlignmentMetadataError, SpecStructureError, SpecConsistencyError
from .utils.errors import UERConfigurationError

logger = logging.getLogger(__name__)


class UERSpecLoader:
    """Loads UER specification files."""

    def __init__(self):
        self._spec_cache = {}

    def load_spec(self, spec_file: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a UER specification from YAML or JSON file.

        Args:
            spec_file: Path to specification file

        Returns:
            Dict containing the UER specification

        Raises:
            FileNotFoundError: If spec file doesn't exist
            ValueError: If spec format is invalid
        """
        spec_path = Path(spec_file)

        if not spec_path.exists():
            raise FileNotFoundError(f"UER spec file not found: {spec_path}")

        # Cache check
        if str(spec_path) in self._spec_cache:
            return self._spec_cache[str(spec_path)]

        # Load based on extension
        if spec_path.suffix.lower() == '.yaml':
            with open(spec_path, 'r', encoding='utf-8') as f:
                spec = yaml.safe_load(f)
        elif spec_path.suffix.lower() == '.json':
            with open(spec_path, 'r', encoding='utf-8') as f:
                spec = json.load(f)
        else:
            raise ValueError(f"Unsupported spec file format: {spec_path.suffix}")

        # Validate spec structure
        self._validate_spec_structure(spec)

        # Cache and return
        self._spec_cache[str(spec_path)] = spec
        return spec

    def _validate_spec_structure(self, spec: Dict[str, Any]) -> None:
        """Validate UER specification structure with backwards compatibility."""
        version = spec.get('uer_version', 'unknown')

        # Determine spec version and apply appropriate validation
        if self._is_v02_spec(version):
            # Strict v0.2 validation using SpecValidator
            self._validate_v02_spec(spec)
        else:
            # Backwards compatible v0.1 validation
            self._validate_v01_spec(spec)

    def _is_v02_spec(self, version: str) -> bool:
        """Check if spec version requires v0.2 validation."""
        if not isinstance(version, str):
            return False
        try:
            # Parse version like "0.2.0" -> (0, 2, 0)
            parts = version.split('.')
            major, minor = int(parts[0]), int(parts[1])
            return major > 0 or (major == 0 and minor >= 2)
        except (ValueError, IndexError):
            # If version can't be parsed, assume v0.1
            return False

    def _validate_v01_spec(self, spec: Dict[str, Any]) -> None:
        """Backwards compatible validation for v0.1 specs."""
        logger.info("Validating v0.1 spec - consider upgrading to v0.2 for enhanced security")

        required_fields = [
            'uer_version',
            'vector_dimension',
            'normalization_rules',
            'metric',
            'dtype',
            'geometry'
        ]

        missing_fields = [field for field in required_fields if field not in spec]

        if missing_fields:
            raise ValueError(f"UER spec missing required fields: {missing_fields}")

        # Version check
        if not isinstance(spec['uer_version'], str):
            raise ValueError("uer_version must be a string")

        # Dimension check
        if not isinstance(spec['vector_dimension'], int) or spec['vector_dimension'] <= 0:
            raise ValueError("vector_dimension must be a positive integer")

    def _validate_v02_spec(self, spec: Dict[str, Any]) -> None:
        """Strict validation for v0.2 specs using SpecValidator."""
        try:
            validator = SpecValidator()
            validator.validate_spec(spec)
            logger.debug("v0.2 spec validation passed")
        except UERConfigurationError as e:
            # Specification errors are critical - always raise
            logger.error(f"v0.2 spec validation failed: {e}")
            raise e
        except Exception as e:
            # Any other validation error
            logger.error(f"Unexpected spec validation error: {e}")
            raise UERConfigurationError(f"Spec validation failed: {e}")

    def validate_spec_for_writing(self, spec: Dict[str, Any]) -> None:
        """
        Validate spec before writing/modifying.

        This is stricter than loading validation and always uses v0.2 rules.
        """
        try:
            validator = SpecValidator()
            validator.validate_spec(spec)
        except Exception as e:
            raise UERConfigurationError(f"Cannot write invalid spec: {e}")


def load_uer_spec(spec_file: Union[str, Path]) -> Dict[str, Any]:
    """Convenience function to load UER spec."""
    loader = UERSpecLoader()
    return loader.load_spec(spec_file)
