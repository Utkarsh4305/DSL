"""
UER Specification Validator Module

Strict validation of UER specification files with detailed error reporting.
Enforces structure, types, ranges, and required fields for v0.2.
"""

import os
import re
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path

from .utils.errors import UERConfigurationError


class SpecValidator:
    """
    Validates UER specification files against strict v0.2 requirements.

    Ensures all required fields are present with correct types,
    validates enums, numerical bounds, and alignment metadata.
    """

    # Strict enum definitions for v0.2
    ALLOWED_METRICS = ["cosine", "dot_product", "euclidean", "manhattan"]
    ALLOWED_GEOMETRIES = ["spherical", "hyperspherical", "euclidean"]
    ALLOWED_DTYPES = ["float16", "float32", "float64", "int8", "uint8"]
    ALLOWED_NORM_METHODS = ["l2", "l1", "none"]
    ALLOWED_MAPPING_TYPES = ["identity", "linear", "procrustes", "fine_tuned"]
    ALLOWED_DOMAINS = ["general", "scientific", "legal", "medical", "multilingual"]
    ALLOWED_CORPUS_SIZES = ["small", "medium", "large_scale", "massive"]
    ALLOWED_TRAINING_METHODS = ["contrastive", "supervised", "unsupervised", "generative"]

    # Required fields with types for v0.2
    REQUIRED_FIELDS = {
        'uer_version': str,
        'vector_dimension': int,
        'normalization_rules': dict,
        'metric': str,
        'dtype': str,
        'geometry': str,
        'version_compatibility': dict,
        'numeric_precision_guarantees': dict,
        'embedding_statistics': dict,
        'robustness_profiles': dict
    }

    def __init__(self):
        """Initialize spec validator with strict validation rules."""
        pass

    def validate_spec(self, spec: Dict[str, Any]) -> None:
        """
        Validate complete UER specification.

        Args:
            spec: Parsed specification dictionary

        Raises:
            SpecMissingFieldError: Required field missing
            SpecTypeError: Field has wrong type
            SpecBoundsError: Numerical value out of bounds
            SpecEnumError: Value not in allowed enum
            AlignmentMetadataError: Invalid alignment configuration
        """

        # Basic structure validation
        self._validate_required_fields(spec)
        self._validate_field_types(spec)

        # Content validation
        self._validate_numerical_constraints(spec)
        self._validate_enums(spec)
        self._validate_alignment_metadata(spec)
        self._validate_nested_structures(spec)

        # Semantic consistency checks
        self._validate_semantic_consistency(spec)

    def _validate_required_fields(self, spec: Dict[str, Any]) -> None:
        """Ensure all required fields are present."""
        missing_fields = []
        for field in self.REQUIRED_FIELDS.keys():
            if field not in spec:
                missing_fields.append(field)

        if missing_fields:
            raise SpecMissingFieldError(f"Missing required fields: {missing_fields}", missing_fields)

    def _validate_field_types(self, spec: Dict[str, Any]) -> None:
        """Validate types of all fields."""
        type_errors = []

        for field, expected_type in self.REQUIRED_FIELDS.items():
            if field in spec:
                if not isinstance(spec[field], expected_type):
                    type_errors.append(f"{field}: expected {expected_type.__name__}, got {type(spec[field]).__name__}")

        # Additional type validations
        if 'vector_dimension' in spec and not isinstance(spec['vector_dimension'], int):
            type_errors.append("vector_dimension: must be positive integer")

        if 'normalization_rules' in spec and 'epsilon' in spec['normalization_rules']:
            epsilon = spec['normalization_rules']['epsilon']
            if not isinstance(epsilon, (int, float)):
                type_errors.append("normalization_rules.epsilon: must be numeric")

        if type_errors:
            raise SpecTypeError(f"Field type errors: {type_errors}", type_errors)

    def _validate_numerical_constraints(self, spec: Dict[str, Any]) -> None:
        """Validate numerical bounds and constraints."""
        bounds_errors = []

        # Vector dimension bounds
        if 'vector_dimension' in spec:
            dim = spec['vector_dimension']
            if not isinstance(dim, int) or dim <= 0:
                bounds_errors.append(f"vector_dimension: must be positive integer, got {dim}")
            else:
                min_dim = spec.get('validation_rules', {}).get('dimension_min', 1)
                max_dim = spec.get('validation_rules', {}).get('dimension_max', 10000)
                if dim < min_dim or dim > max_dim:
                    bounds_errors.append(f"vector_dimension: {dim} outside bounds [{min_dim}, {max_dim}]")

        # Normalization epsilon bounds
        if 'normalization_rules' in spec and 'epsilon' in spec['normalization_rules']:
            epsilon = spec['normalization_rules']['epsilon']
            min_eps = spec.get('validation_rules', {}).get('epsilon_min', 0)
            max_eps = spec.get('validation_rules', {}).get('epsilon_max', 1e-3)
            if epsilon < min_eps or epsilon > max_eps:
                bounds_errors.append(f"normalization_rules.epsilon: {epsilon} outside bounds [{min_eps}, {max_eps}]")

        # Version compatibility numerical checks
        if 'version_compatibility' in spec:
            ver_compat = spec['version_compatibility']
            if 'migration_required' in ver_compat:
                if not isinstance(ver_compat['migration_required'], bool):
                    bounds_errors.append("version_compatibility.migration_required: must be boolean")

        if bounds_errors:
            raise SpecBoundsError(f"Numerical bounds violations: {bounds_errors}", bounds_errors)

    def _validate_enums(self, spec: Dict[str, Any]) -> None:
        """Validate enum values."""
        enum_errors = []

        # Metric validation
        if 'metric' in spec and spec['metric'] not in self.ALLOWED_METRICS:
            enum_errors.append(f"metric: '{spec['metric']}' not in allowed values {self.ALLOWED_METRICS}")

        # Geometry validation
        if 'geometry' in spec and spec['geometry'] not in self.ALLOWED_GEOMETRIES:
            enum_errors.append(f"geometry: '{spec['geometry']}' not in allowed values {self.ALLOWED_GEOMETRIES}")

        # Dtype validation
        if 'dtype' in spec and spec['dtype'] not in self.ALLOWED_DTYPES:
            enum_errors.append(f"dtype: '{spec['dtype']}' not in allowed values {self.ALLOWED_DTYPES}")

        # Normalization method
        norm_rules = spec.get('normalization_rules', {})
        if 'method' in norm_rules and norm_rules['method'] not in self.ALLOWED_NORM_METHODS:
            enum_errors.append(f"normalization_rules.method: '{norm_rules['method']}' not in allowed values {self.ALLOWED_NORM_METHODS}")

        # Mapping type in alignment metadata
        align_meta = spec.get('alignment_metadata', {})
        if 'mapping_type' in align_meta and align_meta['mapping_type'] not in self.ALLOWED_MAPPING_TYPES:
            enum_errors.append(f"alignment_metadata.mapping_type: '{align_meta['mapping_type']}' not in allowed values {self.ALLOWED_MAPPING_TYPES}")

        # Domain validation
        semantic_space = spec.get('semantic_space', {})
        if 'domain' in semantic_space and semantic_space['domain'] not in self.ALLOWED_DOMAINS:
            enum_errors.append(f"semantic_space.domain: '{semantic_space['domain']}' not in allowed values {self.ALLOWED_DOMAINS}")

        # Corpus size validation
        if 'corpus_size' in semantic_space and semantic_space['corpus_size'] not in self.ALLOWED_CORPUS_SIZES:
            enum_errors.append(f"semantic_space.corpus_size: '{semantic_space['corpus_size']}' not in allowed values {self.ALLOWED_CORPUS_SIZES}")

        # Training method validation
        if 'training_method' in semantic_space and semantic_space['training_method'] not in self.ALLOWED_TRAINING_METHODS:
            enum_errors.append(f"semantic_space.training_method: '{semantic_space['training_method']}' not in allowed values {self.ALLOWED_TRAINING_METHODS}")

        if enum_errors:
            raise SpecEnumError(f"Invalid enum values: {enum_errors}", enum_errors)

    def _validate_alignment_metadata(self, spec: Dict[str, Any]) -> None:
        """Validate alignment metadata structure and file existence."""
        align_meta = spec.get('alignment_metadata', {})
        alignment_errors = []

        # Mapping file validation
        if 'mapping_file' in align_meta and align_meta['mapping_file'] is not None:
            mapping_file = Path(align_meta['mapping_file'])
            if not mapping_file.exists():
                alignment_errors.append(f"mapping_file: '{mapping_file}' does not exist")
            elif not mapping_file.is_file():
                alignment_errors.append(f"mapping_file: '{mapping_file}' is not a file")

        # Matrix shape validation for identity mappings
        if 'matrix_shape' in align_meta and align_meta['matrix_shape'] is not None:
            shape = align_meta['matrix_shape']
            if not isinstance(shape, list) or len(shape) != 2:
                alignment_errors.append(f"matrix_shape: must be [rows, cols], got {shape}")
            else:
                rows, cols = shape
                if not isinstance(rows, int) or not isinstance(cols, int) or rows <= 0 or cols <= 0:
                    alignment_errors.append(f"matrix_shape: invalid dimensions {shape}")
                elif rows != cols and align_meta.get('mapping_type') in ['identity']:
                    alignment_errors.append("matrix_shape: identity mapping requires square matrix")
                else:
                    # Check consistency with vector dimension
                    expected_dim = spec.get('vector_dimension')
                    if expected_dim and rows != expected_dim:
                        alignment_errors.append(f"matrix_shape: first dimension {rows} != vector_dimension {expected_dim}")

        # Alignment version format
        if 'alignment_version' in align_meta:
            ver = align_meta['alignment_version']
            if not isinstance(ver, str) or not re.match(r'^\d+\.\d+\.\d+$', ver):
                alignment_errors.append(f"alignment_version: '{ver}' must match semver format (e.g., '0.2.0')")

        if alignment_errors:
            raise AlignmentMetadataError(f"Alignment metadata errors: {alignment_errors}", alignment_meta, alignment_errors)

    def _validate_nested_structures(self, spec: Dict[str, Any]) -> None:
        """Validate nested dictionary structures."""
        structure_errors = []

        # Normalization rules structure
        norm_rules = spec.get('normalization_rules', {})
        if 'method' not in norm_rules:
            structure_errors.append("normalization_rules: missing 'method' field")
        if norm_rules.get('method') == 'l2' and 'epsilon' not in norm_rules:
            structure_errors.append("normalization_rules: L2 method requires 'epsilon' field")

        # Validation rules structure
        val_rules = spec.get('validation_rules', {})
        required_val_fields = ['dimension_min', 'dimension_max', 'epsilon_min', 'epsilon_max']
        for field in required_val_fields:
            if field not in val_rules:
                structure_errors.append(f"validation_rules: missing '{field}' field")

        # Version compatibility structure
        ver_compat = spec.get('version_compatibility', {})
        required_ver_fields = ['previous_versions', 'backwards_compatible', 'migration_required']
        for field in required_ver_fields:
            if field not in ver_compat:
                structure_errors.append(f"version_compatibility: missing '{field}' field")

        if 'previous_versions' in ver_compat and not isinstance(ver_compat['previous_versions'], list):
            structure_errors.append("version_compatibility.previous_versions: must be list")

        # Numeric precision guarantees
        num_prec = spec.get('numeric_precision_guarantees', {})
        if 'supported_precisions' not in num_prec:
            structure_errors.append("numeric_precision_guarantees: missing 'supported_precisions' field")
        elif not isinstance(num_prec['supported_precisions'], list):
            structure_errors.append("numeric_precision_guarantees.supported_precisions: must be list")

        # Embedding statistics
        emb_stats = spec.get('embedding_statistics', {})
        required_stats_fields = ['expected_distribution', 'mean_norm', 'norm_variance', 'dimension_scaling']
        for field in required_stats_fields:
            if field not in emb_stats:
                structure_errors.append(f"embedding_statistics: missing '{field}' field")

        # Robustness profiles
        robust_prof = spec.get('robustness_profiles', {})
        required_robust_fields = ['adversarial_resistance', 'corruption_tolerance', 'numerical_stability', 'provider_variability']
        for field in required_robust_fields:
            if field not in robust_prof:
                structure_errors.append(f"robustness_profiles: missing '{field}' field")

        if structure_errors:
            raise SpecStructureError(f"Nested structure errors: {structure_errors}", structure_errors)

    def _validate_semantic_consistency(self, spec: Dict[str, Any]) -> None:
        """Validate semantic consistency across spec fields."""
        consistency_errors = []

        # Spherical geometry should use cosine metric
        if spec.get('geometry') in ['spherical', 'hyperspherical'] and spec.get('metric') not in ['cosine', 'dot_product']:
            consistency_errors.append("Spherical geometry requires cosine or dot_product metric")

        # Euclidean geometry should avoid cosine
        if spec.get('geometry') == 'euclidean' and spec.get('metric') in ['cosine', 'dot_product']:
            consistency_errors.append("Euclidean geometry should not use cosine or dot_product metric")

        # Low precision dtypes should have quantization impact warnings
        if spec.get('dtype') in ['int8', 'uint8'] and spec.get('numeric_precision_guarantees', {}).get('quantization_impact') == 'minimal':
            consistency_errors.append("Low precision int8/uint8 dtypes cannot have minimal quantization impact")

        # Floating point precision should be consistent with dtype
        fp_prec = spec.get('numeric_precision_guarantees', {}).get('fp_precision')
        if fp_prec and spec.get('dtype') and spec['dtype'] != fp_prec:
            consistency_errors.append(f"numeric_precision_guarantees.fp_precision '{fp_prec}' != dtype '{spec['dtype']}'")

        # Version compatibility backwards compatibility flag should be boolean
        ver_compat = spec.get('version_compatibility', {})
        if 'backwards_compatible' in ver_compat and not isinstance(ver_compat['backwards_compatible'], bool):
            consistency_errors.append("version_compatibility.backwards_compatible: must be boolean")

        if consistency_errors:
            raise SpecConsistencyError(f"Semantic consistency violations: {consistency_errors}", consistency_errors)


# Custom exception classes for detailed spec validation errors

class SpecMissingFieldError(UERConfigurationError):
    """Error for missing required specification fields."""
    def __init__(self, message: str, missing_fields: List[str]):
        self.missing_fields = missing_fields
        super().__init__(message, {
            'tip': f"Add these fields to your UER spec: {', '.join(missing_fields)}",
            'migration': f"For v0.1 to v0.2, these fields are now mandatory"
        })


class SpecTypeError(UERConfigurationError):
    """Error for incorrect field types in specification."""
    def __init__(self, message: str, type_errors: List[str]):
        self.type_errors = type_errors
        fix_examples = {error.split(':')[0]: f"Use correct type: {error.split(':')[1].strip()}" for error in type_errors[:3]}
        super().__init__(message, fix_examples)


class SpecBoundsError(UERConfigurationError):
    """Error for numerical values outside allowed bounds."""
    def __init__(self, message: str, bounds_errors: List[str]):
        self.bounds_errors = bounds_errors
        fix_examples = {}
        for error in bounds_errors[:3]:
            field = error.split(':')[0]
            fix_examples[field] = f"Adjust {field} to be within specified bounds"
        super().__init__(message, fix_examples)


class SpecEnumError(UERConfigurationError):
    """Error for invalid enum values in specification."""
    def __init__(self, message: str, enum_errors: List[str]):
        self.enum_errors = enum_errors
        fix_examples = {}
        for error in enum_errors[:3]:
            field = error.split(':')[0]
            fix_examples[field] = f"Use valid enum value for {field}"
        super().__init__(message, fix_examples)


class AlignmentMetadataError(UERConfigurationError):
    """Error in alignment metadata configuration."""
    def __init__(self, message: str, alignment_metadata: Dict[str, Any], metadata_errors: List[str]):
        self.alignment_metadata = alignment_metadata
        self.metadata_errors = metadata_errors
        fix_examples = {}
        if any('mapping_file' in error for error in metadata_errors):
            fix_examples['mapping_file'] = "Set to null or provide path to existing matrix file"
        if any('matrix_shape' in error for error in metadata_errors):
            fix_examples['matrix_shape'] = f"Set to [{alignment_metadata.get('vector_dimension', 'dim')}, {alignment_metadata.get('vector_dimension', 'dim')}] for identity"
        super().__init__(message, fix_examples)


class SpecStructureError(UERConfigurationError):
    """Error in nested structure of specification."""
    def __init__(self, message: str, structure_errors: List[str]):
        self.structure_errors = structure_errors
        fix_examples = {}
        for error in structure_errors[:3]:
            parts = error.split(':')
            if len(parts) > 1:
                section = parts[0].replace('_', '.')
                field = parts[1].strip().strip("'").replace('missing ', '')
                fix_examples[f"{section}.{field}"] = f"Add required '{field}' field to {section}"
        super().__init__(message, fix_examples)


class SpecConsistencyError(UERConfigurationError):
    """Error for semantic consistency violations."""
    def __init__(self, message: str, consistency_errors: List[str]):
        self.consistency_errors = consistency_errors
        fix_examples = {}
        for error in consistency_errors[:3]:
            if 'geometry' in error.lower() and 'metric' in error.lower():
                fix_examples['geometry'] = "Change geometry or metric for consistency"
            elif 'dtype' in error.lower() and 'precision' in error.lower():
                fix_examples['dtype'] = "Adjust dtype to match precision guarantees"
            elif 'backwards_compatible' in error.lower():
                fix_examples['backwards_compatible'] = "Set to true or false explicitly"
        super().__init__(message, fix_examples)


def validate_uer_spec(spec: Dict[str, Any]) -> None:
    """
    Convenience function to validate a UER specification.

    Args:
        spec: Specification dictionary to validate

    Raises:
        Various Spec*Error exceptions on validation failure
    """
    validator = SpecValidator()
    validator.validate_spec(spec)
