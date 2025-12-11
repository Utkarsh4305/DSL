"""
UER Validation Module

Provides comprehensive validation for UER-compliant embeddings in v0.2.
Enhanced checks for dimension bounds, zero-vectors, dtype enforcement,
norm tolerance, and per-geometry rules.
"""

import numpy as np
from typing import Union, Dict, Any, Optional, List
import logging

from .utils.errors import UERValidationError

logger = logging.getLogger(__name__)


class UERValidator:
    """Validates embeddings against UER specification."""

    def __init__(self, spec: Dict[str, Any]):
        """
        Initialize validator with UER spec.

        Args:
            spec: UER specification dictionary (v0.2 expected)
        """
        self.spec = spec
        self._norm_eps = float(spec.get('normalization_rules', {}).get('epsilon', 1e-12))
        self._norm_tolerance = float(spec.get('validation_rules', {}).get('norm_tolerance', 1e-6))
        self._dim_tolerance = float(spec.get('validation_rules', {}).get('dimension_tolerance', 0.0))

        # v0.2 specific attributes
        self._dim_min = int(spec.get('validation_rules', {}).get('dimension_min', 1))
        self._dim_max = int(spec.get('validation_rules', {}).get('dimension_max', 10000))
        self._zero_vector_reject = bool(spec.get('validation_rules', {}).get('zero_vector_reject', True))
        self._geometry = spec.get('geometry', 'spherical')

    def validate_embedding(self, embedding: np.ndarray, strict: bool = True) -> bool:
        """
        Validate a single embedding vector with v0.2 enhanced checks.

        Args:
            embedding: Embedding vector to validate
            strict: If True, raises UERValidationError on failure. If False, returns bool.

        Returns:
            True if valid, False if invalid (when strict=False)

        Raises:
            UERValidationError: If validation fails and strict=True
        """
        try:
            embedding_info = {
                'shape': embedding.shape,
                'dtype': str(embedding.dtype),
                'expected_dim': self.spec['vector_dimension']
            }

            # v0.2 enhanced validation sequence
            self._validate_dtype_enhanced(embedding, embedding_info)
            self._validate_dimension_bounds(embedding, embedding_info)
            self._validate_finite_enhanced(embedding, embedding_info)
            self._validate_zero_vector(embedding, embedding_info)
            self._validate_normalization_enhanced(embedding, embedding_info)
            self._validate_geometry_rules(embedding, embedding_info)

            return True

        except UERValidationError:
            raise  # Re-raise UERValidationError as-is
        except Exception as e:
            # Wrap generic errors in UERValidationError
            validation_error = UERValidationError(
                f"Unexpected validation error: {e}",
                embedding_info={'shape': getattr(embedding, 'shape', 'unknown'),
                               'dtype': str(getattr(embedding, 'dtype', 'unknown'))}
            )
            if strict:
                raise validation_error
            logger.warning(f"Embedding validation failed: {validation_error}")
            return False

    def _validate_dtype(self, embedding: np.ndarray) -> None:
        """Validate embedding dtype."""
        expected_dtype = np.dtype(self.spec['dtype'])

        if embedding.dtype != expected_dtype:
            if self.spec.get('validation_rules', {}).get('dtype_strict', True):
                raise ValueError(f"Expected dtype {expected_dtype}, got {embedding.dtype}")

        # Check for quantization dtypes
        if self.spec['dtype'] in ['int8', 'uint8']:
            if np.any((embedding < -128) | (embedding > 127)):
                raise ValueError("int8 values out of range [-128, 127]")

    def _validate_dimension(self, embedding: np.ndarray) -> None:
        """Validate embedding dimension."""
        expected_dim = self.spec['vector_dimension']
        actual_dim = embedding.shape[-1] if embedding.ndim > 1 else len(embedding)

        if abs(actual_dim - expected_dim) > self._dim_tolerance:
            raise ValueError(f"Expected dimension {expected_dim}, got {actual_dim}")

    def _validate_finite(self, embedding: np.ndarray) -> None:
        """Validate embedding contains no NaN or infinite values."""
        if self.spec.get('validation_rules', {}).get('nan_check', True):
            if not np.all(np.isfinite(embedding)):
                if np.any(np.isnan(embedding)):
                    raise ValueError("Embedding contains NaN values")
                if np.any(np.isinf(embedding)):
                    raise ValueError("Embedding contains infinite values")

    def _validate_normalization(self, embedding: np.ndarray) -> None:
        """Validate embedding normalization based on spec rules."""
        norm_method = self.spec.get('normalization_rules', {}).get('method')

        if norm_method == 'l2':
            # For spherical geometry, L2 norm should be ~1.0
            norms = np.linalg.norm(embedding, axis=-1)
            expected_norm = 1.0

            if np.any(np.abs(norms - expected_norm) > self._norm_tolerance):
                raise ValueError(f"L2 normalization violation: norms {norms}, expected {expected_norm}")

        elif norm_method == 'none':
            # No normalization check
            pass
        else:
            logger.warning(f"Unknown normalization method: {norm_method}")

    # ===== v0.2 Enhanced Validation Methods =====

    def _validate_dtype_enhanced(self, embedding: np.ndarray, embedding_info: Dict[str, Any]) -> None:
        """Strict dtype enforcement before and after operations."""
        expected_dtype = np.dtype(self.spec['dtype'])

        if embedding.dtype != expected_dtype:
            if self.spec.get('validation_rules', {}).get('dtype_strict', True):
                raise UERValidationError(
                    f"Dtype mismatch: expected {expected_dtype}, got {embedding.dtype}",
                    {"validation_issue": "dtype_mismatch", **embedding_info},
                    ["Convert embedding with: arr.astype(np.dtype('{expected_dtype}'))"]
                )

        # Enhanced quantization checks
        if self.spec['dtype'] in ['int8', 'uint8']:
            if np.any((embedding < -128) | (embedding > 127)):
                raise UERValidationError(
                    f"Quantized dtype {self.spec['dtype']} values out of range [-128, 127]",
                    {"validation_issue": "quantization_range", **embedding_info},
                    ["Apply quantization with proper scaling before validation"]
                )

    def _validate_dimension_bounds(self, embedding: np.ndarray, embedding_info: Dict[str, Any]) -> None:
        """Validate dimension with strict bounds checking."""
        expected_dim = self.spec['vector_dimension']
        actual_dim = embedding.shape[-1] if embedding.ndim > 1 else len(embedding)

        # Check exact match within tolerance
        if abs(actual_dim - expected_dim) > self._dim_tolerance:
            raise UERValidationError(
                f"Dimension mismatch: expected {expected_dim}, got {actual_dim}",
                {"validation_issue": "dimension_mismatch", **embedding_info},
                [f"Check your model's output dimension (tolerance: {self._dim_tolerance})",
                 "Use dimension_aware models or apply padding/truncation"]
            )

        # Lower bound checking
        if actual_dim < self._dim_min:
            raise UERValidationError(
                f"Dimension {actual_dim} below minimum allowed {self._dim_min}",
                {"validation_issue": "dimension_too_small", **embedding_info},
                ["Increase embedding dimension to meet minimum requirements"]
            )

        # Upper bound checking
        if actual_dim > self._dim_max:
            raise UERValidationError(
                f"Dimension {actual_dim} exceeds maximum allowed {self._dim_max}",
                {"validation_issue": "dimension_too_large", **embedding_info},
                ["Reduce embedding dimension or update spec bounds"]
            )

    def _validate_finite_enhanced(self, embedding: np.ndarray, embedding_info: Dict[str, Any]) -> None:
        """Enhanced NaN/Inf detection with consistent rejection."""
        if self.spec.get('validation_rules', {}).get('nan_check', True):
            nan_count = np.sum(np.isnan(embedding))
            inf_count = np.sum(np.isinf(embedding))

            if nan_count > 0:
                raise UERValidationError(
                    f"Embedding contains {nan_count} NaN values (not allowed)",
                    {"validation_issue": "contains_nan", **embedding_info},
                    [f"Check for numerical instability in your embedding pipeline",
                     "Add gradient clipping or numerical stabilization"]
                )

            if inf_count > 0:
                raise UERValidationError(
                    f"Embedding contains {inf_count} infinite values (not allowed)",
                    {"validation_issue": "contains_inf", **embedding_info},
                    ["Check for division by zero in computations",
                     "Add numerical safeguards in preprocessing"]
                )

    def _validate_zero_vector(self, embedding: np.ndarray, embedding_info: Dict[str, Any]) -> None:
        """Reject all-zero vectors if configured."""
        if self._zero_vector_reject:
            # Check if all elements are zero (within floating point tolerance)
            is_zero = np.allclose(embedding, 0, atol=1e-8)
            if is_zero:
                raise UERValidationError(
                    "Zero vector detected (rejected per spec rules)",
                    {"validation_issue": "zero_vector", **embedding_info},
                    ["Ensure your embedding model produces non-zero outputs",
                     "Check for initialization or scaling issues"]
                )

    def _validate_normalization_enhanced(self, embedding: np.ndarray, embedding_info: Dict[str, Any]) -> None:
        """Enhanced normalization validation with configurable tolerance."""
        norm_method = self.spec.get('normalization_rules', {}).get('method')

        if norm_method == 'l2':
            norms = np.linalg.norm(embedding, axis=-1, keepdims=True)
            expected_norm = 1.0

            # Check violation against tolerance
            violations = np.abs(norms - expected_norm) > self._norm_tolerance
            if np.any(violations):
                max_deviation = np.max(np.abs(norms - expected_norm))
                raise UERValidationError(
                    f"L2 normalization violation: max deviation {max_deviation:.6f} > tolerance {self._norm_tolerance}",
                    {"validation_issue": "normalization_violation", **embedding_info},
                    ["Apply proper L2 normalization: vec / (np.linalg.norm(vec) + eps)",
                     f"Check epsilon value: {self._norm_eps}"]
                )

        elif norm_method not in ['l1', 'none']:
            logger.warning(f"Unknown normalization method ignored: {norm_method}")

    def _validate_geometry_rules(self, embedding: np.ndarray, embedding_info: Dict[str, Any]) -> None:
        """Validate embedding properties based on geometry type."""
        if self._geometry == 'spherical':
            # Spherical geometry: ensure unit vectors
            norms = np.linalg.norm(embedding, axis=-1)
            # Additional check beyond normalization - ensure reasonable unit vector properties
            if np.any(norms > 2.0):  # Extremely large norms indicate issues
                raise UERValidationError(
                    "Spherical geometry: norms excessively large (>2.0)",
                    {"validation_issue": "geometry_spherical_large_norm", **embedding_info},
                    ["Normalize with L2 normalization for spherical embeddings"]
                )

        elif self._geometry == 'hyperspherical':
            # Similar to spherical but potentially higher dimensions
            norms = np.linalg.norm(embedding, axis=-1)
            expected_dim = self.spec['vector_dimension']
            if expected_dim > 768:  # Heuristic for hyperspherical
                # Higher dim spheres have different properties, but still should be roughly unit
                if np.any(norms > 1.5):
                    raise UERValidationError(
                        "Hyperspherical geometry: norms too large for high-dimensional space",
                        {"validation_issue": "geometry_hyperspherical_large_norm", **embedding_info},
                        ["Verify hyperspherical coordinate transformations"]
                    )

        # Future: add geometry-specific rules for euclidean, etc.

    def validate_batch(self, embeddings: np.ndarray, strict: bool = True):
        """
        Validate a batch of embeddings.

        Args:
            embeddings: Batch of embeddings (2D array)
            strict: If True, raises on any invalid. If False, returns list of invalid indices.

        Returns:
            True if all valid, or list of invalid indices if any failed (when strict=False)

        Raises:
            UERValidationError: If any embedding invalid and strict=True
        """
        if embeddings.ndim != 2:
            raise ValueError(f"Batch embeddings must be 2D, got {embeddings.ndim}D")

        invalid_indices = []

        for i, embedding in enumerate(embeddings):
            try:
                self.validate_embedding(embedding, strict=True)
            except UERValidationError:
                invalid_indices.append(i)

        if invalid_indices:
            if strict:
                raise UERValidationError(
                    f"Batch validation failed: {len(invalid_indices)}/{len(embeddings)} invalid embeddings at indices {invalid_indices[:10]}",
                    {"validation_issue": "batch_validation_failure",
                     "invalid_count": len(invalid_indices),
                     "batch_size": len(embeddings)},
                    ["Check embedding preprocessing pipeline",
                     "Enable batch normalization and filtering"]
                )
            return invalid_indices

        return True


def validate_uer_embedding(embedding: np.ndarray, spec: Dict[str, Any]) -> bool:
    """
    Convenience function to validate a UER embedding with v0.2 enhanced checks.

    Args:
        embedding: Embedding to validate
        spec: UER specification

    Returns:
        True if valid

    Raises:
        UERValidationError: If invalid
    """
    validator = UERValidator(spec)
    return validator.validate_embedding(embedding, strict=True)


def validate_uer_embedding_batch(embeddings: np.ndarray, spec: Dict[str, Any],
                                strict: bool = True) -> Union[bool, List[int]]:
    """
    Convenience function to validate a batch of UER embeddings.

    Args:
        embeddings: Batch of embeddings (2D array)
        spec: UER specification
        strict: If True, raises on invalid. If False, returns list of failed indices.

    Returns:
        True if all valid, or list of invalid indices (when strict=False)

    Raises:
        UERValidationError: If any invalid and strict=True
    """
    validator = UERValidator(spec)
    return validator.validate_batch(embeddings, strict=strict)
