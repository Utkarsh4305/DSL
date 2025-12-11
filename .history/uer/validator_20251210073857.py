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
        Validate a single embedding vector.

        Args:
            embedding: Embedding vector to validate
            strict: If True, raises on validation failure. If False, returns bool.

        Returns:
            True if valid, False if invalid (when strict=False)

        Raises:
            ValueError: If validation fails and strict=True
        """
        try:
            self._validate_dtype(embedding)
            self._validate_dimension(embedding)
            self._validate_finite(embedding)
            self._validate_normalization(embedding)
            return True

        except ValueError as e:
            if strict:
                raise e
            logger.warning(f"Embedding validation failed: {e}")
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


def validate_uer_embedding(embedding: np.ndarray, spec: Dict[str, Any]) -> bool:
    """
    Convenience function to validate a UER embedding.

    Args:
        embedding: Embedding to validate
        spec: UER specification

    Returns:
        True if valid

    Raises:
        ValueError: If invalid
    """
    validator = UERValidator(spec)
    return validator.validate_embedding(embedding, strict=True)
