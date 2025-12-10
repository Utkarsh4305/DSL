"""
UER Compiler Module

Transforms raw embeddings from any model into UER-compliant vectors.
Handles projection/alignment and normalization according to UER spec.
"""

import numpy as np
from typing import Dict, Any, Union, Optional, Callable
from pathlib import Path
import logging

from .validator import UERValidator
from .ops.align import AlignmentOperator

logger = logging.getLogger(__name__)


class UERCompiler:
    """Compiles raw embeddings into UER-compliant vectors."""

    def __init__(self, spec: Dict[str, Any]):
        """
        Initialize UER compiler.

        Args:
            spec: UER specification dictionary
        """
        self.spec = spec
        self.validator = UERValidator(spec)
        self.alignment_op = self._load_alignment_operator(spec)

    def compile(self, embedding: np.ndarray, validate_input: bool = True) -> np.ndarray:
        """
        Compile raw embedding into UER-compliant vector.

        Args:
            embedding: Raw embedding from any model
            validate_input: Whether to validate input embedding

        Returns:
            UER-compliant embedding vector

        Raises:
            ValueError: If compilation fails validation
        """
        # Optional input validation
        if validate_input:
            # Note: Raw embedding may not conform to UER spec, so use non-strict validation
            if embedding.ndim != 1 and embedding.ndim != 2:
                raise ValueError(f"Embedding must be 1D or 2D array, got {embedding.ndim}D")

        # Step 1: Apply alignment/projection
        aligned = self.alignment_op.align(embedding)

        # Step 2: Apply IR normalization
        normalized = self._apply_normalization(aligned)

        # Step 3: Convert to specified dtype
        compile_result = normalized.astype(np.dtype(self.spec['dtype']))

        # Step 4: Final UER validation
        self.validator.validate_embedding(compile_result, strict=True)

        return compile_result

    def _load_alignment_operator(self, spec: Dict[str, Any]) -> AlignmentOperator:
        """Load appropriate alignment operator based on spec."""
        align_meta = spec.get('alignment_metadata', {})
        mapping_type = align_meta.get('mapping_type', 'identity')

        # For v0.1, we use placeholder identity mapping
        # Future versions will load trained alignment models
        return AlignmentOperator(spec, mapping_type=mapping_type)

    def _apply_normalization(self, embedding: np.ndarray) -> np.ndarray:
        """Apply normalization according to UER spec."""
        norm_rules = self.spec.get('normalization_rules', {})
        method = norm_rules.get('method', 'l2')

        if method == 'l2':
            # L2 normalize for spherical geometry
            axis = embedding.ndim - 1
            norms = np.linalg.norm(embedding, axis=axis, keepdims=True)
            eps = float(norm_rules.get('epsilon', 1e-12))
            return embedding / np.maximum(norms, eps)

        elif method == 'none':
            return embedding

        else:
            raise ValueError(f"Unsupported normalization method: {method}")


def compile_to_uer(embedding: np.ndarray, spec: Dict[str, Any]) -> np.ndarray:
    """
    Convenience function to compile embedding to UER format.

    Args:
        embedding: Raw embedding
        spec: UER specification

    Returns:
        UER-compliant embedding
    """
    compiler = UERCompiler(spec)
    return compiler.compile(embedding)
