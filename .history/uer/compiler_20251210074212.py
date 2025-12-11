"""
UER Compiler Module v0.2

Transforms raw embeddings from any model into UER-compliant vectors.
Enhanced with dimension mismatch handling, projection safeguards,
normalization guards, and batch processing groundwork.
"""

import numpy as np
from typing import Dict, Any, Union, Optional, Callable, List
from pathlib import Path
import logging

from .validator import UERValidator
from .ops.align import AlignmentOperator
from .utils.errors import UERCompilationError

logger = logging.getLogger(__name__)


class UERCompiler:
    """
    Compiles raw embeddings into UER-compliant vectors with v0.2 safeguards.

    Enhanced with dimension mismatch handling, projection failure detection,
    normalization guards, and preliminary batch support.
    """

    def __init__(self, spec: Dict[str, Any]):
        """
        Initialize UER compiler with v0.2 enhancements.

        Args:
            spec: UER v0.2 specification dictionary
        """
        self.spec = spec
        self.validator = UERValidator(spec)
        self.alignment_op = self._load_alignment_operator(spec)

        # v0.2 attributes
        self._target_dim = spec['vector_dimension']
        self._target_dtype = np.dtype(spec['dtype'])
        self._enable_batch_mode = False  # Scaffolding for future batch support

    def compile(self, embedding: np.ndarray, validate_input: bool = True,
                handle_mismatches: bool = True) -> np.ndarray:
        """
        Compile raw embedding into UER-compliant vector with enhanced safeguards.

        Args:
            embedding: Raw embedding from any model (1D or 2D)
            validate_input: Whether to validate input embedding
            handle_mismatches: Whether to attempt intelligent dimension handling

        Returns:
            UER-compliant embedding vector

        Raises:
            UERCompilationError: If compilation encounters critical issues
        """
        try:
            # Step 0: Pre-compilation dimension validation and handling
            checked_embedding = self._handle_dimension_mismatch(embedding, handle_mismatches)
            self._validate_input_dimensions(checked_embedding)

            # Step 1: Apply alignment/projection with failure detection
            aligned = self._apply_alignment_safely(checked_embedding)

            # Step 2: Apply normalization with safeguards
            normalized = self._apply_normalization_safely(aligned)

            # Step 3: Convert to specified dtype with bounds checking
            dtype_normalized = self._enforce_dtype_safely(normalized, self._target_dtype)

            # Step 4: Final UER validation (now uses v0.2 checks)
            self.validator.validate_embedding(dtype_normalized, strict=True)

            return dtype_normalized

        except Exception as e:
            if isinstance(e, UERCompilationError):
                raise
            # Wrap generic errors in UERCompilationError
            raise UERCompilationError(
                f"Compilation failed for embedding shape {embedding.shape}: {e}",
                input_info={'original_shape': embedding.shape,
                           'dtype': str(embedding.dtype),
                           'target_dim': self._target_dim},
                suggested_fixes=["Check input embedding dimensions match spec",
                               "Ensure embedding contains no NaN/inf values",
                               "Verify alignment matrix compatibility"],
                fixes_applied=[]
            )

    def _handle_dimension_mismatch(self, embedding: np.ndarray, handle_mismatches: bool) -> np.ndarray:
        """
        Handle dimension mismatches intelligently.

        Args:
            embedding: Input embedding
            handle_mismatches: Whether to attempt padding/truncation

        Returns:
            Embedding with corrected dimensions

        Raises:
            UERCompilationError: If mismatch cannot be handled
        """
        input_dim = embedding.shape[-1] if embedding.ndim > 1 else len(embedding)

        if input_dim == self._target_dim:
            return embedding
        elif not handle_mismatches:
            raise UERCompilationError(
                f"Dimension mismatch: got {input_dim}, expected {self._target_dim}. "
                "Set handle_mismatches=True to enable auto-correction.",
                input_info={'input_dim': input_dim, 'target_dim': self._target_dim},
                suggested_fixes=["Enable handle_mismatches=True for auto padding/truncation",
                               "Pre-process embeddings to match target dimension"]
            )

        # Intelligent dimension correction
        if input_dim < self._target_dim:
            # Pad with zeros (or smart replication for semantic vectors)
            if embedding.ndim == 1:
                padding = np.zeros(self._target_dim - input_dim)
                return np.concatenate([embedding, padding])
            else:
                padding_shape = (embedding.shape[0], self._target_dim - input_dim)
                padding = np.zeros(padding_shape)
                return np.concatenate([embedding, padding], axis=1)
        else:
            # Truncate (warn about potential semantic loss)
            logger.warning(f"Truncating embedding from {input_dim} to {self._target_dim} dimensions")
            if embedding.ndim == 1:
                return embedding[:self._target_dim]
            else:
                return embedding[:, :self._target_dim]

    def _validate_input_dimensions(self, embedding: np.ndarray) -> None:
        """Validate input embedding has reasonable dimensions."""
        if embedding.ndim not in [1, 2]:
            raise ValueError(f"Embedding must be 1D or 2D array, got {embedding.ndim}D")

        # Check for extremely large arrays that might cause memory issues
        total_elements = np.prod(embedding.shape)
        max_elements = 100 * 1000 * 1000  # 100M elements limit
        if total_elements > max_elements:
            raise UERCompilationError(
                f"Embedding too large: {total_elements} elements > {max_elements} limit",
                input_info={'shape': embedding.shape, 'total_elements': total_elements},
                suggested_fixes=["Reduce batch size", "Use smaller embeddings"]
            )

    def _apply_alignment_safely(self, embedding: np.ndarray) -> np.ndarray:
        """Apply alignment with projection failure detection."""
        try:
            aligned = self.alignment_op.align(embedding)

            # Detect projection failures
            if aligned.shape[-1] != self._target_dim:
                raise UERCompilationError(
                    f"Alignment failed: output dimension {aligned.shape[-1]} != target {self._target_dim}",
                    input_info={'pre_align_shape': embedding.shape,
                               'post_align_shape': aligned.shape},
                    suggested_fixes=["Check alignment matrix dimensions",
                                   "Verify alignment operator configuration"]
                )

            # Check for NaN/inf in aligned output
            if not np.all(np.isfinite(aligned)):
                nan_count = np.sum(np.isnan(aligned))
                inf_count = np.sum(np.isinf(aligned))
                raise UERCompilationError(
                    f"Alignment produced invalid values: {nan_count} NaN, {inf_count} inf",
                    input_info={'aligned_shape': aligned.shape,
                               'nan_count': nan_count, 'inf_count': inf_count},
                    suggested_fixes=["Check for unstable alignment transformations",
                                   "Verify alignment matrix conditioning"]
                )

            return aligned

        except Exception as e:
            raise UERCompilationError(
                f"Alignment operation failed: {e}",
                input_info={'input_shape': embedding.shape,
                           'mapping_type': self.spec.get('alignment_metadata', {}).get('mapping_type')},
                suggested_fixes=["Use identity mapping for initial testing",
                               "Train/fit alignment matrices with sufficient data"]
            )

    def _apply_normalization_safely(self, embedding: np.ndarray) -> np.ndarray:
        """Apply normalization with division-by-zero safeguards."""
        norm_rules = self.spec.get('normalization_rules', {})
        method = norm_rules.get('method', 'l2')
        eps = float(norm_rules.get('epsilon', 1e-12))

        if method == 'l2':
            # Enhanced L2 normalization with safeguards
            axis = embedding.ndim - 1
            norms = np.linalg.norm(embedding, axis=axis, keepdims=True)

            # Check for zero norms that could cause issues
            zero_norm_mask = norms == 0
            if np.any(zero_norm_mask):
                # Handle zero vectors by replacing with small epsilon
                norms = np.where(zero_norm_mask, eps, norms)
                logger.warning(f"Found {np.sum(zero_norm_mask)} zero vectors during normalization, using epsilon")

            # Final normalization with guaranteed non-zero denominators
            return embedding / np.maximum(norms, eps)

        elif method == 'none':
            return embedding

        elif method == 'l1':
            # L1 normalization for potential euclidean geometry
            axis = embedding.ndim - 1
            norms = np.sum(np.abs(embedding), axis=axis, keepdims=True)
            return embedding / np.maximum(norms, eps)

        else:
            raise ValueError(f"Unsupported normalization method: {method}")

    def _enforce_dtype_safely(self, embedding: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
        """Convert to target dtype with bounds checking."""
        # Attempt conversion
        converted = embedding.astype(target_dtype)

        # For integer dtypes, verify values are in range
        if np.issubdtype(target_dtype, np.integer):
            if np.issubdtype(target_dtype, np.signedinteger):
                min_val, max_val = np.iinfo(target_dtype).min, np.iinfo(target_dtype).max
            else:  # unsigned
                min_val, max_val = 0, np.iinfo(target_dtype).max

            if np.any((converted < min_val) | (converted > max_val)):
                raise UERCompilationError(
                    f"Values out of range for dtype {target_dtype}",
                    input_info={'dtype': str(target_dtype),
                               'min_val': min_val, 'max_val': max_val},
                    suggested_fixes=["Use floating point dtype for continuous values",
                                   "Apply quantization transformation before compilation"]
                )

        return converted

    def _load_alignment_operator(self, spec: Dict[str, Any]) -> AlignmentOperator:
        """Load appropriate alignment operator with v0.2 validation."""
        align_meta = spec.get('alignment_metadata', {})
        mapping_type = align_meta.get('mapping_type', 'identity')

        # Validate matrix shape if provided
        if 'matrix_shape' in align_meta and align_meta['matrix_shape']:
            matrix_shape = align_meta['matrix_shape']
            if not (isinstance(matrix_shape, list) and len(matrix_shape) == 2):
                raise UERCompilationError(
                    f"Invalid matrix_shape: {matrix_shape}. Must be [rows, cols].",
                    input_info={'matrix_shape': matrix_shape}
                )

        # For v0.2, validate alignment metadata is properly structured
        required_metadata_fields = ['mapping_type', 'alignment_version']
        missing_fields = [field for field in required_metadata_fields if field not in align_meta]
        if missing_fields:
            raise UERCompilationError(
                f"Missing alignment metadata fields: {missing_fields}",
                input_info={'alignment_metadata': align_meta},
                suggested_fixes=["Add missing fields to alignment_metadata in spec"]
            )

        return AlignmentOperator(spec, mapping_type=mapping_type)

    # ===== Batch Processing Scaffolding =====

    def compile_batch(self, embeddings: np.ndarray, validate_input: bool = True,
                     handle_mismatches: bool = True) -> Union[np.ndarray, List[Optional[np.ndarray]]]:
        """
        Compile batch of embeddings (scaffolding for v0.2).

        Args:
            embeddings: Batch of embeddings (2D array)
            validate_input: Whether to validate inputs
            handle_mismatches: Whether to handle dimension mismatches

        Returns:
            Compiled embeddings or list with None for failed compilations
        """
        if not self._enable_batch_mode:
            raise NotImplementedError("Batch mode not yet enabled in v0.2")

        if embeddings.ndim != 2:
            raise ValueError(f"Batch embeddings must be 2D, got {embeddings.ndim}D")

        results = []
        for i, embedding in enumerate(embeddings):
            try:
                compiled = self.compile(embedding, validate_input, handle_mismatches)
                results.append(compiled)
            except Exception as e:
                logger.error(f"Batch compilation failed for embedding {i}: {e}")
                results.append(None)

        return results

    def enable_batch_mode(self, enabled: bool = True) -> None:
        """Enable/disable batch processing mode (v0.2 scaffolding)."""
        self._enable_batch_mode = enabled
        logger.info(f"Batch mode {'enabled' if enabled else 'disabled'}")


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
