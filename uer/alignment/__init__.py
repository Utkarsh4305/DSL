"""
UER Alignment Operations

Cross-model consistency: linear projection, Procrustes alignment, anisotropy correction.
Advanced alignment strategies for preserving semantic relationships.
"""

from .linear import LinearAlignment, derive_projection_matrix, evaluate_alignment_quality
from .procrustes import (ProcrustesAlignment, IterativeProcrustes, procrustes_align,
                        rigid_align, similarity_align, affine_align)

__all__ = [
    # Linear alignment
    'LinearAlignment', 'derive_projection_matrix', 'evaluate_alignment_quality',

    # Procrustes alignment (geometry-preserving)
    'ProcrustesAlignment', 'IterativeProcrustes', 'procrustes_align',
    'rigid_align', 'similarity_align', 'affine_align'
]
