"""
UER Alignment Operations

Cross-model consistency: linear projection, Procrustes alignment, anisotropy correction.
"""

from .linear import LinearAlignment, derive_projection_matrix, evaluate_alignment_quality

__all__ = ['LinearAlignment', 'derive_projection_matrix', 'evaluate_alignment_quality']
