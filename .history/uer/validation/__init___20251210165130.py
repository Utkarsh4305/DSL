"""
UER Validation System

Enhanced validation with anisotropy checks, distribution analysis, and auto-repair.
Completely rewritten for robustness and semantic preservation validation.
"""

from .validator import (
    UERValidator,
    validate_uer_embedding,
    validate_batch_uer,  # v0.1 compatibility
    validate_uer_embedding_batch  # v0.2 function
)
from .validator import validate_uer_embedding_batch as validate_batch_uer_v2  # Alias
from .checks import AnisotropyChecker, DistributionChecker, SemanticChecker

__all__ = [
    'UERValidator',
    'validate_uer_embedding',
    'validate_batch_uer',
    'UERValidationReport',
    'AnisotropyChecker',
    'DistributionChecker',
    'SemanticChecker'
]
