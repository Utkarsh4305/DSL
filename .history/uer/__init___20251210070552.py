"""
Universal Embedding Representation (UER)

A vendor-neutral, model-agnostic intermediate representation for text embeddings.
Production-ready system with advanced alignment, validation, and performance optimizations.
"""

__version__ = "0.2.0"
__description__ = "Universal Embedding Representation - Production Ready"

from .core.compiler import UERCompiler, compile_to_uer, compile_batch_to_uer
from .core.config import load_uer_spec, UERSpecLoader
from .validation.validator import UERValidator, validate_uer_embedding, validate_batch_uer
from .providers.registry import ProviderRegistry, register_provider, get_provider_alignment

# Legacy compatibility
from .loader import load_uer_spec as legacy_load_uer_spec
from .validator import validate_uer_embedding as legacy_validate_uer_embedding

__all__ = [
    # Core compilation
    'UERCompiler',
    'compile_to_uer',
    'compile_batch_to_uer',

    # Configuration
    'load_uer_spec',
    'UERSpecLoader',

    # Validation
    'UERValidator',
    'validate_uer_embedding',
    'validate_batch_uer',

    # Provider registry
    'ProviderRegistry',
    'register_provider',
    'get_provider_alignment',

    # Version and meta
    '__version__',
    '__description__',
]
