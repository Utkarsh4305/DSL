"""
Universal Embedding Representation (UER)

A vendor-neutral, model-agnostic intermediate representation for text embeddings.
Production-ready system with advanced alignment, validation, and performance optimizations.
"""

__version__ = "0.2.0"
__description__ = "Universal Embedding Representation - Production Ready"

from .core.compiler import UERCompiler, compile_to_uer, compile_batch_to_uer
from .loader import load_uer_spec, UERSpecLoader
from .core.config import UERConfig
from .validation.validator import UERValidator, validate_uer_embedding, validate_uer_embedding_batch
from .providers.registry import ProviderRegistry, register_provider, get_provider_alignment

__all__ = [
    # Core compilation
    'UERCompiler',
    'compile_to_uer',
    'compile_batch_to_uer',

    # Configuration
    'load_uer_spec',
    'UERSpecLoader',
    'UERConfig',

    # Validation
    'UERValidator',
    'validate_uer_embedding',
    'validate_uer_embedding_batch',

    # Provider registry
    'ProviderRegistry',
    'register_provider',
    'get_provider_alignment',

    # Version and meta
    '__version__',
    '__description__',
]
