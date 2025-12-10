"""
Universal Embedding Representation (UER)

A vendor-neutral, model-agnostic intermediate representation for text embeddings.
"""

__version__ = "0.1.0"
__description__ = "Universal Embedding Representation - Open Source Standard"

from .loader import load_uer_spec, UERSpecLoader
from .validator import validate_uer_embedding, UERValidator
from .compiler import compile_to_uer, UERCompiler

__all__ = [
    # Core classes
    'UERSpecLoader',
    'UERValidator',
    'UERCompiler',
    
    # Convenience functions
    'load_uer_spec',
    'validate_uer_embedding', 
    'compile_to_uer',
    
    # Version
    '__version__',
]
