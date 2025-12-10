"""
UER Compilation Example

Demonstrates how to transform raw embeddings from various models
into UER-compliant vectors using the UER compiler.
"""

import numpy as np
import sys
from pathlib import Path

# Add uer package to path
uer_path = Path(__file__).parent.parent / 'uer'
sys.path.insert(0, str(uer_path))

from uer import load_uer_spec, compile_to_uer


def example_transform():
    """Example of transforming a raw embedding to UER format."""

    print("UER Compilation Example")
    print("=" * 50)

    # Load UER specification
    spec_file = Path(__file__).parent.parent / 'specs' / 'uer_v0.1.yaml'
    spec = load_uer_spec(spec_file)

    print(f"Loaded UER spec v{spec['uer_version']}")
    print(f"Vector dimension: {spec['vector_dimension']}")
    print(f"Dtype: {spec['dtype']}")
    print(f"Normalization: {spec['normalization_rules']['method']}")
    print()

    # Simulate a raw embedding from a hypothetical model
    # This could come from OpenAI, Cohere, BGE, etc.
    raw_embedding = np.random.randn(384)  # Example: 384-dim from some model

    print(f"Raw embedding shape: {raw_embedding.shape}")
    print(f"Raw embedding dtype: {raw_embedding.dtype}")
    print(".4f")
    print()

    # Compile to UER format
    uer_embedding = compile_to_uer(raw_embedding, spec)

    print(f"Compiled UER embedding shape: {uer_embedding.shape}")
    print(f"UER embedding dtype: {uer_embedding.dtype}")
    print(".6f")
    print(".6f")
    print()

    # Verify the embedding matches spec requirements
    expected_norm = 1.0 if spec['normalization_rules']['method'] == 'l2' else None

    if expected_norm:
        actual_norm = np.linalg.norm(uer_embedding)
        print(".6f")
        print(".2f")
    else:
        print("No normalization applied")

    print("\nCompilation successful! ✓")


def example_batch_transform():
    """Example of batch processing multiple embeddings."""

    print("\nBatch Compilation Example")
    print("=" * 50)

    # Load spec
    spec_file = Path(__file__).parent.parent / 'specs' / 'uer_v0.1.yaml'
    spec = load_uer_spec(spec_file)

    # Simulate batch of raw embeddings (e.g., from a vector database query)
    batch_size = 100
    raw_embeddings = np.random.randn(batch_size, 512)  # Different model dims

    print(f"Raw batch shape: {raw_embeddings.shape}")
    print()

    # Compile entire batch
    uer_batch = compile_to_uer(raw_embeddings, spec)

    print(f"Compiled UER batch shape: {uer_batch.shape}")
    print(f"Target dimension: {spec['vector_dimension']}")
    print(".3f")
    print(".3f")

    print("\nBatch compilation successful! ✓")


if __name__ == "__main__":
    example_transform()
    example_batch_transform()
