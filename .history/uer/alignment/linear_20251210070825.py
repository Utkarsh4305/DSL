"""
UER Linear Alignment Operations

Learn projection matrices to map embeddings from different models into UER space.
Fixes cross-model consistency issues with learned transformations.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path

from ..core.config import UERConfig

logger = logging.getLogger(__name__)


class LinearAlignment:
    """
    Learned linear alignment for cross-model consistency.

    Trains projection matrices to map embeddings from specific providers
    into the UER intermediate representation while preserving semantic structure.
    """

    def __init__(self, config: UERConfig, provider_name: Optional[str] = None):
        """
        Initialize linear alignment for a provider.

        Args:
            config: UER configuration
            provider_name: Name of the embedding provider (e.g., 'openai', 'cohere')
        """
        self.config = config
        self.provider_name = provider_name or 'unknown'
        self.projection_matrix = None
        self.is_fitted = False
        self.target_dim = config.vector_dim

        # Try to load existing alignment
        self._load_alignment_if_available()

    def align(self, embedding: np.ndarray) -> np.ndarray:
        """
        Apply learned linear alignment to embedding.

        Args:
            embedding: Input embedding from the provider

        Returns:
            Aligned embedding in UER space
        """
        if not self.is_fitted:
            # Fallback to simple dimension adjustment
            return self._fallback_align(embedding)

        # Apply learned projection
        current_dim = embedding.shape[-1]
        if current_dim != self.projection_matrix.shape[1]:
            logger.warning(f"Embedding dimension {current_dim} doesn't match projection matrix "
                         f"dimension {self.projection_matrix.shape[1]}, applying fallback")
            return self._fallback_align(embedding)

        aligned = embedding @ self.projection_matrix.T

        # Ensure target dimension
        if aligned.shape[-1] != self.target_dim:
            if aligned.shape[-1] < self.target_dim:
                # Pad with zeros if needed
                padding_shape = list(aligned.shape)
                padding_shape[-1] = self.target_dim - aligned.shape[-1]
                padding = np.zeros(padding_shape)
                aligned = np.concatenate([aligned, padding], axis=-1)
            else:
                # Truncate if too large
                aligned = aligned[..., :self.target_dim]

        return aligned

    def _fallback_align(self, embedding: np.ndarray) -> np.ndarray:
        """Simple fallback alignment for unfitted or mismatched matrices."""
        current_dim = embedding.shape[-1]
        target_dim = self.target_dim

        if current_dim == target_dim:
            return embedding
        elif current_dim < target_dim:
            # Zero-pad
            padding_shape = list(embedding.shape)
            padding_shape[-1] = target_dim - current_dim
            padding = np.zeros(padding_shape)
            return np.concatenate([embedding, padding], axis=-1)
        else:
            # Truncate
            return embedding[..., :target_dim]

    def _load_alignment_if_available(self) -> None:
        """Load saved alignment matrix if available."""
        # This would load from a provider-specific alignment store in Phase 2
        # For now, always use fallback
        pass

    fit = None  # Placeholder for learning alignment matrices


def derive_projection_matrix(source_embeddings: np.ndarray,
                           target_embeddings: np.ndarray,
                           method: str = 'least_squares') -> np.ndarray:
    """
    Learn projection matrix from source to target embedding space.

    Args:
        source_embeddings: Embeddings from source model (N, D_source)
        target_embeddings: Corresponding embeddings in target space (N, D_target)
        method: Alignment method ('least_squares', 'orthogonal')

    Returns:
        Projection matrix W such that source @ W ≈ target
    """
    # Basic least squares solution
    try:
        # Solve W * source.T = target.T for W
        W, residuals, rank, s = np.linalg.lstsq(source_embeddings, target_embeddings, rcond=None)
        logger.info(f"Learned projection matrix with shape {W.shape}, rank {rank}")

        return W

    except np.linalg.LinAlgError as e:
        logger.error(f"Failed to learn projection matrix: {e}")
        raise


def evaluate_alignment_quality(source_embeddings: np.ndarray,
                             projected_embeddings: np.ndarray,
                             target_embeddings: np.ndarray,
                             sample_pairs: int = 1000) -> Dict[str, float]:
    """
    Evaluate quality of learned alignment by measuring preserved relationships.

    Args:
        source_embeddings: Original source embeddings
        projected_embeddings: Aligned embeddings in UER space
        target_embeddings: Target UER embeddings
        sample_pairs: Number of pairs to sample for evaluation

    Returns:
        Dictionary with quality metrics
    """
    # Sample pairs for evaluation
    n_samples = min(sample_pairs, len(source_embeddings))
    indices = np.random.choice(len(source_embeddings), n_samples, replace=False)

    source_sample = source_embeddings[indices]
    projected_sample = projected_embeddings[indices]
    target_sample = target_embeddings[indices]

    # Compute cosine similarities
    def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms
        return normalized @ normalized.T

    source_sim = cosine_similarity_matrix(source_sample)
    projected_sim = cosine_similarity_matrix(projected_sample)
    target_sim = cosine_similarity_matrix(target_sample)

    # Correlation between similarity matrices
    correlation_source_target = np.corrcoef(source_sim.flatten(), target_sim.flatten())[0, 1]
    correlation_projected_target = np.corrcoef(projected_sim.flatten(), target_sim.flatten())[0, 1]

    return {
        'source_target_correlation': correlation_source_target,
        'projected_target_correlation': correlation_projected_target,
        'alignment_quality': correlation_projected_target / max(correlation_source_target, 1e-6),
        'samples_evaluated': n_samples
    }
