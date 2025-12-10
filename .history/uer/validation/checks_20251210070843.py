"""
UER Advanced Validation Checks

Anisotropy detection, distribution analysis, semantic preservation validation.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class AnisotropyChecker:
    """
    Detects and measures embedding space anisotropy.

    Anisotropic spaces have non-uniform dimensional importance,
    leading to unreliable nearest neighbor searches.
    """

    def __init__(self, sigma_threshold: float = 100.0):
        """
        Initialize anisotropy checker.

        Args:
            sigma_threshold: Ratio threshold for anisotropy detection
        """
        self.sigma_threshold = sigma_threshold

    def check_anisotropy(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Compute anisotropy metrics for embedding space.

        Args:
            embeddings: Batch of embeddings (N, D)

        Returns:
            Anisotropy assessment metrics
        """
        # Compute covariance matrix
        cov_matrix = np.cov(embeddings.T)

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvalues in descending order
        eigenvalues = np.sort(eigenvalues)[::-1]

        # Remove zero/near-zero eigenvalues
        positive_eigenvals = eigenvalues[eigenvalues > 1e-10]

        if len(positive_eigenvals) < 2:
            return {
                'is_anisotropic': False,
                'anisotropy_ratio': 1.0,
                'max_eigenvalue_ratio': 1.0,
                'effective_dimensions': len(positive_eigenvals),
                'warning': 'Too few non-zero eigenvalues for reliable anisotropy assessment'
            }

        # Anisotropy ratio (largest to smallest eigenvalue)
        anisotropy_ratio = positive_eigenvals[0] / positive_eigenvals[-1]

        # Maximum eigenvalue ratio (for hubness detection)
        max_ratio = positive_eigenvals[0] / np.mean(positive_eigenvals)

        # Effective dimensionality using participation ratio
        participation_ratio = np.sum(positive_eigenvals)**2 / np.sum(positive_eigenvals**2)
        effective_dim = len(positive_eigenvals) / participation_ratio

        is_anisotropic = anisotropy_ratio > self.sigma_threshold

        return {
            'is_anisotropic': is_anisotropic,
            'anisotropy_ratio': float(anisotropy_ratio),
            'max_eigenvalue_ratio': float(max_ratio),
            'effective_dimensions': float(effective_dim),
            'total_dimensions': embeddings.shape[1],
            'eigenvalue_spectrum': positive_eigenvals[:10].tolist(),  # First 10 for inspection
            'recommendation': 'Consider anisotropy correction' if is_anisotropic else 'Space is isotropic'
        }


class DistributionChecker:
    """
    Validates embedding distribution characteristics.

    Ensures embeddings follow reasonable statistical distributions.
    """

    def check_distribution(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Validate embedding distribution statistics.

        Args:
            embeddings: Batch of embeddings (N, D)

        Returns:
            Distribution health assessment
        """
        norms = np.linalg.norm(embeddings, axis=1)

        # Norm statistics
        norm_mean = float(np.mean(norms))
        norm_std = float(np.std(norms))
        norm_min = float(np.min(norms))
        norm_max = float(np.max(norms))

        # For spherical geometry, norms should be ~1.0
        expected_norm = 1.0
        norm_deviation = abs(norm_mean - expected_norm)

        # Mean and std across dimensions
        dim_means = np.mean(embeddings, axis=0)
        dim_stds = np.std(embeddings, axis=0)

        overall_mean = float(np.mean(dim_means))
        overall_std = float(np.std(dim_stds))

        # Check for problematic distributions
        warnings = []

        if norm_deviation > 0.1:
            warnings.append('.2f')

        if np.any(norms < 1e-6):
            zero_count = np.sum(norms < 1e-6)
            warnings.append(f"Found {zero_count} near-zero norm embeddings")

        if overall_std > 0.5 and np.any(dim_stds < 0.01):
            warnings.append("Some dimensions have near-zero variance - consider whitening")

        return {
            'norm_stats': {
                'mean': norm_mean,
                'std': norm_std,
                'min': norm_min,
                'max': norm_max,
                'expected': expected_norm,
                'deviation': norm_deviation
            },
            'dimension_stats': {
                'mean_across_dims': overall_mean,
                'std_across_dims': overall_std,
                'dimensions_with_low_var': int(np.sum(dim_stds < 0.01))
            },
            'warnings': warnings,
            'distribution_healthy': len(warnings) == 0
        }


class SemanticChecker:
    """
    Validates semantic preservation in aligned embeddings.

    Ensures that alignment operations maintain meaningful relationships.
    """

    def __init__(self, k_neighbors: int = 10):
        """
        Initialize semantic checker.

        Args:
            k_neighbors: Number of neighbors to check for preservation
        """
        self.k_neighbors = k_neighbors

    def check_semantic_preservation(self,
                                   source_embeddings: np.ndarray,
                                   aligned_embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate how well semantic relationships are preserved through alignment.

        Args:
            source_embeddings: Original embeddings (N, D_source)
            aligned_embeddings: Aligned embeddings (N, D_target)

        Returns:
            Semantic preservation metrics
        """
        from sklearn.metrics.pairwise import cosine_similarity

        # Sample a subset to avoid quadratic complexity
        n_samples = min(1000, len(source_embeddings))
        indices = np.random.choice(len(source_embeddings), n_samples, replace=False)

        source_sample = source_embeddings[indices]
        aligned_sample = aligned_embeddings[indices]

        # Compute similarity matrices
        source_sim = cosine_similarity(source_sample)
        aligned_sim = cosine_similarity(aligned_sample)

        # For each embedding, find k nearest neighbors in both spaces
        preservation_scores = []

        for i in range(n_samples):
            # Get neighbor indices (excluding self)
            source_neighbors = np.argsort(source_sim[i])[::-1][1:self.k_neighbors+1]
            aligned_neighbors = np.argsort(aligned_sim[i])[::-1][1:self.k_neighbors+1]

            # Jaccard similarity between neighbor sets
            intersection = len(set(source_neighbors) & set(aligned_neighbors))
            union = len(set(source_neighbors) | set(aligned_neighbors))
            jaccard = intersection / union if union > 0 else 0.0

            preservation_scores.append(jaccard)

        avg_preservation = float(np.mean(preservation_scores))
        std_preservation = float(np.std(preservation_scores))

        return {
            'avg_neighbor_preservation': avg_preservation,
            'std_neighbor_preservation': std_preservation,
            'semantic_preservation_healthy': avg_preservation > 0.3,  # 30% overlap threshold
            'sample_size': n_samples,
            'k_neighbors': self.k_neighbors,
            'recommendation': 'Good preservation' if avg_preservation > 0.5 else 'Consider better alignment'
        }
