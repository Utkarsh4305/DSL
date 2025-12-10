"""
UER Procrustes Alignment

Optimal orthogonal alignment using classical Procrustes analysis.
Finds optimal rotation/translation to align source embeddings to target space
while preserving all geometric relationships (distances, angles, neighborhoods).
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class ProcrustesAlignment:
    """
    Classical Procrustes analysis for embedding alignment.

    Mathematically optimal solution for mapping one point cloud to another
    in a way that minimizes squared error while preserving distances/angles.
    """

    def __init__(self):
        """Initialize Procrustes alignment."""
        self.rotation_matrix = None
        self.translation_vector = None
        self.scale_factor = None
        self.is_fitted = False
        self.alignment_quality = {}

    def fit(self, source_points: np.ndarray, target_points: np.ndarray,
            allow_scaling: bool = True, allow_translation: bool = True,
            center_data: bool = True) -> Dict[str, Any]:
        """
        Fit Procrustes alignment between source and target point clouds.

        Args:
            source_points: Source embeddings (N, D)
            target_points: Target embeddings (N, D) - same N
            allow_scaling: Whether to allow uniform scaling
            allow_translation: Whether to allow translation
            center_data: Whether to center data before alignment

        Returns:
            Dictionary with alignment quality metrics
        """
        if source_points.shape != target_points.shape:
            raise ValueError(f"Source and target point clouds must have same shape. "
                           f"Got {source_points.shape} vs {target_points.shape}")

        if source_points.shape[0] < source_points.shape[1]:
            logger.warning("Fewer samples than dimensions, Procrustes alignment may be unstable")

        # Store original shapes
        n_points, n_dims = source_points.shape

        # Center the data
        source_centered, target_centered = source_points.copy(), target_points.copy()
        source_mean, target_mean = None, None

        if center_data:
            source_mean = np.mean(source_points, axis=0)
            target_mean = np.mean(target_points, axis=0)
            source_centered -= source_mean
            target_centered -= target_mean

        # Scale to unit variance (optional uniform scaling)
        source_scale, target_scale = 1.0, 1.0

        if allow_scaling:
            source_scale = np.sqrt(np.sum(source_centered**2) / (n_points * n_dims))
            target_scale = np.sqrt(np.sum(target_centered**2) / (n_points * n_dims))

            if source_scale > 0:
                source_centered /= source_scale
            if target_scale > 0:
                target_centered /= target_scale

        # Compute covariance matrix (cross-correlation)
        covariance = source_centered.T @ target_centered

        # Singular Value Decomposition
        try:
            U, s, Vt = np.linalg.svd(covariance)
        except np.linalg.LinAlgError:
            logger.error("SVD failed during Procrustes fitting, using identity alignment")
            self.rotation_matrix = np.eye(n_dims)
            self.translation_vector = np.zeros(n_dims)
            self.scale_factor = 1.0
            self.is_fitted = True
            return {'alignment_error': float('inf'), 'convergence': False}

        # Construct optimal rotation matrix
        self.rotation_matrix = U @ Vt

        # Ensure proper rotation (not reflection) by checking determinant
        if np.linalg.det(self.rotation_matrix) < 0:
            # Flip the smallest singular value
            Vt = Vt.copy()
            Vt[-1] *= -1  # Flip last column
            self.rotation_matrix = U @ Vt

        # Scaling factor
        if allow_scaling:
            self.scale_factor = np.sum(s) / np.sum(source_centered**2 / n_points)
            # Alternative: self.scale_factor = target_scale / source_scale if source_scale > 0 else 1.0
        else:
            self.scale_factor = 1.0

        # Translation vector
        if allow_translation and source_mean is not None and target_mean is not None:
            self.translation_vector = target_mean - self.scale_factor * (self.rotation_matrix @ source_mean)
        else:
            self.translation_vector = np.zeros(n_dims)

        self.is_fitted = True

        # Compute alignment quality
        quality_metrics = self._compute_alignment_quality(source_points, target_points)

        logger.info(f"Procrustes alignment fitted - "
                   f"Rotation det: {np.linalg.det(self.rotation_matrix):.3f}, "
                   f"Scale: {self.scale_factor:.3f}, "
                   f"MSE: {quality_metrics['mse']:.6f}")

        return quality_metrics

    def transform(self, points: np.ndarray) -> np.ndarray:
        """
        Apply learned Procrustes transformation to new points.

        Args:
            points: Points to transform (M, D)

        Returns:
            Transformed points
        """
        if not self.is_fitted:
            raise RuntimeError("Procrustes alignment must be fitted before transform")

        # Apply: scale * (rotation @ points) + translation
        transformed = self.scale_factor * (points @ self.rotation_matrix.T)

        if self.translation_vector is not None:
            transformed += self.translation_vector

        return transformed

    def inverse_transform(self, points: np.ndarray) -> np.ndarray:
        """
        Apply inverse Procrustes transformation.

        Useful for mapping back to original space.
        """
        if not self.is_fitted:
            raise RuntimeError("Procrustes alignment must be fitted before inverse_transform")

        # Inverse: (points - translation) @ rotation.T / scale
        if self.translation_vector is not None:
            points_centered = points - self.translation_vector
        else:
            points_centered = points

        if self.scale_factor != 0:
            points_centered /= self.scale_factor

        # Apply transpose of rotation (since R is orthogonal, R^T = R^-1)
        original = points_centered @ self.rotation_matrix

        return original

    def _compute_alignment_quality(self, source_points: np.ndarray,
                                  target_points: np.ndarray) -> Dict[str, Any]:
        """
        Compute comprehensive alignment quality metrics.

        Measures how well the transformation preserves geometric relationships.
        """
        # Transform source points
        aligned_source = self.transform(source_points)

        # Basic alignment error
        mse = np.mean(np.sum((aligned_source - target_points)**2, axis=1))
        rmse = np.sqrt(mse)
        mae = np.mean(np.sum(np.abs(aligned_source - target_points), axis=1))

        # Distance preservation (how well pairwise distances are maintained)
        # Use numpy broadcasting for efficient distance calculation
        source_norm = np.sum(source_points**2, axis=1, keepdims=True)
        target_norm = np.sum(target_points**2, axis=1, keepdims=True)
        cross_term = source_points @ target_points.T

        source_distances = np.sqrt(
            source_norm - 2 * cross_term + source_norm.T
        )
        target_distances = np.sqrt(
            target_norm - 2 * cross_term + target_norm.T
        )
        aligned_norm = np.sum(aligned_source**2, axis=1, keepdims=True)
        aligned_distances = np.sqrt(
            aligned_norm - 2 * (aligned_source @ target_points.T) + target_norm.T
        )

        # Distance correlation (how well alignment preserves relative distances)
        # Sample a subset to avoid memory issues with large matrices
        n_points = len(source_points)
        if n_points > 1000:
            sample_indices = np.random.choice(n_points, 1000, replace=False)
            source_sample = source_distances[np.ix_(sample_indices, sample_indices)]
            aligned_sample = aligned_distances[np.ix_(sample_indices, sample_indices)]
        else:
            source_sample = source_distances
            aligned_sample = aligned_distances

        dist_preservation = np.corrcoef(
            source_sample.flatten(),
            aligned_sample.flatten()
        )[0, 1]

        # Neighborhood preservation (k-NN accuracy)
        k_values = [1, 3, 5, 10]

        neighborhood_metrics = {}
        for k in k_values:
            if source_points.shape[0] > k:
                # Find k nearest neighbors in source space
                source_neighbors = self._compute_knn(source_points, source_points, k)

                # Find k nearest neighbors in aligned space (to target space)
                aligned_neighbors = self._compute_knn(aligned_source, target_points, k)

                # Compute agreement
                intersection = np.array([len(set(src) & set(aln))
                                       for src, aln in zip(source_neighbors, aligned_neighbors)])
                neighborhood_metrics[f'knn_{k}_overlap'] = float(np.mean(intersection / k))

        # Converged check (determinant should be close to 1 for proper rotations)
        det_rotation = abs(np.linalg.det(self.rotation_matrix))
        converged = 0.9 <= det_rotation <= 1.1

        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'distance_preservation': float(dist_preservation),
            'neighborhood_preservation': neighborhood_metrics,
            'convergence': converged,
            'rotation_determinant': float(det_rotation),
            'scale_factor': float(self.scale_factor),
            'translation_magnitude': float(np.linalg.norm(self.translation_vector)) if self.translation_vector is not None else 0.0
        }

    def _compute_knn(self, query_points: np.ndarray, search_points: np.ndarray,
                     k: int) -> List[List[int]]:
        """
        Compute k-nearest neighbors for each query point.

        Args:
            query_points: Points to find neighbors for
            search_points: Points to search in
            k: Number of neighbors

        Returns:
            List of lists containing neighbor indices
        """
        distances = cdist(query_points, search_points, metric='euclidean')
        # Don't include self in neighbors by setting diagonal to infinity
        np.fill_diagonal(distances, np.inf)

        # Get k smallest indices (ignoring self)
        neighbor_indices = []
        for i in range(len(query_points)):
            # Get k+1 neighbors (may include self, but we'll exclude it)
            neighbors = np.argsort(distances[i])[:k+1]

            # Exclude self if present
            if i in neighbors:
                neighbors = neighbors[neighbors != i][:k]

            neighbor_indices.append(neighbors[:k].tolist())

        return neighbor_indices

    def get_transformation_matrix(self) -> np.ndarray:
        """Get the full affine transformation matrix (if needed for compatibility)."""
        if not self.is_fitted:
            raise RuntimeError("Alignment not fitted")

        n_dims = self.rotation_matrix.shape[0]

        # Create affine transformation matrix: [R, t; 0, 1] * scale
        transformation = np.eye(n_dims + 1)
        transformation[:n_dims, :n_dims] = self.scale_factor * self.rotation_matrix
        if self.translation_vector is not None:
            transformation[:n_dims, n_dims] = self.translation_vector

        return transformation


def procrustes_align(source_embeddings: np.ndarray, target_embeddings: np.ndarray,
                    **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convenience function for Procrustes alignment.

    Args:
        source_embeddings: Source embeddings (N, D)
        target_embeddings: Target embeddings (N, D)
        **kwargs: Additional arguments for fit()

    Returns:
        Tuple of (rotation_matrix, alignment_info)
    """
    aligner = ProcrustesAlignment()
    quality = aligner.fit(source_embeddings, target_embeddings, **kwargs)

    return aligner.rotation_matrix, {
        'rotation_matrix': aligner.rotation_matrix,
        'translation_vector': aligner.translation_vector,
        'scale_factor': aligner.scale_factor,
        'alignment_quality': quality
    }


class IterativeProcrustes:
    """
    Iterative Procrustes alignment for noisy or imperfect data.

    Uses multiple random subsets to find robust alignment.
    """

    def __init__(self, n_iterations: int = 10, subset_fraction: float = 0.8):
        """
        Initialize iterative Procrustes.

        Args:
            n_iterations: Number of iterations with different subsets
            subset_fraction: Fraction of data to use per iteration
        """
        self.n_iterations = n_iterations
        self.subset_fraction = subset_fraction
        self.best_alignment = None
        self.best_quality = -float('inf')

    def fit(self, source_points: np.ndarray, target_points: np.ndarray) -> Dict[str, Any]:
        """
        Find best alignment using iterative subset selection.

        Args:
            source_points: Source embeddings
            target_points: Target embeddings

        Returns:
            Best alignment information
        """
        n_points = len(source_points)

        for iteration in range(self.n_iterations):
            # Select random subset
            subset_size = int(n_points * self.subset_fraction)
            indices = np.random.choice(n_points, subset_size, replace=False)

            # Fit Procrustes on subset
            aligner = ProcrustesAlignment()
            quality = aligner.fit(source_points[indices], target_points[indices])

            # Evaluate on full dataset
            aligned_full = aligner.transform(source_points)
            full_quality = aligner._compute_alignment_quality(source_points, target_points)

            # Keep best alignment
            score = full_quality.get('distance_preservation', 0)
            if score > self.best_quality:
                self.best_quality = score
                self.best_alignment = aligner

                logger.debug(f"Iteration {iteration}: Improved alignment score to {score:.4f}")

        return self.best_alignment._compute_alignment_quality(source_points, target_points)

    def transform(self, points: np.ndarray) -> np.ndarray:
        """Transform points using best alignment."""
        if self.best_alignment is None:
            raise RuntimeError("Iterative Procrustes must be fitted first")
        return self.best_alignment.transform(points)


# Geometry preserving alignment functions
def rigid_align(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rigid alignment (rotation + translation, no scaling).

    Preserves distances and angles perfectly.
    """
    return procrustes_align(source, target, allow_scaling=False)


def similarity_align(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Similarity alignment (rotation + translation + uniform scaling).

    Preserves shape but allows overall scaling.
    """
    return procrustes_align(source, target, allow_scaling=True)


def affine_align(source: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
    """
    General affine alignment using least squares.

    More flexible but may not preserve geometry.
    """
    from ..alignment.linear import derive_projection_matrix

    # Add bias term (affine transformation)
    source_homogeneous = np.column_stack([source, np.ones(len(source))])

    # Solve for affine transformation matrix
    W, residuals, rank, s = np.linalg.lstsq(source_homogeneous, target, rcond=None)

    # Extract components
    rotation_scaling = W[:-1]  # Affine part
    translation = W[-1]       # Translation part

    return {
        'transformation_matrix': W,
        'rotation_scaling': rotation_scaling,
        'translation': translation,
        'residuals': float(np.sum(residuals)) if residuals.size > 0 else 0.0,
        'rank': rank
    }
