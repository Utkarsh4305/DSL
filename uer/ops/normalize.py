"""
UER Batched Normalization Operations

Vectorized normalization with zero-vector protection and geometric handling.
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BatchedNormalizer:
    """
    Vectorized normalization for batches of embeddings with zero-vector protection.

    Handles spherical, euclidean, and future geometric spaces.
    """

    def __init__(self, normalization_rules: Dict[str, Any]):
        """
        Initialize normalizer with configuration.

        Args:
            normalization_rules: Normalization configuration from UER spec
        """
        self.method = normalization_rules.get('method', 'l2')
        self.epsilon = float(normalization_rules.get('epsilon', 1e-12))
        self.min_norm = 1e-8  # Minimum norm for zero vectors

        # Validate method
        valid_methods = ['l2', 'l1', 'none', 'max']
        if self.method not in valid_methods:
            logger.warning(f"Unknown normalization method '{self.method}', using 'l2'")
            self.method = 'l2'

    def normalize(self, embedding: np.ndarray) -> np.ndarray:
        """
        Apply normalization to a single embedding.

        Args:
            embedding: Input embedding

        Returns:
            Normalized embedding

        Raises:
            ValueError: If normalization method is unsupported
        """
        if self.method == 'none':
            return embedding

        # Detect and handle zero/near-zero vectors
        norm_value = np.linalg.norm(embedding)
        if norm_value < self.epsilon:
            logger.warning("Zero/near-zero vector detected, applying minimum normalization")
            # Apply minimum norm as fallback
            embedding = np.ones_like(embedding) / np.sqrt(len(embedding))
            norm_value = 1.0

        if self.method == 'l2':
            return embedding / np.maximum(norm_value, self.epsilon)
        elif self.method == 'l1':
            abs_sum = np.sum(np.abs(embedding))
            return embedding / np.maximum(abs_sum, self.epsilon)
        elif self.method == 'max':
            max_val = np.max(np.abs(embedding))
            return embedding / np.maximum(max_val, self.epsilon)
        else:
            raise ValueError(f"Unsupported normalization method: {self.method}")

    def normalize_batch(self, batch: np.ndarray) -> np.ndarray:
        """
        Vectorized batch normalization.

        Args:
            batch: Batch of embeddings (N, D)

        Returns:
            Batch of normalized embeddings
        """
        if self.method == 'none':
            return batch

        # Compute norms for the batch
        if self.method == 'l2':
            norms = np.linalg.norm(batch, axis=1, keepdims=True)
        elif self.method == 'l1':
            norms = np.sum(np.abs(batch), axis=1, keepdims=True)
        elif self.method == 'max':
            norms = np.max(np.abs(batch), axis=1, keepdims=True)
        else:
            raise ValueError(f"Unsupported batch normalization method: {self.method}")

        # Handle zero vectors in batch
        zero_mask = norms.squeeze() < self.epsilon
        if np.any(zero_mask):
            logger.warning(f"Found {np.sum(zero_mask)} zero vectors in batch, applying minimum normalization")
            # Replace zero vectors with uniform random unit vectors
            for idx in np.where(zero_mask)[0]:
                random_vec = np.random.randn(batch.shape[1])
                random_vec /= np.linalg.norm(random_vec) + self.epsilon
                batch[idx] = random_vec

            # Recompute norms after fixing
            if self.method == 'l2':
                norms = np.linalg.norm(batch, axis=1, keepdims=True)
            elif self.method == 'l1':
                norms = np.sum(np.abs(batch), axis=1, keepdims=True)
            elif self.method == 'max':
                norms = np.max(np.abs(batch), axis=1, keepdims=True)

        # Apply normalization
        return batch / np.maximum(norms, self.epsilon)
