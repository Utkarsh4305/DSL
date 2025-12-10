"""
UER Provider Registry

Centralized management of embedding providers with their characteristics.
Placeholder for Phase 3 implementation.
"""

from typing import Dict, Any, Optional, List, Union
import json
import os
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """
    Registry for embedding providers and their characteristics.

    Manages provider metadata, alignment matrices, and distribution patterns.
    """

    def __init__(self):
        """Initialize provider registry."""
        self.providers = {}
        self._load_default_providers()

    def _load_default_providers(self) -> None:
        """Load default provider configurations."""
        # Placeholder for default providers
        pass

    def register_provider(self, name: str, config: Dict[str, Any]) -> None:
        """Register a new embedding provider."""
        profile = ProviderProfile(name, config)
        self.providers[name] = profile
        logger.info(f"Registered provider '{name}' with {profile.native_dimension}d embeddings")

    def get_provider(self, name: str) -> Optional[ProviderProfile]:
        """Get provider profile by name."""
        return self.providers.get(name)

    def get_provider_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a provider."""
        profile = self.get_provider(name)
        return profile.to_dict() if profile else None

    def list_providers(self) -> List[str]:
        """Get list of all registered provider names."""
        return list(self.providers.keys())

    def get_provider_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            'total_providers': len(self.providers),
            'dimensions': list(set(p.native_dimension for p in self.providers.values())),
            'vendors': list(set(p.metadata.get('vendor', 'Unknown')
                              for p in self.providers.values())),
            'providers_with_alignment': sum(1 for p in self.providers.values()
                                          if p.alignment_available)
        }

    def get_alignment_matrix(self, provider_name: str, target_dimension: int) -> Optional[Any]:
        """Get alignment matrix for a provider."""
        profile = self.get_provider(provider_name)
        if profile:
            return profile.get_alignment_for_target(target_dimension)
        return None

    def find_providers_by_dimension(self, dimension: int) -> List[str]:
        """Find providers with specific native dimension."""
        return [name for name, profile in self.providers.items()
                if profile.native_dimension == dimension]

    def recommend_alignment_approach(self, source_provider: str, target_config) -> Dict[str, Any]:
        """Recommend best alignment approach for source provider to target UER spec."""
        if source_provider not in self.providers:
            return {
                'approach': 'identity',
                'reason': 'Provider not registered, falling back to identity alignment',
                'confidence': 'low'
            }

        provider = self.providers[source_provider]
        target_dim = target_config.vector_dim

        # Check if exact alignment matrix is available
        alignment_matrix = provider.get_alignment_for_target(target_dim)
        if alignment_matrix is not None:
            return {
                'approach': 'pretrained_matrix',
                'matrix_shape': alignment_matrix.shape,
                'confidence': 'high',
                'alignment_type': 'linear_projection'
            }

        # Check if dimensions match
        if provider.native_dimension == target_dim:
            return {
                'approach': 'identity',
                'reason': 'Dimensions match exactly',
                'confidence': 'high'
            }

        # Recommend based on provider characteristics
        if provider.expected_distribution.get('anisotropic', False):
            return {
                'approach': 'linear_with_whitening',
                'reason': 'Provider has anisotropic embeddings, requires distribution correction',
                'confidence': 'medium'
            }

        # Default linear projection
        return {
            'approach': 'linear_projection',
            'reason': 'Standard linear alignment recommended',
            'confidence': 'medium'
        }


# Global registry instance
_registry = None

def get_provider_registry() -> ProviderRegistry:
    """Get the global provider registry."""
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry

def register_provider(name: str, config: Dict[str, Any]) -> None:
    """Global convenience function for provider registration."""
    get_provider_registry().register_provider(name, config)

def get_provider_alignment(provider_name: str, target_dimension: int) -> Optional[Any]:
    """Global convenience function for provider alignment."""
    return get_provider_registry().get_alignment_matrix(provider_name, target_dimension)

def get_provider_recommendations(provider_name: str) -> Dict[str, Any]:
    """Get processing recommendations for a provider."""
    registry = get_provider_registry()
    profile = registry.get_provider(provider_name)
    if profile:
        # Analyze preprocessing needs
        recommendations = profile.needs_preprocessing(np.random.randn(1, profile.native_dimension))
        return {
            'native_dimension': profile.native_dimension,
            'distribution': profile.expected_distribution,
            'quirks': profile.post_processing_quirks,
            'alignment_available': profile.alignment_available,
            'preprocessing_recommendations': recommendations
        }
    return {}

def auto_detect_provider(embeddings: np.ndarray) -> List[str]:
    """
    Attempt to auto-detect provider based on embedding characteristics.

    Args:
        embeddings: Sample embeddings

    Returns:
        List of potential provider matches
    """
    registry = get_provider_registry()
    candidates = []

    # Simple heuristics based on dimension and distribution
    dim = embeddings.shape[-1]
    mean_norm = np.mean(np.linalg.norm(embeddings, axis=1))

    for name, profile in registry.providers.items():
        if profile.native_dimension == dim:
            norm_range = profile.expected_distribution.get('norm_range', [0, 2])
            if norm_range[0] <= mean_norm <= norm_range[1]:
                candidates.append(name)

    return candidates[:5]  # Top 5 candidates


class ProviderProfile:
    """
    Comprehensive profile for an embedding provider.

    Captures all characteristics needed for optimal UER compilation.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize provider profile.

        Args:
            name: Provider name (e.g., 'openai-ada002')
            config: Provider configuration dictionary
        """
        self.name = name
        self.native_dimension = config.get('native_dimension', 768)
        self.expected_distribution = config.get('distribution', {
            'anisotropic': False,
            'hubness': 'medium',
            'norm_range': [0.8, 1.2]
        })
        self.post_processing_quirks = config.get('quirks', [])
        self.alignment_available = config.get('alignment_available', False)
        self.alignment_matrix_path = config.get('alignment_matrix_path')
        self.alignment_matrix = None
        self.metadata = config.get('metadata', {})

        # Load alignment matrix if available
        if self.alignment_matrix_path:
            self._load_alignment_matrix()

    def _load_alignment_matrix(self) -> None:
        """Load pre-trained alignment matrix from disk."""
        if not os.path.exists(self.alignment_matrix_path):
            logger.warning(f"Alignment matrix not found: {self.alignment_matrix_path}")
            return

        try:
            # Assume .npy file for now
            if self.alignment_matrix_path.endswith('.npy'):
                self.alignment_matrix = np.load(self.alignment_matrix_path)
            elif self.alignment_matrix_path.endswith('.json'):
                # Placeholder for serialized alignment
                with open(self.alignment_matrix_path, 'r') as f:
                    alignment_data = json.load(f)
                    self.alignment_matrix = np.array(alignment_data['matrix'])

            logger.info(f"Loaded alignment matrix for {self.name}: {self.alignment_matrix.shape}")

        except Exception as e:
            logger.error(f"Failed to load alignment matrix for {self.name}: {e}")

    def get_alignment_for_target(self, target_dim: int, source_dim: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Get alignment matrix for specific target dimension.

        Args:
            target_dim: Target UER dimension
            source_dim: Source dimension (uses native_dimension if None)

        Returns:
            Alignment matrix or None if not available
        """
        if source_dim is None:
            source_dim = self.native_dimension

        if self.alignment_matrix is not None:
            matrix_source, matrix_target = self.alignment_matrix.shape
            if matrix_source == source_dim and matrix_target == target_dim:
                return self.alignment_matrix

        # Return None if no suitable matrix available
        return None

    def needs_preprocessing(self, embedding: np.ndarray) -> Dict[str, Any]:
        """
        Analyze embedding and recommend preprocessing steps.

        Args:
            embedding: Input embedding from this provider

        Returns:
            Dictionary of preprocessing recommendations
        """
        recommendations = {}

        # Check norm
        norm = np.linalg.norm(embedding)
        norm_range = self.expected_distribution.get('norm_range', [0.5, 1.5])

        if norm < norm_range[0] or norm > norm_range[1]:
            recommendations['normalize'] = 'l2'

        # Check for known quirks
        if 'unnormalized_output' in self.post_processing_quirks:
            recommendations['normalize'] = 'l2'

        if 'asymmetric_similarity' in self.post_processing_quirks:
            recommendations['fix_asymmetry'] = True

        if 'temperature_dependent' in self.post_processing_quirks:
            recommendations['clamp_range'] = [-1.0, 1.0]

        return recommendations

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary for serialization."""
        return {
            'name': self.name,
            'native_dimension': self.native_dimension,
            'distribution': self.expected_distribution,
            'quirks': self.post_processing_quirks,
            'alignment_available': self.alignment_available,
            'alignment_matrix_path': self.alignment_matrix_path,
            'metadata': self.metadata
        }
