"""
UER Validation Tests

Comprehensive tests for UER embedding validation functionality.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add uer package to path
uer_path = Path(__file__).parent.parent / 'uer'
sys.path.insert(0, str(uer_path))

from uer.validator import UERValidator, validate_uer_embedding
from uer.loader import load_uer_spec


class TestUERValidation:
    """Test cases for UER validation functionality."""

    @pytest.fixture
    def sample_spec(self):
        """Load the v0.1 UER spec for testing."""
        spec_file = Path(__file__).parent.parent / 'specs' / 'uer_v0.1.yaml'
        return load_uer_spec(spec_file)

    @pytest.fixture
    def validator(self, sample_spec):
        """Create a validator instance."""
        return UERValidator(sample_spec)

    def test_valid_embedding(self, validator, sample_spec):
        """Test validation of a valid UER embedding."""
        # Create a valid embedding (normalized, correct dim, dtype)
        embedding = np.random.randn(sample_spec['vector_dimension'])
        embedding = embedding / np.linalg.norm(embedding)  # L2 normalize
        embedding = embedding.astype(np.dtype(sample_spec['dtype']))

        # Should validate successfully
        assert validator.validate_embedding(embedding, strict=True) == True

    def test_dimension_validation(self, validator):
        """Test dimension validation."""
        # Wrong dimension
        wrong_dim = np.random.randn(512).astype(np.float32)  # Spec expects 768
        with pytest.raises(ValueError, match="Expected dimension 768"):
            validator.validate_embedding(wrong_dim, strict=True)

        # Check tolerance
        validator._dim_tolerance = 1.0  # Allow some tolerance
        close_dim = np.random.randn(769)  # Within tolerance
        # This would pass if tolerance allows, but our spec has 0.0 tolerance

    def test_dtype_validation(self, sample_spec):
        """Test dtype validation."""
        validator_instance = UERValidator(sample_spec)

        # Wrong dtype
        wrong_dtype = np.random.randn(768).astype(np.int32)  # Spec expects float32
        with pytest.raises(ValueError, match="Expected dtype"):
            validator_instance.validate_embedding(wrong_dtype, strict=True)

        # Correct dtype
        correct_dtype = np.random.randn(768).astype(np.float32)
        correct_dtype = correct_dtype / np.linalg.norm(correct_dtype)
        assert validator_instance.validate_embedding(correct_dtype, strict=True)

    def test_normalization_validation(self, validator, sample_spec):
        """Test normalization validation for L2."""
        embedding = np.random.randn(sample_spec['vector_dimension']).astype(np.float32)

        # Non-normalized embedding should fail
        with pytest.raises(ValueError, match="L2 normalization violation"):
            validator.validate_embedding(embedding, strict=True)

        # Normalized embedding should pass
        normalized = embedding / np.linalg.norm(embedding)
        assert validator.validate_embedding(normalized, strict=True)

    def test_finite_validation(self, validator, sample_spec):
        """Test NaN/inf validation."""
        # Valid embedding
        embedding = np.random.randn(sample_spec['vector_dimension']).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        # Add NaN
        nan_embedding = embedding.copy()
        nan_embedding[0] = np.nan
        with pytest.raises(ValueError, match="NaN values"):
            validator.validate_embedding(nan_embedding, strict=True)

        # Add inf
        inf_embedding = embedding.copy()
        inf_embedding[0] = np.inf
        with pytest.raises(ValueError, match="infinite values"):
            validator.validate_embedding(inf_embedding, strict=True)

    def test_batch_validation(self, validator, sample_spec):
        """Test batch embedding validation."""
        batch_size = 10
        embeddings = np.random.randn(batch_size, sample_spec['vector_dimension']).astype(np.float32)

        # Normalize each embedding
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_batch = embeddings / norms

        # Should validate successfully
        assert validator.validate_embedding(normalized_batch, strict=True)

        # Test batch with one invalid embedding
        invalid_batch = normalized_batch.copy()
        invalid_batch[0] = np.random.randn(sample_spec['vector_dimension'])  # Non-normalized
        with pytest.raises(ValueError):
            validator.validate_embedding(invalid_batch, strict=True)

    def test_convenience_function(self, sample_spec):
        """Test the convenience validation function."""
        embedding = np.random.randn(sample_spec['vector_dimension']).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        # Should work
        assert validate_uer_embedding(embedding, sample_spec)

        # Should raise on invalid
        invalid = embedding * 2  # Non-normalized
        with pytest.raises(ValueError):
            validate_uer_embedding(invalid, sample_spec)

    def test_non_strict_mode(self, validator, sample_spec):
        """Test non-strict validation mode."""
        invalid_embedding = np.random.randn(512).astype(np.float32)  # Wrong dim, non-normalized

        # Strict mode should raise
        with pytest.raises(ValueError):
            validator.validate_embedding(invalid_embedding, strict=True)

        # Non-strict mode should return False
        assert validator.validate_embedding(invalid_embedding, strict=False) == False


class TestSpecValidation:
    """Test specification validation."""

    def test_missing_required_fields(self):
        """Test spec validation with missing fields."""
        from uer.loader import UERSpecLoader

        loader = UERSpecLoader()

        # Spec missing required field
        invalid_spec = {
            'vector_dimension': 768,
            'normalization_rules': {'method': 'l2'},
            'metric': 'cosine',
            'dtype': 'float32',
            # Missing 'uer_version', 'geometry'
        }

        with pytest.raises(ValueError, match="missing required fields"):
            loader._validate_spec_structure(invalid_spec)

    def test_invalid_version(self):
        """Test invalid version validation."""
        from uer.loader import UERSpecLoader

        loader = UERSpecLoader()

        invalid_spec = {
            'uer_version': 123,  # Should be string
            'vector_dimension': 768,
            'normalization_rules': {'method': 'l2'},
            'metric': 'cosine',
            'dtype': 'float32',
            'geometry': 'spherical'
        }

        with pytest.raises(ValueError, match="uer_version must be a string"):
            loader._validate_spec_structure(invalid_spec)

    def test_invalid_dimension(self):
        """Test invalid dimension validation."""
        from uer.loader import UERSpecLoader

        loader = UERSpecLoader()

        invalid_spec = {
            'uer_version': "0.1.0",
            'vector_dimension': 0,  # Invalid
            'normalization_rules': {'method': 'l2'},
            'metric': 'cosine',
            'dtype': 'float32',
            'geometry': 'spherical'
        }

        with pytest.raises(ValueError, match="vector_dimension must be a positive integer"):
            loader._validate_spec_structure(invalid_spec)


if __name__ == "__main__":
    pytest.main([__file__])
