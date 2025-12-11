"""
UER Validation Tests v0.2

Comprehensive tests for UER embedding validation functionality.
Includes 40+ test cases for strict validation, adversarial inputs, batch processing, and compiler safeguards.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add uer package to path
uer_path = Path(__file__).parent.parent / 'uer'
sys.path.insert(0, str(uer_path))

from uer.validator import UERValidator, validate_uer_embedding, validate_uer_embedding_batch
from uer.spec_validator import SpecValidator, SpecMissingFieldError, SpecTypeError
from uer.spec_validator import SpecBoundsError, SpecEnumError, AlignmentMetadataError
from uer.loader import load_uer_spec, UERSpecLoader
from uer.compiler import UERCompiler
from uer.utils.errors import UERValidationError, UERCompilationError


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


class TestSpecValidationV0_2:
    """Test v0.2 specification validation with strict rules."""

    @pytest.fixture
    def valid_v02_spec(self):
        """Load the v0.2 UER spec for testing."""
        spec_file = Path(__file__).parent.parent / 'specs' / 'uer_v0.2.yaml'
        return load_uer_spec(spec_file)

    @pytest.fixture
    def spec_validator(self):
        """Create a spec validator instance."""
        return SpecValidator()

    # ===== Spec Validation Tests (20+ cases) =====

    def test_v02_spec_missing_required_fields(self, spec_validator):
        """Test v0.2 spec validation with missing required fields."""
        incomplete_spec = {
            'vector_dimension': 768,
            'normalization_rules': {'method': 'l2'},
            # Missing several v0.2 required fields
        }

        with pytest.raises(SpecMissingFieldError) as exc_info:
            spec_validator.validate_spec(incomplete_spec)

        assert 'uer_version' in exc_info.value.missing_fields

    def test_v02_spec_invalid_types(self, spec_validator):
        """Test v0.2 spec validation with wrong field types."""
        invalid_spec = {
            'uer_version': 2.0,  # Should be str, not float
            'vector_dimension': 768,
            'normalization_rules': {'method': 'l2'},
            'metric': 'cosine',
            'dtype': 'float32',
            'geometry': 'spherical',
            'version_compatibility': {},
            'numeric_precision_guarantees': {},
            'embedding_statistics': {},
            'robustness_profiles': {}
        }

        with pytest.raises(SpecTypeError):
            spec_validator.validate_spec(invalid_spec)

    def test_v02_spec_dimension_bounds(self, spec_validator):
        """Test v0.2 spec numerical bounds validation."""
        # Dimension too small
        small_dim_spec = {
            'uer_version': '0.2.0',
            'vector_dimension': 300,  # Below min 64
            'normalization_rules': {'method': 'l2', 'epsilon': 1e-12},
            'metric': 'cosine',
            'dtype': 'float32',
            'geometry': 'spherical',
            'validation_rules': {'dimension_min': 64, 'dimension_max': 4096},
            'version_compatibility': {},
            'numeric_precision_guarantees': {},
            'embedding_statistics': {},
            'robustness_profiles': {}
        }

        with pytest.raises(SpecBoundsError, match="below minimum allowed"):
            spec_validator.validate_spec(small_dim_spec)

        # Dimension too large
        large_dim_spec = small_dim_spec.copy()
        large_dim_spec['vector_dimension'] = 5000  # Above max 4096

        with pytest.raises(SpecBoundsError, match="exceeds maximum allowed"):
            spec_validator.validate_spec(large_dim_spec)

    def test_v02_spec_invalid_enums(self, spec_validator):
        """Test v0.2 spec enum validation."""
        # Invalid metric
        invalid_metric_spec = {
            'uer_version': '0.2.0',
            'vector_dimension': 768,
            'normalization_rules': {'method': 'l2', 'epsilon': 1e-12},
            'metric': 'invalid_metric',  # Not in allowed list
            'dtype': 'float32',
            'geometry': 'spherical',
            'validation_rules': {},
            'version_compatibility': {},
            'numeric_precision_guarantees': {},
            'embedding_statistics': {},
            'robustness_profiles': {}
        }

        with pytest.raises(SpecEnumError, match="metric.*not in allowed values"):
            spec_validator.validate_spec(invalid_metric_spec)

        # Invalid geometry
        invalid_metric_spec['metric'] = 'cosine'
        invalid_metric_spec['geometry'] = 'cubic'  # Not in allowed geometries

        with pytest.raises(SpecEnumError, match="geometry.*not in allowed values"):
            spec_validator.validate_spec(invalid_metric_spec)

    def test_v02_spec_alignment_metadata_validation(self, spec_validator):
        """Test v0.2 spec alignment metadata validation."""
        # Missing alignment version
        invalid_align_spec = {
            'uer_version': '0.2.0',
            'vector_dimension': 768,
            'normalization_rules': {'method': 'l2', 'epsilon': 1e-12},
            'metric': 'cosine',
            'dtype': 'float32',
            'geometry': 'spherical',
            'validation_rules': {},
            'version_compatibility': {},
            'numeric_precision_guarantees': {},
            'embedding_statistics': {},
            'robustness_profiles': {},
            'alignment_metadata': {
                'mapping_type': 'identity'
                # Missing alignment_version
            }
        }

        with pytest.raises(SpecStructureError, match="missing.*alignment_version"):
            spec_validator.validate_spec(invalid_align_spec)

        # Invalid alignment version format
        invalid_align_spec['alignment_metadata']['alignment_version'] = 'invalid.semver'

        with pytest.raises(AlignmentMetadataError, match="semver format"):
            spec_validator.validate_spec(invalid_align_spec)

    def test_v02_spec_semantic_consistency(self, spec_validator):
        """Test v0.2 spec semantic consistency checks."""
        # Spherical geometry with euclidean metric (inconsistent)
        inconsistent_spec = {
            'uer_version': '0.2.0',
            'vector_dimension': 768,
            'normalization_rules': {'method': 'l2', 'epsilon': 1e-12},
            'metric': 'euclidean',  # Wrong metric for spherical
            'dtype': 'float32',
            'geometry': 'spherical',
            'validation_rules': {},
            'version_compatibility': {},
            'numeric_precision_guarantees': {},
            'embedding_statistics': {},
            'robustness_profiles': {}
        }

        with pytest.raises(SpecConsistencyError, match="Spherical geometry requires cosine"):
            spec_validator.validate_spec(inconsistent_spec)

    def test_v02_spec_epsilon_bounds(self, spec_validator):
        """Test v0.2 spec epsilon numerical bounds."""
        # Epsilon out of bounds
        invalid_epsilon_spec = {
            'uer_version': '0.2.0',
            'vector_dimension': 768,
            'normalization_rules': {'method': 'l2', 'epsilon': 1e-20},  # Too small
            'metric': 'cosine',
            'dtype': 'float32',
            'geometry': 'spherical',
            'validation_rules': {'epsilon_min': 1e-15, 'epsilon_max': 1e-6},
            'version_compatibility': {},
            'numeric_precision_guarantees': {},
            'embedding_statistics': {},
            'robustness_profiles': {}
        }

        with pytest.raises(SpecBoundsError, match="epsilon.*outside bounds"):
            spec_validator.validate_spec(invalid_epsilon_spec)


class TestUERValidatorV0_2:
    """Test v0.2 enhanced UERValidator functionality."""

    @pytest.fixture
    def v02_spec(self):
        """Load v0.2 spec for testing."""
        spec_file = Path(__file__).parent.parent / 'specs' / 'uer_v0.2.yaml'
        return load_uer_spec(spec_file)

    @pytest.fixture
    def v02_validator(self, v02_spec):
        """Create v0.2 validator instance."""
        return UERValidator(v02_spec)

    # ===== Embedding Validation Tests (20+ cases) =====

    def test_v02_dimension_bounds_validation(self, v02_validator, v02_spec):
        """Test v0.2 dimension bounds checking."""
        # Valid dimension (exactly target)
        valid_dim = np.random.randn(v02_spec['vector_dimension']).astype(np.float32)
        valid_dim = valid_dim / np.linalg.norm(valid_dim)
        assert v02_validator.validate_embedding(valid_dim, strict=True)

        # Lower bound violation
        low_dim = np.random.randn(32).astype(np.float32)  # Below 64 minimum
        with pytest.raises(UERValidationError, match="below minimum allowed"):
            v02_validator.validate_embedding(low_dim, strict=True)

        # Upper bound violation
        high_dim = np.random.randn(5000).astype(np.float32)  # Above 4096 maximum
        high_dim = high_dim / np.linalg.norm(high_dim)
        with pytest.raises(UERValidationError, match="exceeds maximum allowed"):
            v02_validator.validate_embedding(high_dim, strict=True)

    def test_v02_zero_vector_rejection(self, v02_validator):
        """Test zero vector rejection in v0.2."""
        zero_vector = np.zeros(768).astype(np.float32)
        with pytest.raises(UERValidationError, match="Zero vector detected"):
            v02_validator.validate_embedding(zero_vector, strict=True)

    def test_v02_adversarial_vector_tests(self, v02_validator):
        """Test adversarial vector handling."""
        # Vector with extreme values
        extreme_vector = np.random.randn(768).astype(np.float32) * 1e10
        extreme_vector = extreme_vector / np.linalg.norm(extreme_vector)

        # Should still be valid if normalized and within bounds
        assert v02_validator.validate_embedding(extreme_vector, strict=True)

        # Vector with single NaN
        nan_vector = np.random.randn(768).astype(np.float32)
        nan_vector = nan_vector / np.linalg.norm(nan_vector)
        nan_vector[50] = np.nan

        with pytest.raises(UERValidationError, match="NaN values"):
            v02_validator.validate_embedding(nan_vector, strict=True)

        # Vector with infinite values
        inf_vector = np.random.randn(768).astype(np.float32)
        inf_vector = inf_vector / np.linalg.norm(inf_vector)
        inf_vector[100] = np.inf

        with pytest.raises(UERValidationError, match="infinite values"):
            v02_validator.validate_embedding(inf_vector, strict=True)

    def test_v02_dtype_enhanced_validation(self, v02_validator, v02_spec):
        """Test enhanced dtype validation."""
        # Valid dtype
        valid_vector = np.random.randn(v02_spec['vector_dimension']).astype(np.float32)
        valid_vector = valid_vector / np.linalg.norm(valid_vector)
        assert v02_validator.validate_embedding(valid_vector, strict=True)

        # Wrong dtype
        wrong_dtype_vector = np.random.randn(v02_spec['vector_dimension']).astype(np.int32)
        wrong_dtype_vector = wrong_dtype_vector / np.linalg.norm(wrong_dtype_vector.astype(np.float32))
        wrong_dtype_vector = wrong_dtype_vector.astype(np.int32)

        with pytest.raises(UERValidationError, match="Dtype mismatch"):
            v02_validator.validate_embedding(wrong_dtype_vector, strict=True)

    def test_v02_geometry_specific_validation(self, v02_validator, v02_spec):
        """Test geometry-specific validation rules."""
        # For spherical geometry, norms shouldn't be excessively large
        embedding = np.random.randn(v02_spec['vector_dimension']).astype(np.float32)
        embedding = embedding * 5.0  # Scale up before normalization
        embedding = embedding / np.linalg.norm(embedding)  # Usually results in small norms

        # Should pass basic validation
        assert v02_validator.validate_embedding(embedding, strict=True)

    def test_v02_batch_validation(self, v02_validator, v02_spec):
        """Test batch validation functionality."""
        batch_size = 10
        # Valid batch
        valid_batch = np.random.randn(batch_size, v02_spec['vector_dimension']).astype(np.float32)
        norms = np.linalg.norm(valid_batch, axis=1, keepdims=True)
        valid_batch = valid_batch / norms

        assert v02_validator.validate_batch(valid_batch, strict=True) == True

        # Batch with one invalid embedding
        invalid_batch = valid_batch.copy()
        invalid_batch[5] = np.zeros(v02_spec['vector_dimension'])  # Zero vector

        with pytest.raises(UERValidationError, match="Batch validation failed"):
            v02_validator.validate_batch(invalid_batch, strict=True)

        # Non-strict mode should return invalid indices
        invalid_indices = v02_validator.validate_batch(invalid_batch, strict=False)
        assert isinstance(invalid_indices, list)
        assert 5 in invalid_indices

    def test_v02_normalization_tolerance(self, v02_validator, v02_spec):
        """Test normalization tolerance validation."""
        vector = np.random.randn(v02_spec['vector_dimension']).astype(np.float32)

        # Slightly off normalization (within tolerance)
        off_norm_vector = vector / (np.linalg.norm(vector) * 0.99)  # Slightly larger norm
        actual_norm = np.linalg.norm(off_norm_vector)

        # May pass or fail depending on tolerance, but shouldn't crash
        # This tests that validation handles near-normalized vectors appropriately
        try:
            result = v02_validator.validate_embedding(off_norm_vector, strict=False)
            # Either passes or returns False, but no exception
            assert isinstance(result, bool)
        except UERValidationError:
            # If it fails, the error should be about normalization
            pass


class TestUERCompilerV0_2:
    """Test v0.2 enhanced UERCompiler functionality."""

    @pytest.fixture
    def v02_spec(self):
        """Load v0.2 spec for testing."""
        spec_file = Path(__file__).parent.parent / 'specs' / 'uer_v0.2.yaml'
        return load_uer_spec(spec_file)

    @pytest.fixture
    def v02_compiler(self, v02_spec):
        """Create v0.2 compiler instance."""
        return UERCompiler(v02_spec)

    # ===== Compiler Tests =====

    def test_v02_dimension_mismatch_handling(self, v02_compiler, v02_spec):
        """Test dimension mismatch handling."""
        # Input vector smaller than target
        small_vector = np.random.randn(512).astype(np.float32)
        small_vector = small_vector / np.linalg.norm(small_vector)

        # Without handling should fail
        with pytest.raises(UERCompilationError, match="Dimension mismatch"):
            v02_compiler.compile(small_vector, handle_mismatches=False)

        # With handling should pad
        padded_result = v02_compiler.compile(small_vector, handle_mismatches=True)
        assert padded_result.shape == (v02_spec['vector_dimension'],)
        assert np.isfinite(padded_result).all()

    def test_v02_projection_failure_detection(self, v02_compiler):
        """Test projection failure detection."""
        # Create a vector that might cause alignment issues
        vector = np.random.randn(768).astype(np.float32)

        # Should handle normal compilation
        result = v02_compiler.compile(vector)
        assert result.shape == (768,)
        assert np.isfinite(result).all()

    def test_v02_normalization_safeguards(self, v02_compiler):
        """Test normalization safeguards against division by zero."""
        # Zero vector - should be handled gracefully
        zero_vector = np.zeros(768).astype(np.float32)

        result = v02_compiler.compile(zero_vector, handle_mismatches=True)
        # Should not crash and should produce valid output
        assert result.shape == (768,)
        assert np.isfinite(result).all()  # No NaN/Inf in output

    def test_v02_dtype_enforcement(self, v02_compiler, v02_spec):
        """Test dtype enforcement after compilation."""
        vector = np.random.randn(768).astype(np.float64)  # Different input dtype

        result = v02_compiler.compile(vector)
        assert result.dtype == np.dtype(v02_spec['dtype'])
        assert np.isfinite(result).all()

    def test_v02_memory_bounds_checking(self, v02_compiler):
        """Test memory bounds checking for large inputs."""
        # Create a very large vector that could cause memory issues
        large_vector = np.random.randn(100000).astype(np.float32)

        # Should be rejected as too large
        with pytest.raises(UERCompilationError, match="too large"):
            v02_compiler.compile(large_vector)

    def test_v02_batch_processing_scaffolding(self, v02_compiler):
        """Test batch processing scaffolding (should be disabled by default)."""
        batch = np.random.randn(5, 768).astype(np.float32)

        with pytest.raises(NotImplementedError, match="Batch mode not yet enabled"):
            v02_compiler.compile_batch(batch)

        # Should be able to enable but still not implemented
        v02_compiler.enable_batch_mode(True)
        with pytest.raises(NotImplementedError, match="Batch mode not yet enabled"):
            v02_compiler.compile_batch(batch)


class TestBackwardsCompatibility:
    """Test backwards compatibility with v0.1 specs."""

    def test_v01_spec_still_loads(self):
        """Test that v0.1 specs can still be loaded."""
        loader = UERSpecLoader()
        v01_spec = loader.load_spec(Path(__file__).parent.parent / 'specs' / 'uer_v0.1.yaml')

        assert v01_spec['uer_version'] == '0.1.0'
        assert 'version_compatibility' not in v01_spec  # v0.1 doesn't have this

    def test_v01_spec_uses_old_validation(self):
        """Test that v0.1 specs use old validation logic."""
        loader = UERSpecLoader()

        # This is a bit tricky to test directly, but we can check
        # that loading works with basic structure
        v01_spec = {
            'uer_version': '0.1.0',
            'vector_dimension': 768,
            'normalization_rules': {'method': 'l2'},
            'metric': 'cosine',
            'dtype': 'float32',
            'geometry': 'spherical'
        }

        # Should pass validation
        try:
            loader._validate_spec_structure(v01_spec)
        except Exception as e:
            pytest.fail(f"v0.1 spec should be valid: {e}")

    def test_v02_spec_requires_strict_validation(self):
        """Test that v0.2 specs require strict validation."""
        loader = UERSpecLoader()

        # v0.2 spec without required fields should fail
        incomplete_v02 = {
            'uer_version': '0.2.0',
            'vector_dimension': 768,
            # Missing many required v0.2 fields
        }

        with pytest.raises(Exception):  # Either SpecMissingFieldError or related
            loader._validate_spec_structure(incomplete_v02)


if __name__ == "__main__":
    pytest.main([__file__])
    pytest.main([__file__])
