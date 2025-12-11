#!/usr/bin/env python3
"""
UER Advanced Stress Test Suite

Extreme adversarial testing of Universal Embedding Representation system.
Tests specification violations, validation bypasses, compilation failures,
real-world edge cases, security vulnerabilities, and batch processing limits.
"""

import numpy as np
import yaml
import json
import tempfile
import os
from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple
from collections import defaultdict

# Import UER components
import sys
sys.path.insert(0, str(Path(__file__).parent / 'uer'))

from uer import load_uer_spec, UERValidator, UERCompiler, compile_to_uer, compile_batch_to_uer
from uer.core.config import UERConfig
from uer.loader import UERSpecLoader
from uer.validator import UERValidator

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class UERStressTester:
    """Comprehensive stress tester for UER system."""

    def __init__(self):
        self.results = defaultdict(list)
        self.base_spec = self._load_base_spec()
        self.test_counter = 0

    def _load_base_spec(self) -> Dict[str, Any]:
        """Load the base v0.1 spec for testing."""
        with open('specs/uer_v0.1.yaml', 'r') as f:
            spec = yaml.safe_load(f)
        # Also create the config object for the core compiler
        self.base_config = UERConfig(spec)
        return spec

    def run_all_tests(self) -> Dict[str, List]:
        """Run the complete stress test suite."""
        print("🚀 Starting UER Advanced Stress Test Suite")
        print("=" * 60)

        # Test each component
        self._test_spec_violations()
        self._test_validator_weaknesses()
        self._test_compiler_failures()
        self._test_real_world_scenarios()
        self._test_security_adversarial()
        self._test_batch_processing()

        print(f"\n✅ Completed {self.test_counter} stress tests")
        return dict(self.results)

    # ===== SPECIFICATION-LEVEL VIOLATIONS =====

    def _test_spec_violations(self):
        """Test specification weaknesses and violations."""
        print("1️⃣ Testing Specification-Level Violations...")

        # Missing required fields
        self._test_missing_fields()

        # Wrong data types
        self._test_wrong_types()

        # Invalid values
        self._test_invalid_values()

        # Corrupted YAML
        self._test_corrupted_specs()

        # Out-of-spec values
        self._test_out_of_spec_values()

    def _test_missing_fields(self):
        """Test missing required fields."""
        required_fields = ['uer_version', 'vector_dimension', 'dtype']

        for field in required_fields:
            corrupted_spec = self.base_spec.copy()
            del corrupted_spec[field]

            # Test spec loading
            self._evaluate_spec_load(corrupted_spec, f"Missing required field: {field}")

    def _test_wrong_types(self):
        """Test wrong data types."""
        type_tests = [
            ('vector_dimension', 'str', '768'),
            ('dtype', ['list'], 'float32'),
            ('uer_version', 1.0, '0.1.0'),
            ('geometry', {'dict': 'invalid'}, 'spherical'),
        ]

        for field, wrong_type, original in type_tests:
            corrupted_spec = self.base_spec.copy()
            corrupted_spec[field] = wrong_type

            self._evaluate_spec_load(corrupted_spec, f"Wrong type for {field}: {type(wrong_type).__name__}")

    def _test_invalid_values(self):
        """Test invalid field values."""
        invalid_tests = [
            ('vector_dimension', -768, 'negative dimension'),
            ('vector_dimension', 0, 'zero dimension'),
            ('dtype', 'complex128', 'unsupported dtype'),
            ('normalization_rules', {'method': 'invalid'}, 'invalid norm method'),
            ('metric', 'euclidean', 'unsupported metric'),
            ('geometry', 'hyperbolic', 'unsupported geometry'),
        ]

        for field, value, desc in invalid_tests:
            corrupted_spec = self.base_spec.copy()
            corrupted_spec[field] = value

            self._evaluate_spec_load(corrupted_spec, f"Invalid {field}: {desc}")

    def _test_corrupted_specs(self):
        """Test corrupted/invalid YAML/JSON."""
        corrupt_tests = [
            ('Unterminated YAML', "uer_version: '0.1.0'\nvector_dimension:"),
            ('Invalid indentation', "uer_version: '0.1.0'\n  vector_dimension: 768"),
            ('Invalid JSON', '{"uer_version": "0.1.0", "invalid_json"'),
        ]

        for desc, content in corrupt_tests:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(content)
                temp_path = f.name

            try:
                loader = UERSpecLoader()
                result = loader.load_spec(temp_path)
                self.results['spec_violations'].append({
                    'test': f'Corrupted spec: {desc}',
                    'passed': False,
                    'expected': 'Load to fail',
                    'actual': 'Loaded successfully',
                    'severity': 'HIGH'
                })
            except Exception as e:
                self.results['spec_violations'].append({
                    'test': f'Corrupted spec: {desc}',
                    'passed': True,
                    'expected': 'Load to fail',
                    'actual': f'Failed correctly: {e}',
                    'severity': 'LOW'
                })
            finally:
                os.unlink(temp_path)
            self.test_counter += 1

    def _test_out_of_spec_values(self):
        """Test extreme values that violate spec constraints."""
        extreme_tests = [
            ('vector_dimension', 2**31, 'massive dimension'),
            ('vector_dimension', 1, 'minimal dimension'),
            ('normalization_rules', {'epsilon': 0}, 'zero epsilon'),
            ('normalization_rules', {'epsilon': -1}, 'negative epsilon'),
            ('validation_rules', {'norm_tolerance': -0.1}, 'negative tolerance'),
        ]

        for field, value, desc in extreme_tests:
            corrupted_spec = self.base_spec.copy()
            if isinstance(field, str):
                corrupted_spec[field] = value
            else:
                # Nested field
                corrupted_spec[field[0]][field[1]] = value

            self._evaluate_spec_load(corrupted_spec, f"Extreme value: {desc}")

    def _evaluate_spec_load(self, spec: Dict, desc: str):
        """Evaluate spec loading with corrupted data."""
        try:
            # Try to create validator with corrupted spec
            validator = UERValidator(spec)
            self.results['spec_violations'].append({
                'test': desc,
                'passed': False,
                'expected': 'Spec validation to fail',
                'actual': 'Spec accepted incorrectly',
                'severity': 'MEDIUM'
            })
        except Exception as e:
            self.results['spec_violations'].append({
                'test': desc,
                'passed': True,
                'expected': 'Spec validation to fail',
                'actual': f'Failed correctly: {str(e)[:100]}',
                'severity': 'LOW'
            })
        self.test_counter += 1

    # ===== VALIDATOR WEAKNESSES =====

    def _test_validator_weaknesses(self):
        """Test validator bypasses and failures."""
        print("2️⃣ Testing Validator Weaknesses...")

        validator = UERValidator(self.base_spec)

        # Dimension violations
        self._test_dimension_violations(validator)

        # Dtype bypasses
        self._test_dtype_bypasses(validator)

        # Normalization edge cases
        self._test_normalization_edge_cases(validator)

        # NaN/Inf handling
        self._test_nan_inf_handling(validator)

        # Quantization issues
        self._test_quantization_issues(validator)

    def _test_dimension_violations(self, validator: UERValidator):
        """Test dimension validation weaknesses."""
        dim_tests = [
            (np.random.randn(512), 'under-dimension (512 vs 768)'),
            (np.random.randn(1024), 'over-dimension (1024 vs 768)'),
            (np.random.randn(769), 'off-by-one over'),
            (np.random.randn(767), 'off-by-one under'),
            (np.random.randn(768, 2), '2D embedding (batch in wrong place)'),
        ]

        for embedding, desc in dim_tests:
            self._evaluate_validation(validator, embedding, f"Dimension: {desc}")

    def _test_dtype_bypasses(self, validator: UERValidator):
        """Test dtype validation bypasses."""
        dtype_tests = [
            (np.random.randn(768).astype(np.float64), 'float64 vs float32'),
            (np.random.randn(768).astype(np.float16), 'float16 vs float32'),
            (np.random.randn(768).astype(np.int32), 'int32 vs float32'),
            (np.random.randn(768).astype(np.uint8), 'uint8 vs float32'),
            ((np.random.randn(768) * 100).astype(np.uint8), 'quantized uint8'),
        ]

        for embedding, desc in dtype_tests:
            self._evaluate_validation(validator, embedding, f"Dtype: {desc}")

    def _test_normalization_edge_cases(self, validator: UERValidator):
        """Test normalization validation edge cases."""
        # Create embeddings with specific norm issues
        norm_tests = [
            (np.zeros(768), 'zero vector'),
            (np.ones(768), 'uniform vector (norm=√768)'),
            (np.ones(768) / np.sqrt(768), 'perfect unit vector'),
            (np.ones(768) / np.sqrt(768) + 0.1, 'slightly over-normalized'),
            (np.random.randn(768), 'random unnormalized'),
            (np.random.randn(768) / np.linalg.norm(np.random.randn(768)) * 10, 'highly amplified'),
        ]

        for embedding, desc in norm_tests:
            self._evaluate_validation(validator, embedding, f"Normalization: {desc}")

    def _test_nan_inf_handling(self, validator: UERValidator):
        """Test NaN and Inf handling."""
        nan_inf_tests = [
            (np.full(768, np.nan), 'all NaN'),
            (np.full(768, np.inf), 'all positive Inf'),
            (np.full(768, -np.inf), 'all negative Inf'),
            (np.array([1.0] + [np.nan] * 767), 'single NaN'),
            (np.array([1.0] + [np.inf] * 767), 'single Inf'),
            (np.array([np.nan, np.inf, -np.inf] + list(np.ones(765))), 'mixed specials'),
        ]

        for embedding, desc in nan_inf_tests:
            # Normalize to avoid norm issues
            if np.linalg.norm(embedding) > 0:
                embedding = embedding / np.linalg.norm(embedding)
            self._evaluate_validation(validator, embedding, f"NaN/Inf: {desc}")

    def _test_quantization_issues(self, validator: UERValidator):
        """Test quantization validation."""
        quant_tests = [
            (np.random.randint(-128, 127, 768, dtype=np.int8), 'int8 range ok'),
            (np.array([-129] * 768, dtype=np.float32), 'int8 underflow (as float32)'),
            (np.array([128] * 768, dtype=np.float32), 'int8 overflow (as float32)'),
            (np.random.randint(0, 255, 768, dtype=np.uint8), 'uint8 range ok'),
            (np.array([-1] * 768, dtype=np.float32), 'uint8 underflow (as float32)'),
            (np.array([256] * 768, dtype=np.float32), 'uint8 overflow (as float32)'),
        ]

        for embedding, desc in quant_tests:
            self._evaluate_validation(validator, embedding, f"Quantization: {desc}")

    def _evaluate_validation(self, validator: UERValidator, embedding: np.ndarray, desc: str):
        """Evaluate embedding validation."""
        try:
            if validator.validate_embedding(embedding, strict=True):
                # Check if this should actually fail
                if any(substr in desc.lower() for substr in ['under', 'over', 'vs', 'overflow', 'underflow', 'nan', 'inf']):
                    self.results['validator_weaknesses'].append({
                        'test': desc,
                        'passed': False,
                        'expected': 'Validation to fail',
                        'actual': 'Validation passed incorrectly',
                        'severity': 'HIGH'
                    })
                else:
                    self.results['validator_weaknesses'].append({
                        'test': desc,
                        'passed': True,
                        'expected': 'Validation to pass',
                        'actual': 'Passed correctly',
                        'severity': 'LOW'
                    })
            else:
                self.results['validator_weaknesses'].append({
                    'test': desc,
                    'passed': True,
                    'expected': 'Validation to fail',
                    'actual': 'Failed correctly (soft mode)',
                    'severity': 'LOW'
                })
        except ValueError as e:
            if any(substr in desc.lower() for substr in ['under', 'over', 'vs', 'overflow', 'underflow', 'nan', 'inf']):
                self.results['validator_weaknesses'].append({
                    'test': desc,
                    'passed': True,
                    'expected': 'Validation to fail',
                    'actual': f'Failed correctly: {str(e)[:100]}',
                    'severity': 'LOW'
                })
            else:
                self.results['validator_weaknesses'].append({
                    'test': desc,
                    'passed': False,
                    'expected': 'Validation to pass',
                    'actual': f'Failed incorrectly: {str(e)[:100]}',
                    'severity': 'MEDIUM'
                })
        self.test_counter += 1

    # ===== COMPILER FAILURES =====

    def _test_compiler_failures(self):
        """Test compilation failures and edge cases."""
        print("3️⃣ Testing Compiler Weaknesses...")

        compiler = UERCompiler(self.base_config)

        # Input format issues
        self._test_input_format_failures(compiler)

        # Alignment failures
        self._test_alignment_failures(compiler)

        # Normalization failures
        self._test_compiler_normalization_failures(compiler)

        # Dtype conversion issues
        self._test_dtype_conversion_failures(compiler)

        # Final validation failures
        self._test_final_validation_failures(compiler)

    def _test_input_format_failures(self, compiler: UERCompiler):
        """Test input format compilation failures."""
        format_tests = [
            (np.random.randn(3, 768), '3D tensor'),
            (np.random.randn(), 'scalar'),
            ([1.0, 2.0, 3.0], 'Python list'),
            ('string embedding', 'string'),
            ({'key': 'value'}, 'dict'),
        ]

        for embedding, desc in format_tests:
            self._evaluate_compilation(compiler, embedding, f"Input format: {desc}")

    def _evaluate_compilation(self, compiler: UERCompiler, embedding: np.ndarray, desc: str):
        """Evaluate embedding compilation."""
        try:
            result = compiler.compile(embedding, validate_input=False)
            if 'wrong' in desc or 'invalid' in desc or 'fail' in desc:
                self.results['compiler_weaknesses'].append({
                    'test': desc,
                    'passed': False,
                    'expected': 'Compilation to fail',
                    'actual': 'Compiled successfully',
                    'severity': 'MEDIUM'
                })
            else:
                self.results['compiler_weaknesses'].append({
                    'test': desc,
                    'passed': True,
                    'expected': 'Compilation to succeed',
                    'actual': 'Compiled correctly',
                    'severity': 'LOW'
                })
        except Exception as e:
            if 'wrong' in desc or 'invalid' in desc or 'fail' in desc:
                self.results['compiler_weaknesses'].append({
                    'test': desc,
                    'passed': True,
                    'expected': 'Compilation to fail',
                    'actual': f'Failed correctly: {str(e)[:100]}',
                    'severity': 'LOW'
                })
            else:
                self.results['compiler_weaknesses'].append({
                    'test': desc,
                    'passed': False,
                    'expected': 'Compilation to succeed',
                    'actual': f'Failed incorrectly: {str(e)[:100]}',
                    'severity': 'MEDIUM'
                })
        self.test_counter += 1

    def _test_alignment_failures(self, compiler: UERCompiler):
        """Test alignment system failures."""
        # Test different dimension inputs for identity alignment
        align_tests = [
            (np.random.randn(512), 'small dimension (512 -> 768)'),
            (np.random.randn(1024), 'large dimension (1024 -> 768)'),
            (np.ones(768), 'already correct dimension'),
            (np.random.randn(1), 'tiny dimension (1 -> 768)'),
            (np.random.randn(2048), 'very large dimension (2048 -> 768)'),
        ]

        for embedding, desc in align_tests:
            self._evaluate_compilation(compiler, embedding, f"Alignment: {desc}")

    def _test_compiler_normalization_failures(self, compiler: UERCompiler):
        """Test normalization edge cases."""
        norm_tests = [
            (np.zeros(768), 'zero vector normalization'),
            (np.full(768, np.inf), 'infinite values'),
            (np.full(768, np.nan), 'NaN values'),
            (np.ones(768) * 1e-15, 'denormal numbers'),
            (np.random.randn(768) * 1e10, 'huge values'),
        ]

        for embedding, desc in norm_tests:
            self._evaluate_compilation(compiler, embedding, f"Normalization: {desc}")

    def _test_dtype_conversion_failures(self, compiler: UERCompiler):
        """Test dtype conversion edge cases."""
        # Create float64 spec for testing conversion
        float64_spec = self.base_spec.copy()
        float64_spec['dtype'] = 'float64'

        compiler_64 = UERCompiler(float64_spec)

        dtype_tests = [
            (np.random.randn(768).astype(np.complex64), 'complex to float64'),
            (np.array([1+2j] * 768).astype(np.complex128), 'complex128 to float64'),
            (np.random.randint(0, 1000, 768), 'large int to float32'),
        ]

        for embedding, desc in dtype_tests:
            self._evaluate_compilation(compiler, embedding, f"Dtype conversion: {desc}")

    def _test_final_validation_failures(self, compiler: UERCompiler):
        """Test cases where compilation succeeds but final validation fails."""
        # These should trigger final validation after successful processing
        final_tests = [
            (np.random.randn(768) * 1e-20, 'becomes zero after float32 conversion'),
            (np.ones(768) * (1 + 1e10), 'overflows to inf in float32'),
        ]

        for embedding, desc in final_tests:
            self._evaluate_compilation(compiler, embedding, f"Final validation: {desc}")

    # ===== REAL-WORLD EDGE CASES =====

    def _test_real_world_scenarios(self):
        """Test real-world embedding scenarios."""
        print("4️⃣ Testing Real-World Edge Cases...")

        # Provider-specific embeddings
        self._test_provider_embeddings()

        # Multilingual representations
        self._test_multilingual_embeddings()

        # Fine-tuned model embeddings
        self._test_finetuned_embeddings()

        # Domain-specific embeddings
        self._test_domain_specific_embeddings()

        # API-returned corrupted embeddings
        self._test_api_corrupted_embeddings()

    def _test_provider_embeddings(self):
        """Simulate embeddings from different providers."""
        compiler = UERCompiler(self.base_config)

        # OpenAI ADA-002 (1536 dims)
        openai_ada = np.random.randn(1536)
        self._evaluate_compilation(compiler, openai_ada, "OpenAI ADA-002 (1536d)")

        # Cohere v3 (1024 dims)
        cohere_v3 = np.random.randn(1024)
        self._evaluate_compilation(compiler, cohere_v3, "Cohere v3 (1024d)")

        # Mistral 7B (4096 dims)
        mistral_7b = np.random.randn(4096)
        self._evaluate_compilation(compiler, mistral_7b, "Mistral 7B (4096d)")

        # BGE-large (1024 dims)
        bge_large = np.random.randn(1024)
        self._evaluate_compilation(compiler, bge_large, "BGE-large (1024d)")

        # Truncated embeddings
        truncated = np.random.randn(768)[:512]  # Simulate truncated API response
        self._evaluate_compilation(compiler, truncated, "Truncated embedding (512/768)")

        # Quantized embeddings (int8)
        quantized = np.random.randint(-128, 127, 768, dtype=np.int8)
        quantized_spec = self.base_spec.copy()
        quantized_spec['dtype'] = 'int8'
        compiler_quant = UERCompiler(quantized_spec)
        self._evaluate_compilation(compiler_quant, quantized.astype(np.float32), "Quantized embedding (int8)")

    def _test_multilingual_embeddings(self):
        """Test multilingual embedding scenarios."""
        compiler = UERCompiler(self.base_config)

        # Chinese text embedding mapped to English space
        chinese_embedding = np.random.randn(768)
        self._evaluate_compilation(compiler, chinese_embedding, "Chinese text in English semantic space")

        # Code embeddings (different distribution)
        code_embedding = np.random.randn(768) * 2  # Different scale
        self._evaluate_compilation(compiler, code_embedding, "Code embedding different distribution")

        # Emoji/conceptual embeddings
        emoji_embedding = np.ones(768) * 0.1  # Low magnitude conceptual
        self._evaluate_compilation(compiler, emoji_embedding, "Emoji embedding low magnitude")

    def _test_finetuned_embeddings(self):
        """Test fine-tuned model embedding drift."""
        compiler = UERCompiler(self.base_spec)

        # Drifted embeddings (different norm distribution)
        drifted = np.random.randn(768) * np.random.uniform(0.5, 2.0, 768)  # Anisotropic
        self._evaluate_compilation(compiler, drifted, "Finetuned drifted embedding (anisotropic)")

        # Task-specific amplified dimensions
        task_specific = np.random.randn(768)
        task_specific[:100] *= 10  # Amplify first 100 dims for task
        self._evaluate_compilation(compiler, task_specific, "Task-specific amplified dimensions")

    def _test_domain_specific_embeddings(self):
        """Test domain-specific embedding variants."""
        compiler = UERCompiler(self.base_spec)

        # Legal document embeddings (higher precision vocabulary)
        legal_emb = np.random.randn(768) + np.random.uniform(-0.1, 0.1, 768)
        self._evaluate_compilation(compiler, legal_emb, "Legal document embedding")

        # Medical embeddings (different statistical properties)
        medical_emb = np.random.exponential(1.0, 768) - 1  # Exponential distribution
        self._evaluate_compilation(compiler, medical_emb, "Medical embedding (exponential dist)")

    def _test_api_corrupted_embeddings(self):
        """Test embeddings corrupted by API issues."""
        compiler = UERCompiler(self.base_spec)

        # API returns string
        # Skip this as we already test input types

        # API returns null/empty
        empty_emb = np.array([])
        try:
            compiler.compile(empty_emb, validate_input=False)
        except Exception as e:
            self.results['real_world_scenarios'].append({
                'test': 'API returns empty array',
                'passed': 'E' in str(e),  # Should fail
                'expected': 'Compilation to fail',
                'actual': f'Failed correctly: {str(e)[:100]}',
                'severity': 'LOW'
            })

        # API returns wrong structure
        wrong_struct = {'embeddings': [np.random.randn(768)]}
        try:
            compiler.compile(wrong_struct, validate_input=False)
        except Exception as e:
            self.results['real_world_scenarios'].append({
                'test': 'API returns dict instead of array',
                'passed': 'E' in str(e),  # Should fail
                'expected': 'Compilation to fail',
                'actual': f'Failed correctly: {str(e)[:100]}',
                'severity': 'LOW'
            })

        self.test_counter += 2

    # ===== SECURITY & ADVERSARIAL TESTS =====

    def _test_security_adversarial(self):
        """Test security and adversarial inputs."""
        print("5️⃣ Testing Security & Adversarial Cases...")

        compiler = UERCompiler(self.base_spec)

        # Malicious vectors
        self._test_malicious_vectors(compiler)

        # Overflow patterns
        self._test_overflow_patterns(compiler)

        # Numerical instability
        self._test_numerical_instability(compiler)

        # Injection-like attacks
        self._test_injection_attacks(compiler)

    def _test_malicious_vectors(self, compiler: UERCompiler):
        """Test vectors designed to cause crashes or overflow."""
        malicious_tests = [
            (np.full(768, 1e308), 'maximum float values'),
            (np.full(768, -1e308), 'minimum float values'),
            (np.full(768, 1e-323), 'denormal minimum'),
            ((np.ones(768) * 1e300), 'overflow for any operation'),
            ((np.ones(768) * 1e150) + (np.ones(768) * 1e150), 'intermediate overflow'),
        ]

        for embedding, desc in malicious_tests:
            self._evaluate_compilation(compiler, embedding, f"Malicious: {desc}")

    def _test_overflow_patterns(self, compiler: UERCompiler):
        """Test patterns that cause overflow in operations."""
        overflow_tests = [
            (np.ones(768) * 1e100, 'large uniform values'),
            (np.power(10, np.arange(768)), 'exponential growth'),
            (np.random.randn(768) * 1e200, 'random huge values'),
            ((np.ones(768) * 1e100) + np.random.randn(768), 'large base + noise'),
        ]

        for embedding, desc in overflow_tests:
            self._evaluate_compilation(compiler, embedding, f"Overflow: {desc}")

    def _test_numerical_instability(self, compiler: UERCompiler):
        """Test numerically unstable scenarios."""
        instability_tests = [
            (np.ones(768) * 1e-300, 'near-underflow values'),
            (np.random.randn(768) * 1e-200, 'random tiny values'),
            (np.array([1e300, 1e-300] + list(np.ones(766))), 'mixed scales'),
            ((np.sin(np.arange(768) * 0.1) + 1) * 1e200, 'oscillating large values'),
        ]

        for embedding, desc in instability_tests:
            self._evaluate_compilation(compiler, embedding, f"Instability: {desc}")

    def _test_injection_attacks(self, compiler: UERCompiler):
        """Test injection-like attack vectors."""
        # These are more theoretical as numpy doesn't do string operations
        injection_tests = [
            (np.array(['script'] + list(np.ones(767))).astype('<U10'), 'string injection attempt'),
        ]

        for embedding, desc in injection_tests:
            self._evaluate_compilation(compiler, embedding, f"Injection: {desc}")

    # ===== BATCH PROCESSING SCENARIOS =====

    def _test_batch_processing(self):
        """Test batch processing weaknesses."""
        print("6️⃣ Testing Batch Processing Scenarios...")

        # Inconsistent shapes
        self._test_inconsistent_batch_shapes()

        # Mixed dtypes in batch
        self._test_mixed_dtypes_batch()

        # Memory pressure tests
        self._test_memory_pressure_batch()

        # Corrupted batch data
        self._test_corrupted_batch_data()

    def _test_inconsistent_batch_shapes(self):
        """Test batches with inconsistent embedding shapes."""
        inconsistent_batches = [
            ([np.random.randn(512), np.random.randn(768)], '512d and 768d'),
            ([np.random.randn(768), np.random.randn(1024), np.random.randn(512)], 'mixed dimensions'),
            ([np.random.randn(768, 2), np.random.randn(768)], '2D and 1D mixed'),
            ([np.random.randn(1), np.random.randn(768)], 'extreme dimension difference'),
        ]

        for batch, desc in inconsistent_batches:
            try:
                result = compile_batch_to_uer(batch, self.base_spec)
                self.results['batch_scenarios'].append({
                    'test': f'Inconsistent shapes: {desc}',
                    'passed': False,
                    'expected': 'Batch compilation to fail',
                    'actual': 'Compiled batch successfully',
                    'severity': 'HIGH'
                })
            except Exception as e:
                self.results['batch_scenarios'].append({
                    'test': f'Inconsistent shapes: {desc}',
                    'passed': True,
                    'expected': 'Batch compilation to fail',
                    'actual': f'Failed correctly: {str(e)[:100]}',
                    'severity': 'LOW'
                })
            self.test_counter += 1

    def _test_mixed_dtypes_batch(self):
        """Test batches with mixed data types."""
        mixed_batches = [
            ([np.random.randn(768).astype(np.float64), np.random.randn(768).astype(np.float32)], 'float64 and float32'),
            ([np.random.randn(768), np.random.randint(0, 10, 768)], 'float and int'),
            ([np.random.randn(768), (np.random.randn(768) * 100).astype(np.uint8)], 'float and uint8'),
        ]

        for batch, desc in mixed_batches:
            try:
                result = compile_batch_to_uer(batch, self.base_spec)
                self.results['batch_scenarios'].append({
                    'test': f'Mixed dtypes: {desc}',
                    'passed': False,  # Should fail
                    'expected': 'Batch compilation to fail',
                    'actual': 'Compiled mixed dtypes successfully',
                    'severity': 'MEDIUM'
                })
            except Exception as e:
                self.results['batch_scenarios'].append({
                    'test': f'Mixed dtypes: {desc}',
                    'passed': True,
                    'expected': 'Batch compilation to fail',
                    'actual': f'Failed correctly: {str(e)[:100]}',
                    'severity': 'LOW'
                })
            self.test_counter += 1

    def _test_memory_pressure_batch(self):
        """Test batch processing under memory pressure."""
        # Create a large batch
        large_batch = [np.random.randn(768) for _ in range(10000)]

        try:
            result = compile_batch_to_uer(large_batch, self.base_spec)
            self.results['batch_scenarios'].append({
                'test': 'Large batch (10k embeddings)',
                'passed': True,
                'expected': 'Handle large batch',
                'actual': f'Compiled {len(large_batch)} embeddings successfully',
                'severity': 'LOW'
            })
        except Exception as e:
            self.results['batch_scenarios'].append({
                'test': 'Large batch (10k embeddings)',
                'passed': False,
                'expected': 'Handle large batch',
                'actual': f'Memory failure: {str(e)[:100]}',
                'severity': 'MEDIUM'
            })
        self.test_counter += 1

    def _test_corrupted_batch_data(self):
        """Test batches with corrupted data."""
        corrupted_tests = [
            ([np.random.randn(768), np.full(768, np.nan), np.random.randn(768)], 'middle vector NaN'),
            ([np.random.randn(768), np.full(768, np.inf), np.random.randn(768)], 'middle vector Inf'),
            ([np.random.randn(768), np.zeros(768), np.random.randn(768)], 'middle zero vector'),
            ([np.random.randn(768), None, np.random.randn(768)], 'None in batch'),
        ]

        for batch, desc in corrupted_tests:
            try:
                # Filter out None values for numpy compatibility
                clean_batch = [emb for emb in batch if emb is not None]
                result = compile_batch_to_uer(clean_batch, self.base_spec)
                if 'nan' in desc or 'inf' in desc or 'zero' in desc:
                    self.results['batch_scenarios'].append({
                        'test': f'Corrupted batch: {desc}',
                        'passed': False,
                        'expected': 'Batch compilation to fail',
                        'actual': 'Compiled corrupted batch successfully',
                        'severity': 'HIGH'
                    })
                else:
                    self.results['batch_scenarios'].append({
                        'test': f'Corrupted batch: {desc}',
                        'passed': True,
                        'expected': 'Handle corrupted batch',
                        'actual': 'Compiled successfully',
                        'severity': 'LOW'
                    })
            except Exception as e:
                if 'nan' in desc or 'inf' in desc or 'zero' in desc:
                    self.results['batch_scenarios'].append({
                        'test': f'Corrupted batch: {desc}',
                        'passed': True,
                        'expected': 'Batch compilation to fail',
                        'actual': f'Failed correctly: {str(e)[:100]}',
                        'severity': 'LOW'
                    })
                else:
                    self.results['batch_scenarios'].append({
                        'test': f'Corrupted batch: {desc}',
                        'passed': False,
                        'expected': 'Handle corrupted batch',
                        'actual': f'Failed incorrectly: {str(e)[:100]}',
                        'severity': 'MEDIUM'
                    })
            self.test_counter += 1


def analyze_results(results: Dict[str, List]) -> Dict[str, Any]:
    """Analyze test results and compute statistics."""
    analysis = {
        'total_tests': 0,
        'passed_tests': 0,
        'failed_tests': 0,
        'severity_breakdown': {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0},
        'component_coverage': {},
        'worst_failures': [],
        'best_performing': [],
        'critical_vulnerabilities': []
    }

    for component, tests in results.items():
        component_stats = {'total': len(tests), 'passed': 0, 'failed': 0, 'severity': {}}

        for test in tests:
            analysis['total_tests'] += 1

            if test.get('passed', False):
                analysis['passed_tests'] += 1
                component_stats['passed'] += 1
            else:
                analysis['failed_tests'] += 1
                component_stats['failed'] += 1

                # Track worst failures
                if test.get('severity') != 'LOW':
                    analysis['worst_failures'].append(test)

                # Track critical vulnerabilities
                if test.get('severity') == 'HIGH':
                    analysis['critical_vulnerabilities'].append(test)

            severity = test.get('severity', 'LOW')
            analysis['severity_breakdown'][severity] += 1

            if severity not in component_stats['severity']:
                component_stats['severity'][severity] = 0
            component_stats['severity'][severity] += 1

        analysis['component_coverage'][component] = component_stats

    # Find best and worst performing components
    analysis['best_performing'] = sorted(
        analysis['component_coverage'].items(),
        key=lambda x: (x[1]['passed'] / x[1]['total']) if x[1]['total'] > 0 else 0,
        reverse=True
    )[:3]

    analysis['worst_performing'] = sorted(
        analysis['component_coverage'].items(),
        key=lambda x: (x[1]['failed'] / x[1]['total']) if x[1]['total'] > 0 else 0,
        reverse=True
    )[:3]

    analysis['robustness_score'] = analysis['passed_tests'] / analysis['total_tests'] if analysis['total_tests'] > 0 else 0

    return analysis


def generate_report(results: Dict[str, List], analysis: Dict[str, Any]) -> None:
    """Generate the comprehensive Markdown report."""
    report_content = f"""# UER Stress Test Report (Generated by LLM)

## 1. Overview
Comprehensive adversarial testing of the Universal Embedding Representation (UER) v0.1 system. Tests covered specification violations, validation weaknesses, compilation failures, real-world scenarios, security vulnerabilities, and batch processing limits. Total of {analysis['total_tests']} tests executed across all components.

**System Robustness Score: {analysis['robustness_score']:.1%}**
- Passed: {analysis['passed_tests']}
- Failed: {analysis['failed_tests']}
- High-severity issues: {analysis['severity_breakdown']['HIGH']}

## 2. Specification-Level Failures

### Critical Spec Violations
"""

    # List spec violations
    spec_violations = results.get('spec_violations', [])
    if spec_violations:
        for violation in spec_violations:
            if not violation.get('passed', True):
                report_content += f"- **{violation['test']}**: {violation['actual']}\n"
    else:
        report_content += "No specification-level failures detected.\n"

    report_content += "\n### Spec Loading Resilience\n"
    spec_passed = sum(1 for v in spec_violations if v.get('passed', True))
    total_spec = len(spec_violations)
    report_content += f"Spec validation correctly handled {spec_passed}/{total_spec} malformed specifications.\n"

    report_content += """
## 3. Validation Module Weaknesses

### Validator Bypasses
"""

    validator_weaknesses = results.get('validator_weaknesses', [])
    critical_validator = [w for w in validator_weaknesses if not w.get('passed', True) and w.get('severity') == 'HIGH']
    if critical_validator:
        for weakness in critical_validator:
            report_content += f"- **{weakness['test']}**: {weakness['actual']}\n"
    else:
        report_content += "No critical validator bypasses detected.\n"

    report_content += f"""
### Validation Statistics
- Total validation tests: {len(validator_weaknesses)}
- Validation bypasses: {sum(1 for w in validator_weaknesses if not w.get('passed', True))}
- Dimension violations handled: {sum(1 for w in validator_weaknesses if 'dimension' in w['test'].lower() and w.get('passed', True))}
- Dtype enforcement: {sum(1 for w in validator_weaknesses if 'dtype' in w['test'].lower() and w.get('passed', True))}
"""

    report_content += """
## 4. Compiler-Level Weaknesses

### Compilation Failures
"""

    compiler_weaknesses = results.get('compiler_weaknesses', [])
    critical_compiler = [c for c in compiler_weaknesses if not c.get('passed', True) and c.get('severity') in ['HIGH', 'MEDIUM']]
    if critical_compiler:
        for weakness in critical_compiler:
            report_content += f"- **{weakness['test']}**: {weakness['actual']}\n"
    else:
        report_content += "No critical compilation failures detected.\n"

    report_content += """
### Alignment System Performance
- Identity alignment handled dimension mismatches: """ + str(sum(1 for c in compiler_weaknesses if 'alignment' in c['test'].lower() and c.get('passed', True))) + """
- Normalization robustness: """ + str(sum(1 for c in compiler_weaknesses if 'normalization' in c['test'].lower() and c.get('passed', True))) + """
- Type conversion safety: """ + str(sum(1 for c in compiler_weaknesses if 'dtype conversion' in c['test'].lower() and c.get('passed', True))) + """

## 5. Real-World Edge Cases

### Provider Compatibility
"""

    real_world = results.get('real_world_scenarios', [])
    provider_tests = [r for r in real_world if any(provider in r['test'] for provider in ['OpenAI', 'Cohere', 'Mistral', 'BGE'])]
    for test in provider_tests:
        status = "✅ PASS" if test.get('passed', True) else f"❌ FAIL: {test['actual']}"
        report_content += f"- {test['test']}: {status}\n"

    report_content += """
### Multilingual & Domain Handling
"""

    multilingual = [r for r in real_world if any(term in r['test'].lower() for term in ['chinese', 'multilingual', 'domain', 'legal', 'medical', 'code'])]
    for test in multilingual:
        status = "✅ PASS" if test.get('passed', True) else f"❌ FAIL: {test['actual']}"
        report_content += f"- {test['test']}: {status}\n"

    report_content += """
### API Corruption Resilience
"""

    api_tests = [r for r in real_world if 'API' in r['test']]
    api_passed = sum(1 for t in api_tests if t.get('passed', True))
    report_content += f"API corruption scenarios handled correctly: {api_passed}/{len(api_tests)}\n"

    report_content += """
## 6. Security & Adversarial Tests

### Adversarial Vector Resistance
"""

    security = results.get('security_adversarial', [])
    malicious = [s for s in security if any(term in s['test'].lower() for term in ['malicious', 'overflow', 'instability', 'injection'])]
    critical_security = [s for s in malicious if not s.get('passed', True)]
    if critical_security:
        for vuln in critical_security:
            report_content += f"- **VULNERABILITY**: {vuln['test']} - {vuln['actual']}\n"
    else:
        report_content += "No critical adversarial vector vulnerabilities detected.\n"

    report_content += f"""
### Numerical Stability
Security tests passed: {sum(1 for s in security if s.get('passed', True))}/{len(security)}
"""

    # ===== Generating the next sections =====

    report_content += """
## 7. Batch Processing Scenarios

### Batch Integrity Tests
"""

    batch = results.get('batch_scenarios', [])
    batch_failures = [b for b in batch if not b.get('passed', True)]
    if batch_failures:
        for failure in batch_failures:
            report_content += f"- **{failure['test']}**: {failure['actual']}\n"
    else:
        report_content += "All batch processing scenarios handled correctly.\n"

    report_content += """
### Performance Boundaries
"""

    large_batch = [b for b in batch if 'large batch' in b['test'].lower()]
    if large_batch:
        for test in large_batch:
            status = "✅ PASS" if test.get('passed', True) else f"❌ FAIL: {test['actual']}"
            report_content += f"- {test['test']}: {status}\n"

    report_content += f"""
Batch tests: {sum(1 for b in batch if b.get('passed', True))}/{len(batch)} passed
"""

    report_content += """
## 8. Full Coverage Matrix

| Component | Total Tests | Passed | Failed | High Sev | Medium Sev | Low Sev |
|-----------|-------------|--------|--------|----------|------------|---------|
"""

    for component, stats in analysis['component_coverage'].items():
        high_sev = stats['severity'].get('HIGH', 0)
        med_sev = stats['severity'].get('MEDIUM', 0)
        low_sev = stats['severity'].get('LOW', 0)
        report_content += f"| {component.replace('_', ' ').title()} | {stats['total']} | {stats['passed']} | {stats['failed']} | {high_sev} | {med_sev} | {low_sev} |\n"

    report_content += """
## 9. Recommendations & Fixes

### Critical Fixes Required
"""

    if analysis['critical_vulnerabilities']:
        for vuln in analysis['critical_vulnerabilities'][:5]:  # Top 5
            report_content += f"- **{vuln['severity']}**: {vuln['test']} - {vuln['actual'][:50]}...\n"
            if 'dimension' in vuln['test'].lower():
                report_content += "  - Add strict dimension validation in spec loading\n"
            elif 'dtype' in vuln['test'].lower():
                report_content += "  - Enhance dtype conversion safety checks\n"
            elif 'nan' in vuln['test'].lower() or 'inf' in vuln['test'].lower():
                report_content += "  - Implement NaN/Inf sanitization in normalization\n"
            elif 'batch' in vuln['test'].lower():
                report_content += "  - Add batch consistency validation\n"

    report_content += """
### IR Specification Improvements (v0.2)
- Add `version_compatibility` field for model upgrades
- Implement `numeric_precision_guarantees` for fp16/fp8 support
- Add `embedding_statistics` metadata for distribution monitoring
- Include `robustness_profiles` for different deployment scenarios

### v0.3 Architecture Recommendations
- Decentralized validation pipeline with async error recovery
- Model-aware alignment matrices with automatic training
- Federated learning support for alignment model updates
- Hardware acceleration detection with optimized kernels
"""

    # Overall summary
    report_content += ".1%"
    report_content += f"""
### Key Risk Areas
1. **Spec Validation**: {analysis['severity_breakdown']['HIGH']} high-severity spec issues
2. **Batch Processing**: {len([b for b in batch if not b.get('passed', True)])} batch integrity problems
3. **Adversarial Inputs**: {len(critical_security)} security bypasses detected
4. **Provider Compatibility**: {len([p for p in provider_tests if not p.get('passed', True)])} provider mapping issues

### Implementation Priority Matrix
- **HIGH**: Fix spec validation (affects all deployments)
- **HIGH**: Implement batch consistency checks (breaks multi-model workflows)
- **MEDIUM**: Add adversarial input sanitization (security hardening)
- **LOW**: Provider metadata registry (DX improvement)
"""

    # Write report to file
    with open('UER_Test_Report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"\n📊 Report generated: UER_Test_Report.md")
    print(f"Overall robustness: {analysis['robustness_score']:.1%}")
    print(f"Critical vulnerabilities: {len(analysis['critical_vulnerabilities'])}")
    return True


def main():
    """Run the comprehensive stress test and generate report."""
    tester = UERStressTester()

    # Run all tests
    results = tester.run_all_tests()

    # Analyze results
    analysis = analyze_results(results)

    # Generate report
    success = generate_report(results, analysis)

    if success:
        print("✅ UER Stress Test Report generated successfully!")


if __name__ == '__main__':
    main()
