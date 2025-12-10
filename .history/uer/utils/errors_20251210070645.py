"""
UER Error Handling

Human-readable errors with recovery suggestions, auto-fixes, and strict/soft modes.
Replaces generic ValueError with actionable UER-specific exceptions.
"""

from typing import Any, Dict, List, Optional, Callable, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


class UERError(Exception):
    """
    Base exception for UER-related errors.

    Provides actionable error messages with suggestions for resolution.
    """

    def __init__(self, message: str, suggestions: Optional[List[str]] = None,
                 recovery_options: Optional[Dict[str, Any]] = None):
        """
        Initialize UER error with suggestions and recovery options.

        Args:
            message: Human-readable error description
            suggestions: List of actionable suggestions to fix the error
            recovery_options: Dictionary of recovery methods/callbacks
        """
        self.message = message
        self.suggestions = suggestions or []
        self.recovery_options = recovery_options or {}

        # Build full error message
        full_message = f"UER Error: {message}"

        if suggestions:
            full_message += "\n\nSuggestions:"
            for i, suggestion in enumerate(suggestions, 1):
                full_message += f"\n  {i}. {suggestion}"

        super().__init__(full_message)


class UERValidationError(UERError):
    """Errors during embedding validation with auto-recovery options."""

    def __init__(self, validation_issue: str, embedding_info: Dict[str, Any] = None,
                 auto_repair_options: Optional[List[str]] = None):
        """
        Initialize validation error with repair options.

        Args:
            validation_issue: Description of the validation failure
            embedding_info: Information about the problematic embedding
            auto_repair_options: Available auto-repair strategies
        """
        self.validation_issue = validation_issue
        self.embedding_info = embedding_info or {}
        self.auto_repair_options = auto_repair_options or []

        suggestions = []
        recovery_options = {}

        # Generate specific suggestions based on issue type
        if "dimension" in validation_issue.lower():
            suggestions.append("Check your model output dimension matches the UER spec")
            suggestions.append("Use dimension_aware embedding models")
            if "mismatch" in validation_issue.lower():
                suggestions.append("UER can auto-pad embeddings - enable 'auto_repair' in validation_rules")
                recovery_options['auto_pad'] = True

        elif "normalization" in validation_issue.lower():
            suggestions.append("Ensure your embedding model outputs L2-normalized vectors")
            suggestions.append("Apply normalization after encoding if needed")
            recovery_options['auto_normalize'] = True

        elif "dtype" in validation_issue.lower():
            suggestions.append("Configure your UER spec for float32 if using quantized models")
            suggestions.append("Use numpy array conversion: arr.astype(np.float32)")
            recovery_options['auto_cast'] = True

        elif "nan" in validation_issue.lower() or "inf" in validation_issue.lower():
            suggestions.append("Check for numerical instability in your embedding pipeline")
            suggestions.append("Add gradient clipping or numerical stabilization")
            recovery_options['replace_inf_nan'] = True

        message = f"Validation failed: {validation_issue}"
        if self.embedding_info:
            message += f" (shape: {self.embedding_info.get('shape', 'unknown')}, " \
                      f"dtype: {self.embedding_info.get('dtype', 'unknown')})"

        super().__init__(message, suggestions, recovery_options)

    def can_auto_repair(self) -> bool:
        """Check if this error can be automatically repaired."""
        return bool(self.auto_repair_options)

    def try_auto_repair(self, embedding: np.ndarray) -> Optional[np.ndarray]:
        """
        Attempt automatic repair of the problematic embedding.

        Returns:
            Repaired embedding or None if repair failed
        """
        try:
            repaired = embedding.copy()

            for repair_option in self.auto_repair_options:
                if repair_option == 'auto_pad' and 'expected_dim' in self.embedding_info:
                    target_dim = self.embedding_info['expected_dim']
                    current_dim = embedding.shape[-1]
                    if current_dim < target_dim:
                        # Smart padding: Use replication for semantic preservation
                        pad_size = target_dim - current_dim
                        # Use boundary values for smoother padding
                        padding = np.tile(embedding[:, -1:], (1, pad_size))
                        repaired = np.concatenate([embedding, padding], axis=-1)

                elif repair_option == 'auto_normalize':
                    norms = np.linalg.norm(repaired, axis=-1, keepdims=True)
                    eps = 1e-12  # Avoid division by very small numbers
                    repaired = repaired / np.maximum(norms, eps)

                elif repair_option == 'auto_cast':
                    repaired = repaired.astype(np.float32)

                elif repair_option == 'replace_inf_nan':
                    repaired = np.nan_to_num(repaired, nan=0.0, posinf=1.0, neginf=-1.0)

            return repaired

        except Exception as e:
            logger.warning(f"Auto-repair failed: {e}")
            return None


class UERConfigurationError(UERError):
    """Errors in UER specification or configuration."""

    def __init__(self, config_issue: str, fix_examples: Optional[Dict[str, str]] = None):
        """
        Initialize configuration error with fix examples.

        Args:
            config_issue: Description of the configuration problem
            fix_examples: Dictionary of before/after fix examples
        """
        self.config_issue = config_issue
        self.fix_examples = fix_examples or {}

        suggestions = []
        if "missing" in config_issue.lower():
            suggestions.append("Add the missing fields to your UER specification YAML")
            suggestions.append("See specs/uer_v0.1.yaml for complete examples")

        elif "invalid" in config_issue.lower():
            suggestions.append("Check data types in your specification")
            suggestions.append("Validate YAML syntax with an online validator")

        for field, example in self.fix_examples.items():
            suggestions.append(f"For {field}: {example}")

        super().__init__(f"Configuration error: {config_issue}", suggestions)


class UERAlignmentError(UERError):
    """Errors during embedding alignment operations."""

    def __init__(self, alignment_issue: str, semantic_risks: Optional[List[str]] = None):
        """
        Initialize alignment error with semantic risk warnings.

        Args:
            alignment_issue: Description of alignment problem
            semantic_risks: Potential semantic degradation risks
        """
        self.alignment_issue = alignment_issue
        self.semantic_risks = semantic_risks or []

        suggestions = []
        suggestions.append("Consider training a proper alignment matrix instead of identity mapping")
        suggestions.append("Use provider-specific alignment models for better semantic preservation")

        if semantic_risks:
            suggestions.append("Semantic risks detected:")
            suggestions.extend(f"  ⚠ {risk}" for risk in semantic_risks)

        message = f"Alignment error: {alignment_issue}"
        super().__init__(message, suggestions)


class UERCompilationError(UERError):
    """
    Compilation errors with potential auto-recovery.

    Can attempt to fix issues like normalization, padding, etc.
    """

    def __init__(self,
                 compilation_issue: str,
                 input_info: Dict[str, Any] = None,
                 suggested_fixes: Optional[List[str]] = None,
                 auto_fixed_result: Optional[np.ndarray] = None,
                 fixes_applied: Optional[List[str]] = None):
        """
        Initialize compilation error with recovery options.

        Args:
            compilation_issue: Description of compilation failure
            input_info: Information about input embedding
            suggested_fixes: Human suggestions for fixing
            auto_fixed_result: Automatically repaired embedding if available
            fixes_applied: List of auto-fixes that were attempted
        """
        self.compilation_issue = compilation_issue
        self.input_info = input_info or {}
        self.auto_fixed_result = auto_fixed_result
        self.fixes_applied = fixes_applied or []

        suggestions = suggested_fixes or []

        # Generate context-aware suggestions
        if "dimension" in compilation_issue.lower():
            suggestions.append("Your embedding dimension doesn't match UER spec - consider padding or truncation")
        elif "normalization" in compilation_issue.lower():
            suggestions.append("Embeddings must be L2-normalized for cosine similarity")
            suggestions.append("Add normalization before feeding to UER compiler")
        elif "alignment" in compilation_issue.lower():
            suggestions.append("Configure provider-specific alignment in your UER spec")

        if self.auto_fixed_result is not None:
            suggestions.insert(0, f"✓ Auto-repaired {len(self.fixes_applied)} issues: {', '.join(self.fixes_applied)}")

        message = f"Compilation failed: {compilation_issue}"
        super().__init__(message, suggestions, {})


def with_error_handling(strict: bool = True,
                       auto_repair: bool = False,
                       recovery_callback: Optional[Callable] = None):
    """
    Decorator for UER operations with configurable error handling.

    Args:
        strict: If True, raise exceptions. If False, warn and return None
        auto_repair: If True, attempt automatic fixes
        recovery_callback: Optional callback for failed operations
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if auto_repair and hasattr(e, 'try_auto_repair'):
                    result = e.try_auto_repair(args[0] if args else None)
                    if result is not None:
                        logger.info(f"Auto-repaired error in {func.__name__}: {e.validation_issue}")
                        return result

                if recovery_callback:
                    recovery_callback(e, *args, **kwargs)

                if strict:
                    raise e
                else:
                    logger.warning(f"Soft failure in {func.__name__}: {e}")
                    return None
        return wrapper
    return decorator


class ErrorMode:
    """
    Error handling modes for UER operations.

    Provides different strategies for dealing with validation/compilation failures.
    """

    STRICT = "strict"          # Always raise exceptions
    SOFT = "soft"              # Warn and continue with best effort
    AUTO_REPAIR = "auto_repair" # Attempt automatic fixes
    PEDANTIC = "pedantic"      # Detailed logging plus strict mode

    @staticmethod
    def configure(error_mode: str) -> Dict[str, Any]:
        """
        Get configuration for a specific error mode.

        Args:
            error_mode: One of the error mode constants

        Returns:
            Configuration dictionary for error handling
        """
        if error_mode == ErrorMode.STRICT:
            return {'strict': True, 'auto_repair': False, 'detailed_logging': False}

        elif error_mode == ErrorMode.SOFT:
            return {'strict': False, 'auto_repair': False, 'detailed_logging': False}

        elif error_mode == ErrorMode.AUTO_REPAIR:
            return {'strict': False, 'auto_repair': True, 'detailed_logging': True}

        elif error_mode == ErrorMode.PEDANTIC:
            return {'strict': True, 'auto_repair': False, 'detailed_logging': True}

        else:
            raise ValueError(f"Unknown error mode: {error_mode}. Use ErrorMode constants.")
