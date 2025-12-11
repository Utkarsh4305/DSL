"""
UER Specification Loader Module

Loads and parses UER specification files (YAML/JSON).
Provides validation for specification format with v0.2 strict validation.
"""

import yaml
import json
from typing import Dict, Any, Union, Optional
from pathlib import Path
import logging

from .spec_validator import SpecValidator, SpecMissingFieldError, SpecTypeError, SpecBoundsError, SpecEnumError
from .spec_validator import AlignmentMetadataError, SpecStructureError, SpecConsistencyError
from .utils.errors import UERConfigurationError

logger = logging.getLogger(__name__)


class UERSpecLoader:
    """Loads UER specification files."""

    def __init__(self):
        self._spec_cache = {}

    def load_spec(self, spec_file: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a UER specification from YAML or JSON file.

        Args:
            spec_file: Path to specification file

        Returns:
            Dict containing the UER specification

        Raises:
            FileNotFoundError: If spec file doesn't exist
            ValueError: If spec format is invalid
        """
        spec_path = Path(spec_file)

        if not spec_path.exists():
            raise FileNotFoundError(f"UER spec file not found: {spec_path}")

        # Cache check
        if str(spec_path) in self._spec_cache:
            return self._spec_cache[str(spec_path)]

        # Load based on extension
        if spec_path.suffix.lower() == '.yaml':
            with open(spec_path, 'r', encoding='utf-8') as f:
                spec = yaml.safe_load(f)
        elif spec_path.suffix.lower() == '.json':
            with open(spec_path, 'r', encoding='utf-8') as f:
                spec = json.load(f)
        else:
            raise ValueError(f"Unsupported spec file format: {spec_path.suffix}")

        # Validate spec structure
        self._validate_spec_structure(spec)

        # Cache and return
        self._spec_cache[str(spec_path)] = spec
        return spec

    def _validate_spec_structure(self, spec: Dict[str, Any]) -> None:
        """Validate that the spec contains required fields."""
        required_fields = [
            'uer_version',
            'vector_dimension',
            'normalization_rules',
            'metric',
            'dtype',
            'geometry'
        ]

        missing_fields = [field for field in required_fields if field not in spec]

        if missing_fields:
            raise ValueError(f"UER spec missing required fields: {missing_fields}")

        # Version check
        if not isinstance(spec['uer_version'], str):
            raise ValueError("uer_version must be a string")

        # Dimension check
        if not isinstance(spec['vector_dimension'], int) or spec['vector_dimension'] <= 0:
            raise ValueError("vector_dimension must be a positive integer")


def load_uer_spec(spec_file: Union[str, Path]) -> Dict[str, Any]:
    """Convenience function to load UER spec."""
    loader = UERSpecLoader()
    return loader.load_spec(spec_file)
