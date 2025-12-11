"""
UER Configuration Management - Enterprise Grade

Advanced configuration handling with schema evolution, versioning,
and enterprise features for production UER deployments.

Includes UERConfig class for backwards compatibility with v0.1 core compiler.
"""

import json
import os
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import hashlib
from datetime import datetime
import numpy as np

from ..utils.errors import UERConfigurationError

logger = logging.getLogger(__name__)


class UERSchemaManager:
    """
    Manages UER specification schema evolution and compatibility.

    Ensures smooth transitions between versions and prevents
    configuration drift in enterprise environments.
    """

    # Supported schema versions with evolution paths
    SCHEMA_VERSIONS = {
        "0.1.0": {
            "compatible_versions": ["0.1.0"],
            "deprecated_fields": [],
            "required_fields": ["uer_version", "vector_dimension", "normalization_rules", "metric", "dtype", "geometry"],
            "schema_hash": "a1b2c3d4"
        },
        "0.2.0": {
            "compatible_versions": ["0.1.0", "0.2.0"],
            "deprecated_fields": [],
            "required_fields": [
                "uer_version", "vector_dimension", "normalization_rules", "metric", "dtype", "geometry",
                "version_compatibility", "numeric_precision_guarantees",
                "embedding_statistics", "robustness_profiles"
            ],
            "schema_hash": "e5f6g7h8"
        }
    }

    def __init__(self):
        """Initialize schema manager."""
        self._schema_cache = {}

    def validate_schema_compatibility(self, spec: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate if specification is compatible with current schema.

        Returns:
            Tuple of (is_compatible, compatibility_warnings)
        """
        version = spec.get('uer_version', 'unknown')
        if version not in self.SCHEMA_VERSIONS:
            return False, [f"Unsupported schema version: {version}"]

        schema_info = self.SCHEMA_VERSIONS[version]
        warnings = []

        # Check required fields
        missing_fields = []
        for field in schema_info["required_fields"]:
            if field not in spec:
                missing_fields.append(field)

        if missing_fields:
            return False, [f"Missing required fields: {missing_fields}"]

        # Version compatibility warnings
        ver_compat = spec.get('version_compatibility', {})
        if not ver_compat.get('backwards_compatible', True):
            warnings.append("Specification marked as not backwards compatible")

        if ver_compat.get('migration_required', False):
            warnings.append("Migration required for this specification")

        return True, warnings

    def evolve_specification(self, old_spec: Dict[str, Any], target_version: str) -> Dict[str, Any]:
        """
        Evolve specification to target version with compatibility transformations.

        Args:
            old_spec: Original specification
            target_version: Desired version string

        Returns:
            Evolved specification compatible with target version
        """
        current_version = old_spec.get('uer_version', '0.1.0')

        if current_version == target_version:
            return old_spec.copy()

        if target_version not in self.SCHEMA_VERSIONS:
            raise UERConfigurationError(f"Unknown target version: {target_version}")

        evolved_spec = old_spec.copy()
        evolved_spec['uer_version'] = target_version

        # Apply evolution transformations
        if current_version == '0.1.0' and target_version == '0.2.0':
            evolved_spec.update(self._evolve_v01_to_v02(old_spec))

        return evolved_spec

    def _evolve_v01_to_v02(self, v01_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transformations from v0.1 to v0.2."""
        return {
            'version_compatibility': {
                'previous_versions': ['0.1.0'],
                'backwards_compatible': True,
                'deprecated_features': [],
                'migration_required': False
            },
            'numeric_precision_guarantees': {
                'fp_precision': v01_spec.get('dtype', 'float32'),
                'quantization_impact': 'minimal',
                'numerical_stability': 'guaranteed',
                'supported_precisions': ['float32', 'float64']
            },
            'embedding_statistics': {
                'expected_distribution': 'normal',
                'mean_norm': 1.0,
                'norm_variance': 'low',
                'dimension_scaling': 'linear'
            },
            'robustness_profiles': {
                'adversarial_resistance': 'high',
                'corruption_tolerance': 'medium',
                'numerical_stability': 'guaranteed',
                'provider_variability': 'handled'
            }
        }


class UERConfigurationManager:
    """
    Enterprise-grade configuration management for UER.

    Provides configuration versioning, auditing, rollback capabilities,
    and environment-aware configuration loading.
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration manager.

        Args:
            config_dir: Directory for configuration files and history
        """
        self.config_dir = config_dir or Path('config')
        self.schema_manager = UERSchemaManager()
        self._config_cache = {}

        # Ensure config directory exists
        self.config_dir.mkdir(exist_ok=True)

    def load_config_with_audit(self, config_path: Path,
                             environment: str = "production") -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load configuration with comprehensive auditing and validation.

        Args:
            config_path: Path to configuration file
            environment: Deployment environment

        Returns:
            Tuple of (configuration, audit_info)
        """
        config_hash = self._calculate_file_hash(config_path)

        # Check cache
        if config_hash in self._config_cache:
            cached_config, cached_audit = self._config_cache[config_hash]
            return cached_config.copy(), cached_audit.copy()

        # Load and validate
        with open(config_path, 'r') as f:
            raw_config = json.load(f) if config_path.suffix == '.json' else \
                        self._load_yaml_config(f)

        # Schema validation
        is_compatible, warnings = self.schema_manager.validate_schema_compatibility(raw_config)

        # Environment-specific overrides
        config = self._apply_environment_overrides(raw_config, environment)

        # Create audit trail
        audit_info = {
            'loaded_at': datetime.now().isoformat(),
            'file_hash': config_hash,
            'environment': environment,
            'schema_compatible': is_compatible,
            'warnings': warnings,
            'version': config.get('uer_version', 'unknown'),
            'config_size': len(str(config)),
            'server_info': self._get_system_info()
        }

        # Cache result
        self._config_cache[config_hash] = (config.copy(), audit_info.copy())

        # Log warnings
        for warning in warnings:
            logger.warning(f"Config audit: {warning}")

        return config, audit_info

    def save_config_with_backup(self, config: Dict[str, Any], config_path: Path,
                              create_backup: bool = True) -> Path:
        """
        Save configuration with automatic backup creation.

        Returns:
            Path to saved configuration
        """
        if create_backup and config_path.exists():
            backup_path = self._create_backup(config_path)
            logger.info(f"Backup created: {backup_path}")

        # Add metadata
        config_with_meta = config.copy()
        config_with_meta['_metadata'] = {
            'saved_at': datetime.now().isoformat(),
            'schema_version': config.get('uer_version', 'unknown'),
            'config_hash': self._calculate_dict_hash(config)
        }

        # Save
        with open(config_path, 'w') as f:
            json.dump(config_with_meta, f, indent=2, sort_keys=True)

        # Update cache
        config_hash = self._calculate_file_hash(config_path)
        audit_info = {
            'saved_at': config_with_meta['_metadata']['saved_at'],
            'file_hash': config_hash,
            'schema_compatible': True,
            'warnings': []
        }
        self._config_cache[config_hash] = (config.copy(), audit_info)

        return config_path

    def rollback_to_backup(self, config_path: Path, backup_suffix: str) -> Path:
        """
        Rollback configuration to a backup version.

        Args:
            config_path: Current configuration path
            backup_suffix: Backup suffix to restore from

        Returns:
            Restored configuration path
        """
        backup_path = config_path.with_suffix(f"{config_path.suffix}.backup.{backup_suffix}")

        if not backup_path.exists():
            raise UERConfigurationError(f"Backup not found: {backup_path}")

        import shutil
        shutil.copy2(backup_path, config_path)

        logger.info(f"Rolled back to backup: {backup_suffix}")
        return config_path

    def validate_configuration_integrity(self, config: Dict[str, Any]) -> List[str]:
        """
        Perform comprehensive integrity validation of configuration.

        Returns:
            List of integrity issues (empty if all good)
        """
        issues = []

        # Schema integrity
        is_compatible, warnings = self.schema_manager.validate_schema_compatibility(config)
        issues.extend(warnings)

        if not is_compatible:
            issues.append("Schema compatibility validation failed")

        # Logical consistency checks
        issues.extend(self._check_logical_consistency(config))

        # Security validation
        issues.extend(self._check_security_constraints(config))

        # Performance validation
        issues.extend(self._check_performance_constraints(config))

        return issues

    def _apply_environment_overrides(self, config: Dict[str, Any], environment: str) -> Dict[str, Any]:
        """Apply environment-specific configuration overrides."""
        overrides = config.get('environment_overrides', {})
        env_config = overrides.get(environment, {})

        # Deep merge environment overrides
        result = config.copy()
        self._deep_merge(result, env_config)

        return result

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """Perform deep merge of override into base dictionary."""
        for key, value in override.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _check_logical_consistency(self, config: Dict[str, Any]) -> List[str]:
        """Check logical consistency of configuration."""
        issues = []

        # Dimension consistency
        val_rules = config.get('validation_rules', {})
        dim = config.get('vector_dimension', 0)
        min_dim = val_rules.get('dimension_min', 0)
        max_dim = val_rules.get('dimension_max', 100000)

        if not (min_dim <= dim <= max_dim):
            issues.append(f"Vector dimension {dim} outside configured bounds [{min_dim}, {max_dim}]")

        # Precision consistency
        precision = config.get('numeric_precision_guarantees', {}).get('fp_precision')
        dtype = config.get('dtype')

        if precision and dtype and precision != dtype:
            issues.append(f"FP precision '{precision}' != configured dtype '{dtype}'")

        return issues

    def _check_security_constraints(self, config: Dict[str, Any]) -> List[str]:
        """Check security-related constraints."""
        issues = []

        # Large dimension warnings (potential DoS vectors)
        dim = config.get('vector_dimension', 0)
        if dim > 10000:
            issues.append(f"Very large dimension {dim} may cause performance/memory issues")

        # Unsafe validation rules
        val_rules = config.get('validation_rules', {})
        if not val_rules.get('zero_vector_reject', True):
            issues.append("Zero vector rejection disabled - potential security risk")

        if not val_rules.get('nan_check', True):
            issues.append("NaN checking disabled - potential numerical stability issues")

        return issues

    def _check_performance_constraints(self, config: Dict[str, Any]) -> List[str]:
        """Check performance-related constraints."""
        issues = []

        # Memory estimation
        dim = config.get('vector_dimension', 0)
        dtype = config.get('dtype', 'float32')
        dtype_size = {'float16': 2, 'float32': 4, 'float64': 8}.get(dtype, 4)

        estimated_memory = dim * dtype_size
        if estimated_memory > 100 * 1024 * 1024:  # 100MB per vector
            issues.append(f"Large vector estimated memory: {estimated_memory} bytes may impact performance")

        return issues

    def _create_backup(self, config_path: Path) -> Path:
        """Create timestamped backup of configuration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = config_path.with_suffix(f"{config_path.suffix}.backup.{timestamp}")
        import shutil
        shutil.copy2(config_path, backup_path)
        return backup_path

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file contents."""
        hash_obj = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()

    def _calculate_dict_hash(self, data: Dict[str, Any]) -> str:
        """Calculate hash of dictionary contents."""
        # Normalize for consistent hashing
        normalized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(normalized.encode()).hexdigest()

    def _load_yaml_config(self, file_obj) -> Dict[str, Any]:
        """Load YAML configuration (placeholder for when PyYAML is available)."""
        import yaml
        return yaml.safe_load(file_obj)

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for audit trail."""
        import platform
        import psutil
        try:
            return {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(),
                'memory_gb': round(psutil.virtual_memory().total / (1024**3), 1)
            }
        except ImportError:
            return {'system_info_unavailable': True}


def create_production_config(spec: Dict[str, Any],
                           environment: str = "production") -> Dict[str, Any]:
    """
    Create production-ready configuration with enterprise enhancements.

    Args:
        spec: Base UER specification
        environment: Deployment environment

    Returns:
        Production-enhanced configuration
    """
    prod_config = spec.copy()

    # Add production defaults
    prod_config.setdefault('environment_overrides', {
        'production': {
            'validation_rules': {
                'zero_vector_reject': True,
                'nan_check': True,
                'inf_check': True,
                'dtype_strict': True
            },
            'robustness_profiles': {
                'numerical_stability': 'guaranteed',
                'adversarial_resistance': 'high'
            }
        },
        'development': {
            'validation_rules': {
                'zero_vector_reject': False,  # Allow in dev for easier testing
                'nan_check': True,
                'inf_check': True,
                'dtype_strict': False
            }
        }
    })

    # Apply environment overrides
    if environment in prod_config.get('environment_overrides', {}):
        env_override = prod_config['environment_overrides'][environment]
        UERConfigurationManager()._deep_merge(prod_config, env_override)

    return prod_config


def migrate_legacy_config(legacy_path: Path, output_path: Optional[Path] = None) -> Path:
    """
    Migrate legacy configuration to current enterprise format.

    Args:
        legacy_path: Path to legacy configuration
        output_path: Output path (defaults to legacy_path with .migrated suffix)

    Returns:
        Path to migrated configuration
    """
    output_path = output_path or legacy_path.with_suffix(f"{legacy_path.suffix}.migrated")

    manager = UERConfigurationManager()
    config, _ = manager.load_config_with_audit(legacy_path, environment="migration")

    # Evolve schema if needed
    schema_manager = UERSchemaManager()
    migrated_config = schema_manager.evolve_specification(config, "0.2.0")

    # Convert to production format
    production_config = create_production_config(migrated_config)

    # Save with backup
    manager.save_config_with_backup(production_config, output_path)

    logger.info(f"Migrated legacy config: {legacy_path} -> {output_path}")
    return output_path


class UERConfig:
    """
    UER Configuration wrapper for backwards compatibility with v0.1 core compiler.

    Provides a simple interface similar to the original v0.1 configuration,
    while supporting the enhanced v0.2 specification internally.
    """

    def __init__(self, spec: Dict[str, Any]):
        """
        Initialize UER configuration.

        Args:
            spec: UER specification dictionary (can be v0.1 or v0.2)
        """
        self.spec = spec
        self.vector_dim = spec.get('vector_dimension', 768)

        # Auto-upgrade v0.1 specs to v0.2 internally if needed
        if spec.get('uer_version', '0.1.0') == '0.1.0':
            schema_manager = UERSchemaManager()
            self.spec = schema_manager.evolve_specification(spec, '0.2.0')
            logger.info("Automatically upgraded v0.1 spec to v0.2 internally")

    def get_target_dtype(self) -> np.dtype:
        """Get the target dtype for embeddings."""
        dtype_str = self.spec.get('dtype', 'float32')
        return np.dtype(dtype_str)

    def get_spec(self) -> Dict[str, Any]:
        """Get the full specification dictionary."""
        return self.spec

    def is_v2_spec(self) -> bool:
        """Check if this is a v0.2 specification."""
        return self.spec.get('uer_version', '0.1.0') == '0.2.0'

    @property
    def vector_dimension(self) -> int:
        """Get vector dimension."""
        return self.spec.get('vector_dimension', 768)

    @property
    def dtype(self) -> str:
        """Get dtype string."""
        return self.spec.get('dtype', 'float32')

    @property
    def metric(self) -> str:
        """Get distance metric."""
        return self.spec.get('metric', 'cosine')

    @property
    def normalization_rules(self) -> Dict[str, Any]:
        """Get normalization rules."""
        return self.spec.get('normalization_rules', {'method': 'l2', 'epsilon': 1e-12})

    def get_validation_rules(self) -> Dict[str, Any]:
        """Get enhanced validation rules (v0.2 features with defaults)."""
        return self.spec.get('validation_rules', {
            'dimension_tolerance': 0.0,
            'norm_tolerance': 1e-6,
            'dtype_strict': True,
            'nan_check': True,
            'inf_check': True,
            'zero_vector_reject': True
        })

    def should_reject_zero_vectors(self) -> bool:
        """Check if zero vector rejection is enabled."""
        return self.get_validation_rules().get('zero_vector_reject', True)

    @property
    def dim_tolerance(self) -> float:
        """Get dimension tolerance."""
        return self.get_validation_rules().get('dimension_tolerance', 0.0)

    @property
    def norm_tolerance(self) -> float:
        """Get normalization tolerance."""
        return self.get_validation_rules().get('norm_tolerance', 1e-6)
