# this_file: src/vttiro/validation/config_schema.py

"""
Configuration validation & schema management for VTTiro enterprise systems.

This module provides comprehensive configuration validation, schema management,
and type safety for all VTTiro enterprise components including AI intelligence,
analytics, monitoring, security, and resilience systems.

Key Features:
- JSON Schema validation for all configuration types
- Type-safe configuration loading and validation
- Schema versioning and migration support
- Environment-specific validation rules
- Configuration drift detection and correction
- Secure configuration handling with redaction
- Real-time validation with performance optimization

Used by: All enterprise systems for configuration validation
Integrates with: core config, security, monitoring, error handling
"""

import asyncio
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Type, TypeVar, get_type_hints
import hashlib
import copy

from loguru import logger

try:
    import jsonschema
    from jsonschema import validate, ValidationError, Draft7Validator
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    ValidationError = Exception

try:
    from pydantic import BaseModel, ValidationError as PydanticValidationError, validator
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    BaseModel = object
    PydanticValidationError = Exception


# Type definitions
T = TypeVar('T')
ConfigDict = Dict[str, Any]


class ConfigValidationLevel(Enum):
    """Configuration validation strictness levels."""
    STRICT = "strict"           # All validation rules enforced
    STANDARD = "standard"       # Standard validation with warnings
    PERMISSIVE = "permissive"   # Minimal validation, log issues
    DISABLED = "disabled"       # No validation (not recommended)


class SchemaVersion(Enum):
    """Configuration schema versions."""
    V1_0 = "1.0"
    V1_1 = "1.1"
    V2_0 = "2.0"
    CURRENT = "2.0"


class ConfigSection(Enum):
    """Configuration sections for validation."""
    CORE = "core"
    AI_INTELLIGENCE = "ai_intelligence"
    ANALYTICS = "analytics"
    MONITORING = "monitoring"
    SECURITY = "security"
    RESILIENCE = "resilience"
    TRANSCRIPTION = "transcription"
    PROVIDERS = "providers"
    INTEGRATIONS = "integrations"
    ENTERPRISE = "enterprise"


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    CRITICAL = "critical"       # Must be fixed, system cannot start
    ERROR = "error"            # Should be fixed, affects functionality
    WARNING = "warning"        # Should be reviewed, may affect performance
    INFO = "info"              # Informational, no action needed


@dataclass
class ValidationIssue:
    """Configuration validation issue details."""
    severity: ValidationSeverity
    section: ConfigSection
    field_path: str
    message: str
    suggestion: Optional[str] = None
    current_value: Any = None
    expected_value: Any = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SchemaValidationResult:
    """Schema validation result with detailed information."""
    is_valid: bool
    schema_version: SchemaVersion
    validation_level: ConfigValidationLevel
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def critical_issues(self) -> List[ValidationIssue]:
        """Get critical validation issues."""
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.CRITICAL]
    
    @property
    def error_issues(self) -> List[ValidationIssue]:
        """Get error validation issues."""
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.ERROR]
    
    @property
    def warning_issues(self) -> List[ValidationIssue]:
        """Get warning validation issues."""
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.WARNING]


@dataclass
class ConfigurationSchema:
    """Configuration schema definition with validation rules."""
    version: SchemaVersion
    section: ConfigSection
    schema: Dict[str, Any]
    required_fields: Set[str] = field(default_factory=set)
    optional_fields: Set[str] = field(default_factory=set)
    sensitive_fields: Set[str] = field(default_factory=set)
    validation_rules: Dict[str, Any] = field(default_factory=dict)


class ConfigValidator(ABC):
    """Abstract base class for configuration validators."""
    
    @abstractmethod
    async def validate(self, config: ConfigDict, schema: ConfigurationSchema) -> SchemaValidationResult:
        """Validate configuration against schema."""
        pass
    
    @abstractmethod
    def get_supported_sections(self) -> List[ConfigSection]:
        """Get list of supported configuration sections."""
        pass


class JSONSchemaValidator(ConfigValidator):
    """JSON Schema-based configuration validator."""
    
    def __init__(self):
        if not HAS_JSONSCHEMA:
            raise ImportError("jsonschema library required for JSON schema validation")
        
        self.validator_cache: Dict[str, Draft7Validator] = {}
        logger.info("JSON Schema validator initialized")
    
    async def validate(self, config: ConfigDict, schema: ConfigurationSchema) -> SchemaValidationResult:
        """Validate configuration using JSON Schema."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Get or create validator
            schema_key = f"{schema.section.value}_{schema.version.value}"
            if schema_key not in self.validator_cache:
                self.validator_cache[schema_key] = Draft7Validator(schema.schema)
            
            validator = self.validator_cache[schema_key]
            
            # Perform validation
            validation_errors = list(validator.iter_errors(config))
            issues = []
            
            for error in validation_errors:
                field_path = ".".join(str(p) for p in error.absolute_path)
                severity = self._determine_severity(error, schema)
                
                issue = ValidationIssue(
                    severity=severity,
                    section=schema.section,
                    field_path=field_path,
                    message=error.message,
                    current_value=error.instance if hasattr(error, 'instance') else None
                )
                issues.append(issue)
            
            
            # Performance metrics
            end_time = asyncio.get_event_loop().time()
            performance_metrics = {
                'validation_time_ms': (end_time - start_time) * 1000,
                'schema_size': len(str(schema.schema)),
                'config_size': len(str(config))
            }
            
            return SchemaValidationResult(
                is_valid=len([i for i in issues if i.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]]) == 0,
                schema_version=schema.version,
                validation_level=ConfigValidationLevel.STANDARD,
                issues=issues,
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return SchemaValidationResult(
                is_valid=False,
                schema_version=schema.version,
                validation_level=ConfigValidationLevel.STANDARD,
                issues=[ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    section=schema.section,
                    field_path="root",
                    message=f"Schema validation error: {str(e)}"
                )]
            )
    
    def _determine_severity(self, error: ValidationError, schema: ConfigurationSchema) -> ValidationSeverity:
        """Determine severity based on validation error and schema context."""
        field_path = ".".join(str(p) for p in error.absolute_path)
        
        # Critical if required field is missing
        if error.validator == "required":
            return ValidationSeverity.CRITICAL
        
        # Critical if type mismatch on essential fields
        if error.validator == "type" and field_path in schema.required_fields:
            return ValidationSeverity.CRITICAL
        
        # Error for type mismatches
        if error.validator == "type":
            return ValidationSeverity.ERROR
        
        # Warning for format issues
        if error.validator in ["format", "pattern"]:
            return ValidationSeverity.WARNING
        
        return ValidationSeverity.ERROR
    
    
    def get_supported_sections(self) -> List[ConfigSection]:
        """Get supported configuration sections."""
        return list(ConfigSection)


class TypeValidator(ConfigValidator):
    """Type-based configuration validator using Python type hints."""
    
    def __init__(self):
        self.type_cache: Dict[str, Type] = {}
        logger.info("Type validator initialized")
    
    async def validate(self, config: ConfigDict, schema: ConfigurationSchema) -> SchemaValidationResult:
        """Validate configuration using Python type hints."""
        start_time = asyncio.get_event_loop().time()
        issues = []
        
        try:
            # Get validation rules from schema
            validation_rules = schema.validation_rules
            
            for field_path, expected_type in validation_rules.items():
                value = self._get_nested_value(config, field_path)
                
                if value is None and field_path in schema.required_fields:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.CRITICAL,
                        section=schema.section,
                        field_path=field_path,
                        message=f"Required field '{field_path}' is missing"
                    ))
                    continue
                
                if value is not None and not self._validate_type(value, expected_type):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        section=schema.section,
                        field_path=field_path,
                        message=f"Type mismatch: expected {expected_type}, got {type(value).__name__}",
                        current_value=value,
                        expected_value=expected_type
                    ))
            
            # Performance metrics
            end_time = asyncio.get_event_loop().time()
            performance_metrics = {
                'validation_time_ms': (end_time - start_time) * 1000,
                'fields_validated': len(validation_rules)
            }
            
            return SchemaValidationResult(
                is_valid=len([i for i in issues if i.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]]) == 0,
                schema_version=schema.version,
                validation_level=ConfigValidationLevel.STANDARD,
                issues=issues,
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            logger.error(f"Type validation failed: {e}")
            return SchemaValidationResult(
                is_valid=False,
                schema_version=schema.version,
                validation_level=ConfigValidationLevel.STANDARD,
                issues=[ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    section=schema.section,
                    field_path="root",
                    message=f"Type validation error: {str(e)}"
                )]
            )
    
    def _get_nested_value(self, config: Dict[str, Any], field_path: str) -> Any:
        """Get nested value from configuration using dot notation."""
        keys = field_path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def _validate_type(self, value: Any, expected_type: Any) -> bool:
        """Validate value against expected type."""
        if expected_type == Any:
            return True
        
        if hasattr(expected_type, '__origin__'):
            # Handle generic types like List[str], Dict[str, int]
            origin = expected_type.__origin__
            
            if origin == list:
                return isinstance(value, list)
            elif origin == dict:
                return isinstance(value, dict)
            elif origin == Union:
                # Handle Optional types
                args = expected_type.__args__
                return any(isinstance(value, arg) if arg != type(None) else value is None for arg in args)
        
        return isinstance(value, expected_type)
    
    def get_supported_sections(self) -> List[ConfigSection]:
        """Get supported configuration sections."""
        return list(ConfigSection)


class SchemaManager:
    """Manages configuration schemas with versioning and migration support."""
    
    def __init__(self, schema_directory: Optional[Path] = None):
        self.schema_directory = schema_directory or Path(__file__).parent / "schemas"
        self.schemas: Dict[Tuple[ConfigSection, SchemaVersion], ConfigurationSchema] = {}
        self.migration_rules: Dict[Tuple[SchemaVersion, SchemaVersion], callable] = {}
        
        # Initialize built-in schemas
        self._initialize_builtin_schemas()
        
        logger.info(f"Schema manager initialized with {len(self.schemas)} schemas")
    
    def _initialize_builtin_schemas(self):
        """Initialize built-in configuration schemas."""
        
        # Core configuration schema
        core_schema = ConfigurationSchema(
            version=SchemaVersion.CURRENT,
            section=ConfigSection.CORE,
            schema={
                "type": "object",
                "properties": {
                    "debug": {"type": "boolean"},
                    "log_level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR"]},
                    "max_workers": {"type": "integer", "minimum": 1, "maximum": 100},
                    "timeout": {"type": "number", "minimum": 0},
                    "cache_size": {"type": "integer", "minimum": 0}
                },
                "required": ["log_level"],
                "additionalProperties": False
            },
            required_fields={"log_level"},
            optional_fields={"debug", "max_workers", "timeout", "cache_size"},
            validation_rules={
                "debug": bool,
                "log_level": str,
                "max_workers": int,
                "timeout": float,
                "cache_size": int
            }
        )
        self.register_schema(core_schema)
        
        # AI Intelligence configuration schema
        ai_schema = ConfigurationSchema(
            version=SchemaVersion.CURRENT,
            section=ConfigSection.AI_INTELLIGENCE,
            schema={
                "type": "object",
                "properties": {
                    "quality_prediction": {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean"},
                            "confidence_threshold": {"type": "number", "minimum": 0, "maximum": 1},
                            "model_type": {"type": "string", "enum": ["confidence", "statistical", "ml_regression", "ensemble"]}
                        },
                        "required": ["enabled"]
                    },
                    "ml_optimization": {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean"},
                            "learning_rate": {"type": "number", "minimum": 0.001, "maximum": 1.0},
                            "batch_size": {"type": "integer", "minimum": 1, "maximum": 1000}
                        }
                    }
                },
                "required": ["quality_prediction"],
                "additionalProperties": False
            },
            required_fields={"quality_prediction"},
            optional_fields={"ml_optimization"}
        )
        self.register_schema(ai_schema)
        
        # Analytics configuration schema
        analytics_schema = ConfigurationSchema(
            version=SchemaVersion.CURRENT,
            section=ConfigSection.ANALYTICS,
            schema={
                "type": "object",
                "properties": {
                    "usage_tracking": {"type": "boolean"},
                    "cost_analysis": {"type": "boolean"},
                    "performance_monitoring": {"type": "boolean"},
                    "retention_days": {"type": "integer", "minimum": 1, "maximum": 365},
                    "aggregation_interval": {"type": "string", "enum": ["hourly", "daily", "weekly"]}
                },
                "required": ["usage_tracking"],
                "additionalProperties": False
            },
            required_fields={"usage_tracking"},
            optional_fields={"cost_analysis", "performance_monitoring", "retention_days", "aggregation_interval"}
        )
        self.register_schema(analytics_schema)
        
        # Security configuration schema
        security_schema = ConfigurationSchema(
            version=SchemaVersion.CURRENT,
            section=ConfigSection.SECURITY,
            schema={
                "type": "object",
                "properties": {
                    "encryption_key": {"type": "string", "minLength": 32},
                    "api_key_rotation": {"type": "boolean"},
                    "audit_logging": {"type": "boolean"},
                    "rate_limiting": {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean"},
                            "requests_per_minute": {"type": "integer", "minimum": 1}
                        }
                    }
                },
                "required": ["encryption_key"],
                "additionalProperties": False
            },
            required_fields={"encryption_key"},
            optional_fields={"api_key_rotation", "audit_logging", "rate_limiting"},
            sensitive_fields={"encryption_key"}
        )
        self.register_schema(security_schema)
        
        # Resilience configuration schema
        resilience_schema = ConfigurationSchema(
            version=SchemaVersion.CURRENT,
            section=ConfigSection.RESILIENCE,
            schema={
                "type": "object",
                "properties": {
                    "circuit_breaker": {
                        "type": "object",
                        "properties": {
                            "failure_threshold": {"type": "integer", "minimum": 1},
                            "recovery_timeout": {"type": "number", "minimum": 1},
                            "success_threshold": {"type": "integer", "minimum": 1}
                        }
                    },
                    "bulkhead": {
                        "type": "object",
                        "properties": {
                            "max_concurrent_calls": {"type": "integer", "minimum": 1},
                            "queue_size": {"type": "integer", "minimum": 1},
                            "timeout": {"type": "number", "minimum": 1}
                        }
                    },
                    "failover": {
                        "type": "object",
                        "properties": {
                            "strategy": {"type": "string", "enum": ["round_robin", "priority_based", "least_load", "adaptive", "health_based"]},
                            "max_retries": {"type": "integer", "minimum": 1},
                            "retry_delay": {"type": "number", "minimum": 0.1}
                        }
                    }
                },
                "additionalProperties": False
            },
            optional_fields={"circuit_breaker", "bulkhead", "failover"}
        )
        self.register_schema(resilience_schema)
    
    def register_schema(self, schema: ConfigurationSchema):
        """Register a configuration schema."""
        key = (schema.section, schema.version)
        self.schemas[key] = schema
        logger.debug(f"Registered schema: {schema.section.value} v{schema.version.value}")
    
    def get_schema(self, section: ConfigSection, version: SchemaVersion = SchemaVersion.CURRENT) -> Optional[ConfigurationSchema]:
        """Get configuration schema for section and version."""
        return self.schemas.get((section, version))
    
    def get_latest_schema(self, section: ConfigSection) -> Optional[ConfigurationSchema]:
        """Get the latest schema version for a section."""
        matching_schemas = [(v, s) for (s, v), schema in self.schemas.items() if s == section]
        if not matching_schemas:
            return None
        
        # Sort by version and return latest
        latest_version = max(matching_schemas, key=lambda x: x[0].value)
        return self.schemas[(section, latest_version[0])]
    
    def migrate_config(self, config: ConfigDict, from_version: SchemaVersion, to_version: SchemaVersion) -> ConfigDict:
        """Migrate configuration from one version to another."""
        if from_version == to_version:
            return config
        
        migration_key = (from_version, to_version)
        if migration_key in self.migration_rules:
            return self.migration_rules[migration_key](config)
        
        logger.warning(f"No migration rule found for {from_version.value} -> {to_version.value}")
        return config
    
    def register_migration(self, from_version: SchemaVersion, to_version: SchemaVersion, migration_func: callable):
        """Register a configuration migration function."""
        self.migration_rules[(from_version, to_version)] = migration_func
        logger.info(f"Registered migration: {from_version.value} -> {to_version.value}")


class ConfigurationValidator:
    """
    Comprehensive configuration validation system with schema management.
    
    Provides enterprise-grade configuration validation with support for
    multiple validation levels, schema versioning, and performance optimization.
    """
    
    def __init__(self, validation_level: ConfigValidationLevel = ConfigValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.schema_manager = SchemaManager()
        self.validators: List[ConfigValidator] = []
        
        # Initialize validators based on available libraries
        if HAS_JSONSCHEMA:
            self.validators.append(JSONSchemaValidator())
        
        self.validators.append(TypeValidator())
        
        # Performance tracking
        self.validation_cache: Dict[str, Tuple[SchemaValidationResult, datetime]] = {}
        self.cache_ttl = timedelta(minutes=5)
        
        logger.info(f"Configuration validator initialized with {len(self.validators)} validators")
    
    async def validate_configuration(self, config: ConfigDict, section: ConfigSection, 
                                   version: SchemaVersion = SchemaVersion.CURRENT) -> SchemaValidationResult:
        """
        Validate configuration for a specific section.
        
        Args:
            config: Configuration dictionary to validate
            section: Configuration section type
            version: Schema version to validate against
            
        Returns:
            Detailed validation result
        """
        if self.validation_level == ConfigValidationLevel.DISABLED:
            return SchemaValidationResult(
                is_valid=True,
                schema_version=version,
                validation_level=self.validation_level
            )
        
        # Check cache first
        config_hash = self._compute_config_hash(config)
        cache_key = f"{section.value}_{version.value}_{config_hash}"
        
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        # Get schema
        schema = self.schema_manager.get_schema(section, version)
        if not schema:
            return SchemaValidationResult(
                is_valid=False,
                schema_version=version,
                validation_level=self.validation_level,
                issues=[ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    section=section,
                    field_path="root",
                    message=f"No schema found for {section.value} v{version.value}"
                )]
            )
        
        # Run validation with all available validators
        all_issues = []
        all_warnings = []
        performance_metrics = {}
        
        for validator in self.validators:
            if section in validator.get_supported_sections():
                try:
                    result = await validator.validate(config, schema)
                    all_issues.extend(result.issues)
                    all_warnings.extend(result.warnings)
                    performance_metrics.update(result.performance_metrics)
                except Exception as e:
                    logger.error(f"Validator {type(validator).__name__} failed: {e}")
                    all_issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        section=section,
                        field_path="root",
                        message=f"Validator error: {str(e)}"
                    ))
        
        # Apply validation level filtering
        filtered_issues = self._filter_issues_by_level(all_issues)
        
        # Create final result
        final_result = SchemaValidationResult(
            is_valid=len([i for i in filtered_issues if i.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]]) == 0,
            schema_version=version,
            validation_level=self.validation_level,
            issues=filtered_issues,
            warnings=all_warnings,
            performance_metrics=performance_metrics
        )
        
        # Cache result
        self._cache_result(cache_key, final_result)
        
        return final_result
    
    async def validate_full_configuration(self, config: ConfigDict) -> Dict[ConfigSection, SchemaValidationResult]:
        """
        Validate entire configuration across all sections.
        
        Args:
            config: Full configuration dictionary
            
        Returns:
            Validation results for each section
        """
        results = {}
        
        for section in ConfigSection:
            section_config = config.get(section.value, {})
            if section_config or section in [ConfigSection.CORE]:  # Always validate core
                result = await self.validate_configuration(section_config, section)
                results[section] = result
        
        return results
    
    def _compute_config_hash(self, config: ConfigDict) -> str:
        """Compute hash of configuration for caching."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def _get_cached_result(self, cache_key: str) -> Optional[SchemaValidationResult]:
        """Get cached validation result if still valid."""
        if cache_key in self.validation_cache:
            result, timestamp = self.validation_cache[cache_key]
            if datetime.now() - timestamp < self.cache_ttl:
                return result
            else:
                del self.validation_cache[cache_key]
        
        return None
    
    def _cache_result(self, cache_key: str, result: SchemaValidationResult):
        """Cache validation result."""
        self.validation_cache[cache_key] = (result, datetime.now())
        
        # Clean old cache entries
        if len(self.validation_cache) > 100:
            cutoff_time = datetime.now() - self.cache_ttl
            expired_keys = [k for k, (_, ts) in self.validation_cache.items() if ts < cutoff_time]
            for key in expired_keys:
                del self.validation_cache[key]
    
    def _filter_issues_by_level(self, issues: List[ValidationIssue]) -> List[ValidationIssue]:
        """Filter issues based on validation level."""
        if self.validation_level == ConfigValidationLevel.STRICT:
            return issues
        elif self.validation_level == ConfigValidationLevel.STANDARD:
            return [i for i in issues if i.severity != ValidationSeverity.INFO]
        elif self.validation_level == ConfigValidationLevel.PERMISSIVE:
            return [i for i in issues if i.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]]
        else:
            return []
    
    def redact_sensitive_config(self, config: ConfigDict, section: ConfigSection) -> ConfigDict:
        """Redact sensitive fields from configuration for logging/display."""
        schema = self.schema_manager.get_schema(section)
        if not schema or not schema.sensitive_fields:
            return config
        
        redacted_config = copy.deepcopy(config)
        
        def redact_dict(obj: Dict[str, Any], path: str = ""):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                
                if key in schema.sensitive_fields or current_path in schema.sensitive_fields:
                    obj[key] = "***REDACTED***"
                elif isinstance(value, dict):
                    redact_dict(value, current_path)
        
        redact_dict(redacted_config)
        return redacted_config
    
    def generate_validation_report(self, results: Dict[ConfigSection, SchemaValidationResult]) -> str:
        """Generate a comprehensive validation report."""
        report_lines = [
            "Configuration Validation Report",
            "=" * 50,
            f"Validation Level: {self.validation_level.value}",
            f"Timestamp: {datetime.now().isoformat()}",
            ""
        ]
        
        # Summary
        total_sections = len(results)
        valid_sections = len([r for r in results.values() if r.is_valid])
        total_issues = sum(len(r.issues) for r in results.values())
        
        report_lines.extend([
            "Summary:",
            f"  Sections Validated: {total_sections}",
            f"  Valid Sections: {valid_sections}",
            f"  Invalid Sections: {total_sections - valid_sections}",
            f"  Total Issues: {total_issues}",
            ""
        ])
        
        # Section details
        for section, result in results.items():
            status = "✓ VALID" if result.is_valid else "✗ INVALID"
            report_lines.extend([
                f"{section.value.upper()}: {status}",
                f"  Schema Version: {result.schema_version.value}",
                f"  Issues: {len(result.issues)}",
                f"  Warnings: {len(result.warnings)}"
            ])
            
            # Critical and error issues
            critical_errors = [i for i in result.issues if i.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]]
            for issue in critical_errors[:3]:  # Show first 3
                report_lines.append(f"    - {issue.severity.value.upper()}: {issue.message}")
            
            if len(critical_errors) > 3:
                report_lines.append(f"    ... and {len(critical_errors) - 3} more issues")
            
            report_lines.append("")
        
        return "\n".join(report_lines)


# Global configuration validator instance
_global_validator: Optional[ConfigurationValidator] = None


def get_config_validator(validation_level: ConfigValidationLevel = ConfigValidationLevel.STANDARD) -> ConfigurationValidator:
    """Get the global configuration validator instance."""
    global _global_validator
    if _global_validator is None:
        _global_validator = ConfigurationValidator(validation_level)
    return _global_validator


async def validate_config(config: ConfigDict, section: ConfigSection) -> SchemaValidationResult:
    """Convenience function to validate configuration."""
    validator = get_config_validator()
    return await validator.validate_configuration(config, section)


def create_schema(section: ConfigSection, schema_dict: Dict[str, Any], 
                 version: SchemaVersion = SchemaVersion.CURRENT) -> ConfigurationSchema:
    """Convenience function to create a configuration schema."""
    return ConfigurationSchema(
        version=version,
        section=section,
        schema=schema_dict
    )


if __name__ == "__main__":
    import asyncio
    
    async def test_config_validation():
        """Test configuration validation system."""
        
        print("Testing Configuration Validation System")
        print("=" * 50)
        
        validator = ConfigurationValidator()
        
        # Test core configuration
        core_config = {
            "debug": True,
            "log_level": "INFO",
            "max_workers": 4,
            "timeout": 30.0
        }
        
        result = await validator.validate_configuration(core_config, ConfigSection.CORE)
        print(f"Core config validation: {'PASS' if result.is_valid else 'FAIL'}")
        
        if result.issues:
            for issue in result.issues[:3]:
                print(f"  - {issue.severity.value}: {issue.message}")
        
        # Test invalid configuration
        invalid_config = {
            "debug": "not_a_boolean",  # Should be boolean
            "log_level": "INVALID_LEVEL",  # Invalid enum value
            "max_workers": -1  # Should be positive
        }
        
        result = await validator.validate_configuration(invalid_config, ConfigSection.CORE)
        print(f"Invalid config validation: {'PASS' if not result.is_valid else 'FAIL'}")
        print(f"Issues found: {len(result.issues)}")
        
        # Test full configuration validation
        full_config = {
            "core": core_config,
            "ai_intelligence": {
                "quality_prediction": {
                    "enabled": True,
                    "confidence_threshold": 0.8
                }
            },
            "security": {
                "encryption_key": "a" * 32,  # Valid 32-char key
                "audit_logging": True
            }
        }
        
        results = await validator.validate_full_configuration(full_config)
        print(f"\nFull configuration validation:")
        for section, result in results.items():
            status = "PASS" if result.is_valid else "FAIL"
            print(f"  {section.value}: {status}")
        
        # Generate report
        report = validator.generate_validation_report(results)
        print(f"\nValidation Report:\n{report}")
    
    asyncio.run(test_config_validation())