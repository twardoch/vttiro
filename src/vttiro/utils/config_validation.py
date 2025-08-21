#!/usr/bin/env python3
# this_file: src/vttiro/utils/config_validation.py
"""Enhanced configuration validation, health checks, and debugging utilities.

This module provides comprehensive configuration validation, API key testing,
connectivity checks, and configuration debugging capabilities.
"""

import os
import re
import asyncio
import httpx
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

try:
    from loguru import logger
except ImportError:
    import logging as logger

from vttiro.core.config import VttiroConfig


class ValidationSeverity(Enum):
    """Severity levels for configuration validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a configuration validation check."""
    
    check_name: str
    severity: ValidationSeverity
    status: bool  # True = passed, False = failed
    message: str
    details: Optional[str] = None
    recommendation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class ConfigurationHealth:
    """Overall configuration health assessment."""
    
    overall_status: str  # "healthy", "warnings", "errors", "critical"
    total_checks: int
    passed_checks: int
    warning_count: int
    error_count: int
    critical_count: int
    results: List[ValidationResult] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate the percentage of checks that passed."""
        if self.total_checks == 0:
            return 100.0
        return (self.passed_checks / self.total_checks) * 100


class ConfigurationValidator:
    """Comprehensive configuration validation and health checking system.
    
    Validates environment variables, API keys, connectivity, and provides
    detailed debugging information and recommendations.
    """
    
    def __init__(self, config: Optional[VttiroConfig] = None):
        """Initialize the configuration validator.
        
        Args:
            config: VttiroConfig instance to validate (creates default if None)
        """
        self.config = config or VttiroConfig()
        self.results: List[ValidationResult] = []
        
        # API endpoints for connectivity testing
        self.api_endpoints = {
            "gemini": "https://generativelanguage.googleapis.com/v1beta/models",
            "assemblyai": "https://api.assemblyai.com/v2/realtime/token",
            "deepgram": "https://api.deepgram.com/v1/projects",
            "openai": "https://api.openai.com/v1/models",
        }
    
    def validate_all(self, check_connectivity: bool = False) -> ConfigurationHealth:
        """Run comprehensive configuration validation.
        
        Args:
            check_connectivity: Whether to test API connectivity (slower)
            
        Returns:
            ConfigurationHealth with complete validation results
        """
        self.results = []
        
        # Environment variable validation
        self._validate_environment_variables()
        
        # Configuration structure validation
        self._validate_config_structure()
        
        # API key validation
        self._validate_api_keys()
        
        # File path validation
        self._validate_file_paths()
        
        # Value range validation
        self._validate_value_ranges()
        
        # Dependency checks
        self._validate_dependencies()
        
        # API connectivity tests (optional, slower)
        if check_connectivity:
            asyncio.run(self._validate_api_connectivity())
        
        return self._generate_health_report()
    
    def _validate_environment_variables(self):
        """Validate environment variable presence and format."""
        env_vars = {
            "VTTIRO_GEMINI_API_KEY": {
                "required": False,
                "pattern": r"^AIza[0-9A-Za-z\-_]{35}$",
                "description": "Google Gemini API key",
            },
            "VTTIRO_ASSEMBLYAI_API_KEY": {
                "required": False, 
                "pattern": r"^[0-9a-f]{32}$",
                "description": "AssemblyAI API key",
            },
            "VTTIRO_DEEPGRAM_API_KEY": {
                "required": False,
                "pattern": r"^[0-9a-f]{40}$", 
                "description": "Deepgram API key",
            },
            "VTTIRO_OPENAI_API_KEY": {
                "required": False,
                "pattern": r"^sk-[0-9A-Za-z]{20}T3BlbkFJ[0-9A-Za-z]{20}$",
                "description": "OpenAI API key",
            },
            "VTTIRO_MODEL": {
                "required": False,
                "pattern": r"^[a-z][a-z0-9\-_.]*$",
                "description": "Default model preference",
            },
        }
        
        found_keys = 0
        for var_name, requirements in env_vars.items():
            env_value = os.getenv(var_name)
            
            if env_value:
                found_keys += 1
                # Validate format if pattern provided
                if "pattern" in requirements:
                    if re.match(requirements["pattern"], env_value):
                        self.results.append(ValidationResult(
                            check_name=f"env_var_{var_name.lower()}",
                            severity=ValidationSeverity.INFO,
                            status=True,
                            message=f"{requirements['description']} is properly formatted",
                            metadata={"variable": var_name, "masked_value": f"{env_value[:8]}..."}
                        ))
                    else:
                        self.results.append(ValidationResult(
                            check_name=f"env_var_{var_name.lower()}",
                            severity=ValidationSeverity.WARNING,
                            status=False,
                            message=f"{requirements['description']} has invalid format",
                            details=f"Expected pattern: {requirements['pattern']}",
                            recommendation=f"Check {var_name} format in your environment",
                            metadata={"variable": var_name}
                        ))
                else:
                    self.results.append(ValidationResult(
                        check_name=f"env_var_{var_name.lower()}",
                        severity=ValidationSeverity.INFO,
                        status=True,
                        message=f"{requirements['description']} is set",
                        metadata={"variable": var_name}
                    ))
            elif requirements.get("required", False):
                self.results.append(ValidationResult(
                    check_name=f"env_var_{var_name.lower()}",
                    severity=ValidationSeverity.ERROR,
                    status=False,
                    message=f"Required environment variable {var_name} is missing",
                    recommendation=f"Set {var_name} in your environment",
                    metadata={"variable": var_name}
                ))
        
        # Overall environment assessment
        if found_keys == 0:
            self.results.append(ValidationResult(
                check_name="environment_setup",
                severity=ValidationSeverity.WARNING,
                status=False,
                message="No API keys configured - only mock transcription will work",
                recommendation="Configure at least one API key: VTTIRO_GEMINI_API_KEY, VTTIRO_ASSEMBLYAI_API_KEY, VTTIRO_DEEPGRAM_API_KEY, or VTTIRO_OPENAI_API_KEY"
            ))
        elif found_keys == 1:
            self.results.append(ValidationResult(
                check_name="environment_setup",
                severity=ValidationSeverity.INFO,
                status=True,
                message=f"Single API key configured - transcription available with limited fallback options",
                recommendation="Consider configuring additional API keys for better reliability"
            ))
        else:
            self.results.append(ValidationResult(
                check_name="environment_setup", 
                severity=ValidationSeverity.INFO,
                status=True,
                message=f"{found_keys} API keys configured - excellent redundancy",
                metadata={"api_key_count": found_keys}
            ))
    
    def _validate_config_structure(self):
        """Validate the configuration object structure and types."""
        try:
            # Test that config can be serialized (basic structure validation)
            config_dict = self.config.__dict__
            
            # Check required sub-configs and their attributes
            required_subconfigs = {
                "transcription": ["preferred_model", "confidence_threshold"],
                "processing": ["chunk_duration"],
                "output": ["max_chars_per_line"],
            }
            
            missing_attrs = []
            
            # Check main config attributes
            if not hasattr(self.config, 'verbose'):
                missing_attrs.append("verbose")
            
            # Check sub-configs and their attributes
            for subconfig_name, attrs in required_subconfigs.items():
                if not hasattr(self.config, subconfig_name):
                    missing_attrs.append(f"{subconfig_name} sub-config")
                else:
                    subconfig = getattr(self.config, subconfig_name)
                    for attr in attrs:
                        if not hasattr(subconfig, attr):
                            missing_attrs.append(f"{subconfig_name}.{attr}")
            
            if missing_attrs:
                self.results.append(ValidationResult(
                    check_name="config_structure",
                    severity=ValidationSeverity.ERROR,
                    status=False,
                    message=f"Configuration missing required attributes: {', '.join(missing_attrs)}",
                    recommendation="Check VttiroConfig initialization"
                ))
            else:
                self.results.append(ValidationResult(
                    check_name="config_structure",
                    severity=ValidationSeverity.INFO,
                    status=True,
                    message="Configuration structure is valid",
                    metadata={"attributes_count": len(config_dict)}
                ))
                
        except Exception as e:
            self.results.append(ValidationResult(
                check_name="config_structure",
                severity=ValidationSeverity.CRITICAL,
                status=False,
                message=f"Configuration structure validation failed: {str(e)}",
                recommendation="Check VttiroConfig class definition and initialization"
            ))
    
    def _validate_api_keys(self):
        """Validate API key presence and basic format in config object."""
        api_keys = {
            "Gemini": self.config.transcription.gemini_api_key,
            "AssemblyAI": self.config.transcription.assemblyai_api_key, 
            "Deepgram": self.config.transcription.deepgram_api_key,
            "OpenAI": self.config.transcription.openai_api_key,
        }
        
        configured_keys = 0
        for service, api_key in api_keys.items():
            if api_key and api_key.strip():
                configured_keys += 1
                
                # Basic format validation
                key = api_key.strip()
                if len(key) < 10:
                    self.results.append(ValidationResult(
                        check_name=f"api_key_{service.lower()}",
                        severity=ValidationSeverity.WARNING,
                        status=False,
                        message=f"{service} API key appears too short",
                        recommendation=f"Verify {service} API key is complete"
                    ))
                elif len(key) > 200:
                    self.results.append(ValidationResult(
                        check_name=f"api_key_{service.lower()}",
                        severity=ValidationSeverity.WARNING,
                        status=False,
                        message=f"{service} API key appears unusually long",
                        recommendation=f"Verify {service} API key format"
                    ))
                else:
                    self.results.append(ValidationResult(
                        check_name=f"api_key_{service.lower()}",
                        severity=ValidationSeverity.INFO,
                        status=True,
                        message=f"{service} API key is configured",
                        metadata={"key_length": len(key), "key_prefix": key[:8]}
                    ))
            else:
                self.results.append(ValidationResult(
                    check_name=f"api_key_{service.lower()}",
                    severity=ValidationSeverity.INFO,
                    status=False,
                    message=f"{service} API key not configured",
                    details=f"Service will not be available for transcription",
                    recommendation=f"Configure {service} API key if you plan to use this service"
                ))
        
        # Overall API key assessment
        if configured_keys == 0:
            self.results.append(ValidationResult(
                check_name="api_keys_overall",
                severity=ValidationSeverity.CRITICAL,
                status=False,
                message="No API keys are configured in the configuration object",
                recommendation="Configure at least one API key for transcription services"
            ))
    
    def _validate_file_paths(self):
        """Validate file paths and permissions in configuration."""
        # For now, focus on basic path validation
        # This could be extended to validate specific file paths if they're part of config
        
        # Check current working directory permissions
        cwd = Path.cwd()
        
        if cwd.exists() and os.access(cwd, os.W_OK):
            self.results.append(ValidationResult(
                check_name="working_directory",
                severity=ValidationSeverity.INFO,
                status=True,
                message="Working directory is writable",
                metadata={"path": str(cwd)}
            ))
        else:
            self.results.append(ValidationResult(
                check_name="working_directory",
                severity=ValidationSeverity.WARNING,
                status=False,
                message="Working directory may not be writable",
                recommendation="Check file permissions for output file creation",
                metadata={"path": str(cwd)}
            ))
    
    def _validate_value_ranges(self):
        """Validate configuration value ranges and constraints."""
        validations = [
            {
                "field": "chunk_duration",
                "value": self.config.processing.chunk_duration,
                "min_val": 5,
                "max_val": 600,
                "recommended_range": (30, 300),
                "unit": "seconds"
            },
            {
                "field": "max_chars_per_line", 
                "value": self.config.output.max_chars_per_line,
                "min_val": 20,
                "max_val": 200,
                "recommended_range": (40, 80),
                "unit": "characters"
            },
            {
                "field": "confidence_threshold",
                "value": self.config.transcription.confidence_threshold, 
                "min_val": 0.0,
                "max_val": 1.0,
                "recommended_range": (0.5, 0.9),
                "unit": ""
            }
        ]
        
        for validation in validations:
            field = validation["field"]
            value = validation["value"]
            min_val = validation["min_val"]
            max_val = validation["max_val"]
            rec_min, rec_max = validation["recommended_range"]
            unit = validation["unit"]
            
            if value < min_val or value > max_val:
                self.results.append(ValidationResult(
                    check_name=f"value_range_{field}",
                    severity=ValidationSeverity.ERROR,
                    status=False,
                    message=f"{field} value {value}{unit} is outside valid range [{min_val}, {max_val}]",
                    recommendation=f"Set {field} between {min_val} and {max_val} {unit}",
                    metadata={"field": field, "value": value, "valid_range": [min_val, max_val]}
                ))
            elif value < rec_min or value > rec_max:
                self.results.append(ValidationResult(
                    check_name=f"value_range_{field}",
                    severity=ValidationSeverity.WARNING,
                    status=True,
                    message=f"{field} value {value}{unit} is outside recommended range [{rec_min}, {rec_max}]",
                    recommendation=f"Consider setting {field} between {rec_min} and {rec_max} {unit} for optimal performance",
                    metadata={"field": field, "value": value, "recommended_range": [rec_min, rec_max]}
                ))
            else:
                self.results.append(ValidationResult(
                    check_name=f"value_range_{field}",
                    severity=ValidationSeverity.INFO,
                    status=True,
                    message=f"{field} value {value}{unit} is within recommended range",
                    metadata={"field": field, "value": value}
                ))
    
    def _validate_dependencies(self):
        """Validate required dependencies and optional components."""
        required_deps = {
            "loguru": "Advanced logging",
            "pydantic": "Configuration validation",
            "rich": "Terminal output formatting",
            "fire": "CLI framework",
            "httpx": "HTTP client for API calls",
            "psutil": "Performance monitoring",
        }
        
        optional_deps = {
            "ffmpeg-python": "Audio processing",
            "yt-dlp": "Video downloading",
        }
        
        # Check required dependencies
        missing_required = []
        for dep, description in required_deps.items():
            try:
                __import__(dep)
                self.results.append(ValidationResult(
                    check_name=f"dependency_{dep}",
                    severity=ValidationSeverity.INFO,
                    status=True,
                    message=f"Required dependency '{dep}' is available",
                    metadata={"dependency": dep, "description": description}
                ))
            except ImportError:
                missing_required.append(dep)
                self.results.append(ValidationResult(
                    check_name=f"dependency_{dep}",
                    severity=ValidationSeverity.CRITICAL,
                    status=False,
                    message=f"Required dependency '{dep}' is missing",
                    details=f"Used for: {description}",
                    recommendation=f"Install with: uv pip install {dep}",
                    metadata={"dependency": dep}
                ))
        
        # Check optional dependencies
        for dep, description in optional_deps.items():
            try:
                __import__(dep.replace("-", "_"))  # Handle package name differences
                self.results.append(ValidationResult(
                    check_name=f"optional_dependency_{dep}",
                    severity=ValidationSeverity.INFO,
                    status=True,
                    message=f"Optional dependency '{dep}' is available",
                    metadata={"dependency": dep, "description": description}
                ))
            except ImportError:
                self.results.append(ValidationResult(
                    check_name=f"optional_dependency_{dep}",
                    severity=ValidationSeverity.WARNING,
                    status=False,
                    message=f"Optional dependency '{dep}' is missing",
                    details=f"Used for: {description}",
                    recommendation=f"Install with: uv pip install {dep} (if needed)",
                    metadata={"dependency": dep}
                ))
    
    async def _validate_api_connectivity(self):
        """Test connectivity to configured API endpoints."""
        timeout_seconds = 10
        
        apis_to_test = []
        if self.config.transcription.gemini_api_key:
            apis_to_test.append(("gemini", self.config.transcription.gemini_api_key))
        if self.config.transcription.assemblyai_api_key:
            apis_to_test.append(("assemblyai", self.config.transcription.assemblyai_api_key))
        if self.config.transcription.deepgram_api_key:
            apis_to_test.append(("deepgram", self.config.transcription.deepgram_api_key))
        if self.config.transcription.openai_api_key:
            apis_to_test.append(("openai", self.config.transcription.openai_api_key))
        
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            for service, api_key in apis_to_test:
                try:
                    endpoint = self.api_endpoints.get(service)
                    if not endpoint:
                        continue
                    
                    # Prepare service-specific headers
                    headers = {}
                    if service == "gemini":
                        headers = {"x-goog-api-key": api_key}
                    elif service == "assemblyai":
                        headers = {"authorization": api_key}
                    elif service == "deepgram":
                        headers = {"authorization": f"Token {api_key}"}
                    elif service == "openai":
                        headers = {"authorization": f"Bearer {api_key}"}
                    
                    response = await client.get(endpoint, headers=headers)
                    
                    if response.status_code in [200, 401]:  # 401 might indicate valid endpoint but invalid key
                        if response.status_code == 200:
                            self.results.append(ValidationResult(
                                check_name=f"connectivity_{service}",
                                severity=ValidationSeverity.INFO,
                                status=True,
                                message=f"{service.title()} API is accessible and API key is valid",
                                metadata={"service": service, "status_code": response.status_code}
                            ))
                        else:
                            self.results.append(ValidationResult(
                                check_name=f"connectivity_{service}",
                                severity=ValidationSeverity.WARNING,
                                status=False,
                                message=f"{service.title()} API is accessible but API key may be invalid",
                                details=f"HTTP {response.status_code}: API endpoint reachable but authentication failed",
                                recommendation=f"Verify {service.title()} API key is correct and has proper permissions",
                                metadata={"service": service, "status_code": response.status_code}
                            ))
                    else:
                        self.results.append(ValidationResult(
                            check_name=f"connectivity_{service}",
                            severity=ValidationSeverity.WARNING,
                            status=False,
                            message=f"{service.title()} API connectivity test failed",
                            details=f"HTTP {response.status_code}: {response.reason_phrase}",
                            recommendation=f"Check {service.title()} API status and network connectivity",
                            metadata={"service": service, "status_code": response.status_code}
                        ))
                
                except httpx.TimeoutException:
                    self.results.append(ValidationResult(
                        check_name=f"connectivity_{service}",
                        severity=ValidationSeverity.WARNING,
                        status=False,
                        message=f"{service.title()} API connection timed out",
                        details=f"Request timed out after {timeout_seconds} seconds",
                        recommendation="Check network connectivity and firewall settings",
                        metadata={"service": service, "timeout": timeout_seconds}
                    ))
                
                except Exception as e:
                    self.results.append(ValidationResult(
                        check_name=f"connectivity_{service}",
                        severity=ValidationSeverity.ERROR,
                        status=False,
                        message=f"{service.title()} API connectivity test error",
                        details=f"Error: {str(e)}",
                        recommendation=f"Check {service.title()} API configuration and network settings",
                        metadata={"service": service, "error": str(e)}
                    ))
    
    def _generate_health_report(self) -> ConfigurationHealth:
        """Generate comprehensive configuration health report."""
        total_checks = len(self.results)
        passed_checks = sum(1 for r in self.results if r.status)
        
        # Count by severity
        warning_count = sum(1 for r in self.results if r.severity == ValidationSeverity.WARNING)
        error_count = sum(1 for r in self.results if r.severity == ValidationSeverity.ERROR) 
        critical_count = sum(1 for r in self.results if r.severity == ValidationSeverity.CRITICAL)
        
        # Determine overall status
        if critical_count > 0:
            overall_status = "critical"
        elif error_count > 0:
            overall_status = "errors"
        elif warning_count > 0:
            overall_status = "warnings"
        else:
            overall_status = "healthy"
        
        # Generate recommendations
        recommendations = []
        for result in self.results:
            if result.recommendation and result.recommendation not in recommendations:
                recommendations.append(result.recommendation)
        
        return ConfigurationHealth(
            overall_status=overall_status,
            total_checks=total_checks,
            passed_checks=passed_checks,
            warning_count=warning_count,
            error_count=error_count,
            critical_count=critical_count,
            results=self.results,
            recommendations=recommendations[:10]  # Limit to top 10 recommendations
        )
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get comprehensive configuration debugging information."""
        debug_info = {
            "timestamp": logger.info(f"Configuration debug info generated"),
            "config_object": {
                "type": type(self.config).__name__,
                "attributes": {
                    key: "***REDACTED***" if "api_key" in key.lower() else value
                    for key, value in self.config.__dict__.items()
                }
            },
            "environment_variables": {
                key: "***REDACTED***" if key.lower().endswith("_api_key") else value
                for key, value in os.environ.items()
                if key.startswith("VTTIRO_")
            },
            "system_info": {
                "python_version": os.sys.version,
                "platform": os.name,
                "working_directory": str(Path.cwd()),
                "user": os.getenv("USER", "unknown"),
            },
            "validation_summary": {
                "total_results": len(self.results),
                "by_severity": {
                    "info": sum(1 for r in self.results if r.severity == ValidationSeverity.INFO),
                    "warning": sum(1 for r in self.results if r.severity == ValidationSeverity.WARNING),
                    "error": sum(1 for r in self.results if r.severity == ValidationSeverity.ERROR),
                    "critical": sum(1 for r in self.results if r.severity == ValidationSeverity.CRITICAL),
                }
            }
        }
        
        return debug_info


def validate_startup_configuration(config: Optional[VttiroConfig] = None) -> Tuple[bool, List[str]]:
    """Quick startup configuration validation.
    
    Args:
        config: VttiroConfig to validate (creates default if None)
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    validator = ConfigurationValidator(config)
    health = validator.validate_all(check_connectivity=False)  # Fast validation only
    
    # Check for critical issues that would prevent startup
    critical_issues = [
        result.message for result in health.results 
        if result.severity == ValidationSeverity.CRITICAL and not result.status
    ]
    
    is_valid = len(critical_issues) == 0
    return is_valid, critical_issues