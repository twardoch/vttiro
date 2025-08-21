#!/usr/bin/env python3
# this_file: src/vttiro/utils/config_migration.py
"""Configuration migration utilities for vttiro version upgrades.

This module provides tools to migrate configuration files and environment
variables between different versions of vttiro, ensuring backward compatibility
and smooth upgrades.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    from loguru import logger
except ImportError:
    import logging as logger


class ConfigVersion(Enum):
    """Supported configuration versions."""
    V1_0 = "1.0"  # Initial version
    V2_0 = "2.0"  # Added engine/model separation
    V2_1 = "2.1"  # Added performance monitoring, config validation
    CURRENT = V2_1  # Always points to latest


@dataclass
class MigrationStep:
    """A single configuration migration step."""
    
    from_version: ConfigVersion
    to_version: ConfigVersion
    description: str
    required: bool = True  # Whether this migration is required or optional
    
    def apply(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply this migration step to configuration data.
        
        Args:
            config_data: Configuration data dictionary
            
        Returns:
            Migrated configuration data
        """
        # This is a base implementation - specific migrations should override
        return config_data


class V1ToV2Migration(MigrationStep):
    """Migration from v1.0 to v2.0 (engine/model separation)."""
    
    def __init__(self):
        super().__init__(
            from_version=ConfigVersion.V1_0,
            to_version=ConfigVersion.V2_0,
            description="Separate AI engines from models for clearer terminology"
        )
    
    def apply(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply v1 to v2 migration."""
        migrated = config_data.copy()
        
        # Handle old 'preferred_model' field that used to mix engines and models
        if 'preferred_model' in migrated:
            old_model = migrated['preferred_model']
            
            # Map old terminology to new engine/model structure
            model_mappings = {
                'gemini': {'engine': 'gemini', 'model': 'gemini-2.0-flash'},
                'assemblyai': {'engine': 'assemblyai', 'model': 'universal-2'},
                'deepgram': {'engine': 'deepgram', 'model': 'nova-3'},
                'mock': {'engine': 'mock', 'model': 'default'},
            }
            
            if old_model in model_mappings:
                mapping = model_mappings[old_model]
                migrated['preferred_engine'] = mapping['engine']
                migrated['preferred_model'] = mapping['model']
                logger.info(f"Migrated preferred_model '{old_model}' to engine='{mapping['engine']}', model='{mapping['model']}'")
            else:
                # Keep as-is and let validation handle it
                logger.warning(f"Unknown model '{old_model}' during migration - keeping as-is")
        
        # Add version tracking
        migrated['config_version'] = ConfigVersion.V2_0.value
        
        return migrated


class V2ToV2_1Migration(MigrationStep):
    """Migration from v2.0 to v2.1 (performance monitoring, validation)."""
    
    def __init__(self):
        super().__init__(
            from_version=ConfigVersion.V2_0,
            to_version=ConfigVersion.V2_1,
            description="Add performance monitoring and enhanced validation support"
        )
    
    def apply(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply v2.0 to v2.1 migration."""
        migrated = config_data.copy()
        
        # Add new performance monitoring settings with defaults
        if 'performance_monitoring' not in migrated:
            migrated['performance_monitoring'] = {
                'enabled': True,
                'track_resources': True,
                'track_operations': True,
                'max_sessions_history': 100,
            }
            logger.info("Added performance monitoring configuration")
        
        # Add validation settings
        if 'validation' not in migrated:
            migrated['validation'] = {
                'strict_mode': False,
                'validate_on_startup': True,
                'check_api_connectivity': False,
            }
            logger.info("Added validation configuration settings")
        
        # Update version
        migrated['config_version'] = ConfigVersion.V2_1.value
        
        return migrated


class ConfigurationMigrator:
    """Handles configuration migration between vttiro versions."""
    
    def __init__(self):
        """Initialize the configuration migrator."""
        self.migrations: List[MigrationStep] = [
            V1ToV2Migration(),
            V2ToV2_1Migration(),
        ]
    
    def detect_version(self, config_data: Dict[str, Any]) -> ConfigVersion:
        """Detect the version of a configuration.
        
        Args:
            config_data: Configuration data dictionary
            
        Returns:
            Detected configuration version
        """
        if 'config_version' in config_data:
            try:
                return ConfigVersion(config_data['config_version'])
            except ValueError:
                logger.warning(f"Unknown config version: {config_data['config_version']}")
        
        # Heuristic detection for configs without explicit version
        if 'preferred_engine' in config_data:
            return ConfigVersion.V2_0
        elif 'preferred_model' in config_data:
            return ConfigVersion.V1_0
        
        # Default to oldest version for safety
        return ConfigVersion.V1_0
    
    def get_migration_path(self, from_version: ConfigVersion, to_version: ConfigVersion) -> List[MigrationStep]:
        """Get the sequence of migrations needed to go from one version to another.
        
        Args:
            from_version: Starting configuration version
            to_version: Target configuration version
            
        Returns:
            List of migration steps in order
        """
        if from_version == to_version:
            return []
        
        # Build migration path
        migration_path = []
        current_version = from_version
        
        while current_version != to_version:
            # Find next migration step
            next_migration = None
            for migration in self.migrations:
                if migration.from_version == current_version:
                    next_migration = migration
                    break
            
            if next_migration is None:
                raise ValueError(f"No migration path found from {current_version} to {to_version}")
            
            migration_path.append(next_migration)
            current_version = next_migration.to_version
        
        return migration_path
    
    def migrate_config(self, config_data: Dict[str, Any], target_version: Optional[ConfigVersion] = None) -> Tuple[Dict[str, Any], List[str]]:
        """Migrate configuration data to target version.
        
        Args:
            config_data: Configuration data to migrate
            target_version: Target version (defaults to current)
            
        Returns:
            Tuple of (migrated_config, migration_log)
        """
        if target_version is None:
            target_version = ConfigVersion.CURRENT
        
        current_version = self.detect_version(config_data)
        migration_log = []
        
        if current_version == target_version:
            migration_log.append(f"Configuration is already at target version {target_version.value}")
            return config_data, migration_log
        
        migration_log.append(f"Starting migration from {current_version.value} to {target_version.value}")
        
        try:
            migration_path = self.get_migration_path(current_version, target_version)
            migrated_data = config_data.copy()
            
            for migration in migration_path:
                migration_log.append(f"Applying migration: {migration.description}")
                migrated_data = migration.apply(migrated_data)
                migration_log.append(f"Successfully migrated to version {migration.to_version.value}")
            
            migration_log.append("Migration completed successfully")
            return migrated_data, migration_log
            
        except Exception as e:
            error_msg = f"Migration failed: {str(e)}"
            migration_log.append(error_msg)
            logger.error(error_msg)
            raise
    
    def migrate_environment_variables(self) -> List[str]:
        """Migrate environment variables to current standards.
        
        Returns:
            List of migration actions performed
        """
        actions = []
        
        # Check for old environment variable names and suggest new ones
        old_to_new_mappings = {
            'GEMINI_API_KEY': 'VTTIRO_GEMINI_API_KEY',
            'ASSEMBLYAI_API_KEY': 'VTTIRO_ASSEMBLYAI_API_KEY',
            'DEEPGRAM_API_KEY': 'VTTIRO_DEEPGRAM_API_KEY',
            'OPENAI_API_KEY': 'VTTIRO_OPENAI_API_KEY',
            'TRANSCRIPTION_MODEL': 'VTTIRO_MODEL',
        }
        
        for old_name, new_name in old_to_new_mappings.items():
            old_value = os.getenv(old_name)
            new_value = os.getenv(new_name)
            
            if old_value and not new_value:
                actions.append(f"Found old environment variable {old_name}, recommend renaming to {new_name}")
                logger.warning(f"Old environment variable detected: {old_name} -> {new_name}")
            elif old_value and new_value:
                actions.append(f"Both {old_name} and {new_name} are set - using {new_name}")
                logger.info(f"Using new environment variable {new_name} (ignoring old {old_name})")
        
        return actions
    
    def create_migration_report(self, config_path: Optional[Path] = None) -> Dict[str, Any]:
        """Create a comprehensive migration report.
        
        Args:
            config_path: Path to configuration file (optional)
            
        Returns:
            Migration report dictionary
        """
        report = {
            'current_version': ConfigVersion.CURRENT.value,
            'migration_available': False,
            'environment_migration_needed': False,
            'recommendations': [],
        }
        
        # Check environment variables
        env_actions = self.migrate_environment_variables()
        if env_actions:
            report['environment_migration_needed'] = True
            report['environment_actions'] = env_actions
            report['recommendations'].extend([
                "Update environment variable names to use VTTIRO_ prefix",
                "Check environment variable documentation for current standards"
            ])
        
        # Check configuration file if provided
        if config_path and config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                
                current_version = self.detect_version(config_data)
                if current_version != ConfigVersion.CURRENT:
                    report['migration_available'] = True
                    report['config_current_version'] = current_version.value
                    report['migration_path'] = [
                        {
                            'from': m.from_version.value,
                            'to': m.to_version.value,
                            'description': m.description,
                            'required': m.required
                        }
                        for m in self.get_migration_path(current_version, ConfigVersion.CURRENT)
                    ]
                    report['recommendations'].append(f"Update configuration file from version {current_version.value} to {ConfigVersion.CURRENT.value}")
                
            except Exception as e:
                report['config_error'] = str(e)
                report['recommendations'].append("Configuration file could not be read - check file format")
        
        return report


def check_migration_needed() -> bool:
    """Quick check if any migration is needed.
    
    Returns:
        True if migration is recommended
    """
    migrator = ConfigurationMigrator()
    
    # Check environment variables
    env_actions = migrator.migrate_environment_variables()
    if env_actions:
        return True
    
    # Check for common config file locations
    config_paths = [
        Path.home() / '.vttiro' / 'config.json',
        Path.cwd() / 'vttiro.json',
        Path.cwd() / '.vttiro.json',
    ]
    
    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                
                current_version = migrator.detect_version(config_data)
                if current_version != ConfigVersion.CURRENT:
                    return True
                    
            except Exception:
                continue  # Skip files that can't be read
    
    return False


def suggest_migration_command() -> Optional[str]:
    """Suggest appropriate migration command if needed.
    
    Returns:
        Suggested CLI command or None if no migration needed
    """
    if check_migration_needed():
        return "vttiro config_migrate"
    return None