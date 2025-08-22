# this_file: src/vttiro/utils/debugging.py
"""Advanced debugging and diagnostic tools for VTTiro 2.0.

Provides comprehensive debugging utilities, enhanced error reporting,
diagnostic information collection, and troubleshooting guides for
improved developer experience.
"""

import asyncio
import json
import logging
import os
import platform
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
import inspect

from ..core.config import VttiroConfig
from ..core.errors import VttiroError, ProviderError, ConfigurationError
# from ..core.registry import get_registry  # Removed - registry infrastructure removed
# from ..monitoring.health import global_health_monitor, HealthStatus  # Removed - monitoring infrastructure removed
# from ..monitoring.production import production_monitor  # Removed - monitoring infrastructure removed


@dataclass
class DiagnosticInfo:
    """Comprehensive diagnostic information."""
    timestamp: datetime
    platform_info: Dict[str, str]
    python_info: Dict[str, str]
    vttiro_info: Dict[str, Any]
    environment_vars: Dict[str, str]
    installed_packages: List[str]
    configuration: Optional[Dict[str, Any]] = None
    provider_status: Optional[Dict[str, Any]] = None
    recent_errors: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert diagnostic info to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "platform_info": self.platform_info,
            "python_info": self.python_info,
            "vttiro_info": self.vttiro_info,
            "environment_vars": self.environment_vars,
            "installed_packages": self.installed_packages,
            "configuration": self.configuration,
            "provider_status": self.provider_status,
            "recent_errors": self.recent_errors,
            "performance_metrics": self.performance_metrics
        }


@dataclass
class ErrorContext:
    """Enhanced error context for debugging."""
    error_id: str
    timestamp: datetime
    error_type: str
    error_message: str
    stack_trace: str
    function_name: str
    file_path: str
    line_number: int
    local_variables: Dict[str, str]
    call_stack: List[Dict[str, str]]
    environment_context: Dict[str, Any]
    provider_context: Optional[Dict[str, Any]] = None
    configuration_context: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error context to dictionary."""
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp.isoformat(),
            "error_type": self.error_type,
            "error_message": self.error_message,
            "stack_trace": self.stack_trace,
            "function_name": self.function_name,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "local_variables": self.local_variables,
            "call_stack": self.call_stack,
            "environment_context": self.environment_context,
            "provider_context": self.provider_context,
            "configuration_context": self.configuration_context
        }


class EnhancedErrorReporter:
    """Enhanced error reporting with comprehensive context."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.logger = logging.getLogger(__name__)
        self.error_history: List[ErrorContext] = []
        self.max_history = 100
        
    def capture_error(self, 
                     error: Exception, 
                     context: Optional[Dict[str, Any]] = None,
                     provider_name: Optional[str] = None) -> ErrorContext:
        """Capture comprehensive error context for debugging."""
        
        if not self.enabled:
            # Create minimal context if disabled
            return ErrorContext(
                error_id=f"error_{int(time.time())}",
                timestamp=datetime.now(),
                error_type=type(error).__name__,
                error_message=str(error),
                stack_trace="",
                function_name="",
                file_path="",
                line_number=0,
                local_variables={},
                call_stack=[],
                environment_context={}
            )
        
        # Generate unique error ID
        error_id = f"vttiro_error_{int(time.time())}_{id(error)}"
        
        # Get stack trace information
        tb = error.__traceback__
        stack_trace = traceback.format_exception(type(error), error, tb)
        
        # Get frame information
        frame_info = self._extract_frame_info(tb)
        
        # Collect local variables safely
        local_vars = self._collect_safe_variables(tb.tb_frame.f_locals if tb else {})
        
        # Build call stack
        call_stack = self._build_call_stack(tb)
        
        # Collect environment context
        env_context = self._collect_environment_context()
        
        # Provider-specific context
        provider_context = None
        if provider_name:
            provider_context = self._collect_provider_context(provider_name)
        
        # Configuration context
        config_context = self._collect_configuration_context()
        
        error_context = ErrorContext(
            error_id=error_id,
            timestamp=datetime.now(),
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace="".join(stack_trace),
            function_name=frame_info.get("function", "unknown"),
            file_path=frame_info.get("filename", "unknown"),
            line_number=frame_info.get("lineno", 0),
            local_variables=local_vars,
            call_stack=call_stack,
            environment_context=env_context,
            provider_context=provider_context,
            configuration_context=config_context
        )
        
        # Store in history
        self.error_history.append(error_context)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        # Log enhanced error information
        self._log_enhanced_error(error_context)
        
        return error_context
    
    def _extract_frame_info(self, tb) -> Dict[str, Any]:
        """Extract information from traceback frame."""
        if not tb:
            return {}
        
        frame = tb.tb_frame
        return {
            "filename": frame.f_code.co_filename,
            "function": frame.f_code.co_name,
            "lineno": tb.tb_lineno
        }
    
    def _collect_safe_variables(self, variables: Dict[str, Any]) -> Dict[str, str]:
        """Safely collect local variables for debugging."""
        safe_vars = {}
        
        for name, value in variables.items():
            try:
                # Avoid sensitive information
                if any(sensitive in name.lower() for sensitive in ['key', 'password', 'token', 'secret']):
                    safe_vars[name] = "<REDACTED>"
                    continue
                
                # Limit string length and handle various types
                if isinstance(value, str):
                    safe_vars[name] = value[:200] + "..." if len(value) > 200 else value
                elif isinstance(value, (int, float, bool, type(None))):
                    safe_vars[name] = str(value)
                elif isinstance(value, (list, tuple)):
                    safe_vars[name] = f"<{type(value).__name__} with {len(value)} items>"
                elif isinstance(value, dict):
                    safe_vars[name] = f"<dict with {len(value)} keys>"
                else:
                    safe_vars[name] = f"<{type(value).__name__}>"
                    
            except Exception:
                safe_vars[name] = "<unable to serialize>"
        
        return safe_vars
    
    def _build_call_stack(self, tb) -> List[Dict[str, str]]:
        """Build readable call stack from traceback."""
        call_stack = []
        
        while tb:
            frame = tb.tb_frame
            call_stack.append({
                "filename": frame.f_code.co_filename,
                "function": frame.f_code.co_name,
                "line_number": str(tb.tb_lineno),
                "code_context": self._get_code_context(frame.f_code.co_filename, tb.tb_lineno)
            })
            tb = tb.tb_next
        
        return call_stack
    
    def _get_code_context(self, filename: str, line_number: int) -> str:
        """Get code context around the error line."""
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            # Get context lines (3 before, 1 current, 3 after)
            start = max(0, line_number - 4)
            end = min(len(lines), line_number + 3)
            
            context_lines = []
            for i in range(start, end):
                marker = ">>>" if i == line_number - 1 else "   "
                context_lines.append(f"{marker} {i+1:4d}: {lines[i].rstrip()}")
            
            return "\n".join(context_lines)
            
        except Exception:
            return f"Unable to read source code from {filename}"
    
    def _collect_environment_context(self) -> Dict[str, Any]:
        """Collect relevant environment context."""
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "working_directory": str(Path.cwd()),
            "environment_variables": {
                k: v for k, v in os.environ.items() 
                if k.startswith('VTTIRO_') or k.endswith('_API_KEY')
            },
            "memory_usage_mb": self._get_memory_usage(),
            "disk_free_gb": self._get_disk_free()
        }
    
    def _collect_provider_context(self, provider_name: str) -> Dict[str, Any]:
        """Collect provider-specific context."""
        try:
            # registry = get_registry()  # Removed - registry infrastructure removed
            
            context = {
                "provider_name": provider_name,
                "available_providers": ["gemini", "openai", "assemblyai", "deepgram"],  # Simplified list
                "provider_config": None  # registry.get_provider_config(provider_name) if registry else None
            }
            
            # Get health status if available
            try:
                health_data = global_health_monitor.get_all_health_status()
                if provider_name in health_data:
                    context["health_status"] = health_data[provider_name]
            except Exception:
                context["health_status"] = "unavailable"
            
            return context
            
        except Exception as e:
            return {"error": f"Failed to collect provider context: {e}"}
    
    def _collect_configuration_context(self) -> Dict[str, Any]:
        """Collect configuration context."""
        try:
            # Try to get current configuration
            config_context = {
                "config_available": False,
                "config_valid": False
            }
            
            # This would need to be implemented based on how config is accessed
            # For now, return basic info
            return config_context
            
        except Exception as e:
            return {"error": f"Failed to collect configuration context: {e}"}
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
        except Exception:
            return 0.0
    
    def _get_disk_free(self) -> float:
        """Get free disk space in GB."""
        try:
            import psutil
            disk_usage = psutil.disk_usage('.')
            return disk_usage.free / (1024 ** 3)
        except ImportError:
            return 0.0
        except Exception:
            return 0.0
    
    def _log_enhanced_error(self, error_context: ErrorContext):
        """Log enhanced error information."""
        self.logger.error(f"Enhanced Error Report [{error_context.error_id}]")
        self.logger.error(f"  Type: {error_context.error_type}")
        self.logger.error(f"  Message: {error_context.error_message}")
        self.logger.error(f"  Location: {error_context.file_path}:{error_context.line_number} in {error_context.function_name}")
        
        if error_context.provider_context:
            self.logger.error(f"  Provider: {error_context.provider_context.get('provider_name', 'unknown')}")
        
        # Log stack trace at debug level to avoid spam
        self.logger.debug(f"Stack trace for {error_context.error_id}:\n{error_context.stack_trace}")
    
    def get_error_summary(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get summary of recent errors."""
        recent_errors = self.error_history[-limit:] if self.error_history else []
        return [
            {
                "error_id": err.error_id,
                "timestamp": err.timestamp.isoformat(),
                "type": err.error_type,
                "message": err.error_message[:100] + ("..." if len(err.error_message) > 100 else ""),
                "location": f"{Path(err.file_path).name}:{err.line_number}"
            }
            for err in recent_errors
        ]
    
    def export_error_report(self, error_id: str, output_file: Path) -> bool:
        """Export detailed error report to file."""
        try:
            error_context = next((err for err in self.error_history if err.error_id == error_id), None)
            if not error_context:
                return False
            
            with open(output_file, 'w') as f:
                json.dump(error_context.to_dict(), f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export error report: {e}")
            return False


class SystemDiagnostics:
    """Comprehensive system diagnostics for troubleshooting."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def collect_full_diagnostics(self) -> DiagnosticInfo:
        """Collect comprehensive diagnostic information."""
        
        # Platform information
        platform_info = {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "architecture": str(platform.architecture()),
            "hostname": platform.node()
        }
        
        # Python information
        python_info = {
            "version": sys.version,
            "executable": sys.executable,
            "platform": sys.platform,
            "api_version": str(sys.api_version),
            "byte_order": sys.byteorder,
            "max_size": str(sys.maxsize),
            "path": sys.path[:5]  # First 5 entries to avoid spam
        }
        
        # VTTiro information
        vttiro_info = await self._collect_vttiro_info()
        
        # Environment variables
        env_vars = {
            k: v for k, v in os.environ.items()
            if any(prefix in k.upper() for prefix in ['VTTIRO', 'PYTHON', 'PATH']) 
            or k.endswith('_API_KEY')
        }
        
        # Redact sensitive information
        for key in env_vars:
            if 'key' in key.lower() or 'token' in key.lower() or 'secret' in key.lower():
                if env_vars[key]:
                    env_vars[key] = f"<REDACTED:{len(env_vars[key])} chars>"
        
        # Installed packages
        installed_packages = self._get_installed_packages()
        
        # Provider status
        provider_status = await self._get_provider_status()
        
        # Performance metrics
        performance_metrics = self._get_performance_metrics()
        
        # Recent errors
        recent_errors = self._get_recent_errors()
        
        return DiagnosticInfo(
            timestamp=datetime.now(),
            platform_info=platform_info,
            python_info=python_info,
            vttiro_info=vttiro_info,
            environment_vars=env_vars,
            installed_packages=installed_packages,
            provider_status=provider_status,
            performance_metrics=performance_metrics,
            recent_errors=recent_errors
        )
    
    async def _collect_vttiro_info(self) -> Dict[str, Any]:
        """Collect VTTiro-specific information."""
        try:
            from .. import __version__
            vttiro_version = __version__
        except ImportError:
            vttiro_version = "unknown"
        
        try:
            # registry = get_registry()  # Removed - registry infrastructure removed
            available_providers = ["gemini", "openai", "assemblyai", "deepgram"]  # Simplified list
        except Exception:
            available_providers = []
        
        return {
            "version": vttiro_version,
            "available_providers": available_providers,
            "module_path": str(Path(__file__).parent.parent),
            "config_path": str(Path.cwd()),
        }
    
    def _get_installed_packages(self) -> List[str]:
        """Get list of relevant installed packages."""
        try:
            import pkg_resources
            packages = []
            
            # Key packages for VTTiro
            relevant_packages = [
                'vttiro', 'pydantic', 'aiohttp', 'numpy', 'torch', 
                'transformers', 'openai', 'google', 'assemblyai', 'deepgram'
            ]
            
            for package in pkg_resources.working_set:
                if any(rel in package.key for rel in relevant_packages):
                    packages.append(f"{package.key}=={package.version}")
            
            return sorted(packages)
            
        except ImportError:
            return ["pkg_resources not available"]
        except Exception as e:
            return [f"Error collecting packages: {e}"]
    
    async def _get_provider_status(self) -> Dict[str, Any]:
        """Get status of all providers."""
        try:
            return await global_health_monitor.get_all_health_status()
        except Exception as e:
            return {"error": f"Failed to get provider status: {e}"}
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        try:
            return {
                "total_transcriptions": production_monitor.performance_counters.get("total_transcriptions", 0),
                "total_errors": production_monitor.performance_counters.get("total_errors", 0),
                "uptime_hours": (time.time() - production_monitor.performance_counters.get("start_time", time.time())) / 3600,
                "recent_alerts": len(production_monitor.get_recent_alerts(hours=1))
            }
        except Exception as e:
            return {"error": f"Failed to get performance metrics: {e}"}
    
    def _get_recent_errors(self) -> List[Dict[str, Any]]:
        """Get recent error information."""
        try:
            # This would need to integrate with the error reporter
            return []
        except Exception as e:
            return [{"error": f"Failed to get recent errors: {e}"}]
    
    def export_diagnostics(self, output_file: Path, format: str = "json") -> bool:
        """Export diagnostic information to file."""
        try:
            diagnostics = asyncio.run(self.collect_full_diagnostics())
            
            if format.lower() == "json":
                with open(output_file, 'w') as f:
                    json.dump(diagnostics.to_dict(), f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Diagnostics exported to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export diagnostics: {e}")
            return False


class TroubleshootingGuide:
    """Interactive troubleshooting guide for common issues."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.diagnostics = SystemDiagnostics()
    
    async def diagnose_issue(self, symptoms: List[str]) -> Dict[str, Any]:
        """Diagnose issue based on symptoms and provide solutions."""
        
        # Collect current system state
        diagnostics = await self.diagnostics.collect_full_diagnostics()
        
        # Analyze symptoms and system state
        analysis = self._analyze_symptoms(symptoms, diagnostics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(analysis, diagnostics)
        
        return {
            "symptoms": symptoms,
            "analysis": analysis,
            "recommendations": recommendations,
            "system_health": self._assess_system_health(diagnostics),
            "next_steps": self._suggest_next_steps(analysis)
        }
    
    def _analyze_symptoms(self, symptoms: List[str], diagnostics: DiagnosticInfo) -> Dict[str, Any]:
        """Analyze symptoms against system diagnostics."""
        analysis = {
            "likely_causes": [],
            "severity": "unknown",
            "affected_components": []
        }
        
        symptom_text = " ".join(symptoms).lower()
        
        # API key issues
        if any(keyword in symptom_text for keyword in ["api key", "authentication", "unauthorized"]):
            analysis["likely_causes"].append("API key configuration issue")
            analysis["affected_components"].append("authentication")
            analysis["severity"] = "high"
        
        # Provider issues
        if any(keyword in symptom_text for keyword in ["provider", "connection", "timeout"]):
            analysis["likely_causes"].append("Provider connectivity issue")
            analysis["affected_components"].append("provider")
            
            # Check provider status
            if diagnostics.provider_status:
                unhealthy_providers = [
                    name for name, status in diagnostics.provider_status.items()
                    if status.get("status") != "healthy"
                ]
                if unhealthy_providers:
                    analysis["likely_causes"].append(f"Unhealthy providers: {unhealthy_providers}")
        
        # Performance issues
        if any(keyword in symptom_text for keyword in ["slow", "hanging", "timeout", "performance"]):
            analysis["likely_causes"].append("Performance degradation")
            analysis["affected_components"].append("performance")
            
            # Check system resources
            if diagnostics.performance_metrics:
                if diagnostics.performance_metrics.get("recent_alerts", 0) > 0:
                    analysis["likely_causes"].append("Recent performance alerts detected")
        
        # Configuration issues
        if any(keyword in symptom_text for keyword in ["config", "setting", "format"]):
            analysis["likely_causes"].append("Configuration issue")
            analysis["affected_components"].append("configuration")
        
        return analysis
    
    def _generate_recommendations(self, analysis: Dict[str, Any], diagnostics: DiagnosticInfo) -> List[str]:
        """Generate specific recommendations based on analysis."""
        recommendations = []
        
        for cause in analysis["likely_causes"]:
            if "api key" in cause.lower():
                recommendations.extend([
                    "Check that API keys are set as environment variables",
                    "Verify API key format and validity",
                    "Test API key with provider's authentication endpoint"
                ])
            
            elif "provider" in cause.lower():
                recommendations.extend([
                    "Check internet connectivity",
                    "Verify provider service status",
                    "Try switching to a different provider",
                    "Check for rate limiting or quota issues"
                ])
            
            elif "performance" in cause.lower():
                recommendations.extend([
                    "Monitor system resource usage (CPU, memory, disk)",
                    "Check for background processes consuming resources",
                    "Consider reducing chunk size or concurrent operations",
                    "Verify network bandwidth and latency"
                ])
            
            elif "configuration" in cause.lower():
                recommendations.extend([
                    "Validate configuration using VttiroConfig",
                    "Check for deprecated configuration options",
                    "Verify file paths and permissions",
                    "Review migration guide for v2.0 changes"
                ])
        
        # System-specific recommendations
        if diagnostics.platform_info["system"] == "Windows":
            recommendations.append("On Windows: Check for path length limitations")
        
        if diagnostics.python_info["version"].startswith("3.8"):
            recommendations.append("Consider upgrading to Python 3.10+ for better performance")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _assess_system_health(self, diagnostics: DiagnosticInfo) -> Dict[str, str]:
        """Assess overall system health."""
        health = {
            "overall": "unknown",
            "providers": "unknown",
            "configuration": "unknown",
            "performance": "unknown"
        }
        
        # Provider health
        if diagnostics.provider_status:
            healthy_providers = sum(1 for status in diagnostics.provider_status.values() 
                                  if status.get("status") == "healthy")
            total_providers = len(diagnostics.provider_status)
            
            if healthy_providers == total_providers:
                health["providers"] = "good"
            elif healthy_providers > 0:
                health["providers"] = "partial"
            else:
                health["providers"] = "poor"
        
        # Performance health
        if diagnostics.performance_metrics:
            error_rate = diagnostics.performance_metrics.get("total_errors", 0)
            total_ops = diagnostics.performance_metrics.get("total_transcriptions", 1)
            error_percentage = (error_rate / max(total_ops, 1)) * 100
            
            if error_percentage < 5:
                health["performance"] = "good"
            elif error_percentage < 15:
                health["performance"] = "fair"
            else:
                health["performance"] = "poor"
        
        # Overall health
        component_health = [health["providers"], health["performance"]]
        if all(h == "good" for h in component_health):
            health["overall"] = "good"
        elif any(h == "poor" for h in component_health):
            health["overall"] = "poor"
        else:
            health["overall"] = "fair"
        
        return health
    
    def _suggest_next_steps(self, analysis: Dict[str, Any]) -> List[str]:
        """Suggest concrete next steps for troubleshooting."""
        steps = []
        
        if analysis["severity"] == "high":
            steps.extend([
                "1. Check system logs for detailed error messages",
                "2. Run diagnostic tests for affected components",
                "3. Contact support if issue persists"
            ])
        else:
            steps.extend([
                "1. Try the recommended solutions above",
                "2. Monitor system for 10-15 minutes",
                "3. Run diagnostics again to check improvement"
            ])
        
        steps.append("4. Export diagnostic report for further analysis")
        
        return steps


# Global instances
error_reporter = EnhancedErrorReporter()
system_diagnostics = SystemDiagnostics()
troubleshooting_guide = TroubleshootingGuide()


def capture_error(error: Exception, 
                 context: Optional[Dict[str, Any]] = None,
                 provider_name: Optional[str] = None) -> str:
    """Quick function to capture error with enhanced reporting."""
    error_context = error_reporter.capture_error(error, context, provider_name)
    return error_context.error_id


async def diagnose_system() -> DiagnosticInfo:
    """Quick function to collect system diagnostics."""
    return await system_diagnostics.collect_full_diagnostics()


if __name__ == "__main__":
    # CLI tool for debugging operations
    import argparse
    
    parser = argparse.ArgumentParser(description="VTTiro Debugging Tools")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Diagnostic command
    diag_parser = subparsers.add_parser("diagnose", help="Collect system diagnostics")
    diag_parser.add_argument("--output", "-o", type=str, help="Output file for diagnostics")
    
    # Error report command
    error_parser = subparsers.add_parser("errors", help="Show recent errors")
    error_parser.add_argument("--limit", "-l", type=int, default=10, help="Number of recent errors")
    
    # Troubleshoot command
    trouble_parser = subparsers.add_parser("troubleshoot", help="Interactive troubleshooting")
    trouble_parser.add_argument("symptoms", nargs="+", help="Describe symptoms")
    
    args = parser.parse_args()
    
    if args.command == "diagnose":
        print("🔍 Collecting system diagnostics...")
        diagnostics = asyncio.run(diagnose_system())
        
        if args.output:
            output_file = Path(args.output)
            success = system_diagnostics.export_diagnostics(output_file)
            if success:
                print(f"✅ Diagnostics exported to {output_file}")
            else:
                print("❌ Failed to export diagnostics")
        else:
            print(f"📊 Platform: {diagnostics.platform_info['system']} {diagnostics.platform_info['release']}")
            print(f"🐍 Python: {diagnostics.python_info['version'].split()[0]}")
            print(f"📦 VTTiro: {diagnostics.vttiro_info.get('version', 'unknown')}")
            print(f"🔌 Providers: {len(diagnostics.vttiro_info.get('available_providers', []))}")
    
    elif args.command == "errors":
        print("🚨 Recent errors:")
        errors = error_reporter.get_error_summary(args.limit)
        
        if not errors:
            print("  No recent errors found")
        else:
            for error in errors:
                print(f"  {error['timestamp'][:19]} - {error['type']}: {error['message']}")
                print(f"    Location: {error['location']}")
    
    elif args.command == "troubleshoot":
        print("🔧 Analyzing symptoms...")
        symptoms = args.symptoms
        analysis = asyncio.run(troubleshooting_guide.diagnose_issue(symptoms))
        
        print(f"\n📋 Analysis:")
        print(f"  Likely causes: {', '.join(analysis['analysis']['likely_causes'])}")
        print(f"  Severity: {analysis['analysis']['severity']}")
        
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(analysis['recommendations'][:5], 1):
            print(f"  {i}. {rec}")
        
        print(f"\n🎯 Next steps:")
        for step in analysis['next_steps']:
            print(f"  {step}")
    
    else:
        parser.print_help()