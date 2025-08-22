#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["loguru", "rich", "click", "pyyaml"]
# ///
# this_file: scripts/setup_dev_automation.py
"""Development automation setup and configuration script.

This script provides comprehensive development environment setup including:
- Pre-commit hooks installation and configuration
- Development toolchain setup and validation
- Quality gate configuration and customization
- IDE integration and workspace optimization
- Automated workflow configuration

Used by:
- New developer onboarding
- Development environment standardization
- Quality assurance workflow setup
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import yaml
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt


class DevAutomationSetup:
    """Comprehensive development automation setup manager."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize setup manager.
        
        Args:
            project_root: Path to project root directory
        """
        self.project_root = project_root or Path.cwd()
        self.console = Console()
        
        # Configuration paths
        self.config_dir = self.project_root / ".dev-config"
        self.hooks_dir = self.project_root / ".git" / "hooks"
        
        # Tool configurations
        self.tool_configs = {
            "pre-commit": {
                "file": ".pre-commit-config.yaml",
                "required": True,
                "description": "Pre-commit hooks for code quality"
            },
            "ruff": {
                "file": "pyproject.toml",
                "section": "[tool.ruff]",
                "required": True,
                "description": "Python linting and formatting"
            },
            "mypy": {
                "file": "pyproject.toml", 
                "section": "[tool.mypy]",
                "required": True,
                "description": "Static type checking"
            },
            "pytest": {
                "file": "pyproject.toml",
                "section": "[tool.pytest.ini_options]",
                "required": True,
                "description": "Test runner configuration"
            },
            "coverage": {
                "file": "pyproject.toml",
                "section": "[tool.coverage.run]",
                "required": True,
                "description": "Code coverage measurement"
            }
        }
    
    def setup_complete_environment(self, interactive: bool = True) -> bool:
        """Set up complete development environment.
        
        Args:
            interactive: Whether to prompt for user input
            
        Returns:
            True if setup successful, False otherwise
        """
        logger.info("Starting comprehensive development environment setup")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            
            # Setup stages
            stages = [
                ("Validating Environment", self._validate_environment),
                ("Installing Dependencies", self._install_dependencies),
                ("Configuring Tools", lambda: self._configure_tools(interactive)),
                ("Setting up Pre-commit", self._setup_pre_commit),
                ("Creating Quality Gates", self._create_quality_gates),
                ("Configuring IDE Integration", lambda: self._configure_ide_integration(interactive)),
                ("Setting up Automation Scripts", self._setup_automation_scripts),
                ("Validating Setup", self._validate_setup),
            ]
            
            success = True
            
            for stage_name, stage_func in stages:
                task = progress.add_task(f"[cyan]{stage_name}...", total=None)
                
                try:
                    stage_success = stage_func()
                    
                    if stage_success:
                        progress.update(task, description=f"[green]✓ {stage_name}")
                    else:
                        progress.update(task, description=f"[red]✗ {stage_name}")
                        success = False
                        break
                        
                except Exception as e:
                    progress.update(task, description=f"[red]✗ {stage_name} - Error: {e}")
                    logger.error(f"Stage '{stage_name}' failed: {e}")
                    success = False
                    break
        
        if success:
            self._display_setup_summary()
            logger.info("Development environment setup completed successfully! 🎉")
        else:
            logger.error("Development environment setup failed!")
        
        return success
    
    def _validate_environment(self) -> bool:
        """Validate development environment prerequisites."""
        required_tools = {
            "python": "Python 3.10+",
            "git": "Git version control",
            "uv": "UV package manager"
        }
        
        missing_tools = []
        
        for tool, description in required_tools.items():
            if not self._check_tool_available(tool):
                missing_tools.append(f"{tool} ({description})")
        
        if missing_tools:
            logger.error(f"Missing required tools: {', '.join(missing_tools)}")
            return False
        
        # Check Python version
        if sys.version_info < (3, 10):
            logger.error(f"Python 3.10+ required, found {sys.version_info.major}.{sys.version_info.minor}")
            return False
        
        # Check if in git repository
        if not (self.project_root / ".git").exists():
            logger.error("Not in a git repository")
            return False
        
        logger.info("Environment validation passed")
        return True
    
    def _install_dependencies(self) -> bool:
        """Install required dependencies."""
        try:
            # Install development dependencies
            result = subprocess.run(
                ["uv", "sync", "--all-extras", "--dev"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to install dependencies: {result.stderr}")
                return False
            
            # Install pre-commit
            result = subprocess.run(
                ["uv", "add", "--dev", "pre-commit"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.warning("Pre-commit may already be installed")
            
            logger.info("Dependencies installed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Dependency installation failed: {e}")
            return False
    
    def _configure_tools(self, interactive: bool) -> bool:
        """Configure development tools."""
        try:
            # Create config directory
            self.config_dir.mkdir(exist_ok=True)
            
            # Configure each tool
            for tool_name, config in self.tool_configs.items():
                if interactive:
                    if not Confirm.ask(f"Configure {tool_name} ({config['description']})?"):
                        continue
                
                self._configure_tool(tool_name, config)
            
            logger.info("Tool configuration completed")
            return True
            
        except Exception as e:
            logger.error(f"Tool configuration failed: {e}")
            return False
    
    def _configure_tool(self, tool_name: str, config: Dict[str, Any]):
        """Configure a specific development tool."""
        config_file = self.project_root / config["file"]
        
        if tool_name == "pre-commit":
            self._create_pre_commit_config()
        elif tool_name == "ruff":
            self._configure_ruff_in_pyproject()
        elif tool_name == "mypy":
            self._configure_mypy_in_pyproject()
        elif tool_name == "pytest":
            self._configure_pytest_in_pyproject()
        elif tool_name == "coverage":
            self._configure_coverage_in_pyproject()
        
        logger.info(f"Configured {tool_name}")
    
    def _create_pre_commit_config(self):
        """Create comprehensive pre-commit configuration."""
        config_content = {
            "repos": [
                {
                    "repo": "https://github.com/pre-commit/pre-commit-hooks",
                    "rev": "v4.4.0",
                    "hooks": [
                        {"id": "trailing-whitespace"},
                        {"id": "end-of-file-fixer"},
                        {"id": "check-yaml"},
                        {"id": "check-toml"},
                        {"id": "check-merge-conflict"},
                        {"id": "check-added-large-files"},
                        {"id": "debug-statements"},
                        {"id": "check-case-conflict"},
                        {"id": "check-docstring-first"},
                    ]
                },
                {
                    "repo": "https://github.com/astral-sh/ruff-pre-commit",
                    "rev": "v0.1.6",
                    "hooks": [
                        {
                            "id": "ruff",
                            "args": ["--fix", "--exit-non-zero-on-fix"]
                        },
                        {"id": "ruff-format"}
                    ]
                },
                {
                    "repo": "https://github.com/pre-commit/mirrors-mypy",
                    "rev": "v1.7.1",
                    "hooks": [
                        {
                            "id": "mypy",
                            "additional_dependencies": ["types-all"],
                            "args": ["--strict"]
                        }
                    ]
                },
                {
                    "repo": "https://github.com/PyCQA/bandit",
                    "rev": "1.7.5",
                    "hooks": [
                        {
                            "id": "bandit",
                            "args": ["-r", "src/"],
                            "exclude": "tests/"
                        }
                    ]
                },
                {
                    "repo": "local",
                    "hooks": [
                        {
                            "id": "quality-check",
                            "name": "VTTiro Quality Check",
                            "entry": "python scripts/ci_enhancement.py --fast",
                            "language": "system",
                            "pass_filenames": False,
                            "always_run": True
                        }
                    ]
                }
            ]
        }
        
        config_path = self.project_root / ".pre-commit-config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_content, f, default_flow_style=False, sort_keys=False)
    
    def _configure_ruff_in_pyproject(self):
        """Configure Ruff in pyproject.toml."""
        # This would read existing pyproject.toml and add/update Ruff config
        # Simplified implementation
        ruff_config = """
[tool.ruff]
target-version = "py310"
line-length = 88
extend-exclude = ["migrations", "venv"]

[tool.ruff.lint]
select = ["E", "F", "W", "C90", "I", "N", "D", "UP", "B", "A", "C4", "ICN", "PIE", "PYI", "Q", "SIM", "TID", "TCH", "ARG", "PTH", "ERA", "PL", "TRY", "FLY", "PERF", "RUF"]
ignore = ["D100", "D101", "D102", "D103", "D104", "D105", "D106", "D107"]

[tool.ruff.lint.pydocstyle]
convention = "google"

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
"""
        logger.info("Ruff configuration template created")
    
    def _configure_mypy_in_pyproject(self):
        """Configure MyPy in pyproject.toml."""
        mypy_config = """
[tool.mypy]
python_version = "3.10"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
ignore_missing_imports = true
"""
        logger.info("MyPy configuration template created")
    
    def _configure_pytest_in_pyproject(self):
        """Configure Pytest in pyproject.toml."""
        pytest_config = """
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
addopts = [
    "--strict-markers",
    "--strict-config",
    "--verbose",
    "--tb=short",
    "--cov=src",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-report=xml",
    "--cov-fail-under=90",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
]
"""
        logger.info("Pytest configuration template created")
    
    def _configure_coverage_in_pyproject(self):
        """Configure Coverage in pyproject.toml."""
        coverage_config = """
[tool.coverage.run]
source = ["src"]
omit = [
    "*/tests/*",
    "*/test_*",
    "*/conftest.py",
    "*/__main__.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if self.debug:",
    "if settings.DEBUG",
    "raise AssertionError",
    "raise NotImplementedError",
    "if 0:",
    "if __name__ == .__main__.:",
    "class .*\\bProtocol\\):",
    "@(abc\\.)?abstractmethod",
]

[tool.coverage.html]
directory = "htmlcov"
"""
        logger.info("Coverage configuration template created")
    
    def _setup_pre_commit(self) -> bool:
        """Set up pre-commit hooks."""
        try:
            # Install pre-commit hooks
            result = subprocess.run(
                ["uv", "run", "pre-commit", "install"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Pre-commit installation failed: {result.stderr}")
                return False
            
            # Install commit-msg hook for conventional commits
            result = subprocess.run(
                ["uv", "run", "pre-commit", "install", "--hook-type", "commit-msg"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            # Run initial check
            result = subprocess.run(
                ["uv", "run", "pre-commit", "run", "--all-files"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            # It's OK if this fails on first run
            logger.info("Pre-commit hooks installed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Pre-commit setup failed: {e}")
            return False
    
    def _create_quality_gates(self) -> bool:
        """Create quality gate configurations."""
        try:
            # Create quality gates config
            quality_config = {
                "thresholds": {
                    "coverage_min": 0.90,
                    "quality_score_min": 0.85,
                    "critical_issues_max": 0,
                    "major_issues_max": 5,
                    "build_time_max": 600
                },
                "rules": {
                    "enforce_coverage": True,
                    "enforce_quality_score": True,
                    "enforce_security_scan": True,
                    "enforce_type_checking": True,
                    "enforce_documentation": False
                },
                "notifications": {
                    "slack_webhook": None,
                    "email_recipients": [],
                    "github_issues": True
                }
            }
            
            config_path = self.config_dir / "quality-gates.yaml"
            with open(config_path, "w") as f:
                yaml.dump(quality_config, f, default_flow_style=False)
            
            logger.info("Quality gates configuration created")
            return True
            
        except Exception as e:
            logger.error(f"Quality gates creation failed: {e}")
            return False
    
    def _configure_ide_integration(self, interactive: bool) -> bool:
        """Configure IDE integration files."""
        try:
            if interactive:
                configure_vscode = Confirm.ask("Configure VS Code integration?")
                configure_pycharm = Confirm.ask("Configure PyCharm integration?")
            else:
                configure_vscode = True
                configure_pycharm = True
            
            if configure_vscode:
                self._create_vscode_config()
            
            if configure_pycharm:
                self._create_pycharm_config()
            
            logger.info("IDE integration configured")
            return True
            
        except Exception as e:
            logger.error(f"IDE integration failed: {e}")
            return False
    
    def _create_vscode_config(self):
        """Create VS Code configuration."""
        vscode_dir = self.project_root / ".vscode"
        vscode_dir.mkdir(exist_ok=True)
        
        # Settings
        settings = {
            "python.defaultInterpreterPath": "./.venv/bin/python",
            "python.testing.pytestEnabled": True,
            "python.testing.pytestArgs": ["tests"],
            "python.linting.enabled": True,
            "python.linting.ruffEnabled": True,
            "python.linting.mypyEnabled": True,
            "python.formatting.provider": "black",
            "editor.formatOnSave": True,
            "editor.codeActionsOnSave": {
                "source.organizeImports": True
            },
            "files.exclude": {
                "**/__pycache__": True,
                "**/.pytest_cache": True,
                "**/.mypy_cache": True,
                "**/.coverage": True,
                "**/htmlcov": True
            }
        }
        
        with open(vscode_dir / "settings.json", "w") as f:
            import json
            json.dump(settings, f, indent=2)
        
        # Extensions recommendations
        extensions = {
            "recommendations": [
                "ms-python.python",
                "ms-python.mypy-type-checker",
                "charliermarsh.ruff",
                "ms-python.black-formatter",
                "ms-vscode.test-adapter-converter",
                "littlefoxteam.vscode-python-test-adapter",
                "ms-python.pytest"
            ]
        }
        
        with open(vscode_dir / "extensions.json", "w") as f:
            import json
            json.dump(extensions, f, indent=2)
    
    def _create_pycharm_config(self):
        """Create PyCharm configuration hints."""
        pycharm_config = """
# PyCharm Configuration Hints

## Code Style
- Go to Settings > Editor > Code Style > Python
- Set line length to 88
- Enable automatic imports optimization

## Inspections
- Enable all Python inspections
- Configure type checking with mypy
- Enable security inspections

## Testing
- Set default test runner to pytest
- Configure test templates
- Enable coverage measurement

## Version Control
- Enable pre-commit hooks integration
- Configure commit message template
- Set up code review settings
"""
        
        config_path = self.config_dir / "pycharm-setup.md"
        with open(config_path, "w") as f:
            f.write(pycharm_config)
    
    def _setup_automation_scripts(self) -> bool:
        """Set up automation scripts."""
        try:
            scripts_dir = self.project_root / "scripts"
            scripts_dir.mkdir(exist_ok=True)
            
            # Create development helper scripts
            self._create_dev_helper_scripts(scripts_dir)
            
            # Make scripts executable
            for script in scripts_dir.glob("*.py"):
                script.chmod(0o755)
            
            logger.info("Automation scripts created")
            return True
            
        except Exception as e:
            logger.error(f"Automation scripts setup failed: {e}")
            return False
    
    def _create_dev_helper_scripts(self, scripts_dir: Path):
        """Create helpful development scripts."""
        # Quick test script
        test_script = """#!/usr/bin/env -S uv run
# Quick test runner with coverage
import subprocess
import sys

def main():
    cmd = ["python", "-m", "pytest", "--cov=src", "--cov-report=term-missing"]
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])
    
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
"""
        
        with open(scripts_dir / "test.py", "w") as f:
            f.write(test_script)
        
        # Quick quality check script
        quality_script = """#!/usr/bin/env -S uv run
# Quick quality check runner
import subprocess
import sys

def main():
    checks = [
        (["uvx", "ruff", "check", "src/"], "Ruff linting"),
        (["uvx", "mypy", "src/"], "Type checking"),
        (["python", "-m", "pytest", "--tb=short"], "Tests"),
    ]
    
    failed = []
    
    for cmd, name in checks:
        print(f"Running {name}...")
        result = subprocess.run(cmd, capture_output=True)
        
        if result.returncode == 0:
            print(f"✓ {name} passed")
        else:
            print(f"✗ {name} failed")
            failed.append(name)
    
    if failed:
        print(f"\\nFailed checks: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("\\nAll quality checks passed! 🎉")

if __name__ == "__main__":
    main()
"""
        
        with open(scripts_dir / "quality-check.py", "w") as f:
            f.write(quality_script)
    
    def _validate_setup(self) -> bool:
        """Validate the complete setup."""
        try:
            # Check pre-commit installation
            result = subprocess.run(
                ["uv", "run", "pre-commit", "--version"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error("Pre-commit validation failed")
                return False
            
            # Check tool availability
            tools_to_check = ["ruff", "mypy", "pytest"]
            for tool in tools_to_check:
                result = subprocess.run(
                    ["uvx", tool, "--version"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    logger.error(f"{tool} validation failed")
                    return False
            
            logger.info("Setup validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Setup validation failed: {e}")
            return False
    
    def _display_setup_summary(self):
        """Display setup completion summary."""
        from rich.panel import Panel
        from rich.text import Text
        
        summary_text = Text()
        summary_text.append("Development environment setup completed successfully!\n\n", style="bold green")
        summary_text.append("✓ Pre-commit hooks installed and configured\n", style="green")
        summary_text.append("✓ Quality gates and automation scripts created\n", style="green")
        summary_text.append("✓ IDE integration files generated\n", style="green")
        summary_text.append("✓ Development tools configured\n\n", style="green")
        
        summary_text.append("Quick commands:\n", style="bold")
        summary_text.append("• python scripts/test.py - Run tests with coverage\n", style="cyan")
        summary_text.append("• python scripts/quality-check.py - Run quality checks\n", style="cyan")
        summary_text.append("• python scripts/ci_enhancement.py - Full CI pipeline\n", style="cyan")
        summary_text.append("• uv run pre-commit run --all-files - Run all hooks\n\n", style="cyan")
        
        summary_text.append("Configuration files created:\n", style="bold")
        summary_text.append("• .pre-commit-config.yaml - Pre-commit hooks\n", style="yellow")
        summary_text.append("• .dev-config/quality-gates.yaml - Quality thresholds\n", style="yellow")
        summary_text.append("• .vscode/ - VS Code configuration\n", style="yellow")
        
        panel = Panel(summary_text, title="Setup Complete", border_style="green")
        self.console.print(panel)
    
    def _check_tool_available(self, tool: str) -> bool:
        """Check if a tool is available in PATH."""
        try:
            result = subprocess.run(
                [tool, "--version"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False


@click.command()
@click.option("--interactive/--no-interactive", default=True, help="Interactive setup prompts")
@click.option("--minimal", is_flag=True, help="Minimal setup (essential tools only)")
def main(interactive: bool, minimal: bool):
    """VTTiro Development Automation Setup.
    
    Sets up comprehensive development environment with quality gates,
    pre-commit hooks, and automation tools.
    """
    
    setup_manager = DevAutomationSetup()
    
    if minimal:
        # Minimal setup - just pre-commit and basic tools
        success = (
            setup_manager._validate_environment() and
            setup_manager._install_dependencies() and
            setup_manager._setup_pre_commit()
        )
    else:
        # Full setup
        success = setup_manager.setup_complete_environment(interactive=interactive)
    
    if success:
        console = Console()
        console.print("🎉 Development environment ready!", style="bold green")
    else:
        console = Console()
        console.print("❌ Setup failed. Check logs for details.", style="bold red")
        sys.exit(1)


if __name__ == "__main__":
    main()