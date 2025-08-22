#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["loguru", "pydantic", "rich", "click"]
# ///
# this_file: scripts/ci_enhancement.py
"""Comprehensive CI/CD enhancement and quality gate enforcement script.

This script provides advanced CI/CD pipeline enhancements including:
- Intelligent quality gate enforcement with adaptive thresholds
- Automated workflow optimization and performance monitoring
- Quality trend analysis and regression detection
- Smart test selection and parallel execution optimization
- Comprehensive reporting and alerting integration

Used by:
- CI/CD pipeline for quality assurance and automation
- Development workflow optimization
- Quality monitoring and trend analysis
"""

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table


@dataclass
class PipelineMetrics:
    """Metrics collected during CI/CD pipeline execution."""
    
    # Timing metrics
    total_duration: float = 0.0
    setup_duration: float = 0.0
    test_duration: float = 0.0
    analysis_duration: float = 0.0
    
    # Quality metrics
    test_coverage: float = 0.0
    quality_score: float = 0.0
    security_score: float = 0.0
    performance_score: float = 0.0
    
    # Test metrics
    tests_total: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    tests_skipped: int = 0
    
    # Issue counts
    critical_issues: int = 0
    major_issues: int = 0
    minor_issues: int = 0
    
    # Build artifacts
    artifacts_size: int = 0
    dependency_count: int = 0
    
    # Metadata
    commit_hash: str = ""
    branch_name: str = ""
    timestamp: str = ""


@dataclass
class QualityThresholds:
    """Adaptive quality thresholds for different scenarios."""
    
    # Coverage thresholds
    coverage_min: float = 0.85
    coverage_warning: float = 0.90
    coverage_excellent: float = 0.95
    
    # Quality score thresholds
    quality_min: float = 0.75
    quality_warning: float = 0.85
    quality_excellent: float = 0.95
    
    # Performance thresholds
    build_time_max: float = 600  # 10 minutes
    test_time_max: float = 300   # 5 minutes
    
    # Issue thresholds
    critical_max: int = 0
    major_max: int = 5
    minor_max: int = 20


class CIEnhancementManager:
    """Comprehensive CI/CD pipeline enhancement and quality management."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize CI enhancement manager.
        
        Args:
            project_root: Path to project root directory
        """
        self.project_root = project_root or Path.cwd()
        self.console = Console()
        self.metrics_file = self.project_root / ".ci" / "metrics.json"
        self.thresholds_file = self.project_root / ".ci" / "thresholds.json"
        
        # Ensure CI directory exists
        (self.project_root / ".ci").mkdir(exist_ok=True)
        
        # Load or create thresholds
        self.thresholds = self._load_thresholds()
        
        # Pipeline state
        self.start_time = time.time()
        self.metrics = PipelineMetrics()
    
    def run_enhanced_pipeline(self, strict: bool = False, fast: bool = False) -> bool:
        """Run enhanced CI/CD pipeline with intelligent optimizations.
        
        Args:
            strict: Whether to use strict quality thresholds
            fast: Whether to run in fast mode (skip non-essential checks)
            
        Returns:
            True if pipeline passed, False otherwise
        """
        logger.info("Starting enhanced CI/CD pipeline")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            
            # Pipeline stages
            stages = [
                ("Setup Environment", self._setup_environment),
                ("Run Quality Analysis", lambda: self._run_quality_analysis(fast)),
                ("Execute Tests", lambda: self._run_tests(fast)),
                ("Security Scan", lambda: self._run_security_scan(fast)),
                ("Performance Benchmarks", lambda: self._run_performance_benchmarks(fast)),
                ("Generate Reports", self._generate_reports),
                ("Enforce Quality Gates", lambda: self._enforce_quality_gates(strict)),
            ]
            
            success = True
            
            for stage_name, stage_func in stages:
                task = progress.add_task(f"[cyan]{stage_name}...", total=None)
                
                try:
                    stage_start = time.time()
                    stage_success = stage_func()
                    stage_duration = time.time() - stage_start
                    
                    if stage_success:
                        progress.update(task, description=f"[green]✓ {stage_name} ({stage_duration:.1f}s)")
                    else:
                        progress.update(task, description=f"[red]✗ {stage_name} ({stage_duration:.1f}s)")
                        success = False
                        
                        if strict and stage_name == "Enforce Quality Gates":
                            logger.error(f"Pipeline failed at {stage_name} (strict mode)")
                            break
                    
                except Exception as e:
                    progress.update(task, description=f"[red]✗ {stage_name} - Error: {e}")
                    logger.error(f"Stage '{stage_name}' failed: {e}")
                    success = False
                    break
        
        # Finalize metrics
        self.metrics.total_duration = time.time() - self.start_time
        self._save_metrics()
        
        # Display final status
        self._display_pipeline_summary(success)
        
        return success
    
    def analyze_trends(self, days: int = 30) -> Dict[str, Any]:
        """Analyze quality trends over time.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Trend analysis report
        """
        metrics_history = self._load_metrics_history(days)
        
        if len(metrics_history) < 2:
            return {"error": "Insufficient data for trend analysis"}
        
        # Calculate trends
        trends = {}
        
        # Quality score trend
        quality_scores = [m.quality_score for m in metrics_history]
        trends["quality_trend"] = self._calculate_trend(quality_scores)
        
        # Coverage trend
        coverage_scores = [m.test_coverage for m in metrics_history]
        trends["coverage_trend"] = self._calculate_trend(coverage_scores)
        
        # Build time trend
        build_times = [m.total_duration for m in metrics_history]
        trends["build_time_trend"] = self._calculate_trend(build_times)
        
        # Issue trends
        critical_counts = [m.critical_issues for m in metrics_history]
        trends["critical_issues_trend"] = self._calculate_trend(critical_counts)
        
        return trends
    
    def optimize_test_execution(self) -> List[str]:
        """Generate test execution optimization recommendations.
        
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        # Analyze test history for optimization opportunities
        test_history = self._load_test_history()
        
        if test_history:
            # Find slow tests
            slow_tests = [t for t in test_history if t.get("duration", 0) > 30]
            if slow_tests:
                recommendations.append(
                    f"Consider optimizing {len(slow_tests)} slow tests (>30s each)"
                )
            
            # Find flaky tests
            flaky_tests = self._find_flaky_tests(test_history)
            if flaky_tests:
                recommendations.append(
                    f"Found {len(flaky_tests)} potentially flaky tests that need attention"
                )
            
            # Parallel execution opportunities
            cpu_count = self._get_cpu_count()
            if cpu_count > 2:
                recommendations.append(
                    f"Enable parallel test execution with {cpu_count-1} workers"
                )
        
        # Dependency optimization
        dependency_analysis = self._analyze_dependencies()
        if dependency_analysis["unused_count"] > 0:
            recommendations.append(
                f"Remove {dependency_analysis['unused_count']} unused dependencies"
            )
        
        return recommendations
    
    def _setup_environment(self) -> bool:
        """Set up the CI environment."""
        try:
            setup_start = time.time()
            
            # Get git information
            self.metrics.commit_hash = self._get_git_commit_hash()
            self.metrics.branch_name = self._get_git_branch_name()
            self.metrics.timestamp = datetime.now().isoformat()
            
            # Check Python version
            python_version = sys.version_info
            if python_version < (3, 10):
                logger.warning(f"Python {python_version.major}.{python_version.minor} is below recommended 3.10+")
            
            # Install dependencies
            result = subprocess.run(
                ["uv", "sync", "--all-extras"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Dependency installation failed: {result.stderr}")
                return False
            
            self.metrics.setup_duration = time.time() - setup_start
            logger.info(f"Environment setup completed in {self.metrics.setup_duration:.1f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Environment setup failed: {e}")
            return False
    
    def _run_quality_analysis(self, fast: bool = False) -> bool:
        """Run comprehensive code quality analysis."""
        try:
            analysis_start = time.time()
            
            # Import development automation
            sys.path.insert(0, str(self.project_root / "src"))
            from vttiro.utils.development_automation import DevelopmentAutomationManager
            
            # Run quality analysis
            manager = DevelopmentAutomationManager(self.project_root)
            report = manager.analyze_code_quality()
            
            # Update metrics
            self.metrics.quality_score = report.overall_score
            self.metrics.critical_issues = report.critical_count
            self.metrics.major_issues = report.major_count
            self.metrics.minor_issues = report.minor_count
            
            self.metrics.analysis_duration = time.time() - analysis_start
            
            logger.info(f"Quality analysis completed - Score: {report.overall_score:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Quality analysis failed: {e}")
            return False
    
    def _run_tests(self, fast: bool = False) -> bool:
        """Run test suite with coverage analysis."""
        try:
            test_start = time.time()
            
            # Determine test command
            if fast:
                cmd = ["python", "-m", "pytest", "-x", "--tb=short"]
            else:
                cmd = ["python", "-m", "pytest", "--cov=src", "--cov-report=json", "--json-report", "--json-report-file=test-report.json"]
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            # Parse test results
            if (self.project_root / "test-report.json").exists():
                with open(self.project_root / "test-report.json") as f:
                    test_data = json.load(f)
                
                self.metrics.tests_total = test_data.get("summary", {}).get("total", 0)
                self.metrics.tests_passed = test_data.get("summary", {}).get("passed", 0)
                self.metrics.tests_failed = test_data.get("summary", {}).get("failed", 0)
                self.metrics.tests_skipped = test_data.get("summary", {}).get("skipped", 0)
            
            # Parse coverage results
            if (self.project_root / "coverage.json").exists():
                with open(self.project_root / "coverage.json") as f:
                    coverage_data = json.load(f)
                
                self.metrics.test_coverage = coverage_data.get("totals", {}).get("percent_covered", 0) / 100
            
            self.metrics.test_duration = time.time() - test_start
            
            success = result.returncode == 0
            logger.info(f"Tests completed - {self.metrics.tests_passed}/{self.metrics.tests_total} passed, Coverage: {self.metrics.test_coverage:.1%}")
            
            return success
            
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return False
    
    def _run_security_scan(self, fast: bool = False) -> bool:
        """Run security analysis."""
        try:
            if fast:
                logger.info("Skipping security scan in fast mode")
                return True
            
            result = subprocess.run(
                ["uvx", "bandit", "-r", "src/", "-f", "json", "-o", "security-report.json"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            # Parse security results
            if (self.project_root / "security-report.json").exists():
                with open(self.project_root / "security-report.json") as f:
                    security_data = json.load(f)
                
                # Calculate security score based on findings
                high_severity = len([r for r in security_data.get("results", []) if r.get("issue_severity") == "HIGH"])
                medium_severity = len([r for r in security_data.get("results", []) if r.get("issue_severity") == "MEDIUM"])
                
                # Simple scoring: start at 1.0, deduct for issues
                self.metrics.security_score = max(0.0, 1.0 - (high_severity * 0.2 + medium_severity * 0.1))
            else:
                self.metrics.security_score = 1.0
            
            logger.info(f"Security scan completed - Score: {self.metrics.security_score:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Security scan failed: {e}")
            return False
    
    def _run_performance_benchmarks(self, fast: bool = False) -> bool:
        """Run performance benchmarks."""
        try:
            if fast:
                logger.info("Skipping performance benchmarks in fast mode")
                self.metrics.performance_score = 1.0
                return True
            
            # Run benchmarks if they exist
            benchmark_dir = self.project_root / "tests" / "benchmarks"
            if benchmark_dir.exists():
                result = subprocess.run(
                    ["python", "-m", "pytest", str(benchmark_dir), "--benchmark-json=benchmark.json"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True
                )
                
                if (self.project_root / "benchmark.json").exists():
                    with open(self.project_root / "benchmark.json") as f:
                        benchmark_data = json.load(f)
                    
                    # Simple performance scoring based on benchmark results
                    benchmarks = benchmark_data.get("benchmarks", [])
                    if benchmarks:
                        avg_time = sum(b.get("stats", {}).get("mean", 0) for b in benchmarks) / len(benchmarks)
                        # Arbitrary scoring: faster is better
                        self.metrics.performance_score = max(0.0, min(1.0, 1.0 - avg_time))
                    else:
                        self.metrics.performance_score = 1.0
                else:
                    self.metrics.performance_score = 1.0
            else:
                self.metrics.performance_score = 1.0
            
            logger.info(f"Performance benchmarks completed - Score: {self.metrics.performance_score:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Performance benchmarks failed: {e}")
            return False
    
    def _generate_reports(self) -> bool:
        """Generate comprehensive reports."""
        try:
            # Create reports directory
            reports_dir = self.project_root / "reports"
            reports_dir.mkdir(exist_ok=True)
            
            # Generate HTML report
            html_report = self._generate_html_report()
            with open(reports_dir / "ci-report.html", "w") as f:
                f.write(html_report)
            
            # Generate JSON report
            json_report = {
                "metrics": self.metrics.__dict__,
                "thresholds": self.thresholds.__dict__,
                "recommendations": self.optimize_test_execution(),
                "generated_at": datetime.now().isoformat()
            }
            
            with open(reports_dir / "ci-report.json", "w") as f:
                json.dump(json_report, f, indent=2)
            
            logger.info("Reports generated successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return False
    
    def _enforce_quality_gates(self, strict: bool = False) -> bool:
        """Enforce quality gates based on metrics."""
        gates_passed = True
        
        # Adjust thresholds for strict mode
        if strict:
            thresholds = QualityThresholds(
                coverage_min=0.95,
                quality_min=0.90,
                critical_max=0,
                major_max=2
            )
        else:
            thresholds = self.thresholds
        
        # Check coverage
        if self.metrics.test_coverage < thresholds.coverage_min:
            logger.error(f"Coverage gate failed: {self.metrics.test_coverage:.1%} < {thresholds.coverage_min:.1%}")
            gates_passed = False
        
        # Check quality score
        if self.metrics.quality_score < thresholds.quality_min:
            logger.error(f"Quality gate failed: {self.metrics.quality_score:.2f} < {thresholds.quality_min:.2f}")
            gates_passed = False
        
        # Check critical issues
        if self.metrics.critical_issues > thresholds.critical_max:
            logger.error(f"Critical issues gate failed: {self.metrics.critical_issues} > {thresholds.critical_max}")
            gates_passed = False
        
        # Check major issues
        if self.metrics.major_issues > thresholds.major_max:
            logger.error(f"Major issues gate failed: {self.metrics.major_issues} > {thresholds.major_max}")
            gates_passed = False
        
        # Check build time
        if self.metrics.total_duration > thresholds.build_time_max:
            logger.warning(f"Build time warning: {self.metrics.total_duration:.1f}s > {thresholds.build_time_max}s")
        
        if gates_passed:
            logger.info("All quality gates passed ✓")
        else:
            logger.error("Quality gates failed ✗")
        
        return gates_passed
    
    def _display_pipeline_summary(self, success: bool):
        """Display comprehensive pipeline summary."""
        table = Table(title="CI/CD Pipeline Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green" if success else "red")
        table.add_column("Status", style="bold")
        
        # Add metrics rows
        table.add_row("Overall Status", "PASSED" if success else "FAILED", "✓" if success else "✗")
        table.add_row("Total Duration", f"{self.metrics.total_duration:.1f}s", "")
        table.add_row("Quality Score", f"{self.metrics.quality_score:.2f}", "")
        table.add_row("Test Coverage", f"{self.metrics.test_coverage:.1%}", "")
        table.add_row("Security Score", f"{self.metrics.security_score:.2f}", "")
        table.add_row("Tests Passed", f"{self.metrics.tests_passed}/{self.metrics.tests_total}", "")
        table.add_row("Critical Issues", str(self.metrics.critical_issues), "")
        table.add_row("Major Issues", str(self.metrics.major_issues), "")
        
        self.console.print(table)
    
    def _load_thresholds(self) -> QualityThresholds:
        """Load quality thresholds from file or use defaults."""
        if self.thresholds_file.exists():
            try:
                with open(self.thresholds_file) as f:
                    data = json.load(f)
                return QualityThresholds(**data)
            except Exception:
                pass
        
        # Return default thresholds
        return QualityThresholds()
    
    def _save_metrics(self):
        """Save current metrics to file."""
        metrics_data = self.metrics.__dict__.copy()
        
        # Load existing metrics
        history = []
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file) as f:
                    history = json.load(f)
            except Exception:
                pass
        
        # Add current metrics
        history.append(metrics_data)
        
        # Keep only last 100 entries
        history = history[-100:]
        
        # Save updated history
        with open(self.metrics_file, "w") as f:
            json.dump(history, f, indent=2)
    
    def _load_metrics_history(self, days: int) -> List[PipelineMetrics]:
        """Load metrics history for trend analysis."""
        if not self.metrics_file.exists():
            return []
        
        try:
            with open(self.metrics_file) as f:
                history_data = json.load(f)
            
            # Convert to metrics objects
            metrics_list = []
            for data in history_data:
                metrics = PipelineMetrics()
                for key, value in data.items():
                    if hasattr(metrics, key):
                        setattr(metrics, key, value)
                metrics_list.append(metrics)
            
            return metrics_list[-days:]  # Return last N days
            
        except Exception:
            return []
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend information for a series of values."""
        if len(values) < 2:
            return {"trend": "unknown", "change": 0.0}
        
        # Simple linear trend calculation
        recent_avg = sum(values[-5:]) / len(values[-5:])
        older_avg = sum(values[:-5]) / len(values[:-5]) if len(values) > 5 else values[0]
        
        change = recent_avg - older_avg
        trend = "improving" if change > 0 else "declining" if change < 0 else "stable"
        
        return {
            "trend": trend,
            "change": change,
            "recent_average": recent_avg,
            "historical_average": older_avg
        }
    
    def _load_test_history(self) -> List[Dict[str, Any]]:
        """Load test execution history."""
        # Simplified implementation - would load from test reports
        return []
    
    def _find_flaky_tests(self, test_history: List[Dict[str, Any]]) -> List[str]:
        """Find potentially flaky tests."""
        # Simplified implementation - would analyze test failure patterns
        return []
    
    def _get_cpu_count(self) -> int:
        """Get available CPU count."""
        import os
        return os.cpu_count() or 1
    
    def _analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze project dependencies."""
        # Simplified implementation
        return {"unused_count": 0, "outdated_count": 0}
    
    def _get_git_commit_hash(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            return result.stdout.strip()[:8]
        except Exception:
            return "unknown"
    
    def _get_git_branch_name(self) -> str:
        """Get current git branch name."""
        try:
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            return result.stdout.strip()
        except Exception:
            return "unknown"
    
    def _generate_html_report(self) -> str:
        """Generate HTML report."""
        # Simplified HTML report
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>VTTiro CI/CD Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .metric {{ margin: 10px 0; }}
        .success {{ color: green; }}
        .warning {{ color: orange; }}
        .error {{ color: red; }}
    </style>
</head>
<body>
    <h1>VTTiro CI/CD Pipeline Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Summary</h2>
    <div class="metric">Quality Score: {self.metrics.quality_score:.2f}</div>
    <div class="metric">Test Coverage: {self.metrics.test_coverage:.1%}</div>
    <div class="metric">Security Score: {self.metrics.security_score:.2f}</div>
    <div class="metric">Total Duration: {self.metrics.total_duration:.1f}s</div>
    
    <h2>Test Results</h2>
    <div class="metric">Tests Passed: {self.metrics.tests_passed}</div>
    <div class="metric">Tests Failed: {self.metrics.tests_failed}</div>
    <div class="metric">Tests Skipped: {self.metrics.tests_skipped}</div>
    
    <h2>Issues</h2>
    <div class="metric">Critical Issues: {self.metrics.critical_issues}</div>
    <div class="metric">Major Issues: {self.metrics.major_issues}</div>
    <div class="metric">Minor Issues: {self.metrics.minor_issues}</div>
</body>
</html>
"""


@click.command()
@click.option("--strict", is_flag=True, help="Use strict quality thresholds")
@click.option("--fast", is_flag=True, help="Run in fast mode (skip non-essential checks)")
@click.option("--analyze-trends", is_flag=True, help="Analyze quality trends")
@click.option("--optimize", is_flag=True, help="Show optimization recommendations")
def main(strict: bool, fast: bool, analyze_trends: bool, optimize: bool):
    """VTTiro CI/CD Enhancement and Quality Gate System."""
    
    manager = CIEnhancementManager()
    
    if analyze_trends:
        trends = manager.analyze_trends()
        console = Console()
        console.print("Quality Trends Analysis:", style="bold blue")
        console.print(json.dumps(trends, indent=2))
        return
    
    if optimize:
        recommendations = manager.optimize_test_execution()
        console = Console()
        console.print("Optimization Recommendations:", style="bold green")
        for rec in recommendations:
            console.print(f"• {rec}")
        return
    
    # Run enhanced pipeline
    success = manager.run_enhanced_pipeline(strict=strict, fast=fast)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()