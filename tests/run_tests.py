#!/usr/bin/env python3
# this_file: tests/run_tests.py
"""Test runner script for vttiro package with various test categories."""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


class TestRunner:
    """Test runner for different test categories and configurations."""

    def __init__(self, root_dir: Path | None = None):
        """Initialize test runner.

        Args:
            root_dir: Root directory of the project (defaults to parent of tests dir)
        """
        if root_dir is None:
            root_dir = Path(__file__).parent.parent
        self.root_dir = root_dir
        self.tests_dir = root_dir / "tests"

    def run_command(self, cmd: list[str], description: str = "") -> int:
        """Run a command and return exit code.

        Args:
            cmd: Command to run as list of strings
            description: Description of what the command does

        Returns:
            Exit code from the command
        """
        if description:
            print(f"\n=== {description} ===")

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False, cwd=self.root_dir)
        return result.returncode

    def unit_tests(self, coverage: bool = True, parallel: bool = True) -> int:
        """Run unit tests.

        Args:
            coverage: Whether to generate coverage report
            parallel: Whether to run tests in parallel

        Returns:
            Exit code
        """
        cmd = ["python", "-m", "pytest"]

        if parallel:
            cmd.extend(["-n", "auto"])

        if coverage:
            cmd.extend(
                ["--cov=src/vttiro", "--cov-report=term-missing", "--cov-report=html:htmlcov", "--cov-fail-under=85"]
            )

        cmd.extend(["-m", "unit", "-v", "tests/"])

        return self.run_command(cmd, "Running Unit Tests")

    def integration_tests(self) -> int:
        """Run integration tests.

        Returns:
            Exit code
        """
        cmd = ["python", "-m", "pytest", "-m", "integration", "-v", "--tb=short", "tests/"]

        return self.run_command(cmd, "Running Integration Tests")

    def performance_tests(self) -> int:
        """Run performance and benchmark tests.

        Returns:
            Exit code
        """
        cmd = [
            "python",
            "-m",
            "pytest",
            "-m",
            "benchmark or performance",
            "-v",
            "--benchmark-only",
            "--benchmark-sort=mean",
            "tests/",
        ]

        return self.run_command(cmd, "Running Performance Tests")

    def property_tests(self, max_examples: int = 100) -> int:
        """Run property-based tests with Hypothesis.

        Args:
            max_examples: Maximum number of examples to generate

        Returns:
            Exit code
        """
        cmd = [
            "python",
            "-m",
            "pytest",
            "-m",
            "property",
            "-v",
            f"--hypothesis-max-examples={max_examples}",
            "--hypothesis-show-statistics",
            "tests/",
        ]

        return self.run_command(cmd, "Running Property-Based Tests")

    def error_handling_tests(self) -> int:
        """Run error handling and resilience tests.

        Returns:
            Exit code
        """
        cmd = [
            "python",
            "-m",
            "pytest",
            "-v",
            "--tb=long",
            "tests/test_error_handling.py",
            "tests/test_transcriber_integration.py",
        ]

        return self.run_command(cmd, "Running Error Handling Tests")

    def slow_tests(self) -> int:
        """Run slow tests (network, API, large file processing).

        Returns:
            Exit code
        """
        cmd = [
            "python",
            "-m",
            "pytest",
            "-m",
            "slow",
            "-v",
            "--timeout=300",  # 5 minute timeout
            "tests/",
        ]

        return self.run_command(cmd, "Running Slow Tests")

    def api_tests(self) -> int:
        """Run tests requiring API keys.

        Returns:
            Exit code
        """
        cmd = ["python", "-m", "pytest", "-m", "api", "-v", "--tb=short", "tests/"]

        return self.run_command(cmd, "Running API Tests")

    def quick_tests(self) -> int:
        """Run quick test suite (unit tests only, no slow tests).

        Returns:
            Exit code
        """
        cmd = [
            "python",
            "-m",
            "pytest",
            "-m",
            "not slow and not api and not network",
            "-n",
            "auto",
            "--cov=src/vttiro",
            "--cov-report=term-missing",
            "-v",
            "tests/",
        ]

        return self.run_command(cmd, "Running Quick Test Suite")

    def full_test_suite(self) -> int:
        """Run complete test suite with all categories.

        Returns:
            Exit code
        """
        cmd = [
            "python",
            "-m",
            "pytest",
            "-n",
            "auto",
            "--cov=src/vttiro",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-report=xml:coverage.xml",
            "--cov-fail-under=85",
            "--benchmark-skip",  # Skip benchmarks in full suite
            "-v",
            "tests/",
        ]

        return self.run_command(cmd, "Running Full Test Suite")

    def regression_tests(self) -> int:
        """Run regression tests to ensure no performance degradation.

        Returns:
            Exit code
        """
        cmd = [
            "python",
            "-m",
            "pytest",
            "-m",
            "benchmark",
            "--benchmark-only",
            "--benchmark-compare=benchmark/baseline.json",
            "--benchmark-compare-fail=mean:10%",  # Fail if 10% slower
            "tests/",
        ]

        return self.run_command(cmd, "Running Regression Tests")

    def security_tests(self) -> int:
        """Run security-focused tests.

        Returns:
            Exit code
        """
        # Run bandit security scanner
        bandit_cmd = ["bandit", "-r", "src/vttiro", "-f", "json", "-o", "security_report.json"]

        bandit_result = self.run_command(bandit_cmd, "Running Security Scan")

        # Run safety check for vulnerable dependencies
        safety_cmd = ["safety", "check", "--json"]
        safety_result = self.run_command(safety_cmd, "Checking Dependencies for Vulnerabilities")

        return max(bandit_result, safety_result)

    def type_check(self) -> int:
        """Run type checking with mypy.

        Returns:
            Exit code
        """
        cmd = ["mypy", "--install-types", "--non-interactive", "src/vttiro", "tests"]

        return self.run_command(cmd, "Running Type Checking")

    def lint_check(self) -> int:
        """Run code quality checks with ruff.

        Returns:
            Exit code
        """
        # Check code style
        check_cmd = ["ruff", "check", "src/vttiro", "tests"]
        check_result = self.run_command(check_cmd, "Checking Code Style")

        # Check formatting
        format_cmd = ["ruff", "format", "--check", "src/vttiro", "tests"]
        format_result = self.run_command(format_cmd, "Checking Code Formatting")

        return max(check_result, format_result)

    def pre_commit_tests(self) -> int:
        """Run pre-commit test suite (quick tests + linting + type check).

        Returns:
            Exit code
        """
        results = []

        # Run quick tests
        results.append(self.quick_tests())

        # Run linting
        results.append(self.lint_check())

        # Run type checking
        results.append(self.type_check())

        return max(results) if results else 0

    def ci_tests(self) -> int:
        """Run CI test suite optimized for continuous integration.

        Returns:
            Exit code
        """
        cmd = [
            "python",
            "-m",
            "pytest",
            "-n",
            "auto",
            "--cov=src/vttiro",
            "--cov-report=xml:coverage.xml",
            "--cov-fail-under=85",
            "--tb=short",
            "--maxfail=5",  # Stop after 5 failures
            "-m",
            "not slow and not api",  # Skip slow and API tests in CI
            "tests/",
        ]

        return self.run_command(cmd, "Running CI Test Suite")


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description="vttiro test runner")
    parser.add_argument(
        "test_type",
        choices=[
            "unit",
            "integration",
            "performance",
            "property",
            "error",
            "slow",
            "api",
            "quick",
            "full",
            "regression",
            "security",
            "typecheck",
            "lint",
            "precommit",
            "ci",
        ],
        help="Type of tests to run",
    )
    parser.add_argument("--no-coverage", action="store_true", help="Disable coverage reporting")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel test execution")
    parser.add_argument("--max-examples", type=int, default=100, help="Maximum examples for property-based tests")

    args = parser.parse_args()

    runner = TestRunner()

    # Map test types to methods
    test_methods = {
        "unit": lambda: runner.unit_tests(coverage=not args.no_coverage, parallel=not args.no_parallel),
        "integration": runner.integration_tests,
        "performance": runner.performance_tests,
        "property": lambda: runner.property_tests(args.max_examples),
        "error": runner.error_handling_tests,
        "slow": runner.slow_tests,
        "api": runner.api_tests,
        "quick": runner.quick_tests,
        "full": runner.full_test_suite,
        "regression": runner.regression_tests,
        "security": runner.security_tests,
        "typecheck": runner.type_check,
        "lint": runner.lint_check,
        "precommit": runner.pre_commit_tests,
        "ci": runner.ci_tests,
    }

    if args.test_type in test_methods:
        exit_code = test_methods[args.test_type]()
        sys.exit(exit_code)
    else:
        print(f"Unknown test type: {args.test_type}")
        sys.exit(1)


if __name__ == "__main__":
    main()
