# this_file: src/vttiro/tests/test_golden_file_versioning.py

"""
Golden file versioning and change detection system.

Provides version control tracking for golden files to detect unexpected changes,
validate intentional updates, and maintain baseline consistency across test runs.
Integrates with git to track golden file modifications and provides automated
change detection with approval workflows.
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pytest

from vttiro.core.types import TranscriptionResult, TranscriptSegment


class GoldenFileVersion:
    """Represents a versioned golden file with metadata."""
    
    def __init__(self, file_path: Path, content_hash: str, timestamp: str, git_commit: str = ""):
        self.file_path = file_path
        self.content_hash = content_hash
        self.timestamp = timestamp
        self.git_commit = git_commit
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "file_path": str(self.file_path),
            "content_hash": self.content_hash,
            "timestamp": self.timestamp,
            "git_commit": self.git_commit
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "GoldenFileVersion":
        """Create from dictionary."""
        return cls(
            file_path=Path(data["file_path"]),
            content_hash=data["content_hash"],
            timestamp=data["timestamp"],
            git_commit=data.get("git_commit", "")
        )


class GoldenFileTracker:
    """Tracks golden file versions and detects changes."""
    
    def __init__(self, fixtures_dir: Path):
        self.fixtures_dir = fixtures_dir
        self.versions_file = fixtures_dir / ".golden_versions.json"
        self.versions: Dict[str, GoldenFileVersion] = {}
        self._load_versions()
    
    def _load_versions(self):
        """Load existing version information."""
        if self.versions_file.exists():
            try:
                with open(self.versions_file, 'r') as f:
                    data = json.load(f)
                    self.versions = {
                        path: GoldenFileVersion.from_dict(version_data)
                        for path, version_data in data.items()
                    }
            except (json.JSONDecodeError, KeyError) as e:
                # Handle corrupted version file
                print(f"Warning: Corrupted version file, creating new: {e}")
                self.versions = {}
    
    def _save_versions(self):
        """Save version information to file."""
        self.versions_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.versions_file, 'w') as f:
            data = {
                path: version.to_dict()
                for path, version in self.versions.items()
            }
            json.dump(data, f, indent=2, sort_keys=True)
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file content."""
        if not file_path.exists():
            return ""
        
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""
    
    def register_golden_file(self, file_path: Path) -> GoldenFileVersion:
        """Register a golden file and track its version."""
        relative_path = file_path.relative_to(self.fixtures_dir)
        content_hash = self._get_file_hash(file_path)
        timestamp = datetime.now().isoformat()
        git_commit = self._get_git_commit()
        
        version = GoldenFileVersion(
            file_path=relative_path,
            content_hash=content_hash,
            timestamp=timestamp,
            git_commit=git_commit
        )
        
        self.versions[str(relative_path)] = version
        self._save_versions()
        return version
    
    def check_for_changes(self, file_path: Path) -> Tuple[bool, Optional[GoldenFileVersion], str]:
        """
        Check if golden file has changed since last registration.
        
        Returns:
            (has_changed, previous_version, current_hash)
        """
        relative_path = file_path.relative_to(self.fixtures_dir)
        current_hash = self._get_file_hash(file_path)
        
        previous_version = self.versions.get(str(relative_path))
        if not previous_version:
            return True, None, current_hash
        
        has_changed = previous_version.content_hash != current_hash
        return has_changed, previous_version, current_hash
    
    def get_all_changes(self) -> List[Tuple[Path, bool, Optional[GoldenFileVersion], str]]:
        """Get all changes in golden files."""
        changes = []
        
        # Check tracked files
        for relative_path_str, version in self.versions.items():
            file_path = self.fixtures_dir / relative_path_str
            if file_path.exists():
                has_changed, prev_version, current_hash = self.check_for_changes(file_path)
                if has_changed:
                    changes.append((file_path, True, prev_version, current_hash))
            else:
                # File was deleted
                changes.append((file_path, True, version, ""))
        
        # Check for new files
        for file_path in self.fixtures_dir.rglob("*.webvtt"):
            if file_path.name != ".golden_versions.json":
                relative_path = file_path.relative_to(self.fixtures_dir)
                if str(relative_path) not in self.versions:
                    current_hash = self._get_file_hash(file_path)
                    changes.append((file_path, True, None, current_hash))
        
        for file_path in self.fixtures_dir.rglob("*.json"):
            if file_path.name != ".golden_versions.json":
                relative_path = file_path.relative_to(self.fixtures_dir)
                if str(relative_path) not in self.versions:
                    current_hash = self._get_file_hash(file_path)
                    changes.append((file_path, True, None, current_hash))
        
        return changes
    
    def approve_changes(self, file_paths: List[Path]) -> List[GoldenFileVersion]:
        """Approve changes to golden files and update versions."""
        approved_versions = []
        
        for file_path in file_paths:
            if file_path.exists():
                version = self.register_golden_file(file_path)
                approved_versions.append(version)
            else:
                # Remove deleted file from tracking
                relative_path = file_path.relative_to(self.fixtures_dir)
                if str(relative_path) in self.versions:
                    del self.versions[str(relative_path)]
        
        self._save_versions()
        return approved_versions
    
    def generate_change_report(self) -> str:
        """Generate a human-readable change report."""
        changes = self.get_all_changes()
        
        if not changes:
            return "✅ No changes detected in golden files."
        
        report = ["🔍 Golden File Changes Detected:"]
        report.append("=" * 50)
        
        for file_path, has_changed, prev_version, current_hash in changes:
            if prev_version:
                if current_hash == "":
                    status = "DELETED"
                else:
                    status = "MODIFIED"
                report.append(f"\n📝 {status}: {file_path.name}")
                report.append(f"   Previous: {prev_version.content_hash[:12]}...")
                report.append(f"   Current:  {current_hash[:12] if current_hash else 'N/A'}")
                report.append(f"   Last updated: {prev_version.timestamp}")
                if prev_version.git_commit:
                    report.append(f"   Git commit: {prev_version.git_commit[:12]}...")
            else:
                report.append(f"\n🆕 NEW: {file_path.name}")
                report.append(f"   Hash: {current_hash[:12]}...")
        
        report.append("\n" + "=" * 50)
        report.append("To approve these changes, run:")
        report.append("pytest src/vttiro/tests/test_golden_file_versioning.py::test_approve_golden_changes")
        
        return "\n".join(report)


class TestGoldenFileVersioning:
    """Test golden file versioning system."""
    
    @pytest.fixture
    def fixtures_dir(self, tmp_path):
        """Create temporary fixtures directory."""
        fixtures = tmp_path / "fixtures"
        fixtures.mkdir()
        return fixtures
    
    @pytest.fixture
    def tracker(self, fixtures_dir):
        """Create golden file tracker."""
        return GoldenFileTracker(fixtures_dir)
    
    @pytest.fixture
    def sample_webvtt_content(self):
        """Sample WebVTT content for testing."""
        return """WEBVTT

00:00:00.000 --> 00:00:02.000
Hello, this is a test transcription.

00:00:02.000 --> 00:00:04.000
This is used for golden file testing.
"""
    
    @pytest.fixture
    def sample_json_content(self):
        """Sample JSON content for testing."""
        return {
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "Hello, this is a test transcription.",
                    "confidence": 0.95
                },
                {
                    "start": 2.0,
                    "end": 4.0,
                    "text": "This is used for golden file testing.",
                    "confidence": 0.92
                }
            ],
            "language": "en",
            "provider": "test"
        }
    
    def test_register_new_golden_file(self, tracker, fixtures_dir, sample_webvtt_content):
        """Test registering a new golden file."""
        # Create golden file
        golden_file = fixtures_dir / "test_output.webvtt"
        golden_file.write_text(sample_webvtt_content)
        
        # Register file
        version = tracker.register_golden_file(golden_file)
        
        # Verify version information
        assert version.file_path == Path("test_output.webvtt")
        assert version.content_hash != ""
        assert version.timestamp != ""
        
        # Verify tracking
        assert str(version.file_path) in tracker.versions
        
        # Verify versions file was created
        assert tracker.versions_file.exists()
    
    def test_detect_file_changes(self, tracker, fixtures_dir, sample_webvtt_content):
        """Test detection of changes to golden files."""
        # Create and register golden file
        golden_file = fixtures_dir / "test_output.webvtt"
        golden_file.write_text(sample_webvtt_content)
        original_version = tracker.register_golden_file(golden_file)
        
        # Verify no changes initially
        has_changed, prev_version, current_hash = tracker.check_for_changes(golden_file)
        assert not has_changed
        assert prev_version.content_hash == current_hash
        
        # Modify file
        modified_content = sample_webvtt_content.replace("test transcription", "modified transcription")
        golden_file.write_text(modified_content)
        
        # Verify change detection
        has_changed, prev_version, current_hash = tracker.check_for_changes(golden_file)
        assert has_changed
        assert prev_version.content_hash != current_hash
        assert prev_version.content_hash == original_version.content_hash
    
    def test_detect_file_deletion(self, tracker, fixtures_dir, sample_webvtt_content):
        """Test detection of deleted golden files."""
        # Create and register golden file
        golden_file = fixtures_dir / "test_output.webvtt"
        golden_file.write_text(sample_webvtt_content)
        tracker.register_golden_file(golden_file)
        
        # Delete file
        golden_file.unlink()
        
        # Verify deletion detection
        has_changed, prev_version, current_hash = tracker.check_for_changes(golden_file)
        assert has_changed
        assert prev_version is not None
        assert current_hash == ""
    
    def test_detect_new_files(self, tracker, fixtures_dir, sample_webvtt_content, sample_json_content):
        """Test detection of new golden files."""
        # Create files without registering
        webvtt_file = fixtures_dir / "new_output.webvtt"
        webvtt_file.write_text(sample_webvtt_content)
        
        json_file = fixtures_dir / "new_output.json"
        json_file.write_text(json.dumps(sample_json_content, indent=2))
        
        # Check for changes
        changes = tracker.get_all_changes()
        
        # Verify new files detected
        assert len(changes) == 2
        new_files = [change[0] for change in changes]
        assert webvtt_file in new_files
        assert json_file in new_files
        
        # Verify they're marked as new (no previous version)
        for file_path, has_changed, prev_version, current_hash in changes:
            assert has_changed
            assert prev_version is None
            assert current_hash != ""
    
    def test_approve_changes(self, tracker, fixtures_dir, sample_webvtt_content):
        """Test approving changes to golden files."""
        # Create and register file
        golden_file = fixtures_dir / "test_output.webvtt"
        golden_file.write_text(sample_webvtt_content)
        original_version = tracker.register_golden_file(golden_file)
        
        # Modify file
        modified_content = sample_webvtt_content.replace("test", "approved")
        golden_file.write_text(modified_content)
        
        # Verify change detected
        changes = tracker.get_all_changes()
        assert len(changes) == 1
        
        # Approve changes
        approved_versions = tracker.approve_changes([golden_file])
        
        # Verify approval
        assert len(approved_versions) == 1
        new_version = approved_versions[0]
        assert new_version.content_hash != original_version.content_hash
        
        # Verify no more changes
        changes = tracker.get_all_changes()
        assert len(changes) == 0
    
    def test_generate_change_report(self, tracker, fixtures_dir, sample_webvtt_content):
        """Test generating change reports."""
        # No changes initially
        report = tracker.generate_change_report()
        assert "No changes detected" in report
        
        # Create file changes
        golden_file = fixtures_dir / "test_output.webvtt"
        golden_file.write_text(sample_webvtt_content)
        tracker.register_golden_file(golden_file)
        
        # Modify file
        modified_content = sample_webvtt_content.replace("test", "modified")
        golden_file.write_text(modified_content)
        
        # Add new file
        new_file = fixtures_dir / "new_file.webvtt"
        new_file.write_text(sample_webvtt_content)
        
        # Generate report
        report = tracker.generate_change_report()
        
        # Verify report content
        assert "Golden File Changes Detected" in report
        assert "MODIFIED: test_output.webvtt" in report
        assert "NEW: new_file.webvtt" in report
        assert "Previous:" in report
        assert "Current:" in report
    
    def test_version_persistence(self, tracker, fixtures_dir, sample_webvtt_content):
        """Test that version information persists across tracker instances."""
        # Create and register file
        golden_file = fixtures_dir / "test_output.webvtt"
        golden_file.write_text(sample_webvtt_content)
        original_version = tracker.register_golden_file(golden_file)
        
        # Create new tracker instance
        new_tracker = GoldenFileTracker(fixtures_dir)
        
        # Verify version information loaded
        assert str(original_version.file_path) in new_tracker.versions
        loaded_version = new_tracker.versions[str(original_version.file_path)]
        assert loaded_version.content_hash == original_version.content_hash
        assert loaded_version.timestamp == original_version.timestamp
    
    def test_corrupted_version_file_recovery(self, fixtures_dir):
        """Test recovery from corrupted version file."""
        # Create corrupted version file
        versions_file = fixtures_dir / ".golden_versions.json"
        versions_file.write_text("invalid json content")
        
        # Create tracker - should handle corruption gracefully
        tracker = GoldenFileTracker(fixtures_dir)
        
        # Verify empty versions (recovered state)
        assert len(tracker.versions) == 0
        
        # Verify can still register files
        golden_file = fixtures_dir / "test_output.webvtt"
        golden_file.write_text("test content")
        version = tracker.register_golden_file(golden_file)
        
        assert version.content_hash != ""
        assert str(version.file_path) in tracker.versions


# CLI integration for golden file management
def main():
    """CLI entry point for golden file management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Golden file version management")
    parser.add_argument("--fixtures-dir", default="src/vttiro/tests/fixtures", 
                       help="Path to fixtures directory")
    parser.add_argument("--check", action="store_true", 
                       help="Check for changes without approval")
    parser.add_argument("--approve", nargs="*", 
                       help="Approve changes to specific files (or all if none specified)")
    parser.add_argument("--report", action="store_true", 
                       help="Generate change report")
    
    args = parser.parse_args()
    
    fixtures_dir = Path(args.fixtures_dir)
    tracker = GoldenFileTracker(fixtures_dir)
    
    if args.check:
        changes = tracker.get_all_changes()
        if changes:
            print(f"Found {len(changes)} changed files:")
            for file_path, _, _, _ in changes:
                print(f"  - {file_path}")
        else:
            print("No changes detected.")
    
    elif args.approve is not None:
        if not args.approve:
            # Approve all changes
            changes = tracker.get_all_changes()
            file_paths = [change[0] for change in changes]
        else:
            # Approve specific files
            file_paths = [Path(path) for path in args.approve]
        
        approved = tracker.approve_changes(file_paths)
        print(f"Approved {len(approved)} file changes.")
    
    elif args.report:
        print(tracker.generate_change_report())
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()