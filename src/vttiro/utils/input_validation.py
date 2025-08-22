# this_file: src/vttiro/utils/input_validation.py
"""Simple input validation for VTTiro - bare minimum checks only."""

from pathlib import Path

from vttiro.core.errors import ValidationError


def validate_file_path(file_path: str | Path) -> Path:
    """Validate file exists and is readable."""
    path = Path(file_path)

    if not path.exists():
        raise ValidationError(f"File does not exist: {file_path}")

    if not path.is_file():
        raise ValidationError(f"Path is not a file: {file_path}")

    return path
