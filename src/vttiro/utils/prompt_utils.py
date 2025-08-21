#!/usr/bin/env python3
# this_file: src/vttiro/utils/prompt_utils.py
"""Prompt file utilities for safe reading and validation."""

import chardet
from pathlib import Path
from typing import Optional, Tuple

try:
    from loguru import logger
except ImportError:
    import logging as logger

from vttiro.utils.exceptions import ValidationError


# Maximum file size for prompt files (1MB)
MAX_PROMPT_FILE_SIZE = 1024 * 1024

# Minimum length for a valid prompt
MIN_PROMPT_LENGTH = 10

# Maximum length for a prompt (to prevent memory issues)
MAX_PROMPT_LENGTH = 50000

# Required WebVTT keywords that should be present in prompts
WEBVTT_KEYWORDS = ["webvtt", "timestamp", "format"]


def read_prompt_file(file_path: str) -> str:
    """Safely read and validate a prompt file.

    Args:
        file_path: Path to the prompt file

    Returns:
        Content of the prompt file

    Raises:
        ValidationError: If file is invalid, too large, or contains invalid content
    """
    try:
        prompt_path = Path(file_path)

        # Validate file exists
        if not prompt_path.exists():
            raise ValidationError(f"Prompt file not found: {file_path}")

        # Validate file is actually a file (not directory)
        if not prompt_path.is_file():
            raise ValidationError(f"Path is not a file: {file_path}")

        # Check file size
        file_size = prompt_path.stat().st_size
        if file_size == 0:
            raise ValidationError(f"Prompt file is empty: {file_path}")

        if file_size > MAX_PROMPT_FILE_SIZE:
            size_mb = file_size / (1024 * 1024)
            raise ValidationError(
                f"Prompt file too large: {size_mb:.1f}MB (max: {MAX_PROMPT_FILE_SIZE / (1024 * 1024):.1f}MB)"
            )

        # Read file with encoding detection
        with open(prompt_path, "rb") as f:
            raw_data = f.read()

        # Detect encoding
        encoding_info = chardet.detect(raw_data)
        encoding = encoding_info.get("encoding", "utf-8")
        confidence = encoding_info.get("confidence", 0)

        if confidence < 0.7:
            logger.warning(
                f"Low confidence ({confidence:.1%}) in encoding detection for {file_path}"
            )

        # Decode content
        try:
            content = raw_data.decode(encoding)
        except UnicodeDecodeError:
            # Fallback to utf-8 with error handling
            logger.warning(f"Failed to decode with {encoding}, falling back to utf-8")
            content = raw_data.decode("utf-8", errors="replace")

        # Validate content length
        if len(content.strip()) < MIN_PROMPT_LENGTH:
            raise ValidationError(
                f"Prompt file too short (minimum {MIN_PROMPT_LENGTH} characters)"
            )

        if len(content) > MAX_PROMPT_LENGTH:
            raise ValidationError(
                f"Prompt file too long (maximum {MAX_PROMPT_LENGTH} characters)"
            )

        logger.debug(
            f"Successfully read prompt file: {file_path} ({len(content)} chars, {encoding})"
        )
        return content.strip()

    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(f"Failed to read prompt file {file_path}: {e}")


def validate_webvtt_prompt(
    prompt: str, is_full_prompt: bool = False
) -> Tuple[bool, list[str]]:
    """Validate that a prompt contains necessary WebVTT instructions.

    Args:
        prompt: Prompt content to validate
        is_full_prompt: Whether this is a full prompt replacement (stricter validation)

    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    warnings = []
    prompt_lower = prompt.lower()

    # Check for essential WebVTT keywords
    missing_keywords = [kw for kw in WEBVTT_KEYWORDS if kw not in prompt_lower]

    if is_full_prompt and missing_keywords:
        warnings.append(
            f"Full prompt missing WebVTT keywords: {', '.join(missing_keywords)}"
        )
        warnings.append("This may result in plain text output instead of WebVTT format")

    # Check for common WebVTT format indicators
    webvtt_indicators = ["webvtt", "timestamp", "hh:mm:ss", "subtitle", "caption"]

    found_indicators = sum(
        1 for indicator in webvtt_indicators if indicator in prompt_lower
    )

    if is_full_prompt and found_indicators < 2:
        warnings.append("Prompt appears to lack WebVTT format instructions")
        warnings.append(
            "Consider including timestamp format and WebVTT structure requirements"
        )

    # Check for potentially problematic instructions
    problematic_phrases = ["plain text", "no timestamps", "text only", "without format"]

    found_problematic = [
        phrase for phrase in problematic_phrases if phrase in prompt_lower
    ]
    if found_problematic:
        warnings.append(
            f"Prompt contains potentially problematic instructions: {', '.join(found_problematic)}"
        )
        warnings.append("This may conflict with WebVTT format requirements")

    # Length validation
    if len(prompt.strip()) > 5000:
        warnings.append("Very long prompt may exceed AI model limits or increase costs")

    is_valid = len(warnings) == 0 or not any(
        "missing WebVTT keywords" in w for w in warnings
    )
    return is_valid, warnings


def process_prompt_argument(
    prompt_arg: Optional[str], arg_name: str, is_full_prompt: bool = False
) -> Optional[str]:
    """Process a prompt argument that could be file path or direct text.

    Args:
        prompt_arg: The prompt argument value (could be file path or text)
        arg_name: Name of the argument (for error messages)
        is_full_prompt: Whether this is a full prompt replacement

    Returns:
        Processed prompt content, or None if no argument provided

    Raises:
        ValidationError: If prompt is invalid
    """
    if not prompt_arg:
        return None

    prompt_arg = prompt_arg.strip()
    if not prompt_arg:
        return None

    # Check if it looks like a file path
    if (
        prompt_arg.startswith("/")
        or prompt_arg.startswith("./")
        or prompt_arg.startswith("~/")
        or "\\" in prompt_arg
        or (len(prompt_arg) < 200 and Path(prompt_arg).exists())
    ):

        # Treat as file path
        logger.debug(f"Processing {arg_name} as file path: {prompt_arg}")
        try:
            content = read_prompt_file(prompt_arg)

            # Validate WebVTT compatibility
            is_valid, warnings = validate_webvtt_prompt(content, is_full_prompt)

            if warnings:
                for warning in warnings:
                    logger.warning(f"{arg_name} validation: {warning}")

            return content

        except ValidationError as e:
            raise ValidationError(f"Error reading {arg_name} file: {e}")

    else:
        # Treat as direct text
        logger.debug(f"Processing {arg_name} as direct text ({len(prompt_arg)} chars)")

        # Basic validation for direct text
        if len(prompt_arg) < MIN_PROMPT_LENGTH:
            raise ValidationError(
                f"{arg_name} too short (minimum {MIN_PROMPT_LENGTH} characters)"
            )

        if len(prompt_arg) > MAX_PROMPT_LENGTH:
            raise ValidationError(
                f"{arg_name} too long (maximum {MAX_PROMPT_LENGTH} characters)"
            )

        # Validate WebVTT compatibility for full prompts
        if is_full_prompt:
            is_valid, warnings = validate_webvtt_prompt(prompt_arg, True)
            if warnings:
                for warning in warnings:
                    logger.warning(f"{arg_name} validation: {warning}")

        return prompt_arg


def get_prompt_preview(prompt: str, max_length: int = 100) -> str:
    """Get a preview of a prompt for display purposes.

    Args:
        prompt: Full prompt content
        max_length: Maximum length of preview

    Returns:
        Truncated prompt preview
    """
    if not prompt:
        return "(empty)"

    # Clean up whitespace
    clean_prompt = " ".join(prompt.split())

    if len(clean_prompt) <= max_length:
        return clean_prompt

    # Find a good break point near the limit
    preview = clean_prompt[:max_length]
    last_space = preview.rfind(" ")

    if last_space > max_length - 20:  # If space is reasonably close to end
        preview = preview[:last_space]

    return preview + "..."


def validate_prompt_combination(
    full_prompt: Optional[str], xtra_prompt: Optional[str]
) -> Tuple[bool, list[str]]:
    """Validate the combination of full and extra prompts.

    Args:
        full_prompt: Full prompt replacement (if any)
        xtra_prompt: Extra prompt to append (if any)

    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    warnings = []

    if full_prompt and xtra_prompt:
        combined_length = len(full_prompt) + len(xtra_prompt)
        if combined_length > MAX_PROMPT_LENGTH:
            warnings.append(
                f"Combined prompt length ({combined_length} chars) exceeds maximum ({MAX_PROMPT_LENGTH} chars)"
            )

    if full_prompt:
        # Validate full prompt more strictly
        is_valid, full_warnings = validate_webvtt_prompt(full_prompt, True)
        warnings.extend(full_warnings)

    if xtra_prompt:
        # Check if extra prompt conflicts with WebVTT requirements
        extra_lower = xtra_prompt.lower()
        conflicting_terms = ["plain text", "no format", "text only"]
        found_conflicts = [term for term in conflicting_terms if term in extra_lower]

        if found_conflicts:
            warnings.append(
                f"Extra prompt may conflict with WebVTT format: {', '.join(found_conflicts)}"
            )

    is_valid = (
        len(
            [
                w
                for w in warnings
                if "exceeds maximum" in w or "missing WebVTT keywords" in w
            ]
        )
        == 0
    )
    return is_valid, warnings
