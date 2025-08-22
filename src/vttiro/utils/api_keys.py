# this_file: src/vttiro/utils/api_keys.py
"""API key management utilities.

Provides fallback logic for common API key environment variable names
to ensure compatibility with different configuration styles.

Used by:
- All provider transcribers for API key resolution
- Configuration validation
- Environment setup utilities
"""

import os


def get_api_key_with_fallbacks(provider: str, api_key: str | None = None) -> str | None:
    """Get API key with fallback to common environment variable names.

    Args:
        provider: Provider name (gemini, openai, deepgram, assemblyai)
        api_key: Explicitly provided API key (takes priority)

    Returns:
        API key string or None if not found

    Environment Variable Priority:
        - VTTIRO_{PROVIDER}_API_KEY (project-specific)
        - {PROVIDER}_API_KEY (standard)
        - GOOGLE_API_KEY (for Gemini fallback)
    """
    if api_key:
        return api_key

    provider_upper = provider.upper()

    # Define fallback patterns for each provider
    fallback_patterns = {
        "GEMINI": [
            f"VTTIRO_{provider_upper}_API_KEY",
            f"{provider_upper}_API_KEY",
            "GOOGLE_API_KEY",  # Common Google API key
            "GOOGLE_GENERATIVEAI_API_KEY",  # Alternative Google naming
        ],
        "OPENAI": [f"VTTIRO_{provider_upper}_API_KEY", f"{provider_upper}_API_KEY"],
        "DEEPGRAM": [
            f"VTTIRO_{provider_upper}_API_KEY",
            f"{provider_upper}_API_KEY",
            "DG_API_KEY",  # Deepgram SDK default
        ],
        "ASSEMBLYAI": [
            f"VTTIRO_{provider_upper}_API_KEY",
            f"{provider_upper}_API_KEY",
            "AAI_API_KEY",  # AssemblyAI SDK default
        ],
    }

    # Try each fallback pattern
    for env_var in fallback_patterns.get(provider_upper, [f"{provider_upper}_API_KEY"]):
        api_key = os.getenv(env_var)
        if api_key:
            return api_key

    return None


def get_all_available_api_keys() -> dict[str, str]:
    """Get all available API keys for debugging and validation.

    Returns:
        Dictionary mapping provider names to their API keys (first 10 chars only)
    """
    providers = ["gemini", "openai", "deepgram", "assemblyai"]
    available_keys = {}

    for provider in providers:
        api_key = get_api_key_with_fallbacks(provider)
        if api_key:
            # Only show first 10 characters for security
            available_keys[provider] = f"{api_key[:10]}..."
        else:
            available_keys[provider] = "Not found"

    return available_keys
