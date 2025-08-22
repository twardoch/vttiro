# this_file: src/vttiro/tests/conftest.py
"""Shared test configuration and fixtures for VTTiro tests.

This module provides common test fixtures and setup to reduce redundant
test code and improve test suite performance.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
import pytest


@pytest.fixture(scope="session", autouse=True)
def mock_api_keys():
    """Set up mock API keys for all providers for the entire test session.
    
    This prevents each individual test from having to patch environment
    variables, significantly improving test performance.
    """
    with patch.dict(os.environ, {
        'GEMINI_API_KEY': 'test-gemini-key',
        'OPENAI_API_KEY': 'test-openai-key', 
        'ASSEMBLYAI_API_KEY': 'test-assemblyai-key',
        'DEEPGRAM_API_KEY': 'test-deepgram-key',
    }):
        yield


@pytest.fixture(scope="function")
def temp_audio_file():
    """Create a temporary audio file for testing.
    
    Returns:
        Path: Path to temporary audio file that will be cleaned up after test
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        # Write minimal WAV header for a valid audio file
        wav_header = b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
        tmp_file.write(wav_header)
        tmp_file.flush()
        
        yield Path(tmp_file.name)
        
        # Cleanup
        try:
            os.unlink(tmp_file.name)
        except OSError:
            pass


@pytest.fixture(scope="function")
def mock_transcription_response():
    """Create a mock transcription response for testing.
    
    Returns:
        Mock: Mock object with typical transcription response structure
    """
    mock_response = Mock()
    mock_response.text = "Hello world, this is a test transcription."
    mock_response.segments = [
        Mock(start=0.0, end=2.5, text="Hello world,", confidence=0.95),
        Mock(start=2.5, end=5.0, text="this is a test transcription.", confidence=0.92)
    ]
    mock_response.confidence = 0.935
    mock_response.language = "en"
    return mock_response


@pytest.fixture(scope="function") 
def mock_provider_client():
    """Create a mock provider client for testing API interactions.
    
    Returns:
        Mock: Mock client with common provider methods
    """
    mock_client = Mock()
    mock_client.transcribe = Mock()
    mock_client.get_transcript = Mock()
    mock_client.submit = Mock()
    mock_client.wait_for_completion = Mock()
    return mock_client


class MockProviderMixin:
    """Mixin class providing common mock provider functionality.
    
    This reduces code duplication across provider test classes.
    """
    
    @staticmethod
    def create_mock_segments(count=3):
        """Create mock transcript segments for testing.
        
        Args:
            count: Number of segments to create
            
        Returns:
            List of mock segments with realistic timing and text
        """
        segments = []
        for i in range(count):
            start_time = i * 2.0
            end_time = start_time + 1.8
            mock_segment = Mock()
            mock_segment.start = start_time
            mock_segment.end = end_time  
            mock_segment.text = f"Test segment {i + 1}"
            mock_segment.confidence = 0.9 + (i * 0.02)  # Slightly varying confidence
            mock_segment.speaker = f"Speaker {(i % 2) + 1}" if i % 3 == 0 else None
            segments.append(mock_segment)
        return segments
    
    @staticmethod
    def create_mock_words(segment_text="Hello world test", confidence=0.95):
        """Create mock word-level transcription data.
        
        Args:
            segment_text: Text to split into words
            confidence: Base confidence score
            
        Returns:
            List of mock word objects
        """
        words = segment_text.split()
        mock_words = []
        
        for i, word in enumerate(words):
            start_time = i * 0.5
            end_time = start_time + 0.4
            
            mock_word = Mock()
            mock_word.word = word
            mock_word.start = start_time
            mock_word.end = end_time
            mock_word.confidence = confidence + (i * 0.01)  # Slight variation
            mock_words.append(mock_word)
            
        return mock_words


# Performance optimization settings for tests
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest for optimal test performance."""
    # Disable unnecessary warnings for faster test execution
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="assemblyai")
    warnings.filterwarnings("ignore", category=UserWarning, module="fs")


def pytest_collection_modifyitems(config, items):
    """Optimize test collection and execution order."""
    # Sort tests to run faster unit tests before slower integration tests
    def test_priority(item):
        if "integration" in item.name.lower():
            return 2  # Lower priority (run later)
        elif "provider" in item.fspath.basename:
            return 1  # Medium priority  
        else:
            return 0  # High priority (run first)
    
    items.sort(key=test_priority)