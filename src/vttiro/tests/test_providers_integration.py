# this_file: src/vttiro/tests/test_providers_integration.py
"""Integration tests for transcription providers using real APIs.

These tests run against actual provider APIs and are typically executed
in scheduled CI jobs to validate provider integrations and detect service issues.
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import patch

import pytest

from ..core.config import VttiroConfig
from ..core.types import TranscriptionResult
from ..providers.gemini.transcriber import GeminiTranscriber
from ..providers.openai.transcriber import OpenAITranscriber
from ..providers.assemblyai.transcriber import AssemblyAITranscriber
from ..providers.deepgram.transcriber import DeepgramTranscriber
from ..tests.test_data_generator import SyntheticAudioGenerator
from ..tests.test_advanced_quality import MemoryProfiler, PerformanceBenchmark


class IntegrationTestFramework:
    """Framework for running integration tests against real provider APIs."""
    
    def __init__(self):
        self.audio_generator = SyntheticAudioGenerator()
        self.profiler = MemoryProfiler()
        self.benchmark = PerformanceBenchmark()
        self.test_files = []
        
    def create_test_audio(self, duration: float = 5.0, description: str = "integration_test") -> Path:
        """Create test audio file for integration testing."""
        audio_path = self.audio_generator.generate_speech_simulation(
            transcript="Hello, this is a test audio file for integration testing. The quick brown fox jumps over the lazy dog.",
            duration=duration,
            filename_prefix=description
        )
        self.test_files.append(audio_path)
        return audio_path
        
    def cleanup_test_files(self):
        """Clean up test audio files."""
        for file_path in self.test_files:
            if file_path.exists():
                file_path.unlink()
        self.test_files.clear()
        
    def validate_transcription_result(self, result: TranscriptionResult, expected_provider: str) -> Dict[str, Any]:
        """Validate transcription result and return quality metrics."""
        validation = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "metrics": {}
        }
        
        # Basic structure validation
        if not isinstance(result, TranscriptionResult):
            validation["valid"] = False
            validation["errors"].append("Result is not a TranscriptionResult instance")
            return validation
            
        if result.provider != expected_provider:
            validation["warnings"].append(f"Provider mismatch: expected {expected_provider}, got {result.provider}")
            
        if not result.segments:
            validation["valid"] = False
            validation["errors"].append("No transcription segments found")
            return validation
            
        # Segment validation
        total_duration = 0
        for i, segment in enumerate(result.segments):
            if segment.start < 0:
                validation["errors"].append(f"Segment {i} has negative start time: {segment.start}")
                
            if segment.end <= segment.start:
                validation["errors"].append(f"Segment {i} has invalid timing: start={segment.start}, end={segment.end}")
                
            if not segment.text or not segment.text.strip():
                validation["warnings"].append(f"Segment {i} has empty or whitespace-only text")
                
            total_duration = max(total_duration, segment.end)
            
        # Quality metrics
        validation["metrics"] = {
            "segment_count": len(result.segments),
            "total_duration": total_duration,
            "average_segment_length": total_duration / len(result.segments) if result.segments else 0,
            "total_text_length": sum(len(seg.text) for seg in result.segments),
            "average_confidence": sum(seg.confidence or 0 for seg in result.segments) / len(result.segments) if result.segments else 0
        }
        
        # Mark as invalid if there are errors
        if validation["errors"]:
            validation["valid"] = False
            
        return validation


@pytest.fixture
def integration_framework():
    """Fixture providing integration test framework."""
    framework = IntegrationTestFramework()
    yield framework
    framework.cleanup_test_files()


@pytest.fixture
def skip_if_no_api_key():
    """Fixture to skip tests if API keys are not available."""
    def _skip_if_no_key(provider: str):
        key_map = {
            "gemini": "GEMINI_API_KEY",
            "openai": "OPENAI_API_KEY", 
            "assemblyai": "ASSEMBLYAI_API_KEY",
            "deepgram": "DEEPGRAM_API_KEY"
        }
        
        if provider not in key_map:
            pytest.skip(f"Unknown provider: {provider}")
            
        if not os.getenv(key_map[provider]):
            pytest.skip(f"API key not available for {provider}: {key_map[provider]}")
            
        if not os.getenv("INTEGRATION_TEST_ENABLED", "false").lower() == "true":
            pytest.skip("Integration tests not enabled. Set INTEGRATION_TEST_ENABLED=true")
            
    return _skip_if_no_key


class TestGeminiIntegration:
    """Integration tests for Gemini provider."""
    
    @pytest.mark.integration
    async def test_gemini_integration(self, integration_framework, skip_if_no_api_key):
        """Test Gemini provider integration with real API."""
        skip_if_no_api_key("gemini")
        
        # Create test audio
        test_audio = integration_framework.create_test_audio(duration=3.0, description="gemini_integration")
        
        # Initialize provider
        api_key = os.getenv("GEMINI_API_KEY")
        transcriber = GeminiTranscriber(api_key=api_key)
        
        # Benchmark the transcription
        integration_framework.profiler.start_profiling()
        start_time = time.time()
        
        try:
            result = await transcriber.transcribe(test_audio, language="en")
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
            
        end_time = time.time()
        memory_stats = integration_framework.profiler.stop_profiling()
        
        # Log performance metrics
        duration = end_time - start_time
        print(f"Gemini transcription took {duration:.2f}s, memory: {memory_stats}")
        
        # Validate result if successful
        if success and result:
            validation = integration_framework.validate_transcription_result(result, "gemini")
            
            assert validation["valid"], f"Transcription validation failed: {validation['errors']}"
            
            # Quality assertions
            metrics = validation["metrics"]
            assert metrics["segment_count"] > 0, "Should have at least one segment"
            assert metrics["total_text_length"] > 10, "Should have meaningful text content"
            
            # Performance assertions
            assert duration < 30.0, f"Transcription should complete within 30s, took {duration:.2f}s"
            
            print(f"Gemini integration test passed: {metrics}")
        else:
            pytest.fail(f"Gemini transcription failed: {error}")


class TestOpenAIIntegration:
    """Integration tests for OpenAI provider."""
    
    @pytest.mark.integration
    async def test_openai_integration(self, integration_framework, skip_if_no_api_key):
        """Test OpenAI provider integration with real API."""
        skip_if_no_api_key("openai")
        
        # Create test audio
        test_audio = integration_framework.create_test_audio(duration=3.0, description="openai_integration")
        
        # Initialize provider
        api_key = os.getenv("OPENAI_API_KEY")
        transcriber = OpenAITranscriber(api_key=api_key)
        
        # Benchmark the transcription
        result, benchmark = integration_framework.benchmark.benchmark_operation(
            "openai_transcription",
            lambda: transcriber.transcribe(test_audio, language="en")
        )
        
        print(f"OpenAI benchmark: {benchmark}")
        
        if benchmark["success"] and result:
            validation = integration_framework.validate_transcription_result(result, "openai")
            
            assert validation["valid"], f"Transcription validation failed: {validation['errors']}"
            
            # Quality assertions
            metrics = validation["metrics"]
            assert metrics["segment_count"] > 0, "Should have at least one segment"
            assert metrics["total_text_length"] > 10, "Should have meaningful text content"
            
            print(f"OpenAI integration test passed: {metrics}")
        else:
            pytest.fail(f"OpenAI transcription failed: {benchmark.get('error')}")


class TestAssemblyAIIntegration:
    """Integration tests for AssemblyAI provider."""
    
    @pytest.mark.integration
    async def test_assemblyai_integration(self, integration_framework, skip_if_no_api_key):
        """Test AssemblyAI provider integration with real API."""
        skip_if_no_api_key("assemblyai")
        
        # Create test audio
        test_audio = integration_framework.create_test_audio(duration=3.0, description="assemblyai_integration")
        
        # Initialize provider
        api_key = os.getenv("ASSEMBLYAI_API_KEY")
        transcriber = AssemblyAITranscriber(api_key=api_key)
        
        # Note: AssemblyAI typically has longer processing times due to async nature
        integration_framework.profiler.start_profiling()
        
        try:
            result = await transcriber.transcribe(test_audio, language="en")
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
            
        memory_stats = integration_framework.profiler.stop_profiling()
        print(f"AssemblyAI memory usage: {memory_stats}")
        
        if success and result:
            validation = integration_framework.validate_transcription_result(result, "assemblyai")
            
            assert validation["valid"], f"Transcription validation failed: {validation['errors']}"
            
            # Quality assertions
            metrics = validation["metrics"]
            assert metrics["segment_count"] > 0, "Should have at least one segment"
            assert metrics["total_text_length"] > 10, "Should have meaningful text content"
            
            print(f"AssemblyAI integration test passed: {metrics}")
        else:
            pytest.fail(f"AssemblyAI transcription failed: {error}")


class TestDeepgramIntegration:
    """Integration tests for Deepgram provider."""
    
    @pytest.mark.integration
    async def test_deepgram_integration(self, integration_framework, skip_if_no_api_key):
        """Test Deepgram provider integration with real API."""
        skip_if_no_api_key("deepgram")
        
        # Create test audio
        test_audio = integration_framework.create_test_audio(duration=3.0, description="deepgram_integration")
        
        # Initialize provider
        api_key = os.getenv("DEEPGRAM_API_KEY")
        transcriber = DeepgramTranscriber(api_key=api_key)
        
        # Benchmark the transcription
        result, benchmark = integration_framework.benchmark.benchmark_operation(
            "deepgram_transcription",
            lambda: transcriber.transcribe(test_audio, language="en")
        )
        
        print(f"Deepgram benchmark: {benchmark}")
        
        if benchmark["success"] and result:
            validation = integration_framework.validate_transcription_result(result, "deepgram")
            
            assert validation["valid"], f"Transcription validation failed: {validation['errors']}"
            
            # Quality assertions
            metrics = validation["metrics"]
            assert metrics["segment_count"] > 0, "Should have at least one segment"
            assert metrics["total_text_length"] > 10, "Should have meaningful text content"
            
            print(f"Deepgram integration test passed: {metrics}")
        else:
            pytest.fail(f"Deepgram transcription failed: {benchmark.get('error')}")


class TestCrossProviderIntegration:
    """Cross-provider integration tests."""
    
    @pytest.mark.integration
    async def test_provider_quality_comparison(self, integration_framework):
        """Compare transcription quality across providers."""
        if not os.getenv("COMPREHENSIVE_TEST", "false").lower() == "true":
            pytest.skip("Comprehensive testing not enabled")
            
        # Create test audio
        test_audio = integration_framework.create_test_audio(
            duration=5.0, 
            description="quality_comparison"
        )
        
        # Test all available providers
        providers = []
        results = {}
        
        if os.getenv("GEMINI_API_KEY"):
            providers.append(("gemini", GeminiTranscriber(api_key=os.getenv("GEMINI_API_KEY"))))
            
        if os.getenv("OPENAI_API_KEY"):
            providers.append(("openai", OpenAITranscriber(api_key=os.getenv("OPENAI_API_KEY"))))
            
        if len(providers) < 2:
            pytest.skip("Need at least 2 providers for quality comparison")
            
        # Run transcriptions
        for provider_name, transcriber in providers:
            try:
                result = await transcriber.transcribe(test_audio, language="en")
                validation = integration_framework.validate_transcription_result(result, provider_name)
                results[provider_name] = {
                    "result": result,
                    "validation": validation,
                    "success": validation["valid"]
                }
            except Exception as e:
                results[provider_name] = {
                    "result": None,
                    "validation": {"valid": False, "errors": [str(e)]},
                    "success": False
                }
                
        # Compare results
        successful_providers = [name for name, data in results.items() if data["success"]]
        
        assert len(successful_providers) >= 1, f"At least one provider should succeed. Results: {results}"
        
        # Log comparison metrics
        for provider_name, data in results.items():
            if data["success"]:
                metrics = data["validation"]["metrics"]
                print(f"{provider_name}: {metrics['segment_count']} segments, {metrics['total_text_length']} chars, {metrics['average_confidence']:.2f} confidence")
                
        print(f"Quality comparison completed for {len(successful_providers)} providers")


if __name__ == "__main__":
    # Direct execution for testing
    print("Integration test framework ready")
    
    framework = IntegrationTestFramework()
    
    # Create a test audio file
    test_audio = framework.create_test_audio(duration=2.0, description="manual_test")
    print(f"Test audio created: {test_audio}")
    
    # Cleanup
    framework.cleanup_test_files()
    print("Integration tests ready! ✅")