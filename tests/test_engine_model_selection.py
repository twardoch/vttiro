#!/usr/bin/env python3
# this_file: tests/test_engine_model_selection.py
"""Tests for engine/model selection logic and validation."""

import pytest
from vttiro.models.base import (
    TranscriptionEngine,
    GeminiModel,
    AssemblyAIModel, 
    DeepgramModel,
    get_default_model,
    get_available_models,
    validate_engine_model_combination,
    get_model_enum_class,
    ENGINE_DEFAULT_MODELS,
    ENGINE_AVAILABLE_MODELS
)


class TestTranscriptionEngine:
    """Test TranscriptionEngine enum."""
    
    def test_engine_values(self):
        """Test that all expected engines are available."""
        expected_engines = {"gemini", "assemblyai", "deepgram"}
        actual_engines = {engine.value for engine in TranscriptionEngine}
        assert actual_engines == expected_engines
    
    def test_engine_from_string(self):
        """Test creating engines from string values."""
        assert TranscriptionEngine("gemini") == TranscriptionEngine.GEMINI
        assert TranscriptionEngine("assemblyai") == TranscriptionEngine.ASSEMBLYAI
        assert TranscriptionEngine("deepgram") == TranscriptionEngine.DEEPGRAM
    
    def test_invalid_engine_raises_error(self):
        """Test that invalid engine strings raise ValueError."""
        with pytest.raises(ValueError):
            TranscriptionEngine("invalid_engine")


class TestModelEnums:
    """Test individual model enums."""
    
    def test_gemini_models(self):
        """Test GeminiModel enum values."""
        expected_models = {
            "gemini-2.0-flash",
            "gemini-2.0-flash-exp", 
            "gemini-2.0-flash-lite",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-pro"
        }
        actual_models = {model.value for model in GeminiModel}
        assert actual_models == expected_models
    
    def test_assemblyai_models(self):
        """Test AssemblyAIModel enum values.""" 
        expected_models = {"universal-1", "universal-2", "nano", "best"}
        actual_models = {model.value for model in AssemblyAIModel}
        assert actual_models == expected_models
    
    def test_deepgram_models(self):
        """Test DeepgramModel enum values."""
        expected_models = {"nova-2", "nova-3", "enhanced", "base", "whisper-cloud"}
        actual_models = {model.value for model in DeepgramModel}
        assert actual_models == expected_models


class TestDefaultModels:
    """Test default model functionality."""
    
    def test_get_default_model_for_all_engines(self):
        """Test that each engine has a default model."""
        for engine in TranscriptionEngine:
            default_model = get_default_model(engine)
            assert isinstance(default_model, str)
            assert len(default_model) > 0
    
    def test_default_models_exist_in_available(self):
        """Test that default models are in the available models list."""
        for engine in TranscriptionEngine:
            default_model = get_default_model(engine) 
            available_models = get_available_models(engine)
            assert default_model in available_models
    
    def test_specific_default_values(self):
        """Test specific expected default values."""
        assert get_default_model(TranscriptionEngine.GEMINI) == "gemini-2.0-flash"
        assert get_default_model(TranscriptionEngine.ASSEMBLYAI) == "universal-2"
        assert get_default_model(TranscriptionEngine.DEEPGRAM) == "nova-3"


class TestAvailableModels:
    """Test available models functionality."""
    
    def test_get_available_models_returns_list(self):
        """Test that get_available_models returns a list."""
        for engine in TranscriptionEngine:
            models = get_available_models(engine)
            assert isinstance(models, list)
            assert len(models) > 0
    
    def test_available_models_are_strings(self):
        """Test that all available models are strings."""
        for engine in TranscriptionEngine:
            models = get_available_models(engine)
            for model in models:
                assert isinstance(model, str)
                assert len(model) > 0
    
    def test_specific_available_models(self):
        """Test specific models are available for each engine."""
        gemini_models = get_available_models(TranscriptionEngine.GEMINI)
        assert "gemini-2.0-flash" in gemini_models
        assert "gemini-2.5-pro" in gemini_models
        
        assemblyai_models = get_available_models(TranscriptionEngine.ASSEMBLYAI)
        assert "universal-2" in assemblyai_models
        assert "universal-1" in assemblyai_models
        
        deepgram_models = get_available_models(TranscriptionEngine.DEEPGRAM)
        assert "nova-3" in deepgram_models
        assert "nova-2" in deepgram_models


class TestValidateEngineModel:
    """Test engine/model combination validation."""
    
    def test_valid_combinations(self):
        """Test that valid engine/model combinations return True."""
        valid_combinations = [
            ("gemini", "gemini-2.0-flash"),
            ("gemini", "gemini-2.5-pro"),
            ("assemblyai", "universal-2"),
            ("assemblyai", "nano"),
            ("deepgram", "nova-3"),
            ("deepgram", "enhanced")
        ]
        
        for engine, model in valid_combinations:
            assert validate_engine_model_combination(engine, model) is True
    
    def test_invalid_engine(self):
        """Test that invalid engines return False."""
        assert validate_engine_model_combination("invalid_engine", "some_model") is False
    
    def test_invalid_model_for_valid_engine(self):
        """Test that invalid models for valid engines return False."""
        invalid_combinations = [
            ("gemini", "universal-2"),  # AssemblyAI model with Gemini engine
            ("assemblyai", "nova-3"),   # Deepgram model with AssemblyAI engine
            ("deepgram", "gemini-2.0-flash"),  # Gemini model with Deepgram engine
            ("gemini", "invalid_model"),  # Completely invalid model
        ]
        
        for engine, model in invalid_combinations:
            assert validate_engine_model_combination(engine, model) is False
    
    def test_case_sensitivity(self):
        """Test that validation is case sensitive."""
        # These should fail due to case differences
        assert validate_engine_model_combination("Gemini", "gemini-2.0-flash") is False
        assert validate_engine_model_combination("gemini", "Gemini-2.0-Flash") is False


class TestModelEnumClass:
    """Test get_model_enum_class functionality."""
    
    def test_get_model_enum_class_for_all_engines(self):
        """Test that each engine returns correct enum class."""
        assert get_model_enum_class(TranscriptionEngine.GEMINI) == GeminiModel
        assert get_model_enum_class(TranscriptionEngine.ASSEMBLYAI) == AssemblyAIModel
        assert get_model_enum_class(TranscriptionEngine.DEEPGRAM) == DeepgramModel
    
    def test_invalid_engine_raises_error(self):
        """Test that invalid engine raises ValueError."""
        # Create a mock engine that doesn't exist
        with pytest.raises(ValueError, match="Unknown engine"):
            get_model_enum_class("invalid_engine")


class TestEngineModelMappings:
    """Test the underlying mappings are consistent."""
    
    def test_engine_default_models_consistency(self):
        """Test that ENGINE_DEFAULT_MODELS has all engines."""
        for engine in TranscriptionEngine:
            assert engine in ENGINE_DEFAULT_MODELS
    
    def test_engine_available_models_consistency(self):
        """Test that ENGINE_AVAILABLE_MODELS has all engines."""
        for engine in TranscriptionEngine:
            assert engine in ENGINE_AVAILABLE_MODELS
    
    def test_default_models_in_available_models(self):
        """Test that default models are in available models."""
        for engine in TranscriptionEngine:
            default_model = ENGINE_DEFAULT_MODELS[engine]
            available_models = ENGINE_AVAILABLE_MODELS[engine]
            assert default_model in available_models
    
    def test_available_models_match_enum_values(self):
        """Test that available models match actual enum values."""
        # Gemini
        gemini_enum_values = [model.value for model in GeminiModel]
        gemini_available = ENGINE_AVAILABLE_MODELS[TranscriptionEngine.GEMINI]
        assert set(gemini_enum_values) == set(gemini_available)
        
        # AssemblyAI
        assemblyai_enum_values = [model.value for model in AssemblyAIModel]
        assemblyai_available = ENGINE_AVAILABLE_MODELS[TranscriptionEngine.ASSEMBLYAI]
        assert set(assemblyai_enum_values) == set(assemblyai_available)
        
        # Deepgram
        deepgram_enum_values = [model.value for model in DeepgramModel]
        deepgram_available = ENGINE_AVAILABLE_MODELS[TranscriptionEngine.DEEPGRAM]
        assert set(deepgram_enum_values) == set(deepgram_available)


class TestEngineModelIntegration:
    """Integration tests for engine/model system."""
    
    def test_full_workflow_validation(self):
        """Test a complete workflow of engine/model selection and validation."""
        # Select an engine
        engine = TranscriptionEngine.GEMINI
        engine_str = engine.value
        
        # Get available models
        available_models = get_available_models(engine)
        assert len(available_models) > 0
        
        # Get default model
        default_model = get_default_model(engine)
        assert default_model in available_models
        
        # Validate the default combination
        assert validate_engine_model_combination(engine_str, default_model) is True
        
        # Test with each available model
        for model in available_models:
            assert validate_engine_model_combination(engine_str, model) is True
        
        # Get the model enum class
        model_enum_class = get_model_enum_class(engine)
        assert issubclass(model_enum_class, str)  # All our enums inherit from str
    
    def test_cross_engine_contamination(self):
        """Test that models from one engine don't work with another."""
        gemini_models = get_available_models(TranscriptionEngine.GEMINI)
        assemblyai_models = get_available_models(TranscriptionEngine.ASSEMBLYAI)
        deepgram_models = get_available_models(TranscriptionEngine.DEEPGRAM)
        
        # Test that Gemini models don't work with other engines
        for model in gemini_models:
            assert validate_engine_model_combination("assemblyai", model) is False
            assert validate_engine_model_combination("deepgram", model) is False
        
        # Test that AssemblyAI models don't work with other engines
        for model in assemblyai_models:
            assert validate_engine_model_combination("gemini", model) is False
            assert validate_engine_model_combination("deepgram", model) is False
        
        # Test that Deepgram models don't work with other engines
        for model in deepgram_models:
            assert validate_engine_model_combination("gemini", model) is False
            assert validate_engine_model_combination("assemblyai", model) is False


# Test fixtures for parametrized tests
@pytest.fixture
def all_engine_model_combinations():
    """Generate all valid engine/model combinations."""
    combinations = []
    for engine in TranscriptionEngine:
        engine_str = engine.value
        available_models = get_available_models(engine)
        for model in available_models:
            combinations.append((engine_str, model))
    return combinations


class TestParametrizedValidation:
    """Parametrized tests for comprehensive validation."""
    
    def test_all_valid_combinations(self, all_engine_model_combinations):
        """Test all valid engine/model combinations."""
        for engine, model in all_engine_model_combinations:
            assert validate_engine_model_combination(engine, model) is True, \
                f"Valid combination {engine}/{model} failed validation"