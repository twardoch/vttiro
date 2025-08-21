---
# this_file: PLAN.md
---

# vttiro Transcription Engine Fix Plan (Issue 204)

## Problem Analysis

Based on the test output in `issues/204.txt`, we identified two critical issues affecting transcription reliability:

### Issue 1: Gemini-2.5-pro Safety Filter Blocking (finish_reason: 2)
- **Root Cause**: Gemini's safety filters are blocking transcription responses due to overly restrictive safety settings
- **Error**: `Invalid operation: The response.text quick accessor requires the response to contain a valid Part, but none were returned. The candidate's finish_reason is 2`
- **Impact**: Complete failure of gemini-2.5-pro model for legitimate audio content

### Issue 2: OpenAI Models Silent Failure  
- **Root Cause**: OpenAI transcriber initialization or execution failing silently without error logging
- **Symptoms**: No output, no error messages, no logging - complete silent failure
- **Models Affected**: whisper-1, gpt-4o-transcribe, gpt-4o-mini-transcribe
- **Impact**: 100% failure rate for all OpenAI models

## Project Overview

This plan addresses these critical transcription engine failures to restore reliability across all supported AI models.

## Technical Solution Plan

### Phase 1: Fix Gemini Safety Filter Issue

#### 1.1 Implement Configurable Safety Settings
- **Location**: `src/vttiro/models/gemini.py`
- **Changes**:
  - Add safety settings configuration to `GeminiTranscriber.__init__()`
  - Create `_get_safety_settings()` method with reasonable defaults
  - Allow override via config or environment variables
  - Set default safety thresholds to `BLOCK_ONLY_HIGH` or `BLOCK_NONE` for transcription use case

#### 1.2 Add Safety Filter Error Handling  
- **Location**: `src/vttiro/models/gemini.py` in `transcribe()` method
- **Changes**:
  - Detect `finish_reason: 2` specifically
  - Provide user-friendly error message explaining safety filter blocking
  - Suggest retry with different settings or alternative models
  - Log the specific safety category that triggered the block

#### 1.3 Configuration Integration
- **Location**: `src/vttiro/core/config.py`
- **Changes**:
  - Add `gemini_safety_settings` configuration option
  - Allow per-category safety threshold configuration
  - Document safety implications in configuration

### Phase 2: Fix OpenAI Silent Failure Issue

#### 2.1 Enhanced Error Handling and Logging
- **Location**: `src/vttiro/models/openai.py`
- **Changes**:
  - Add comprehensive try-catch blocks around all initialization steps
  - Add debug logging for API key validation, client initialization
  - Add specific error handling for missing dependencies, API failures
  - Ensure all exceptions are properly logged and re-raised

#### 2.2 Graceful Dependency Management
- **Location**: `src/vttiro/models/openai.py`
- **Changes**:
  - Improve OpenAI package import error handling
  - Add validation for API key format and availability
  - Test API connectivity during initialization
  - Provide clear error messages for configuration issues

#### 2.3 CLI Integration Debugging
- **Location**: `src/vttiro/core/file_transcriber.py`
- **Changes**:
  - Add debug logging in `_create_transcriber()` for OpenAI path
  - Ensure transcriber creation errors are properly propagated
  - Add model validation before transcriber instantiation

### Phase 3: Robustness Improvements

#### 3.1 Universal Error Handling Framework
- **Location**: `src/vttiro/core/file_transcriber.py`
- **Changes**:
  - Implement standardized error handling pattern across all engines
  - Add correlation IDs to track errors across operations
  - Implement graceful degradation when models fail
  - Add retry logic with exponential backoff for transient failures

#### 3.2 Model Fallback System
- **Location**: `src/vttiro/core/file_transcriber.py`
- **Changes**:
  - Implement automatic fallback to alternative models when primary fails
  - Create model compatibility matrix for intelligent fallbacks
  - Add user notification when fallback occurs
  - Preserve user's engine preference while allowing model switching

#### 3.3 Configuration Validation
- **Location**: `src/vttiro/core/config.py`
- **Changes**:
  - Add startup validation for all configured engines
  - Test API connectivity and credentials during config load
  - Provide clear feedback on configuration issues
  - Add config doctor command for troubleshooting

### Phase 4: Testing and Validation

#### 4.1 Comprehensive Test Suite
- **Location**: `tests/integration/`
- **Changes**:
  - Create integration tests for all engine/model combinations
  - Add specific tests for safety filter scenarios
  - Add tests for API failure conditions
  - Implement test fixtures for various audio content types

#### 4.2 Error Scenario Testing
- **Location**: `tests/unit/`
- **Changes**:
  - Test safety filter error handling paths
  - Test OpenAI initialization failure scenarios
  - Test network failure and retry logic
  - Test configuration validation edge cases

#### 4.3 End-to-End Validation
- **Location**: `tests/e2e/`
- **Changes**:
  - Create automated test script similar to `temp/test1.sh`
  - Test all models with known-good audio samples
  - Validate error reporting and user experience
  - Performance regression testing

## Implementation Priority

1. **High Priority**: Fix OpenAI silent failure (blocking all OpenAI usage)
2. **High Priority**: Fix Gemini safety filter issue (blocking gemini-2.5-pro)
3. **Medium Priority**: Enhanced error handling framework
4. **Medium Priority**: Model fallback system
5. **Low Priority**: Comprehensive testing suite

## Success Criteria

- All OpenAI models produce transcription output or clear error messages
- Gemini-2.5-pro successfully transcribes legitimate audio content
- No silent failures - all errors provide actionable feedback
- Fallback system prevents total transcription failure
- Test suite achieves >95% pass rate across all engine/model combinations

## Risk Mitigation

- **Safety Settings**: Document security implications of relaxed safety filters
- **API Costs**: Implement cost estimation and limits for retry logic
- **Breaking Changes**: Maintain backward compatibility for existing configurations
- **Performance**: Ensure error handling doesn't impact transcription speed

## Future Enhancements

- Adaptive safety settings based on content type
- Machine learning-based model selection
- Real-time health monitoring for all engines
- User feedback integration for model quality assessment

### Phase 1: Core Prompt Infrastructure
**Priority: HIGH | Duration: 1-2 days**

#### 1.1 Create Common Prompt Module
- Create `src/vttiro/core/prompts.py` with:
  - Base WebVTT prompt templates
  - Example WebVTT output with proper formatting
  - Speaker diarization examples using WebVTT `<v Speaker>` syntax
  - Emotion/nonverbal indicators in square brackets
  - Language-specific prompt variations

#### 1.2 WebVTT Example Integration
- Include properly formatted WebVTT examples in prompts:
  ```webvtt
  WEBVTT

  00:00:01.000 --> 00:00:04.000
  <v Speaker1>Hello everyone, welcome to the session.

  00:00:04.500 --> 00:00:07.200
  <v Speaker2>[excited] Thanks for having me! This is great.
  ```

#### 1.3 Prompt Customization System
- Add CLI arguments:
  - `--full_prompt`: Replace default prompt entirely
  - `--xtra_prompt`: Append to default/custom prompt
  - `--add_cues`: Include cue identifiers in WebVTT (default: False)

### Phase 2: Gemini Integration Overhaul
**Priority: HIGH | Duration: 2-3 days**

#### 2.1 Update Gemini API Usage
- Migrate to new `google.genai` API format as shown in issue:
  ```python
  from google.genai import types
  
  response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=[
      'Generate WebVTT subtitles for this audio...',
      types.Part.from_bytes(
        data=audio_bytes,
        mime_type='audio/mp3',
      )
    ]
  )
  ```

#### 2.2 WebVTT-First Prompting
- Modify Gemini transcriber to request WebVTT format directly
- Include timing precision requirements in prompt
- Add chunk context for multi-segment processing
- Implement WebVTT validation and error handling

#### 2.3 Chunk Processing Enhancement
- Implement proper WebVTT merging for multi-chunk audio
- Handle timestamp continuity across chunks  
- Add overlap detection and resolution
- Preserve speaker continuity across segments

### Phase 3: Audio Processing Improvements
**Priority: MEDIUM | Duration: 1-2 days**

#### 3.1 Audio Format Migration
- Switch default audio extraction from WAV to MP3:
  - Update FFmpeg commands in `SimpleAudioProcessor`
  - Maintain quality while reducing file sizes
  - Add format detection and conversion logic

#### 3.2 Audio File Management
- Implement `--keep_audio` functionality:
  - Save extracted audio next to original video with same basename
  - Check for existing audio files before extraction
  - Add audio file reuse logic for repeated processing

#### 3.3 Chunk Size Validation
- Add 20MB limit validation for audio chunks
- Implement automatic chunk splitting:
  - Detect oversized chunks during processing
  - Split at natural audio boundaries (silence detection preferred)
  - Fallback to time-based splitting at ~50% duration mark
  - Update chunk metadata tracking

### Phase 4: WebVTT Generation Fixes
**Priority: HIGH | Duration: 1-2 days**

#### 4.1 Optional Cue Identifiers
- Modify `SimpleWebVTTGenerator` to support optional cue IDs:
  - Add `include_cue_ids: bool = False` parameter
  - Update `_build_webvtt_content()` method
  - Preserve existing behavior with opt-in flag

#### 4.2 Timestamp Validation
- Add comprehensive timestamp validation:
  - Ensure end_time > start_time for all cues
  - Detect and fix non-sequential timestamps
  - Add minimum gap enforcement between cues
  - Log and report timing irregularities

#### 4.3 WebVTT Parser Integration
- Create WebVTT parser for engine-generated content:
  - Parse native WebVTT from AI engines
  - Extract and validate cue structure
  - Handle malformed content gracefully
  - Convert to internal segment format

### Phase 5: Multi-Engine Native Format Support
**Priority: MEDIUM | Duration: 2-3 days**

#### 5.1 AssemblyAI Native Timing
- Research AssemblyAI's native WebVTT/SRT export capabilities
- Implement direct format retrieval if available
- Update transcription workflow to prefer native formats

#### 5.2 Deepgram Native Timing
- Research Deepgram's WebVTT support in API responses
- Implement native timestamp extraction
- Add Deepgram-specific WebVTT processing

#### 5.3 Unified Format Handling
- Create abstraction layer for native format processing:
  - `NativeFormatProcessor` base class
  - Engine-specific implementations
  - Fallback to current estimation approach
  - Format validation and conversion utilities

### Phase 6: Enhanced CLI Features
**Priority: LOW | Duration: 1 day**

#### 6.1 Prompt Control Arguments
- Implement `--full_prompt` flag:
  - Accept file path or direct text
  - Replace entire default prompt
  - Validate prompt structure for WebVTT requests

#### 6.2 Prompt Enhancement Arguments  
- Implement `--xtra_prompt` flag:
  - Append additional instructions to base prompt
  - Support both file and direct text input
  - Merge with context-aware prompting

#### 6.3 WebVTT Display Options
- Add `--add_cues` flag for cue identifier inclusion
- Implement cue naming patterns (numbered, speaker-based, timestamp-based)
- Add preview/validation mode for generated WebVTT

## Specific Implementation Steps

### Step 1: Create Core Prompt Infrastructure

1. **Create `src/vttiro/core/prompts.py`:**
   ```python
   class WebVTTPromptGenerator:
       def __init__(self, include_examples: bool = True, include_diarization: bool = True):
           self.include_examples = include_examples
           self.include_diarization = include_diarization
       
       def generate_webvtt_prompt(self, language: str = None, context: dict = None) -> str:
           # Base WebVTT generation prompt with examples
           
       def get_webvtt_example(self) -> str:
           # Return properly formatted WebVTT example
   ```

2. **WebVTT Example Template:**
   ```webvtt
   WEBVTT
   Language: auto
   
   00:00:00.000 --> 00:00:03.240
   <v Speaker1>Welcome to this presentation on font design.
   
   00:00:03.240 --> 00:00:06.180
   <v Speaker1>[enthusiastic] I'm excited to share my workflow with you.
   
   00:00:06.180 --> 00:00:08.920
   <v Speaker2>Thanks for having me! Let's dive right in.
   ```

### Step 2: Update Gemini Integration

1. **Modify `GeminiTranscriber.transcribe()` method:**
   - Change prompt to request WebVTT format
   - Add WebVTT parsing logic
   - Handle chunk merging with timestamp adjustment

2. **Update `_generate_transcription_prompt()` method:**
   - Integrate `WebVTTPromptGenerator`
   - Add WebVTT-specific instructions
   - Include format validation requirements

### Step 3: Audio Processing Updates

1. **Modify `SimpleAudioProcessor.extract_audio()`:**
   - Change output format from WAV to MP3
   - Add `--keep_audio` support with file reuse logic
   - Implement chunk size validation with splitting

2. **Update FFmpeg command:**
   ```python
   cmd = [
       'ffmpeg', '-i', str(video_path),
       '-vn', '-acodec', 'mp3', '-ar', '16000', '-ac', '1',
       '-b:a', '128k', '-y', str(output_path)
   ]
   ```

### Step 4: WebVTT Generator Enhancement

1. **Update `SimpleWebVTTGenerator.__init__()`:**
   ```python
   def __init__(self, ..., include_cue_ids: bool = False):
       self.include_cue_ids = include_cue_ids
   ```

2. **Modify `_build_webvtt_content()` method:**
   - Make cue ID generation conditional
   - Add timestamp validation before output
   - Ensure proper WebVTT structure

## Testing and Validation Criteria

### Critical Success Metrics
1. **Timing Accuracy**: All generated WebVTT files have valid timestamp ranges (end > start)
2. **Sequential Order**: Timestamps progress monotonically forward
3. **Format Compliance**: Generated WebVTT files validate against WebVTT specification
4. **Content Preservation**: Transcription quality maintains or improves current levels

### Test Cases
1. **Short audio clips** (< 1 minute): Perfect timing expected
2. **Medium audio clips** (1-10 minutes): Accurate segmentation with proper boundaries
3. **Long audio clips** (> 10 minutes): Multi-chunk processing with seamless merging
4. **Multiple speakers**: Proper diarization in WebVTT format
5. **Technical content**: Specialized terminology preserved with timing
6. **Various languages**: Multilingual support with appropriate timing

### Validation Process
1. **Automated Testing**: Unit tests for timestamp validation, format parsing
2. **Integration Testing**: End-to-end transcription with various file types
3. **Manual Validation**: Visual inspection of generated WebVTT in video players
4. **Performance Testing**: Processing time and accuracy comparisons

## Edge Cases and Error Handling

### Timestamp Edge Cases
- **Zero-duration segments**: Assign minimum 0.5-second duration
- **Overlapping segments**: Detect and resolve with gap insertion
- **Out-of-order segments**: Sort and validate timestamp sequence
- **Malformed engine responses**: Parse and repair or fallback to estimation

### Audio Processing Edge Cases  
- **Corrupted audio files**: Graceful failure with user feedback
- **Unsupported formats**: Clear error messages with format suggestions
- **Large files**: Progressive processing with memory management
- **Network interruptions**: Retry logic with partial result preservation

### Multi-Engine Compatibility
- **Format differences**: Normalize WebVTT structure across engines
- **Capability variations**: Feature detection and graceful degradation
- **API changes**: Version-aware integration with fallback options
- **Rate limiting**: Respect engine limits with queue management

## Future Considerations

### Performance Optimization
- **Parallel processing**: Multi-threaded audio chunk processing
- **Caching system**: Store intermediate results for large files
- **Progressive output**: Stream WebVTT cues as they're generated
- **Batch processing**: Optimize multiple file handling

### Advanced Features  
- **Custom timestamp adjustment**: User-defined offset and scaling
- **Style preservation**: WebVTT styling and positioning support
- **Metadata integration**: Rich metadata embedding in WebVTT files
- **Quality metrics**: Confidence scoring and accuracy reporting

### Integration Enhancements
- **Plugin system**: Extensible engine integration framework
- **Configuration profiles**: Predefined settings for common use cases
- **External tool integration**: Direct output to video editing software
- **Cloud deployment**: Scalable processing infrastructure

This comprehensive plan addresses all identified issues while establishing a robust foundation for future enhancements. The phased approach ensures critical timing issues are resolved first, followed by systematic improvements to audio processing and multi-engine support.