---
this_file: PLAN.md
---

# vttiro Issue 105 - WebVTT Timing and Format Fixes

## Project Overview

This plan addresses critical issues identified in the WebVTT generation process, particularly the timing problems with Gemini transcription that result in invalid timestamp ranges and non-sequential timelines.

## Root Cause Analysis

The main issue is that we're asking Gemini for plain text transcription and then attempting to artificially generate timestamps based on estimated word timing. This approach fails because:

1. **No real timing information**: Gemini returns plain text without actual audio timing
2. **Artificial segmentation**: Our current code splits text arbitrarily and assigns estimated timestamps
3. **Timestamp calculation errors**: The estimation logic produces invalid ranges where end times precede start times
4. **Missing audio timing context**: We lose the relationship between audio segments and transcribed text

## Technical Architecture Decisions

### Core Strategy Change
Switch from requesting plain text to requesting WebVTT format directly from AI engines, leveraging their native timestamp generation capabilities.

### Key Design Principles
1. **Native format support**: Use engines' built-in WebVTT/SRT output when available
2. **Fallback compatibility**: Maintain current approach as fallback for engines without native timing
3. **Modular prompting**: Create reusable prompt templates with proper WebVTT examples
4. **Audio format optimization**: Use MP3 instead of WAV for better performance
5. **Optional identifiers**: Make WebVTT cue identifiers optional based on user preference

## Phase-by-Phase Implementation Plan

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