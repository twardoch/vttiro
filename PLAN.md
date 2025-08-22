# VTTiro Current State & Future Improvements Plan

## Current State (Post-Bloat Removal)

**VTTiro is now a clean, focused transcription tool with core functionality:**

- **Validation Simplification Complete**: `input_validation.py` reduced from 357 to 18 lines (95% reduction)
- **Configuration Debloating Complete**: Removed all profile management, validation decorators, and enterprise complexity from config
- **Core Architecture Preserved**: All transcription providers (Gemini, OpenAI, AssemblyAI, Deepgram) functional
- **Working CLI**: Basic functionality intact, user experience preserved
- **Test Suite**: Basic integration tests passing

**Project Philosophy**: Simple video transcription to WebVTT subtitles using AI models - no enterprise bloat.

## Success Criteria Met
- [x] All core transcription functionality preserved
- [x] WebVTT generation works perfectly  
- [x] All AI providers (Gemini, OpenAI, AssemblyAI, Deepgram) functional
- [x] CLI interface remains unchanged for users
- [x] Codebase is maintainable and focused

**Status: Ready for quality improvements focused on core functionality reliability and robustness.**

## Phase 1: Code Quality & Import Cleanup ⏳

**Objective**: Fix linting errors and organize imports properly following validation cleanup

**Technical Details**:
- **Fix Test Suite**: Clean up unused imports (`Path`, `MagicMock`, `patch`) and broken assertions (`assert validator is not None`) from validation removal
- **Import Organization**: Convert relative imports to absolute imports to fix TID252 violations across `src/vttiro/` modules
- **Import Placement**: Move imports like `import json` and `import shutil` to module top level (fix PLC0415 violations)
- **Unused Code Removal**: Remove leftover variables and clean up test fixtures that reference removed validation components

**Files to Modify**:
- `tests/test_basic_integration.py` - Fix broken test code
- `src/vttiro/__init__.py` - Convert relative imports
- `src/vttiro/utils/__init__.py` - Convert relative imports 
- All provider modules - Fix import organization
- `src/vttiro/processing/audio.py` - Move imports to top level

**Success Criteria**: All TID252 and PLC0415 linting errors resolved, tests run cleanly

## Phase 2: Exception Handling Improvements ⏳

**Objective**: Improve exception handling patterns for better debugging and reliability

**Technical Details**:
- **Remove F-Strings from Exceptions**: Fix EM102 violations by assigning f-string results to variables before passing to exceptions
- **Add Exception Chaining**: Fix B904 violations by adding `raise ... from err` or `raise ... from None` to distinguish error sources
- **Convert String Literals**: Fix EM101 violations by assigning string literals to variables before exception constructors
- **Standardize Patterns**: Ensure consistent error handling across all transcription providers

**Files to Modify**:
- `src/vttiro/processing/audio.py` - Multiple exception handling improvements
- All provider modules - Standardize exception patterns
- `src/vttiro/core/transcriber.py` - Review and improve error handling

**Success Criteria**: All EM101, EM102, B904 linting errors resolved, better error traceability

## Phase 3: Constants & Magic Values Cleanup ⏳

**Objective**: Replace magic numbers with named constants for maintainability

**Technical Details**:
- **Create Constants Module**: Define `src/vttiro/core/constants.py` with commonly used values
- **Audio Processing Constants**: Replace magic values like `300` (5 minutes), `1024` (1KB minimum file size)
- **Memory & Timeout Constants**: Define standard limits for file sizes, processing timeouts, retry counts
- **Documentation**: Add comments explaining the rationale for each constant value

**Constants to Define**:
- `MAX_AUDIO_DURATION_SECONDS = 300  # 5 minutes before chunking required`
- `MIN_CHUNK_FILE_SIZE_BYTES = 1024  # 1KB minimum for valid audio chunks`
- `DEFAULT_TIMEOUT_SECONDS = 300  # Standard API timeout`
- `MAX_FILE_SIZE_MB = 512  # Maximum input file size`

**Files to Modify**:
- `src/vttiro/core/constants.py` - New constants module
- `src/vttiro/processing/audio.py` - Replace magic values
- `src/vttiro/core/config.py` - Use constants for defaults
- Provider modules - Use timeout constants

**Success Criteria**: All PLR2004 linting errors resolved, code more maintainable and configurable