# VTTiro Simplification Tasks

## URGENT

- [x] You must TEST THIS APP with @temp/test2.sh ✅

- [x] At the beginnig of @CLAUDE.md write a detailed DEVELOPMENT LOOP of this specific project. That is, after we do changes, what do we test, how we test, where the error logs are etc. ✅

- [x] Actually MAKE THIS WORK see @issues/402.txt ✅ (API key fallback system implemented)

- [x] Remove validation and profile bloat from codebase ✅ (v1.0.9 - Validation & Profile Bloat Removal completed)

---

## 🎯 QUALITY IMPROVEMENT PHASE: 3 SMALL-SCALE RELIABILITY TASKS

Based on analysis of the current streamlined codebase, these specific improvements will increase quality, reliability & robustness:

- [x] **Task 1: Code Quality & Import Cleanup** ✅
  - ✅ Fixed unused imports and broken test code (~50+ imports eliminated)
  - ✅ Organized imports properly at module top level  
  - ✅ Cleaned up test files (conftest.py, test_property_based.py)
  - ✅ Simplified provider imports (AssemblyAI)

- [x] **Task 2: Exception Handling Improvements** ✅  
  - ✅ Added proper exception chaining with `raise ... from e` (6 locations)
  - ✅ Improved error traceability and debugging context
  - ✅ Enhanced error information preservation across providers

- [x] **Task 3: Constants & Magic Values Cleanup** ✅
  - ✅ Created `src/vttiro/core/constants.py` with 20+ named constants
  - ✅ Replaced magic values with meaningful names throughout codebase
  - ✅ Added memory thresholds, processing limits, provider limits, audio constants
  - ✅ Improved code readability and maintainability

**All tasks focus on core functionality reliability without adding bloat or complexity.**

---

## 🔧 NEW QUALITY IMPROVEMENT PHASE: 3 TARGETED RELIABILITY FIXES

Based on concrete analysis of current codebase issues, these specific fixes will improve quality & robustness:

- [x] **Task 1: Test Infrastructure Fixes** ✅  
  - ✅ **Fixed OutputGenerationError**: Added missing error class to `core.errors` module
  - ✅ **Test Collection Fixed**: Resolved ImportError in `test_transcriber_integration.py`
  - ✅ **Verification Complete**: All 77 tests now collect successfully (up from 63 with errors)
  - ✅ **Import Resolution**: All test imports resolve correctly without errors

- [x] **Task 2: Type Annotation Completeness** ✅  
  - ✅ **Return Type Annotations**: Added missing `-> None` and specific return types to 8+ functions
  - ✅ **Audio Processing**: Fixed all missing annotations in `AudioProcessor` methods
  - ✅ **Context Managers**: Added proper typing for `__enter__`, `__exit__`, and `log_timing`
  - ✅ **CLI Functions**: Added annotation to progress callback function
  - ✅ **Import Updates**: Added `Iterator` import for context manager typing

- [x] **Task 3: Error Handling Framework Completion** ✅
  - ✅ **OutputGenerationError Added**: Complete error class with proper inheritance from VttiroError
  - ✅ **Error Hierarchy Verified**: All error classes referenced in tests now exist and are importable
  - ✅ **Test Compatibility**: All 77 tests collect without import errors
  - ✅ **Consistent Patterns**: Error class follows same patterns as other VttiroError subclasses

**All tasks address concrete, identified issues without adding bloat or new features.**

---

## 🔍 THIRD QUALITY IMPROVEMENT PHASE: 3 CODE CONSISTENCY FIXES

Based on concrete analysis using ruff linting and code review, these specific fixes will improve code quality & consistency:

### Task 1: Code Quality & Linting Fixes
**Goal**: Resolve remaining linting issues identified by ruff analysis  
**Specific Issues**:
- `UP035`: Import `Iterator` from `collections.abc` instead of `typing` (modern Python pattern)
- `F401`: Remove unused imports in `enhanced_webvtt.py` (`Optional`, `TranscriptSegment`)
- `E501`: Fix line length violations in `utils/prompt.py` (6+ lines >120 characters)
- Apply consistent import and formatting patterns

### Task 2: Exception Handling Pattern Improvements
**Goal**: Enhance exception handling specificity and reliability
**Focus Areas**:
- Fix bare `except:` clause in `utils/timestamp.py` (replace with specific exceptions)
- Review and improve exception handling patterns for better debugging
- Ensure consistent error reporting across modules
- Add proper exception context where missing

### Task 3: Code Consistency & Documentation Enhancement
**Goal**: Ensure consistent coding patterns and improve documentation coverage
**Focus Areas**:
- Verify docstring consistency across provider modules
- Standardize async function patterns and error handling
- Review function complexity and consider refactoring opportunities  
- Ensure consistent naming and code organization patterns

**All tasks focus on code quality, consistency, and maintainability without adding new features.**


