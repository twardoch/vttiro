# VTTiro Simplification Tasks

## URGENT

- [x] You must TEST THIS APP with @temp/test2.sh ✅

- [x] At the beginnig of @CLAUDE.md write a detailed DEVELOPMENT LOOP of this specific project. That is, after we do changes, what do we test, how we test, where the error logs are etc. ✅

- [x] Actually MAKE THIS WORK see @issues/402.txt ✅ (API key fallback system implemented)

- [x] Remove validation and profile bloat from codebase ✅ (v1.0.9 - Validation & Profile Bloat Removal completed)

---

## 🎯 QUALITY IMPROVEMENT PHASE: 3 SMALL-SCALE RELIABILITY TASKS

Based on analysis of the current streamlined codebase, these specific improvements will increase quality, reliability & robustness:

### Task 1: Code Quality & Import Cleanup
**Goal**: Fix linting errors, organize imports, remove unused code from validation cleanup
**Focus Areas**:
- Fix unused imports and broken test code leftover from validation removal
- Convert relative imports to absolute imports (TID252 violations)
- Remove unused variables and clean up test assertions
- Organize imports properly at module top level

### Task 2: Exception Handling Improvements  
**Goal**: Improve exception handling patterns for better debugging and reliability
**Focus Areas**:
- Remove f-strings from exception constructors (EM102 violations)
- Add proper exception chaining with `raise ... from err` (B904 violations)
- Convert string literals in exceptions to variables (EM101 violations)
- Ensure consistent error handling patterns across providers

### Task 3: Constants & Magic Values Cleanup
**Goal**: Replace magic numbers with named constants for maintainability
**Focus Areas**:
- Replace magic values like `300`, `1024` with named constants (PLR2004 violations)
- Create constants module for commonly used values (timeouts, size limits, etc.)
- Document the meaning and rationale for key numeric values
- Improve code readability and make values configurable

**All tasks focus on core functionality reliability without adding bloat or complexity.**


