# VTTiro Simplification Tasks (Task 302 Analysis)

## Phase 1: Remove Over-Engineered Components

### A. Enterprise-Level Systems (Priority: HIGH)
- [ ] Remove Resilience Framework (`src/vttiro/core/resilience.py`)
- [ ] Replace with simple retry decorator using `tenacity` library
- [ ] Update all providers to use basic retry pattern
- [ ] Remove circuit breaker complexity
- [ ] Remove Configuration Schema System (`src/vttiro/validation/config_schema.py`)
- [ ] Keep basic Pydantic validation in VttiroConfig only
- [ ] Remove schema versioning and migration support
- [ ] Simplify config validation to essential checks
- [ ] Remove Output Quality Analyzer (`src/vttiro/output/quality_analyzer.py`)
- [ ] Replace with basic WebVTT format validation
- [ ] Remove quality metrics, accessibility scoring
- [ ] Keep only essential timestamp and content validation

### B. Unnecessary Abstractions (Priority: HIGH)
- [ ] Remove Multi-Format Exporter (`src/vttiro/output/multi_format_exporter.py`)
- [ ] Focus on WebVTT only (core objective)
- [ ] Remove SRT, TTML, ASS format support
- [ ] Simplify enhanced_webvtt.py to standard WebVTT generation
- [ ] Remove LLM Helper (`src/vttiro/utils/llm_helper.py`)
- [ ] Remove context enhancement via OpenAI
- [ ] Revert prompt.py changes that depend on LLM helper

## Phase 2: Simplify Over-Complicated Components

### A. Input Validation System (Priority: MEDIUM)
- [ ] Simplify Input Validation (`src/vttiro/utils/input_validation.py`)
- [ ] Replace complex validation with basic file checks
- [ ] Use pathlib and mimetypes for simple validation
- [ ] Remove detailed error reporting system

### B. Enhanced WebVTT Formatter (Priority: MEDIUM)
- [ ] Simplify WebVTT Formatter (`src/vttiro/output/enhanced_webvtt.py`)
- [ ] Remove accessibility scoring and WCAG compliance
- [ ] Remove advanced line breaking algorithms
- [ ] Keep basic WebVTT format with timestamps and speaker labels

## Phase 3: Remove Redundant Components

### A. Testing Infrastructure (Priority: HIGH)
- [ ] Consolidate Test Directories
- [ ] Remove entire `src/vttiro/tests/` directory
- [ ] Move essential tests to main `tests/` directory
- [ ] Remove duplicate test configurations
- [ ] Remove Security Module (`src/vttiro/security/security.py`)
- [ ] Use direct environment variable access
- [ ] Update providers to use standard env vars

### B. Development Infrastructure (Priority: LOW)
- [ ] Remove CI Enhancement System (`scripts/ci_enhancement.py`)
- [ ] Keep simple CI workflows only
- [ ] Remove pipeline management complexity
- [ ] Simplify GitHub Actions workflows

## Phase 4: Remove Non-Core Components

### A. External Dependencies (Priority: MEDIUM)
- [ ] Remove External Repository Integration (`external/repos/`)
- [ ] Use proper pip dependencies instead
- [ ] Remove local repository copies
- [ ] Update imports to use installed packages

### B. Development Utilities (Priority: LOW)
- [ ] Remove Advanced Testing
- [ ] Remove property-based testing files
- [ ] Remove memory profiling and benchmarks
- [ ] Keep basic integration tests only
- [ ] Remove Debugging Infrastructure (`src/vttiro/utils/debugging.py`)
- [ ] Remove comprehensive diagnostic system
- [ ] Keep basic logging only

## Phase 5: Clean Up and Consolidation

### A. Code Cleanup (Priority: MEDIUM)
- [ ] Fix Duplicate Imports in provider files
- [ ] Consolidate Configuration (remove pytest.ini, use pyproject.toml)
- [ ] Standardize Logging (loguru only, remove standard logging)
- [ ] Update Documentation to reflect simplified architecture

### B. Dependencies Cleanup (Priority: LOW)
- [ ] Remove Unused Dependencies from pyproject.toml
- [ ] Simplify Optional Dependencies groups
- [ ] Update Requirements to minimal set

---

## ✅ PREVIOUS MAJOR CLEANUP COMPLETED (85% reduction achieved)
**Removed 123,420 lines (85% reduction from 145,274 to 21,854 lines)**