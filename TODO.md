# VTTiro Simplification Tasks

## URGENT

- [ ] You must TEST THIS APP with @temp/test2.sh 

- [ ] At the beginnig of @CLAUDE.md write a detailed DEVELOPMENT LOOP of this specific project. That is, after we do changes, what do we test, how we test, where the error logs are etc. 

- [ ] Actually MAKE THIS WORK see @issues/402.txt  



---

## 🎯 NEXT PHASE: 3 SMALL-SCALE QUALITY IMPROVEMENTS


#### Task 2: Remove Development Infrastructure Bloat  
**Goal**: Clean up unnecessary development scripts and configurations
**Files**: `scripts/ci_enhancement.py`, `scripts/generate_ci_test_data.py`, `scripts/setup_dev_automation.py`
**Actions**:
- Delete CI enhancement scripts (keep simple GitHub Actions only)
- Remove development automation complexity
- Clean up script dependencies from pyproject.toml if any
- Target: Remove ~500+ lines of development bloat

#### Task 3: Consolidate Configuration Management
**Goal**: Standardize project configuration to single source
**Files**: `pytest.ini` (if exists), `pyproject.toml`  
**Actions**:
- Move all pytest configuration to pyproject.toml [tool.pytest] section
- Remove standalone pytest.ini file
- Ensure consistent configuration approach
- Verify test configuration still works
- Target: Single source of truth for project configuration

---

## 🔄 LOWER PRIORITY REMAINING TASKS

### Phase 3: Optional Cleanup (Priority: LOW)

#### Dependencies Cleanup
- [ ] **Remove Unused Dependencies** from pyproject.toml
- [ ] **Simplify Optional Dependencies** groups  
- [ ] **Update Requirements** to minimal set

#### External Dependencies  
- [ ] **Remove External Repository Integration** (`external/repos/`) - if exists
  - Use proper pip dependencies instead
  - Remove local repository copies

#### Documentation Updates
- [ ] **Update Documentation** to reflect simplified architecture
- [ ] **Standardize Logging** - loguru only, remove standard logging fallbacks


