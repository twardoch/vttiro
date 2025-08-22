# VTTiro Simplification Tasks

## URGENT

- [x] You must TEST THIS APP with @temp/test2.sh ✅

- [x] At the beginnig of @CLAUDE.md write a detailed DEVELOPMENT LOOP of this specific project. That is, after we do changes, what do we test, how we test, where the error logs are etc. ✅

- [x] Actually MAKE THIS WORK see @issues/402.txt ✅ (API key fallback system implemented)



---

## 🎯 NEXT PHASE: 3 SMALL-SCALE QUALITY IMPROVEMENTS


#### Task 2: Remove Development Infrastructure Bloat ✅ COMPLETED
**Goal**: Clean up unnecessary development scripts and configurations
**Status**: No scripts/ directory exists - task was already completed

#### Task 3: Consolidate Configuration Management ✅ COMPLETED
**Goal**: Standardize project configuration to single source
**Status**: All pytest configuration already consolidated in pyproject.toml - no standalone config files exist

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


