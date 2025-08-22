# VTTiro Test Fixtures

This directory contains test fixtures for VTTiro's testing suite, including golden files for output validation.

## Directory Structure

```
fixtures/
├── golden/                 # Golden files for output comparison
│   ├── *.vtt              # WebVTT format golden files
│   ├── *.json             # JSON format golden files
│   └── *.diff             # Diff files (auto-generated when tests fail)
├── audio/                 # Sample audio files for testing
├── video/                 # Sample video files for testing
└── configs/               # Sample configuration files
```

## Golden File Testing

Golden files (also known as snapshot tests) compare actual output against known-good reference outputs. This ensures output consistency across code changes.

### How Golden Files Work

1. **Initial Creation**: Run tests with `update_golden=True` to create initial golden files
2. **Comparison**: Subsequent test runs compare actual output against golden files
3. **Failure Handling**: When outputs differ, a `.diff` file is created for debugging
4. **Updates**: Use `update_golden=True` to update golden files when changes are intentional

### Golden File Categories

#### WebVTT Files (*.vtt)
- `basic_webvtt.vtt` - Basic WebVTT output format
- `webvtt_with_speakers.vtt` - WebVTT with speaker identification
- `webvtt_with_confidence.vtt` - WebVTT with confidence annotations
- `webvtt_long_text.vtt` - WebVTT with text wrapping
- `webvtt_special_chars.vtt` - WebVTT with special characters

#### JSON Files (*.json)
- `transcription_result.json` - Standard transcription result format
- `minimal_transcription.json` - Minimal transcription with no metadata
- `detailed_transcription.json` - Transcription with all fields populated
- `config_output.json` - Configuration serialization format
- `provider_comparison.json` - Provider comparison output
- `error_*.json` - Error message formats

### Usage Examples

```python
def test_webvtt_output(golden_manager):
    \"\"\"Test WebVTT output matches golden file.\"\"\"
    webvtt_content = generate_webvtt(transcription_result)
    
    # Compare with golden file (fails if different)
    assert golden_manager.compare_with_golden(
        "test_webvtt", webvtt_content, "vtt"
    )

def test_json_output_update(golden_manager):
    \"\"\"Test JSON output and update golden file.\"\"\"
    json_content = generate_json(transcription_result)
    
    # Update golden file if output changed
    assert golden_manager.compare_with_golden(
        "test_json", json_content, "json", update_golden=True
    )
```

### Best Practices

1. **Version Control**: Always commit golden files to version control
2. **Review Changes**: Carefully review golden file changes in PRs
3. **Deterministic Output**: Ensure test outputs are deterministic (avoid timestamps, random IDs)
4. **Meaningful Names**: Use descriptive names for golden files
5. **Small Files**: Keep golden files small and focused on specific scenarios

### Updating Golden Files

To update golden files when output format changes intentionally:

```bash
# Run specific test with update flag
pytest tests/test_golden_files.py::test_name -k "update"

# Or set environment variable
UPDATE_GOLDEN=1 pytest tests/test_golden_files.py
```

### Debugging Failed Golden File Tests

When a golden file test fails:

1. Check the generated `.diff` file in the golden directory
2. Compare expected vs actual output
3. Determine if change is intentional (update golden) or a bug (fix code)
4. Use `update_golden=True` to update if change is correct

## Sample Audio Files

Place small sample audio files here for testing:

- `sample_short.wav` - 5-second sample for quick tests
- `sample_multi_speaker.wav` - Multi-speaker sample for diarization tests
- `sample_noisy.wav` - Noisy audio sample for robustness tests
- `sample_quiet.wav` - Low-volume audio for sensitivity tests

## Sample Video Files

Place small sample video files here for testing:

- `sample_video.mp4` - Basic video with audio track
- `sample_no_audio.mp4` - Video without audio track
- `sample_multi_track.mkv` - Video with multiple audio tracks

## Configuration Files

Sample configuration files for testing different scenarios:

- `basic_config.yaml` - Basic configuration
- `advanced_config.yaml` - Configuration with all options
- `provider_configs/` - Provider-specific configurations

## Guidelines for Adding Fixtures

1. **Keep Files Small**: Use minimal file sizes for fast test execution
2. **No Copyright Content**: Only use content you own or public domain
3. **Document Purpose**: Add comments explaining what each fixture tests
4. **Organize by Category**: Group related fixtures in subdirectories
5. **Clean Periodically**: Remove unused fixtures to avoid clutter

## Automated Fixture Generation

Some fixtures can be generated automatically:

```python
# Generate synthetic audio for testing
from vttiro.tests.utils import generate_synthetic_audio
generate_synthetic_audio("fixtures/audio/synthetic_test.wav", duration=5.0)

# Generate test configurations
from vttiro.core.config import VttiroConfig
config = VttiroConfig(provider="gemini")
with open("fixtures/configs/test_config.json", "w") as f:
    json.dump(config.to_dict(), f, indent=2)
```