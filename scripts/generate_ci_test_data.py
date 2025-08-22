#!/usr/bin/env python3
# this_file: scripts/generate_ci_test_data.py
"""CI test data generation script for VTTiro testing.

This script generates consistent synthetic audio and video files for use in
CI/CD pipelines, ensuring reproducible test results across different environments.

Usage:
    python scripts/generate_ci_test_data.py [options]

Features:
- Generates standard test files for all providers
- Creates files with known properties for validation testing
- Supports multiple formats and durations
- Generates metadata for test validation
- Caches files for efficient CI runs

Used by:
- GitHub Actions CI pipeline
- Local development testing
- Performance benchmarking
- Provider integration testing
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vttiro.tests.test_data_generator import (
    TestDataManager, 
    generate_provider_test_files,
    TestAudioMetadata,
    TestVideoMetadata
)


def generate_ci_test_suite(output_dir: Path, formats: List[str] = None) -> Dict[str, Any]:
    """Generate complete test suite for CI.
    
    Args:
        output_dir: Directory to store generated files
        formats: List of formats to generate (default: all supported)
        
    Returns:
        Dictionary with test file metadata and paths
    """
    if formats is None:
        formats = ["wav", "mp3", "m4a"]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize test data manager with output directory
    manager = TestDataManager(output_dir)
    
    test_suite = {
        "generated_files": {},
        "provider_files": {},
        "validation_files": {},
        "metadata": {
            "formats": formats,
            "generation_timestamp": str(Path(__file__).stat().st_mtime),
            "total_files": 0,
            "total_size_bytes": 0
        }
    }
    
    # Generate standard provider test files
    print("Generating provider-specific test files...")
    provider_files = {}
    providers = ["openai", "gemini", "assemblyai", "deepgram"]
    
    for provider in providers:
        print(f"  Generating files for {provider}...")
        provider_files[provider] = {}
        
        # Small files for quick tests
        for format in formats:
            file_path, metadata = manager.get_test_audio(
                f"{provider}_small_tone",
                content_type="tone",
                duration=2.0,
                format=format,
                frequency=440.0
            )
            provider_files[provider][f"small_tone_{format}"] = {
                "path": str(file_path),
                "metadata": metadata.__dict__
            }
        
        # Medium files for standard tests
        file_path, metadata = manager.get_test_audio(
            f"{provider}_medium_speech",
            content_type="speech",
            duration=10.0,
            format="wav",
            transcript="This is a medium duration test file for transcription validation."
        )
        provider_files[provider]["medium_speech_wav"] = {
            "path": str(file_path),
            "metadata": metadata.__dict__
        }
        
        # Large files for stress tests (only for providers with higher limits)
        if provider in ["assemblyai", "deepgram"]:
            file_path, metadata = manager.get_test_audio(
                f"{provider}_large_silence",
                content_type="silence",
                duration=60.0,
                format="wav"
            )
            provider_files[provider]["large_silence_wav"] = {
                "path": str(file_path),
                "metadata": metadata.__dict__
            }
    
    test_suite["provider_files"] = provider_files
    
    # Generate validation test files
    print("Generating validation test files...")
    validation_files = {}
    
    # Different sample rates for compatibility testing
    sample_rates = [8000, 16000, 22050, 44100]
    for sr in sample_rates:
        file_path, metadata = manager.get_test_audio(
            f"validation_sr_{sr}",
            content_type="tone",
            duration=5.0,
            sample_rate=sr,
            frequency=1000.0
        )
        validation_files[f"sample_rate_{sr}"] = {
            "path": str(file_path),
            "metadata": metadata.__dict__
        }
    
    # Different channel configurations
    for channels in [1, 2]:
        file_path, metadata = manager.get_test_audio(
            f"validation_channels_{channels}",
            content_type="tone",
            duration=3.0,
            channels=channels,
            frequency=440.0
        )
        validation_files[f"channels_{channels}"] = {
            "path": str(file_path),
            "metadata": metadata.__dict__
        }
    
    # Different bit depths
    for bit_depth in [16, 24]:
        file_path, metadata = manager.get_test_audio(
            f"validation_bit_depth_{bit_depth}",
            content_type="tone",
            duration=2.0,
            bit_depth=bit_depth,
            frequency=440.0
        )
        validation_files[f"bit_depth_{bit_depth}"] = {
            "path": str(file_path),
            "metadata": metadata.__dict__
        }
    
    test_suite["validation_files"] = validation_files
    
    # Generate edge case files
    print("Generating edge case test files...")
    edge_case_files = {}
    
    # Very short file (minimum duration)
    file_path, metadata = manager.get_test_audio(
        "edge_very_short",
        content_type="tone",
        duration=0.1,
        frequency=440.0
    )
    edge_case_files["very_short"] = {
        "path": str(file_path),
        "metadata": metadata.__dict__
    }
    
    # Different frequencies for tone detection
    frequencies = [100, 440, 1000, 5000, 10000]
    for freq in frequencies:
        file_path, metadata = manager.get_test_audio(
            f"edge_freq_{freq}",
            content_type="tone",
            duration=2.0,
            frequency=freq
        )
        edge_case_files[f"frequency_{freq}hz"] = {
            "path": str(file_path),
            "metadata": metadata.__dict__
        }
    
    test_suite["edge_case_files"] = edge_case_files
    
    # Calculate total statistics
    total_files = 0
    total_size = 0
    
    for category in ["provider_files", "validation_files", "edge_case_files"]:
        for item in test_suite[category].values():
            if isinstance(item, dict) and "metadata" in item:
                total_files += 1
                total_size += item["metadata"].get("file_size_bytes", 0)
            else:
                for subitem in item.values():
                    if isinstance(subitem, dict) and "metadata" in subitem:
                        total_files += 1
                        total_size += subitem["metadata"].get("file_size_bytes", 0)
    
    test_suite["metadata"]["total_files"] = total_files
    test_suite["metadata"]["total_size_bytes"] = total_size
    
    print(f"Generated {total_files} test files ({total_size / 1024 / 1024:.1f} MB total)")
    
    return test_suite


def generate_test_manifest(test_suite: Dict[str, Any], output_file: Path):
    """Generate test manifest file for CI consumption.
    
    Args:
        test_suite: Test suite metadata
        output_file: Path to write manifest JSON
    """
    manifest = {
        "version": "1.0",
        "generated_at": test_suite["metadata"]["generation_timestamp"],
        "total_files": test_suite["metadata"]["total_files"],
        "total_size_mb": round(test_suite["metadata"]["total_size_bytes"] / 1024 / 1024, 2),
        "providers": list(test_suite["provider_files"].keys()),
        "formats": test_suite["metadata"]["formats"],
        "test_categories": {
            "provider_specific": len(test_suite["provider_files"]),
            "validation": len(test_suite["validation_files"]),
            "edge_cases": len(test_suite["edge_case_files"])
        },
        "files": {}
    }
    
    # Flatten file structure for easier CI consumption
    file_id = 0
    for category, files in test_suite.items():
        if category.endswith("_files"):
            category_name = category.replace("_files", "")
            
            for key, value in files.items():
                if isinstance(value, dict) and "path" in value:
                    # Direct file entry
                    manifest["files"][f"{category_name}_{key}"] = {
                        "id": file_id,
                        "path": value["path"],
                        "category": category_name,
                        "duration": value["metadata"].get("duration_seconds"),
                        "format": value["metadata"].get("format"),
                        "content_type": value["metadata"].get("content_type"),
                        "size_bytes": value["metadata"].get("file_size_bytes"),
                        "checksum": value["metadata"].get("checksum")
                    }
                    file_id += 1
                else:
                    # Nested structure (provider files)
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, dict) and "path" in subvalue:
                            manifest["files"][f"{category_name}_{key}_{subkey}"] = {
                                "id": file_id,
                                "path": subvalue["path"],
                                "category": category_name,
                                "provider": key,
                                "test_type": subkey,
                                "duration": subvalue["metadata"].get("duration_seconds"),
                                "format": subvalue["metadata"].get("format"),
                                "content_type": subvalue["metadata"].get("content_type"),
                                "size_bytes": subvalue["metadata"].get("file_size_bytes"),
                                "checksum": subvalue["metadata"].get("checksum")
                            }
                            file_id += 1
    
    # Write manifest
    with open(output_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Generated test manifest: {output_file}")


def validate_generated_files(test_suite: Dict[str, Any]) -> bool:
    """Validate that all generated files exist and have correct properties.
    
    Args:
        test_suite: Test suite metadata
        
    Returns:
        True if all files are valid
    """
    print("Validating generated files...")
    
    all_valid = True
    file_count = 0
    
    for category, files in test_suite.items():
        if not category.endswith("_files"):
            continue
            
        for key, value in files.items():
            if isinstance(value, dict) and "path" in value:
                # Direct file entry
                file_path = Path(value["path"])
                metadata = value["metadata"]
                
                if not file_path.exists():
                    print(f"  ERROR: Missing file {file_path}")
                    all_valid = False
                    continue
                
                # Validate file size
                actual_size = file_path.stat().st_size
                expected_size = metadata.get("file_size_bytes", 0)
                
                if actual_size != expected_size:
                    print(f"  ERROR: Size mismatch for {file_path}: {actual_size} != {expected_size}")
                    all_valid = False
                    continue
                
                file_count += 1
                
            else:
                # Nested structure
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, dict) and "path" in subvalue:
                        file_path = Path(subvalue["path"])
                        metadata = subvalue["metadata"]
                        
                        if not file_path.exists():
                            print(f"  ERROR: Missing file {file_path}")
                            all_valid = False
                            continue
                        
                        # Validate file size
                        actual_size = file_path.stat().st_size
                        expected_size = metadata.get("file_size_bytes", 0)
                        
                        if actual_size != expected_size:
                            print(f"  ERROR: Size mismatch for {file_path}: {actual_size} != {expected_size}")
                            all_valid = False
                            continue
                        
                        file_count += 1
    
    if all_valid:
        print(f"  ✅ All {file_count} files validated successfully")
    else:
        print(f"  ❌ Validation failed for some files")
    
    return all_valid


def main():
    """Main entry point for CI test data generation."""
    parser = argparse.ArgumentParser(description="Generate CI test data for VTTiro")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_data"),
        help="Output directory for generated files"
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["wav", "mp3", "m4a"],
        help="Audio formats to generate"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate generated files after creation"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Path to write test manifest JSON file"
    )
    
    args = parser.parse_args()
    
    try:
        # Generate test suite
        print(f"Generating CI test data in {args.output_dir}")
        test_suite = generate_ci_test_suite(args.output_dir, args.formats)
        
        # Validate if requested
        if args.validate:
            if not validate_generated_files(test_suite):
                print("❌ File validation failed")
                return 1
        
        # Generate manifest if requested
        if args.manifest:
            generate_test_manifest(test_suite, args.manifest)
        
        print("✅ CI test data generation completed successfully")
        return 0
        
    except Exception as e:
        print(f"❌ Error generating test data: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())