# vttiro

> Simple video transcription to WebVTT subtitles using AI models

**vttiro** is a straightforward Python tool that converts audio and video files to WebVTT subtitles using powerful AI transcription models. No complex setup, no over-engineering - just clean, simple transcription that works.

## ✨ Features

- **Multi-AI Support**: Gemini 2.0 Flash, OpenAI Whisper, AssemblyAI, Deepgram
- **Video & Audio**: Supports MP4, AVI, MOV, MKV, WebM, MP3, WAV, M4A, FLAC
- **WebVTT Output**: Industry-standard subtitle format with precise timestamps
- **Speaker Diarization**: Identify different speakers in audio
- **Simple CLI**: Clean command-line interface using Fire
- **Modern Python**: Built for Python 3.12+ with type hints

## 🚀 Quick Start

### Installation

```bash
# Core installation (includes CLI and configuration)
uv pip install vttiro

# Essential transcription (includes all AI providers)
uv pip install vttiro[basic]

# With local inference capabilities
uv pip install vttiro[local]

# YouTube download integration
uv pip install vttiro[youtube]

# Complete installation with all features
uv pip install vttiro[all]
```

### Usage

```bash
# Basic transcription with Gemini
vttiro transcribe video.mp4

# Use different AI engine
vttiro transcribe audio.mp3 --engine openai

# Specify model and output path
vttiro transcribe video.mp4 --engine gemini --model gemini-2.5-flash --output subtitles.vtt

# Add custom prompting
vttiro transcribe video.mp4 --prompt "Focus on technical terminology"

# Verbose mode for debugging
vttiro transcribe video.mp4 --verbose
```

### Available Commands

```bash
vttiro transcribe    # Transcribe audio/video to WebVTT
vttiro providers     # List available AI engines and models
vttiro config        # Show current configuration
vttiro version       # Display version information
```

## 🤖 Supported AI Engines

| Engine | Models | Features |
|--------|--------|----------|
| **Gemini** | 2.5-flash, 2.5-pro, 2.0-flash | Context-aware, multi-language |
| **OpenAI** | whisper-1, gpt-4o-transcribe | High accuracy, 25MB limit |
| **AssemblyAI** | universal-2 | Speaker diarization, 500MB limit |
| **Deepgram** | nova-3 | Real-time capable, 2GB limit |

## 🔧 Configuration

Set API keys as environment variables:

```bash
export VTTIRO_GEMINI_API_KEY="your-key"
export VTTIRO_OPENAI_API_KEY="your-key"
export VTTIRO_ASSEMBLYAI_API_KEY="your-key"
export VTTIRO_DEEPGRAM_API_KEY="your-key"
```

## 📝 Examples

### Basic Video Transcription
```bash
vttiro transcribe presentation.mp4
# Output: presentation.vtt
```

### Multi-language Support
```bash
vttiro transcribe spanish_video.mp4 --language es --engine gemini
```

### Custom Prompting
```bash
vttiro transcribe lecture.mp4 --prompt "Include speaker emotions and pauses"
```

### Dry Run (Validation Only)
```bash
vttiro transcribe video.mp4 --dry-run --verbose
```

## 🏗️ Architecture

vttiro v2.0 features a clean, modular architecture:

```
src/vttiro/
├── cli.py                 # Command-line interface
├── core/
│   ├── config.py         # Configuration management
│   ├── transcriber.py    # Main transcription orchestrator
│   ├── types.py          # Type definitions
│   └── errors.py         # Error handling
├── providers/            # AI engine implementations
│   ├── gemini/          # Gemini transcriber
│   ├── openai/          # OpenAI transcriber
│   ├── assemblyai/      # AssemblyAI transcriber
│   └── deepgram/        # Deepgram transcriber
├── utils/               # Utilities
│   ├── timestamp.py     # Time formatting
│   ├── prompt.py        # Prompt building
│   └── input_validation.py  # Input validation
└── output/              # Output formatting
    └── enhanced_webvtt.py    # WebVTT generation
```

## 🔄 Version 2.0 Changes

vttiro v2.0 represents a major simplification:

- **85% code reduction**: Removed enterprise bloat and over-engineering
- **Simplified CLI**: No deprecated flags, clean modern interface
- **Core focus**: Pure transcription functionality without unnecessary complexity
- **Better performance**: Streamlined codebase for faster execution
- **Easier maintenance**: Clean architecture for reliable operation

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 🐛 Issues & Support

Report issues at: https://github.com/twardoch/vttiro/issues