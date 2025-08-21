# vttiro

> Simple video transcription to WebVTT subtitles using AI models

**vttiro** is a straightforward Python tool that converts audio and video files to WebVTT subtitles using powerful AI transcription models. No complex setup, no over-engineering - just clean, simple transcription that works.

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style: Ruff](https://img.shields.io/badge/Code%20Style-Ruff-000000.svg)](https://github.com/astral-sh/ruff)

## ✨ What It Does

- **Transcribes local files**: Convert MP4, MP3, WAV, MOV, AVI, MKV, WebM to WebVTT subtitles
- **Multiple AI engines**: Google Gemini, AssemblyAI, Deepgram, OpenAI with specific model selection
- **Simple CLI**: `vttiro transcribe video.mp4` - that's it!
- **Clean output**: Properly formatted WebVTT with readable timestamps
- **Auto audio extraction**: Handles video files by extracting audio with ffmpeg

## 🚀 Quick Start

### 1. Install

```bash
uv pip install --system vttiro
```

### 2. Set API Key

```bash
export VTTIRO_GEMINI_API_KEY="your-gemini-api-key"
# OR
export VTTIRO_ASSEMBLYAI_API_KEY="your-assemblyai-key"  
# OR
export VTTIRO_DEEPGRAM_API_KEY="your-deepgram-key"
```

### 3. Transcribe

```bash
vttiro transcribe video.mp4
```

That's it! Your WebVTT file will be saved as `video.vtt`.

## 📖 Usage Examples

### Basic Transcription
```bash
# Transcribe video to WebVTT (uses default: gemini/gemini-2.0-flash)
vttiro transcribe meeting.mp4

# Custom output file
vttiro transcribe lecture.mp4 --output subtitles.vtt

# Use different AI engine
vttiro transcribe podcast.mp3 --engine assemblyai
vttiro transcribe audio.mp3 --engine openai

# Use specific model within engine
vttiro transcribe interview.mp4 --engine gemini --model gemini-2.5-pro
vttiro transcribe speech.mp4 --engine openai --model whisper-1
```

### Discovery Commands
```bash
# List available AI engines
vttiro engines

# List all available models
vttiro models

# List models for specific engine
vttiro models --engine gemini
```

### Check Supported Formats
```bash
vttiro formats
```

### Get Help
```bash
vttiro help
```

## 🎯 Supported Formats

**Input formats:**
- Video: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`
- Audio: `.mp3`, `.wav`, `.m4a`, `.aac`, `.ogg`, `.flac`

**Output format:**
- WebVTT (`.vtt`) - properly formatted with timestamps

## ⚙️ Configuration

vttiro uses environment variables for configuration:

| Variable | Description | Default |
|----------|-------------|---------|
| `VTTIRO_GEMINI_API_KEY` | Google Gemini API key | - |
| `VTTIRO_ASSEMBLYAI_API_KEY` | AssemblyAI API key | - |
| `VTTIRO_DEEPGRAM_API_KEY` | Deepgram API key | - |
| `VTTIRO_OPENAI_API_KEY` | OpenAI API key | - |
| `VTTIRO_ENGINE` | Default AI engine to use | `gemini` |
| `VTTIRO_CHUNK_DURATION` | Audio chunk size in seconds | `600` |

## 🤖 AI Engines and Models

vttiro supports multiple AI engines, each with various model options:

### **Gemini** (Google) - Default Engine
- `gemini-2.0-flash` (default) - Latest model with excellent accuracy
- `gemini-2.0-flash-exp` - Experimental version with cutting-edge features  
- `gemini-2.5-pro` - Highest quality model for critical transcription
- `gemini-2.5-flash` - Balanced speed and accuracy
- And more models available

### **AssemblyAI** - Professional-Grade Transcription  
- `universal-2` (default) - Latest universal transcription model
- `universal-1` - Previous generation universal model
- `nano` - Fast lightweight model
- `best` - Automatically selects optimal model

### **Deepgram** - Fast and Multilingual
- `nova-3` (default) - Latest Nova model with 30+ languages
- `nova-2` - Previous generation Nova model
- `enhanced` - Enhanced accuracy model
- `base` - Fast basic transcription
- `whisper-cloud` - OpenAI Whisper via Deepgram

### **OpenAI** - Industry-Standard Speech Recognition
- `gpt-4o-transcribe` (default) - GPT-4o Omni with native audio understanding
- `whisper-1` - OpenAI Whisper model optimized for speech transcription
- `gpt-4o-mini-transcribe` - Lightweight GPT-4o Mini with audio capabilities

**Usage:**
```bash
vttiro transcribe video.mp4 --engine gemini --model gemini-2.5-pro
vttiro transcribe audio.mp3 --engine deepgram --model nova-3
vttiro transcribe meeting.mp4 --engine assemblyai --model universal-2
vttiro transcribe interview.mp4 --engine openai --model whisper-1
```

## 📋 Requirements

- **Python 3.12+**
- **ffmpeg** (for video processing)
- **API key** for at least one supported service

### Install ffmpeg:

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html)

## 🔧 Installation Options

### Basic Installation (Recommended)
```bash
uv pip install --system vttiro
```
Includes all core functionality for API-based transcription.

### With Local Models
```bash
uv pip install --system vttiro[local]
```
Adds PyTorch and local inference capabilities.

### All Features
```bash
uv pip install --system vttiro[all]
```
Complete installation with all optional dependencies.

## 📝 Example Output

Input: `meeting.mp4`

Output: `meeting.vtt`
```webvtt
WEBVTT

cue-0001
00:00:00.000 --> 00:00:03.500
Hello everyone, welcome to today's meeting.

cue-0002
00:00:03.500 --> 00:00:07.200
We'll be discussing the quarterly results and future plans.
```

## 🚨 Troubleshooting

### "File not found" error
- Check that the input file exists and path is correct
- Use quotes around filenames with spaces: `vttiro transcribe "my video.mp4"`

### "No API key" error  
- Set at least one API key environment variable
- Verify the key is valid and has sufficient credits

### "ffmpeg not found" error
- Install ffmpeg using the instructions above
- Ensure ffmpeg is in your system PATH

### Audio extraction fails
- Check that the video file isn't corrupted
- Try with a different video format
- Ensure sufficient disk space for temporary files

## 🧑‍💻 Development

### Local Development Setup
```bash
git clone https://github.com/twardoch/vttiro.git
cd vttiro
uv venv --python 3.12
uv sync --all-extras
```

### Run Tests
```bash
uv run pytest
```

### Code Quality
```bash
uv run ruff check src/
uv run ruff format src/
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions welcome! Please read our [contributing guidelines](CONTRIBUTING.md) first.

## 📞 Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/twardoch/vttiro/issues)
- **Documentation**: [Read the full docs](https://github.com/twardoch/vttiro#readme)

---

**vttiro** - Simple transcription that just works! 🎬→📝