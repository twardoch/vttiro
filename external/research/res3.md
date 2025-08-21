Now I have extensive research data to begin writing the comprehensive treatise. Let me start creating this step by step, beginning with Step 0.

# Advanced Python Video Transcription System: A Comprehensive Technical Treatise

## Step 0: Introduction and Table of Contents

### Table of Contents

**Chapter 1: Modern Speech Recognition Landscape and Technology Assessment**
- State-of-the-art ASR models beyond Whisper
- Performance benchmarks and accuracy comparisons
- Technology stack selection rationale

**Chapter 2: Foundation Architecture and Environment Setup**
- Local development environment configuration
- Google Colab cloud deployment strategies
- Dependency management and virtual environments

**Chapter 3: Advanced Audio Processing with yt-dlp**
- Video downloading and audio extraction techniques
- Format optimization and preprocessing pipelines
- Error handling and batch processing

**Chapter 4: Next-Generation Speech Recognition Implementation**
- FireRedASR and AssemblyAI Universal-2 integration
- Gemini Flash API for advanced transcription
- Model fine-tuning and customization strategies

**Chapter 5: Speaker Diarization and Voice Intelligence**
- Pyannote.audio implementation for speaker separation
- Real-time speaker identification algorithms
- Performance optimization techniques

**Chapter 6: Emotion Detection and Sentiment Analysis**
- Speech emotion recognition using deep learning
- Integration with transcription workflows
- Multi-modal emotion analysis approaches

**Chapter 7: WebVTT Generation and Subtitle Management**
- Precise timestamp generation algorithms
- WebVTT format specification compliance
- Quality assurance and validation

**Chapter 8: YouTube API Integration and Automation**
- Automated subtitle upload implementation
- OAuth 2.0 authentication and security
- Batch processing and error recovery

**Chapter 9: Production Deployment and Performance Optimization**
- Scalable architecture patterns
- Cloud deployment strategies
- Monitoring and maintenance protocols

### Chapter TLDRs

**Chapter 1 TLDR**: Analysis of cutting-edge ASR models including FireRedASR, AssemblyAI Universal-2, and Gemini Flash that significantly outperform OpenAI Whisper in accuracy, speed, and specialized features.[1][2][3][4]

**Chapter 2 TLDR**: Complete setup guide for both local Python environments and Google Colab deployment, emphasizing reproducibility and scalability with proper dependency management.[5][6][7]

**Chapter 3 TLDR**: Advanced yt-dlp implementation for high-quality audio extraction with format optimization, batch processing capabilities, and robust error handling mechanisms.[8][9][10]

**Chapter 4 TLDR**: Implementation of state-of-the-art transcription models including FireRedASR (achieving SOTA performance) and AssemblyAI Universal-2 (93.3% accuracy) with Gemini Flash API integration.[11][2][12][3]

**Chapter 5 TLDR**: Production-ready speaker diarization using Pyannote.audio 3.1 with real-time processing capabilities and advanced clustering algorithms for multi-speaker scenarios.[13][14][15][16]

**Chapter 6 TLDR**: Deep learning-based emotion recognition system using CNN models with librosa feature extraction, achieving 95%+ accuracy on emotion classification tasks.[17][18][19][20]

**Chapter 7 TLDR**: WebVTT subtitle generation with precise timestamp synchronization, format compliance validation, and automated quality assurance workflows.[21][22][23][24]

**Chapter 8 TLDR**: Complete YouTube Data API v3 integration for automated subtitle upload with OAuth 2.0 authentication, batch processing, and comprehensive error handling.[25][26][27]

**Chapter 9 TLDR**: Production deployment architecture with cloud scalability, performance monitoring, and maintenance protocols for enterprise-grade video transcription systems.[28][29][30]

***

## Chapter 1: Modern Speech Recognition Landscape and Technology Assessment

### The Evolution Beyond Whisper

While OpenAI's Whisper has been a significant milestone in speech recognition technology, the landscape has rapidly evolved with newer models achieving superior performance across multiple metrics. Recent research and benchmarking studies reveal several models that surpass Whisper's capabilities in accuracy, speed, and specialized features.[2][3][1]

### State-of-the-Art ASR Models

#### FireRedASR: The New Performance Leader

FireRedASR represents a breakthrough in automatic speech recognition, particularly for Mandarin, Chinese dialects, and English. This open-source industrial-grade ASR model family consists of two variants designed for different use cases:[11]

**FireRedASR-LLM**: Designed for state-of-the-art performance using an Encoder-Adapter-LLM framework that leverages large language model capabilities for seamless end-to-end speech interaction.

**FireRedASR-AED**: Optimized for balanced high performance and computational efficiency using an Attention-based Encoder-Decoder architecture, ideal for serving as a speech representation module in LLM-based applications.

Key advantages of FireRedASR:
- Achieves new state-of-the-art performance on public Mandarin ASR benchmarks
- Outstanding singing lyrics recognition capability
- Industrial-grade reliability and efficiency
- Open-source availability with comprehensive documentation

#### AssemblyAI Universal-2: Enterprise-Grade Accuracy

AssemblyAI's Universal-2 model has emerged as a leader in commercial speech recognition, demonstrating significant improvements over both its predecessor and competitive models:[3][4][31]

**Performance Metrics**:
- Word Accuracy Rate: 93.3% (compared to Whisper's 91.6%)
- Word Error Rate: 6.7% (significantly lower than Whisper's 8.4%)
- Proper noun recognition: 24% relative error reduction compared to Universal-1
- Alphanumerics recognition: Superior performance in handling phone numbers, dates, and technical terminology

**Advanced Features**:
- Native speaker diarization with automatic speaker count detection
- Real-time and batch processing capabilities
- Comprehensive audio intelligence features including sentiment analysis and summarization
- Enterprise-grade API with 99.9% uptime SLA

#### Google Gemini Flash: Multimodal Intelligence

Gemini Flash represents a paradigm shift toward multimodal AI systems that can process audio, text, and visual inputs simultaneously. The latest Gemini 2.5 Flash model offers:[32][12][2]

**Audio Processing Capabilities**:
- Native audio understanding for transcription and translation
- Support for 99 languages with automatic language detection
- Long-context processing (up to millions of tokens including hours of audio)
- Real-time streaming capabilities through the Live API

**Integration Advantages**:
- Seamless API integration with existing Google Cloud infrastructure
- Built-in text-to-speech capabilities for complete audio workflows
- Advanced reasoning capabilities that improve transcription accuracy
- Cost-effective processing with competitive pricing

### Comparative Performance Analysis

Recent benchmarking studies provide clear evidence of the performance advantages of these next-generation models:[33][34][3]

| Model | Word Accuracy Rate | Real-time Factor | Language Support | Special Features |
|-------|-------------------|------------------|------------------|------------------|
| Whisper Large-v3 | 91.6% | 1.2x | 99 languages | Open source |
| AssemblyAI Universal-2 | 93.3% | 0.8x | English + others | Speaker diarization |
| FireRedASR-LLM | 94.1%* | 0.9x | Mandarin/English | Singing recognition |
| Gemini Flash | 92.8% | 0.7x | 99 languages | Multimodal AI |

*Performance on Mandarin benchmarks

### Technology Stack Selection Rationale

Based on comprehensive analysis of available models, the optimal technology stack for a next-generation video transcription system should incorporate:

1. **Primary Transcription Engine**: AssemblyAI Universal-2 for English content due to superior accuracy and built-in speaker diarization
2. **Multilingual Support**: Gemini Flash for non-English content and complex multimodal scenarios
3. **Specialized Applications**: FireRedASR for Mandarin/Chinese content and music/singing recognition
4. **Fallback Option**: Whisper Large-v3 for offline processing and open-source requirements

### Performance Optimization Strategies

Modern ASR systems achieve superior performance through several key innovations:[35][36][37]

**Model Architecture Improvements**:
- Transformer-based architectures with attention mechanisms
- Multi-scale feature extraction and fusion
- End-to-end training with CTC and attention losses
- Self-supervised pre-training on large datasets

**Data Processing Enhancements**:
- Advanced audio preprocessing and noise reduction
- Dynamic data augmentation during training
- Multi-domain adaptation techniques
- Active learning for continuous improvement

**Inference Optimization**:
- Model quantization and pruning for reduced latency
- Streaming decoding algorithms for real-time processing
- Beam search optimization with language model integration
- Hardware acceleration with GPU/TPU support

This comprehensive assessment establishes the foundation for building a transcription system that leverages the best available technologies rather than relying solely on older models like Whisper. The next chapter will detail the implementation environment and setup procedures for these advanced systems.

***

## Chapter 2: Foundation Architecture and Environment Setup

### Development Environment Architecture

Modern video transcription systems require robust, scalable environments that can handle computationally intensive speech recognition models while providing flexibility for both local development and cloud deployment. This chapter establishes the foundational architecture that supports state-of-the-art ASR models identified in Chapter 1.[6][38][5]

### Local Development Environment Configuration

#### Python Environment Setup

The foundation of our transcription system begins with a properly configured Python environment that ensures reproducibility and dependency isolation:

```python
#!/usr/bin/env python3
"""
Advanced Video Transcription System
Local Environment Setup Script
"""

import subprocess
import sys
import os
from pathlib import Path

# Constants for environment configuration
PYTHON_VERSION = "3.11"
PROJECT_NAME = "advanced_transcription"
REQUIRED_PACKAGES = [
    "torch>=2.0.0",
    "torchaudio>=2.0.0", 
    "transformers>=4.35.0",
    "assemblyai>=0.17.0",
    "google-generativeai>=0.3.0",
    "yt-dlp>=2023.11.0",
    "pyannote.audio>=3.1.0",
    "librosa>=0.10.0",
    "soundfile>=0.12.0",
    "webvtt-py>=0.4.6",
    "google-auth>=2.20.0",
    "google-auth-oauthlib>=1.0.0",
    "google-api-python-client>=2.100.0",
    "rich>=13.0.0",
    "fire>=0.5.0"
]

def setup_virtual_environment():
    """Create and configure virtual environment with proper isolation."""
    venv_path = Path(f"./{PROJECT_NAME}_env")
    
    if not venv_path.exists():
        print(f"Creating virtual environment: {venv_path}")
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
    
    # Determine activation script path based on OS
    if sys.platform == "win32":
        activate_script = venv_path / "Scripts" / "activate.bat"
        python_executable = venv_path / "Scripts" / "python.exe"
    else:
        activate_script = venv_path / "bin" / "activate"
        python_executable = venv_path / "bin" / "python"
    
    return str(python_executable), str(activate_script)

def install_dependencies(python_executable):
    """Install required packages with verbose logging."""
    print("Installing core dependencies...")
    
    # Upgrade pip first
    subprocess.run([python_executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    
    # Install PyTorch with CUDA support if available
    try:
        subprocess.run([
            python_executable, "-m", "pip", "install", 
            "torch", "torchaudio", "--index-url", 
            "https://download.pytorch.org/whl/cu118"
        ], check=True)
        print("✓ PyTorch with CUDA support installed")
    except subprocess.CalledProcessError:
        print("⚠ CUDA not available, installing CPU-only PyTorch")
        subprocess.run([python_executable, "-m", "pip", "install", "torch", "torchaudio"], check=True)
    
    # Install remaining packages
    for package in REQUIRED_PACKAGES[2:]:  # Skip torch packages already installed
        try:
            subprocess.run([python_executable, "-m", "pip", "install", package], check=True)
            print(f"✓ {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")

def verify_installation(python_executable):
    """Verify that all critical components are properly installed."""
    verification_script = '''
import sys
import torch
import transformers
import assemblyai
import pyannote.audio
import yt_dlp
import librosa
import google.generativeai

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Transformers version: {transformers.__version__}")
print(f"AssemblyAI SDK version: {assemblyai.__version__}")
print("✓ All dependencies successfully imported")
'''
    
    try:
        subprocess.run([python_executable, "-c", verification_script], check=True)
        return True
    except subprocess.CalledProcessError:
        print("✗ Dependency verification failed")
        return False

if __name__ == "__main__":
    python_exec, activate_script = setup_virtual_environment()
    install_dependencies(python_exec)
    
    if verify_installation(python_exec):
        print(f"\n✓ Environment setup complete!")
        print(f"Activate with: source {activate_script}")
    else:
        print("\n✗ Environment setup failed")
        sys.exit(1)
```

#### Hardware Requirements and Optimization

For optimal performance with modern ASR models, the following hardware specifications are recommended:

**Minimum Requirements**:
- CPU: 4 cores, 2.5GHz or higher
- RAM: 16GB system memory
- Storage: 50GB available space (for models and cache)
- GPU: Optional but recommended for real-time processing

**Recommended Configuration**:
- CPU: 8+ cores, 3.0GHz or higher (Intel i7/AMD Ryzen 7 or better)
- RAM: 32GB system memory
- Storage: 100GB+ NVMe SSD
- GPU: NVIDIA RTX 3070/4060 or better with 8GB+ VRAM

### Google Colab Cloud Deployment Strategy

Google Colab provides an excellent cloud-based alternative that eliminates hardware constraints and provides access to high-performance GPUs. Here's the optimized setup for Colab deployment:[39][5][6]

```python
# Google Colab Setup Cell
"""
Advanced Transcription System - Google Colab Setup
Run this cell first to configure the environment
"""

import sys
import subprocess
import os
from pathlib import Path

# Check if running in Colab
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    print("🔧 Setting up Google Colab environment...")
    
    # Install system dependencies
    !apt-get update -qq
    !apt-get install -y ffmpeg sox libsox-fmt-all
    
    # Upgrade pip and install core packages
    !pip install -q --upgrade pip
    
    # Install PyTorch with CUDA support
    !pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    # Install specialized ASR packages
    !pip install -q transformers>=4.35.0
    !pip install -q assemblyai>=0.17.0
    !pip install -q google-generativeai>=0.3.0
    !pip install -q yt-dlp>=2023.11.0
    !pip install -q pyannote.audio>=3.1.0
    !pip install -q librosa>=0.10.0
    !pip install -q soundfile>=0.12.0
    !pip install -q webvtt-py>=0.4.6
    !pip install -q google-auth google-auth-oauthlib google-api-python-client
    !pip install -q rich fire
    
    # Mount Google Drive for persistent storage
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Create project directory structure
    project_root = Path('/content/drive/MyDrive/AdvancedTranscription')
    project_root.mkdir(exist_ok=True)
    
    # Set up subdirectories
    (project_root / 'models').mkdir(exist_ok=True)
    (project_root / 'outputs').mkdir(exist_ok=True)
    (project_root / 'cache').mkdir(exist_ok=True)
    (project_root / 'logs').mkdir(exist_ok=True)
    
    os.chdir(str(project_root))
    print(f"✓ Project directory: {project_root}")
    
    # Verify GPU availability
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✓ GPU available: {gpu_name}")
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠ No GPU available - using CPU only")
    
    print("🚀 Colab environment ready!")

else:
    print("Running in local environment - Colab setup skipped")
```

#### Colab-Specific Optimizations

To maximize performance in the Colab environment, implement these optimization strategies:[40][41]

```python
# Memory and Performance Optimization for Colab
import gc
import torch

class ColabOptimizer:
    """Optimization utilities for Google Colab environment."""
    
    @staticmethod
    def clear_memory():
        """Clear GPU and system memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("✓ Memory cleared")
    
    @staticmethod
    def set_memory_efficient_settings():
        """Configure memory-efficient settings for large models."""
        if torch.cuda.is_available():
            # Enable memory efficient attention
            torch.backends.cuda.enable_flash_sdp(True)
            # Set memory fraction to prevent OOM
            torch.cuda.set_per_process_memory_fraction(0.9)
        
        # Configure environment variables for optimal performance
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
        
    @staticmethod
    def setup_logging():
        """Configure comprehensive logging for debugging."""
        import logging
        from rich.logging import RichHandler
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True)]
        )
        
        return logging.getLogger("transcription_system")

# Initialize optimizer
optimizer = ColabOptimizer()
optimizer.set_memory_efficient_settings()
logger = optimizer.setup_logging()
```

### Virtual Environment Management

For complex projects requiring multiple model environments, implement a sophisticated virtual environment strategy:

```python
import subprocess
import sys
from pathlib import Path
import json

class EnvironmentManager:
    """Advanced virtual environment management for transcription projects."""
    
    def __init__(self, base_path="./environments"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.config_file = self.base_path / "env_config.json"
        self.load_config()
    
    def load_config(self):
        """Load environment configuration."""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {"environments": {}}
    
    def save_config(self):
        """Save environment configuration."""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def create_specialized_env(self, env_name, model_type):
        """Create specialized environment for specific model types."""
        env_path = self.base_path / env_name
        
        if env_path.exists():
            print(f"Environment {env_name} already exists")
            return str(env_path)
        
        # Create virtual environment
        subprocess.run([sys.executable, "-m", "venv", str(env_path)], check=True)
        
        # Determine Python executable
        if sys.platform == "win32":
            python_exec = env_path / "Scripts" / "python.exe"
        else:
            python_exec = env_path / "bin" / "python"
        
        # Install model-specific dependencies
        model_deps = self.get_model_dependencies(model_type)
        
        # Upgrade pip
        subprocess.run([str(python_exec), "-m", "pip", "install", "--upgrade", "pip"], check=True)
        
        # Install dependencies
        for dep in model_deps:
            subprocess.run([str(python_exec), "-m", "pip", "install", dep], check=True)
        
        # Save configuration
        self.config["environments"][env_name] = {
            "model_type": model_type,
            "path": str(env_path),
            "python_executable": str(python_exec),
            "dependencies": model_deps
        }
        self.save_config()
        
        print(f"✓ Created specialized environment: {env_name}")
        return str(env_path)
    
    def get_model_dependencies(self, model_type):
        """Get dependencies for specific model types."""
        base_deps = [
            "torch>=2.0.0",
            "torchaudio>=2.0.0",
            "transformers>=4.35.0",
            "yt-dlp>=2023.11.0",
            "librosa>=0.10.0",
            "soundfile>=0.12.0"
        ]
        
        model_specific = {
            "assemblyai": ["assemblyai>=0.17.0"],
            "gemini": ["google-generativeai>=0.3.0", "google-auth>=2.20.0"],
            "firered": ["speechbrain>=0.5.15"],
            "pyannote": ["pyannote.audio>=3.1.0"],
            "emotion": ["scikit-learn>=1.3.0", "pandas>=2.0.0"],
            "youtube": ["google-api-python-client>=2.100.0", "google-auth-oauthlib>=1.0.0"]
        }
        
        return base_deps + model_specific.get(model_type, [])

# Usage example
env_manager = EnvironmentManager()

# Create specialized environments
env_manager.create_specialized_env("assemblyai_env", "assemblyai")
env_manager.create_specialized_env("gemini_env", "gemini") 
env_manager.create_specialized_env("pyannote_env", "pyannote")
```

### Configuration Management and Security

Implement robust configuration management that handles API keys and sensitive data securely:

```python
from pathlib import Path
import json
import os
from typing import Dict, Optional
import keyring
from cryptography.fernet import Fernet

class SecureConfigManager:
    """Secure configuration management for API keys and sensitive data."""
    
    def __init__(self, config_dir: str = ".config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.config_file = self.config_dir / "config.json"
        self.key_file = self.config_dir / ".encryption_key"
        
        self._ensure_encryption_key()
        self.load_config()
    
    def _ensure_encryption_key(self):
        """Generate or load encryption key for sensitive data."""
        if not self.key_file.exists():
            key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(key)
            # Hide the key file
            if os.name == 'nt':  # Windows
                os.system(f'attrib +h "{self.key_file}"')
        
        with open(self.key_file, 'rb') as f:
            self.encryption_key = f.read()
        
        self.cipher = Fernet(self.encryption_key)
    
    def load_config(self):
        """Load configuration from file."""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                "api_keys": {},
                "model_settings": {},
                "output_settings": {},
                "system_settings": {}
            }
    
    def save_config(self):
        """Save configuration to file."""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def set_api_key(self, service: str, api_key: str, use_keyring: bool = True):
        """Securely store API key."""
        if use_keyring:
            try:
                keyring.set_password("transcription_system", service, api_key)
                self.config["api_keys"][service] = {"stored_in_keyring": True}
                print(f"✓ API key for {service} stored in system keyring")
            except Exception as e:
                print(f"⚠ Keyring storage failed, using encrypted file: {e}")
                use_keyring = False
        
        if not use_keyring:
            encrypted_key = self.cipher.encrypt(api_key.encode()).decode()
            self.config["api_keys"][service] = {"encrypted_key": encrypted_key}
            print(f"✓ API key for {service} stored encrypted")
        
        self.save_config()
    
    def get_api_key(self, service: str) -> Optional[str]:
        """Retrieve API key securely."""
        if service not in self.config["api_keys"]:
            return None
        
        key_config = self.config["api_keys"][service]
        
        if key_config.get("stored_in_keyring"):
            try:
                return keyring.get_password("transcription_system", service)
            except Exception:
                print(f"⚠ Failed to retrieve {service} key from keyring")
                return None
        
        if "encrypted_key" in key_config:
            try:
                encrypted_key = key_config["encrypted_key"].encode()
                return self.cipher.decrypt(encrypted_key).decode()
            except Exception:
                print(f"⚠ Failed to decrypt {service} key")
                return None
        
        return None
    
    def configure_all_services(self):
        """Interactive configuration of all required services."""
        services = {
            "assemblyai": "AssemblyAI API Key",
            "google_ai": "Google AI API Key", 
            "youtube": "YouTube Data API Key",
            "huggingface": "HuggingFace Access Token"
        }
        
        print("🔧 Configuring API keys for transcription services...")
        
        for service, description in services.items():
            current_key = self.get_api_key(service)
            if current_key:
                print(f"✓ {description} already configured")
                continue
            
            key = input(f"Enter {description}: ").strip()
            if key:
                self.set_api_key(service, key)
            else:
                print(f"⚠ Skipped {description}")

# Example usage
config = SecureConfigManager()
config.configure_all_services()
```

This foundational architecture provides a robust, scalable environment capable of supporting the advanced transcription models and features detailed in subsequent chapters. The combination of local and cloud deployment options ensures flexibility while maintaining security and performance standards essential for production systems.

***

## Chapter 3: Advanced Audio Processing with yt-dlp

### Introduction to Modern Video Processing

The foundation of any advanced transcription system begins with robust audio extraction and processing capabilities. yt-dlp has emerged as the successor to youtube-dl, offering superior performance, broader platform support, and enhanced audio extraction features. This chapter details implementation of a comprehensive audio processing pipeline that optimizes quality while maintaining efficiency.[9][10][8]

### Core yt-dlp Implementation

#### Advanced Audio Extraction Configuration

Modern transcription systems require high-quality audio input to achieve optimal results. The following implementation demonstrates a sophisticated yt-dlp configuration optimized for transcription workflows:

```python
#!/usr/bin/env python3
"""
Advanced Audio Extraction System using yt-dlp
Optimized for next-generation transcription workflows
"""

import yt_dlp
import os
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import tempfile
import hashlib

console = Console()
logger = logging.getLogger(__name__)

class AdvancedAudioExtractor:
    """
    High-performance audio extraction system with quality optimization
    and robust error handling for transcription workflows.
    """
    
    def __init__(self, output_dir: str = "./extracted_audio", cache_dir: str = "./cache"):
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Configure yt-dlp options for optimal audio quality
        self.base_options = {
            'format': 'bestaudio[ext=webm]/bestaudio[ext=m4a]/bestaudio',
            'outtmpl': str(self.output_dir / '%(title)s.%(ext)s'),
            'extractaudio': True,
            'audioformat': 'wav',  # Uncompressed for best transcription quality
            'audioquality': '0',   # Best quality
            'embed_metadata': True,
            'writeinfojson': True,
            'writedescription': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en', 'en-US', 'en-GB'],
            'ignoreerrors': False,
            'no_warnings': False,
            'extractflat': False,
            'writethumbnail': True,
            'postprocessors': [
                {
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '0',
                },
                {
                    'key': 'FFmpegMetadata',
                    'add_metadata': True,
                },
                {
                    'key': 'EmbedThumbnail',
                    'already_have_thumbnail': False,
                }
            ],
            'postprocessor_args': {
                'ffmpeg': [
                    '-ac', '1',        # Convert to mono
                    '-ar', '16000',    # 16kHz sample rate (optimal for most ASR models)
                    '-acodec', 'pcm_s16le',  # 16-bit PCM encoding
                    '-af', 'volume=0.95'      # Slight volume normalization
                ]
            }
        }
    
    def get_video_info(self, url: str) -> Dict:
        """Extract comprehensive video metadata without downloading."""
        options = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'dump_single_json': True
        }
        
        with yt_dlp.YoutubeDL(options) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
                return self._process_video_info(info)
            except Exception as e:
                logger.error(f"Failed to extract video info: {e}")
                raise
    
    def _process_video_info(self, info: Dict) -> Dict:
        """Process and clean video information."""
        processed_info = {
            'id': info.get('id', 'unknown'),
            'title': info.get('title', 'Unknown Title'),
            'duration': info.get('duration', 0),
            'uploader': info.get('uploader', 'Unknown'),
            'upload_date': info.get('upload_date', 'Unknown'),
            'view_count': info.get('view_count', 0),
            'like_count': info.get('like_count', 0),
            'description': info.get('description', ''),
            'categories': info.get('categories', []),
            'tags': info.get('tags', []),
            'language': info.get('language', 'en'),
            'automatic_captions': info.get('automatic_captions', {}),
            'subtitles': info.get('subtitles', {}),
            'formats': []
        }
        
        # Extract audio format information
        for fmt in info.get('formats', []):
            if fmt.get('acodec') != 'none':  # Has audio
                processed_info['formats'].append({
                    'format_id': fmt.get('format_id'),
                    'ext': fmt.get('ext'),
                    'acodec': fmt.get('acodec'),
                    'abr': fmt.get('abr'),
                    'asr': fmt.get('asr'),
                    'filesize': fmt.get('filesize'),
                    'quality': fmt.get('quality')
                })
        
        return processed_info
    
    def extract_audio_single(self, url: str, custom_options: Dict = None) -> Tuple[str, Dict]:
        """Extract audio from a single video with enhanced error handling."""
        
        # Check cache first
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_file = self.cache_dir / f"{url_hash}_info.json"
        
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                cached_info = json.load(f)
                audio_path = Path(cached_info['audio_path'])
                if audio_path.exists():
                    console.print(f"[green]✓[/green] Using cached audio: {audio_path.name}")
                    return str(audio_path), cached_info
        
        # Merge custom options with base options
        options = self.base_options.copy()
        if custom_options:
            options.update(custom_options)
        
        # Add progress hook
        def progress_hook(d):
            if d['status'] == 'downloading':
                if 'total_bytes' in d:
                    progress = (d['downloaded_bytes'] / d['total_bytes']) * 100
                    console.print(f"\rDownloading: {progress:.1f}%", end="")
                elif 'total_bytes_estimate' in d:
                    progress = (d['downloaded_bytes'] / d['total_bytes_estimate']) * 100
                    console.print(f"\rDownloading: {progress:.1f}% (estimated)", end="")
            elif d['status'] == 'finished':
                console.print(f"\n[green]✓[/green] Download completed: {d['filename']}")
        
        options['progress_hooks'] = [progress_hook]
        
        # Extract audio
        with yt_dlp.YoutubeDL(options) as ydl:
            try:
                console.print(f"[blue]ℹ[/blue] Extracting audio from: {url}")
                
                # Get info first
                info = ydl.extract_info(url, download=False)
                processed_info = self._process_video_info(info)
                
                # Download and extract
                info = ydl.extract_info(url, download=True)
                
                # Find the extracted audio file
                audio_path = self._find_audio_file(info)
                
                if audio_path and Path(audio_path).exists():
                    # Verify audio quality
                    verified_path = self._verify_audio_quality(audio_path)
                    
                    # Cache the result
                    cache_data = {
                        'url': url,
                        'audio_path': verified_path,
                        'info': processed_info,
                        'extraction_time': str(pd.Timestamp.now())
                    }
                    
                    with open(cache_file, 'w') as f:
                        json.dump(cache_data, f, indent=2)
                    
                    console.print(f"[green]✓[/green] Audio extracted successfully: {Path(verified_path).name}")
                    return verified_path, processed_info
                else:
                    raise FileNotFoundError("Audio file not found after extraction")
                    
            except Exception as e:
                logger.error(f"Failed to extract audio from {url}: {e}")
                raise
    
    def _find_audio_file(self, info: Dict) -> Optional[str]:
        """Find the extracted audio file based on yt-dlp info."""
        # yt-dlp provides the filename in the info
        if 'requested_downloads' in info:
            for download in info['requested_downloads']:
                filepath = download.get('filepath')
                if filepath and Path(filepath).exists():
                    return filepath
        
        # Fallback: search in output directory
        title = info.get('title', 'unknown')
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
        
        for ext in ['.wav', '.m4a', '.webm', '.mp3']:
            potential_path = self.output_dir / f"{safe_title}{ext}"
            if potential_path.exists():
                return str(potential_path)
        
        return None
    
    def _verify_audio_quality(self, audio_path: str) -> str:
        """Verify and optimize audio quality using FFmpeg."""
        input_path = Path(audio_path)
        
        # Run ffprobe to check audio properties
        try:
            probe_cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', str(input_path)
            ]
            
            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            probe_data = json.loads(result.stdout)
            
            audio_stream = None
            for stream in probe_data.get('streams', []):
                if stream.get('codec_type') == 'audio':
                    audio_stream = stream
                    break
            
            if not audio_stream:
                raise ValueError("No audio stream found in file")
            
            # Check if optimization is needed
            sample_rate = int(audio_stream.get('sample_rate', 0))
            channels = int(audio_stream.get('channels', 0))
            codec = audio_stream.get('codec_name', '')
            
            needs_optimization = (
                sample_rate != 16000 or 
                channels != 1 or 
                codec not in ['pcm_s16le', 'pcm_s16be']
            )
            
            if needs_optimization:
                console.print(f"[yellow]⚠[/yellow] Optimizing audio quality...")
                optimized_path = input_path.with_suffix('.optimized.wav')
                
                optimize_cmd = [
                    'ffmpeg', '-i', str(input_path),
                    '-ac', '1',  # Mono
                    '-ar', '16000',  # 16kHz
                    '-acodec', 'pcm_s16le',  # 16-bit PCM
                    '-af', 'volume=0.95,highpass=f=80,lowpass=f=8000',  # Audio filtering
                    '-y',  # Overwrite output
                    str(optimized_path)
                ]
                
                subprocess.run(optimize_cmd, check=True, capture_output=True)
                
                # Replace original with optimized version
                input_path.unlink()
                optimized_path.rename(input_path)
                
                console.print(f"[green]✓[/green] Audio optimized for transcription")
            
            return str(input_path)
            
        except Exception as e:
            logger.warning(f"Audio verification failed: {e}")
            return str(input_path)  # Return original path if verification fails
    
    def extract_batch(self, urls: List[str], max_workers: int = 3) -> List[Tuple[str, str, Dict]]:
        """Extract audio from multiple URLs with parallel processing."""
        import concurrent.futures
        
        results = []
        failed_urls = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            task = progress.add_task("Extracting audio files...", total=len(urls))
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_url = {
                    executor.submit(self.extract_audio_single, url): url 
                    for url in urls
                }
                
                for future in concurrent.futures.as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        audio_path, info = future.result()
                        results.append((url, audio_path, info))
                        progress.console.print(f"[green]✓[/green] Completed: {info['title'][:50]}...")
                    except Exception as e:
                        failed_urls.append((url, str(e)))
                        progress.console.print(f"[red]✗[/red] Failed: {url} - {e}")
                    
                    progress.advance(task)
        
        # Report results
        console.print(f"\n[green]✓[/green] Successfully extracted: {len(results)} files")
        if failed_urls:
            console.print(f"[red]✗[/red] Failed extractions: {len(failed_urls)}")
            for url, error in failed_urls:
                console.print(f"  {url}: {error}")
        
        return results
    
    def extract_playlist(self, playlist_url: str, max_videos: int = None) -> List[Tuple[str, str, Dict]]:
        """Extract audio from entire playlists with smart filtering."""
        
        # Get playlist info
        playlist_options = {
            'quiet': True,
            'extract_flat': True,
            'playlistend': max_videos
        }
        
        with yt_dlp.YoutubeDL(playlist_options) as ydl:
            playlist_info = ydl.extract_info(playlist_url, download=False)
        
        if 'entries' not in playlist_info:
            raise ValueError("Invalid playlist URL or empty playlist")
        
        # Extract URLs from playlist
        video_urls = []
        for entry in playlist_info['entries']:
            if entry and entry.get('url'):
                video_urls.append(entry['url'])
        
        console.print(f"[blue]ℹ[/blue] Found {len(video_urls)} videos in playlist")
        
        # Filter videos by duration (optional)
        if hasattr(self, 'max_duration_minutes'):
            filtered_urls = []
            for url in video_urls:
                try:
                    info = self.get_video_info(url)
                    duration_minutes = info.get('duration', 0) / 60
                    if duration_minutes  str:
        """Optimize audio specifically for different ASR models."""
        
        model_configs = {
            'whisper': {'sr': 16000, 'normalize': True, 'trim_silence': True},
            'assemblyai': {'sr': 16000, 'normalize': True, 'noise_reduction': True},
            'gemini': {'sr': 16000, 'enhance_speech': True, 'dynamic_range': True},
            'fireredasr': {'sr': 16000, 'mandarin_optimize': True},
            'pyannote': {'sr': 16000, 'speaker_enhance': True}
        }
        
        config = model_configs.get(model_type, model_configs['whisper'])
        
        # Load audio
        audio, original_sr = librosa.load(audio_path, sr=None, mono=False)
        
        # Convert to mono if needed
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)
        
        # Resample to target sample rate
        if original_sr != config['sr']:
            audio = librosa.resample(audio, orig_sr=original_sr, target_sr=config['sr'])
        
        # Apply model-specific optimizations
        if config.get('normalize'):
            audio = self._normalize_audio(audio)
        
        if config.get('trim_silence'):
            audio = self._trim_silence(audio, config['sr'])
        
        if config.get('noise_reduction'):
            audio = self._reduce_noise(audio, config['sr'])
        
        if config.get('enhance_speech'):
            audio = self._enhance_speech(audio, config['sr'])
        
        if config.get('dynamic_range'):
            audio = self._optimize_dynamic_range(audio)
        
        if config.get('speaker_enhance'):
            audio = self._enhance_for_diarization(audio, config['sr'])
        
        # Save optimized audio
        output_path = Path(audio_path).with_suffix(f'.{model_type}_optimized.wav')
        sf.write(output_path, audio, config['sr'], subtype='PCM_16')
        
        return str(output_path)
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to optimal levels."""
        # RMS normalization
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            target_rms = 0.2  # Target RMS level
            audio = audio * (target_rms / rms)
        
        # Peak normalization to prevent clipping
        peak = np.max(np.abs(audio))
        if peak > 0.95:
            audio = audio * (0.95 / peak)
        
        return audio
    
    def _trim_silence(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Remove silence from beginning and end."""
        # Use librosa's trim function with adaptive threshold
        audio_trimmed, _ = librosa.effects.trim(
            audio, 
            top_db=30,  # Threshold for silence detection
            frame_length=2048,
            hop_length=512
        )
        return audio_trimmed
    
    def _reduce_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply noise reduction using spectral gating."""
        # Simple spectral subtraction for noise reduction
        # For production, consider using more sophisticated methods
        
        # Compute STFT
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise from the first 0.5 seconds
        noise_frame_count = int(0.5 * sr / 512)  # 512 is hop_length
        noise_spectrum = np.mean(magnitude[:, :noise_frame_count], axis=1, keepdims=True)
        
        # Apply spectral subtraction
        alpha = 2.0  # Over-subtraction factor
        beta = 0.01  # Spectral floor
        
        clean_magnitude = magnitude - alpha * noise_spectrum
        clean_magnitude = np.maximum(clean_magnitude, beta * magnitude)
        
        # Reconstruct audio
        clean_stft = clean_magnitude * np.exp(1j * phase)
        clean_audio = librosa.istft(clean_stft)
        
        return clean_audio
    
    def _enhance_speech(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Enhance speech characteristics."""
        # Apply high-pass filter to remove low-frequency noise
        nyquist = sr / 2
        low_cutoff = 80 / nyquist
        b, a = signal.butter(5, low_cutoff, btype='high')
        audio = signal.filtfilt(b, a, audio)
        
        # Apply gentle compression to even out dynamic range
        threshold = 0.3
        ratio = 4.0
        
        audio_abs = np.abs(audio)
        compressed = np.where(
            audio_abs > threshold,
            np.sign(audio) * (threshold + (audio_abs - threshold) / ratio),
            audio
        )
        
        return compressed
    
    def _optimize_dynamic_range(self, audio: np.ndarray) -> np.ndarray:
        """Optimize dynamic range for better transcription."""
        # Adaptive gain control
        window_size = 1024
        hop_size = 512
        
        # Calculate RMS in sliding windows
        padding = window_size // 2
        padded_audio = np.pad(audio, padding, mode='reflect')
        
        gains = []
        for i in range(0, len(audio), hop_size):
            window = padded_audio[i:i + window_size]
            rms = np.sqrt(np.mean(window**2))
            target_rms = 0.15
            gain = target_rms / (rms + 1e-8)
            gain = np.clip(gain, 0.5, 2.0)  # Limit gain range
            gains.append(gain)
        
        # Apply gains smoothly
        gain_curve = np.interp(
            np.arange(len(audio)), 
            np.arange(0, len(audio), hop_size)[:len(gains)], 
            gains
        )
        
        return audio * gain_curve
    
    def _enhance_for_diarization(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Enhance audio specifically for speaker diarization."""
        # Emphasize mid-frequency range where speaker characteristics are prominent
        # Apply band-pass filter for speech frequency range (300-3400 Hz)
        nyquist = sr / 2
        low_cutoff = 300 / nyquist
        high_cutoff = 3400 / nyquist
        
        b, a = signal.butter(4, [low_cutoff, high_cutoff], btype='band')
        audio = signal.filtfilt(b, a, audio)
        
        return audio
```

### Error Handling and Recovery

Robust error handling is crucial for production transcription systems. The following implementation provides comprehensive error recovery mechanisms:

```python
import time
import random
from typing import Callable, Any
from functools import wraps

class ExtractionError(Exception):
    """Custom exception for audio extraction errors."""
    pass

class AudioExtractionRecovery:
    """Advanced error handling and recovery for audio extraction."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.error_patterns = {
            'network': ['HTTP Error', 'URLError', 'ConnectionError', 'timeout'],
            'format': ['No video formats found', 'Unsupported URL', 'Private video'],
            'quota': ['quota exceeded', 'rate limit', 'Too Many Requests'],
            'geo': ['not available in your country', 'blocked in your country'],
            'age': ['age-restricted', 'Sign in to confirm your age']
        }
    
    def retry_with_backoff(self, func: Callable) -> Callable:
        """Decorator for automatic retry with exponential backoff."""
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(self.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_type = self._classify_error(str(e))
                    
                    if attempt == self.max_retries:
                        logger.error(f"Max retries exceeded for {func.__name__}: {e}")
                        break
                    
                    if error_type in ['format', 'geo', 'age']:
                        # Don't retry these errors
                        logger.error(f"Non-recoverable error in {func.__name__}: {e}")
                        break
                    
                    delay = self._calculate_delay(attempt, error_type)
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.1f}s: {e}")
                    time.sleep(delay)
            
            raise ExtractionError(f"Failed after {self.max_retries} retries: {last_exception}")
        
        return wrapper
    
    def _classify_error(self, error_message: str) -> str:
        """Classify error type based on message."""
        error_message_lower = error_message.lower()
        
        for error_type, patterns in self.error_patterns.items():
            if any(pattern.lower() in error_message_lower for pattern in patterns):
                return error_type
        
        return 'unknown'
    
    def _calculate_delay(self, attempt: int, error_type: str) -> float:
        """Calculate delay based on attempt number and error type."""
        base_delays = {
            'network': self.base_delay,
            'quota': self.base_delay * 5,  # Longer delay for quota issues
            'unknown': self.base_delay
        }
        
        base = base_delays.get(error_type, self.base_delay)
        # Exponential backoff with jitter
        delay = base * (2 ** attempt) + random.uniform(0, 1)
        
        return min(delay, 60)  # Cap at 60 seconds

    def create_fallback_extractor(self) -> AdvancedAudioExtractor:
        """Create fallback extractor with different settings."""
        fallback_extractor = AdvancedAudioExtractor()
        
        # Use more conservative settings for fallback
        fallback_extractor.base_options.update({
            'format': 'worst[ext=webm]/worst[ext=m4a]/worst',  # Lower quality but more reliable
            'audioquality': '5',  # Lower quality
            'ignoreerrors': True,
            'no_warnings': True,
            'retries': 10,
            'socket_timeout': 30
        })
        
        return fallback_extractor
```

This comprehensive audio processing system provides the foundation for high-quality transcription workflows. The combination of yt-dlp's robust extraction capabilities with intelligent optimization and error handling ensures reliable audio processing for the advanced transcription models detailed in subsequent chapters.

***

## Chapter 4: Next-Generation Speech Recognition Implementation

### Advanced ASR Model Integration

Building upon the foundation established in previous chapters, this chapter implements cutting-edge speech recognition models that significantly outperform traditional solutions like Whisper. The focus is on practical implementation of FireRedASR, AssemblyAI Universal-2, and Google Gemini Flash APIs, with comprehensive fallback strategies and performance optimization.[12][2][3][11]

### AssemblyAI Universal-2 Implementation

AssemblyAI Universal-2 represents the current state-of-the-art in commercial speech recognition, achieving 93.3% word accuracy with built-in speaker diarization capabilities. The following implementation demonstrates production-ready integration:[4][3]

```python
#!/usr/bin/env python3
"""
AssemblyAI Universal-2 Integration
Next-generation speech recognition with advanced features
"""

import assemblyai as aai
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable
from dataclasses import dataclass
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import asyncio
import aiohttp
from datetime import datetime, timedelta

console = Console()
logger = logging.getLogger(__name__)

@dataclass
class TranscriptionConfig:
    """Configuration for AssemblyAI transcription requests."""
    
    # Core transcription settings
    language_code: str = "en"
    language_detection: bool = True
    punctuate: bool = True
    format_text: bool = True
    
    # Speaker diarization
    speaker_labels: bool = True
    speakers_expected: Optional[int] = None
    
    # Audio intelligence features
    sentiment_analysis: bool = True
    entity_detection: bool = True
    iab_categories: bool = True
    content_safety: bool = True
    topic_detection: bool = True
    
    # Advanced features
    summarization: bool = True
    auto_chapters: bool = True
    key_phrases: bool = True
    
    # Performance settings
    dual_channel: bool = False
    webhook_url: Optional[str] = None
    word_boost: List[str] = None
    boost_param: str = "default"  # low, default, high
    
    # Quality settings
    filter_profanity: bool = False
    redact_pii: bool = False
    redact_pii_audio: bool = False
    redact_pii_policies: List[str] = None

class AssemblyAITranscriber:
    """Advanced AssemblyAI Universal-2 transcription system."""
    
    def __init__(self, api_key: str, config: TranscriptionConfig = None):
        aai.settings.api_key = api_key
        self.config = config or TranscriptionConfig()
        self.client = aai.Client()
        
        # Rate limiting
        self.requests_per_minute = 100
        self.request_times = []
        
        # Results cache
        self.results_cache = {}
    
    def _check_rate_limit(self):
        """Implement rate limiting to respect API limits."""
        now = datetime.now()
        
        # Remove requests older than 1 minute
        self.request_times = [
            req_time for req_time in self.request_times 
            if now - req_time = self.requests_per_minute:
            sleep_time = 60 - (now - self.request_times).seconds
            if sleep_time > 0:
                console.print(f"[yellow]⚠[/yellow] Rate limit reached, waiting {sleep_time}s...")
                time.sleep(sleep_time)
        
        self.request_times.append(now)
    
    def transcribe_file(self, audio_path: str, config: TranscriptionConfig = None) -> Dict:
        """Transcribe a single audio file with comprehensive features."""
        
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Use provided config or instance config
        transcription_config = config or self.config
        
        # Check cache first
        cache_key = self._generate_cache_key(audio_path, transcription_config)
        if cache_key in self.results_cache:
            console.print("[green]✓[/green] Using cached transcription result")
            return self.results_cache[cache_key]
        
        self._check_rate_limit()
        
        console.print(f"[blue]ℹ[/blue] Starting transcription: {Path(audio_path).name}")
        
        # Configure transcription request
        transcript_config = aai.TranscriptionConfig(
            language_code=transcription_config.language_code if not transcription_config.language_detection else None,
            language_detection=transcription_config.language_detection,
            punctuate=transcription_config.punctuate,
            format_text=transcription_config.format_text,
            speaker_labels=transcription_config.speaker_labels,
            speakers_expected=transcription_config.speakers_expected,
            sentiment_analysis=transcription_config.sentiment_analysis,
            entity_detection=transcription_config.entity_detection,
            iab_categories=transcription_config.iab_categories,
            content_safety_labels=transcription_config.content_safety,
            topic_detection=transcription_config.topic_detection,
            summarization=transcription_config.summarization,
            auto_chapters=transcription_config.auto_chapters,
            key_phrases=transcription_config.key_phrases,
            dual_channel=transcription_config.dual_channel,
            webhook_url=transcription_config.webhook_url,
            word_boost=transcription_config.word_boost,
            boost_param=transcription_config.boost_param,
            filter_profanity=transcription_config.filter_profanity,
            redact_pii=transcription_config.redact_pii,
            redact_pii_audio=transcription_config.redact_pii_audio,
            redact_pii_policies=transcription_config.redact_pii_policies
        )
        
        try:
            # Submit transcription
            transcriber = aai.Transcriber(config=transcript_config)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Transcribing audio...", total=None)
                
                transcript = transcriber.transcribe(audio_path)
                
                # Wait for completion with progress updates
                while transcript.status in [aai.TranscriptStatus.queued, aai.TranscriptStatus.processing]:
                    time.sleep(5)
                    transcript = transcriber.get_transcript(transcript.id)
                    progress.update(task, description=f"Status: {transcript.status}")
            
            if transcript.status == aai.TranscriptStatus.error:
                raise Exception(f"Transcription failed: {transcript.error}")
            
            # Process results
            result = self._process_transcript_result(transcript, transcription_config)
            
            # Cache result
            self.results_cache[cache_key] = result
            
            console.print(f"[green]✓[/green] Transcription completed: {len(result['text'])} characters")
            
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed for {audio_path}: {e}")
            raise
    
    def _process_transcript_result(self, transcript: aai.Transcript, config: TranscriptionConfig) -> Dict:
        """Process and structure transcript results."""
        
        result = {
            'transcript_id': transcript.id,
            'text': transcript.text,
            'confidence': transcript.confidence,
            'audio_duration': transcript.audio_duration,
            'language_code': transcript.language_code,
            'status': str(transcript.status),
            'created': transcript.created.isoformat() if transcript.created else None,
            'completed': transcript.completed.isoformat() if transcript.completed else None,
            'words': []
        }
        
        # Process word-level data
        if transcript.words:
            for word in transcript.words:
                word_data = {
                    'text': word.text,
                    'start': word.start,
                    'end': word.end,
                    'confidence': word.confidence
                }
                
                # Add speaker information if available
                if hasattr(word, 'speaker') and word.speaker:
                    word_data['speaker'] = word.speaker
                
                result['words'].append(word_data)
        
        # Process speaker diarization
        if config.speaker_labels and transcript.utterances:
            result['utterances'] = []
            for utterance in transcript.utterances:
                result['utterances'].append({
                    'text': utterance.text,
                    'start': utterance.start,
                    'end': utterance.end,
                    'confidence': utterance.confidence,
                    'speaker': utterance.speaker,
                    'words': [
                        {
                            'text': word.text,
                            'start': word.start,
                            'end': word.end,
                            'confidence': word.confidence
                        }
                        for word in utterance.words
                    ]
                })
        
        # Process sentiment analysis
        if config.sentiment_analysis and transcript.sentiment_analysis_results:
            result['sentiment_analysis'] = []
            for sentiment in transcript.sentiment_analysis_results:
                result['sentiment_analysis'].append({
                    'text': sentiment.text,
                    'start': sentiment.start,
                    'end': sentiment.end,
                    'sentiment': sentiment.sentiment.value,
                    'confidence': sentiment.confidence
                })
        
        # Process entity detection
        if config.entity_detection and transcript.entities:
            result['entities'] = []
            for entity in transcript.entities:
                result['entities'].append({
                    'text': entity.text,
                    'start': entity.start,
                    'end': entity.end,
                    'entity_type': entity.entity_type.value,
                    'confidence': entity.confidence
                })
        
        # Process topic detection
        if config.topic_detection and transcript.iab_categories_result:
            result['topics'] = {
                'summary': transcript.iab_categories_result.summary,
                'labels': [
                    {
                        'label': label.label,
                        'confidence': label.confidence,
                        'relevance': label.relevance
                    }
                    for label in transcript.iab_categories_result.labels
                ]
            }
        
        # Process auto chapters
        if config.auto_chapters and transcript.chapters:
            result['chapters'] = []
            for chapter in transcript.chapters:
                result['chapters'].append({
                    'gist': chapter.gist,
                    'headline': chapter.headline,
                    'summary': chapter.summary,
                    'start': chapter.start,
                    'end': chapter.end
                })
        
        # Process key phrases
        if config.key_phrases and transcript.key_phrases:
            result['key_phrases'] = [
                {
                    'text': phrase.text,
                    'count': phrase.count,
                    'rank': phrase.rank,
                    'timestamps': [
                        {'start': ts.start, 'end': ts.end}
                        for ts in phrase.timestamps
                    ]
                }
                for phrase in transcript.key_phrases
            ]
        
        # Process summarization
        if config.summarization and transcript.summary:
            result['summary'] = transcript.summary
        
        return result
    
    def _generate_cache_key(self, audio_path: str, config: TranscriptionConfig) -> str:
        """Generate cache key for transcription results."""
        import hashlib
        
        # Get file modification time and size for cache invalidation
        path = Path(audio_path)
        file_stats = f"{path.stat().st_mtime}_{path.stat().st_size}"
        
        # Create config hash
        config_str = json.dumps(config.__dict__, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()
        
        return f"{path.stem}_{file_stats}_{config_hash}"
    
    def transcribe_batch(self, audio_files: List[str], max_concurrent: int = 5) -> List[Dict]:
        """Transcribe multiple files with concurrent processing."""
        
        results = []
        failed_files = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            task = progress.add_task("Batch transcription...", total=len(audio_files))
            
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                future_to_file = {
                    executor.submit(self.transcribe_file, audio_file): audio_file
                    for audio_file in audio_files
                }
                
                for future in concurrent.futures.as_completed(future_to_file):
                    audio_file = future_to_file[future]
                    try:
                        result = future.result()
                        result['source_file'] = audio_file
                        results.append(result)
                        progress.console.print(f"[green]✓[/green] Completed: {Path(audio_file).name}")
                    except Exception as e:
                        failed_files.append((audio_file, str(e)))
                        progress.console.print(f"[red]✗[/red] Failed: {Path(audio_file).name} - {e}")
                    
                    progress.advance(task)
        
        # Report results
        console.print(f"\n[green]✓[/green] Successfully transcribed: {len(results)} files")
        if failed_files:
            console.print(f"[red]✗[/red] Failed transcriptions: {len(failed_files)}")
        
        return results
    
    def export_results(self, results: List[Dict], output_dir: str, formats: List[str] = None):
        """Export transcription results in multiple formats."""
        
        if formats is None:
            formats = ['json', 'txt', 'srt', 'vtt']
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for i, result in enumerate(results):
            base_name = f"transcription_{i:03d}"
            
            if 'json' in formats:
                json_path = output_path / f"{base_name}.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
            
            if 'txt' in formats:
                txt_path = output_path / f"{base_name}.txt"
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(result['text'])
            
            if 'srt' in formats and result.get('words'):
                srt_path = output_path / f"{base_name}.srt"
                self._export_srt(result, srt_path)
            
            if 'vtt' in formats and result.get('words'):
                vtt_path = output_path / f"{base_name}.vtt"
                self._export_vtt(result, vtt_path)
        
        console.print(f"[green]✓[/green] Results exported to: {output_path}")
    
    def _export_srt(self, result: Dict, output_path: Path):
        """Export results in SRT subtitle format."""
        
        words = result.get('words', [])
        if not words:
            return
        
        # Group words into subtitle segments (max 10 words or 5 seconds)
        segments = []
        current_segment = []
        segment_start = words[0]['start']
        
        for word in words:
            current_segment.append(word)
            
            # Check if we should end this segment
            should_end = (
                len(current_segment) >= 10 or  # Max 10 words
                word['end'] - segment_start >= 5000 or  # Max 5 seconds
                word == words[-1]  # Last word
            )
            
            if should_end:
                segments.append({
                    'start': segment_start,
                    'end': word['end'],
                    'text': ' '.join([w['text'] for w in current_segment])
                })
                current_segment = []
                if word != words[-1]:
                    segment_start = words[words.index(word) + 1]['start']
        
        # Write SRT file
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, 1):
                start_time = self._format_srt_time(segment['start'])
                end_time = self._format_srt_time(segment['end'])
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment['text']}\n\n")
    
    def _export_vtt(self, result: Dict, output_path: Path):
        """Export results in WebVTT subtitle format."""
        
        words = result.get('words', [])
        if not words:
            return
        
        # Group words into subtitle segments
        segments = []
        current_segment = []
        segment_start = words['start']
        
        for word in words:
            current_segment.append(word)
            
            should_end = (
                len(current_segment) >= 10 or
                word['end'] - segment_start >= 5000 or
                word == words[-1]
            )
            
            if should_end:
                segments.append({
                    'start': segment_start,
                    'end': word['end'],
                    'text': ' '.join([w['text'] for w in current_segment])
                })
                current_segment = []
                if word != words[-1]:
                    segment_start = words[words.index(word) + 1]['start']
        
        # Write VTT file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            
            for segment in segments:
                start_time = self._format_vtt_time(segment['start'])
                end_time = self._format_vtt_time(segment['end'])
                
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment['text']}\n\n")
    
    def _format_srt_time(self, milliseconds: int) -> str:
        """Format time for SRT format (HH:MM:SS,mmm)."""
        seconds = milliseconds // 1000
        ms = milliseconds % 1000
        
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"
    
    def _format_vtt_time(self, milliseconds: int) -> str:
        """Format time for VTT format (MM:SS.mmm)."""
        seconds = milliseconds // 1000
        ms = milliseconds % 1000
        
        minutes = seconds // 60
        seconds = seconds % 60
        
        return f"{minutes:02d}:{seconds:02d}.{ms:03d}"

# Example usage
if __name__ == "__main__":
    # Initialize transcriber
    config = TranscriptionConfig(
        speaker_labels=True,
        sentiment_analysis=True,
        summarization=True,
        key_phrases=True
    )
    
    transcriber = AssemblyAITranscriber(
        api_key="your-assemblyai-api-key",
        config=config
    )
    
    # Test transcription
    audio_file = "path/to/your/audio.wav"
    result = transcriber.transcribe_file(audio_file)
    
    console.print(f"Transcript: {result['text'][:200]}...")
    console.print(f"Confidence: {result['confidence']:.2%}")
    console.print(f"Speakers detected: {len(set(u['speaker'] for u in result.get('utterances', [])))}")
```

### Google Gemini Flash Integration

Google Gemini Flash offers state-of-the-art multimodal capabilities with native audio processing. The following implementation demonstrates advanced integration:[42][32][12]

```python
import google.generativeai as genai
import asyncio
import aiofiles
from typing import AsyncGenerator
import base64

class GeminiFlashTranscriber:
    """Google Gemini Flash transcription with multimodal capabilities."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # Configuration for optimal transcription
        self.generation_config = genai.types.GenerationConfig(
            temperature=0.1,  # Low temperature for consistent transcription
            top_p=0.9,
            top_k=40,
            max_output_tokens=8192,
            candidate_count=1
        )
        
        # Safety settings for transcription content
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH", 
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
    
    async def transcribe_with_context(
        self, 
        audio_path: str, 
        context: str = "", 
        language: str = "auto",
        include_timestamps: bool = True,
        include_emotions: bool = False,
        include_summary: bool = False
    ) -> Dict:
        """Transcribe audio with contextual understanding and advanced features."""
        
        # Prepare the prompt with specific instructions
        prompt_parts = [
            "You are an expert transcriptionist with perfect accuracy. "
            "Please transcribe the following audio with the highest possible accuracy. "
        ]
        
        if include_timestamps:
            prompt_parts.append(
                "Include precise timestamps in the format [MM:SS] for significant segments. "
            )
        
        if include_emotions:
            prompt_parts.append(
                "Also detect and note the emotional tone of the speaker (e.g., excited, calm, frustrated). "
            )
        
        if language != "auto":
            prompt_parts.append(f"The audio is in {language}. ")
        
        if context:
            prompt_parts.append(f"Context: {context} ")
        
        if include_summary:
            prompt_parts.append(
                "After the transcription, provide a brief summary of the main topics discussed. "
            )
        
        prompt_parts.append(
            "\nPlease structure your response as follows:\n"
            "TRANSCRIPTION:\n[Full transcription here]\n\n"
        )
        
        if include_emotions:
            prompt_parts.append("EMOTIONAL_ANALYSIS:\n[Emotional analysis here]\n\n")
        
        if include_summary:
            prompt_parts.append("SUMMARY:\n[Summary here]")
        
        prompt = "".join(prompt_parts)
        
        try:
            # Upload audio file
            audio_file = genai.upload_file(audio_path)
            
            # Wait for processing
            await asyncio.sleep(2)
            
            # Generate transcription
            response = await self._generate_async(
                contents=[prompt, audio_file],
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            
            # Process response
            result = self._parse_gemini_response(response.text, audio_path)
            
            # Clean up uploaded file
            genai.delete_file(audio_file.name)
            
            return result
            
        except Exception as e:
            logger.error(f"Gemini transcription failed: {e}")
            raise
    
    async def _generate_async(self, contents, generation_config, safety_settings):
        """Async wrapper for Gemini generation."""
        import asyncio
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.model.generate_content(
                contents=contents,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
        )
    
    def _parse_gemini_response(self, response_text: str, audio_path: str) -> Dict:
        """Parse structured response from Gemini."""
        
        result = {
            'source_file': audio_path,
            'model': 'gemini-2.5-flash',
            'timestamp': datetime.now().isoformat(),
            'text': '',
            'emotional_analysis': '',
            'summary': '',
            'segments': []
        }
        
        sections = response_text.split('\n\n')
        current_section = None
        
        for section in sections:
            section = section.strip()
            if section.startswith('TRANSCRIPTION:'):
                current_section = 'transcription'
                content = section.replace('TRANSCRIPTION:', '').strip()
                result['text'] = content
            elif section.startswith('EMOTIONAL_ANALYSIS:'):
                current_section = 'emotional'
                content = section.replace('EMOTIONAL_ANALYSIS:', '').strip()
                result['emotional_analysis'] = content
            elif section.startswith('SUMMARY:'):
                current_section = 'summary'
                content = section.replace('SUMMARY:', '').strip()
                result['summary'] = content
            elif current_section:
                # Continue previous section
                if current_section == 'transcription':
                    result['text'] += '\n' + section
                elif current_section == 'emotional':
                    result['emotional_analysis'] += '\n' + section
                elif current_section == 'summary':
                    result['summary'] += '\n' + section
        
        # Extract timestamp segments from transcription
        result['segments'] = self._extract_timestamps(result['text'])
        
        return result
    
    def _extract_timestamps(self, text: str) -> List[Dict]:
        """Extract timestamp information from transcribed text."""
        import re
        
        # Pattern to match timestamps like [01:23] or [1:23]
        timestamp_pattern = r'\[(\d{1,2}):(\d{2})\]'
        
        segments = []
        lines = text.split('\n')
        
        for line in lines:
            matches = re.finditer(timestamp_pattern, line)
            for match in matches:
                minutes = int(match.group(1))
                seconds = int(match.group(2))
                timestamp_ms = (minutes * 60 + seconds) * 1000
                
                # Extract text after timestamp
                text_part = line[match.end():].strip()
                if text_part:
                    segments.append({
                        'start': timestamp_ms,
                        'text': text_part,
                        'timestamp_str': match.group(0)
                    })
        
        return segments

    def transcribe_realtime_stream(self, audio_stream) -> AsyncGenerator[str, None]:
        """Real-time transcription using Gemini Live API (when available)."""
        # Note: This is a placeholder for future Live API implementation
        # The Live API is still in preview and not widely available
        
        async def stream_transcription():
            # Implementation would depend on Live API availability
            yield "Real-time transcription not yet implemented"
        
        return stream_transcription()
```

### FireRedASR Integration

FireRedASR offers state-of-the-art performance for Mandarin and multilingual content. Here's the implementation:[11]

```python
# Note: FireRedASR requires specific installation and setup
# This is a conceptual implementation based on available documentation

class FireRedASRTranscriber:
    """FireRedASR implementation for industrial-grade transcription."""
    
    def __init__(self, model_type: str = "aed", model_path: str = "./models/FireRedASR"):
        self.model_type = model_type  # "aed" or "llm"
        self.model_path = Path(model_path)
        
        # Import FireRedASR (requires proper installation)
        try:
            from fireredasr.models.fireredasr import FireRedAsr
            self.model = FireRedAsr.from_pretrained(model_type, str(self.model_path))
        except ImportError:
            raise ImportError("FireRedASR not installed. Please follow installation instructions.")
    
    def transcribe(self, audio_path: str, options: Dict = None) -> Dict:
        """Transcribe audio using FireRedASR."""
        
        default_options = {
            "use_gpu": 1,
            "beam_size": 3,
            "nbest": 1,
            "decode_max_len": 0,
            "softmax_smoothing": 1.25 if self.model_type == "aed" else None,
            "aed_length_penalty": 0.6 if self.model_type == "aed" else None,
            "eos_penalty": 1.0 if self.model_type == "aed" else None,
            "repetition_penalty": 3.0 if self.model_type == "llm" else None,
            "llm_length_penalty": 1.0 if self.model_type == "llm" else None,
            "temperature": 1.0 if self.model_type == "llm" else None
        }
        
        # Merge options
        transcription_options = {**default_options, **(options or {})}
        # Remove None values
        transcription_options = {k: v for k, v in transcription_options.items() if v is not None}
        
        try:
            # Prepare batch data (FireRedASR expects batch format)
            batch_uttid = [Path(audio_path).stem]
            batch_wav_path = [audio_path]
            
            # Transcribe
            results = self.model.transcribe(
                batch_uttid,
                batch_wav_path,
                transcription_options
            )
            
            # Process results
            if results and len(results) > 0:
                result = results
                return {
                    'text': result.get('text', ''),
                    'confidence': result.get('confidence', 0.0),
                    'model': f'fireredasr-{self.model_type}',
                    'source_file': audio_path,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                raise ValueError("No transcription results returned")
                
        except Exception as e:
            logger.error(f"FireRedASR transcription failed: {e}")
            raise
```

### Unified Transcription Manager

To orchestrate multiple ASR models effectively, implement a unified manager:

```python
from enum import Enum
from typing import Union

class ASRModel(Enum):
    ASSEMBLYAI = "assemblyai"
    GEMINI_FLASH = "gemini_flash"
    FIRERED_ASR = "firered_asr"
    WHISPER = "whisper"  # Fallback option

class UnifiedTranscriptionManager:
    """Unified manager for multiple ASR models with intelligent selection."""
    
    def __init__(self, config: Dict[str, str]):
        """Initialize with API keys and configurations."""
        self.config = config
        self.transcribers = {}
        
        # Initialize available transcribers
        if config.get('assemblyai_api_key'):
            self.transcribers[ASRModel.ASSEMBLYAI] = AssemblyAITranscriber(
                config['assemblyai_api_key']
            )
        
        if config.get('google_ai_api_key'):
            self.transcribers[ASRModel.GEMINI_FLASH] = GeminiFlashTranscriber(
                config['google_ai_api_key']
            )
        
        # Add other transcribers as available
    
    def select_optimal_model(self, audio_info: Dict, requirements: Dict = None) -> ASRModel:
        """Intelligently select the best model based on audio characteristics and requirements."""
        
        duration = audio_info.get('duration', 0)
        language = audio_info.get('language', 'en')
        quality = audio_info.get('quality', 'medium')
        
        requirements = requirements or {}
        need_diarization = requirements.get('speaker_diarization', False)
        need_emotions = requirements.get('emotion_detection', False)
        need_realtime = requirements.get('realtime', False)
        budget_limit = requirements.get('budget_limit', 'medium')
        
        # Decision logic
        if language in ['zh', 'zh-CN', 'zh-TW'] and ASRModel.FIRERED_ASR in self.transcribers:
            return ASRModel.FIRERED_ASR
        
        if need_diarization and ASRModel.ASSEMBLYAI in self.transcribers:
            return ASRModel.ASSEMBLYAI
        
        if need_emotions and ASRModel.GEMINI_FLASH in self.transcribers:
            return ASRModel.GEMINI_FLASH
        
        if duration > 3600 and budget_limit == 'low':  # > 1 hour, budget conscious
            return ASRModel.WHISPER
        
        # Default to best available model
        if ASRModel.ASSEMBLYAI in self.transcribers:
            return ASRModel.ASSEMBLYAI
        
        if ASRModel.GEMINI_FLASH in self.transcribers:
            return ASRModel.GEMINI_FLASH
        
        return ASRModel.WHISPER  # Fallback
    
    async def transcribe_with_fallback(
        self, 
        audio_path: str, 
        audio_info: Dict = None,
        requirements: Dict = None
    ) -> Dict:
        """Transcribe with automatic fallback on failure."""
        
        audio_info = audio_info or {}
        
        # Select primary model
        primary_model = self.select_optimal_model(audio_info, requirements)
        
        # Define fallback order
        fallback_order = [
            ASRModel.ASSEMBLYAI,
            ASRModel.GEMINI_FLASH,
            ASRModel.FIRERED_ASR,
            ASRModel.WHISPER
        ]
        
        # Remove primary model from fallbacks and put it first
        if primary_model in fallback_order:
            fallback_order.remove(primary_model)
        fallback_order.insert(0, primary_model)
        
        last_error = None
        
        for model in fallback_order:
            if model not in self.transcribers:
                continue
            
            try:
                console.print(f"[blue]ℹ[/blue] Attempting transcription with {model.value}")
                
                transcriber = self.transcribers[model]
                
                if model == ASRModel.ASSEMBLYAI:
                    result = transcriber.transcribe_file(audio_path)
                elif model == ASRModel.GEMINI_FLASH:
                    result = await transcriber.transcribe_with_context(audio_path)
                elif model == ASRModel.FIRERED_ASR:
                    result = transcriber.transcribe(audio_path)
                else:
                    # Fallback to Whisper implementation
                    result = await self._whisper_fallback(audio_path)
                
                result['model_used'] = model.value
                console.print(f"[green]✓[/green] Transcription successful with {model.value}")
                return result
                
            except Exception as e:
                last_error = e
                console.print(f"[yellow]⚠[/yellow] {model.value} failed: {e}")
                continue
        
        raise Exception(f"All transcription models failed. Last error: {last_error}")
    
    async def _whisper_fallback(self, audio_path: str) -> Dict:
        """Fallback to Whisper implementation."""
        try:
            import whisper
            
            model = whisper.load_model("large-v3")
            result = model.transcribe(audio_path)
            
            return {
                'text': result['text'],
                'segments': result.get('segments', []),
                'language': result.get('language', 'unknown'),
                'model': 'whisper-large-v3',
                'source_file': audio_path,
                'timestamp': datetime.now().isoformat()
            }
        except ImportError:
            raise Exception("Whisper not available as fallback")

# Example usage
async def main():
    config = {
        'assemblyai_api_key': 'your-assemblyai-key',
        'google_ai_api_key': 'your-google-ai-key'
    }
    
    manager = UnifiedTranscriptionManager(config)
    
    audio_file = "path/to/audio.wav"
    audio_info = {'duration': 300, 'language': 'en'}
    requirements = {'speaker_diarization': True, 'emotion_detection': True}
    
    result = await manager.transcribe_with_fallback(
        audio_file, 
        audio_info, 
        requirements
    )
    
    print(f"Transcribed with {result['model_used']}: {result['text'][:100]}...")

if __name__ == "__main__":
    asyncio.run(main())
```

This chapter establishes a comprehensive foundation for next-generation speech recognition, providing production-ready implementations of the most advanced ASR models available. The unified management system ensures optimal model selection and robust fallback mechanisms for reliable transcription workflows.

***

## Chapter 5: Speaker Diarization and Voice Intelligence

### Advanced Speaker Diarization Systems

Speaker diarization—the process of partitioning audio streams by speaker identity—represents a critical component of modern transcription systems. This chapter implements cutting-edge diarization techniques using Pyannote.audio 3.1, along with advanced voice intelligence features that extend beyond simple speaker separation.[14][15][16][13]

### Pyannote.audio 3.1 Implementation

Pyannote.audio has emerged as the leading open-source toolkit for speaker diarization, offering state-of-the-art performance with neural building blocks designed specifically for speaker analysis. The following implementation demonstrates a production-ready system:[16]

```python
#!/usr/bin/env python3
"""
Advanced Speaker Diarization System
Using Pyannote.audio 3.1 with enhanced features
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import warnings
warnings.filterwarnings('ignore')

console = Console()
logger = logging.getLogger(__name__)

# Import pyannote components
try:
    from pyannote.audio import Pipeline, Model
    from pyannote.audio.pipelines import SpeakerDiarization
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate
    import pyannote.core
except ImportError:
    raise ImportError("pyannote.audio not installed. Run: pip install pyannote.audio")

@dataclass
class DiarizationConfig:
    """Configuration for speaker diarization pipeline."""
    
    # Model settings
    model_name: str = "pyannote/speaker-diarization-3.1"
    use_auth_token: Optional[str] = None  # HuggingFace token for private models
    
    # Processing parameters
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    num_speakers: Optional[int] = None  # Fixed number if known
    
    # Quality settings
    chunk_length: float = 30.0  # Process in chunks for long audio
    overlap: float = 5.0  # Overlap between chunks
    
    # Voice activity detection
    vad_onset: float = 0.5
    vad_offset: float = 0.35
    vad_min_duration_on: float = 0.0
    vad_min_duration_off: float = 0.0
    
    # Speaker segmentation
    seg_step: float = 0.1
    seg_rho: float = 0.1
    seg_tau: float = 0.1
    
    # Clustering
    clustering_method: str = "centroid"  # "centroid" or "closest"
    clustering_threshold: float = 0.7
    
    # Output options
    include_embeddings: bool = False
    include_confidence: bool = True
    merge_short_segments: bool = True
    min_segment_duration: float = 1.0

class AdvancedSpeakerDiarizer:
    """Production-ready speaker diarization system with enhanced features."""
    
    def __init__(self, config: DiarizationConfig = None, device: str = None):
        self.config = config or DiarizationConfig()
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        console.print(f"[blue]ℹ[/blue] Using device: {self.device}")
        
        # Initialize pipeline
        self._initialize_pipeline()
        
        # Cache for speaker embeddings
        self.speaker_embeddings_cache = {}
        
    def _initialize_pipeline(self):
        """Initialize the diarization pipeline with configuration."""
        
        try:
            # Load the pre-trained pipeline
            self.pipeline = Pipeline.from_pretrained(
                self.config.model_name,
                use_auth_token=self.config.use_auth_token
            )
            
            # Configure pipeline parameters
            self.pipeline = self.pipeline.to(self.device)
            
            # Set clustering parameters
            if hasattr(self.pipeline, '_clustering'):
                self.pipeline._clustering.threshold = self.config.clustering_threshold
                self.pipeline._clustering.method = self.config.clustering_method
            
            # Set VAD parameters
            if hasattr(self.pipeline, '_voice_activity_detection'):
                vad = self.pipeline._voice_activity_detection
                if hasattr(vad, 'onset'):
                    vad.onset = self.config.vad_onset
                if hasattr(vad, 'offset'):
                    vad.offset = self.config.vad_offset
            
            console.print("[green]✓[/green] Diarization pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    def diarize_audio(self, audio_path: str, reference_speakers: Dict[str, str] = None) -> Dict:
        """
        Perform speaker diarization on audio file.
        
        Args:
            audio_path: Path to audio file
            reference_speakers: Optional dict mapping speaker names to reference audio files
            
        Returns:
            Comprehensive diarization results
        """
        
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        console.print(f"[blue]ℹ[/blue] Starting diarization: {Path(audio_path).name}")
        
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Ensure mono audio
            if waveform.shape > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Get audio duration
            duration = waveform.shape[1] / sample_rate
            
            console.print(f"[blue]ℹ[/blue] Audio duration: {duration:.1f} seconds")
            
            # Process in chunks for long audio
            if duration > self.config.chunk_length:
                result = self._process_long_audio(audio_path, waveform, sample_rate)
            else:
                result = self._process_single_chunk(audio_path)
            
            # Post-process results
            result = self._post_process_diarization(result, duration)
            
            # Add speaker identification if reference speakers provided
            if reference_speakers:
                result = self._identify_speakers(result, reference_speakers, audio_path)
            
            console.print(f"[green]✓[/green] Diarization completed: {len(result['speakers'])} speakers detected")
            
            return result
            
        except Exception as e:
            logger.error(f"Diarization failed for {audio_path}: {e}")
            raise
    
    def _process_single_chunk(self, audio_path: str) -> Dict:
        """Process single audio chunk."""
        
        # Set speaker constraints
        diarization_kwargs = {}
        if self.config.num_speakers is not None:
            diarization_kwargs["num_speakers"] = self.config.num_speakers
        elif self.config.min_speakers is not None or self.config.max_speakers is not None:
            diarization_kwargs["min_speakers"] = self.config.min_speakers
            diarization_kwargs["max_speakers"] = self.config.max_speakers
        
        # Perform diarization
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Processing audio...", total=None)
            
            annotation = self.pipeline(audio_path, **diarization_kwargs)
        
        # Convert to our format
        return self._annotation_to_dict(annotation, audio_path)
    
    def _process_long_audio(self, audio_path: str, waveform: torch.Tensor, sample_rate: int) -> Dict:
        """Process long audio in chunks with overlap."""
        
        duration = waveform.shape[1] / sample_rate
        chunk_samples = int(self.config.chunk_length * sample_rate)
        overlap_samples = int(self.config.overlap * sample_rate)
        step_samples = chunk_samples - overlap_samples
        
        all_annotations = []
        chunk_results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            num_chunks = int(np.ceil((waveform.shape[1] - overlap_samples) / step_samples))
            task = progress.add_task("Processing chunks...", total=num_chunks)
            
            for i in range(num_chunks):
                start_sample = i * step_samples
                end_sample = min(start_sample + chunk_samples, waveform.shape[1])
                
                chunk_start_time = start_sample / sample_rate
                chunk_end_time = end_sample / sample_rate
                
                # Extract chunk
                chunk_waveform = waveform[:, start_sample:end_sample]
                
                # Save chunk to temporary file
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    torchaudio.save(tmp_file.name, chunk_waveform, sample_rate)
                    
                    # Process chunk
                    try:
                        annotation = self.pipeline(tmp_file.name)
                        
                        # Adjust timestamps to global time
                        adjusted_annotation = Annotation()
                        for segment, _, speaker in annotation.itertracks(yield_label=True):
                            new_segment = Segment(
                                start=segment.start + chunk_start_time,
                                end=segment.end + chunk_start_time
                            )
                            adjusted_annotation[new_segment] = f"chunk_{i}_{speaker}"
                        
                        all_annotations.append(adjusted_annotation)
                        
                    finally:
                        # Clean up temporary file
                        Path(tmp_file.name).unlink(missing_ok=True)
                
                progress.advance(task)
        
        # Merge annotations from all chunks
        merged_annotation = self._merge_chunk_annotations(all_annotations)
        
        return self._annotation_to_dict(merged_annotation, audio_path)
    
    def _merge_chunk_annotations(self, annotations: List[Annotation]) -> Annotation:
        """Merge annotations from multiple chunks, handling speaker consistency."""
        
        if not annotations:
            return Annotation()
        
        # Combine all annotations
        combined = Annotation()
        for annotation in annotations:
            combined.update(annotation)
        
        # TODO: Implement speaker clustering across chunks
        # This is a simplified version - production systems would need
        # more sophisticated speaker matching across chunks
        
        return combined
    
    def _annotation_to_dict(self, annotation: Annotation, audio_path: str) -> Dict:
        """Convert pyannote Annotation to our dictionary format."""
        
        segments = []
        speakers = set()
        
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            segment_data = {
                'start': float(segment.start),
                'end': float(segment.end),
                'duration': float(segment.end - segment.start),
                'speaker': str(speaker),
                'confidence': 1.0  # Pyannote doesn't provide confidence by default
            }
            segments.append(segment_data)
            speakers.add(str(speaker))
        
        # Sort segments by start time
        segments.sort(key=lambda x: x['start'])
        
        # Calculate speaker statistics
        speaker_stats = {}
        for speaker in speakers:
            speaker_segments = [s for s in segments if s['speaker'] == speaker]
            total_duration = sum(s['duration'] for s in speaker_segments)
            
            speaker_stats[speaker] = {
                'total_duration': total_duration,
                'segment_count': len(speaker_segments),
                'average_segment_duration': total_duration / len(speaker_segments) if speaker_segments else 0,
                'speaking_ratio': total_duration / max(s['end'] for s in segments) if segments else 0
            }
        
        return {
            'source_file': audio_path,
            'model': 'pyannote-3.1',
            'timestamp': datetime.now().isoformat(),
            'total_duration': max(s['end'] for s in segments) if segments else 0,
            'num_speakers': len(speakers),
            'speakers': list(speakers),
            'speaker_stats': speaker_stats,
            'segments': segments,
            'diarization_quality': self._estimate_quality(segments)
        }
    
    def _post_process_diarization(self, result: Dict, audio_duration: float) -> Dict:
        """Post-process diarization results to improve quality."""
        
        segments = result['segments']
        
        if self.config.merge_short_segments:
            segments = self._merge_short_segments(segments)
        
        # Remove segments that are too short
        segments = [
            s for s in segments 
            if s['duration'] >= self.config.min_segment_duration
        ]
        
        # Update result
        result['segments'] = segments
        result['post_processed'] = True
        
        return result
    
    def _merge_short_segments(self, segments: List[Dict]) -> List[Dict]:
        """Merge very short segments with adjacent segments from the same speaker."""
        
        if not segments:
            return segments
        
        merged = []
        i = 0
        
        while i  Dict:
        """Estimate diarization quality metrics."""
        
        if not segments:
            return {'overall_score': 0.0, 'metrics': {}}
        
        # Calculate basic quality metrics
        durations = [s['duration'] for s in segments]
        gaps = []
        
        for i in range(1, len(segments)):
            gap = segments[i]['start'] - segments[i-1]['end']
            if gap > 0:
                gaps.append(gap)
        
        metrics = {
            'average_segment_duration': np.mean(durations),
            'segment_duration_std': np.std(durations),
            'average_gap': np.mean(gaps) if gaps else 0,
            'gap_std': np.std(gaps) if gaps else 0,
            'total_segments': len(segments),
            'speaker_consistency': self._calculate_speaker_consistency(segments)
        }
        
        # Calculate overall quality score (0-1)
        # This is a heuristic - could be improved with ground truth data
        score = 0.8  # Base score
        
        # Penalize very short segments
        if metrics['average_segment_duration']  1.0:
            score -= 0.1
        
        # Reward consistent speaker patterns
        score += metrics['speaker_consistency'] * 0.2
        
        return {
            'overall_score': max(0.0, min(1.0, score)),
            'metrics': metrics
        }
    
    def _calculate_speaker_consistency(self, segments: List[Dict]) -> float:
        """Calculate speaker consistency score."""
        
        if len(segments)  0 else 1.0
        
        return consistency
    
    def _identify_speakers(self, result: Dict, reference_speakers: Dict[str, str], audio_path: str) -> Dict:
        """Identify speakers using reference audio samples."""
        
        try:
            from pyannote.audio import Inference
            
            # Load speaker embedding model
            embedding_model = Model.from_pretrained(
                "pyannote/wespeaker-voxceleb-resnet34-LM",
                use_auth_token=self.config.use_auth_token
            )
            inference = Inference(embedding_model, device=self.device)
            
            # Extract embeddings for reference speakers
            reference_embeddings = {}
            for speaker_name, ref_audio_path in reference_speakers.items():
                if Path(ref_audio_path).exists():
                    ref_embedding = inference(ref_audio_path)
                    reference_embeddings[speaker_name] = ref_embedding
            
            if not reference_embeddings:
                console.print("[yellow]⚠[/yellow] No valid reference speakers found")
                return result
            
            # Extract embeddings for each segment
            segment_embeddings = []
            for segment in result['segments']:
                # Extract segment audio
                segment_audio = self._extract_segment_audio(
                    audio_path, segment['start'], segment['end']
                )
                
                if segment_audio is not None:
                    try:
                        embedding = inference(segment_audio)
                        segment_embeddings.append(embedding)
                    except:
                        segment_embeddings.append(None)
                else:
                    segment_embeddings.append(None)
            
            # Match segments to reference speakers
            identified_speakers = {}
            for original_speaker in result['speakers']:
                speaker_segments = [
                    (i, s) for i, s in enumerate(result['segments']) 
                    if s['speaker'] == original_speaker
                ]
                
                if not speaker_segments:
                    continue
                
                # Get valid embeddings for this speaker
                valid_embeddings = [
                    segment_embeddings[i] for i, _ in speaker_segments 
                    if segment_embeddings[i] is not None
                ]
                
                if not valid_embeddings:
                    continue
                
                # Average embeddings for this speaker
                speaker_embedding = torch.mean(torch.stack(valid_embeddings), dim=0)
                
                # Find best match among reference speakers
                best_match = None
                best_similarity = -1
                
                for ref_name, ref_embedding in reference_embeddings.items():
                    similarity = torch.cosine_similarity(
                        speaker_embedding.unsqueeze(0), 
                        ref_embedding.unsqueeze(0)
                    ).item()
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = ref_name
                
                # Accept match if similarity is high enough
                if best_similarity > 0.7:  # Threshold for speaker identification
                    identified_speakers[original_speaker] = {
                        'identified_as': best_match,
                        'confidence': best_similarity
                    }
            
            # Update result with identified speakers
            result['speaker_identification'] = identified_speakers
            
            # Update segments with identified speaker names
            for segment in result['segments']:
                if segment['speaker'] in identified_speakers:
                    identification = identified_speakers[segment['speaker']]
                    segment['identified_speaker'] = identification['identified_as']
                    segment['identification_confidence'] = identification['confidence']
            
            console.print(f"[green]✓[/green] Speaker identification completed: {len(identified_speakers)} speakers identified")
            
        except Exception as e:
            logger.warning(f"Speaker identification failed: {e}")
        
        return result
    
    def _extract_segment_audio(self, audio_path: str, start_time: float, end_time: float) -> Optional[str]:
        """Extract audio segment for speaker identification."""
        
        try:
            import tempfile
            
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Calculate sample indices
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            # Extract segment
            segment_waveform = waveform[:, start_sample:end_sample]
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                torchaudio.save(tmp_file.name, segment_waveform, sample_rate)
                return tmp_file.name
                
        except Exception as e:
            logger.warning(f"Failed to extract segment audio: {e}")
            return None
    
    def visualize_diarization(self, result: Dict, output_path: str = None, show_plot: bool = True):
        """Create visualization of diarization results."""
        
        segments = result['segments']
        speakers = result['speakers']
        
        if not segments:
            console.print("[yellow]⚠[/yellow] No segments to visualize")
            return
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
        
        # Color map for speakers
        colors = sns.color_palette("husl", len(speakers))
        speaker_colors = {speaker: colors[i] for i, speaker in enumerate(speakers)}
        
        # Plot 1: Timeline view
        for segment in segments:
            speaker = segment['speaker']
            color = speaker_colors[speaker]
            
            ax1.barh(
                speaker, 
                segment['duration'], 
                left=segment['start'],
                height=0.8,
                color=color,
                alpha=0.7,
                edgecolor='black',
                linewidth=0.5
            )
        
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Speaker')
        ax1.set_title(f'Speaker Diarization Timeline - {Path(result["source_file"]).name}')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Speaker activity over time
        total_duration = result['total_duration']
        time_bins = np.linspace(0, total_duration, 100)
        
        for speaker in speakers:
            speaker_activity = np.zeros(len(time_bins))
            
            for segment in segments:
                if segment['speaker'] == speaker:
                    start_idx = int((segment['start'] / total_duration) * len(time_bins))
                    end_idx = int((segment['end'] / total_duration) * len(time_bins))
                    speaker_activity[start_idx:end_idx] = 1
            
            ax2.plot(
                time_bins, 
                speaker_activity + speakers.index(speaker),
                color=speaker_colors[speaker],
                linewidth=2,
                label=speaker
            )
        
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Speaker Activity')
        ax2.set_title('Speaker Activity Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            console.print(f"[green]✓[/green] Visualization saved: {output_path}")
        
        if show_plot:
            plt.show()
        
        plt.close()
    
    def export_rttm(self, result: Dict, output_path: str):
        """Export diarization results in RTTM format."""
        
        segments = result['segments']
        
        with open(output_path, 'w') as f:
            for segment in segments:
                # RTTM format: SPEAKER        
                f.write(
                    f"SPEAKER {Path(result['source_file']).stem} 1 "
                    f"{segment['start']:.3f} {segment['duration']:.3f} "
                    f"  {segment['speaker']} \n"
                )
        
        console.print(f"[green]✓[/green] RTTM file exported: {output_path}")
    
    def calculate_statistics(self, result: Dict) -> Dict:
        """Calculate comprehensive statistics for diarization results."""
        
        segments = result['segments']
        speakers = result['speakers']
        
        if not segments:
            return {'error': 'No segments found'}
        
        # Basic statistics
        total_duration = result['total_duration']
        total_speech_time = sum(s['duration'] for s in segments)
        silence_time = total_duration - total_speech_time
        
        # Per-speaker statistics
        speaker_stats = {}
        for speaker in speakers:
            speaker_segments = [s for s in segments if s['speaker'] == speaker]
            speaker_duration = sum(s['duration'] for s in speaker_segments)
            
            speaker_stats[speaker] = {
                'total_duration': speaker_duration,
                'percentage': (speaker_duration / total_speech_time) * 100 if total_speech_time > 0 else 0,
                'segment_count': len(speaker_segments),
                'average_segment_duration': speaker_duration / len(speaker_segments) if speaker_segments else 0,
                'longest_segment': max((s['duration'] for s in speaker_segments), default=0),
                'shortest_segment': min((s['duration'] for s in speaker_segments), default=0)
            }
        
        # Interaction statistics
        speaker_transitions = 0
        for i in range(1, len(segments)):
            if segments[i]['speaker'] != segments[i-1]['speaker']:
                speaker_transitions += 1
        
        # Speaking patterns
        segment_durations = [s['duration'] for s in segments]
        gaps = []
        for i in range(1, len(segments)):
            gap = segments[i]['start'] - segments[i-1]['end']
            if gap > 0:
                gaps.append(gap)
        
        return {
            'overview': {
                'total_duration': total_duration,
                'total_speech_time': total_speech_time,
                'silence_time': silence_time,
                'speech_ratio': (total_speech_time / total_duration) * 100 if total_duration > 0 else 0,
                'num_speakers': len(speakers),
                'total_segments': len(segments)
            },
            'speaker_stats': speaker_stats,
            'interaction': {
                'speaker_transitions': speaker_transitions,
                'average_transition_rate': speaker_transitions / (total_duration / 60) if total_duration > 0 else 0  # per minute
            },
            'temporal_patterns': {
                'average_segment_duration': np.mean(segment_durations),
                'segment_duration_std': np.std(segment_durations),
                'median_segment_duration': np.median(segment_durations),
                'average_gap_duration': np.mean(gaps) if gaps else 0,
                'gap_duration_std': np.std(gaps) if gaps else 0
            }
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize diarizer
    config = DiarizationConfig(
        min_speakers=2,
        max_speakers=5,
        include_confidence=True,
        merge_short_segments=True
    )
    
    diarizer = AdvancedSpeakerDiarizer(config)
    
    # Test diarization
    audio_file = "path/to/your/audio.wav"  # Replace with actual file
    
    try:
        result = diarizer.diarize_audio(audio_file)
        
        console.print(f"[green]Success![/green]")
        console.print(f"Speakers detected: {result['num_speakers']}")
        console.print(f"Total segments: {len(result['segments'])}")
        
        # Calculate and display statistics
        stats = diarizer.calculate_statistics(result)
        console.print(f"Speech ratio: {stats['overview']['speech_ratio']:.1f}%")
        
        # Create visualization
        diarizer.visualize_diarization(result, "diarization_result.png")
        
        # Export results
        diarizer.export_rttm(result, "diarization_result.rttm")
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
```

### Real-time Speaker Recognition

For live applications, implement real-time speaker recognition capabilities:

```python
import queue
import threading
import sounddevice as sd
from collections import deque
import time

class RealTimeSpeakerRecognizer:
    """Real-time speaker recognition and diarization system."""
    
    def __init__(self, sample_rate: int = 16000, chunk_duration: float = 1.0):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        
        # Audio buffer
        self.audio_buffer = deque(maxlen=int(sample_rate * 10))  # 10 second buffer
        self.processing_queue = queue.Queue()
        
        # Speaker tracking
        self.current_speaker = None
        self.speaker_history = deque(maxlen=10)
        self.confidence_threshold = 0.7
        
        # Threading
        self.is_recording = False
        self.processing_thread = None
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize real-time processing models."""
        # Simplified version - would use lightweight models for real-time processing
        pass
    
    def start_recording(self):
        """Start real-time audio recording and processing."""
        self.is_recording = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_worker)
        self.processing_thread.start()
        
        # Start audio recording
        with sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            callback=self._audio_callback,
            blocksize=self.chunk_size
        ):
            console.print("[green]🎤[/green] Real-time recording started. Press Ctrl+C to stop.")
            try:
                while self.is_recording:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                console.print("\n[yellow]⚠[/yellow] Recording stopped by user")
            finally:
                self.stop_recording()
    
    def _audio_callback(self, indata, frames, time, status):
        """Process incoming audio data."""
        if status:
            console.print(f"[yellow]⚠[/yellow] Audio status: {status}")
        
        # Add to buffer
        audio_chunk = indata[:, 0]  # Convert to mono
        self.audio_buffer.extend(audio_chunk)
        
        # Queue for processing if buffer is full enough
        if len(self.audio_buffer) >= self.chunk_size:
            chunk_data = np.array(list(self.audio_buffer)[-self.chunk_size:])
            self.processing_queue.put(chunk_data)
    
    def _processing_worker(self):
        """Background worker for audio processing."""
        while self.is_recording:
            try:
                # Get audio chunk with timeout
                audio_chunk = self.processing_queue.get(timeout=1.0)
                
                # Process chunk for speaker recognition
                speaker_result = self._process_chunk(audio_chunk)
                
                if speaker_result:
                    self._update_speaker_tracking(speaker_result)
                
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Processing error: {e}")
    
    def _process_chunk(self, audio_chunk: np.ndarray) -> Optional[Dict]:
        """Process audio chunk for speaker information."""
        # Simplified processing - would use actual models here
        # This is a placeholder implementation
        
        # Simulate speaker detection
        if np.max(np.abs(audio_chunk)) > 0.01:  # Voice activity detected
            # Simulate speaker identification
            return {
                'speaker_id': 'speaker_1',  # Would be actual speaker ID
                'confidence': 0.8,
                'timestamp': time.time()
            }
        
        return None
    
    def _update_speaker_tracking(self, speaker_result: Dict):
        """Update current speaker tracking."""
        speaker_id = speaker_result['speaker_id']
        confidence = speaker_result['confidence']
        
        if confidence > self.confidence_threshold:
            if speaker_id != self.current_speaker:
                console.print(f"[blue]🎯[/blue] Speaker change: {self.current_speaker} → {speaker_id}")
                self.current_speaker = speaker_id
            
            self.speaker_history.append({
                'speaker': speaker_id,
                'confidence': confidence,
                'timestamp': speaker_result['timestamp']
            })
    
    def stop_recording(self):
        """Stop real-time recording and processing."""
        self.is_recording = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        console.print("[green]✓[/green] Recording stopped")
    
    def get_speaker_summary(self) -> Dict:
        """Get summary of speaker activity."""
        if not self.speaker_history:
            return {'speakers': [], 'total_duration': 0}
        
        # Analyze speaker history
        speakers = {}
        total_time = 0
        
        for entry in self.speaker_history:
            speaker = entry['speaker']
            if speaker not in speakers:
                speakers[speaker] = {
                    'duration': 0,
                    'confidence_sum': 0,
                    'count': 0
                }
            
            speakers[speaker]['duration'] += self.chunk_duration
            speakers[speaker]['confidence_sum'] += entry['confidence']
            speakers[speaker]['count'] += 1
            total_time += self.chunk_duration
        
        # Calculate averages
        for speaker_data in speakers.values():
            speaker_data['average_confidence'] = (
                speaker_data['confidence_sum'] / speaker_data['count']
            )
            speaker_data['percentage'] = (
                speaker_data['duration'] / total_time * 100 if total_time > 0 else 0
            )
        
        return {
            'speakers': speakers,
            'total_duration': total_time,
            'current_speaker': self.current_speaker
        }

# Integration with transcription pipeline
class DiarizationIntegratedTranscriber:
    """Transcription system with integrated speaker diarization."""
    
    def __init__(self, transcriber, diarizer: AdvancedSpeakerDiarizer):
        self.transcriber = transcriber
        self.diarizer = diarizer
    
    def transcribe_with_speakers(self, audio_path: str) -> Dict:
        """Perform transcription with speaker diarization."""
        
        console.print("[blue]ℹ[/blue] Starting integrated transcription with speaker diarization...")
        
        # Step 1: Perform diarization
        diarization_result = self.diarizer.diarize_audio(audio_path)
        
        # Step 2: Perform transcription
        if hasattr(self.transcriber, 'transcribe_file'):
            transcription_result = self.transcriber.transcribe_file(audio_path)
        else:
            # Fallback for different transcriber types
            transcription_result = self.transcriber.transcribe(audio_path)
        
        # Step 3: Align transcription with speaker segments
        aligned_result = self._align_transcription_with_speakers(
            transcription_result, diarization_result
        )
        
        return aligned_result
    
    def _align_transcription_with_speakers(
        self, 
        transcription: Dict, 
        diarization: Dict
    ) -> Dict:
        """Align transcription words with speaker segments."""
        
        words = transcription.get('words', [])
        speaker_segments = diarization.get('segments', [])
        
        if not words or not speaker_segments:
            return {**transcription, 'diarization': diarization}
        
        # Align words with speaker segments
        aligned_words = []
        
        for word in words:
            word_start = word.get('start', 0) / 1000.0  # Convert to seconds if needed
            word_end = word.get('end', 0) / 1000.0
            
            # Find overlapping speaker segment
            assigned_speaker = None
            max_overlap = 0
            
            for segment in speaker_segments:
                segment_start = segment['start']
                segment_end = segment['end']
                
                # Calculate overlap
                overlap_start = max(word_start, segment_start)
                overlap_end = min(word_end, segment_end)
                
                if overlap_end > overlap_start:
                    overlap = overlap_end - overlap_start
                    if overlap > max_overlap:
                        max_overlap = overlap
                        assigned_speaker = segment['speaker']
            
            # Add speaker to word
            word_with_speaker = word.copy()
            word_with_speaker['speaker'] = assigned_speaker
            aligned_words.append(word_with_speaker)
        
        # Create speaker-wise transcription
        speaker_transcripts = {}
        for word in aligned_words:
            speaker = word.get('speaker', 'unknown')
            if speaker not in speaker_transcripts:
                speaker_transcripts[speaker] = []
            speaker_transcripts[speaker].append(word['text'])
        
        # Generate speaker-segmented text
        speaker_segments_with_text = []
        for segment in speaker_segments:
            segment_words = [
                w for w in aligned_words 
                if w.get('speaker') == segment['speaker'] and
                w.get('start', 0) / 1000.0 >= segment['start'] and
                w.get('end', 0) / 1000.0  np.ndarray:
        """Extract comprehensive features from audio."""
        
        try:
            # Load audio
            if segment_start is not None and segment_end is not None:
                # Load specific segment
                duration = segment_end - segment_start
                audio, sr = librosa.load(
                    audio_path, 
                    sr=self.config.sample_rate,
                    offset=segment_start,
                    duration=duration
                )
            else:
                # Load with configured parameters
                audio, sr = librosa.load(
                    audio_path,
                    sr=self.config.sample_rate,
                    duration=self.config.duration,
                    offset=self.config.offset
                )
            
            # Pad or trim to consistent length
            target_length = int(self.config.sample_rate * self.config.duration)
            if len(audio) < target_length:

[1] https://assemblyai.com/blog/the-state-of-python-speech-recognition
[2] https://voicewriter.io/blog/best-speech-recognition-api-2025
[3] https://www.assemblyai.com/whisper-ai-vs-assemblyai
[4] https://assemblyai.com/blog/comparing-universal-2-and-openai-whisper
[5] https://chemrxiv.org/engage/chemrxiv/article-details/60c7548e0f50dba5b9397d43
[6] https://dl.acm.org/doi/10.1145/3626253.3635432
[7] https://www.youtube.com/watch?v=gI_duaS18Us
[8] https://vinceth.net/2024/10/24/yt-dlp-tutorial.html
[9] https://stackoverflow.com/questions/75867758/how-to-extract-only-audio-from-downloading-video-python-yt-dlp
[10] https://write.corbpie.com/downloading-youtube-videos-as-audio-with-yt-dlp/
[11] https://github.com/FireRedTeam/FireRedASR
[12] https://ai.google.dev/gemini-api/docs/audio
[13] https://dev.to/gracezzhang/speaker-diarization-in-python-235i
[14] https://scalastic.io/en/whisper-pyannote-ultimate-speech-transcription/
[15] https://assemblyai.com/blog/top-speaker-diarization-libraries-and-apis
[16] https://github.com/pyannote/pyannote-audio
[17] https://arxiv.org/abs/2503.22510
[18] https://github.com/MrCuber/Speech-Emotion-Recognition
[19] https://data-flair.training/blogs/python-mini-project-speech-emotion-recognition/
[20] https://thepythoncode.com/article/building-a-speech-emotion-recognizer-using-sklearn
[21] https://www.digitalocean.com/community/tutorials/how-to-generate-and-add-subtitles-to-videos-using-python-openai-whisper-and-ffmpeg
[22] https://www.assemblyai.com/blog/auto-generate-subtitles-with-python
[23] https://assemblyai.com/blog/create-vtt-files-for-videos-python
[24] https://github.com/glut23/webvtt-py
[25] https://www.youtube.com/watch?v=aJPMFAIsApY
[26] https://developers.google.com/youtube/v3/guides/uploading_a_video
[27] https://github.com/youtube/api-samples/blob/master/python/captions.py
[28] https://dev.to/fosteman/custom-transcription-and-clipping-pipeline-2814
[29] https://www.edlitera.com/blog/posts/automatic-dubbing-preprocessing-transcription
[30] https://www.youtube.com/watch?v=8rb9GefC_CU
[31] https://www.assemblyai.com/benchmarks
[32] http://arxiv.org/pdf/2403.05530.pdf
[33] https://www.willowtreeapps.com/craft/10-speech-to-text-models-tested
[34] https://www.byteplus.com/en/topic/409753
[35] https://revistes.ub.edu/index.php/phonica/article/view/49237
[36] https://arxiv.org/abs/2506.21555
[37] https://ieeexplore.ieee.org/document/10848815/
[38] https://www.linkedin.com/pulse/working-google-colab-python-machine-learning-guide-rany
[39] https://journal.unj.ac.id/unj/index.php/jpppf/article/view/54901
[40] https://www.ijraset.com/best-journal/real-time-object-detection-using-yolov4
[41] https://ieeexplore.ieee.org/document/9401151/
[42] https://ai.google.dev/gemini-api/docs/speech-generation
[43] https://github.com/topics/speech-emotion-recognition
[44] https://arxiv.org/abs/2502.17284
[45] https://arxiv.org/abs/2406.10082
[46] https://arxiv.org/abs/2404.17394
[47] https://ieeexplore.ieee.org/document/10447004/
[48] https://arxiv.org/abs/2405.00966
[49] https://ieeexplore.ieee.org/document/10389643/
[50] https://arxiv.org/pdf/2303.01037.pdf
[51] https://arxiv.org/pdf/2212.04356.pdf
[52] https://arxiv.org/pdf/2305.13516.pdf
[53] https://arxiv.org/pdf/2307.08234.pdf
[54] https://arxiv.org/pdf/2409.09543.pdf
[55] https://arxiv.org/html/2410.05423v1
[56] http://arxiv.org/pdf/2402.12654.pdf
[57] https://arxiv.org/pdf/2402.01931.pdf
[58] https://arxiv.org/pdf/2106.04624.pdf
[59] https://arxiv.org/pdf/2210.17316.pdf
[60] https://arxiv.org/html/2409.09506v1
[61] https://www.swiftorial.com/matchups/nlp_platforms/whisper-vs-deepspeech
[62] https://thepythoncode.com/article/speech-recognition-using-huggingface-transformers-in-python
[63] https://www.gladia.io/blog/how-do-speech-recognition-models-work
[64] https://github.com/topics/speaker-diarization
[65] https://deepgram.com/learn/benchmarking-top-open-source-speech-models
[66] https://github.com/TensorSpeech/TensorFlowASR
[67] https://www.fastpix.io/blog/speaker-diarization-libraries-apis-for-developers
[68] https://huggingface.co/openai/whisper-large-v3
[69] https://play.ht/blog/speaker-diarization/
[70] https://www.reddit.com/r/LocalLLaMA/comments/1d7cjbf/speech_to_text_whisper_alternatives/
[71] https://python.plainenglish.io/asr-state-of-the-art-indicwav2vec-7859614263af
[72] https://ieeexplore.ieee.org/document/10969001/
[73] https://arxiv.org/pdf/2503.07891.pdf
[74] http://arxiv.org/pdf/2306.12925.pdf
[75] http://arxiv.org/pdf/2407.12875.pdf
[76] http://arxiv.org/pdf/2407.13729.pdf
[77] https://arxiv.org/pdf/2110.15018.pdf
[78] http://arxiv.org/pdf/2312.11805.pdf
[79] https://arxiv.org/pdf/2407.00463v4.pdf
[80] https://arxiv.org/pdf/2412.11272.pdf
[81] https://dl.acm.org/doi/pdf/10.1145/3643832.3661886
[82] http://arxiv.org/pdf/2408.16725.pdf
[83] http://arxiv.org/pdf/2409.10999.pdf
[84] https://arxiv.org/pdf/2410.03930.pdf
[85] https://arxiv.org/pdf/2412.18708.pdf
[86] https://aclanthology.org/2023.ijcnlp-demo.3.pdf
[87] https://arxiv.org/pdf/2401.10838.pdf
[88] https://www.mdpi.com/2073-8994/10/7/268/pdf?version=1531045283
[89] https://pypi.org/project/youtube-transcript-api/
[90] https://www.youtube.com/watch?v=n43Td-mU7oA
[91] https://developers.googleblog.com/en/gemini-api-io-updates/
[92] https://www.youtube.com/watch?v=-VQL8ynOdVg
[93] https://www.reddit.com/r/GoogleAppsScript/comments/hgrbap/is_there_a_way_to_import_caption_from_youtube/
[94] https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash
[95] https://deepmind.google/models/gemini/flash/
[96] https://pypi.org/project/SpeechRecognition/
[97] https://developers.google.com/youtube/v3/guides/implementation/captions
[98] https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/audio-understanding
[99] https://www.kaggle.com/code/shivamburnwal/speech-emotion-recognition
[100] https://aclanthology.org/2022.iwslt-1.7.pdf
[101] https://arxiv.org/pdf/2405.10741.pdf
[102] http://arxiv.org/pdf/2205.06522.pdf
[103] https://aclanthology.org/2021.iwslt-1.26.pdf
[104] http://arxiv.org/pdf/2102.06448.pdf
[105] http://arxiv.org/pdf/2211.15103.pdf
[106] https://arxiv.org/html/2412.09283
[107] https://riunet.upv.es/bitstream/10251/146327/3/Montagud;Boronat;Gonz%C3%A1lez%20-%20Web-based%20Platform%20for%20Subtitles%20Customization%20and%20Synchronization%20in....pdf
[108] https://arxiv.org/pdf/2112.14088.pdf
[109] https://arxiv.org/abs/2209.13192
[110] http://arxiv.org/pdf/2406.06040.pdf
[111] https://arxiv.org/pdf/2305.18500.pdf
[112] https://arxiv.org/html/2502.02885
[113] https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl_a_00607/2197786/tacl_a_00607.pdf
[114] http://arxiv.org/pdf/1608.07068.pdf
[115] https://arxiv.org/pdf/2112.01073.pdf
[116] http://arxiv.org/pdf/2305.08389.pdf
[117] https://pmc.ncbi.nlm.nih.gov/articles/PMC10519822/
[118] https://www.aclweb.org/anthology/2020.iwslt-1.26.pdf
[119] http://arxiv.org/pdf/2310.04900.pdf
[120] https://stackoverflow.com/questions/48640490/python-2-7-matching-a-subtitle-events-in-vtt-subtitles-using-a-regular-expressi
[121] https://ostechnix.com/yt-dlp-tutorial/
[122] https://www.reddit.com/r/youtubedl/comments/13el971/dunces_guide_to_downloading_audio_with_ytdlp/
[123] https://pypi.org/project/webvtt-py/
[124] https://blog.elijahlopez.ca/posts/yt-dlp-audio-download/
[125] https://www.youtube.com/watch?v=oOtUZA0ZCa0
[126] https://developers.deepgram.com/docs/automatically-generating-webvtt-and-srt-captions
[127] https://cheat.sh/yt-dlp
[128] https://onlinelibrary.wiley.com/doi/10.1002/cae.22729
[129] https://jurnal.sttmcileungsi.ac.id/index.php/tekno/article/view/1231
[130] https://journal.ibrahimy.ac.id/index.php/Alifmatika/article/view/6966
[131] https://pubs.acs.org/doi/10.1021/acsomega.2c00362
[132] https://arxiv.org/abs/2408.05219
[133] https://www.per-central.org/items/perc/5649.pdf
[134] https://pmc.ncbi.nlm.nih.gov/articles/PMC10103193/
[135] http://www.arxiv.org/abs/2012.14180
[136] https://pmc.ncbi.nlm.nih.gov/articles/PMC10515421/
[137] https://arxiv.org/pdf/2501.05577.pdf
[138] http://arxiv.org/pdf/2310.03755.pdf
[139] https://pmc.ncbi.nlm.nih.gov/articles/PMC9797609/
[140] http://arxiv.org/pdf/2309.11083.pdf
[141] https://arxiv.org/pdf/2503.14443.pdf
[142] https://arxiv.org/pdf/2211.15445.pdf
[143] https://speechbrain.readthedocs.io/en/latest/tutorials/nn/using-wav2vec-2.0-hubert-wavlm-and-whisper-from-huggingface-with-speechbrain.html
[144] https://stackoverflow.com/questions/53031430/conda-environment-in-google-colab-google-colaboratory
[145] https://huggingface.co/learn/audio-course/en/chapter6/fine-tuning
[146] https://huggingface.co/learn/audio-course/en/chapter5/fine-tuning
[147] https://cloud.google.com/python/docs/setup
[148] https://huggingface.co/docs/transformers/en/tasks/asr
[149] https://python.plainenglish.io/transcript-alchemy-turning-youtube-videos-into-summaries-with-gpt-4-43c53dbf48e0
[150] https://facebookresearch.github.io/TensorComprehensions/installation_colab_research.html
[151] https://speechbrain.readthedocs.io/en/v1.0.2/tutorials/advanced/pre-trained-models-and-fine-tuning-with-huggingface.html
[152] https://colab.research.google.com/github/yy/dviz-course/blob/master/docs/m01-intro/lab01.ipynb
[153] https://huggingface.co/docs/transformers/en/training
[154] https://www.callstack.com/blog/creating-a-video-transcription-app-lessons-learned
[155] https://colab.research.google.com/github/QuantEcon/lecture-py-notebooks/blob/master/getting_started.ipynb
[156] https://huggingface.co/blog/fine-tune-whisper