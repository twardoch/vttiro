---
this_file: plan/part2.md
---

# Part 2: Video Processing & Audio Extraction

## Overview

Implement robust video processing capabilities using yt-dlp for downloading and extracting audio from various sources, with intelligent handling of different formats, durations, and quality levels. Focus on efficiency, reliability, and preparing audio data optimally for transcription processing.

## Detailed Tasks

### 2.1 Enhanced yt-dlp Integration

- [ ] Create `VideoProcessor` class with comprehensive yt-dlp integration
- [ ] Support multiple video sources beyond YouTube:
  - [ ] YouTube (primary focus)
  - [ ] Vimeo, Twitch, Facebook, Twitter
  - [ ] Direct video file URLs
  - [ ] Local video files
- [ ] Implement intelligent format selection:
  - [ ] Prefer best audio quality available
  - [ ] Fallback strategies for restricted content
  - [ ] Adaptive bitrate selection based on content length
- [ ] Add comprehensive metadata extraction:
  - [ ] Title, description, duration, language hints
  - [ ] Existing captions/subtitles for context
  - [ ] Uploader information and timestamps
  - [ ] Thumbnail and chapter information

### 2.2 Robust Download Management

- [ ] Implement retry mechanisms with exponential backoff
- [ ] Add progress tracking and cancellation support
- [ ] Handle network interruptions and resume capabilities
- [ ] Create download queue management for batch operations
- [ ] Implement concurrent downloads with rate limiting
- [ ] Add support for proxy and authentication requirements
- [ ] Handle geo-restricted content with appropriate fallbacks

### 2.3 Audio Extraction & Preprocessing

- [ ] Extract audio with optimal settings:
  - [ ] Target format: WAV 16kHz mono for consistency
  - [ ] Support multiple input formats (MP4, WebM, etc.)
  - [ ] Preserve original quality when possible
  - [ ] Normalize audio levels to prevent clipping
- [ ] Implement audio quality assessment:
  - [ ] Signal-to-noise ratio analysis
  - [ ] Dynamic range detection
  - [ ] Background noise profiling
- [ ] Add audio enhancement preprocessing:
  - [ ] Noise reduction for poor quality audio
  - [ ] Automatic gain control
  - [ ] Echo and reverb reduction

### 2.4 Intelligent Content Analysis

- [ ] Implement content type detection:
  - [ ] Identify music vs speech content
  - [ ] Detect multiple speakers vs single speaker
  - [ ] Language detection from audio characteristics
  - [ ] Content domain classification (news, podcast, lecture, etc.)
- [ ] Extract contextual information for better transcription:
  - [ ] Speaker gender detection for diarization hints
  - [ ] Estimated speaker count
  - [ ] Audio quality metrics
  - [ ] Background music/noise levels

### 2.5 Smart Chunking Implementation

- [ ] Develop energy-based segmentation algorithm:
  - [ ] Analyze audio energy levels using RMS and spectral features
  - [ ] Identify natural break points in speech
  - [ ] Balance between low energy points and maximum chunk duration
  - [ ] Prefer segmentation at full integer seconds for timestamp accuracy
- [ ] Implement configurable chunking strategies:
  - [ ] Default: 10-minute chunks with energy-based boundaries
  - [ ] Short content: Process as single chunk if under 5 minutes
  - [ ] Long content: Hierarchical chunking for videos over 2 hours
- [ ] Add overlap handling for chunk boundaries:
  - [ ] 30-second overlap between chunks to prevent word loss
  - [ ] Smart overlap detection during reassembly
  - [ ] Cross-fade analysis for seamless merging

### 2.6 Memory-Efficient Processing

- [ ] Implement streaming audio processing:
  - [ ] Process audio in chunks to avoid memory issues
  - [ ] Support videos up to 10 hours duration
  - [ ] Efficient temporary file management
  - [ ] Automatic cleanup of intermediate files
- [ ] Add memory usage monitoring and limits:
  - [ ] Adaptive chunk sizes based on available memory
  - [ ] Garbage collection optimization for large files
  - [ ] Warning systems for memory pressure situations

### 2.7 Format Handling & Conversion

- [ ] Support multiple input formats:
  - [ ] Video files: MP4, WebM, AVI, MOV, MKV
  - [ ] Audio files: WAV, MP3, FLAC, OGG, M4A
  - [ ] Stream formats: HLS, DASH, RTMP
- [ ] Implement FFmpeg integration for format conversion:
  - [ ] Audio codec optimization for transcription
  - [ ] Sample rate conversion and normalization
  - [ ] Multi-channel to mono conversion
  - [ ] Bitrate optimization based on content type

### 2.8 Metadata Enrichment

- [ ] Extract comprehensive metadata for transcription context:
  - [ ] Video title and description for domain hints
  - [ ] Upload date and channel information
  - [ ] Existing captions in multiple languages
  - [ ] Video categories and tags
- [ ] Create metadata-driven model selection:
  - [ ] Use title/description to identify technical content
  - [ ] Language hints for multilingual model routing
  - [ ] Content type for specialized model selection
  - [ ] Quality metrics for processing parameter tuning

### 2.9 Error Handling & Recovery

- [ ] Comprehensive error handling for all failure modes:
  - [ ] Network timeouts and connection failures
  - [ ] Unsupported formats or codecs
  - [ ] Copyright restrictions and geo-blocking
  - [ ] Disk space and permission issues
- [ ] Implement graceful degradation:
  - [ ] Alternative format selection when preferred fails
  - [ ] Quality reduction for problematic content
  - [ ] Partial processing for partially downloaded content
- [ ] Create detailed error reporting and diagnostics

### 2.10 Performance Optimization

- [ ] Multi-threaded download and processing:
  - [ ] Parallel audio extraction for multiple chunks
  - [ ] Concurrent downloads for batch operations
  - [ ] Efficient CPU utilization for preprocessing
- [ ] Implement caching strategies:
  - [ ] Metadata caching for repeated URLs
  - [ ] Processed audio caching with checksums
  - [ ] Smart cache invalidation and cleanup
- [ ] Add performance monitoring:
  - [ ] Download speed tracking
  - [ ] Processing time metrics
  - [ ] Memory usage profiling

## Technical Specifications

### VideoProcessor Class Interface
```python
class VideoProcessor:
    def __init__(self, 
                 cache_dir: Path,
                 max_duration: int = 36000,  # 10 hours
                 chunk_duration: int = 600,   # 10 minutes
                 overlap_duration: int = 30): # 30 seconds
        
    async def process_url(self, url: str) -> ProcessedVideo:
        """Download and process video/audio from URL"""
        
    async def process_file(self, file_path: Path) -> ProcessedVideo:
        """Process local video/audio file"""
        
    def extract_metadata(self, source: str) -> VideoMetadata:
        """Extract comprehensive metadata"""
        
    def segment_audio(self, audio_path: Path) -> List[AudioChunk]:
        """Smart energy-based audio segmentation"""
```

### Audio Segmentation Algorithm
```python
def energy_based_segmentation(
    audio: np.ndarray, 
    sr: int,
    max_chunk_duration: int = 600,
    min_energy_window: float = 2.0,
    energy_threshold_percentile: int = 20
) -> List[Tuple[float, float]]:
    """
    Segment audio at energy minima within duration constraints
    
    Returns list of (start_time, end_time) tuples in seconds
    """
```

### Configuration Options
```yaml
video_processing:
  max_duration: 36000  # 10 hours maximum
  chunk_duration: 600  # 10 minutes default chunks
  overlap_duration: 30  # 30 seconds overlap
  quality_preference: "best"  # best, medium, fast
  audio_format: "wav"
  sample_rate: 16000
  channels: 1  # mono
  
  yt_dlp:
    retry_attempts: 3
    timeout: 300  # 5 minutes per download
    rate_limit: "1M"  # 1MB/s default rate limit
    
  preprocessing:
    normalize_audio: true
    noise_reduction: "auto"  # auto, aggressive, off
    enhance_speech: true
```

## Dependencies

### Required Libraries
- `yt-dlp >= 2025.01.01` - Video downloading
- `ffmpeg-python >= 0.2.0` - Audio processing
- `librosa >= 0.10.0` - Audio analysis
- `numpy >= 1.24.0` - Numerical processing
- `scipy >= 1.10.0` - Signal processing

### Optional Libraries
- `whisper-timestamped` - Enhanced audio segmentation
- `webrtcvad` - Voice activity detection
- `noisereduce` - Audio noise reduction
- `pyloudnorm` - Audio normalization

## Success Criteria

- [ ] Successfully download and process videos from major platforms
- [ ] Handle videos up to 10 hours duration without memory issues
- [ ] Achieve sub-second accuracy in audio segmentation timing
- [ ] Maintain 99% success rate for supported video formats
- [ ] Process audio at 5x real-time speed or better
- [ ] Generate high-quality metadata for transcription context
- [ ] Robust error handling with graceful degradation
- [ ] Memory usage stays under 2GB for any single video

## Integration Points

### With Part 3 (Multi-Model Transcription)
- Provide optimally segmented audio chunks
- Supply rich metadata for model selection
- Enable parallel processing of multiple chunks

### With Part 4 (Smart Audio Segmentation)  
- Enhance basic segmentation with advanced algorithms
- Provide energy analysis data for intelligent chunking
- Support custom segmentation strategies per content type

### With Part 8 (YouTube Integration)
- Extract YouTube-specific metadata for upload optimization
- Preserve original video information for subtitle upload
- Handle YouTube-specific format requirements

## Timeline

**Week 3-4**: Core yt-dlp integration and basic processing  
**Week 5**: Smart chunking and energy-based segmentation  
**Week 6**: Error handling, optimization, and testing  
**Week 7**: Integration with transcription pipeline  

This robust video processing foundation enables the entire transcription pipeline to operate efficiently on diverse content types while maintaining high reliability and performance standards.