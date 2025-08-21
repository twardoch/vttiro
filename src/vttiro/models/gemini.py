#!/usr/bin/env python3
# this_file: src/vttiro/models/gemini.py
"""Google Gemini 2.0 Flash transcription engine with advanced context awareness."""

import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
import time

try:
    from loguru import logger
except ImportError:
    import logging as logger

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Google GenerativeAI not available. Install with: uv add google-generativeai")
    # Create dummy objects to avoid NameError
    genai = None
    HarmCategory = object()
    HarmBlockThreshold = object()

from vttiro.core.transcription import TranscriptionEngine
from vttiro.core.config import VttiroConfig, TranscriptionResult
from vttiro.models.base import GeminiModel
from vttiro.core.prompts import WebVTTPromptGenerator, PromptTemplate


class GeminiTranscriber(TranscriptionEngine):
    """Google Gemini 2.0 Flash transcription engine with factual prompting and context awareness."""
    
    def __init__(self, config: VttiroConfig, model: GeminiModel = GeminiModel.GEMINI_2_0_FLASH):
        super().__init__(config)
        self.model_variant = model
        
        if not GEMINI_AVAILABLE:
            raise ImportError("Google GenerativeAI not available. Install with: uv add google-generativeai")
        
        # Configure Gemini API
        api_key = config.transcription.gemini_api_key
        if not api_key:
            raise ValueError("Gemini API key not configured. Set GEMINI_API_KEY environment variable.")
            
        genai.configure(api_key=api_key)
        
        # Initialize model with optimal settings for transcription
        self.model = genai.GenerativeModel(
            model_name=model.value,  # Use specified model variant
            generation_config={
                "temperature": 0.1,  # Low temperature for consistent transcription
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 8192,
            },
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        # Model capabilities
        self._supported_languages = [
            "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", 
            "ar", "hi", "bn", "te", "mr", "ta", "ur", "gu", "kn", "ml",
            "pa", "ne", "si", "my", "th", "vi", "id", "ms", "tl", "sw"
        ]
        
        # Initialize WebVTT prompt generator for direct format requests
        self.prompt_generator = WebVTTPromptGenerator(
            include_examples=True,
            include_diarization=True,
            include_emotions=True,
            template=PromptTemplate.SPEAKER_DIARIZATION
        )
        
    @property
    def name(self) -> str:
        return f"gemini/{self.model_variant.value}"
        
    async def transcribe(
        self, 
        audio_path: Path, 
        language: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> TranscriptionResult:
        """Transcribe audio using Gemini 2.0 Flash with context-aware prompting."""
        start_time = time.time()
        
        logger.info(f"Transcribing with Gemini: {audio_path}")
        
        try:
            # Read audio file
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
                
            # Upload audio file to Gemini
            audio_file = await self._upload_audio_file(audio_path)
            
            # Generate context-aware prompt
            prompt = self._generate_transcription_prompt(language, context)
            
            # Perform transcription
            response = await self._transcribe_with_gemini(audio_file, prompt)
            
            # Process WebVTT response
            webvtt_content = response.text.strip()
            processing_time = time.time() - start_time
            
            # Log raw response for debugging (truncated if too long)
            try:
                # Check if debug logging is enabled
                import loguru
                current_level = logger._core.min_level
                debug_enabled = current_level <= 10  # DEBUG level or lower
            except:
                # Fallback: always log if we can't determine level
                debug_enabled = True
                
            if debug_enabled:
                if webvtt_content:
                    content_preview = webvtt_content[:500] + ("..." if len(webvtt_content) > 500 else "")
                    logger.debug(f"Raw Gemini response ({len(webvtt_content)} chars):\n{content_preview}")
                else:
                    logger.error("CRITICAL: Gemini returned empty response!")
            
            # Parse WebVTT content to extract text and timing information
            transcribed_text, word_timestamps = self._parse_webvtt_response(webvtt_content)
            
            # Calculate confidence score (estimated from response quality)
            confidence = self._estimate_confidence(response, transcribed_text)
            
            logger.info(f"Gemini WebVTT transcription completed in {processing_time:.2f}s")
            
            return TranscriptionResult(
                text=transcribed_text,
                confidence=confidence,
                word_timestamps=word_timestamps,
                processing_time=processing_time,
                model_name=self.model_variant.value,
                language=language or "auto",
                metadata={
                    "engine": "gemini",
                    "model_version": self.model_variant.value,
                    "context_used": bool(context),
                    "prompt_tokens": len(prompt.split()),
                    "safety_ratings": getattr(response, 'safety_ratings', []),
                    "finish_reason": getattr(response, 'finish_reason', 'complete'),
                    "format": "webvtt",
                    "webvtt_content": webvtt_content,  # Store original WebVTT
                    "native_timing": True  # Flag indicating real timestamps
                }
            )
            
        except Exception as e:
            logger.error(f"Gemini transcription failed for {audio_path}: {e}")
            raise
            
    async def _upload_audio_file(self, audio_path: Path) -> Any:
        """Upload audio file to Gemini for processing."""
        try:
            # Upload file using genai.upload_file
            logger.debug(f"Uploading audio file to Gemini: {audio_path}")
            
            # Use the proper Gemini file upload API
            audio_file = await asyncio.to_thread(
                genai.upload_file, 
                path=str(audio_path)
            )
            
            logger.debug(f"Audio file uploaded successfully: {audio_file.display_name}")
            return audio_file
            
        except Exception as e:
            logger.error(f"Failed to upload audio file: {e}")
            raise
            
        
    def _generate_transcription_prompt(
        self, 
        language: Optional[str], 
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Generate WebVTT format prompt using WebVTTPromptGenerator.
        
        This replaces the old plain text approach with direct WebVTT format requests,
        eliminating the need for artificial timestamp estimation.
        """
        logger.debug("Generating WebVTT format prompt for Gemini transcription")
        
        # Use the WebVTT prompt generator to create a comprehensive prompt
        # that requests proper WebVTT format with timestamps directly from Gemini
        prompt = self.prompt_generator.generate_webvtt_prompt(
            language=language,
            context=context
        )
        
        # Add Gemini-specific optimization instructions
        gemini_optimizations = """

GEMINI-SPECIFIC OPTIMIZATIONS:
- Leverage your audio understanding capabilities for precise timing
- Use natural speech boundary detection for segment splitting  
- Apply contextual knowledge for proper noun recognition
- Maintain consistent speaker identification throughout
- Ensure WebVTT timestamps reflect actual speech timing, not estimation"""
        
        # Combine with the main prompt
        enhanced_prompt = prompt + gemini_optimizations
        
        logger.debug(f"Generated WebVTT prompt ({len(enhanced_prompt)} chars) for Gemini")
        return enhanced_prompt
        
    async def _transcribe_with_gemini(self, audio_file: Any, prompt: str) -> Any:
        """Perform transcription using Gemini API."""
        try:
            # Create the content parts with the prompt and uploaded audio file
            content_parts = [prompt, audio_file]
            
            # Generate content with the uploaded audio file
            response = await asyncio.to_thread(
                self.model.generate_content,
                content_parts
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            raise
            
    def _parse_webvtt_response(self, webvtt_content: str) -> tuple[str, List[Dict[str, Any]]]:
        """Parse WebVTT content to extract text and timing information.
        
        Args:
            webvtt_content: Raw WebVTT content from Gemini
            
        Returns:
            Tuple of (transcribed_text, word_timestamps)
        """
        import re
        
        logger.debug("Parsing WebVTT response from Gemini")
        
        # Initialize results
        transcribed_text_parts = []
        word_timestamps = []
        
        try:
            # Split into lines and process
            lines = webvtt_content.strip().split('\n')
            
            # Track current cue processing
            current_cue = None
            i = 0
            
            while i < len(lines):
                line = lines[i].strip()
                
                # Skip WEBVTT header and metadata
                if line.startswith('WEBVTT') or line.startswith('Language:') or not line:
                    i += 1
                    continue
                
                # Check for timestamp line - flexible format to handle Gemini's sometimes malformed timestamps
                # Standard: HH:MM:SS.mmm, but also handle: HH:MM:SS.m, HH:MM:SS, MM:SS, MM:SSS, etc.
                timestamp_match = re.match(r'([\d:\.]+)\s*-->\s*([\d:\.]+)', line)
                if timestamp_match:
                    start_time_str, end_time_str = timestamp_match.groups()
                    
                    # Convert to seconds
                    start_seconds = self._parse_timestamp(start_time_str)
                    end_seconds = self._parse_timestamp(end_time_str)
                    
                    # Collect cue text from following lines
                    cue_text_parts = []
                    i += 1
                    
                    while i < len(lines) and lines[i].strip():
                        cue_line = lines[i].strip()
                        
                        # Handle speaker tags <v Speaker>
                        speaker_match = re.match(r'<v\s+([^>]+)>(.*)', cue_line)
                        if speaker_match:
                            speaker_name, speaker_text = speaker_match.groups()
                            cue_text_parts.append(speaker_text.strip())
                        else:
                            cue_text_parts.append(cue_line)
                        
                        i += 1
                    
                    # Join cue text
                    cue_text = ' '.join(cue_text_parts).strip()
                    
                    if cue_text:
                        # Add to transcribed text
                        transcribed_text_parts.append(cue_text)
                        
                        # Create word timestamps for this cue
                        words = cue_text.split()
                        if words:
                            duration = end_seconds - start_seconds
                            for j, word in enumerate(words):
                                # Distribute words evenly across cue duration
                                word_start = start_seconds + (j / len(words)) * duration
                                word_end = start_seconds + ((j + 1) / len(words)) * duration
                                
                                # Clean word of punctuation for timestamp matching
                                clean_word = re.sub(r'[^\w]', '', word)
                                if clean_word:  # Only add non-empty words
                                    word_timestamps.append({
                                        "word": word,
                                        "start": word_start,
                                        "end": word_end,
                                        "confidence": 0.95  # High confidence for Gemini WebVTT
                                    })
                else:
                    # Skip non-timestamp lines (cue IDs, etc.)
                    i += 1
            
            # Join all transcribed text
            final_text = ' '.join(transcribed_text_parts)
            
            # Log parsing results and check for issues
            logger.debug(f"Parsed WebVTT: {len(word_timestamps)} words, {len(final_text)} chars")
            
            # CRITICAL: Check for zero results
            if len(word_timestamps) == 0 and len(final_text) == 0:
                logger.error("CRITICAL: WebVTT parsing produced zero results!")
                logger.error(f"Original WebVTT content ({len(webvtt_content)} chars):")
                logger.error(webvtt_content)
                if len(lines) == 0:
                    logger.error("WebVTT content split into zero lines")
                else:
                    logger.error(f"WebVTT content had {len(lines)} lines:")
                    for i, line in enumerate(lines[:10]):  # Show first 10 lines
                        logger.error(f"Line {i+1}: '{line}'")
                    if len(lines) > 10:
                        logger.error(f"... and {len(lines) - 10} more lines")
            
            return final_text, word_timestamps
            
        except Exception as e:
            logger.warning(f"Failed to parse WebVTT response: {e}")
            logger.warning("Falling back to plain text extraction")
            
            # Fallback: extract text without timestamps
            plain_text = re.sub(r'WEBVTT.*?\n', '', webvtt_content)
            plain_text = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}', '', plain_text)
            plain_text = re.sub(r'<v\s+[^>]+>', '', plain_text)
            plain_text = ' '.join(plain_text.split())
            
            return plain_text, []
    
    def _parse_timestamp(self, timestamp_str: str) -> float:
        """Parse WebVTT timestamp to seconds - handles malformed Gemini timestamps.
        
        Args:
            timestamp_str: Timestamp in various formats (HH:MM:SS.mmm, MM:SS, MM:SSS, etc.)
            
        Returns:
            Time in seconds as float
        """
        try:
            # Handle Gemini's malformed timestamps like "00:05:700" (should be "00:05:07.000")
            parts = timestamp_str.split(':')
            
            if len(parts) == 3:
                # HH:MM:SS.mmm or HH:MM:SSS format
                hours = int(parts[0])
                minutes = int(parts[1])
                
                # Handle seconds part - could be SS.mmm or SSS (malformed)
                seconds_part = parts[2]
                if '.' in seconds_part:
                    # Standard format: SS.mmm
                    sec_parts = seconds_part.split('.')
                    seconds = int(sec_parts[0])
                    milliseconds = int(sec_parts[1]) if len(sec_parts) > 1 else 0
                else:
                    # Malformed format: treat as raw number and convert intelligently
                    raw_number = int(seconds_part)
                    if raw_number >= 1000:
                        # Looks like milliseconds: 5700 -> 5.7 seconds
                        seconds = raw_number // 1000
                        milliseconds = raw_number % 1000
                    elif raw_number >= 100:
                        # Ambiguous: 700 could be 7.00 sec or 0.7 sec - assume seconds for Gemini
                        seconds = raw_number // 100
                        milliseconds = (raw_number % 100) * 10  # 700 -> 7 sec, 0 ms
                    else:
                        # Small number, treat as seconds
                        seconds = raw_number
                        milliseconds = 0
                        
            elif len(parts) == 2:
                # MM:SS.mmm or MM:SSS format
                hours = 0
                minutes = int(parts[0])
                seconds_part = parts[1]
                
                if '.' in seconds_part:
                    sec_parts = seconds_part.split('.')
                    seconds = int(sec_parts[0])
                    milliseconds = int(sec_parts[1]) if len(sec_parts) > 1 else 0
                else:
                    # Handle MM:SSS format
                    raw_number = int(seconds_part)
                    if raw_number >= 100:
                        seconds = raw_number // 100
                        milliseconds = (raw_number % 100) * 10
                    else:
                        seconds = raw_number
                        milliseconds = 0
                        
            else:
                logger.warning(f"Unexpected timestamp format: '{timestamp_str}'")
                return 0.0
            
            total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
            logger.debug(f"Parsed timestamp '{timestamp_str}' -> {total_seconds:.3f}s")
            return total_seconds
            
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse timestamp '{timestamp_str}': {e}")
            return 0.0
        
    def _estimate_confidence(self, response: Any, text: str) -> float:
        """Estimate confidence score from Gemini response quality."""
        # Gemini doesn't provide explicit confidence scores
        # Estimate based on response characteristics
        
        base_confidence = 0.90  # High baseline for Gemini
        
        # Adjust based on text characteristics
        if len(text) < 10:
            base_confidence -= 0.10  # Very short responses might be incomplete
        elif any(char in text for char in ["[", "]", "(", ")"]):
            base_confidence -= 0.05  # Bracketed text might indicate uncertainty
            
        # Check for finish reason if available
        finish_reason = getattr(response, 'finish_reason', 'complete')
        if finish_reason != 'complete':
            base_confidence -= 0.15
            
        return max(0.0, min(1.0, base_confidence))
        
    def estimate_cost(self, duration_seconds: float) -> float:
        """Estimate Gemini transcription cost in USD."""
        # Gemini 2.0 Flash pricing (estimated)
        # Input: $0.075 per 1M tokens, Output: $0.30 per 1M tokens
        # Audio processing cost varies, estimated at ~$1.20 per hour
        
        hours = duration_seconds / 3600.0
        cost_per_hour = 1.20
        return hours * cost_per_hour
        
    def get_supported_languages(self) -> List[str]:
        """Return list of supported language codes."""
        return self._supported_languages.copy()