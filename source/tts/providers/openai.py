"""OpenAI TTS provider implementation."""

import logging
import openai
from typing import List
from ..base import SpeakerSegment, TTSProvider
from ...schemas import TTSConfig

logger = logging.getLogger(__name__)


class OpenAITTS(TTSProvider):
    """OpenAI Text-to-Speech provider."""

    # Provider-specific SSML tags
    PROVIDER_SSML_TAGS: List[str] = ["break", "emphasis"]

    def __init__(self, config: TTSConfig):
        """
        Initialize OpenAI TTS provider.

        Args:
            config (TTSConfig): Configuration object.
        """
        super().__init__(config)
        api_key = config.api_key
        if api_key:
            openai.api_key = api_key
        elif (
            not openai.api_key
        ):  # Check if it was set via environment variable if not in config
            raise ValueError(
                "OpenAI API key must be provided in config or set in environment"
            )
        # Use values from config or defaults if None
        self.model = config.model or "tts-1-hd"
        self.response_format = config.response_format
        self.streaming = config.streaming
        self.speed = config.speed
        self.language = config.language

        # Validate response format
        valid_formats = ["mp3", "opus", "aac", "flac", "wav", "pcm"]
        if self.response_format not in valid_formats:
            raise ValueError(
                f"Invalid response format: {self.response_format}. Must be one of {valid_formats}."
            )

        # Validate speed
        if not (0.5 <= self.speed <= 2.0):
            raise ValueError(
                f"Invalid speed: {self.speed}. Must be between 0.5 and 2.0."
            )

    def get_supported_tags(self) -> List[str]:
        """Get all supported SSML tags including provider-specific ones."""
        return self.PROVIDER_SSML_TAGS

    def generate_audio(self, segments: List[SpeakerSegment]) -> List[bytes]:
        """
        Generate audio using OpenAI API for all SpeakerSegments in a single call.
        """
        audio_chunks = []
        for segment in segments:
            logger.info(
                f"Generating audio for Speaker {segment.speaker_id}: {segment.text}"
            )
            try:
                response = openai.audio.speech.create(
                    model=self.model,
                    voice=segment.voice_config.voice,
                    input=segment.text,
                    response_format=self.response_format,
                    speed=self.speed,
                    language=self.language,
                    stream=self.streaming,
                )

                if self.streaming:
                    logger.info("Streaming audio in real-time...")
                    audio_chunks.append(b"".join(chunk for chunk in response))
                else:
                    logger.info("Saving audio to memory...")
                    audio_chunks.append(response.content)

            except Exception as e:
                logger.error(
                    f"Failed to generate audio for Speaker {segment.speaker_id}: {str(e)}"
                )
                raise RuntimeError(f"Failed to generate audio: {str(e)}") from e
        return audio_chunks
