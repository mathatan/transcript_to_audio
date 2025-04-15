"""Azure OpenAI TTS provider implementation."""

import logging
from openai import AzureOpenAI
from typing import List
from ..base import SpeakerSegment, TTSProvider
from ...schemas import TTSConfig

logger = logging.getLogger("transcript_to_audio_logger")


class AzureOpenAITTS(TTSProvider):
    """Azure OpenAI Text-to-Speech provider."""

    def __init__(self, config: TTSConfig):
        """
        Initialize Azure OpenAI TTS provider.

        Args:
            config (TTSConfig): Configuration object.
        """
        super().__init__(config)
        self.api_base = config.api_base
        self.api_key = config.api_key
        self.api_version = config.api_version
        self.deployment = config.deployment

        if not all([self.api_base, self.api_key, self.deployment]):
            raise ValueError(
                "Azure OpenAI API base, API key, and deployment name must be provided."
            )

        self.client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.api_base,
            api_key=self.api_key,
        )

    def get_supported_tags(self) -> List[str]:
        """Get supported SSML tags."""
        return self.COMMON_SSML_TAGS

    def generate_audio(self, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        """
        Generate audio using Azure OpenAI TTS API for all SpeakerSegments in a single call.
        """
        for segment in segments:
            logger.info(
                f"Generating audio for Speaker {segment.speaker_id}: {segment.text}"
            )
            try:
                response = self.client.audio.speech.create(
                    model=self.config.model or "gpt-4o-audio-preview",
                    voice=segment.voice_config.voice or "alloy",
                    input=segment.text,
                )
                segment.audio = response.content
            except Exception as e:
                logger.error(
                    f"Failed to generate audio for Speaker {segment.speaker_id}: {str(e)}"
                )
                raise RuntimeError(f"Failed to generate audio: {str(e)}") from e
        return segments
