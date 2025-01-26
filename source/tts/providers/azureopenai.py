"""Azure OpenAI TTS provider implementation."""

import logging
from openai import AzureOpenAI
from typing import List, Dict, Any
from ..base import SpeakerSegment, TTSProvider

logger = logging.getLogger(__name__)


class AzureOpenAITTS(TTSProvider):
    """Azure OpenAI Text-to-Speech provider."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Azure OpenAI TTS provider.

        Args:
            config (Dict[str, Any]): Configuration dictionary from tts_config.
        """
        self.api_base = config.get("api_base")
        self.api_key = config.get("api_key")
        self.api_version = config.get("api_version", "2025-01-01-preview")
        self.deployment = config.get("deployment")

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

    def generate_audio(self, segments: List[SpeakerSegment]) -> List[bytes]:
        """
        Generate audio using Azure OpenAI TTS API for all SpeakerSegments in a single call.
        """
        audio_chunks = []
        for segment in segments:
            logger.info(
                f"Generating audio for Speaker {segment.speaker_id}: {segment.text}"
            )
            try:
                response = self.client.audio.speech.create(
                    model="gpt-4o-audio-preview",
                    voice=segment.voice_config.get("voice", "alloy"),
                    input=segment.text,
                )
                audio_chunks.append(response.content)
            except Exception as e:
                logger.error(
                    f"Failed to generate audio for Speaker {segment.speaker_id}: {str(e)}"
                )
                raise RuntimeError(f"Failed to generate audio: {str(e)}") from e
        return audio_chunks
