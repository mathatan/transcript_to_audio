"""ElevenLabs TTS provider implementation."""

import logging
from elevenlabs import VoiceSettings, client as elevenlabs_client
from ..base import SpeakerSegment, TTSProvider
from typing import List
from ...schemas import TTSConfig

logger = logging.getLogger(__name__)


class ElevenLabsTTS(TTSProvider):
    def __init__(self, config: TTSConfig):
        """
        Initialize ElevenLabs TTS provider.

        Args:
            config (TTSConfig): Configuration object.
        """
        super().__init__(config)
        if not config.api_key:
            raise ValueError(
                "ElevenLabs API key must be provided in the configuration."
            )
        self.client = elevenlabs_client.ElevenLabs(api_key=config.api_key)
        # Use the model from config or default to "eleven_multilingual_v2" if None
        self.model = config.model or "eleven_multilingual_v2"

    def generate_audio(self, segments: List[SpeakerSegment]) -> List[bytes]:
        """
        Generate audio using ElevenLabs API for all SpeakerSegments in a single call.
        """
        audio_chunks = []
        # Initialize variables for history and request tracking
        previous_request_ids: List[str] = []
        previous_history_id: str | None = None

        for i, segment in enumerate(segments):
            logger.info(
                f"Generating audio for Speaker {segment.speaker_id}: {segment.text}"
            )

            # Determine previous_text and next_text
            previous_text: str | None = segments[i - 1].text if i > 0 else None
            next_text: str | None = (
                segments[i + 1].text if i < len(segments) - 1 else None
            )

            # Prepare voice settings
            voice_settings: VoiceSettings = VoiceSettings(
                stability=segment.voice_config.stability,
                similarity_boost=segment.voice_config.similarity_boost,
                style=segment.voice_config.style,
                use_speaker_boost=segment.voice_config.use_speaker_boost,
            )

            # Generate audio
            audio = self.client.generate(
                text=segment.text,
                voice=segment.voice_config.voice,
                model=self.model,
                previous_text=previous_text,
                next_text=next_text,
                previous_request_ids=previous_request_ids[
                    -3:
                ],  # Use up to 3 previous IDs
                voice_settings=voice_settings,
            )

            # Append audio chunks
            audio_chunks.append(b"".join(chunk for chunk in audio if chunk))

            # Fetch updated history after generation
            history_response = self.client.history.get_all(
                page_size=3,
                start_after_history_item_id=previous_history_id,
            )
            previous_request_ids = previous_request_ids + [
                item.request_id for item in history_response.history if item.request_id
            ]
            if history_response.history:
                previous_history_id = history_response.history[-1].history_item_id
        return audio_chunks

    def get_supported_tags(self) -> List[str]:
        """Get supported SSML tags."""
        return ["lang", "p", "phoneme", "s", "sub"]
