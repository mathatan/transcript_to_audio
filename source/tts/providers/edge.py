"""Edge TTS provider implementation."""

import edge_tts
import os
import tempfile
from typing import List
from ..base import SpeakerSegment, TTSProvider
from ...schemas import TTSConfig


class EdgeTTS(TTSProvider):
    def __init__(self, config: TTSConfig):
        """
        Initialize Edge TTS provider.

        Args:
            config (TTSConfig): Configuration object.
        """
        super().__init__(config)
        # Use the model from config or default to "default" if None
        self.model = config.model or "default"

    def generate_audio(self, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        """
        Generate audio using Edge TTS for all SpeakerSegments in a single call.
        """
        import nest_asyncio
        import asyncio

        # Apply nest_asyncio to allow nested event loops
        nest_asyncio.apply()

        async def _generate(segment: SpeakerSegment) -> bytes:
            communicate = edge_tts.Communicate(segment.text, segment.voice_config.voice)
            # Ensure temp file is created in the configured directory
            with tempfile.NamedTemporaryFile(
                suffix=".mp3", delete=False, dir=self.config.temp_audio_dir
            ) as tmp_file:
                temp_path = tmp_file.name

            try:
                # Save audio to temporary file
                await communicate.save(temp_path)
                # Read the audio data
                with open(temp_path, "rb") as f:
                    segment.audio = f.read()
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return segment

        # Use asyncio to process all segments
        loop = asyncio.get_event_loop()
        return [loop.run_until_complete(_generate(segment)) for segment in segments]

    def get_supported_tags(self) -> List[str]:
        """Get supported SSML tags."""
        return self.COMMON_SSML_TAGS
