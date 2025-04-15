"""Google Cloud Text-to-Speech provider implementation for single speaker."""

from google.cloud import texttospeech
from typing import List
from ..base import SpeakerSegment, TTSProvider
from ...schemas import TTSConfig
import logging

logger = logging.getLogger("transcript_to_audio_logger")


class GeminiTTS(TTSProvider):
    """Google Cloud Text-to-Speech provider for single speaker."""

    def __init__(self, config: TTSConfig):
        """
        Initialize Google Cloud TTS provider.

        Args:
            config (TTSConfig): Configuration object.
        """
        super().__init__(config)
        # Use the model from config or default to "en-US-Journey-F" if None
        self.model = config.model or "en-US-Journey-F"
        try:
            self.client = texttospeech.TextToSpeechClient(
                client_options={"api_key": config.api_key}
            )
            logger.info("Successfully initialized GeminiTTS client")
        except Exception as e:
            logger.error(f"Failed to initialize Google TTS client: {str(e)}")
            raise

    def generate_audio(self, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        """
        Generate audio using Google Cloud TTS API for all SpeakerSegments in a single call.
        """
        for segment in segments:
            logger.info(
                f"Generating audio for Speaker {segment.speaker_id}: {segment.text}"
            )

            try:
                # Create synthesis input
                synthesis_input = texttospeech.SynthesisInput(text=segment.text)

                # Set voice parameters
                # Ensure ssml_gender is uppercase and handle potential None
                ssml_gender = (segment.voice_config.ssml_gender or "NEUTRAL").upper()
                ssml_gender_enum = getattr(
                    texttospeech.SsmlVoiceGender,
                    ssml_gender,
                    texttospeech.SsmlVoiceGender.NEUTRAL,
                )

                voice_params = texttospeech.VoiceSelectionParams(
                    language_code=segment.voice_config.language,
                    name=segment.voice_config.voice,
                    ssml_gender=ssml_gender_enum,
                )

                # Set audio config
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.MP3
                )

                # Generate speech
                response = self.client.synthesize_speech(
                    request={
                        "input": synthesis_input,
                        "voice": voice_params,
                        "audio_config": audio_config,
                    }
                )

                segment.audio = response.audio_content

            except Exception as e:
                logger.error(
                    f"Failed to generate audio for Speaker {segment.speaker_id}: {str(e)}"
                )
                raise RuntimeError(f"Failed to generate audio: {str(e)}") from e

        return segments

    def get_supported_tags(self) -> List[str]:
        """Get supported SSML tags."""
        return self.COMMON_SSML_TAGS

    def validate_parameters(self, text: str, voice: str, model: str) -> None:
        """
        Validate input parameters before generating audio.

        Args:
            text (str): Input text
            voice (str): Voice ID/name
            model (str): Model name

        Raises:
            ValueError: If parameters are invalid
        """
        super().validate_parameters(text, voice, model)

        if not text:
            raise ValueError("Text cannot be empty")

        if not voice:
            raise ValueError("Voice must be specified")
