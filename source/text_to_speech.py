"""
Text-to-Speech Module for converting text into speech using various providers.

This module provides functionality to convert text into speech using various TTS models.
It supports ElevenLabs, Google, OpenAI and Edge TTS services and handles the conversion process,
including cleaning of input text and merging of audio files.
"""

import logging
import os
import tempfile
from typing import List, Tuple, Optional, Dict, Any
from pydub import AudioSegment

from .tts.base import TTSProvider
from .tts.factory import TTSProviderFactory

logger = logging.getLogger(__name__)


class TextToSpeech:
    provider_config: Dict[str, Any] = None
    tts_config: Dict[str, Any] = None
    provider: TTSProvider

    def __init__(
        self,
        provider: str = "elevenlabs",
        tts_config: Optional[Dict[str, Any]] = {},
    ):
        """
        Initialize the TextToSpeech class.

        Args:
            provider (str): The provider to use for text-to-speech conversion.
                Options are 'elevenlabs', 'gemini', 'openai', 'edge' or 'geminimulti'.
                Defaults to 'elevenlabs'.
            tts_config (Optional[Dict]): Configuration for TTS settings.
        """
        self.tts_config = tts_config
        default_speaker_configs = {
            1: {
                "voice": "default_voice_1",
                "language": "en-US",
                "pitch": "default",
                "speaking_rate": "1.0",
                "stability": 0.75,  # ElevenLabs-specific
                "similarity_boost": 0.85,  # ElevenLabs-specific
                "style": 0,  # ElevenLabs-specific
                "ssml_gender": "NEUTRAL",  # Gemini-specific
            },
            2: {
                "voice": "default_voice_2",
                "language": "en-US",
                "pitch": "default",
                "speaking_rate": "1.0",
                "stability": 0.75,
                "similarity_boost": 0.85,
                "style": 0,
                "ssml_gender": "NEUTRAL",
            },
        }
        self.tts_config["speaker_configs"] = {
            speaker_id: {
                **default_speaker_configs.get(speaker_id, {}),
                **self.tts_config.get("speaker_configs", {}).get(speaker_id, {}),
            }
            for speaker_id in set(default_speaker_configs)
            | set(self.tts_config.get("speaker_configs", {}))
        }

        # Initialize provider using factory
        self.provider = TTSProviderFactory.create(
            provider_name=provider, config=self.tts_config
        )

        # Setup directories and config
        self._setup_directories()
        self.audio_format = self.tts_config.get("audio_format", "mp3")

    def convert_to_speech(self, text: str, output_file: str) -> None:
        """
        Convert input text to speech and save as an audio file.

        Args:
            text (str): Input text to convert to speech.
            output_file (str): Path to save the output audio file.

        Raises:
            ValueError: If the input text is not properly formatted
        """
        cleaned_text = text

        try:
            with tempfile.TemporaryDirectory(dir=self.temp_audio_dir) as temp_dir:
                # Generate audio segments
                audio_segments = self._generate_audio_segments(cleaned_text, temp_dir)

                # Merge audio files into a single output
                self._merge_audio_files(audio_segments, output_file)
                logger.info(f"Audio saved to {output_file}")

        except Exception as e:
            logger.error(f"Error converting text to speech: {str(e)}")
            raise

    def _generate_audio_segments(self, text: str, temp_dir: str) -> List[str]:
        """Generate audio segments for each Q&A pair."""
        # Parse the input text into SpeakerSegment instances
        qa_pairs = self.provider.split_qa(text, self.provider.get_supported_tags())
        segments = [
            segment for pair in qa_pairs for segment in pair
        ]  # Flatten the list of pairs
        audio_files = []

        # Generate audio for all segments in a single call
        audio_data_list = self.provider.generate_audio(segments)

        # Save each audio chunk to a temporary file
        for idx, (segment, audio_data) in enumerate(zip(segments, audio_data_list), 1):
            temp_file = os.path.join(
                temp_dir, f"{idx}_speaker{segment.speaker_id}.{self.audio_format}"
            )
            with open(temp_file, "wb") as f:
                f.write(audio_data)
            audio_files.append(temp_file)

        return audio_files

    def _merge_audio_files(self, audio_files: List[str], output_file: str) -> None:
        """
        Merge the provided audio files sequentially, ensuring questions come before answers.

        Args:
            audio_files: List of paths to audio files to merge
            output_file: Path to save the merged audio file
        """
        try:

            def get_sort_key(file_path: str) -> Tuple[int, int]:
                """
                Create sort key from filename that orders files by index and speaker ID.
                Example filenames: "1_speaker1.mp3", "2_speaker2.mp3"
                """
                basename = os.path.basename(file_path)
                idx = int(basename.split("_")[0])
                speaker_id = int(
                    basename.split("_")[1].replace("speaker", "").split(".")[0]
                )
                return (idx, speaker_id)

            # Sort files by index and type (question/answer)
            audio_files.sort(key=get_sort_key)

            # Create empty audio segment
            combined = AudioSegment.empty()

            # Add each audio file to the combined segment
            for file_path in audio_files:
                combined += AudioSegment.from_file(file_path, format=self.audio_format)

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # Export the combined audio
            combined.export(output_file, format=self.audio_format)
            logger.info(f"Merged audio saved to {output_file}")

        except Exception as e:
            logger.error(f"Error merging audio files: {str(e)}")
            raise

    def _setup_directories(self) -> None:
        """Setup required directories for audio processing."""
        self.output_directories = self.tts_config.get("output_directories", {})
        temp_dir = (
            self.tts_config.get("temp_audio_dir", "data/audio/tmp/")
            .rstrip("/")
            .split("/")
        )
        self.temp_audio_dir = os.path.join(*temp_dir)
        base_dir = os.path.abspath(os.path.dirname(__file__))
        self.temp_audio_dir = os.path.join(base_dir, self.temp_audio_dir)

        os.makedirs(self.temp_audio_dir, exist_ok=True)

        # Create directories if they don't exist
        for dir_path in [
            self.output_directories.get("transcripts"),
            self.output_directories.get("audio"),
            self.temp_audio_dir,
        ]:
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path)
