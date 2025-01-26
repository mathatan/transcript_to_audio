"""
Text-to-Speech Module for converting text into speech using various providers.

This module provides functionality to convert text into speech using various TTS models.
It supports ElevenLabs, Google, OpenAI and Edge TTS services and handles the conversion process,
including cleaning of input text and merging of audio files.
"""

import io
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
        model: str = None,
        api_key: Optional[str] = None,
        tts_config: Optional[Dict[str, Any]] = {},
    ):
        """
        Initialize the TextToSpeech class.

        Args:
                        model (str): The model to use for
                            text-to-speech conversion.
                            Options are 'elevenlabs', 'gemini', 'openai',
                            'edge' or 'geminimulti'. Defaults to 'openai'.
                        api_key (Optional[str]): API key for the selected
                            text-to-speech service.
                        tts_config (Optional[Dict]):
                            Configuration for tts settings.
        """
        self.tts_config = tts_config

        # Get API key from config if not provided
        if not api_key:
            raise Exception("Api key should be provided")

        # Initialize provider using factory
        self.provider = TTSProviderFactory.create(
            provider_name=model, api_key=api_key, model=model
        )

        # Setup directories and config
        self._setup_directories()
        self.audio_format = self.tts_config.get("audio_format", "mp3")

    def _get_provider_config(self) -> Dict[str, Any]:
        """Get provider-specific configuration."""
        # Get provider name in lowercase without 'TTS' suffix
        provider_name = self.provider.__class__.__name__.lower().replace("tts", "")

        # Get provider config from tts_config
        provider_config = self.tts_config.get(provider_name, {})

        # If provider config is empty, try getting from default config
        if not provider_config:
            provider_config = {
                "model": self.tts_config.get("default_model"),
                "default_voices": {
                    "question": self.tts_config.get("default_voice_question"),
                    "answer": self.tts_config.get("default_voice_answer"),
                },
            }

        logger.debug(f"Using provider config: {provider_config}")
        return provider_config

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

            if (
                "multi" in self.provider.model.lower()
            ):  # refactor: We should have instead MultiSpeakerTTS and SingleSpeakerTTS classes
                audio_data_list = self.provider.generate_audio(
                    cleaned_text,
                    voice="S",
                    model="en-US-Studio-MultiSpeaker",
                    voice2="R",
                )

                try:
                    # First verify we have data
                    if not audio_data_list:
                        raise ValueError("No audio data chunks provided")

                    logger.info(
                        f"Starting audio processing with {len(audio_data_list)} chunks"
                    )
                    combined = AudioSegment.empty()

                    for i, chunk in enumerate(audio_data_list):
                        # Save chunk to temporary file
                        # temp_file = "./tmp.mp3"
                        # with open(temp_file, "wb") as f:
                        #    f.write(chunk)

                        segment = AudioSegment.from_file(io.BytesIO(chunk))
                        logger.info(
                            f"################### Loaded chunk {i}, duration: {len(segment)}ms"
                        )

                        combined += segment

                    # Export with high quality settings
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    combined.export(
                        output_file,
                        format=self.audio_format,
                        codec="libmp3lame",
                        bitrate="320k",
                    )

                except Exception as e:
                    logger.error(f"Error during audio processing: {str(e)}")
                    raise
            else:
                with tempfile.TemporaryDirectory(dir=self.temp_audio_dir) as temp_dir:
                    audio_segments = self._generate_audio_segments(
                        cleaned_text, temp_dir
                    )
                    self._merge_audio_files(audio_segments, output_file)
                    logger.info(f"Audio saved to {output_file}")

        except Exception as e:
            logger.error(f"Error converting text to speech: {str(e)}")
            raise

    def _generate_audio_segments(self, text: str, temp_dir: str) -> List[str]:
        """Generate audio segments for each Q&A pair."""
        qa_pairs = self.provider.split_qa(text, self.provider.get_supported_tags())
        audio_files = []
        provider_config = self._get_provider_config()

        for idx, (question, answer) in enumerate(qa_pairs, 1):
            for speaker_type, content in [("question", question), ("answer", answer)]:
                temp_file = os.path.join(
                    temp_dir, f"{idx}_{speaker_type}.{self.audio_format}"
                )
                voice = provider_config.get("default_voices", {}).get(speaker_type)
                model = provider_config.get("model")

                audio_data = self.provider.generate_audio(content, voice, model)
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
                Create sort key from filename that puts questions before answers.
                Example filenames: "1_question.mp3", "1_answer.mp3"
                """
                basename = os.path.basename(file_path)
                # Extract the index number and type (question/answer)
                idx = int(basename.split("_")[0])
                is_answer = basename.split("_")[1].startswith("answer")
                return (
                    idx,
                    1 if is_answer else 0,
                )  # Questions (0) come before answers (1)

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
