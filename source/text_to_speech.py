"""
Text-to-Speech Module for converting text into speech using various providers.

This module provides functionality to convert text into speech using various TTS models.
It supports ElevenLabs, Google, OpenAI and Edge TTS services and handles the conversion process,
including cleaning of input text and merging of audio files.
"""

import logging
import math
import os
import tempfile
from typing import List, Tuple, Optional, Dict, Any, Union
from pydub import AudioSegment
from pydub.silence import split_on_silence

from .tts.base import TTSProvider
from .tts.factory import TTSProviderFactory
from .schemas import SpeakerConfig, TTSConfig  # Import from the new schemas file

logger = logging.getLogger(__name__)


# Default speaker configurations using the Pydantic model
DEFAULT_SPEAKER_1 = SpeakerConfig()
DEFAULT_SPEAKER_2 = SpeakerConfig(voice="default_voice_2")


class TextToSpeech:
    tts_config: TTSConfig
    provider: TTSProvider
    speaker_configs: Dict[int, SpeakerConfig] = (
        None  # Will hold the final SpeakerConfig objects
    )

    def __init__(
        self,
        provider: str = "elevenlabs",
        tts_config: Optional[Union[Dict[str, Any], TTSConfig]] = None,
    ):
        """
        Initialize the TextToSpeech class.

        Args:
            provider (str): The provider to use for text-to-speech conversion.
                Options are 'elevenlabs', 'gemini', 'openai', 'edge' or 'geminimulti'.
                Defaults to 'elevenlabs'.
            tts_config (Optional[Union[Dict, TTSConfig]]): Configuration for TTS settings.
                                       Can be a dictionary or a TTSConfig instance.
                                       If a dictionary is provided, it will be parsed into TTSConfig.
                                       If None, default TTSConfig settings will be used.
        """
        # Instantiate TTSConfig if a dict is passed, or use the instance directly.
        if isinstance(tts_config, dict):
            self.tts_config = TTSConfig(**tts_config)
        elif isinstance(tts_config, TTSConfig):
            self.tts_config = tts_config
        else:
            self.tts_config = TTSConfig()  # Use default if None or invalid type

        processed_speaker_configs = {}

        # Check if user provided speaker configurations and process them
        # Access speaker_configs via TTSConfig attribute
        incoming_configs = self.tts_config.speaker_configs
        if isinstance(
            incoming_configs, dict
        ):  # Still check if it's a dict, as it comes from TTSConfig
            for speaker_id, config_value in incoming_configs.items():
                try:
                    sid = int(speaker_id)
                    if isinstance(config_value, SpeakerConfig):
                        # Use the instance directly if it's already a SpeakerConfig
                        processed_speaker_configs[sid] = config_value
                    elif isinstance(config_value, dict):
                        # Parse the dictionary into a SpeakerConfig instance
                        processed_speaker_configs[sid] = SpeakerConfig(**config_value)
                    else:
                        logger.warning(
                            f"Invalid type for speaker config {sid}: {type(config_value)}. Skipping."
                        )
                        raise ValueError(
                            f"Invalid type for speaker config {sid}: {type(config_value)}"
                        )
                except (ValueError, TypeError) as e:
                    #  logger.warning(f"Error processing speaker config for ID '{speaker_id}': {e}. Skipping.")
                    raise ValueError(
                        f"Error processing speaker config for ID '{speaker_id}': {e}"
                    )

        # If no valid configs were processed or provided, use defaults
        if not processed_speaker_configs:
            self.speaker_configs = {1: DEFAULT_SPEAKER_1, 2: DEFAULT_SPEAKER_2}
        else:
            # Ensure default speakers 1 and 2 are present if not provided by user
            if 1 not in processed_speaker_configs:
                processed_speaker_configs[1] = DEFAULT_SPEAKER_1
            if 2 not in processed_speaker_configs:
                processed_speaker_configs[2] = DEFAULT_SPEAKER_2
            self.speaker_configs = processed_speaker_configs

        # Update the speaker_configs within the TTSConfig instance
        self.tts_config.speaker_configs = self.speaker_configs

        # Initialize provider using factory, passing the TTSConfig instance
        self.provider = TTSProviderFactory.create(
            provider_name=provider, config=self.tts_config
        )

        # Setup directories and config
        self._setup_directories()
        # Access audio_format via TTSConfig attribute
        self.audio_format = self.tts_config.audio_format

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
        segments = self.provider.split_qa(text, self.provider.get_supported_tags())

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

    @classmethod
    def _normalize_audio_segments(
        cls, audio_segments: List[AudioSegment]
    ) -> List[AudioSegment]:
        """
        Normalize a list of audio segments to a consistent loudness level.

        Args:
            audio_segments: A list of AudioSegment objects to normalize.

        Returns:
            A list of normalized AudioSegment objects.
        """
        # Step 1: Calculate the RMS (loudness level) for each segment
        loudness_levels = [segment.rms for segment in audio_segments]

        # Step 2: Determine the target loudness level (mean RMS)
        target_loudness = sum(loudness_levels) / len(loudness_levels)

        # Step 3: Normalize each segment to the target loudness
        normalized_segments = []
        for segment, segment_loudness in zip(audio_segments, loudness_levels):
            gain = 20 * math.log10(target_loudness / segment_loudness)
            normalized_audio = segment.apply_gain(gain)
            normalized_segments.append(normalized_audio)

        return normalized_segments

    @classmethod
    def _split_audio_on_silence(cls, audio: AudioSegment) -> List[AudioSegment]:
        """
        Split the given audio into chunks based on silence detection.

        Args:
            audio: The combined audio segment to be split

        Returns:
            A list of audio chunks split based on silence
        """
        silence_threshold = -40  # Silence threshold in dB
        min_silence_len = 2000  # Minimum silence duration (2 seconds)

        try:
            chunks = split_on_silence(  # Using split_on_silence from pydub.silence
                audio,
                min_silence_len=min_silence_len,
                silence_thresh=silence_threshold,
                keep_silence=500,
            )

            logger.info(
                f"Detected {len(chunks)} chunks in the audio after silence detection."
            )
            return chunks[0] if len(chunks) > 1 else audio
        except Exception as e:
            logger.error(f"Error during silence detection: {str(e)}")
            raise

    def _merge_audio_files(self, audio_files: List[str], output_file: str) -> None:
        """
        Merge the provided audio files sequentially, ensuring questions come before answers,
        and normalize the audio segments to a consistent loudness level.

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

            # Step 1: Load all audio segments
            audio_segments = [
                self._split_audio_on_silence(
                    AudioSegment.from_file(file_path, format=self.audio_format)
                )
                for file_path in audio_files
            ]

            # Step 2: Normalize all audio segments
            normalized_segments = self._normalize_audio_segments(audio_segments)

            # Step 3: Combine all normalized audio segments
            combined = AudioSegment.empty()
            for segment in normalized_segments:
                combined += segment

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # Step 4: Export the combined audio file
            combined.export(output_file, format=self.audio_format)
            logger.info(f"Merged and normalized audio saved to {output_file}")

        except Exception as e:
            logger.error(f"Error merging and normalizing audio files: {str(e)}")
            raise

    def _ensure_directory_exists(self, path: str) -> str:
        """Ensure a directory exists, resolving relative paths to absolute."""
        if not os.path.isabs(path):
            base_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), os.pardir)
            )
            path = os.path.join(base_dir, path)
        os.makedirs(path, exist_ok=True)
        return path

    def _setup_directories(self) -> None:
        """Setup required directories for audio processing."""
        self.output_directories = self.tts_config.output_directories
        temp_dir_path = self.tts_config.temp_audio_dir
        self.temp_audio_dir = self._ensure_directory_exists(temp_dir_path)

        # Create output directories if they don't exist
        for dir_key in ["transcripts", "audio"]:
            dir_path = self.output_directories.get(dir_key)
            if dir_path:
                self.output_directories[dir_key] = self._ensure_directory_exists(
                    dir_path
                )
