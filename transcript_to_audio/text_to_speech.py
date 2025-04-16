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
import uuid
from pydub import AudioSegment
from pydub.silence import split_on_silence

from transcript_to_audio.tts.providers.geminimulti import GeminiMultiTTS

from .tts.base import TTSProvider
from .tts.factory import TTSProviderFactory
from .schemas import (
    SpeakerSegment,
    SpeakerConfig,
    TTSConfig,
)  # Import from the new schemas file

logger = logging.getLogger("transcript_to_audio_logger")


# Default speaker configurations using the Pydantic model
DEFAULT_SPEAKER_1 = SpeakerConfig()
DEFAULT_SPEAKER_2 = SpeakerConfig(voice="default_voice_2")


class TextToSpeech:
    provider: TTSProvider
    tts_config: TTSConfig

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

        # Initialize provider using factory, passing the TTSConfig instance
        self.provider = TTSProviderFactory.create(
            provider_name=provider, config=self.tts_config
        )

        # Setup directories and config
        self._setup_directories()
        # Access audio_format via TTSConfig attribute
        self.audio_format = self.tts_config.audio_format

    def convert_to_speech(
        self,
        text: str,
        speaker_configs: Dict[int, SpeakerConfig] = {
            1: DEFAULT_SPEAKER_1,
            2: DEFAULT_SPEAKER_2,
        },
        output_file: Optional[str] = None,
        save_to_file: bool = False,
    ) -> Tuple[str, AudioSegment]:
        """
        Convert input text to speech and save as an audio file.

        Args:
            text (str): Input text to convert to speech.
            output_file (str): Path to save the output audio file.

        Raises:
            ValueError: If the input text is not properly formatted
        """

        for key, config in speaker_configs.items():
            if not isinstance(config, SpeakerConfig):
                speaker_configs[key] = SpeakerConfig(**config)

        try:
            with tempfile.TemporaryDirectory(dir=self.temp_audio_dir) as temp_dir:
                # Generate audio segments
                audio_segments = self._generate_audio_segments(
                    text, speaker_configs, temp_dir
                )

                # Merge audio files into a single output
                segments, audio = self._merge_audio_files(
                    audio_segments, output_file, save_to_file
                )

                new_transcript = "\n".join([segment.to_tag() for segment in segments])
                if save_to_file:
                    logger.info(f"Audio saved to {output_file}")
                    transcript_file = (
                        f"{os.path.splitext(output_file)[0]}_transcript.txt"
                    )

                    with open(
                        transcript_file, "w", encoding="utf-8"
                    ) as transcript_output:
                        transcript_output.write(new_transcript)
                    logger.info(f"Transcript saved to {transcript_file}")

                return (new_transcript, audio)

        except Exception as e:
            logger.error(f"Error converting text to speech: {str(e)}")
            raise

    def _generate_audio_segments(
        self, text: str, speaker_configs: Dict[int, SpeakerConfig], temp_dir: str
    ) -> Tuple[List[SpeakerSegment], Union[str | None]]:
        """Generate audio segments for each Q&A pair."""
        # Parse the input text into SpeakerSegment instances
        segments = self.provider.split_qa(
            text, speaker_configs, self.provider.get_supported_tags()
        )

        # audio_files = []
        audio_file = None

        if not isinstance(self.provider, GeminiMultiTTS):
            generated_id = str(uuid.uuid4())
            # Generate audio for all segments in a single call
            segments = self.provider.generate_audio(segments)

            # Save each audio chunk to a temporary file
            for idx, segment in enumerate(segments):
                if segment.audio:
                    temp_file = os.path.join(
                        temp_dir,
                        f"{generated_id}_{idx}_speaker{segment.speaker_id}.{self.audio_format}",
                    )
                    with open(temp_file, "wb") as f:
                        f.write(segment.audio)
                    segment.audio_file = temp_file
        else:
            audio = self.provider.generate_joint_audio(segments)
            generated_id = str(uuid.uuid4())
            audio_file = os.path.join(
                temp_dir, f"{generated_id}_full_audio.{self.audio_format}"
            )
            with open(audio_file, "wb") as f:
                f.write(audio)
        return [segments, audio_file]

    @classmethod
    def _normalize_audio_segments(
        cls, audio_segments: List[Tuple[Union[SpeakerSegment, None], AudioSegment]]
    ) -> List[AudioSegment]:
        """
        Normalize a list of audio segments to a consistent loudness level.

        Args:
            audio_segments: A list of AudioSegment objects to normalize.

        Returns:
            A list of normalized AudioSegment objects.
        """
        # Step 1: Calculate the RMS (loudness level) for each segment
        loudness_levels = [segment[1].rms for segment in audio_segments]

        # Step 2: Determine the target loudness level (mean RMS)
        target_loudness = sum(loudness_levels) / len(loudness_levels)

        # Step 3: Normalize each segment to the target loudness
        normalized_segments = []
        for segment_tuple, segment_loudness in zip(audio_segments, loudness_levels):

            gain = 20 * math.log10(target_loudness / segment_loudness)
            normalized_audio = segment_tuple[1].apply_gain(gain)
            if segment_tuple[0] is not None:
                segment_tuple[0].audio_segment = normalized_audio

            normalized_segments.append(normalized_audio)

        return normalized_segments

    def _split_audio_on_silence(
        self, audio: AudioSegment, speaker_segment: SpeakerSegment
    ) -> Tuple[SpeakerSegment, AudioSegment]:
        """
        Split the given audio into chunks based on silence detection.

        Args:
            audio: The combined audio segment to be split

        Returns:
            A list of audio chunks split based on silence
        """
        if (
            speaker_segment.voice_config.use_emote
            and speaker_segment.parameters.get("emote") is not None
        ):
            silence_threshold = -40  # Silence threshold in dB
            min_silence_len = (
                int(round(float(speaker_segment.voice_config.emote_pause) * 1000))
                or 2000
            )

            try:
                chunks = split_on_silence(  # Using split_on_silence from pydub.silence
                    audio,
                    min_silence_len=min_silence_len,
                    silence_thresh=silence_threshold,
                    keep_silence=speaker_segment.voice_config.emote_merge_pause or 500,
                )

                logger.info(
                    f"Detected {len(chunks)} chunks in the audio after silence detection."
                )
                if len(chunks) > 1:
                    # Drop the last chunk and concatenate the remaining ones
                    merged_audio = sum(chunks[:-1])
                    speaker_segment.audio_segment = merged_audio
                    return (speaker_segment, merged_audio)
                speaker_segment.audio_segment = audio
                return (
                    speaker_segment,
                    audio,
                )  # Return the original audio if only one chunk
            except Exception as e:
                logger.error(f"Error during silence detection: {str(e)}")
                raise
        else:
            speaker_segment.audio_segment = audio
            return (speaker_segment, audio)

    def _merge_audio_files(
        self,
        audio_files: tuple[List[SpeakerSegment], Union[str | None]],
        output_file: Optional[str],
        save_to_file: bool = False,
    ) -> Tuple[List[SpeakerSegment], AudioSegment]:
        """
        Merge the provided audio files sequentially, ensuring questions come before answers,
        and normalize the audio segments to a consistent loudness level.

        Args:
            audio_files: Tuple of list of speaker segments with files to merge and audio file (if GeminiMultiSpeaker)
            output_file: Path to save the merged audio file
        """

        try:
            if audio_files[1] is not None:
                audio_segments: List[
                    Tuple[Union[SpeakerSegment, None], AudioSegment]
                ] = [
                    (
                        None,
                        AudioSegment.from_file(
                            audio_files[1], format=self.audio_format
                        ),
                    )
                ]
            else:
                audio_segments: List[
                    Tuple[Union[SpeakerSegment, None], AudioSegment]
                ] = [
                    self._split_audio_on_silence(
                        AudioSegment.from_file(
                            segment.audio_file, format=self.audio_format
                        ),
                        segment,
                    )
                    for segment in audio_files[0]
                ]

            # Step 2: Normalize all audio segments
            normalized_segments = self._normalize_audio_segments(audio_segments)

            # Step 3: Combine all normalized audio segments
            combined = AudioSegment.empty()
            for segment in normalized_segments:
                combined += segment

            segments: List[SpeakerSegment] = []
            cur_time = 0
            for segment in audio_files[0]:
                segment.audio_length = segment.audio_segment.__len__()
                segment.start_time = cur_time
                cur_time += segment.audio_length
                segment.end_time = cur_time
                segments.append(segment)

            if save_to_file:
                # Ensure output directory exists
                os.makedirs(os.path.dirname(output_file), exist_ok=True)

                # Step 4: Export the combined audio file
                combined.export(output_file, format=self.audio_format)
                logger.info(f"Merged and normalized audio saved to {output_file}")

            return (segments, combined)

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
