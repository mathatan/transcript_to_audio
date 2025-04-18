"""Google Cloud Text-to-Speech provider implementation."""

import os  # Added
import tempfile  # Added
from google.cloud import texttospeech
from typing import List
from ..base import SpeakerSegment, TTSProvider
from ...schemas import TTSConfig
import re
import logging
from io import BytesIO
from pydub import AudioSegment

logger = logging.getLogger("transcript_to_audio_logger")


class GeminiMultiTTS(TTSProvider):
    """Google Cloud Text-to-Speech provider with multi-speaker support."""

    def __init__(self, config: TTSConfig):
        """
        Initialize Google Cloud TTS provider with multi-speaker support.

        Args:
            config (TTSConfig): Configuration object.
        """
        super().__init__(config)
        # Use the model from config or default to "en-US-Studio-MultiSpeaker" if None
        self.model = config.model or "en-US-Studio-MultiSpeaker"
        try:
            self.client = texttospeech.TextToSpeechClient(
                client_options={"api_key": config.api_key}
            )
            logger.info("Successfully initialized GeminiMultiTTS client")
        except Exception as e:
            logger.error(f"Failed to initialize GeminiMultiTTS client: {str(e)}")
            raise

    def chunk_text(self, text: str, max_bytes: int = 1300) -> List[str]:
        """
        Split text into chunks that fit within Google TTS byte limit while preserving speaker tags.

        Args:
            text (str): Input text with Person1/Person2 tags
            max_bytes (int): Maximum bytes per chunk

        Returns:
            List[str]: List of text chunks with proper speaker tags preserved
        """
        logger.debug(f"Starting chunk_text with text length: {len(text)} bytes")

        # Split text into tagged sections, preserving both Person1 and Person2 tags
        pattern = r"(<Person[12]>.*?</Person[12]>)"
        sections = re.split(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        sections = [s.strip() for s in sections if s.strip()]
        logger.debug(f"Split text into {len(sections)} sections")

        chunks = []
        current_chunk = ""

        for section in sections:
            # Extract speaker tag and content if this is a tagged section
            tag_match = re.match(
                r"<(Person[12])>(.*?)</Person[12]>",
                section,
                flags=re.DOTALL | re.IGNORECASE,
            )

            if tag_match:
                speaker_tag = tag_match.group(1)  # Will be either Person1 or Person2
                content = tag_match.group(2).strip()

                # Test if adding this entire section would exceed limit
                test_chunk = current_chunk
                if current_chunk:
                    test_chunk += f"<{speaker_tag}>{content}</{speaker_tag}>"
                else:
                    test_chunk = f"<{speaker_tag}>{content}</{speaker_tag}>"

                if len(test_chunk.encode("utf-8")) > max_bytes and current_chunk:
                    # Store current chunk and start new one
                    chunks.append(current_chunk)
                    current_chunk = f"<{speaker_tag}>{content}</{speaker_tag}>"
                else:
                    # Add to current chunk
                    current_chunk = test_chunk

        # Add final chunk if it exists
        if current_chunk:
            chunks.append(current_chunk)

        logger.info(f"Created {len(chunks)} chunks from input text")
        return chunks

    def split_turn_text(self, text: str, max_chars: int = 500) -> List[str]:
        """
        Split turn text into smaller chunks at sentence boundaries.

        Args:
            text (str): Text content of a single turn
            max_chars (int): Maximum characters per chunk

        Returns:
            List[str]: List of text chunks
        """
        if len(text) <= max_chars:
            return [text]

        chunks = []
        sentences = re.split(r"([.!?]+(?:\s+|$))", text)
        sentences = [s for s in sentences if s]

        current_chunk = ""
        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            separator = sentences[i + 1] if i + 1 < len(sentences) else ""
            complete_sentence = sentence + separator

            if len(current_chunk) + len(complete_sentence) > max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = complete_sentence
                else:
                    # If a single sentence is too long, split at word boundaries
                    words = complete_sentence.split()
                    temp_chunk = ""
                    for word in words:
                        if len(temp_chunk) + len(word) + 1 > max_chars:
                            chunks.append(temp_chunk.strip())
                            temp_chunk = word
                        else:
                            temp_chunk += " " + word if temp_chunk else word
                    current_chunk = temp_chunk
            else:
                current_chunk += complete_sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def merge_audio(self, audio_chunks: List[bytes]) -> bytes:
        """
        Merge multiple MP3 audio chunks into a single audio file using temporary files.

        Args:
            audio_chunks (List[bytes]): List of MP3 audio data

        Returns:
            bytes: Combined MP3 audio data
        """
        if not audio_chunks:
            return b""

        if len(audio_chunks) == 1:
            return audio_chunks[0]

        try:
            combined = None
            valid_chunks = []
            temp_files_to_clean = []  # Keep track of temp files

            for i, chunk in enumerate(audio_chunks):
                temp_file_path = None
                try:
                    if not chunk or len(chunk) == 0:
                        logger.warning(f"Skipping empty chunk {i}")
                        continue

                    # Create a named temporary file in the configured directory
                    with tempfile.NamedTemporaryFile(
                        suffix=".mp3", delete=False, dir=self.config.temp_audio_dir
                    ) as tmp_file:
                        temp_file_path = tmp_file.name
                        tmp_file.write(chunk)
                        temp_files_to_clean.append(
                            temp_file_path
                        )  # Add to cleanup list

                    # Create audio segment from the temporary file
                    segment = AudioSegment.from_file(temp_file_path, format="mp3")
                    if len(segment) > 0:
                        valid_chunks.append(segment)
                        logger.debug(
                            f"Successfully processed chunk {i} using {temp_file_path}"
                        )
                    else:
                        logger.warning(f"Zero-length segment in chunk {i}")

                except Exception as e:
                    logger.error(f"Error processing chunk {i}: {str(e)}")
                    # Attempt cleanup even if processing failed for this chunk
                    if temp_file_path and os.path.exists(temp_file_path):
                        try:
                            os.remove(temp_file_path)
                            if temp_file_path in temp_files_to_clean:
                                temp_files_to_clean.remove(temp_file_path)
                        except OSError as rm_err:
                            logger.warning(
                                f"Could not remove temp file {temp_file_path} after error: {rm_err}"
                            )
                    continue  # Continue to next chunk

            if not valid_chunks:
                raise RuntimeError("No valid audio chunks to merge")

            # Merge valid chunks
            combined = valid_chunks[0]
            for segment in valid_chunks[1:]:
                combined = combined + segment

            # Export combined audio
            output = BytesIO()
            combined.export(output, format="mp3", codec="libmp3lame", bitrate="320k")
            result = output.getvalue()

            if len(result) == 0:
                raise RuntimeError("Export produced empty output")

            return result

        except Exception as e:
            logger.error(f"Audio merge failed: {str(e)}", exc_info=True)
            # Fallback logic remains the same
            if audio_chunks:
                return audio_chunks[0]
            raise RuntimeError(
                f"Failed to merge audio chunks and no valid fallback found: {str(e)}"
            )
        finally:
            # Final cleanup of all successfully created temp files
            for temp_file_path in temp_files_to_clean:
                if os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                        logger.debug(f"Cleaned up temp file: {temp_file_path}")
                    except OSError as e:
                        logger.warning(
                            f"Could not remove temp file {temp_file_path} during final cleanup: {e}"
                        )

    def generate_joint_audio(self, segments: List[SpeakerSegment]) -> bytes:
        """
        Generate audio using Google Cloud TTS API with multi-speaker support.
        Handles all SpeakerSegment instances in a single call.
        """
        logger.info(f"Starting audio generation for {len(segments)} segments")

        try:
            # Create multi-speaker markup
            multi_speaker_markup = texttospeech.MultiSpeakerMarkup(
                turns=[
                    texttospeech.MultiSpeakerMarkup.Turn(
                        text=segment.text.strip(),
                        speaker=segment.voice_config.voice,
                    )
                    for segment in segments
                ]
            )

            # Create synthesis input with multi-speaker markup
            synthesis_input = texttospeech.SynthesisInput(
                multi_speaker_markup=multi_speaker_markup
            )

            # Set voice parameters
            voice_params = texttospeech.VoiceSelectionParams(
                language_code="en-US", name=self.model
            )

            # Set audio config
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )

            # Generate speech
            logger.debug("Calling synthesize_speech API")
            response = self.client.synthesize_speech(
                input=synthesis_input, voice=voice_params, audio_config=audio_config
            )

            return response.audio_content

        except Exception as e:
            # logger.error(f"Failed to generate audio: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to generate audio: {str(e)}")  # from e

    def get_supported_tags(self) -> List[str]:
        """Get supported SSML tags."""
        # Add any Google-specific SSML tags to the common ones
        return self.COMMON_SSML_TAGS

    def validate_parameters(self, text: str, voice: str, model: str) -> None:
        """
        Validate input parameters before generating audio.

        Args:
            text (str): Input text
            voice (str): Voice ID
            model (str): Model name

        Raises:
            ValueError: If parameters are invalid
        """
        super().validate_parameters(text, voice, model)

        # Additional validation for multi-speaker model
        if model != "en-US-Studio-MultiSpeaker":
            raise ValueError(
                "Google Multi-speaker TTS requires model='en-US-Studio-MultiSpeaker'"
            )
