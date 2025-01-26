"""Google Cloud Text-to-Speech provider implementation."""

from google.cloud import texttospeech
from typing import List, Dict, Any
from ..base import SpeakerSegment, TTSProvider
import re
import logging
from io import BytesIO
from pydub import AudioSegment

logger = logging.getLogger(__name__)


class GeminiMultiTTS(TTSProvider):
    """Google Cloud Text-to-Speech provider with multi-speaker support."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Google Cloud TTS provider with multi-speaker support.

        Args:
            config (Dict[str, Any]): Configuration dictionary from tts_config.
        """
        self.model = config.get("model", "en-US-Studio-MultiSpeaker")
        try:
            self.client = texttospeech.TextToSpeechClient(
                client_options={"api_key": config.get("api_key")}
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
        Merge multiple MP3 audio chunks into a single audio file.

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
            # Initialize combined audio with first chunk
            combined = None
            valid_chunks = []

            for i, chunk in enumerate(audio_chunks):
                try:
                    # Ensure chunk is not empty
                    if not chunk or len(chunk) == 0:
                        logger.warning(f"Skipping empty chunk {i}")
                        continue

                    # Save chunk to temporary file for ffmpeg to process
                    temp_file = f"temp_chunk_{i}.mp3"
                    with open(temp_file, "wb") as f:
                        f.write(chunk)

                    # Create audio segment from temp file
                    try:
                        segment = AudioSegment.from_file(temp_file, format="mp3")
                        if len(segment) > 0:
                            valid_chunks.append(segment)
                            logger.debug(f"Successfully processed chunk {i}")
                        else:
                            logger.warning(f"Zero-length segment in chunk {i}")
                    except Exception as e:
                        logger.error(f"Error processing chunk {i}: {str(e)}")

                    # Clean up temp file
                    import os

                    try:
                        os.remove(temp_file)
                    except Exception as e:
                        logger.warning(
                            f"Failed to remove temp file {temp_file}: {str(e)}"
                        )

                except Exception as e:
                    logger.error(f"Error handling chunk {i}: {str(e)}")
                    continue

            if not valid_chunks:
                raise RuntimeError("No valid audio chunks to merge")

            # Merge valid chunks
            combined = valid_chunks[0]
            for segment in valid_chunks[1:]:
                combined = combined + segment

            # Export with specific parameters
            output = BytesIO()
            combined.export(output, format="mp3", codec="libmp3lame", bitrate="320k")

            result = output.getvalue()
            if len(result) == 0:
                raise RuntimeError("Export produced empty output")

            return result

        except Exception as e:
            logger.error(f"Audio merge failed: {str(e)}", exc_info=True)
            # If merging fails, return the first valid chunk as fallback
            if audio_chunks:
                return audio_chunks[0]
            raise RuntimeError(
                f"Failed to merge audio chunks and no valid fallback found: {str(e)}"
            )

    def generate_audio(self, segments: List[SpeakerSegment]) -> List[bytes]:
        """
        Generate audio using Google Cloud TTS API with multi-speaker support.
        Handles all SpeakerSegment instances in a single call.
        """
        logger.info(f"Starting audio generation for {len(segments)} segments")
        audio_chunks = []

        try:
            # Create multi-speaker markup
            multi_speaker_markup = texttospeech.MultiSpeakerMarkup(
                turns=[
                    texttospeech.MultiSpeakerMarkup.Turn(
                        text=segment.text.strip(),
                        speaker=segment.voice_config["voice"],
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

            audio_chunks.append(response.audio_content)

            return audio_chunks

        except Exception as e:
            logger.error(f"Failed to generate audio: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to generate audio: {str(e)}") from e

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
