"""Abstract base class for Text-to-Speech providers."""

from abc import ABC, abstractmethod
from typing import List, ClassVar, Tuple, Dict
import re


class SpeakerSegment:
    """
    Represents a segment of text associated with a specific speaker.

    Attributes:
        speaker_id (int): The ID of the speaker (e.g., 1 for <speaker1>).
        parameters (dict): Additional parameters extracted from the tag (e.g., {"param": "value"}).
        text (str): The text content of the segment.
        voice_config (dict): The voice configuration for the speaker.
    """

    def __init__(
        self,
        speaker_id: int,
        parameters: Dict[str, str],
        text: str,
        voice_config: Dict[str, str],
    ):
        self.speaker_id = speaker_id
        self.parameters = parameters
        self.text = text
        self.voice_config = voice_config

    def __repr__(self):
        return (
            f"SpeakerSegment(speaker_id={self.speaker_id}, "
            f"parameters={self.parameters}, text={self.text}, voice_config={self.voice_config})"
        )


class TTSProvider(ABC):
    """Abstract base class that defines the interface for TTS providers."""

    # Common SSML tags supported by most providers
    COMMON_SSML_TAGS: ClassVar[List[str]] = ["lang", "p", "phoneme", "s", "sub"]

    @abstractmethod
    def generate_audio(self, segments: List[SpeakerSegment]) -> List[bytes]:
        """
        Generate audio for a list of SpeakerSegment objects using the provider's API.

        Args:
            segments: List of SpeakerSegment objects containing text and voice configurations.

        Returns:
            List of audio data as bytes for each segment.

        Raises:
            ValueError: If invalid parameters are provided.
            RuntimeError: If audio generation fails.
        """
        raise NotImplementedError(
            "Subclasses must implement the generate_audio method."
        )

    def get_supported_tags(self) -> List[str]:
        """
        Get set of SSML tags supported by this provider.

        Returns:
            Set of supported SSML tag names
        """
        return self.COMMON_SSML_TAGS.copy()

    def validate_parameters(
        self, text: str, voice: str, model: str, voice2: str = None
    ) -> None:
        """
        Validate input parameters before generating audio.

        Raises:
            ValueError: If any parameter is invalid
        """
        if not text:
            raise ValueError("Text cannot be empty")
        if not voice:
            raise ValueError("Voice must be specified")
        if not model:
            raise ValueError("Model must be specified")

    def split_qa(
        self, input_text: str, supported_tags: List[str] = None
    ) -> List[Tuple[SpeakerSegment, SpeakerSegment]]:
        """
        Parse input text into tuples of SpeakerSegment instances.

        Args:
            input_text (str): The input text containing <speakerN> tags.
            supported_tags (List[str]): Supported SSML tags.

        Returns:
            List[Tuple[SpeakerSegment, SpeakerSegment]]: A list of tuples containing SpeakerSegment instances.
        """
        # Retrieve predefined configurations from tts_config
        predefined_configs = self.tts_config.get("speaker_configs", {})

        # Clean the input text
        input_text = self.clean_tss_markup(input_text, supported_tags=supported_tags)

        # Regular expression pattern to match <speakerN> tags
        pattern = r"<speaker(\d+)(.*?)>(.*?)</speaker\1>"
        matches = re.findall(pattern, input_text, re.DOTALL | re.IGNORECASE)

        segments = []
        for speaker_id, params, text in matches:
            speaker_id = int(speaker_id)

            # Parse parameters into a dictionary
            param_dict = dict(re.findall(r'(\w+)="(.*?)"', params))

            # Retrieve default configuration for the speaker ID
            default_config = predefined_configs.get(speaker_id, {}).copy()

            # Overwrite default configuration with parameters from the tag
            voice_config = {**default_config, **param_dict}

            segments.append(
                SpeakerSegment(speaker_id, param_dict, text.strip(), voice_config)
            )

        # Combine consecutive segments with the same speaker_id
        combined_segments = []
        for segment in segments:
            if (
                combined_segments
                and combined_segments[-1].speaker_id == segment.speaker_id
            ):
                combined_segments[-1].text += " " + segment.text
            else:
                combined_segments.append(segment)

        # Return tuples of question-answer pairs (default: speaker 1 = question, speaker 2 = answer)
        return [
            (combined_segments[i], combined_segments[i + 1])
            for i in range(0, len(combined_segments), 2)
        ]

    def clean_tss_markup(
        self,
        input_text: str,
        additional_tags: List[str] = ["Person1", "Person2"],
        supported_tags: List[str] = None,
        person_tag: str = "Person",
    ) -> str:
        """
        Remove unsupported TSS markup tags from the input text while preserving supported SSML tags.

        Args:
            input_text (str): The input text containing TSS markup tags.
            additional_tags (List[str]): Optional list of additional tags to preserve. Defaults to ["Person1", "Person2"].
            supported_tags (List[str]): Optional list of supported tags. If None, use COMMON_SSML_TAGS.
            person_tag (str): Base name for person tags (e.g., "Person"). Defaults to "Person".

        Returns:
            str: Cleaned text with unsupported TSS markup tags removed.
        """
        if supported_tags is None:
            supported_tags = self.COMMON_SSML_TAGS.copy()

        # Dynamically generate person tags pattern (e.g., Person1, Person2, ..., PersonN)
        person_tag_pattern = f"{person_tag}\\d+"

        # Combine supported tags and additional tags
        all_supported_tags = supported_tags + additional_tags

        # Create a pattern that matches any tag not in the supported list
        pattern = (
            r"</?(?!(?:"
            + "|".join(all_supported_tags + [person_tag_pattern])
            + r")\b)[^>]+>"
        )

        # Remove unsupported tags
        cleaned_text = re.sub(pattern, "", input_text)

        # Remove any leftover empty lines
        cleaned_text = re.sub(r"\n\s*\n", "\n", cleaned_text)

        # Ensure closing tags for additional tags are preserved
        for tag in additional_tags:
            cleaned_text = re.sub(
                f"<{tag}>(.*?)(?=<|$)",
                f"<{tag}>\\1</{tag}>",
                cleaned_text,
                flags=re.DOTALL | re.IGNORECASE,
            )

        # Ensure closing tags for person tags are preserved
        cleaned_text = re.sub(
            f"<({person_tag}\\d+)>(.*?)(?=<|$)",
            "<\\1>\\2</\\1>",
            cleaned_text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        return cleaned_text.strip()
