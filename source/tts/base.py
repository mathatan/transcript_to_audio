"""Abstract base class for Text-to-Speech providers."""

from abc import ABC, abstractmethod
from typing import List, ClassVar, Dict
import re

from ..schemas import SpeakerConfig, SpeakerSegment, TTSConfig


class TTSProvider(ABC):
    """Abstract base class that defines the interface for TTS providers."""

    # Common SSML tags supported by most providers
    COMMON_SSML_TAGS: ClassVar[List[str]] = ["lang", "p", "phoneme", "s", "sub"]

    def __init__(self, config: TTSConfig):
        """
        Initialize the TTS provider with configuration.

        Args:
            config (TTSConfig): Configuration object.
        """
        self.config = config

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
    ) -> List[SpeakerSegment]:
        """
        Parse input text into a list of SpeakerSegment instances.

        Args:
            input_text (str): The input text containing <personN> tags.
            supported_tags (List[str]): Supported SSML tags.

        Returns:
            List[SpeakerSegment]: A list of SpeakerSegment instances.
        """
        # Retrieve predefined configurations from the TTSConfig object
        predefined_configs: Dict[int, SpeakerConfig] = self.config.speaker_configs

        # Clean the input text
        input_text = self.clean_tss_markup(input_text, supported_tags=supported_tags)

        # Regular expression pattern to match <personN> tags
        pattern = r"<person(\d+)(.*?)>(.*?)</person\1>"
        matches = re.findall(pattern, input_text, re.DOTALL | re.IGNORECASE)

        segments: List[SpeakerSegment] = []
        for speaker_id, params, text in matches:
            speaker_id = int(speaker_id)

            # Parse parameters into a dictionary
            param_dict = dict(re.findall(r'(\w+)="(.*?)"', params))

            # Retrieve default configuration for the speaker ID
            default_config = predefined_configs.get(speaker_id, {}).copy()

            # Identify fields in param_dict that match the SpeakerConfig schema
            schema_fields = SpeakerConfig.model_fields.keys()
            matched_fields = {
                key: value for key, value in param_dict.items() if key in schema_fields
            }

            # Only create a new instance if there are matching fields; otherwise, use default_config directly
            if matched_fields:
                updated_config = {**default_config, **matched_fields}
                speaker_config = SpeakerConfig.model_validate(updated_config)
            else:
                speaker_config = SpeakerConfig.model_validate(default_config)

            segments.append(
                SpeakerSegment(speaker_id, param_dict, text.strip(), speaker_config)
            )

        # Return the list of combined SpeakerSegments
        return segments

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
        # pattern = (
        #     r"</?(?!(?:"
        #     + "|".join(all_supported_tags + [person_tag_pattern])
        #     + r")\b)[^>]+>"
        # )
        # Define a regex pattern for unsupported tags to be removed
        # Define a regex pattern to look for unsupported tags (excluding supported tags with optional properties)
        pattern = (
            r"</?(?!("
            + "|".join(all_supported_tags + [person_tag_pattern])
            + r")\b)[^>]+>"
        )

        # Remove unsupported tags using the defined `pattern`
        cleaned_text = re.sub(
            pattern,
            "",
            input_text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # Remove any leftover empty lines
        cleaned_text = re.sub(
            r"\n\s*\n",
            "\n",
            cleaned_text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # Ensure closing tags for additional tags are preserved, while retaining properties
        for tag in additional_tags:
            cleaned_text = re.sub(
                f"<{tag}( [^>]+)?>(.*?)(?=<|$)",  # Match opening tags with optional properties
                f"<{tag}\\1>\\2</{tag}>",  # Reinsert the opening tag with its properties if present
                cleaned_text,
                flags=re.DOTALL | re.IGNORECASE,
            )

        # Ensure closing tags for person-specific tags are preserved, while retaining properties
        cleaned_text = re.sub(
            f"<({person_tag}\\d+)( [^>]+)?>(.*?)(?=<|$)",  # Match opening tags with optional properties
            "<\\1\\2>\\3</\\1>",  # Reinsert the opening tag with its properties if present
            cleaned_text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        return cleaned_text.strip()
