"""
Pydantic models and data structures for TTS configuration and processing.
"""

from typing import Optional, Dict, Union
from pydantic import BaseModel, Field


class SpeakerConfig(BaseModel):
    """
    Configuration for the speaker's properties.

    Attributes:
        voice (str): Specifies the speaker's voice identifier. Defaults to "default_voice_1".
        language (str): Language code for the speech, e.g., "en-US". Defaults to "en-US".
        pitch (str): Pitch of the speaker's voice. Defaults to "default".
        speaking_rate (Union[str, float]): Rate at which the speaker talks.
            Can be a string or a float value. Defaults to 1.0.
        stability (Optional[float]): Stability of the speech generation (ElevenLabs-specific). Defaults to 0.75.
        similarity_boost (Optional[float]): Similarity boost in speech tone (ElevenLabs-specific). Defaults to 0.85.
        style (Optional[int]): Style of speech delivery (ElevenLabs-specific). Defaults to 0.
        ssml_gender (Optional[str]): Gender identifier for SSML-driven TTS (Gemini-specific). Defaults to "NEUTRAL".
    """

    voice: str = Field(
        default="default_voice_1",
        description="Specifies the speaker's voice identifier.",
    )
    language: str = Field(
        default="en-US", description="Language code for the speech, e.g., 'en-US'."
    )
    pitch: str = Field(default="default", description="Pitch of the speaker's voice.")
    speaking_rate: Union[str, float] = Field(
        default=1.0,
        description="Rate at which the speaker talks. Can be string or float.",
    )
    stability: Optional[float] = Field(
        default=0.75,
        description="Stability of the speech generation (ElevenLabs-specific).",
    )
    similarity_boost: Optional[float] = Field(
        default=0.85,
        description="Similarity boost in speech tone (ElevenLabs-specific).",
    )
    style: Optional[float] = Field(
        default=0, description="Style of speech delivery (ElevenLabs-specific)."
    )
    use_speaker_boost: Optional[bool] = Field(
        default=True, description="Whether to use speaker boost (ElevenLabs-specific)."
    )

    ssml_gender: Optional[str] = Field(
        default="NEUTRAL",
        description="Gender identifier for SSML-driven TTS (Gemini-specific).",
    )

    class Config:
        extra = "allow"  # Allow extra fields for specific providers.


class TTSConfig(BaseModel):
    """
    Configuration for the Text-to-Speech system and providers.

    Attributes:
        speaker_configs (Dict[int, SpeakerConfig]): Configurations for each speaker ID. Defaults to an empty dict.
        audio_format (str): The desired output audio format (e.g., 'mp3', 'wav'). Defaults to 'mp3'.
        output_directories (Dict[str, str]): Dictionary mapping output types (e.g., 'audio', 'transcripts') to directory paths. Defaults to an empty dict.
        temp_audio_dir (str): Path to the directory for temporary audio files. Defaults to 'data/audio/tmp/'.
        api_base (Optional[str]): The base URL for the API endpoint (Azure OpenAI-specific). Defaults to None.
        api_key (Optional[str]): API key for the TTS provider (Used by Azure, ElevenLabs, Gemini, OpenAI). Defaults to None. Can often be set via environment variables.
        api_version (str): API version for the TTS provider (Azure OpenAI-specific). Defaults to '2025-01-01-preview'.
        deployment (Optional[str]): Deployment name or ID for the TTS provider (Azure OpenAI-specific). Defaults to None.
        model (Optional[str]): The specific TTS model to use. Defaults vary by provider (Edge, ElevenLabs, Gemini, OpenAI). Defaults to None.
        response_format (str): The audio format for the API response (OpenAI-specific). Defaults to 'mp3'.
        streaming (bool): Whether to use streaming audio generation (OpenAI-specific). Defaults to False.
        speed (float): The speaking speed multiplier (OpenAI-specific). Defaults to 1.0.
        language (str): The language for the TTS request (OpenAI-specific, distinct from SpeakerConfig language). Defaults to 'en'.
    """

    speaker_configs: Dict[int, SpeakerConfig] = Field(
        default_factory=dict,
        description="Configurations for each speaker ID.",
    )
    audio_format: str = Field(
        default="mp3",
        description="The desired output audio format (e.g., 'mp3', 'wav').",
    )
    output_directories: Dict[str, str] = Field(
        default_factory=lambda: {
            "audio": "data/audio/",
            "transcripts": "data/transcripts/",
        },
        description="Dictionary mapping output types (e.g., 'audio', 'transcripts') to directory paths.",
    )
    temp_audio_dir: str = Field(
        default="data/audio/tmp/",
        description="Path to the directory for temporary audio files.",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="The base URL for the API endpoint (Azure OpenAI-specific).",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for the TTS provider (Used by Azure, ElevenLabs, Gemini, OpenAI). Can often be set via environment variables.",
    )
    api_version: str = Field(
        default="2025-01-01-preview",
        description="API version for the TTS provider (Azure OpenAI-specific).",
    )
    deployment: Optional[str] = Field(
        default=None,
        description="Deployment name or ID for the TTS provider (Azure OpenAI-specific).",
    )
    model: Optional[str] = Field(
        default=None,
        description="The specific TTS model to use. Defaults vary by provider (Edge, ElevenLabs, Gemini, OpenAI).",
    )
    response_format: str = Field(
        default="mp3",
        description="The audio format for the API response (OpenAI-specific).",
    )
    streaming: bool = Field(
        default=False,
        description="Whether to use streaming audio generation (OpenAI-specific).",
    )
    speed: float = Field(
        default=1.0, description="The speaking speed multiplier (OpenAI-specific)."
    )
    language: str = Field(
        default="en",
        description="The language for the TTS request (OpenAI-specific, distinct from SpeakerConfig language).",
    )
    use_emote: Optional[bool] = Field(
        default=True,
        description="Whether to use emotes with audio generation (ElevenLabs-specific).",
    )
    emote_pause: Optional[str] = Field(
        default="1.5",
        description="Length to use for splitting emote description from text (ElevenLabs-specific).",
    )
    emote_merge_pause: Optional[int] = Field(
        default=500,
        description="Pause between speaker turns when using emote (ElevenLabs-specific).",
    )

    class Config:
        extra = "allow"  # Allow extra fields for specific providers or future use.


class SpeakerSegment:
    """
    Represents a segment of text associated with a specific speaker.

    Attributes:
        speaker_id (int): The ID of the speaker (e.g., 1 for <person1>).
        parameters (dict): Additional parameters extracted from the tag (e.g., {"param": "value"}).
        text (str): The text content of the segment.
        voice_config (SpeakerConfig): The voice configuration for the speaker.
        audio (Optional[bytes]): Binary audio data if available.
        audio_file (Optional[str]): Path to an audio file if available.
    """

    def __init__(
        self,
        speaker_id: int,
        parameters: Optional[Dict[str, str]] = None,
        text: str = "",
        voice_config: Optional["SpeakerConfig"] = None,
        audio: Optional[bytes] = None,
        audio_file: Optional[str] = None,
    ):
        self.speaker_id = speaker_id
        self.parameters = parameters or {}
        self.text = text
        self.voice_config = voice_config
        self.audio = audio
        self.audio_file = audio_file

    def __repr__(self):
        return (
            f"SpeakerSegment(speaker_id={self.speaker_id}, "
            f"parameters={self.parameters}, text={self.text}, "
            f"voice_config={self.voice_config}, audio={self.audio}, audio_file={self.audio_file})"
        )
