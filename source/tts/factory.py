"""Factory for creating TTS provider instances."""

from .providers.edge import EdgeTTS
from .providers.elevenlabs import ElevenLabsTTS
from .providers.gemini import GeminiTTS
from .providers.geminimulti import GeminiMultiTTS
from .providers.openai import OpenAITTS
from .providers.azureopenai import AzureOpenAITTS
from .base import TTSProvider
from ..schemas import TTSConfig


class TTSProviderFactory:
    """
    Factory class for creating TTS provider instances.
    """

    _providers = {
        "edge": EdgeTTS,
        "elevenlabs": ElevenLabsTTS,
        "gemini": GeminiTTS,
        "geminimulti": GeminiMultiTTS,
        "openai": OpenAITTS,
        "azureopenai": AzureOpenAITTS,
    }

    @staticmethod
    def create(provider_name: str, config: TTSConfig) -> TTSProvider:
        """
        Create a TTS provider instance.

        Args:
            provider_name (str): Name of the provider (e.g., 'edge', 'elevenlabs').
            config (TTSConfig): Configuration object for the provider.

        Returns:
            TTSProvider: An instance of the requested TTS provider.
        """
        if provider_name not in TTSProviderFactory._providers:
            raise ValueError(f"Unknown provider: {provider_name}")
        return TTSProviderFactory._providers[provider_name](config)
