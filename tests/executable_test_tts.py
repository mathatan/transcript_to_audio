import os

# import tempfile # No longer needed
from dotenv import load_dotenv

from transcript_to_audio.text_to_speech import TextToSpeech
from transcript_to_audio.schemas import TTSConfig, SpeakerConfig

import logging

logger = logging.getLogger("transcript_to_audio_logger")
logger.setLevel(logging.DEBUG)

# Create a StreamHandler to output logs to the shell
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)  # Handler level

# Optional: Add a formatter for better readability
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(stream_handler)


class ExecutableTestTTS:
    def __init__(
        self, transcript_filename="tests/transcripts/sample_transcript_en.txt"
    ):
        # Load environment variables from .env file
        load_dotenv()

        # transcript_filename = "tests/transcripts/article_transcript_fi.txt"

        # Load transcript from file
        try:
            with open(transcript_filename, "r", encoding="utf-8") as f:
                self.transcript_text = f.read()
        except FileNotFoundError:
            logger.debug(f"Error: Transcript file not found at {transcript_filename}")
            self.transcript_text = ""  # Set to empty string or handle error as needed

        # Supported providers
        self.providers = [
            "elevenlabs",
            # "azureopenai",
            # "openai",
            # "edge",
            # "geminimulti",
            # "gemini",
        ]

    def run_tests(self):
        # Define the output directory relative to the project root
        output_dir = "data/audio"
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        logger.debug(f"Saving test outputs to: {os.path.abspath(output_dir)}")

        for provider in self.providers:
            try:
                logger.debug(f"Testing provider: {provider}")

                # Load provider-specific environment variables
                provider_specific_config_args = {}
                if provider == "elevenlabs":
                    provider_specific_config_args["api_key"] = os.getenv(
                        "ELEVENLABS_API_KEY"
                    )
                    # provider_specific_config_args["model"] = "eleven_flash_v2_5"
                    provider_specific_config_args["use_emote"] = True
                    # provider_specific_config_args["emote_merge_pause"] = (
                    #     600  # works better for FI
                    # )
                elif provider == "azureopenai":
                    provider_specific_config_args["api_key"] = os.getenv(
                        "AZURE_OPENAI_API_KEY"
                    )
                    provider_specific_config_args["api_base"] = os.getenv(
                        "AZURE_OPENAI_ENDPOINT"
                    )
                    provider_specific_config_args["deployment"] = os.getenv(
                        "AZURE_OPENAI_DEPLOYMENT_NAME"
                    )
                elif provider == "openai":
                    provider_specific_config_args["api_key"] = os.getenv(
                        "OPENAI_API_KEY"
                    )
                elif provider == "gemini" or provider == "geminimulti":
                    provider_specific_config_args["api_key"] = os.getenv(
                        "GOOGLE_API_KEY"
                    )
                # Edge TTS does not require specific env vars in config

                # Check if api_key is available
                if (
                    "api_key" in provider_specific_config_args
                    and not provider_specific_config_args["api_key"]
                ):
                    logger.debug(f"Skipping test for {provider}: Missing API key")
                    continue

                # Define provider-specific speaker configurations
                speaker1_config = None
                speaker2_config = None

                if provider == "elevenlabs":
                    # Example voice IDs, replace with actual IDs if needed
                    # English
                    speaker1_config = SpeakerConfig(
                        voice="Liam",
                        language="en-US",
                        stability=0.25,
                        similarity_boost=0.5,
                    )
                    speaker2_config = SpeakerConfig(
                        voice="Juniper",
                        language="en-US",
                        stability=0.25,
                        similarity_boost=0.5,
                    )

                    # Finnish
                    # speaker1_config = SpeakerConfig(
                    #     voice="Chris",
                    #     language="fi-FI",
                    #     stability=0.5,
                    #     similarity_boost=0.5,
                    # )
                    # speaker2_config = SpeakerConfig(
                    #     voice="tDgDlJWDwRpTqPofl3HU",
                    #     language="fi-FI",
                    #     stability=0.2,
                    #     similarity_boost=0.5,
                    # )
                elif provider == "azureopenai":
                    speaker1_config = SpeakerConfig(voice="alloy", language="en-US")
                    speaker2_config = SpeakerConfig(voice="nova", language="en-US")
                elif provider == "openai":
                    speaker1_config = SpeakerConfig(voice="alloy", language="en-US")
                    speaker2_config = SpeakerConfig(voice="nova", language="en-US")
                elif provider == "edge":
                    speaker1_config = SpeakerConfig(
                        voice="en-US-JennyNeural", language="en-US"
                    )
                    speaker2_config = SpeakerConfig(
                        voice="en-US-GuyNeural", language="en-US"
                    )
                elif provider == "geminimulti":
                    # Gemini Multi uses speaker roles defined in the markup
                    speaker1_config = SpeakerConfig(voice="Person1", language="en-US")
                    speaker2_config = SpeakerConfig(voice="Person2", language="en-US")
                elif provider == "gemini":
                    speaker1_config = SpeakerConfig(
                        voice="en-US-Journey-F", language="en-US"
                    )
                    speaker2_config = SpeakerConfig(
                        voice="en-US-Journey-M", language="en-US"
                    )
                else:
                    # Default fallback (might need adjustment)
                    speaker1_config = SpeakerConfig(
                        voice="default_voice_1", language="en-US"
                    )
                    speaker2_config = SpeakerConfig(
                        voice="default_voice_2", language="en-US"
                    )

                # Create TTS Configuration with provider-specific speakers and env vars
                test_tts_config = TTSConfig(
                    speaker_configs={1: speaker1_config, 2: speaker2_config},
                    **provider_specific_config_args,
                )

                logger.debug(f"{provider=} {test_tts_config=}")

                # Initialize TextToSpeech with the current provider and config
                tts = TextToSpeech(provider=provider, tts_config=test_tts_config)

                # Output file path within the defined output directory
                output_file = os.path.join(output_dir, f"{provider}_output.mp3")

                # Convert text to speech
                tts.convert_to_speech(self.transcript_text, output_file, True)

                # Verify the file was created
                if os.path.exists(output_file):
                    logger.debug(
                        f"Success: MP3 file created for {provider} at {output_file}"
                    )
                else:
                    logger.debug(f"Error: MP3 file not created for {provider}")

            except Exception as e:
                logger.debug(f"Error testing provider {provider}: {e}")


if __name__ == "__main__":
    tester = ExecutableTestTTS()
    tester.run_tests()
