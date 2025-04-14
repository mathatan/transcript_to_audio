import os

# import tempfile # No longer needed
from dotenv import load_dotenv

from source.text_to_speech import TextToSpeech
from source.schemas import TTSConfig, SpeakerConfig


class ExecutableTestTTS:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()

        # Supported providers
        self.providers = [
            "elevenlabs",
            # "azureopenai",
            # "openai",
            # "edge",
            # "geminimulti",
            # "gemini",
        ]

        # Sample transcript-style text for testing
        # self.sample_text = (
        #     "<person1>Hello, how are you today?</person1> "
        #     "<person2>I'm doing well, thank you! How about you?</person2> "
        #     "<person1>I'm great, thanks for asking.</person1>"
        # )
        self.sample_text = (
            "<person1>Hello, how are you today?  </person1>"
            "<person2>I'm doing well, thank you! How about you?</person2>"
            "<person1>I'm great, thanks for asking.</person1>"
            "<person1>Do you have any plans for the weekend?</person1>"
            "<person2>Not sure yet, I might just stay in and read. What about you?</person2>"
            "<person1>I was thinking of hiking! The weather should be perfect.</person1>"
            "<person2>Oh, that's a lovely idea! I do love the outdoors.</person2>"
            "<person1>You should come along if you're free.</person1>"
            "<person2>Are you sure you can keep up with me on a trail?</person2>"
            "<person1>Oh, I'm sure! But I'll let you win if you promise to bring snacks.</person1>"
            "<person2>Deal! Now that's a hike I can definitely get behind.</person2>"
        )

        # self.sample_text_with_emoting = (
        #     '<person1 emote="Started enthusiasticly">Hello, how are you today?  </person1>'
        #     '<person2 emote="Replied a bit sarcastically">I\'m doing well, thank you! How about you?</person2>'
        #     '<person1 emote="Said with a smile in his voice">I\'m great, thanks for asking.</person1>'
        # )

        self.sample_text_with_emoting = (
            '<person1 emote="He spoke with excitement and a cheerful tone">Hello, how are you today?  </person1>'
            '<person2 emote="She replied warmly and invitingly">I\'m doing well, thank you! How about you?</person2>'
            '<person1 emote="He said happily and with energy">I\'m great, thanks for asking.</person1>'
            '<person1 emote="He asked curiously with a rising tone">Do you have any plans for the weekend?</person1>'
            '<person2 emote="She answered thoughtfully with a calm tone">Not sure yet, I might just stay in and read. What about you?</person2>'
            '<person1 emote="He suggested enthusiastically with excitement">I was thinking of hiking! The weather should be perfect.</person1>'
            '<person2 emote="She responded with intrigue and interest">Oh, that\'s a lovely idea! I do love the outdoors.</person2>'
            '<person1 emote="He said playfully with a teasing tone">You should come along if you\'re free.</person1>'
            '<person2 emote="She replied challengingly with a bold tone">Are you sure you can keep up with me on a trail?</person2>'
            "<person1 emote=\"He teased confidently with a playful tone\">Oh, I'm sure! But I'll let you win if you promise to bring snacks.</person1>"
            '<person2 emote="She agreed cheerfully with enthusiasm">Deal! Now that\'s a hike I can definitely get behind.</person2>'
        )

    def run_tests(self):
        # Define the output directory relative to the project root
        output_dir = "data/audio"
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving test outputs to: {os.path.abspath(output_dir)}")

        for provider in self.providers:
            try:
                print(f"Testing provider: {provider}")

                # Load provider-specific environment variables
                provider_specific_config_args = {}
                if provider == "elevenlabs":
                    provider_specific_config_args["api_key"] = os.getenv(
                        "ELEVENLABS_API_KEY"
                    )
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
                    print(f"Skipping test for {provider}: Missing API key")
                    continue

                # Define provider-specific speaker configurations
                speaker1_config = None
                speaker2_config = None

                if provider == "elevenlabs":
                    # Example voice IDs, replace with actual IDs if needed
                    speaker1_config = SpeakerConfig(
                        voice="Liam",
                        language="en-US",
                        # stability=0.15,
                        # similarity_boost=0.4,
                    )
                    speaker2_config = SpeakerConfig(
                        voice="Juniper",
                        language="en-US",
                        # stability=0.15,
                        # similarity_boost=0.4,
                    )
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

                # Initialize TextToSpeech with the current provider and config
                tts = TextToSpeech(provider=provider, tts_config=test_tts_config)

                # Output file path within the defined output directory
                output_file = os.path.join(output_dir, f"{provider}_output.mp3")

                # Convert text to speech
                tts.convert_to_speech(
                    (
                        self.sample_text
                        if provider != "elevenlabs"
                        else self.sample_text_with_emoting
                    ),
                    output_file,
                )

                # Verify the file was created
                if os.path.exists(output_file):
                    print(f"Success: MP3 file created for {provider} at {output_file}")
                else:
                    print(f"Error: MP3 file not created for {provider}")

            except Exception as e:
                print(f"Error testing provider {provider}: {e}")


if __name__ == "__main__":
    tester = ExecutableTestTTS()
    tester.run_tests()
