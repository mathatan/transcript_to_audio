import os
import tempfile
from dotenv import load_dotenv
from source.text_to_speech import TextToSpeech


class ExecutableTestTTS:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()

        # Supported providers
        self.providers = ["elevenlabs", "azureopenai", "openai", "edge", "geminimulti"]

        # Sample transcript-style text for testing
        self.sample_text = (
            "<person1>Hello, how are you today?</person1> "
            "<person2>I'm doing well, thank you! How about you?</person2> "
            "<person1>I'm great, thanks for asking.</person1>"
        )

    def run_tests(self):
        # Create a temporary directory for output files
        with tempfile.TemporaryDirectory() as temp_dir:
            for provider in self.providers:
                try:
                    print(f"Testing provider: {provider}")

                    # Initialize TextToSpeech with the current provider
                    tts = TextToSpeech(provider=provider)

                    # Output file path
                    output_file = os.path.join(temp_dir, f"{provider}_output.mp3")

                    # Convert text to speech
                    tts.convert_to_speech(self.sample_text, output_file)

                    # Verify the file was created
                    if os.path.exists(output_file):
                        print(
                            f"Success: MP3 file created for {provider} at {output_file}"
                        )
                    else:
                        print(f"Error: MP3 file not created for {provider}")

                except Exception as e:
                    print(f"Error testing provider {provider}: {e}")


if __name__ == "__main__":
    tester = ExecutableTestTTS()
    tester.run_tests()
