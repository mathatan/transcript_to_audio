"""ElevenLabs TTS provider implementation."""

import logging
import time
from elevenlabs import (
    SpeechHistoryItemResponse,
    VoiceSettings,
    client as elevenlabs_client,
)
from elevenlabs.client import is_voice_id
from ..base import TTSProvider
from typing import List
from ...schemas import SpeakerSegment, TTSConfig

logger = logging.getLogger("transcript_to_audio_logger")


class ElevenLabsTTS(TTSProvider):
    def __init__(self, config: TTSConfig):
        """
        Initialize ElevenLabs TTS provider.

        Args:
            config (TTSConfig): Configuration object.
        """
        super().__init__(config)
        if not config.api_key:
            raise ValueError(
                "ElevenLabs API key must be provided in the configuration."
            )
        self.client = elevenlabs_client.ElevenLabs(api_key=config.api_key)
        # Use the model from config or default to "eleven_multilingual_v2" if None
        self.model = config.model or "eleven_multilingual_v2"

    def generate_audio(self, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        """
        Generate audio using ElevenLabs API for all SpeakerSegments in a single call.
        """
        # audio_chunks = []
        # Initialize variables for history and request tracking
        previous_request_ids: List[tuple[str, str]] = []
        # previous_history_id: str | None = None

        for i, segment in enumerate(segments):
            logger.info(
                f"Generating audio for Speaker {segment.speaker_id}: {segment.text}"
            )

            # Determine previous_text and next_text
            # previous_text: str | None = (
            #     segments[i - 1].text
            #     if i > 0 and segments[i - 1].speaker_id == segment.speaker_id
            #     else None
            # )
            # next_text: str | None = (
            #     segments[i + 1].text
            #     if i < len(segments) - 1
            #     and segments[i + 1].speaker_id == segment.speaker_id
            #     else None
            # )
            previous_text: str | None = (
                (
                    segments[i - 1].text
                    if segments[i - 1].speaker_id == segment.speaker_id
                    else segments[i - 1].text
                    + (
                        '" said. '
                        if segments[i - 1].voice_config.use_emote
                        and segments[i - 1].parameters.get("emote") is None
                        else ('" ' + segments[i - 1].parameters.get("emote", "said"))
                    )
                )
                if i > 0
                else None
            )
            next_text: str | None = (
                (
                    segments[i + 1].text
                    if segments[i + 1].speaker_id == segment.speaker_id
                    else (
                        '" said. ' + segments[i + 1].text
                        if segment.voice_config.use_emote
                        and segment.parameters.get("emote") is None
                        else (
                            segments[i + 1].text
                            + '" '
                            + segment.parameters.get("emote", "said")
                        )
                    )
                )
                if i < (len(segments) - 1)
                else None
            )

            # Prepare voice settings
            voice_settings: VoiceSettings = VoiceSettings(
                stability=segment.voice_config.stability,
                similarity_boost=segment.voice_config.similarity_boost,
                style=segment.voice_config.style,
                use_speaker_boost=segment.voice_config.use_speaker_boost,
            )

            voice = segment.voice_config.voice

            if isinstance(voice, str) and is_voice_id(voice):
                voice_id = voice
            elif isinstance(voice, str):
                voices_response = self.client.voices.get_all(show_legacy=True)
                maybe_voice_id = next(
                    (v.voice_id for v in voices_response.voices if v.name == voice),
                    None,
                )
                if maybe_voice_id is None:
                    raise ValueError(body=f"Voice model {voice} not found.")
                voice_id = maybe_voice_id

            prev_requests = previous_request_ids[-3:]
            # prev_requests.reverse()

            logger.info(
                f"previous segments: {"\n".join([str(req) for req in prev_requests])}"
            )

            text = segment.text
            if (
                segment.voice_config.use_emote
                and segment.voice_config.emote_pause is not None
                and segment.parameters.get("emote", None) is not None
            ):
                text += (
                    f'<break time="{segment.voice_config.emote_pause}s" />" '
                    + segment.parameters.get("emote")
                )

            audio_chunks = None
            max_repeats = 3
            rep = 0
            gen_e = None
            while rep < max_repeats and audio_chunks is None:
                try:
                    # Generate audiochunks_
                    audio_chunks = self.client.text_to_speech.convert(
                        enable_logging=True,
                        text=text,
                        voice_id=voice_id,
                        model_id=self.model,
                        previous_text=previous_text,
                        next_text=next_text,
                        previous_request_ids=[
                            item[0] for item in prev_requests
                        ],  # Use up to 3 previous IDs
                        voice_settings=voice_settings,
                    )
                except Exception as e:
                    gen_e = e
                    time.sleep(2)
                    rep += 1

            if audio_chunks is None:
                raise ValueError(
                    f"Unable to generate audio_chunks. \nError: {gen_e}"
                ) from gen_e

            # Append audio chunks
            segment.audio = b"".join(chunk for chunk in audio_chunks if chunk)
            prev_len = len(previous_request_ids)
            max_repeats = 3
            rep = 0
            while rep < max_repeats and prev_len == len(previous_request_ids):
                logger.info("Try to find history item")

                # Fetch updated history after generation
                history_response = self.client.history.get_all(
                    page_size=4,
                    # start_after_history_item_id=previous_history_id,
                )
                history_response_items: List[SpeechHistoryItemResponse] = sorted(
                    history_response.history,
                    key=lambda item: item.date_unix,
                    reverse=True,
                )
                # logger.info(
                #     f"history response {[tuple([item.request_id, item.text]) for item in history_response.history]}"
                # )
                first_match = next(
                    (
                        tuple([item.request_id, item.text])
                        for item in history_response_items
                        if item.request_id and item.text == text
                    ),
                    None,  # Default value if no match is found
                )
                if first_match is not None:
                    logger.info("Found item")
                    previous_request_ids = previous_request_ids + [first_match]
                else:
                    time.sleep(2)
                    rep += 1

            # if history_response.history:
            #     previous_history_id = history_response.history[-1].history_item_id
        return segments

    def get_supported_tags(self) -> List[str]:
        """Get supported SSML tags."""
        return ["lang", "p", "phoneme", "s", "sub"]
