"""Same-language routing: source == target uses transcription.

The translation model omits same-language audio (no output deltas), which
breaks subtitles. When the configured source and destination languages
match, audio chunks are routed through the transcription path
(``gpt-realtime-transcribe``): transcription session updates, transcription
append/commit types, and transcription transcript events.
"""

import asyncio
import base64
import json

import pytest

from app.api.websocket import build_upstream_client
from app.core.config import Settings
from app.schemas.events import (
    AudioAppendEvent,
    AudioCommitEvent,
    TranscriptCompletedEvent,
    TranscriptDeltaEvent,
    is_same_language,
    map_upstream_event,
    normalize_language_code,
)
from app.services.openai_realtime import build_transcription_session_update
from app.services.relay import RelaySession, to_upstream_payload
from app.services.webrtc_session import (
    WebRTCRelaySession,
    webrtc_audio_to_upstream_payload,
)

AUDIO_B64 = "AQID"  # 3 decoded bytes, valid base64
PCM_SMALL = b"\x01\x02\x03\x04"


def make_settings(**overrides):
    base = {
        "OPENAI_API_KEY": "test-key",
        "SESSION_IDLE_TIMEOUT_SECONDS": 5.0,
        "MAX_AUDIO_BYTES_PER_SECOND": 96000,
        "MAX_WEBSOCKET_MESSAGE_BYTES": 65536,
    }
    base.update(overrides)
    return Settings(**base)


class FakeClient:
    """In-memory extension socket (same event loop)."""

    def __init__(self, incoming: list | None = None):
        self.incoming: asyncio.Queue = asyncio.Queue()
        for item in incoming or []:
            self.incoming.put_nowait(item)
        self.sent: list[str] = []
        self.closed: tuple | None = None

    async def send_text(self, message: str) -> None:
        self.sent.append(message)

    async def receive_text(self) -> str:
        item = await self.incoming.get()
        if item is None:
            raise RuntimeError("client disconnected")
        return item

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = (code, reason)

    def disconnect(self) -> None:
        self.incoming.put_nowait(None)


class FakeUpstream:
    """In-memory OpenAI socket (same event loop)."""

    def __init__(self, incoming: list | None = None):
        self.incoming: asyncio.Queue = asyncio.Queue()
        for item in incoming or []:
            self.incoming.put_nowait(item)
        self.sent: list[str] = []
        self.closed = False

    async def send_text(self, payload: str) -> None:
        self.sent.append(payload)

    async def recv_text(self) -> str:
        item = await self.incoming.get()
        if item is None:
            raise RuntimeError("upstream disconnected")
        return item

    async def close(self) -> None:
        self.closed = True

    def disconnect(self) -> None:
        self.incoming.put_nowait(None)


class FakeChannel:
    """In-memory stand-in for aiortc's RTCDataChannel (sync send)."""

    def __init__(self):
        self.sent: list = []
        self.readyState = "open"

    def send(self, message) -> None:
        self.sent.append(message)


def test_normalize_language_code():
    assert normalize_language_code("EN") == "en"
    assert normalize_language_code("en-US") == "en-us"
    assert normalize_language_code(" en ") == "en"
    assert normalize_language_code("") == ""


def test_is_same_language():
    assert is_same_language("en", "en") is True
    assert is_same_language("EN", "en") is True
    assert is_same_language("en-US", "en") is True
    assert is_same_language("es", "en") is False
    assert is_same_language("", "en") is False
    assert is_same_language("en", "") is False


def test_build_transcription_session_update_shape():
    payload = json.loads(
        build_transcription_session_update("gpt-realtime-transcribe", ["en"])
    )
    assert payload["type"] == "session.update"
    session = payload["session"]
    assert session["type"] == "transcription"
    assert session["audio"]["input"]["format"] == {
        "type": "audio/pcm",
        "rate": 24000,
    }
    assert session["audio"]["input"]["transcription"]["model"] == (
        "gpt-realtime-transcribe"
    )
    assert session["audio"]["input"]["transcription"]["languages"] == ["en"]
    assert session["audio"]["input"]["turn_detection"] is None


def test_transcription_deltas_map_downstream():
    delta = map_upstream_event(
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "delta": "Hello",
            "item_id": "item_1",
        }
    )
    assert isinstance(delta, TranscriptDeltaEvent)
    assert delta.delta == "Hello"
    assert delta.item_id == "item_1"
    completed = map_upstream_event(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "Hello.",
            "item_id": "item_1",
        }
    )
    assert isinstance(completed, TranscriptCompletedEvent)
    assert completed.transcript == "Hello."


def test_input_transcript_mapped_only_in_transcription_mode():
    event = {"type": "session.input_transcript.delta", "delta": "Hi"}
    assert map_upstream_event(event) is None
    mapped = map_upstream_event(event, transcription_mode=True)
    assert isinstance(mapped, TranscriptDeltaEvent)
    assert mapped.delta == "Hi"


def test_to_upstream_payload_modes():
    event = AudioAppendEvent(type="audio.append", audio=AUDIO_B64)
    assert json.loads(to_upstream_payload(event)) == {
        "type": "session.input_audio_buffer.append",
        "audio": AUDIO_B64,
    }
    assert json.loads(
        to_upstream_payload(event, transcription_mode=True)
    ) == {"type": "input_audio_buffer.append", "audio": AUDIO_B64}
    # Commits are swallowed in translation mode but forwarded in
    # transcription mode, which has a commit/turn lifecycle.
    assert to_upstream_payload(AudioCommitEvent(type="audio.commit")) is None
    assert json.loads(
        to_upstream_payload(
            AudioCommitEvent(type="audio.commit"), transcription_mode=True
        )
    ) == {"type": "input_audio_buffer.commit"}


def test_webrtc_audio_payload_modes():
    assert json.loads(webrtc_audio_to_upstream_payload(PCM_SMALL)) == {
        "type": "session.input_audio_buffer.append",
        "audio": base64.b64encode(PCM_SMALL).decode("ascii"),
    }
    assert json.loads(
        webrtc_audio_to_upstream_payload(PCM_SMALL, transcription_mode=True)
    ) == {
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(PCM_SMALL).decode("ascii"),
    }


def test_transcribe_flag_defaults_to_translate():
    settings = make_settings()
    assert settings.transcribe_mode is False
    assert settings.starts_in_transcription() is False
    assert settings.active_model() == "gpt-realtime-translate"
    assert settings.openai_transcription_model == "gpt-realtime-transcribe"
    assert (
        settings.openai_transcription_url
        == "wss://api.openai.com/v1/realtime?intent=transcription"
    )


def test_upstream_factory_selects_transcription_for_flag():
    client = build_upstream_client(make_settings(TRANSCRIBE_MODE=True))
    assert client.is_transcription is True
    assert client.model == "gpt-realtime-transcribe"

    client = build_upstream_client(make_settings())
    assert client.is_transcription is False
    assert client.model == "gpt-realtime-translate"


def test_upstream_factory_selects_transcription_for_same_language():
    client = build_upstream_client(
        make_settings(
            OPENAI_TRANSLATION_SOURCE_LANGUAGE="en",
            OPENAI_TRANSLATION_TARGET_LANGUAGE="en",
        )
    )
    assert client.is_transcription is True
    assert client.model == "gpt-realtime-transcribe"

    client = build_upstream_client(make_settings())
    assert client.is_transcription is False
    assert client.model == "gpt-realtime-translate"


@pytest.mark.asyncio
async def test_relay_same_language_config_routes_to_transcription():
    client = FakeClient(
        [
            json.dumps(
                {
                    "type": "language-config",
                    "source_lang": "en",
                    "target_lang": "en",
                }
            ),
            json.dumps({"type": "audio.append", "audio": AUDIO_B64}),
        ]
    )
    upstream = FakeUpstream(
        [
            json.dumps(
                {
                    "type": (
                        "conversation.item.input_audio_transcription.delta"
                    ),
                    "delta": "Hello",
                }
            ),
        ]
    )
    relay = RelaySession(client, upstream, make_settings())
    task = asyncio.create_task(relay.run())
    await asyncio.sleep(0.4)
    client.disconnect()
    upstream.disconnect()
    await asyncio.wait_for(task, timeout=5)

    assert relay._transcription_mode is True
    update = json.loads(upstream.sent[0])
    assert update["type"] == "session.update"
    assert update["session"]["type"] == "transcription"
    assert update["session"]["audio"]["input"]["transcription"][
        "model"
    ] == "gpt-realtime-transcribe"
    assert json.loads(upstream.sent[1]) == {
        "type": "input_audio_buffer.append",
        "audio": AUDIO_B64,
    }
    deltas = [
        json.loads(m) for m in client.sent if "transcript.delta" in m
    ]
    assert deltas and deltas[0]["delta"] == "Hello"


@pytest.mark.asyncio
async def test_relay_transcribe_flag_forwards_commits_and_completed():
    client = FakeClient(
        [
            json.dumps({"type": "audio.append", "audio": AUDIO_B64}),
            json.dumps({"type": "audio.commit"}),
        ]
    )
    upstream = FakeUpstream(
        [
            json.dumps(
                {
                    "type": (
                        "conversation.item.input_audio_transcription.completed"
                    ),
                    "transcript": "Hello.",
                    "item_id": "item_1",
                }
            ),
        ]
    )
    relay = RelaySession(client, upstream, make_settings(TRANSCRIBE_MODE=True))
    task = asyncio.create_task(relay.run())
    await asyncio.sleep(0.4)
    client.disconnect()
    upstream.disconnect()
    await asyncio.wait_for(task, timeout=5)

    assert relay._transcription_mode is True
    assert json.loads(upstream.sent[0]) == {
        "type": "input_audio_buffer.append",
        "audio": AUDIO_B64,
    }
    assert json.loads(upstream.sent[1]) == {"type": "input_audio_buffer.commit"}
    completed = [
        json.loads(m) for m in client.sent if "transcript.completed" in m
    ]
    assert completed and completed[0]["transcript"] == "Hello."
    assert completed[0]["item_id"] == "item_1"


async def test_relay_mode_flips_back_to_translation():
    client = FakeClient(
        [
            json.dumps(
                {
                    "type": "language-config",
                    "source_lang": "en",
                    "target_lang": "en",
                }
            ),
            json.dumps(
                {
                    "type": "language-config",
                    "source_lang": "hi",
                    "target_lang": "en",
                }
            ),
        ]
    )
    upstream = FakeUpstream()
    relay = RelaySession(client, upstream, make_settings())
    task = asyncio.create_task(relay.run())
    await asyncio.sleep(0.3)
    client.disconnect()
    upstream.disconnect()
    await asyncio.wait_for(task, timeout=5)

    assert relay._transcription_mode is False
    assert json.loads(upstream.sent[0])["session"]["type"] == "transcription"
    update = json.loads(upstream.sent[1])
    assert update["session"]["audio"]["output"] == {"language": "en"}


@pytest.mark.asyncio
async def test_webrtc_same_language_config_routes_to_transcription():
    channel, upstream = FakeChannel(), FakeUpstream()
    relay = WebRTCRelaySession(channel, upstream, make_settings())
    task = asyncio.create_task(relay.run())
    await asyncio.sleep(0.05)
    relay.on_channel_message(
        json.dumps(
            {"type": "language-config", "source_lang": "en", "target_lang": "en"}
        )
    )
    await asyncio.sleep(0.05)
    relay.on_channel_message(PCM_SMALL)
    await asyncio.sleep(0.1)
    relay.on_channel_close()
    await asyncio.wait_for(task, timeout=5)

    assert relay._transcription_mode is True
    update = json.loads(upstream.sent[0])
    assert update["session"]["type"] == "transcription"
    assert json.loads(upstream.sent[1]) == {
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(PCM_SMALL).decode("ascii"),
    }
