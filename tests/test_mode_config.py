"""Extension session-mode config: translation vs transcription.

The extension sends ``{"type": "language-config", "mode": ...}`` with no
source language (the upstream model auto-detects it). ``target_lang``
is only sent in translation mode. Both transports share the behavior.
"""

import asyncio
import base64
import json

import pytest

from app.core.config import Settings
from app.schemas.events import parse_client_event
from app.services.relay import RelaySession
from app.services.webrtc_session import WebRTCRelaySession

AUDIO_B64 = "AQID"  # 3 decoded bytes, valid base64
PCM_SMALL = b"\x01\x02\x03\x04"


def make_settings(**overrides):
    base = {
        "OPENAI_API_KEY": "test-key",
        "SESSION_IDLE_TIMEOUT_SECONDS": 5.0,
        "MAX_AUDIO_BYTES_PER_SECOND": 96000,
        "MAX_WEBSOCKET_MESSAGE_BYTES": 65536,
        "TRANSCRIBE_MODE": False,
        "OPENAI_TRANSLATION_SOURCE_LANGUAGE": "",
        "SUBTITLE_PROVIDER": "none",
        "STT_PROVIDER": "openai",
    }
    base.update(overrides)
    return Settings(**base)


class FakeClient:
    def __init__(self, incoming=None):
        self.incoming: asyncio.Queue = asyncio.Queue()
        for item in incoming or []:
            self.incoming.put_nowait(item)
        self.sent: list[str] = []
        self.closed = None

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
    def __init__(self, incoming=None):
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
    def __init__(self):
        self.sent: list = []
        self.readyState = "open"

    def send(self, message) -> None:
        self.sent.append(message)


async def run_relay(client, upstream, settings, subtitle=None):
    relay = RelaySession(client, upstream, settings, subtitle=subtitle)
    task = asyncio.create_task(relay.run())
    await asyncio.sleep(0.3)
    client.disconnect()
    upstream.disconnect()
    await asyncio.wait_for(task, timeout=5)
    return relay


# --- schema ---


def test_transcription_update_omits_languages_for_auto_detect():
    # Regression: an empty "languages": [] list makes the upstream
    # reject the session.update with an "empty_array" error, which
    # ended the WebRTC session when switching to translation mode.
    from app.services.openai_realtime import build_transcription_session_update

    for languages in ([], None):
        payload = json.loads(
            build_transcription_session_update("gpt-live-transcribe", languages)
        )
        transcription = payload["session"]["audio"]["input"]["transcription"]
        assert transcription["model"] == "gpt-live-transcribe"
        assert "languages" not in transcription
    hinted = json.loads(
        build_transcription_session_update("gpt-live-transcribe", ["en"])
    )
    assert hinted["session"]["audio"]["input"]["transcription"][
        "languages"
    ] == ["en"]


def test_parse_transcription_mode_needs_no_languages():
    event = parse_client_event(
        {"type": "language-config", "mode": "transcription"}
    )
    assert event.effective_mode == "transcription"
    assert event.target_lang is None
    assert event.source_lang is None


def test_parse_translation_mode_with_target():
    event = parse_client_event(
        {"type": "language-config", "mode": "translation", "target_lang": "es"}
    )
    assert event.effective_mode == "translation"
    assert event.target_lang == "es"


def test_parse_translation_mode_requires_target():
    with pytest.raises(ValueError):
        parse_client_event({"type": "language-config", "mode": "translation"})


def test_parse_rejects_unknown_mode():
    with pytest.raises(ValueError):
        parse_client_event(
            {
                "type": "language-config",
                "mode": "interpretation",
                "target_lang": "es",
            }
        )


# --- WebSocket relay ---


@pytest.mark.asyncio
async def test_relay_transcription_mode_auto_detects_source():
    settings = make_settings()
    client = FakeClient(
        [
            json.dumps({"type": "language-config", "mode": "transcription"}),
            json.dumps({"type": "audio.append", "audio": AUDIO_B64}),
        ]
    )
    upstream = FakeUpstream()
    relay = await run_relay(client, upstream, settings)

    assert relay._session_mode == "transcription"
    assert relay._transcription_mode is True
    update = json.loads(upstream.sent[0])
    assert update["session"]["type"] == "transcription"
    transcription = update["session"]["audio"]["input"]["transcription"]
    assert transcription["model"] == settings.openai_transcription_model
    # No languages key: the model auto-detects the spoken language.
    # (An empty list is rejected upstream with "empty_array".)
    assert "languages" not in transcription
    assert json.loads(upstream.sent[1]) == {
        "type": "input_audio_buffer.append",
        "audio": AUDIO_B64,
    }


@pytest.mark.asyncio
async def test_relay_translation_mode_sets_only_target():
    settings = make_settings()
    client = FakeClient(
        [
            json.dumps(
                {
                    "type": "language-config",
                    "mode": "translation",
                    "target_lang": "es",
                }
            ),
            json.dumps({"type": "audio.append", "audio": AUDIO_B64}),
        ]
    )
    upstream = FakeUpstream()
    relay = await run_relay(client, upstream, settings)

    assert relay._session_mode == "translation"
    assert relay._transcription_mode is False
    update = json.loads(upstream.sent[0])
    assert update["session"]["audio"]["output"] == {"language": "es"}
    assert json.loads(upstream.sent[1]) == {
        "type": "session.input_audio_buffer.append",
        "audio": AUDIO_B64,
    }


@pytest.mark.asyncio
async def test_relay_mode_flips_transcription_to_translation():
    settings = make_settings()
    client = FakeClient(
        [
            json.dumps({"type": "language-config", "mode": "transcription"}),
            json.dumps(
                {
                    "type": "language-config",
                    "mode": "translation",
                    "target_lang": "fr",
                }
            ),
        ]
    )
    upstream = FakeUpstream()
    relay = await run_relay(client, upstream, settings)

    assert relay._session_mode == "translation"
    assert relay._transcription_mode is False
    assert json.loads(upstream.sent[0])["session"]["type"] == "transcription"
    assert json.loads(upstream.sent[1])["session"]["audio"]["output"] == {
        "language": "fr"
    }


class RecordingSubtitleTranslator:
    enabled = True

    def __init__(self):
        self.calls: list = []

    def clear_history(self) -> None:
        pass

    async def translate(self, text, **kwargs):
        self.calls.append(text)
        return "TRANSLATED:" + text

    async def aclose(self) -> None:
        pass


@pytest.mark.asyncio
async def test_relay_translation_mode_in_transcribe_deployment():
    # Production shape (TRANSCRIBE_MODE + subtitle translator): asking
    # for translation must not send "languages": [] upstream (that
    # ended the session with upstream_error:empty_array). It keeps
    # transcribing with auto-detect and translates the subtitle text.
    settings = make_settings(
        TRANSCRIBE_MODE=True, OPENAI_TRANSLATION_TARGET_LANGUAGE="hi"
    )
    client = FakeClient(
        [
            json.dumps(
                {
                    "type": "language-config",
                    "mode": "translation",
                    "target_lang": "hi",
                }
            )
        ]
    )
    upstream = FakeUpstream(
        [
            json.dumps(
                {
                    "type": "conversation.item.input_audio_transcription.delta",
                    "delta": "Hello",
                    "item_id": "item_1",
                }
            )
        ]
    )
    subtitle = RecordingSubtitleTranslator()
    relay = await run_relay(client, upstream, settings, subtitle=subtitle)

    assert relay._session_mode == "translation"
    assert relay._transcription_mode is True
    update = json.loads(upstream.sent[0])
    assert update["session"]["type"] == "transcription"
    assert (
        "languages"
        not in update["session"]["audio"]["input"]["transcription"]
    )
    assert subtitle.calls == ["Hello"]
    deltas = [json.loads(m) for m in client.sent if "transcript.delta" in m]
    assert deltas and "TRANSLATED:Hello" in deltas[0]["delta"]


@pytest.mark.asyncio
async def test_relay_transcription_mode_skips_subtitle_translation():
    settings = make_settings(TRANSCRIBE_MODE=True)
    client = FakeClient(
        [json.dumps({"type": "language-config", "mode": "transcription"})]
    )
    upstream = FakeUpstream(
        [
            json.dumps(
                {
                    "type": "conversation.item.input_audio_transcription.delta",
                    "delta": "Hello",
                    "item_id": "item_1",
                }
            )
        ]
    )
    subtitle = RecordingSubtitleTranslator()
    relay = await run_relay(client, upstream, settings, subtitle=subtitle)

    assert relay._session_mode == "transcription"
    assert subtitle.calls == []
    deltas = [json.loads(m) for m in client.sent if "transcript.delta" in m]
    assert deltas and deltas[0]["delta"] == "Hello"


# --- WebRTC relay ---


@pytest.mark.asyncio
async def test_webrtc_translation_mode_in_transcribe_deployment():
    # The reported failure: switching to translation on WebRTC ended
    # the session with upstream_error:empty_array.
    settings = make_settings(
        TRANSCRIBE_MODE=True, OPENAI_TRANSLATION_TARGET_LANGUAGE="hi"
    )
    channel, upstream = FakeChannel(), FakeUpstream()
    subtitle = RecordingSubtitleTranslator()
    relay = WebRTCRelaySession(channel, upstream, settings, subtitle=subtitle)
    task = asyncio.create_task(relay.run())
    await asyncio.sleep(0.05)
    relay.on_channel_message(
        json.dumps(
            {
                "type": "language-config",
                "mode": "translation",
                "target_lang": "hi",
            }
        )
    )
    await asyncio.sleep(0.1)
    relay.on_channel_close()
    await asyncio.wait_for(task, timeout=5)

    assert relay._session_mode == "translation"
    assert relay._transcription_mode is True
    update = json.loads(upstream.sent[0])
    assert update["session"]["type"] == "transcription"
    assert (
        "languages"
        not in update["session"]["audio"]["input"]["transcription"]
    )


@pytest.mark.asyncio
async def test_webrtc_transcription_mode_auto_detects_source():
    settings = make_settings()
    channel, upstream = FakeChannel(), FakeUpstream()
    relay = WebRTCRelaySession(channel, upstream, settings)
    task = asyncio.create_task(relay.run())
    await asyncio.sleep(0.05)
    relay.on_channel_message(
        json.dumps({"type": "language-config", "mode": "transcription"})
    )
    await asyncio.sleep(0.05)
    relay.on_channel_message(PCM_SMALL)
    await asyncio.sleep(0.1)
    relay.on_channel_close()
    await asyncio.wait_for(task, timeout=5)

    assert relay._session_mode == "transcription"
    assert relay._transcription_mode is True
    update = json.loads(upstream.sent[0])
    assert update["session"]["type"] == "transcription"
    assert (
        "languages"
        not in update["session"]["audio"]["input"]["transcription"]
    )
    assert json.loads(upstream.sent[1]) == {
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(PCM_SMALL).decode("ascii"),
    }


@pytest.mark.asyncio
async def test_webrtc_translation_mode_sets_only_target():
    settings = make_settings()
    channel, upstream = FakeChannel(), FakeUpstream()
    relay = WebRTCRelaySession(channel, upstream, settings)
    task = asyncio.create_task(relay.run())
    await asyncio.sleep(0.05)
    relay.on_channel_message(
        json.dumps(
            {
                "type": "language-config",
                "mode": "translation",
                "target_lang": "es",
            }
        )
    )
    await asyncio.sleep(0.05)
    relay.on_channel_message(PCM_SMALL)
    await asyncio.sleep(0.1)
    relay.on_channel_close()
    await asyncio.wait_for(task, timeout=5)

    assert relay._session_mode == "translation"
    assert relay._transcription_mode is False
    update = json.loads(upstream.sent[0])
    assert update["session"]["audio"]["output"] == {"language": "es"}
    assert json.loads(upstream.sent[1]) == {
        "type": "session.input_audio_buffer.append",
        "audio": base64.b64encode(PCM_SMALL).decode("ascii"),
    }
