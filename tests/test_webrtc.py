"""WebRTC DataChannel relay tests (mocked channel + upstream, no aiortc).

Covers the extensionRTC contract: raw Int16 PCM binary messages become
base64 upstream appends, commits are auto-issued while audio flows and
flushed on channel close, transcripts map back onto the channel, and the
signaling endpoint rejects misconfigured/over-limit offers before
touching aiortc or the network.
"""

import asyncio
import base64
import json

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.webrtc_session import (
    WebRTCRelaySession,
    webrtc_audio_to_upstream_payload,
)

# One nominal extension chunk: 200 ms of 24 kHz mono Int16 = 9600 bytes.
PCM_CHUNK = b"\x01\x02" * 4800
PCM_SMALL = b"\x01\x02\x03\x04"


def make_settings(**overrides):
    base = {
        "OPENAI_API_KEY": "test-key",
        "SESSION_IDLE_TIMEOUT_SECONDS": 5.0,
        "MAX_AUDIO_BYTES_PER_SECOND": 96000,
        "MAX_WEBSOCKET_MESSAGE_BYTES": 65536,
        "WEBRTC_AUTO_COMMIT_SECONDS": 1000.0,
    }
    base.update(overrides)
    return Settings(**base)


class FakeChannel:
    """In-memory stand-in for aiortc's RTCDataChannel (sync send)."""

    def __init__(self):
        self.sent: list = []
        self.readyState = "open"
        self.fail_send = False

    def send(self, message) -> None:
        if self.fail_send:
            raise RuntimeError("channel closed")
        self.sent.append(message)


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


async def run_session(channel, upstream, settings, feed: list):
    """Feed messages, run until channel close, return the stop reason."""
    relay = WebRTCRelaySession(channel, upstream, settings)
    task = asyncio.create_task(relay.run())
    await asyncio.sleep(0.05)
    for message in feed:
        relay.on_channel_message(message)
        await asyncio.sleep(0.05)
    relay.on_channel_close()
    return await asyncio.wait_for(task, timeout=5)


def channel_jsons(channel):
    return [json.loads(m) for m in channel.sent if isinstance(m, str)]


def test_payload_roundtrip_pcm16():
    payload = webrtc_audio_to_upstream_payload(PCM_SMALL)
    decoded = json.loads(payload)
    assert decoded["type"] == "input_audio_buffer.append"
    assert base64.b64decode(decoded["audio"]) == PCM_SMALL


def test_payload_rejects_empty_and_odd_length():
    with pytest.raises(ValueError):
        webrtc_audio_to_upstream_payload(b"")
    with pytest.raises(ValueError):
        webrtc_audio_to_upstream_payload(b"\x01\x02\x03")


@pytest.mark.asyncio
async def test_binary_pcm_forwarded_as_base64_append():
    channel, upstream = FakeChannel(), FakeUpstream()
    reason = await run_session(
        channel, upstream, make_settings(), [PCM_CHUNK]
    )
    assert reason == "channel_closed"
    assert json.loads(channel.sent[0])["type"] == "session.ready"
    assert json.loads(upstream.sent[0]) == {
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(PCM_CHUNK).decode("ascii"),
    }
    assert upstream.closed


@pytest.mark.asyncio
async def test_odd_length_and_empty_binary_rejected():
    channel, upstream = FakeChannel(), FakeUpstream()
    await run_session(channel, upstream, make_settings(), [b"\x01\x02\x03", b""])
    errors = [m for m in channel_jsons(channel) if m["type"] == "error"]
    assert len(errors) == 2
    assert all(e["code"] == "invalid_event" for e in errors)
    assert upstream.sent == []


@pytest.mark.asyncio
async def test_oversized_binary_rejected():
    channel, upstream = FakeChannel(), FakeUpstream()
    reason = await run_session(
        channel,
        upstream,
        make_settings(MAX_WEBSOCKET_MESSAGE_BYTES=10),
        [b"\x01\x02" * 8],
    )
    assert reason == "message_too_large"
    assert channel_jsons(channel)[-1]["code"] == "message_too_large"
    assert upstream.sent == []


@pytest.mark.asyncio
async def test_explicit_text_commit_forwarded():
    channel, upstream = FakeChannel(), FakeUpstream()
    await run_session(
        channel, upstream, make_settings(), [json.dumps({"type": "audio.commit"})]
    )
    assert json.loads(upstream.sent[0]) == {
        "type": "input_audio_buffer.commit"
    }


@pytest.mark.asyncio
async def test_text_base64_append_accepted_for_compat():
    channel, upstream = FakeChannel(), FakeUpstream()
    audio_b64 = base64.b64encode(PCM_SMALL).decode("ascii")
    await run_session(
        channel,
        upstream,
        make_settings(),
        [json.dumps({"type": "audio.append", "audio": audio_b64})],
    )
    assert json.loads(upstream.sent[0]) == {
        "type": "input_audio_buffer.append",
        "audio": audio_b64,
    }


@pytest.mark.asyncio
async def test_auto_commit_while_streaming():
    channel, upstream = FakeChannel(), FakeUpstream()
    relay = WebRTCRelaySession(
        channel, upstream, make_settings(WEBRTC_AUTO_COMMIT_SECONDS=0.05)
    )
    task = asyncio.create_task(relay.run())
    await asyncio.sleep(0.05)
    relay.on_channel_message(PCM_SMALL)
    await asyncio.sleep(0.15)
    relay.on_channel_message(PCM_SMALL)
    await asyncio.sleep(0.1)
    relay.on_channel_close()
    await asyncio.wait_for(task, timeout=5)
    kinds = [json.loads(p)["type"] for p in upstream.sent]
    assert kinds.count("input_audio_buffer.append") == 2
    assert "input_audio_buffer.commit" in kinds


@pytest.mark.asyncio
async def test_final_commit_on_channel_close():
    channel, upstream = FakeChannel(), FakeUpstream()
    await run_session(channel, upstream, make_settings(), [PCM_SMALL])
    kinds = [json.loads(p)["type"] for p in upstream.sent]
    assert kinds == ["input_audio_buffer.append", "input_audio_buffer.commit"]


@pytest.mark.asyncio
async def test_throughput_limit_drops_chunk_with_error():
    channel, upstream = FakeChannel(), FakeUpstream()
    await run_session(
        channel,
        upstream,
        make_settings(MAX_AUDIO_BYTES_PER_SECOND=3),
        [PCM_SMALL],
    )
    assert channel_jsons(channel)[-1]["code"] == "rate_limited"
    assert upstream.sent == []


@pytest.mark.asyncio
async def test_transcript_events_mapped_to_channel():
    upstream = FakeUpstream(
        [
            json.dumps(
                {
                    "type": "conversation.item.input_audio_transcription.delta",
                    "item_id": "item_1",
                    "delta": "Hel",
                }
            ),
            json.dumps(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": "item_1",
                    "transcript": "Hello.",
                }
            ),
        ]
    )
    channel = FakeChannel()
    relay = WebRTCRelaySession(channel, upstream, make_settings())
    task = asyncio.create_task(relay.run())
    await asyncio.sleep(0.3)
    relay.on_channel_close()
    upstream.disconnect()
    await asyncio.wait_for(task, timeout=5)
    kinds = [m["type"] for m in channel_jsons(channel)]
    assert kinds == ["session.ready", "transcript.delta", "transcript.completed"]


@pytest.mark.asyncio
async def test_upstream_error_forwarded_safely():
    upstream = FakeUpstream(
        [json.dumps({"type": "error", "error": {"code": "rate_limited"}})]
    )
    channel = FakeChannel()
    relay = WebRTCRelaySession(channel, upstream, make_settings())
    reason = await asyncio.wait_for(relay.run(), timeout=5)
    assert reason == "upstream_error:rate_limited"
    assert channel_jsons(channel)[-1]["type"] == "error"
    assert upstream.closed


@pytest.mark.asyncio
async def test_idle_timeout_closes_session():
    channel, upstream = FakeChannel(), FakeUpstream()
    relay = WebRTCRelaySession(
        channel, upstream, make_settings(SESSION_IDLE_TIMEOUT_SECONDS=0.1)
    )
    reason = await asyncio.wait_for(relay.run(), timeout=5)
    assert reason == "idle_timeout"
    assert channel_jsons(channel)[-1]["code"] == "idle_timeout"
    assert upstream.closed


@pytest.mark.asyncio
async def test_channel_send_failure_ends_session():
    channel, upstream = FakeChannel(), FakeUpstream()
    channel.fail_send = True
    relay = WebRTCRelaySession(channel, upstream, make_settings())
    reason = await asyncio.wait_for(relay.run(), timeout=5)
    assert reason == "channel_closed"


def make_client(**overrides):
    base = {"OPENAI_API_KEY": "test-key"}
    base.update(overrides)
    return TestClient(create_app(Settings(**base)))


def test_offer_rejected_without_api_key():
    client = make_client(OPENAI_API_KEY="")
    response = client.post(
        "/api/v1/transcription/webrtc/offer",
        json={"sdp": "v=0", "type": "offer"},
    )
    assert response.status_code == 503
    assert response.json()["code"] == "not_configured"


def test_offer_rejected_disallowed_origin():
    client = make_client(ALLOWED_WEBSOCKET_ORIGINS="https://example.com")
    response = client.post(
        "/api/v1/transcription/webrtc/offer",
        json={"sdp": "v=0", "type": "offer"},
        headers={"origin": "https://evil.test"},
    )
    assert response.status_code == 403
    assert response.json()["code"] == "origin_not_allowed"


def test_offer_rejected_when_full():
    client = make_client(MAX_CONCURRENT_SESSIONS=0)
    response = client.post(
        "/api/v1/transcription/webrtc/offer",
        json={"sdp": "v=0", "type": "offer"},
    )
    assert response.status_code == 429
    assert response.json()["code"] == "server_busy"


def test_offer_validation_rejects_empty_sdp():
    client = make_client()
    response = client.post(
        "/api/v1/transcription/webrtc/offer",
        json={"sdp": "", "type": "offer"},
    )
    assert response.status_code == 422
