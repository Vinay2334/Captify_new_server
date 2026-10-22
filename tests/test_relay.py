"""Unit tests for the bidirectional relay (mocked sockets, no network)."""

import asyncio
import json

import pytest

from app.core.config import Settings
from app.services.relay import (
    AudioThroughputLimiter,
    RelaySession,
    decode_audio_size,
    to_upstream_payload,
)
from app.schemas.events import AudioAppendEvent, UpstreamError


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
        self.close_event = asyncio.Event()

    async def send_text(self, message: str) -> None:
        self.sent.append(message)

    async def receive_text(self) -> str:
        item = await self.incoming.get()
        if item is None:
            raise RuntimeError("client disconnected")
        return item

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = (code, reason)
        self.close_event.set()

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


AUDIO_B64 = "AQID"  # 3 decoded bytes, valid base64


def test_decode_audio_size_passthrough():
    assert decode_audio_size(AUDIO_B64) == 3
    with pytest.raises(ValueError):
        decode_audio_size("!!!not-base64!!!")


def test_to_upstream_payload_keeps_audio_verbatim():
    payload = to_upstream_payload(AudioAppendEvent(type="audio.append", audio=AUDIO_B64))
    assert json.loads(payload) == {
        "type": "input_audio_buffer.append",
        "audio": AUDIO_B64,
    }


def test_throughput_limiter():
    limiter = AudioThroughputLimiter(10)
    assert limiter.allow(6) is True
    assert limiter.allow(5) is False
    assert limiter.allow(4) is True


@pytest.mark.asyncio
async def test_audio_append_forwarded_verbatim():
    client = FakeClient([json.dumps({"type": "audio.append", "audio": AUDIO_B64})])
    upstream = FakeUpstream()
    relay = RelaySession(client, upstream, make_settings())
    task = asyncio.create_task(relay.run())
    await asyncio.sleep(0.2)
    client.disconnect()
    upstream.disconnect()
    reason = await asyncio.wait_for(task, timeout=5)
    assert reason in ("client_disconnected", "upstream_disconnected", "closed")
    assert len(upstream.sent) == 1
    assert json.loads(upstream.sent[0]) == {
        "type": "input_audio_buffer.append",
        "audio": AUDIO_B64,
    }
    assert upstream.closed and client.closed is not None


@pytest.mark.asyncio
async def test_audio_commit_forwarded():
    client = FakeClient([json.dumps({"type": "audio.commit"})])
    upstream = FakeUpstream()
    relay = RelaySession(client, upstream, make_settings())
    task = asyncio.create_task(relay.run())
    await asyncio.sleep(0.2)
    client.disconnect()
    upstream.disconnect()
    await asyncio.wait_for(task, timeout=5)
    assert json.loads(upstream.sent[0]) == {"type": "input_audio_buffer.commit"}


@pytest.mark.asyncio
async def test_transcript_events_mapped_downstream():
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
            json.dumps({"type": "session.updated", "session": {}}),
        ]
    )
    client = FakeClient()
    relay = RelaySession(client, upstream, make_settings())
    task = asyncio.create_task(relay.run())
    await asyncio.sleep(0.3)
    client.disconnect()
    upstream.disconnect()
    await asyncio.wait_for(task, timeout=5)
    kinds = [json.loads(m)["type"] for m in client.sent]
    assert kinds == ["transcript.delta", "transcript.completed"]
    assert json.loads(client.sent[0])["delta"] == "Hel"
    assert json.loads(client.sent[1])["transcript"] == "Hello."


@pytest.mark.asyncio
async def test_invalid_client_event_rejected_connection_survives():
    client = FakeClient(
        [
            json.dumps({"type": "nope"}),
            "not json at all",
            json.dumps({"type": "audio.append", "audio": AUDIO_B64}),
        ]
    )
    upstream = FakeUpstream()
    relay = RelaySession(client, upstream, make_settings())
    task = asyncio.create_task(relay.run())
    await asyncio.sleep(0.3)
    client.disconnect()
    upstream.disconnect()
    await asyncio.wait_for(task, timeout=5)
    errors = [json.loads(m) for m in client.sent if "error" in m]
    assert len(errors) == 2
    assert all(e["type"] == "error" for e in errors)
    # Only the valid event reached upstream.
    assert len(upstream.sent) == 1


@pytest.mark.asyncio
async def test_oversized_message_rejected_and_closed():
    big = "A" * 100
    client = FakeClient([json.dumps({"type": "audio.append", "audio": big})])
    upstream = FakeUpstream()
    relay = RelaySession(
        client, upstream, make_settings(MAX_WEBSOCKET_MESSAGE_BYTES=20)
    )
    reason = await asyncio.wait_for(relay.run(), timeout=5)
    assert reason == "message_too_large"
    assert json.loads(client.sent[0])["code"] == "message_too_large"
    assert client.closed is not None and client.closed[0] == 1009
    assert upstream.sent == []


@pytest.mark.asyncio
async def test_throughput_limit_drops_chunk_with_error():
    client = FakeClient(
        [
            json.dumps({"type": "audio.append", "audio": AUDIO_B64}),
            json.dumps({"type": "audio.commit"}),
        ]
    )
    upstream = FakeUpstream()
    # 3 decoded bytes per chunk, budget of 3: first ok, rest dropped.
    relay = RelaySession(client, upstream, make_settings(MAX_AUDIO_BYTES_PER_SECOND=3))
    task = asyncio.create_task(relay.run())
    await asyncio.sleep(0.3)
    client.disconnect()
    upstream.disconnect()
    await asyncio.wait_for(task, timeout=5)
    assert len(upstream.sent) == 1  # commit dropped, append forwarded
    assert json.loads(client.sent[0])["code"] == "rate_limited"


@pytest.mark.asyncio
async def test_upstream_error_forwarded_safely():
    upstream = FakeUpstream(
        [json.dumps({"type": "error", "error": {"code": "rate_limited"}})]
    )
    client = FakeClient()
    relay = RelaySession(client, upstream, make_settings())
    reason = await asyncio.wait_for(relay.run(), timeout=5)
    assert reason == "upstream_error:rate_limited"
    assert json.loads(client.sent[0])["type"] == "error"
    assert upstream.closed and client.closed is not None


@pytest.mark.asyncio
async def test_upstream_error_event_redacts_raw_payload():
    err = UpstreamError.from_event(
        {"type": "error", "error": {"code": "x" * 200, "message": "secret detail"}}
    )
    assert err.code == "upstream_error"
    assert "secret" not in err.message


@pytest.mark.asyncio
async def test_idle_timeout_closes_client():
    client = FakeClient()
    upstream = FakeUpstream()
    relay = RelaySession(
        client, upstream, make_settings(SESSION_IDLE_TIMEOUT_SECONDS=0.1)
    )
    reason = await asyncio.wait_for(relay.run(), timeout=5)
    assert reason == "idle_timeout"
    assert json.loads(client.sent[0])["code"] == "idle_timeout"
    assert upstream.closed


@pytest.mark.asyncio
async def test_relay_tasks_cancelled_on_disconnect():
    client = FakeClient()
    upstream = FakeUpstream()
    relay = RelaySession(client, upstream, make_settings())
    task = asyncio.create_task(relay.run())
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # run() shields nothing: cancellation propagates after cleanup.
    assert client.closed is not None or upstream.closed
