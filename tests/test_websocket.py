"""WebSocket endpoint tests with a mocked upstream (no network, no API key)."""

import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.schemas.events import UpstreamError

AUDIO_B64 = "AQID"


class FakeUpstream:
    """Thread-safe fake: app loop drives recv, test thread injects events."""

    def __init__(self, connect_error: Exception | None = None):
        self.sent: list[str] = []
        self.closed = False
        self.connected = False
        self._pending: list[str] = []
        self._event: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._connect_error = connect_error

    async def connect(self) -> None:
        if self._connect_error is not None:
            raise self._connect_error
        self._loop = asyncio.get_running_loop()
        self._event = asyncio.Event()
        self.connected = True

    async def send_text(self, payload: str) -> None:
        self.sent.append(payload)

    async def recv_text(self) -> str:
        assert self._event is not None
        await self._event.wait()
        if self.closed and not self._pending:
            raise RuntimeError("upstream disconnected")
        message = self._pending.pop(0)
        if not self._pending:
            self._event.clear()
        return message

    def inject(self, event: dict) -> None:
        assert self._loop is not None and self._event is not None
        payload = json.dumps(event)
        self._loop.call_soon_threadsafe(self._pending.append, payload)
        self._loop.call_soon_threadsafe(self._event.set)

    async def close(self) -> None:
        self.closed = True
        if self._loop is not None and self._event is not None:
            self._loop.call_soon_threadsafe(self._event.set)


def make_client(**overrides):
    base = {"OPENAI_API_KEY": "test-key-fake-12345"}
    base.update(overrides)
    settings = Settings(**base)
    fakes: list[FakeUpstream] = []
    connect_error = overrides.pop("_connect_error", None) if "_connect_error" in overrides else None

    app = create_app(settings)

    def factory() -> FakeUpstream:
        fake = FakeUpstream(connect_error=connect_error)
        fakes.append(fake)
        return fake

    app.state.upstream_factory = factory
    return TestClient(app), fakes, settings


def wait_for(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_connect_without_auth_gets_ready():
    client, fakes, _ = make_client()
    with client.websocket_connect("/api/v1/transcription/ws") as ws:
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "session.ready"
        assert msg["model"] == "gpt-realtime-translate"
    assert fakes[0].connected
    assert wait_for(lambda: fakes[0].closed)


def test_audio_append_forwarded_verbatim_upstream():
    client, fakes, _ = make_client()
    with client.websocket_connect("/api/v1/transcription/ws") as ws:
        ws.receive_text()  # session.ready
        ws.send_text(json.dumps({"type": "audio.append", "audio": AUDIO_B64}))
        assert wait_for(lambda: len(fakes[0].sent) == 1)
        assert json.loads(fakes[0].sent[0]) == {
            "type": "session.input_audio_buffer.append",
            "audio": AUDIO_B64,
        }


def test_audio_commit_swallowed_never_forwarded():
    client, fakes, _ = make_client()
    with client.websocket_connect("/api/v1/transcription/ws") as ws:
        ws.receive_text()  # session.ready
        ws.send_text(json.dumps({"type": "audio.commit"}))
        # Give the relay a beat, then verify nothing went upstream.
        assert not wait_for(lambda: len(fakes[0].sent) == 1, timeout=0.5)
        assert fakes[0].sent == []


def test_native_openai_audio_events_accepted():
    client, fakes, _ = make_client()
    with client.websocket_connect("/api/v1/transcription/ws") as ws:
        ws.receive_text()
        ws.send_text(
            json.dumps({"type": "input_audio_buffer.append", "audio": AUDIO_B64})
        )
        assert wait_for(lambda: len(fakes[0].sent) == 1)
        assert json.loads(fakes[0].sent[0]) == {
            "type": "session.input_audio_buffer.append",
            "audio": AUDIO_B64,
        }


def test_native_translate_append_event_accepted():
    client, fakes, _ = make_client()
    with client.websocket_connect("/api/v1/transcription/ws") as ws:
        ws.receive_text()
        ws.send_text(
            json.dumps(
                {"type": "session.input_audio_buffer.append", "audio": AUDIO_B64}
            )
        )
        assert wait_for(lambda: len(fakes[0].sent) == 1)
        assert json.loads(fakes[0].sent[0]) == {
            "type": "session.input_audio_buffer.append",
            "audio": AUDIO_B64,
        }


def test_translated_transcript_deltas_forwarded():
    client, fakes, _ = make_client()
    with client.websocket_connect("/api/v1/transcription/ws") as ws:
        assert json.loads(ws.receive_text())["type"] == "session.ready"
        fakes[0].inject(
            {
                "type": "session.output_transcript.delta",
                "delta": "Hol",
            }
        )
        fakes[0].inject(
            {
                "type": "session.output_transcript.delta",
                "delta": "a.",
            }
        )
        first = json.loads(ws.receive_text())
        second = json.loads(ws.receive_text())
        assert first == {
            "type": "transcript.delta",
            "delta": "Hol",
            "item_id": "",
        }
        assert second == {
            "type": "transcript.delta",
            "delta": "a.",
            "item_id": "",
        }


def test_invalid_event_rejected_connection_survives():
    client, fakes, _ = make_client()
    with client.websocket_connect("/api/v1/transcription/ws") as ws:
        ws.receive_text()
        ws.send_text(json.dumps({"type": "bogus.event"}))
        err = json.loads(ws.receive_text())
        assert err["type"] == "error" and err["code"] == "invalid_event"
        ws.send_text("this is not json")
        err = json.loads(ws.receive_text())
        assert err["code"] == "invalid_event"
        # Connection still usable afterwards.
        ws.send_text(json.dumps({"type": "audio.append", "audio": AUDIO_B64}))
        assert wait_for(lambda: len(fakes[0].sent) == 1)


def test_oversized_message_rejected():
    client, fakes, _ = make_client(MAX_WEBSOCKET_MESSAGE_BYTES=32)
    with client.websocket_connect("/api/v1/transcription/ws") as ws:
        ws.receive_text()
        ws.send_text(json.dumps({"type": "audio.append", "audio": "A" * 64}))
        err = json.loads(ws.receive_text())
        assert err["code"] == "message_too_large"
    assert fakes[0].sent == []


def test_upstream_connect_failure_returns_safe_error():
    client, fakes, _ = make_client(
        _connect_error=UpstreamError(code="auth_failed", message="bad")
    )
    with client.websocket_connect("/api/v1/transcription/ws") as ws:
        err = json.loads(ws.receive_text())
        assert err["type"] == "error"
        assert err["code"] == "auth_failed"


def test_missing_api_key_rejected():
    client, _, _ = make_client(OPENAI_API_KEY="")
    with client.websocket_connect("/api/v1/transcription/ws") as ws:
        err = json.loads(ws.receive_text())
        assert err["code"] == "not_configured"


def test_client_disconnect_closes_upstream():
    client, fakes, _ = make_client()
    with client.websocket_connect("/api/v1/transcription/ws"):
        pass
    assert wait_for(lambda: fakes and fakes[0].closed)


def test_connection_limit_second_rejected():
    client, fakes, _ = make_client(MAX_CONCURRENT_SESSIONS=1)
    with client.websocket_connect("/api/v1/transcription/ws") as first:
        assert json.loads(first.receive_text())["type"] == "session.ready"
        with client.websocket_connect("/api/v1/transcription/ws") as second:
            err = json.loads(second.receive_text())
            assert err["code"] == "server_busy"
    assert len(fakes) == 1  # rejected session never touched upstream


def test_disallowed_origin_rejected():
    client, _, _ = make_client(ALLOWED_WEBSOCKET_ORIGINS="https://example.com")
    with client.websocket_connect(
        "/api/v1/transcription/ws", headers={"origin": "https://evil.test"}
    ) as ws:
        with pytest.raises(Exception):
            ws.receive_text()


def test_no_api_key_leakage_in_responses(caplog):
    secret = "test-key-fake-12345"
    client, fakes, _ = make_client()
    seen: list[str] = []
    with client.websocket_connect("/api/v1/transcription/ws") as ws:
        seen.append(ws.receive_text())
        ws.send_text(json.dumps({"type": "audio.append", "audio": AUDIO_B64}))
        assert wait_for(lambda: len(fakes[0].sent) == 1)
        fakes[0].inject(
            {
                "type": "session.output_transcript.delta",
                "delta": "hi",
            }
        )
        seen.append(ws.receive_text())
        ws.send_text(json.dumps({"type": "bogus"}))
        seen.append(ws.receive_text())
    assert all(secret not in m for m in seen)
    assert all(secret not in f for f in fakes[0].sent)
