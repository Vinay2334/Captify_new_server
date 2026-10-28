"""Focused coverage for the debug audio recorder (SAVE_RECEIVED_AUDIO).

The writer is stdlib-only; the relay hook is exercised with the same
fake channel/upstream pattern as test_webrtc.py.
"""

import asyncio
import base64
import wave

import pytest

from app.core.config import Settings
from app.services.audio_dump import AudioDumpWriter
from app.services.webrtc_session import WebRTCRelaySession

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


class FakeChannel:
    def __init__(self):
        self.sent: list = []
        self.readyState = "open"

    def send(self, message) -> None:
        self.sent.append(message)


class FakeUpstream:
    def __init__(self):
        self.sent: list[str] = []
        self.closed = False
        self._disconnected = asyncio.Event()

    async def send_text(self, payload: str) -> None:
        self.sent.append(payload)

    async def recv_text(self) -> str:
        await self._disconnected.wait()
        raise RuntimeError("upstream disconnected")

    async def close(self) -> None:
        self.closed = True


def test_writer_roundtrips_pcm16(tmp_path):
    writer = AudioDumpWriter(str(tmp_path))
    writer.write(PCM_SMALL)
    writer.write(PCM_SMALL)
    path = writer.close()
    assert path is not None
    with wave.open(path, "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 24000
        assert wav.readframes(wav.getnframes()) == PCM_SMALL + PCM_SMALL


def test_writer_creates_no_file_without_audio(tmp_path):
    writer = AudioDumpWriter(str(tmp_path))
    writer.write(b"")
    assert writer.close() is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_webrtc_binary_audio_recorded(tmp_path):
    settings = make_settings(
        SAVE_RECEIVED_AUDIO=True, RECEIVED_AUDIO_DIR=str(tmp_path)
    )
    channel, upstream = FakeChannel(), FakeUpstream()
    relay = WebRTCRelaySession(channel, upstream, settings)
    task = asyncio.create_task(relay.run())
    await asyncio.sleep(0.05)
    relay.on_channel_message(PCM_SMALL)
    await asyncio.sleep(0.05)
    relay.on_channel_close()
    await asyncio.wait_for(task, timeout=5)

    wavs = list(tmp_path.glob("*.wav"))
    assert len(wavs) == 1
    with wave.open(str(wavs[0]), "rb") as wav:
        assert wav.getframerate() == 24000
        assert wav.readframes(wav.getnframes()) == PCM_SMALL


@pytest.mark.asyncio
async def test_webrtc_base64_text_audio_recorded(tmp_path):
    settings = make_settings(
        SAVE_RECEIVED_AUDIO=True, RECEIVED_AUDIO_DIR=str(tmp_path)
    )
    channel, upstream = FakeChannel(), FakeUpstream()
    relay = WebRTCRelaySession(channel, upstream, settings)
    task = asyncio.create_task(relay.run())
    await asyncio.sleep(0.05)
    import json

    relay.on_channel_message(
        json.dumps(
            {
                "type": "audio.append",
                "audio": base64.b64encode(PCM_SMALL).decode("ascii"),
            }
        )
    )
    await asyncio.sleep(0.05)
    relay.on_channel_close()
    await asyncio.wait_for(task, timeout=5)

    wavs = list(tmp_path.glob("*.wav"))
    assert len(wavs) == 1
    with wave.open(str(wavs[0]), "rb") as wav:
        assert wav.readframes(wav.getnframes()) == PCM_SMALL


@pytest.mark.asyncio
async def test_recorder_off_by_default(tmp_path):
    settings = make_settings(RECEIVED_AUDIO_DIR=str(tmp_path))
    assert settings.save_received_audio is False
    channel, upstream = FakeChannel(), FakeUpstream()
    relay = WebRTCRelaySession(channel, upstream, settings)
    task = asyncio.create_task(relay.run())
    await asyncio.sleep(0.05)
    relay.on_channel_message(PCM_SMALL)
    await asyncio.sleep(0.05)
    relay.on_channel_close()
    await asyncio.wait_for(task, timeout=5)
    assert list(tmp_path.iterdir()) == []
