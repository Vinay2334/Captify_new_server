"""STT provider factory and Soniox workflow tests (mocked, no network)."""

import asyncio
import base64
import json

import pytest

from app.api.websocket import build_upstream_client
from app.core.config import Settings
from app.schemas.events import (
    AudioAppendEvent,
    AudioCommitEvent,
    TranscriptDeltaEvent,
    UpstreamError,
)
from app.services.openai_realtime import OpenAIRealtimeClient
from app.services.relay import RelaySession
from app.services.soniox_realtime import (
    SonioxRealtimeClient,
    build_soniox_config,
    build_soniox_transcribe_config,
    map_soniox_event,
)
from app.services.stt_factory import (
    OpenAIWorkflow,
    SonioxWorkflow,
    STTWorkflowFactory,
)
from app.services.webrtc_session import WebRTCRelaySession

AUDIO_B64 = "AQID"  # 3 decoded bytes, valid base64
PCM_SMALL = b"\x01\x02\x03\x04"


def make_settings(**overrides):
    base = {
        "OPENAI_API_KEY": "test-key",
        "SONIOX_API_KEY": "soniox-test-key",
        "TRANSCRIBE_MODE": False,
        "SESSION_IDLE_TIMEOUT_SECONDS": 5.0,
        "MAX_AUDIO_BYTES_PER_SECOND": 96000,
        "MAX_WEBSOCKET_MESSAGE_BYTES": 65536,
    }
    base.update(overrides)
    return Settings(**base)


def soniox_settings(**overrides):
    return make_settings(STT_PROVIDER="soniox", **overrides)


class FakeClient:
    def __init__(self, incoming=None):
        self.incoming = asyncio.Queue()
        for item in incoming or []:
            self.incoming.put_nowait(item)
        self.sent = []
        self.closed = None

    async def send_text(self, message):
        self.sent.append(message)

    async def receive_text(self):
        item = await self.incoming.get()
        if item is None:
            raise RuntimeError("client disconnected")
        return item

    async def close(self, code=1000, reason=""):
        self.closed = (code, reason)

    def disconnect(self):
        self.incoming.put_nowait(None)


class FakeUpstream:
    def __init__(self, incoming=None):
        self.incoming = asyncio.Queue()
        for item in incoming or []:
            self.incoming.put_nowait(item)
        self.sent = []
        self.closed = False

    async def send_text(self, payload):
        self.sent.append(payload)

    async def recv_text(self):
        item = await self.incoming.get()
        if item is None:
            raise RuntimeError("upstream disconnected")
        return item

    async def close(self):
        self.closed = True

    def disconnect(self):
        self.incoming.put_nowait(None)


class FakeChannel:
    def __init__(self):
        self.sent = []
        self.readyState = "open"

    def send(self, message):
        self.sent.append(message)


# --- provider switch ---


def test_default_provider_is_openai():
    assert make_settings().stt_provider == "openai"
    assert make_settings().is_soniox_provider() is False
    assert make_settings().upstream_api_key() == "test-key"


def test_invalid_provider_rejected():
    import pydantic

    with pytest.raises((ValueError, pydantic.ValidationError)):
        make_settings(STT_PROVIDER="whisper")


def test_soniox_settings_helpers():
    settings = soniox_settings()
    assert settings.is_soniox_provider() is True
    assert settings.upstream_api_key() == "soniox-test-key"
    assert settings.active_model() == "stt-rt-v5"
    assert settings.soniox_language_hints() == ["en"]
    hinted = soniox_settings(OPENAI_TRANSLATION_SOURCE_LANGUAGE="hi")
    assert hinted.soniox_language_hints() == ["hi", "en"]


def test_factory_selects_workflow():
    assert isinstance(STTWorkflowFactory.get_workflow(make_settings()), OpenAIWorkflow)
    assert isinstance(
        STTWorkflowFactory.get_workflow(soniox_settings()), SonioxWorkflow
    )
    assert STTWorkflowFactory.provider_name(soniox_settings()) == "soniox"


def test_factory_builds_openai_client_by_default():
    client = STTWorkflowFactory.create_upstream_client(make_settings())
    assert isinstance(client, OpenAIRealtimeClient)
    assert client.is_transcription is False
    assert client.model == "gpt-realtime-translate"


def test_factory_builds_soniox_client():
    client = STTWorkflowFactory.create_upstream_client(soniox_settings())
    assert isinstance(client, SonioxRealtimeClient)
    assert client.model == "stt-rt-v5"
    assert client.target_language == "en"
    assert build_upstream_client(soniox_settings()).model == "stt-rt-v5"


def test_factory_builds_soniox_transcribe_client_for_flag():
    client = STTWorkflowFactory.create_upstream_client(
        soniox_settings(TRANSCRIBE_MODE=True)
    )
    assert isinstance(client, SonioxRealtimeClient)
    assert client.is_transcription is True


# --- soniox config builders ---


def test_soniox_translation_config_shape():
    config = json.loads(build_soniox_config("es", language_hints=["en", "es"]))
    assert config["model"] == "stt-rt-v5"
    assert config["audio_format"] == "pcm_s16le"
    assert config["sample_rate"] == 24000
    assert config["num_channels"] == 1
    assert config["language_hints"] == ["en", "es"]
    assert config["translation"] == {"type": "one_way", "target_language": "es"}


def test_soniox_transcribe_config_has_no_translation():
    config = json.loads(build_soniox_transcribe_config("en"))
    assert config["model"] == "stt-rt-v5"
    assert "translation" not in config
    assert config["language_hints"] == ["en"]


# --- soniox payload / event mapping ---


def test_soniox_workflow_audio_payload_is_raw_base64():
    workflow = SonioxWorkflow()
    payload = workflow.to_upstream_payload(
        AudioAppendEvent(type="audio.append", audio=AUDIO_B64)
    )
    assert payload == AUDIO_B64
    assert workflow.to_upstream_payload(AudioCommitEvent(type="audio.commit")) is None


def test_soniox_workflow_webrtc_audio_payload():
    workflow = SonioxWorkflow()
    assert workflow.webrtc_audio_payload(PCM_SMALL) == base64.b64encode(PCM_SMALL).decode(
        "ascii"
    )
    with pytest.raises(ValueError):
        workflow.webrtc_audio_payload(b"")
    with pytest.raises(ValueError):
        workflow.webrtc_audio_payload(b"\x01\x02\x03")


def test_soniox_final_translation_tokens_map_to_delta():
    mapped = map_soniox_event(
        {
            "tokens": [
                {"text": "Hola", "is_final": True, "translation_status": "translation"},
                {"text": "Hel", "is_final": True, "translation_status": "original"},
                {"text": "…", "is_final": False, "translation_status": "translation"},
            ],
            "finished": False,
        }
    )
    assert isinstance(mapped, TranscriptDeltaEvent)
    assert mapped.delta == "Hola"


def test_soniox_provisional_and_empty_responses_map_to_none():
    assert (
        map_soniox_event(
            {
                "tokens": [
                    {"text": "Ho", "is_final": False, "translation_status": "translation"}
                ],
                "finished": False,
            }
        )
        is None
    )
    assert map_soniox_event({"tokens": [], "finished": False}) is None
    assert map_soniox_event({"tokens": [], "finished": True}) is None


def test_soniox_original_tokens_map_only_in_transcription_mode():
    event = {
        "tokens": [
            {"text": "Hello", "is_final": True, "translation_status": "original"}
        ],
        "finished": False,
    }
    assert map_soniox_event(event) is None
    mapped = map_soniox_event(event, transcription_mode=True)
    assert isinstance(mapped, TranscriptDeltaEvent)
    assert mapped.delta == "Hello"


def test_soniox_error_raises_upstream_error():
    with pytest.raises(UpstreamError):
        map_soniox_event({"type": "error", "error": {"code": "rate_limited"}})
    with pytest.raises(UpstreamError):
        map_soniox_event({"tokens": [], "error": "boom"})


def test_soniox_language_updates_need_no_upstream_send():
    workflow = SonioxWorkflow()
    assert workflow.build_language_update("es", soniox_settings()) is None
    assert workflow.build_transcription_update("m", ["en"]) is None


# --- relay integration on the soniox workflow ---


@pytest.mark.asyncio
async def test_relay_soniox_audio_and_translation_deltas():
    client = FakeClient([json.dumps({"type": "audio.append", "audio": AUDIO_B64})])
    upstream = FakeUpstream(
        [
            json.dumps(
                {
                    "tokens": [
                        {
                            "text": "Hola.",
                            "is_final": True,
                            "translation_status": "translation",
                        }
                    ],
                    "finished": False,
                }
            ),
        ]
    )
    relay = RelaySession(client, upstream, soniox_settings())
    task = asyncio.create_task(relay.run())
    await asyncio.sleep(0.3)
    client.disconnect()
    upstream.disconnect()
    await asyncio.wait_for(task, timeout=5)
    # Audio travels as a raw base64 text frame, not OpenAI JSON.
    assert upstream.sent == [AUDIO_B64]
    deltas = [json.loads(m) for m in client.sent if "transcript.delta" in m]
    assert [d["delta"] for d in deltas] == ["Hola."]


@pytest.mark.asyncio
async def test_relay_soniox_language_config_flips_mode_without_send():
    client = FakeClient(
        [
            json.dumps(
                {"type": "language-config", "source_lang": "en", "target_lang": "en"}
            ),
            json.dumps({"type": "audio.commit"}),
        ]
    )
    upstream = FakeUpstream()
    relay = RelaySession(client, upstream, soniox_settings())
    task = asyncio.create_task(relay.run())
    await asyncio.sleep(0.3)
    client.disconnect()
    upstream.disconnect()
    await asyncio.wait_for(task, timeout=5)
    assert relay._transcription_mode is True
    # Fixed at connect: nothing re-sent upstream, commits swallowed.
    assert upstream.sent == []


@pytest.mark.asyncio
async def test_webrtc_soniox_binary_audio_forwarded_as_base64_frame():
    channel, upstream = FakeChannel(), FakeUpstream()
    relay = WebRTCRelaySession(channel, upstream, soniox_settings())
    task = asyncio.create_task(relay.run())
    await asyncio.sleep(0.05)
    relay.on_channel_message(PCM_SMALL)
    await asyncio.sleep(0.1)
    relay.on_channel_close()
    await asyncio.wait_for(task, timeout=5)
    assert upstream.sent == [base64.b64encode(PCM_SMALL).decode("ascii")]
    assert json.loads(channel.sent[0]) == {
        "type": "session.ready",
        "model": "stt-rt-v5",
    }
