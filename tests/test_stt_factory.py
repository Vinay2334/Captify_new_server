"""STT provider factory and Soniox workflow tests (mocked, no network).

Soniox is transcription-only: the session config never carries a
``translation`` block and only original (source-language) final tokens
map downstream. Translation (when a session asks for it) is a separate
text-to-text step over those transcripts, identical for every tier.

Tiers: ``base`` transcribes with Soniox ``stt-rt-v5``; ``pro``
transcribes with OpenAI ``gpt-live-transcribe``. Both emit the same
``transcript.delta`` / ``transcript.completed`` downstream shape.
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
    TranscriptDeltaEvent,
    UpstreamError,
    normalize_model_tier,
)
from app.services.openai_realtime import OpenAIRealtimeClient
from app.services.relay import RelaySession
from app.services.soniox_realtime import (
    SonioxRealtimeClient,
    build_soniox_transcribe_config,
    map_soniox_event,
)
from app.services.stt_factory import (
    OpenAITranscriptionWorkflow,
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
        # Pinned so ambient shell exports (e.g. a SONIOX_MODEL experiment)
        # cannot change what these unit tests verify (wiring, not names).
        "SONIOX_MODEL": "stt-rt-v5",
        "OPENAI_TRANSCRIPTION_MODEL": "gpt-live-transcribe",
        "TRANSCRIBE_MODE": False,
        "OPENAI_TRANSLATION_SOURCE_LANGUAGE": "",
        "OPENAI_TRANSLATION_TARGET_LANGUAGE": "en",
        "SESSION_IDLE_TIMEOUT_SECONDS": 5.0,
        "MAX_AUDIO_BYTES_PER_SECOND": 96000,
        "MAX_WEBSOCKET_MESSAGE_BYTES": 65536,
    }
    base.update(overrides)
    base.setdefault("SUBTITLE_PROVIDER", "none")
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
        # Like a real socket close, unblock a pending recv so the relay's
        # reader task observes the generation bump of a hot-swap instead
        # of hanging on the old socket.
        self.closed = True
        self.incoming.put_nowait(None)

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
    assert client.is_transcription is True
    assert build_upstream_client(soniox_settings()).model == "stt-rt-v5"


def test_factory_builds_soniox_transcribe_client_for_flag():
    client = STTWorkflowFactory.create_upstream_client(
        soniox_settings(TRANSCRIBE_MODE=True)
    )
    assert isinstance(client, SonioxRealtimeClient)
    assert client.is_transcription is True


# --- model tiers (base Soniox / pro gpt-live-transcribe) ---


def test_normalize_model_tier():
    assert normalize_model_tier("base") == "base"
    assert normalize_model_tier(" Pro ") == "pro"
    assert normalize_model_tier("BASE") == "base"
    assert normalize_model_tier(None) is None
    assert normalize_model_tier("") is None
    assert normalize_model_tier("ultra") is None


def test_default_model_tier_follows_provider():
    assert make_settings().default_model_tier() == "pro"
    assert soniox_settings().default_model_tier() == "base"


def test_active_model_for_tier():
    settings = make_settings()
    assert settings.active_model_for_tier("base") == "stt-rt-v5"
    assert (
        settings.active_model_for_tier("pro")
        == settings.openai_transcription_model
    )
    assert settings.active_model_for_tier(None) == settings.active_model()
    assert settings.active_model_for_tier("bogus") == settings.active_model()


def test_upstream_api_key_for_tier():
    settings = make_settings()
    assert settings.upstream_api_key_for_tier("base") == "soniox-test-key"
    assert settings.upstream_api_key_for_tier("pro") == "test-key"
    assert settings.upstream_api_key_for_tier(None) == settings.upstream_api_key()


def test_factory_tier_base_selects_soniox_transcription():
    workflow = STTWorkflowFactory.get_workflow(make_settings(), "base")
    assert isinstance(workflow, SonioxWorkflow)
    assert workflow.tier == "base"
    client = STTWorkflowFactory.create_upstream_client(make_settings(), "base")
    assert isinstance(client, SonioxRealtimeClient)
    assert client.is_transcription is True


def test_factory_tier_pro_selects_openai_transcription():
    workflow = STTWorkflowFactory.get_workflow(make_settings(), "pro")
    assert isinstance(workflow, OpenAITranscriptionWorkflow)
    assert workflow.tier == "pro"
    client = STTWorkflowFactory.create_upstream_client(make_settings(), "pro")
    assert isinstance(client, OpenAIRealtimeClient)
    assert client.is_transcription is True
    assert client.model == "gpt-live-transcribe"


def test_factory_unknown_tier_falls_back_to_legacy_provider():
    assert isinstance(
        STTWorkflowFactory.get_workflow(make_settings(), "ultra"), OpenAIWorkflow
    )
    assert isinstance(
        STTWorkflowFactory.get_workflow(make_settings(), None), OpenAIWorkflow
    )
    assert isinstance(
        STTWorkflowFactory.get_workflow(soniox_settings(), None), SonioxWorkflow
    )


def test_build_upstream_client_tier():
    assert build_upstream_client(make_settings(), "base").model == "stt-rt-v5"
    assert (
        build_upstream_client(make_settings(), "pro").model
        == "gpt-live-transcribe"
    )


# --- soniox config builders (transcription-only) ---


def test_soniox_transcribe_config_has_no_translation():
    config = json.loads(build_soniox_transcribe_config("en"))
    assert config["model"] == "stt-rt-v5"
    assert "translation" not in config
    assert config["language_hints"] == ["en"]


def test_soniox_session_config_never_carries_translation_block():
    client = SonioxRealtimeClient(
        api_key="k", target_language="es", language_hints=["en", "es"]
    )
    config = json.loads(client._config_for_mode())
    assert "translation" not in config
    assert config["language_hints"] == ["en", "es"]


# --- soniox payload / event mapping (transcription-only) ---


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


def test_soniox_translation_tokens_are_ignored():
    # Transcription-only: translated tokens never map downstream, even
    # when the caller passes transcription_mode=False for compatibility.
    event = {
        "tokens": [
            {"text": "Hola", "is_final": True, "translation_status": "translation"},
        ],
        "finished": False,
    }
    assert map_soniox_event(event) is None
    assert map_soniox_event(event, transcription_mode=True) is None


def test_soniox_original_tokens_map_in_both_modes():
    event = {
        "tokens": [
            {"text": "Hello", "is_final": True, "translation_status": "original"}
        ],
        "finished": False,
    }
    for mode in (False, True):
        mapped = map_soniox_event(event, transcription_mode=mode)
        assert isinstance(mapped, TranscriptDeltaEvent)
        assert mapped.delta == "Hello"


def test_soniox_provisional_and_empty_responses_map_to_none():
    assert (
        map_soniox_event(
            {
                "tokens": [
                    {"text": "He", "is_final": False, "translation_status": "original"}
                ],
                "finished": False,
            }
        )
        is None
    )
    assert map_soniox_event({"tokens": [], "finished": False}) is None
    assert map_soniox_event({"tokens": [], "finished": True}) is None


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
async def test_relay_soniox_audio_and_transcription_deltas():
    client = FakeClient([json.dumps({"type": "audio.append", "audio": AUDIO_B64})])
    upstream = FakeUpstream(
        [
            json.dumps(
                {
                    "tokens": [
                        {
                            "text": "Hello.",
                            "is_final": True,
                            "translation_status": "original",
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
    assert [d["delta"] for d in deltas] == ["Hello."]


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


# --- tiered sessions (base/pro transcription, decoupled translation) ---


@pytest.mark.asyncio
async def test_relay_base_tier_transcribes_and_reports_soniox_model():
    client = FakeClient(
        [
            json.dumps(
                {
                    "type": "language-config",
                    "mode": "transcription",
                    "model_tier": "base",
                }
            ),
            json.dumps({"type": "audio.append", "audio": AUDIO_B64}),
        ]
    )
    upstream = FakeUpstream(
        [
            json.dumps(
                {
                    "tokens": [
                        {
                            "text": "Hello",
                            "is_final": True,
                            "translation_status": "original",
                        }
                    ],
                    "finished": False,
                }
            ),
        ]
    )
    settings = make_settings(OPENAI_TRANSLATION_SOURCE_LANGUAGE="")
    relay = RelaySession(client, upstream, settings, model_tier="base")
    assert relay._is_tiered is True
    assert relay._transcription_mode is True
    task = asyncio.create_task(relay.run())
    await asyncio.sleep(0.3)
    client.disconnect()
    upstream.disconnect()
    await asyncio.wait_for(task, timeout=5)
    assert upstream.sent == [AUDIO_B64]
    deltas = [json.loads(m) for m in client.sent if "transcript.delta" in m]
    assert deltas and deltas[0]["delta"] == "Hello"


@pytest.mark.asyncio
async def test_relay_pro_tier_transcribes_with_openai_and_same_output_shape():
    client = FakeClient(
        [
            json.dumps(
                {
                    "type": "language-config",
                    "mode": "transcription",
                    "model_tier": "pro",
                }
            ),
            json.dumps({"type": "audio.append", "audio": AUDIO_B64}),
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
            ),
        ]
    )
    settings = make_settings(OPENAI_TRANSLATION_SOURCE_LANGUAGE="")
    relay = RelaySession(client, upstream, settings, model_tier="pro")
    assert relay._is_tiered is True
    assert relay._transcription_mode is True
    task = asyncio.create_task(relay.run())
    await asyncio.sleep(0.3)
    client.disconnect()
    upstream.disconnect()
    await asyncio.wait_for(task, timeout=5)
    assert json.loads(upstream.sent[0]) == {
        "type": "input_audio_buffer.append",
        "audio": AUDIO_B64,
    }
    deltas = [json.loads(m) for m in client.sent if "transcript.delta" in m]
    # Same downstream shape as the base tier above.
    assert deltas and deltas[0]["delta"] == "Hello"
    assert deltas[0]["type"] == "transcript.delta"


@pytest.mark.asyncio
async def test_relay_tier_switches_engine_on_the_fly():
    old = FakeUpstream()
    new = FakeUpstream(
        incoming=[
            json.dumps(
                {
                    "type": "conversation.item.input_audio_transcription.delta",
                    "delta": "Hello",
                    "item_id": "item_1",
                }
            ),
        ]
    )

    def factory(tier=None):
        return new if tier == "pro" else old

    client = FakeClient(
        [
            json.dumps(
                {
                    "type": "language-config",
                    "mode": "transcription",
                    "model_tier": "base",
                }
            ),
            json.dumps({"type": "audio.append", "audio": AUDIO_B64}),
            json.dumps(
                {
                    "type": "language-config",
                    "mode": "transcription",
                    "model_tier": "pro",
                }
            ),
            json.dumps({"type": "audio.append", "audio": AUDIO_B64}),
        ]
    )
    settings = make_settings(OPENAI_TRANSLATION_SOURCE_LANGUAGE="")
    relay = RelaySession(
        client, old, settings, model_tier="base", upstream_factory=factory
    )
    task = asyncio.create_task(relay.run())
    await asyncio.sleep(0.5)
    client.disconnect()
    await asyncio.wait_for(task, timeout=5)
    # First audio went to the base (Soniox) engine as a raw frame...
    assert old.sent == [AUDIO_B64]
    assert old.closed is True
    # ...second audio went to the pro (OpenAI) engine as a JSON append.
    assert json.loads(new.sent[0]) == {
        "type": "input_audio_buffer.append",
        "audio": AUDIO_B64,
    }
    # One session throughout: no mismatch error, tier flipped over.
    assert relay._model_tier == "pro"
    assert relay._workflow.tier == "pro"
    assert not [m for m in client.sent if "model_tier_mismatch" in m]
    deltas = [json.loads(m) for m in client.sent if "transcript.delta" in m]
    assert deltas and deltas[0]["delta"] == "Hello"
    readies = [json.loads(m) for m in client.sent if "session.ready" in m]
    assert readies and readies[-1]["model"] == "gpt-live-transcribe"


@pytest.mark.asyncio
async def test_relay_tier_flip_without_factory_ends_session():
    # Direct-constructed sessions have no factory to build the new
    # engine with, so a tier flip still ends the session (the client
    # re-offers and heals it).
    client = FakeClient(
        [
            json.dumps(
                {
                    "type": "language-config",
                    "mode": "transcription",
                    "model_tier": "pro",
                }
            ),
        ]
    )
    upstream = FakeUpstream()
    settings = make_settings(OPENAI_TRANSLATION_SOURCE_LANGUAGE="")
    relay = RelaySession(client, upstream, settings, model_tier="base")
    task = asyncio.create_task(relay.run())
    await asyncio.wait_for(task, timeout=5)
    errors = [json.loads(m) for m in client.sent if '"model_tier_mismatch"' in m]
    assert errors and errors[0]["code"] == "model_tier_mismatch"
    # No swap happened: still on the old tier (run()'s teardown close of
    # the old socket at session end is expected and unrelated).
    assert relay._model_tier == "base"


@pytest.mark.asyncio
async def test_webrtc_tier_session_ready_reports_tier_model():
    for tier, expected in (
        ("base", "stt-rt-v5"),
        ("pro", "gpt-live-transcribe"),
    ):
        channel, upstream = FakeChannel(), FakeUpstream()
        settings = make_settings(OPENAI_TRANSLATION_SOURCE_LANGUAGE="")
        relay = WebRTCRelaySession(
            channel, upstream, settings, model_tier=tier
        )
        task = asyncio.create_task(relay.run())
        await asyncio.sleep(0.05)
        relay.on_channel_close()
        await asyncio.wait_for(task, timeout=5)
        assert json.loads(channel.sent[0]) == {
            "type": "session.ready",
            "model": expected,
        }


@pytest.mark.asyncio
async def test_webrtc_tier_switches_engine_on_the_fly():
    old = FakeUpstream()
    new = FakeUpstream(
        incoming=[
            json.dumps(
                {
                    "type": "conversation.item.input_audio_transcription.delta",
                    "delta": "Hello",
                    "item_id": "item_1",
                }
            ),
        ]
    )

    def factory(tier=None):
        return new if tier == "pro" else old

    channel = FakeChannel()
    settings = make_settings(OPENAI_TRANSLATION_SOURCE_LANGUAGE="")
    relay = WebRTCRelaySession(
        channel, old, settings, model_tier="base", upstream_factory=factory
    )
    task = asyncio.create_task(relay.run())
    await asyncio.sleep(0.05)
    relay.on_channel_message(
        json.dumps(
            {
                "type": "language-config",
                "mode": "transcription",
                "model_tier": "base",
            }
        )
    )
    await asyncio.sleep(0.05)
    relay.on_channel_message(PCM_SMALL)
    await asyncio.sleep(0.05)
    relay.on_channel_message(
        json.dumps(
            {
                "type": "language-config",
                "mode": "transcription",
                "model_tier": "pro",
            }
        )
    )
    await asyncio.sleep(0.05)
    relay.on_channel_message(PCM_SMALL)
    await asyncio.sleep(0.3)
    relay.on_channel_close()
    await asyncio.wait_for(task, timeout=5)
    # First chunk went to the base (Soniox) engine as a raw frame...
    assert old.sent == [base64.b64encode(PCM_SMALL).decode("ascii")]
    assert old.closed is True
    # ...second chunk went to the pro (OpenAI) engine as a JSON append.
    assert json.loads(new.sent[0]) == {
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(PCM_SMALL).decode("ascii"),
    }
    # One session throughout: tier flipped, fresh ready, same deltas.
    assert relay._model_tier == "pro"
    readies = [
        json.loads(m) for m in channel.sent if isinstance(m, str) and "session.ready" in m
    ]
    assert readies[0]["model"] == "stt-rt-v5"
    assert readies[-1]["model"] == "gpt-live-transcribe"
    deltas = [
        json.loads(m)
        for m in channel.sent
        if isinstance(m, str) and "transcript.delta" in m
    ]
    assert deltas and deltas[0]["delta"] == "Hello"
    assert not [
        m
        for m in channel.sent
        if isinstance(m, str) and "model_tier_mismatch" in m
    ]

