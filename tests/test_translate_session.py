"""Focused coverage for the gpt-realtime-translate wire contract.

Pins the upstream session configuration, the ``session.``-prefixed append
event, and the translated-delta mapping so a regression back to the
transcription protocol fails fast.
"""

import json

import pytest

from app.core.config import Settings
from app.schemas.events import (
    TranscriptDeltaEvent,
    UpstreamAudioAppendEvent,
    UpstreamAudioCommitEvent,
    UpstreamError,
    map_upstream_event,
    parse_client_event,
)
from app.services.openai_realtime import build_session_update


def make_settings(**overrides):
    base = {"OPENAI_API_KEY": "test-key"}
    base.update(overrides)
    return Settings(**base)


def test_settings_default_to_translate_model_and_endpoint():
    settings = make_settings()
    assert settings.openai_realtime_model == "gpt-realtime-translate"
    assert (
        settings.openai_realtime_url
        == "wss://api.openai.com/v1/realtime/translations"
           "?model=gpt-realtime-translate"
    )
    assert settings.translation_target_language == "en"
    assert (
        settings.translation_transcription_model == "gpt-realtime-whisper"
    )


def test_build_session_update_translate_shape():
    payload = json.loads(build_session_update("es"))
    assert payload == {
        "type": "session.update",
        "session": {
            "audio": {
                "input": {
                    "transcription": {"model": "gpt-realtime-whisper"},
                    "noise_reduction": {"type": "near_field"},
                },
                "output": {"language": "es"},
            }
        },
    }


def test_build_session_update_custom_models():
    payload = json.loads(
        build_session_update("fr", transcription_model="gpt-realtime-whisper")
    )
    session = payload["session"]
    assert session["audio"]["output"] == {"language": "fr"}
    assert session["audio"]["input"]["transcription"] == {
        "model": "gpt-realtime-whisper"
    }
    # The translation model lives in the URL, never in session.update.
    assert "model" not in session
    assert "type" not in session


def test_output_transcript_delta_maps_to_transcript_delta():
    mapped = map_upstream_event(
        {"type": "session.output_transcript.delta", "delta": "Bonjour"}
    )
    assert isinstance(mapped, TranscriptDeltaEvent)
    assert mapped.delta == "Bonjour"
    assert mapped.item_id == ""


@pytest.mark.parametrize(
    "event_type",
    [
        "session.created",
        "session.updated",
        "session.closed",
        "session.input_transcript.delta",
        "session.output_audio.delta",
    ],
)
def test_translate_side_channels_are_ignored(event_type):
    assert (
        map_upstream_event({"type": event_type, "delta": "x"}) is None
    )


def test_upstream_error_still_raises():
    with pytest.raises(UpstreamError):
        map_upstream_event({"type": "error", "error": {"code": "x"}})


def test_parse_client_event_accepts_translate_native_names():
    append = parse_client_event(
        {"type": "session.input_audio_buffer.append", "audio": "AQID"}
    )
    assert isinstance(append, UpstreamAudioAppendEvent)
    commit = parse_client_event({"type": "session.input_audio_buffer.commit"})
    assert isinstance(commit, UpstreamAudioCommitEvent)


def test_parse_client_event_keeps_legacy_native_names():
    append = parse_client_event(
        {"type": "input_audio_buffer.append", "audio": "AQID"}
    )
    assert isinstance(append, UpstreamAudioAppendEvent)
    commit = parse_client_event({"type": "input_audio_buffer.commit"})
    assert isinstance(commit, UpstreamAudioCommitEvent)


def test_parse_client_event_accepts_language_config():
    from app.schemas.events import LanguageConfigEvent

    event = parse_client_event(
        {"type": "language-config", "source_lang": "hi", "target_lang": "es"}
    )
    assert isinstance(event, LanguageConfigEvent)
    assert event.source_lang == "hi"
    assert event.target_lang == "es"


def test_parse_client_event_rejects_language_config_without_target():
    import pydantic

    with pytest.raises((ValueError, pydantic.ValidationError)):
        parse_client_event(
            {"type": "language-config", "source_lang": "hi", "target_lang": ""}
        )
    with pytest.raises((ValueError, pydantic.ValidationError)):
        parse_client_event(
            {"type": "language-config", "source_lang": "hi"}
        )
