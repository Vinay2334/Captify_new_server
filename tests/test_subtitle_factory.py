"""Subtitle growing-window tests (mocked, no network)."""

import asyncio
import json

import pytest

from app.core.config import Settings


def make_settings(**overrides):
    base = {
        "OPENAI_API_KEY": "test-key",
        "SESSION_IDLE_TIMEOUT_SECONDS": 5.0,
        "MAX_AUDIO_BYTES_PER_SECOND": 96000,
        "MAX_WEBSOCKET_MESSAGE_BYTES": 65536,
    }
    base.update(overrides)
    return Settings(**base)


def test_subtitle_defaults_to_disabled(monkeypatch):
    from app.core.config import Settings as SettingsClass
    from app.services.subtitle_factory import SubtitleFactory

    assert SettingsClass.model_fields["subtitle_provider"].default == "none"
    monkeypatch.setenv("SUBTITLE_PROVIDER", "none")
    settings = make_settings()
    assert settings.subtitle_provider == "none"
    assert SubtitleFactory.provider_name(settings) == "none"
    translator = SubtitleFactory.get_translator(settings)
    assert translator.enabled is False


def test_subtitle_window_cap_default_and_validation(monkeypatch):
    # The old minimum-size and flush-age knobs are gone: extra env input
    # is ignored, and only the window cap remains.
    for var in (
        "SUBTITLE_MIN_SEGMENT_CHARS",
        "SUBTITLE_FLUSH_SECONDS",
    ):
        monkeypatch.delenv(var, raising=False)
    settings = make_settings()
    assert settings.subtitle_max_segment_chars == 300

    import pydantic

    with pytest.raises((ValueError, pydantic.ValidationError)):
        make_settings(SUBTITLE_MAX_SEGMENT_CHARS=-1)
    # Removed knobs never fail validation; they are simply ignored.
    make_settings(SUBTITLE_MIN_SEGMENT_CHARS=-1, SUBTITLE_FLUSH_SECONDS=0)


def test_subtitle_invalid_provider_rejected():
    import pydantic

    with pytest.raises((ValueError, pydantic.ValidationError)):
        make_settings(SUBTITLE_PROVIDER="google")


def test_subtitle_factory_selects_deepseek():
    from app.services.subtitle_factory import (
        DeepseekFlashSubtitleTranslator,
        SubtitleFactory,
    )

    settings = make_settings(
        SUBTITLE_PROVIDER="deepseek",
        DEEPSEEK_API_KEY="deepseek-test-key",
    )
    translator = SubtitleFactory.get_translator(settings)
    assert isinstance(translator, DeepseekFlashSubtitleTranslator)
    assert translator.enabled is True
    assert SubtitleFactory.provider_name(settings) == "deepseek"


def test_subtitle_factory_passes_window_cap_and_context():
    from app.services.subtitle_factory import SubtitleFactory

    settings = make_settings(
        SUBTITLE_PROVIDER="deepseek",
        DEEPSEEK_API_KEY="deepseek-test-key",
        SUBTITLE_MAX_SEGMENT_CHARS=80,
        SUBTITLE_CONTEXT_SEGMENTS=5,
    )
    translator = SubtitleFactory.get_translator(settings)
    assert translator.max_chars == 80
    assert translator._context_segments == 5


def test_deepseek_default_model_is_flash():
    from app.services.subtitle_factory import DeepseekFlashSubtitleTranslator

    translator = DeepseekFlashSubtitleTranslator(api_key="k")
    assert translator.model == "deepseek-flash"


def test_null_translator_passes_snapshot_through(monkeypatch):
    from app.services.subtitle_factory import SubtitleFactory

    monkeypatch.setenv("SUBTITLE_PROVIDER", "none")
    translator = SubtitleFactory.get_translator(make_settings())

    async def _run():
        assert (
            await translator.translate_snapshot("Hello", target_language="hi")
            == "Hello"
        )
        assert await translator.translate_snapshot("", target_language="hi") == ""
        assert translator.push_fragment("Hello") == "Hello"
        assert translator.should_finalize("Hello.") is False
        assert translator.take_final("full turn") == "full turn"

    asyncio.run(_run())


def test_rolling_window_includes_previous_segments():
    from app.services.subtitle_factory import DeepseekFlashSubtitleTranslator

    seen_messages = []

    class FakeClient:
        async def post(self, url, *, headers, json, timeout):
            seen_messages.append(json["messages"])

            class Resp:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {
                        "choices": [
                            {"message": {"content": "TRANSLATED:" + json["messages"][-1]["content"][-20:]}}
                        ]
                    }

            return Resp()

        async def aclose(self):
            pass

    translator = DeepseekFlashSubtitleTranslator(
        api_key="k",
        model="deepseek-flash",
        context_segments=2,
        http_client=FakeClient(),
    )

    async def _run():
        first = await translator.translate_snapshot(
            "First segment.", target_language="hi"
        )
        assert first.startswith("TRANSLATED:")
        translator.commit("First segment.", first)
        # Second call must carry the first segment as context.
        await translator.translate_snapshot("Second segment.", target_language="hi")
        translator.commit("Second segment.", "second-t")
        prompt_text = json.dumps(seen_messages[-1])
        assert "First segment." in prompt_text
        assert "Second segment." in prompt_text
        # Window of 2: after 3 committed segments the oldest drops out.
        third = await translator.translate_snapshot(
            "Third segment.", target_language="hi"
        )
        translator.commit("Third segment.", third)
        fourth = await translator.translate_snapshot(
            "Fourth segment.", target_language="hi"
        )
        translator.commit("Fourth segment.", fourth)
        prompt_text = json.dumps(seen_messages[-1])
        assert "First segment." not in prompt_text
        assert "Third segment." in prompt_text or "Second segment." in prompt_text

    asyncio.run(_run())


def test_snapshot_never_touches_history_until_commit():
    from app.services.subtitle_factory import DeepseekFlashSubtitleTranslator

    calls = []

    class FakeClient:
        async def post(self, url, *, headers, json, timeout):
            calls.append(json["messages"][-1]["content"])

            class Resp:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"choices": [{"message": {"content": "t"}}]}

            return Resp()

        async def aclose(self):
            pass

    translator = DeepseekFlashSubtitleTranslator(
        api_key="k", context_segments=5, http_client=FakeClient()
    )

    async def _run():
        # Provisional re-translations (one per word) never pollute history.
        await translator.translate_snapshot("parti", target_language="hi")
        await translator.translate_snapshot("partial window", target_language="hi")
        assert translator.history == []
        # Only the segment-final commit records history.
        translator.commit("partial window", "t")
        assert translator.history == [("partial window", "t")]

    asyncio.run(_run())


def test_deepseek_empty_text_never_calls_http():
    from app.services.subtitle_factory import DeepseekFlashSubtitleTranslator

    class ExplodingClient:
        async def post(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("must not call HTTP for empty text")

    translator = DeepseekFlashSubtitleTranslator(
        api_key="k", http_client=ExplodingClient()
    )

    async def _run():
        assert await translator.translate_snapshot("   ", target_language="hi") == "   "

    asyncio.run(_run())


def test_window_grows_word_by_word_until_sentence_end():
    from app.services.subtitle_factory import DeepseekFlashSubtitleTranslator

    translator = DeepseekFlashSubtitleTranslator(api_key="k")
    assert translator.seg_id == "seg-0"
    assert translator.push_fragment("And") == "And"
    assert translator.push_fragment(" ask") == "And ask"
    assert translator.push_fragment(" you") == "And ask you"
    assert translator.should_finalize("And ask you") is False
    assert translator.push_fragment(" guys.") == "And ask you guys."
    assert translator.should_finalize(translator.current_snapshot()) is True
    head, tail = translator.split_for_finalize(translator.current_snapshot())
    assert (head, tail) == ("And ask you guys.", "")
    translator.take_window()
    translator.commit(head, "translation")
    assert translator.seg_id == "seg-1"
    assert translator.history == [("And ask you guys.", "translation")]


def test_window_cap_splits_at_word_boundary():
    from app.services.subtitle_factory import DeepseekFlashSubtitleTranslator

    translator = DeepseekFlashSubtitleTranslator(api_key="k", max_chars=11)
    snapshot = translator.push_fragment("hello world. bye")
    assert snapshot == "hello world. bye"
    assert translator.should_finalize(snapshot) is True
    head, tail = translator.split_for_finalize(snapshot)
    assert head == "hello"
    assert tail == "world. bye"


def test_take_final_prefers_completion_tail():
    from app.services.subtitle_factory import DeepseekFlashSubtitleTranslator

    translator = DeepseekFlashSubtitleTranslator(api_key="k")
    # Trailing punctuation often arrives only in the completion.
    translator.push_fragment("uno")
    translator.push_fragment(" dos")
    assert translator.take_final("uno dos.") == "uno dos."
    # An empty window falls back to the completion itself.
    assert translator.take_final("full turn") == "full turn"
    # A window the completion does not extend wins as-is.
    translator.push_fragment("stale words here")
    assert translator.take_final("unrelated") == "stale words here"


def test_failed_snapshot_keeps_window_and_history():
    from app.services.subtitle_factory import DeepseekFlashSubtitleTranslator

    class FlakyClient:
        def __init__(self):
            self.fail_next = True

        async def post(self, url, *, headers, json, timeout):
            if self.fail_next:
                self.fail_next = False
                raise ConnectionError("boom")

            class Resp:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"choices": [{"message": {"content": "ok"}}]}

            return Resp()

        async def aclose(self):
            pass

    translator = DeepseekFlashSubtitleTranslator(
        api_key="k", http_client=FlakyClient()
    )

    async def _run():
        translator.push_fragment("Hello world.")
        with pytest.raises(Exception):
            await translator.translate_snapshot(
                translator.current_snapshot(), target_language="hi"
            )
        # The stateless snapshot call never touches window or history.
        assert translator.current_snapshot() == "Hello world."
        assert translator.history == []
        assert (
            await translator.translate_snapshot(
                translator.current_snapshot(), target_language="hi"
            )
            == "ok"
        )

    asyncio.run(_run())


def test_deepseek_streams_tokens_as_they_arrive():
    from app.services.subtitle_factory import DeepseekFlashSubtitleTranslator

    http = FakeHTTP(script=[["Hel", "", "lo", " world"]])
    translator = DeepseekFlashSubtitleTranslator(api_key="k", http_client=http)
    seen = []

    async def _collect(piece):
        seen.append(piece)

    async def _run():
        result = await translator.translate_snapshot(
            "Hello world.", target_language="hi", on_chunk=_collect
        )
        assert seen == ["Hel", "lo", " world"]
        assert result == "Hello world"
        # Stateless: streaming alone never records history.
        assert translator.history == []

    asyncio.run(_run())


def test_stream_skips_noise_and_empty_deltas():
    from app.services.subtitle_factory import DeepseekFlashSubtitleTranslator

    http = FakeHTTP(script=[["a", "", "b"]])
    translator = DeepseekFlashSubtitleTranslator(api_key="k", http_client=http)
    seen = []

    async def _collect(piece):
        seen.append(piece)

    async def _run():
        result = await translator.translate_snapshot(
            "Sentence.", target_language="hi", on_chunk=_collect
        )
        assert seen == ["a", "b"]
        assert result == "ab"

    asyncio.run(_run())


def test_stream_http_error_leaves_window_untouched():
    from app.services.subtitle_factory import DeepseekFlashSubtitleTranslator

    class FailingStream:
        def raise_for_status(self):
            raise RuntimeError("500")

        async def aiter_lines(self):  # pragma: no cover
            yield ""

    class FailingContext:
        async def __aenter__(self):
            return FailingStream()

        async def __aexit__(self, *args):
            return False

    class FailingHTTP:
        def stream(self, *args, **kwargs):
            return FailingContext()

        async def aclose(self):
            pass

    translator = DeepseekFlashSubtitleTranslator(
        api_key="k", http_client=FailingHTTP()
    )

    async def _run():
        translator.push_fragment("Hello world.")
        with pytest.raises(Exception):
            await translator.translate_snapshot(
                translator.current_snapshot(), target_language="hi"
            )

    asyncio.run(_run())
    assert translator.current_snapshot() == "Hello world."
    assert translator.history == []


def test_stream_malformed_event_raises():
    from app.services.subtitle_factory import DeepseekFlashSubtitleTranslator

    class BadStream:
        def raise_for_status(self):
            pass

        async def aiter_lines(self):
            yield "data: {not json"

    class BadContext:
        async def __aenter__(self):
            return BadStream()

        async def __aexit__(self, *args):
            return False

    class BadHTTP:
        def stream(self, *args, **kwargs):
            return BadContext()

        async def aclose(self):
            pass

    translator = DeepseekFlashSubtitleTranslator(api_key="k", http_client=BadHTTP())

    async def _run():
        seen = []

        async def _collect(piece):
            seen.append(piece)

        with pytest.raises(Exception):
            await translator.translate_snapshot(
                "Hello world.", target_language="hi", on_chunk=_collect
            )
        assert seen == []

    asyncio.run(_run())


class FakeClient:
    def __init__(self, incoming=None):
        self.incoming = asyncio.Queue()
        for item in incoming or []:
            self.incoming.put_nowait(item)
        self.sent = []

    async def send_text(self, message):
        self.sent.append(message)

    async def receive_text(self):
        item = await self.incoming.get()
        if item is None:
            raise RuntimeError("client disconnected")
        return item

    async def close(self, code=1000, reason=""):
        pass

    def disconnect(self):
        self.incoming.put_nowait(None)


class FakeUpstream:
    def __init__(self, incoming=None):
        self.incoming = asyncio.Queue()
        for item in incoming or []:
            self.incoming.put_nowait(item)
        self.sent = []

    async def send_text(self, payload):
        self.sent.append(payload)

    async def recv_text(self):
        item = await self.incoming.get()
        if item is None:
            raise RuntimeError("upstream disconnected")
        return item

    async def close(self):
        pass

    def disconnect(self):
        self.incoming.put_nowait(None)


class _FakeStreamResponse:
    def __init__(self, pieces):
        self._pieces = pieces

    def raise_for_status(self):
        pass

    async def aiter_lines(self):
        yield ": keep-alive"
        yield ""
        for piece in self._pieces:
            yield "data: " + json.dumps(
                {"choices": [{"delta": {"content": piece}}]}
            )
        yield "data: [DONE]"


class _FakeStreamContext:
    def __init__(self, pieces):
        self._pieces = pieces

    async def __aenter__(self):
        return _FakeStreamResponse(self._pieces)

    async def __aexit__(self, *args):
        return False


class FakeHTTP:
    """DeepSeek stand-in: scripted token pieces per call, SSE or plain."""

    def __init__(self, script=()):
        self.script = [list(pieces) for pieces in script]
        self.calls = []

    def _pieces_for(self, index):
        if index < len(self.script):
            return self.script[index]
        return ["NAMASTE"]

    async def post(self, url, *, headers, json, timeout):
        self.calls.append(json["messages"][-1]["content"])

        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": "NAMASTE"}}]}

        return Resp()

    def stream(self, method, url, *, headers, json, timeout):
        self.calls.append(json["messages"][-1]["content"])
        return _FakeStreamContext(self._pieces_for(len(self.calls) - 1))

    async def aclose(self):
        pass


def _delta_event(frag, item_id="item_1"):
    from app.schemas.events import TranscriptDeltaEvent

    return TranscriptDeltaEvent(delta=frag, item_id=item_id)


def _completed_event(transcript, item_id="item_1"):
    from app.schemas.events import TranscriptCompletedEvent

    return TranscriptCompletedEvent(transcript=transcript, item_id=item_id)


@pytest.mark.asyncio
async def test_relay_replaces_provisional_then_finalizes_turn():
    from app.services.relay import RelaySession
    from app.services.subtitle_factory import DeepseekFlashSubtitleTranslator

    settings = make_settings(
        TRANSCRIBE_MODE=True,
        OPENAI_TRANSLATION_TARGET_LANGUAGE="hi",
        OPENAI_TRANSLATION_SOURCE_LANGUAGE="",
    )
    client = FakeClient()
    upstream = FakeUpstream()
    http = FakeHTTP(script=[["uno"], ["uno dos"], ["UNO DOS"]])
    translator = DeepseekFlashSubtitleTranslator(api_key="k", http_client=http)
    relay = RelaySession(client, upstream, settings, subtitle=translator)

    # Word 1 grows the window and streams its first translation.
    assert await relay._maybe_translate_subtitle(_delta_event("uno")) is None
    await relay._join_subtitle_task()
    # Word 2 re-translates the whole window as a new revision.
    assert await relay._maybe_translate_subtitle(_delta_event(" dos")) is None
    await relay._join_subtitle_task()

    deltas = [json.loads(m) for m in client.sent if '"transcript.delta"' in m]
    assert [d["seg_id"] for d in deltas] == ["seg-0", "seg-0"]
    assert [d["rev"] for d in deltas] == [1, 2]
    assert [d["delta"] for d in deltas] == ["\nuno", "\nuno dos"]
    # Provisional snapshots never reach history.
    assert translator.history == []
    assert len(http.calls) == 2
    assert "uno dos" in http.calls[1]

    # Turn end finalizes the window under the same seg_id.
    assert (
        await relay._maybe_translate_subtitle(_completed_event("uno dos."))
        is None
    )
    messages = [json.loads(m) for m in client.sent]
    assert messages[-1]["type"] == "transcript.completed"
    assert messages[-1]["seg_id"] == "seg-0"
    assert messages[-1]["transcript"] == "\nUNO DOS"
    assert translator.history == [("uno dos.", "UNO DOS")]
    assert translator._seg_index == 1


@pytest.mark.asyncio
async def test_relay_finalizes_sentence_mid_turn_and_starts_next_window():
    from app.services.relay import RelaySession
    from app.services.subtitle_factory import DeepseekFlashSubtitleTranslator

    settings = make_settings(
        TRANSCRIBE_MODE=True,
        OPENAI_TRANSLATION_TARGET_LANGUAGE="hi",
        OPENAI_TRANSLATION_SOURCE_LANGUAGE="",
    )
    client = FakeClient()
    upstream = FakeUpstream()
    http = FakeHTTP(script=[["EK"], ["DO"]])
    translator = DeepseekFlashSubtitleTranslator(api_key="k", http_client=http)
    relay = RelaySession(client, upstream, settings, subtitle=translator)

    assert await relay._maybe_translate_subtitle(_delta_event("Hello.")) is None
    assert await relay._maybe_translate_subtitle(_delta_event("Again.")) is None

    completed = [json.loads(m) for m in client.sent if "transcript.completed" in m]
    assert [c["seg_id"] for c in completed] == ["seg-0", "seg-1"]
    assert [c["transcript"] for c in completed] == ["\nEK", "\nDO"]
    assert [source for source, _ in translator.history] == ["Hello.", "Again."]


@pytest.mark.asyncio
async def test_relay_splits_window_at_char_cap():
    from app.services.relay import RelaySession
    from app.services.subtitle_factory import DeepseekFlashSubtitleTranslator

    settings = make_settings(
        TRANSCRIBE_MODE=True,
        OPENAI_TRANSLATION_TARGET_LANGUAGE="hi",
        OPENAI_TRANSLATION_SOURCE_LANGUAGE="",
        SUBTITLE_MAX_SEGMENT_CHARS=11,
    )
    client = FakeClient()
    upstream = FakeUpstream()
    http = FakeHTTP(script=[["H1"], ["HEAD"], ["TAIL"]])
    translator = DeepseekFlashSubtitleTranslator(
        api_key="k", max_chars=11, http_client=http
    )
    relay = RelaySession(client, upstream, settings, subtitle=translator)

    assert await relay._maybe_translate_subtitle(_delta_event("hello")) is None
    await relay._join_subtitle_task()
    assert await relay._maybe_translate_subtitle(_delta_event(" world. bye")) is None
    await relay._join_subtitle_task()

    completed = [json.loads(m) for m in client.sent if "transcript.completed" in m]
    assert len(completed) == 1
    assert completed[0]["seg_id"] == "seg-0"
    assert completed[0]["transcript"] == "\nHEAD"
    # The tail keeps growing under the next window.
    assert translator.current_snapshot() == "world. bye"
    assert translator.history == [("hello", "HEAD")]


@pytest.mark.asyncio
async def test_relay_same_language_skips_translation():
    from app.services.relay import RelaySession
    from app.services.subtitle_factory import DeepseekFlashSubtitleTranslator

    class ExplodingHTTP:
        def stream(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("same-language text must not reach DeepSeek")

        async def post(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("same-language text must not reach DeepSeek")

    settings = make_settings(
        OPENAI_TRANSLATION_SOURCE_LANGUAGE="hi",
        OPENAI_TRANSLATION_TARGET_LANGUAGE="hi",
    )
    client = FakeClient()
    upstream = FakeUpstream()
    translator = DeepseekFlashSubtitleTranslator(
        api_key="k", http_client=ExplodingHTTP()
    )
    relay = RelaySession(client, upstream, settings, subtitle=translator)
    mapped = _delta_event("Hello.")
    assert await relay._maybe_translate_subtitle(mapped) is mapped


@pytest.mark.asyncio
async def test_relay_subtitle_failure_falls_back_to_source():
    from app.services.relay import RelaySession
    from app.services.subtitle_factory import DeepseekFlashSubtitleTranslator

    class FailingHTTP:
        def stream(self, *args, **kwargs):
            raise ConnectionError("boom")

        async def post(self, *args, **kwargs):
            raise ConnectionError("boom")

        async def aclose(self):
            pass

    settings = make_settings(
        TRANSCRIBE_MODE=True,
        OPENAI_TRANSLATION_TARGET_LANGUAGE="hi",
        OPENAI_TRANSLATION_SOURCE_LANGUAGE="",
    )
    client = FakeClient()
    upstream = FakeUpstream()
    translator = DeepseekFlashSubtitleTranslator(
        api_key="k", http_client=FailingHTTP()
    )
    relay = RelaySession(client, upstream, settings, subtitle=translator)
    mapped = _completed_event("Hello world.")
    # Turn-end failure passes the source turn through untouched.
    assert await relay._maybe_translate_subtitle(mapped) is mapped
    assert client.sent == []
    assert translator.history == []


@pytest.mark.asyncio
async def test_webrtc_replaces_provisional_then_finalizes_turn():
    from app.services.subtitle_factory import DeepseekFlashSubtitleTranslator
    from app.services.webrtc_session import WebRTCRelaySession

    class FakeChannel:
        def __init__(self):
            self.sent = []
            self.readyState = "open"

        def send(self, message):
            self.sent.append(message)

    settings = make_settings(
        TRANSCRIBE_MODE=True,
        OPENAI_TRANSLATION_TARGET_LANGUAGE="hi",
        OPENAI_TRANSLATION_SOURCE_LANGUAGE="",
    )
    channel = FakeChannel()
    upstream = FakeUpstream()
    http = FakeHTTP(script=[["uno"], ["UNO DOS"]])
    translator = DeepseekFlashSubtitleTranslator(api_key="k", http_client=http)
    relay = WebRTCRelaySession(channel, upstream, settings, subtitle=translator)

    assert await relay._maybe_translate_subtitle(_delta_event("uno dos")) is None
    await relay._join_subtitle_task()
    assert (
        await relay._maybe_translate_subtitle(_completed_event("uno dos."))
        is None
    )

    messages = [json.loads(m) for m in channel.sent]
    kinds = [m["type"] for m in messages]
    assert kinds == ["transcript.delta", "transcript.delta", "transcript.completed"]
    assert messages[0]["seg_id"] == "seg-0"
    assert messages[-1]["seg_id"] == "seg-0"
    assert messages[-1]["transcript"] == "\nUNO DOS"
    assert translator.history == [("uno dos.", "UNO DOS")]
