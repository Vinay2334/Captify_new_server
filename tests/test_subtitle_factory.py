"""Subtitle factory and segment-buffer tests (mocked, no network)."""

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


def test_subtitle_buffer_defaults(monkeypatch):
    for var in (
        "SUBTITLE_MIN_SEGMENT_CHARS",
        "SUBTITLE_MAX_SEGMENT_CHARS",
        "SUBTITLE_FLUSH_SECONDS",
    ):
        monkeypatch.delenv(var, raising=False)
    settings = make_settings()
    assert settings.subtitle_min_segment_chars == 20
    assert settings.subtitle_max_segment_chars == 300
    assert settings.subtitle_flush_seconds == 2.0


def test_subtitle_buffer_config_rejected_when_invalid():
    import pydantic

    with pytest.raises((ValueError, pydantic.ValidationError)):
        make_settings(SUBTITLE_MIN_SEGMENT_CHARS=-1)
    with pytest.raises((ValueError, pydantic.ValidationError)):
        make_settings(SUBTITLE_MAX_SEGMENT_CHARS=-1)
    with pytest.raises((ValueError, pydantic.ValidationError)):
        make_settings(SUBTITLE_FLUSH_SECONDS=0)


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


def test_subtitle_factory_configures_buffer_from_settings():
    from app.services.subtitle_factory import SubtitleFactory

    settings = make_settings(
        SUBTITLE_PROVIDER="deepseek",
        DEEPSEEK_API_KEY="deepseek-test-key",
        SUBTITLE_MIN_SEGMENT_CHARS=7,
        SUBTITLE_MAX_SEGMENT_CHARS=80,
        SUBTITLE_FLUSH_SECONDS=1.5,
    )
    translator = SubtitleFactory.get_translator(settings)
    assert translator._buffer._min_chars == 7
    assert translator._buffer._max_chars == 80
    assert translator._buffer._flush_seconds == 1.5


def test_deepseek_default_model_is_flash():
    from app.services.subtitle_factory import DeepseekFlashSubtitleTranslator

    translator = DeepseekFlashSubtitleTranslator(api_key="k")
    assert translator.model == "deepseek-flash"


def test_null_translator_passes_text_through(monkeypatch):
    from app.services.subtitle_factory import SubtitleFactory

    monkeypatch.setenv("SUBTITLE_PROVIDER", "none")
    translator = SubtitleFactory.get_translator(make_settings())

    async def _run():
        assert await translator.translate("Hello", target_language="hi") == "Hello"
        assert await translator.translate("", target_language="hi") == ""
        assert (
            await translator.translate(
                "Hello", target_language="hi", store=False, item_id="x"
            )
            == "Hello"
        )

    asyncio.run(_run())


def test_null_translator_ignores_on_chunk(monkeypatch):
    from app.services.subtitle_factory import SubtitleFactory

    monkeypatch.setenv("SUBTITLE_PROVIDER", "none")
    translator = SubtitleFactory.get_translator(make_settings())
    seen = []

    async def _collect(piece):
        seen.append(piece)

    async def _run():
        result = await translator.translate(
            "Hello", target_language="hi", on_chunk=_collect
        )
        assert result == "Hello"
        assert seen == []

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
        first = await translator.translate("First segment.", target_language="hi")
        assert "First segment" not in first or first.startswith("TRANSLATED:")
        # Second call must carry the first segment as context.
        await translator.translate("Second segment.", target_language="hi")
        prompt_text = json.dumps(seen_messages[-1])
        assert "First segment." in prompt_text
        assert "Second segment." in prompt_text
        # Window of 2: after 3 segments the oldest drops out.
        await translator.translate("Third segment.", target_language="hi")
        await translator.translate("Fourth segment.", target_language="hi")
        prompt_text = json.dumps(seen_messages[-1])
        assert "First segment." not in prompt_text
        assert "Second segment." in prompt_text or "Third segment." in prompt_text

    asyncio.run(_run())


def test_rolling_window_delta_does_not_pollute_history():
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
        await translator.translate("Final one.", target_language="hi")
        assert len(translator.history) == 1
        await translator.translate("parti", target_language="hi", store=False)
        assert len(translator.history) == 1

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
        assert await translator.translate("   ", target_language="hi") == "   "

    asyncio.run(_run())


def test_buffer_accumulates_fragments_until_sentence_end():
    from app.services.subtitle_factory import SubtitleSegmentBuffer

    buf = SubtitleSegmentBuffer()
    assert buf.take_ready("a") == ""
    buf.add("a", "And")
    assert buf.take_ready("a") == ""
    buf.add("a", " ask")
    buf.add("a", " you")
    buf.add("a", " guys")
    assert buf.take_ready("a") == ""
    buf.add("a", ".")
    assert buf.take_ready("a") == "And ask you guys."
    assert buf.take_ready("a") == ""


def test_buffer_splits_long_segments_at_word_boundary():
    from app.services.subtitle_factory import SubtitleSegmentBuffer

    buf = SubtitleSegmentBuffer(min_chars=1000, max_chars=11, flush_seconds=60.0)
    buf.add("a", "hello world. bye", now=100.0)
    assert buf.take_ready("a", now=100.0) == "hello"
    buf.add("a", "!", now=100.0)
    assert buf.take_ready("a", now=100.0) == "world. bye!"


def test_buffer_flushes_on_age_past_minimum():
    from app.services.subtitle_factory import SubtitleSegmentBuffer

    buf = SubtitleSegmentBuffer(min_chars=5, max_chars=0, flush_seconds=2.0)
    buf.add("a", "hello world", now=100.0)
    assert buf.take_ready("a", now=101.0) == ""
    assert buf.take_ready("a", now=102.5) == "hello world"
    assert buf.take_ready("a", now=103.0) == ""
    # Below the minimum: age alone never flushes a stub.
    buf.add("b", "hi", now=200.0)
    assert buf.take_ready("b", now=205.0) == ""


def test_buffer_take_all_prefers_buffer_over_fallback():
    from app.services.subtitle_factory import SubtitleSegmentBuffer

    buf = SubtitleSegmentBuffer()
    assert buf.take_all("a", fallback="full turn") == "full turn"
    buf.add("a", "tail words")
    assert buf.take_all("a", fallback="full turn text") == "tail words"
    assert buf.take_all("a", fallback="full turn text") == "full turn text"


def test_buffer_put_back_restores_failed_flush():
    from app.services.subtitle_factory import DeepseekFlashSubtitleTranslator

    calls = []

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

            calls.append(json["messages"][-1]["content"])
            return Resp()

        async def aclose(self):
            pass

    translator = DeepseekFlashSubtitleTranslator(
        api_key="k", http_client=FlakyClient()
    )

    async def _run():
        with pytest.raises(Exception):
            await translator.translate(
                "Hello world.", target_language="hi", store=False, item_id="t1"
            )
        # The failed flush was restored, so the continuation still
        # carries the whole sentence to DeepSeek.
        result = await translator.translate(
            " Again.", target_language="hi", store=False, item_id="t1"
        )
        assert result == "ok"
        assert "Hello world. Again." in calls[-1]

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


def _delta(frag, item_id="item_1"):
    return json.dumps(
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "delta": frag,
            "item_id": item_id,
        }
    )


def _completed(transcript, item_id="item_1"):
    return json.dumps(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": transcript,
            "item_id": item_id,
        }
    )


def test_deepseek_streams_tokens_as_they_arrive():
    from app.services.subtitle_factory import DeepseekFlashSubtitleTranslator

    http = FakeHTTP(script=[["Hel", "", "lo", " world"]])
    translator = DeepseekFlashSubtitleTranslator(api_key="k", http_client=http)
    seen = []

    async def _collect(piece):
        seen.append(piece)

    async def _run():
        result = await translator.translate(
            "Hello world.", target_language="hi", on_chunk=_collect
        )
        assert seen == ["Hel", "lo", " world"]
        assert result == "Hello world"
        assert translator.history == [("Hello world.", "Hello world")]

    asyncio.run(_run())


def test_stream_skips_noise_and_empty_deltas():
    from app.services.subtitle_factory import DeepseekFlashSubtitleTranslator

    http = FakeHTTP(script=[["a", "", "b"]])
    translator = DeepseekFlashSubtitleTranslator(api_key="k", http_client=http)
    seen = []

    async def _collect(piece):
        seen.append(piece)

    async def _run():
        result = await translator.translate(
            "Sentence.", target_language="hi", store=False, item_id="t",
            on_chunk=_collect,
        )
        assert seen == ["a", "b"]
        assert result == "ab"

    asyncio.run(_run())


def test_stream_http_error_restores_buffer():
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
        with pytest.raises(Exception):
            await translator.translate(
                "Hello world.", target_language="hi", store=False, item_id="t1"
            )

    asyncio.run(_run())
    assert translator._buffer.take_all("t1") == "Hello world."


def test_stream_malformed_event_raises_and_restores():
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
            await translator.translate(
                "Hello world.", target_language="hi", on_chunk=_collect
            )
        assert seen == []

    asyncio.run(_run())


@pytest.mark.asyncio
async def test_relay_streams_sentence_pieces():
    from app.services.relay import RelaySession
    from app.services.subtitle_factory import DeepseekFlashSubtitleTranslator

    settings = make_settings(
        TRANSCRIBE_MODE=True,
        OPENAI_TRANSLATION_TARGET_LANGUAGE="hi",
        OPENAI_TRANSLATION_SOURCE_LANGUAGE="",
    )
    client = FakeClient()
    upstream = FakeUpstream(
        [
            _delta("And"),
            _delta(" ask"),
            _delta(" you"),
            _delta(" guys."),
            _completed("And ask you guys."),
        ]
    )
    http = FakeHTTP(script=[["Nam", "aste"], ["Poora", " anuvaad"]])
    translator = DeepseekFlashSubtitleTranslator(api_key="k", http_client=http)
    relay = RelaySession(client, upstream, settings, subtitle=translator)
    task = asyncio.create_task(relay.run())
    await asyncio.sleep(0.5)
    client.disconnect()
    upstream.disconnect()
    await asyncio.wait_for(task, timeout=5)
    deltas = [json.loads(m) for m in client.sent if "transcript.delta" in m]
    completed = [json.loads(m) for m in client.sent if "transcript.completed" in m]
    # Four word fragments collapsed into one streamed sentence, then turn end.
    # Each translated sentence starts on a new line so captions don't clump.
    assert len(http.calls) == 2
    assert "And ask you guys." in http.calls[0]
    assert [d["delta"] for d in deltas] == ["\nNam", "aste", "\nPoora", " anuvaad"]
    assert [c["transcript"] for c in completed] == ["\nPoora anuvaad"]


@pytest.mark.asyncio
async def test_relay_same_language_skips_buffering():
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
    upstream = FakeUpstream([_delta("Hello.")])
    translator = DeepseekFlashSubtitleTranslator(
        api_key="k", http_client=ExplodingHTTP()
    )
    relay = RelaySession(client, upstream, settings, subtitle=translator)
    task = asyncio.create_task(relay.run())
    await asyncio.sleep(0.3)
    client.disconnect()
    upstream.disconnect()
    await asyncio.wait_for(task, timeout=5)
    deltas = [json.loads(m) for m in client.sent if "transcript.delta" in m]
    assert [d["delta"] for d in deltas] == ["Hello."]


@pytest.mark.asyncio
async def test_webrtc_streams_sentence_pieces():
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
    upstream = FakeUpstream(
        [
            _delta("And"),
            _delta(" ask"),
            _delta(" you"),
            _delta(" guys."),
            _completed("And ask you guys."),
        ]
    )
    http = FakeHTTP(script=[["Nam", "aste"], ["Poora", " anuvaad"]])
    translator = DeepseekFlashSubtitleTranslator(api_key="k", http_client=http)
    relay = WebRTCRelaySession(channel, upstream, settings, subtitle=translator)
    task = asyncio.create_task(relay.run())
    await asyncio.sleep(0.5)
    relay.on_channel_close()
    upstream.disconnect()
    await asyncio.wait_for(task, timeout=5)
    deltas = [json.loads(m) for m in channel.sent if "transcript.delta" in m]
    completed = [json.loads(m) for m in channel.sent if "transcript.completed" in m]
    assert len(http.calls) == 2
    assert [d["delta"] for d in deltas] == ["\nNam", "aste", "\nPoora", " anuvaad"]
    assert [c["transcript"] for c in completed] == ["\nPoora anuvaad"]
