"""Regression tests for batched, send-once subtitle translation.

Guards against the old expanding-window behaviour: a DeepSeek call
every ~10 characters, re-sending and re-generating the whole segment
each time, with a fresh HTTP client per call.
"""

import asyncio

from tests.test_subtitle_factory import FakeHTTP


def test_fragments_make_no_api_call_until_segment_ready():
    from app.services.subtitle_factory import DeepseekFlashSubtitleTranslator

    http = FakeHTTP(script=[["OK"]])
    translator = DeepseekFlashSubtitleTranslator(api_key="k", http_client=http)

    async def _run():
        # 10+ chars without a sentence end: still buffered, zero calls.
        assert (
            await translator.translate(
                "0123456789", target_language="hi", store=False, item_id="x"
            )
            is None
        )
        assert (
            await translator.translate(
                " more text", target_language="hi", store=False, item_id="x"
            )
            is None
        )
        assert http.calls == []
        # Sentence end flushes the accumulated segment exactly once.
        result = await translator.translate(
            " done.", target_language="hi", store=False, item_id="x"
        )
        assert result == "NAMASTE"  # FakeHTTP's non-streaming reply
        assert len(http.calls) == 1
        assert "0123456789 more text done." in http.calls[0]

    asyncio.run(_run())


def test_each_ready_segment_sent_once_not_expanded():
    from app.services.subtitle_factory import DeepseekFlashSubtitleTranslator

    http = FakeHTTP(script=[["A"], ["B"]])
    translator = DeepseekFlashSubtitleTranslator(api_key="k", http_client=http)

    async def _collect(piece):
        pass

    async def _run():
        first = await translator.translate(
            "First part.", target_language="hi", store=False, item_id="s",
            on_chunk=_collect,
        )
        second = await translator.translate(
            "Second part.", target_language="hi", store=False, item_id="s",
            on_chunk=_collect,
        )
        assert (first, second) == ("A", "B")
        assert len(http.calls) == 2
        current = http.calls[1].split("Current segment (translate this only):")[-1]
        assert "Second part." in current
        assert "First part." not in current

    asyncio.run(_run())


def test_history_context_is_character_capped():
    from app.services.subtitle_factory import (
        MAX_HISTORY_CONTEXT_CHARS,
        DeepseekFlashSubtitleTranslator,
    )

    translator = DeepseekFlashSubtitleTranslator(api_key="k", http_client=FakeHTTP())
    for _ in range(3):
        translator._history.append(("s" * 400, "t" * 400))
    messages = translator._build_messages("Now.", "hi")
    assert len(messages[1]["content"]) <= MAX_HISTORY_CONTEXT_CHARS + 200
    assert "Now." in messages[1]["content"]


def test_owned_http_client_is_reused_and_closed(monkeypatch):
    import httpx

    from app.services.subtitle_factory import DeepseekFlashSubtitleTranslator

    created = []

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            self.closed = False
            created.append(self)

        async def post(self, url, *, headers, json, timeout):
            class Resp:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"choices": [{"message": {"content": "ok"}}]}

            return Resp()

        async def aclose(self):
            self.closed = True

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    translator = DeepseekFlashSubtitleTranslator(api_key="k")

    async def _run():
        await translator.translate("One.", target_language="hi")
        await translator.translate("Two.", target_language="hi")
        assert len(created) == 1
        await translator.aclose()
        assert created[0].closed is True
        assert translator._owned_client is None

    asyncio.run(_run())


def test_payload_caps_max_tokens():
    from app.services.subtitle_factory import DeepseekFlashSubtitleTranslator

    payloads = []

    class CapturingHTTP(FakeHTTP):
        async def post(self, url, *, headers, json, timeout):
            payloads.append(json)
            return await super().post(url, headers=headers, json=json, timeout=timeout)

    translator = DeepseekFlashSubtitleTranslator(
        api_key="k", http_client=CapturingHTTP()
    )

    async def _run():
        await translator.translate("Hello world.", target_language="hi")
        assert payloads[0]["max_tokens"] >= 64
        assert payloads[0]["stream"] is False
        assert payloads[0]["model"] == "deepseek-flash"

    asyncio.run(_run())


def test_injected_client_is_never_closed_by_translator():
    from app.services.subtitle_factory import DeepseekFlashSubtitleTranslator

    http = FakeHTTP()
    closed = []

    async def _mark_closed():
        closed.append(True)

    http.aclose = _mark_closed  # type: ignore[method-assign]
    translator = DeepseekFlashSubtitleTranslator(api_key="k", http_client=http)

    async def _run():
        await translator.translate("Hello.", target_language="hi")
        await translator.aclose()
        assert closed == []

    asyncio.run(_run())
