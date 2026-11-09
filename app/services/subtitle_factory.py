"""Subtitle factory: text2text translation for live-transcribe transcripts.

The STT layer (``openai-live-transcribe`` via :mod:`app.services.relay`,
or Soniox transcription mode) produces source-language transcripts. When
``SUBTITLE_PROVIDER`` is set, each session attaches a text2text LLM that
translates those transcripts into the session target language before they
reach the extension.

Provider switch (``SUBTITLE_PROVIDER``):

- ``"none"`` (default) disables subtitle translation; transcripts pass
  through untouched.
- ``"deepseek"`` translates with a DeepSeek chat model (default
  ``deepseek-flash``, configurable via ``SUBTITLE_MODEL``) over the
  existing ``httpx`` dependency. The DeepSeek API is OpenAI-compatible
  (``POST {base_url}/chat/completions``), so no new package is needed.

Rolling window: :class:`RollingSubtitleTranslator` keeps the last
``SUBTITLE_CONTEXT_SEGMENTS`` (default 3) source/translation segment
pairs and sends them as context alongside the segment being translated,
so pronouns, tense, and topic stay consistent across segments. Only the
current segment is translated; history pairs are context only and are
never re-emitted.
"""

from __future__ import annotations

import json
import re
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

from app.core.logging import get_logger

log = get_logger(__name__)

PROVIDER_NONE = "none"
PROVIDER_DEEPSEEK = "deepseek"

DEEPSEEK_DEFAULT_MODEL = "deepseek-flash"
DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com"


class SubtitleError(Exception):
    """Safe subtitle failure. Never carries keys or transcript content."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class NullSubtitleTranslator:
    """Pass-through translator used when subtitles are disabled."""

    name = PROVIDER_NONE
    enabled = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @property
    def history(self) -> list[tuple[str, str]]:
        return []

    async def translate(
        self,
        text: str,
        *,
        target_language: str,
        source_language: str | None = None,
        store: bool = True,
        item_id: str = "",
        on_chunk: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        del target_language, source_language, store, item_id, on_chunk
        return text

    def clear_history(self) -> None:
        pass


class RollingSubtitleTranslator:
    """Base translator with a rolling window of previous segments.

    Subclasses implement :meth:`_complete` (one translated string for the
    current segment). :meth:`translate` builds the context prompt from
    history, calls :meth:`_complete`, then records the new pair unless
    ``store`` is False (used for provisional deltas, which must not
    pollute the window with overlapping fragments).
    """

    name = "rolling"
    enabled = True

    def __init__(self, context_segments: int = 3) -> None:
        self._context_segments = max(0, int(context_segments))
        self._buffer = SubtitleSegmentBuffer()
        self._history: deque[tuple[str, str]] = deque(
            maxlen=self._context_segments if self._context_segments > 0 else None
        )

    @property
    def history(self) -> list[tuple[str, str]]:
        return list(self._history)

    def clear_history(self) -> None:
        self._history.clear()
        self._buffer.clear()

    def _history_context(self) -> list[tuple[str, str]]:
        if self._context_segments <= 0:
            return []
        return list(self._history)[-self._context_segments :]

    def _build_messages(
        self,
        text: str,
        target_language: str,
        source_language: str | None = None,
    ) -> list[dict[str, str]]:
        source_label = source_language or "the speaker's language"
        system = (
            "You are a live subtitle translator. Translate the CURRENT "
            f"segment from {source_label} into {target_language}. "
            "PREVIOUS segments are context only (pronouns, tense, topic): "
            "do NOT re-translate or repeat them. Return ONLY the translated "
            "current segment, with no quotes, no explanations, and no extra "
            "sentences."
        )
        lines = []
        for index, (prev_source, prev_translated) in enumerate(
            self._history_context(), start=1
        ):
            lines.append(
                f"{index}. Source: {prev_source}\n   Translation: {prev_translated}"
            )
        if lines:
            user = (
                "Previous segments (context only, do NOT re-translate):\n"
                + "\n".join(lines)
                + f"\n\nCurrent segment (translate this only):\n{text}"
            )
        else:
            user = f"Translate this segment into {target_language}:\n{text}"
        print("RollingSubtitleTranslator._build_messages called with system:", system)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _build_messages_deque(
            self,
            text: str,
            target_language: str,
            source_language: str | None = None,
        ) -> list[dict[str, str]]:
            source_label = source_language or "the speaker's language"
            system = (
                "You are a live subtitle translator. Translate the CURRENT "
                f"segment from {source_label} into {target_language}. "
                "PREVIOUS segments are context only (pronouns, tense, topic): "
                "do NOT re-translate or repeat them. Return ONLY the translated "
                "current segment, with no quotes, no explanations, and no extra "
                "sentences."
            )
            lines = self._buffer.build_context()
            if lines:
                user = (
                    "Previous segments (context only, do NOT re-translate):\n"
                    + "\n".join(lines)
                    + f"\n\nCurrent segment (translate this only):\n{text}"
                )
            else:
                user = f"Translate this segment into {target_language}:\n{text}"
            print("RollingSubtitleTranslator._build_messages called with system:", system)
            return [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]

    async def _complete(self, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError

    async def translate(
        self,
        text: str,
        *,
        target_language: str,
        source_language: str | None = None,
        store: bool = True,
        item_id: str = "",
        on_chunk: Callable[[str], Awaitable[None]] | None = None,
    ) -> str | None:
        if not text or not text.strip():
            return text
        key = item_id or ""
        seq, context, ready = -1, "", ""
        fragment = self._buffer.add_fragment(text)
        print("RollingSubtitleTranslator.translate called with text:", fragment)
        if fragment:
            seq, context, text = fragment
        else:
            return text
        if not store:
            self._buffer.add(key, text)
            if not ready:
                return None
            text = ready
        else:
            text = self._buffer.add_fragment(text) or text
            if not text.strip():
                return text
        messages = self._build_messages_deque(text, target_language, source_language)
        try:
            if on_chunk is None:
                translated = (await self._complete(messages)).strip()
            else:
                translated = (
                    await self._complete_stream(messages, on_chunk)
                ).strip()
            self._buffer.set_translation(seq, translated)
        except Exception:
            self._buffer.put_back(key, text)
            raise
        if not translated:
            self._buffer.put_back(key, text)
            return text
        if self._context_segments > 0:
            self._history.append((text, translated))
        print("RollingSubtitleTranslator.translate returning translated:", translated)
        return translated


class DeepseekFlashSubtitleTranslator(RollingSubtitleTranslator):
    """DeepSeek chat subtitle translator with rolling context.

    Uses the OpenAI-compatible ``POST {base_url}/chat/completions`` API
    over ``httpx`` (already a project dependency). ``model`` defaults to
    ``deepseek-flash`` and is passed through verbatim so a rename only
    needs a config change. An ``http_client`` with a ``post`` method may
    be injected for tests; otherwise a fresh ``httpx.AsyncClient`` is
    used per call and closed afterwards.
    """

    name = PROVIDER_DEEPSEEK

    def __init__(
        self,
        api_key: str = "",
        model: str = DEEPSEEK_DEFAULT_MODEL,
        base_url: str = DEEPSEEK_DEFAULT_BASE_URL,
        timeout_seconds: float = 15.0,
        context_segments: int = 3,
        http_client: Any | None = None,
    ) -> None:
        super().__init__(context_segments=context_segments)
        self._api_key = api_key
        self._model = model
        self._base_url = (base_url or DEEPSEEK_DEFAULT_BASE_URL).rstrip("/")
        try:
            self._timeout = float(timeout_seconds)
        except (TypeError, ValueError):
            self._timeout = 15.0
        self._http_client = http_client

    def configure_buffer(self, settings: Any) -> None:
        self._buffer.configure(
            min_chars=getattr(settings, "subtitle_min_segment_chars", None),
            max_chars=getattr(settings, "subtitle_max_segment_chars", None),
            flush_seconds=getattr(settings, "subtitle_flush_seconds", None),
        )

    @property
    def model(self) -> str:
        return self._model

    async def _complete(self, messages: list[dict[str, str]]) -> str:
        print("DeepseekFlashSubtitleTranslator._complete called with messages:", messages)
        if not self._api_key:
            raise SubtitleError(
                code="not_configured",
                message="subtitle service is not configured",
            )
        url = f"{self._base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.0,
            "stream": False,
        }
        client = self._http_client
        owns_client = False
        if client is None:
            import httpx

            client = httpx.AsyncClient()
            owns_client = True
        try:
            try:
                response = await client.post(
                    url, headers=headers, json=payload, timeout=self._timeout
                )
            except Exception as exc:
                raise SubtitleError(
                    code="upstream_error",
                    message="subtitle service unavailable",
                ) from exc
            try:
                response.raise_for_status()
            except Exception as exc:
                raise SubtitleError(
                    code="upstream_error",
                    message="subtitle service unavailable",
                ) from exc
            try:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
            except Exception as exc:
                raise SubtitleError(
                    code="upstream_error",
                    message="subtitle service gave an invalid response",
                ) from exc
            if not isinstance(content, str) or not content.strip():
                raise SubtitleError(
                    code="upstream_error",
                    message="subtitle service gave an empty response",
                )
            return content.strip()
        finally:
            if owns_client:
                try:
                    await client.aclose()
                except Exception:
                    pass


    async def _complete_stream(
        self,
        messages: list[dict[str, str]],
        on_chunk: Callable[[str], Awaitable[None]],
    ) -> str:
        if not self._api_key:
            raise SubtitleError(
                code="not_configured",
                message="subtitle service is not configured",
            )
        url = f"{self._base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.0,
            "stream": True,
        }
        client = self._http_client
        owns_client = False
        if client is None:
            import httpx

            client = httpx.AsyncClient()
            owns_client = True
        try:
            try:
                async with client.stream(
                    "POST",
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self._timeout,
                ) as response:
                    try:
                        response.raise_for_status()
                    except Exception as exc:
                        raise SubtitleError(
                            code="upstream_error",
                            message="subtitle service unavailable",
                        ) from exc
                    full_parts: list[str] = []
                    async for line in response.aiter_lines():
                        stripped = line.strip()
                        if not stripped or not stripped.startswith("data:"):
                            continue
                        data = stripped[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            event = json.loads(data)
                            piece = event["choices"][0]["delta"].get("content", "")
                        except Exception as exc:
                            raise SubtitleError(
                                code="upstream_error",
                                message="subtitle service gave an invalid response",
                            ) from exc
                        if not piece:
                            continue
                        full_parts.append(piece)
                        await on_chunk(piece)
            except SubtitleError:
                raise
            except Exception as exc:
                raise SubtitleError(
                    code="upstream_error",
                    message="subtitle service unavailable",
                ) from exc
        finally:
            if owns_client:
                try:
                    await client.aclose()
                except Exception:
                    pass
        return "".join(full_parts)


class SubtitleSegmentBuffer:
    def __init__(
        self,
        min_chars: int = 20,
        max_chars: int = 100,
        flush_seconds: float = 0.2,
        max_items: int = 32,
    ) -> None:
        self._min_chars = max(0, int(min_chars))
        self._max_chars = max(0, int(max_chars))
        self._max_current = 5
        self._prev_budget = 50
        self._current = deque()
        self._previous = deque()
        try:
            self._flush_seconds = float(flush_seconds)
        except (TypeError, ValueError):
            self._flush_seconds = 1.0
        self._max_items = max(1, int(max_items))
        self._items: dict[str, list] = {}

    @property
    def min_chars(self) -> int:
        return self._min_chars

    def configure(
        self,
        min_chars: int | None = None,
        max_chars: int | None = None,
        flush_seconds: float | None = None,
    ) -> None:
        if min_chars is not None:
            self._min_chars = max(0, int(min_chars))
        if max_chars is not None:
            self._max_chars = max(0, int(max_chars))
        if flush_seconds is not None:
            try:
                value = float(flush_seconds)
            except (TypeError, ValueError):
                value = self._flush_seconds
            if value > 0:
                self._flush_seconds = value

    def add(self, item_id: str, fragment: str, *, now: float | None = None) -> None:
        if not fragment:
            return
        stamp = time.monotonic() if now is None else now
        entry = self._items.get(item_id)
        if entry is None:
            if len(self._items) >= self._max_items:
                oldest = next(iter(self._items))
                del self._items[oldest]
            entry = [fragment, stamp]
            self._items[item_id] = entry
            return
        current = entry[0]
        if current and not current[-1].isspace() and not fragment[0].isspace():
            current += " "
        entry[0] = current + fragment

    def add_fragment(self, text: str):
        """Returns (seq, context, chunk) when a chunk is ready, otherwise None."""
        if not text or not text.strip():
            return None
        self._current.append(text.strip())
        if self._ends_sentence(text) or len(self._current) >= self._max_current:
            return self.flush()
        return None

    def flush(self):
        """Call this directly on a VAD pause so trailing words don't stall."""
        if not self._current:
            return None

        chunk = " ".join(self._current)
        self._current.clear()

        seq = self._seq
        self._seq += 1

        # Build context BEFORE adding this chunk, so it only contains older chunks
        context = self._build_context()

        # Move the chunk into previous now (tgt=None until the translation returns)
        self._previous.append({"seq": seq, "src": chunk, "tgt": None})
        self._trim_previous()

        return seq, context, chunk

    def set_translation(self, seq: int, tgt: str):
        """Call when the translator responds, so later calls get the translated context."""
        for entry in self._previous:
            if entry["seq"] == seq:
                entry["tgt"] = tgt
                return
    def build_context(self) -> str:
        lines = []
        for e in self._previous:
            if e["tgt"] is not None:
                lines.append(f'{e["src"]} => {e["tgt"]}')
            else:
                lines.append(f'{e["src"]} => (translation pending)')
        return "\n".join(lines)
    def _trim_previous(self):
        # Drop whole chunks from the left, never partial ones
        while len(self._previous) > 1 and self._word_count() > self._prev_budget:
            self._previous.popleft()

    def take_ready(self, item_id: str, *, now: float | None = None) -> str:
        entry = self._items.get(item_id)
        if entry is None:
            return ""
        stamp = time.monotonic() if now is None else now
        text = self._normalize(entry[0])
        if not text:
            del self._items[item_id]
            return ""
        if self._ends_sentence(text):
            del self._items[item_id]
            return text
        if self._max_chars > 0 and len(text) >= self._max_chars:
            head, tail = self._split_head(text, self._max_chars)
            if tail:
                entry[0] = tail
            else:
                del self._items[item_id]
            return head
        if len(text) >= self._min_chars and (stamp - entry[1]) >= self._flush_seconds:
            del self._items[item_id]
            return text
        return ""

    def take_all(self, item_id: str, fallback: str = "") -> str:
        entry = self._items.pop(item_id, None)
        if entry is not None:
            text = self._normalize(entry[0])
            if text:
                return text
        return self._normalize(fallback)

    def put_back(self, item_id: str, text: str, *, now: float | None = None) -> None:
        if not text or not text.strip():
            return
        stamp = time.monotonic() if now is None else now
        entry = self._items.get(item_id)
        if entry is not None and entry[0].strip():
            entry[0] = text + " " + entry[0]
        else:
            self._items[item_id] = [text, stamp]

    def clear(self) -> None:
        self._items.clear()

    @staticmethod
    def _normalize(text: str) -> str:
        collapsed = re.sub(r"\s+", " ", text).strip()
        return re.sub(r"\s+([.,!?;:])", r"\1", collapsed)

    @staticmethod
    def _ends_sentence(text: str) -> bool:
        return text[-1:] in (".", "!", "?", "\u2026", "\u3002", "\uff01", "\uff1f")

    @staticmethod
    def _split_head(text: str, max_chars: int) -> tuple[str, str]:
        segment = text[:max_chars]
        for marker in (".", "!", "?", "\u2026"):
            index = segment.rfind(marker)
            if index > 0:
                return text[: index + 1], text[index + 1 :].strip()
        space = segment.rfind(" ")
        if space > 0:
            return text[:space], text[space + 1 :]
        return segment, text[max_chars:]


class SubtitleFactory:
    """Attaches the text2text subtitle LLM configured by ``settings``."""

    @staticmethod
    def get_translator(
        settings: Any,
    ) -> NullSubtitleTranslator | DeepseekFlashSubtitleTranslator:
        provider = getattr(settings, "subtitle_provider", PROVIDER_NONE)
        if provider == PROVIDER_DEEPSEEK:
            translator = DeepseekFlashSubtitleTranslator(
                api_key=getattr(settings, "deepseek_api_key", ""),
                model=getattr(settings, "subtitle_model", DEEPSEEK_DEFAULT_MODEL)
                or DEEPSEEK_DEFAULT_MODEL,
                base_url=getattr(
                    settings, "deepseek_base_url", DEEPSEEK_DEFAULT_BASE_URL
                )
                or DEEPSEEK_DEFAULT_BASE_URL,
                timeout_seconds=getattr(settings, "subtitle_timeout_seconds", 15.0),
                context_segments=getattr(settings, "subtitle_context_segments", 3),
            )
            translator.configure_buffer(settings)
            return translator
        return NullSubtitleTranslator()

    @staticmethod
    def provider_name(settings: Any) -> str:
        return SubtitleFactory.get_translator(settings).name
