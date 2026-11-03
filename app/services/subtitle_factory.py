"""Subtitle factory: text2text translation for live-transcribe transcripts.

The STT layer (``openai-live-transcribe`` via :mod:`app.services.relay`,
or Soniox transcription mode) produces source-language transcripts word
by word. When ``SUBTITLE_PROVIDER`` is set, each session attaches a
text2text LLM that translates those transcripts into the session target
language before they reach the extension.

Provider switch (``SUBTITLE_PROVIDER``):

- ``"none"`` (default) disables subtitle translation; transcripts pass
  through untouched.
- ``"deepseek"`` translates with a DeepSeek chat model (default
  ``deepseek-flash``, configurable via ``SUBTITLE_MODEL``) over the
  existing ``httpx`` dependency. The DeepSeek API is OpenAI-compatible
  (``POST {base_url}/chat/completions``), so no new package is needed.

Growing window: the relay accumulates transcriptor words into a source
window and re-translates the whole window on every word, up to
``SUBTITLE_MAX_SEGMENT_CHARS``. Each re-translation is emitted as a
``transcript.delta`` carrying the full translated snapshot plus a
``seg_id``/``rev`` pair; the frontend replaces the line for ``seg_id``
whenever ``rev`` bumps. When the window hits a sentence end, the char
cap, or the turn end, the backend translates once more and emits a
``transcript.completed`` with the same ``seg_id`` as the segment-final
identifier; the next window then starts under the next ``seg_id``.

Rolling history: :class:`RollingSubtitleTranslator` keeps the last
``SUBTITLE_CONTEXT_SEGMENTS`` (default 3) finalized source/translation
segment pairs and sends them as context alongside the window being
translated, so pronouns, tense, and topic stay consistent across
segments. Provisional snapshots never enter history; only finalized
segments do.
"""

from __future__ import annotations

import json
import re
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

    async def translate_snapshot(
        self,
        text: str,
        *,
        target_language: str,
        source_language: str | None = None,
        item_id: str = "",
        on_chunk: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        del target_language, source_language, item_id, on_chunk
        return text

    @property
    def seg_id(self) -> str:
        return "seg-0"

    def push_fragment(self, fragment: str) -> str:
        return fragment or ""

    def current_snapshot(self) -> str:
        return ""

    def take_window(self) -> str:
        return ""

    def take_final(self, fallback: str) -> str:
        return fallback

    def set_window(self, text: str) -> None:
        del text

    def should_finalize(self, snapshot: str) -> bool:
        del snapshot
        return False

    def split_for_finalize(self, snapshot: str) -> tuple[str, str]:
        return snapshot, ""

    def commit(self, source: str, translated: str) -> None:
        del source, translated

    def clear_history(self) -> None:
        pass


class RollingSubtitleTranslator:
    """Base translator with a growing source window and rolling history.

    The relay accumulates transcriptor words via :meth:`push_fragment`
    and re-translates the whole window snapshot on every word with
    :meth:`translate_snapshot` (stateless: history is only read, never
    written). When the window is finalized (sentence end, char cap, or
    turn end) the relay calls :meth:`commit`, which records the
    source/translation pair in history and opens the next ``seg_id``.
    Subclasses implement :meth:`_complete` / :meth:`_complete_stream`.
    """

    name = "rolling"
    enabled = True

    def __init__(self, context_segments: int = 3, max_chars: int = 1) -> None:
        self._context_segments = max(0, int(context_segments))
        self._max_chars = max(0, int(max_chars))
        self._history: deque[tuple[str, str]] = deque(
            maxlen=self._context_segments if self._context_segments > 0 else None
        )
        self._window = ""
        self._seg_index = 0

    @property
    def history(self) -> list[tuple[str, str]]:
        return list(self._history)

    @property
    def seg_id(self) -> str:
        return f"seg-{self._seg_index}"

    @property
    def max_chars(self) -> int:
        return self._max_chars

    def clear_history(self) -> None:
        self._history.clear()
        self._window = ""

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
        log.debug("subtitle prompt built chars=%d", len(text))
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    async def _complete(self, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError

    async def _complete_stream(
        self,
        messages: list[dict[str, str]],
        on_chunk: Callable[[str], Awaitable[None]],
    ) -> str:
        raise NotImplementedError

    def push_fragment(self, fragment: str) -> str:
        """Append one transcriptor word to the window; return the snapshot."""
        if fragment:
            if self._window and not self._window[-1].isspace() and not fragment[0].isspace():
                self._window += " "
            self._window += fragment
            self._window = _normalize(self._window)
        return self._window

    def current_snapshot(self) -> str:
        return self._window

    def take_window(self) -> str:
        """Return the window snapshot and clear the window (for finalize)."""
        snapshot, self._window = self._window, ""
        return snapshot

    def take_final(self, fallback: str) -> str:
        """Snapshot for a turn end: the window, or the completion itself.

        Prefers the completed transcript when it extends the window
        (trailing punctuation often arrives only in the completion);
        otherwise the accumulated window wins, mirroring the old
        buffer-over-fallback behavior.
        """
        window, self._window = self._window, ""
        if not window.strip():
            return _normalize(fallback)
        done = _normalize(fallback)
        if done.startswith(window) and len(done) >= len(window):
            return done
        return window

    def set_window(self, text: str) -> None:
        self._window = _normalize(text)

    def should_finalize(self, snapshot: str) -> bool:
        """Whether a snapshot ends the current segment.

        True on sentence-ending punctuation or once the snapshot reaches
        the configured character cap (0 disables the cap).
        """
        if not snapshot or not snapshot.strip():
            return False
        if _ends_sentence(snapshot):
            return True
        return self._max_chars > 0 and len(snapshot) >= self._max_chars

    def split_head(self, snapshot: str) -> tuple[str, str]:
        """Split an over-long snapshot at a word boundary for the char cap."""
        return _split_head(snapshot, self._max_chars)

    def split_for_finalize(self, snapshot: str) -> tuple[str, str]:
        """Head to finalize now plus the tail that opens the next window.

        Whole snapshots ending in sentence punctuation finalize as one
        segment; snapshots that only hit the character cap split at a
        word boundary so the tail keeps growing under the next ``seg``.
        """
        if (
            self._max_chars > 0
            and len(snapshot) >= self._max_chars
            and not _ends_sentence(snapshot)
        ):
            return _split_head(snapshot, self._max_chars)
        return snapshot, ""

    def commit(self, source: str, translated: str) -> None:
        """Record a finalized segment and open the next ``seg_id``."""
        if self._context_segments > 0 and source.strip() and translated.strip():
            self._history.append((source, translated))
        self._window = ""
        self._seg_index += 1

    async def translate_snapshot(
        self,
        text: str,
        *,
        target_language: str,
        source_language: str | None = None,
        on_chunk: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """Translate one window snapshot. Stateless: history is read-only."""
        if not text or not text.strip():
            return text
        messages = self._build_messages(text, target_language, source_language)
        if on_chunk is None:
            return (await self._complete(messages)).strip()
        return (await self._complete_stream(messages, on_chunk)).strip()


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
        max_chars: int = 1,
        http_client: Any | None = None,
    ) -> None:
        super().__init__(context_segments=context_segments, max_chars=max_chars)
        self._api_key = api_key
        self._model = model
        self._base_url = (base_url or DEEPSEEK_DEFAULT_BASE_URL).rstrip("/")
        try:
            self._timeout = float(timeout_seconds)
        except (TypeError, ValueError):
            self._timeout = 15.0
        self._http_client = http_client

    @property
    def model(self) -> str:
        return self._model

    async def _complete(self, messages: list[dict[str, str]]) -> str:
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


def _normalize(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([.,!?;:])", r"\1", collapsed)


def _ends_sentence(text: str) -> bool:
    return text[-1:] in (".", "!", "?", "…", "。", "！", "？")


def _split_head(text: str, max_chars: int) -> tuple[str, str]:
    """Split an over-long window at a sentence end or word boundary."""
    if max_chars <= 0:
        return text, ""
    segment = text[:max_chars]
    for marker in (".", "!", "?", "…"):
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
                max_chars=getattr(settings, "subtitle_max_segment_chars", 300),
            )
            return translator
        return NullSubtitleTranslator()

    @staticmethod
    def provider_name(settings: Any) -> str:
        return SubtitleFactory.get_translator(settings).name
