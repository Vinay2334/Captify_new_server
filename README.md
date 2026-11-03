# Captify Translation Backend

FastAPI backend relaying tab audio to OpenAI Realtime translation
(`gpt-realtime-translate`, or transcription with `gpt-realtime-transcribe`).
Two client transports are supported; both feed the same upstream session.

## Transports

- WebSocket JSON: `WS /api/v1/transcription/ws`
  (`Extension/` build). Text events: `audio.append` (`base64` PCM16
  24 kHz mono) / `audio.commit` upstream; `session.ready` /
  `transcript.delta` / `error` downstream.
- WebRTC DataChannel: `POST /api/v1/transcription/webrtc/offer`
  (`extensionRTC/` build, see `extensionRTC/rtc-config.js` `SIGNAL_URL`).
  Request `{ sdp, type: "offer" }`, reply `{ sdp, type: "answer" }`
  (non-trickle ICE). Audio arrives as raw Int16 PCM 24 kHz mono
  messages (~200 ms each) on the `audio` channel; translated transcripts
  return as JSON text on the same channel. The server streams audio
  continuously with no commits, matching the upstream translation
  session, which has no commit / turn lifecycle.

## Translation behavior

- Source language is auto-detected; the target language defaults to
  `OPENAI_TRANSLATION_TARGET_LANGUAGE` (`en`). The extension can switch
  it per session by sending
  `{"type": "language-config", "source_lang": "...", "target_lang": "es"}`:
  `target_lang` is forwarded upstream as a `session.update` (repeats for
  the same language are skipped).
- Transcribe mode: set `TRANSCRIBE_MODE=true` to transcribe every session
  with `gpt-realtime-transcribe` (`OPENAI_TRANSCRIPTION_MODEL`) instead of
  translating. Commits close each turn and transcription events map to the
  same `transcript.delta` / `transcript.completed` stream, so the extension
  protocol is unchanged; `session.ready` and `/api/v1/health` report the
  transcription model.
- Same-language sessions transcribe instead of translating: the
  translation model omits same-language audio, which would otherwise
  break subtitles. When `source_lang` matches `target_lang` (for example
  English captions for English audio), the backend routes audio chunks
  through `gpt-realtime-transcribe` (`OPENAI_TRANSCRIPTION_MODEL`), forwards
  commits per turn, and maps transcription events to the same
  `transcript.delta` / `transcript.completed` stream. Set
  `OPENAI_TRANSLATION_SOURCE_LANGUAGE` to match the target when a
  deployment is same-language from the start, so the upstream socket
  opens on the transcription endpoint.
- The backend forwards translated transcript deltas
  (`session.output_transcript.delta` → `transcript.delta`). Translation
  emits deltas only, so `transcript.completed` never fires.
- Subtitle translation: set `SUBTITLE_PROVIDER=deepseek` (plus
  `DEEPSEEK_API_KEY`) to translate `openai-live-transcribe` transcripts
  into the session target language with the `deepseek-flash` chat model
  (`SUBTITLE_MODEL`). The last `SUBTITLE_CONTEXT_SEGMENTS` (default 3)
  finalized segments travel as context so pronouns and topic stay
  consistent; only the current window is translated. Every transcriptor
  word grows the source window and re-triggers the translation (up to
  `SUBTITLE_MAX_SEGMENT_CHARS`), so the first translated words stream
  back as `transcript.delta` events immediately instead of waiting for
  the sentence to finish. Each window carries a `seg_id` plus a `rev`
  counter: the frontend replaces the line for `seg_id` whenever `rev`
  bumps (token pieces within one `rev` append). A window finalizes on
  sentence end, at the character cap, or at turn end; the backend then
  sends `transcript.completed` with the same `seg_id` as the
  segment-final identifier and the next window starts under the next
  `seg_id`. Only finalized segments enter the history context.
  Applies to transcription-mode
  transcripts on both transports; failures fall back to the original text.
  Each translated segment starts
  with a newline so consecutive captions don't run together.
- Source-language deltas and translated audio (`session.output_audio.delta`)
  are dropped: the downstream protocol is translated-text-only.
- Client `audio.commit` events are accepted (the extension still sends
  them) but swallowed in translation mode; in transcribe mode commits close

  the turn and are forwarded upstream.

## Run

```
uv sync
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.

## Tests

```
uv run pytest
```
