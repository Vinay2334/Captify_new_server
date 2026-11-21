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

## Transcription tiers and translation behavior

- The extension's Model dropdown picks the transcription (speech-to-text)
  engine per session: `base` transcribes with Soniox `stt-rt-v5`
  (`SONIOX_MODEL`), `pro` transcribes with `gpt-live-transcribe`
  (`OPENAI_TRANSCRIPTION_MODEL`). The tier travels in the WebRTC offer
  (`model_tier`) and in every `language-config` (`model_tier`); when a
  request names a different tier, the backend connects the new engine
  and swaps it in on the fly — the transport never drops and the client
  never re-offers. A fresh `session.ready` announces the new model.
  Sessions without a tier keep the legacy `STT_PROVIDER` behavior. Unit
  economics per tier are unsettled on purpose: the tier only selects
  the engine, never pricing.
- Every tiered session transcribes: both engines emit verbatim
  source-language `transcript.delta` / `transcript.completed` events in
  the same downstream shape, so transcription output is identical
  whichever tier produced it.
- Translation is a separate text-to-text step over those transcripts,
  decoupled from transcription: changing the transcription tier never
  changes translation, and changing the subtitle provider never changes
  transcription. Session mode is still picked per session in the
  extension's Mode dropdown (`translation` / `transcription`). Source
  language is always auto-detected (the extension no longer sends one);
  the target language defaults to `OPENAI_TRANSLATION_TARGET_LANGUAGE`
  (`en`). The extension can switch it per session by sending
  `{"type": "language-config", "mode": "translation", "target_lang": "es"}`
  (transcription sends just `{"type": "language-config", "mode": "transcription"}`,
  no target, and captions stay in the auto-detected source language):
  `target_lang` sets the subtitle translation target (repeats for the
  same language are skipped). Tiered sessions never send `target_lang`
  upstream: they keep transcribing and let the subtitle step render the
  target text.
- Transcribe mode: set `TRANSCRIBE_MODE=true` to transcribe every session
  with `gpt-realtime-transcribe` (`OPENAI_TRANSCRIPTION_MODEL`) instead of
  translating. Commits close each turn and transcription events map to the
  same `transcript.delta` / `transcript.completed` stream, so the extension
  protocol is unchanged; `session.ready` and `/api/v1/health` report the
  transcription model.
- Legacy same-language picks transcribe instead of translating: the
  translation model omits same-language audio, which would otherwise
  break subtitles. When `source_lang` matches `target_lang` (for example
  English captions for English audio), the backend routes audio chunks
  through `gpt-realtime-transcribe` (`OPENAI_TRANSCRIPTION_MODEL`), forwards
  commits per turn, and maps transcription events to the same
  `transcript.delta` / `transcript.completed` stream. Set
  `OPENAI_TRANSLATION_SOURCE_LANGUAGE` to match the target when a
  deployment is same-language from the start, so the upstream socket
  opens on the transcription endpoint.
- Legacy (untiered) OpenAI translation sessions still forward translated
  transcript deltas (`session.output_transcript.delta` → `transcript.delta`).
  Translation emits deltas only, so `transcript.completed` never fires.
  Soniox never translates: its one-way translation workflow was removed,
  and Soniox sessions are transcription-only (verbatim source text).
- Subtitle translation: set `SUBTITLE_PROVIDER=deepseek` (plus
  `DEEPSEEK_API_KEY`) to translate `openai-live-transcribe` transcripts
  into the session target language with the `deepseek-flash` chat model
  (`SUBTITLE_MODEL`). The last `SUBTITLE_CONTEXT_SEGMENTS` (default 3)
  segments travel as context so pronouns and topic stay consistent; only
  the current segment is translated. Transcription deltas arrive word by
  word, so fragments are buffered per turn and DeepSeek is only called for
  flush-ready segments (sentence end, `SUBTITLE_MAX_SEGMENT_CHARS` cap, or
  `SUBTITLE_FLUSH_SECONDS` age with `SUBTITLE_MIN_SEGMENT_CHARS` minimum).
  Applies to transcription-mode
  transcripts on both transports; failures fall back to the original text.
  Sentences stream token-by-token as `transcript.delta` events while
  DeepSeek responds, then land as `transcript.completed`; committed
  sentences are never re-translated. Each translated sentence starts
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

## Authentication + quota (Supabase)

When `SUPABASE_URL` is set, the WebSocket and WebRTC transports require a
verified Supabase access token and every subtitle session is billed in
whole seconds against the user's quota (`Remaining = limit - used`).
Without it the endpoints keep the open local-dev behavior above.

While a session runs, the backend checkpoints its billed-so-far seconds
to the DB every `QUOTA_COMMIT_SECONDS` (default 15) and then re-reads
quota, so concurrent sessions see each other's usage within one interval;
between checkpoints a session's own uncommitted audio already counts
against its remainder, and it is cut off at zero.

- Schema + RLS + atomic RPCs: `supabase/migrations/0001_auth_quota.sql`
  (plus `20261119000000_quota_checkpoint.sql` for the checkpoint RPC —
  apply it with `supabase db push` on existing projects)
- Setup (Google OAuth, extension config, dashboard plan changes):
  `SUPABASE_SETUP.md`
- Quota tests: `tests/test_auth_quota.py`

## Tests

```
uv run pytest
```
