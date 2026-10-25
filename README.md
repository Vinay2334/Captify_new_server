# Captify Translation Backend

FastAPI backend relaying tab audio to OpenAI Realtime translation
(`gpt-realtime-translate`). Two client transports are supported; both feed
the same upstream session.

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
  the same language are skipped). `source_lang` is accepted but unused —
  the API has no source-language field.
- The backend forwards translated transcript deltas
  (`session.output_transcript.delta` → `transcript.delta`). Translation
  emits deltas only, so `transcript.completed` never fires.
- Source-language deltas and translated audio (`session.output_audio.delta`)
  are dropped: the downstream protocol is translated-text-only.
- Client `audio.commit` events are accepted (the extension still sends
  them) but swallowed, never forwarded upstream.

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
