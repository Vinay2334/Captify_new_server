# Captify Transcription Backend

FastAPI backend relaying tab audio to OpenAI Realtime transcription.
Two client transports are supported; both feed the same upstream session.

## Transports

- WebSocket JSON: `WS /api/v1/transcription/ws`
  (`Extension/` build). Text events: `audio.append` (`base64` PCM16
  24 kHz mono) / `audio.commit` upstream; `session.ready` /
  `transcript.delta` / `transcript.completed` / `error` downstream.
- WebRTC DataChannel: `POST /api/v1/transcription/webrtc/offer`
  (`extensionRTC/` build, see `extensionRTC/rtc-config.js` `SIGNAL_URL`).
  Request `{ sdp, type: "offer" }`, reply `{ sdp, type: "answer" }`
  (non-trickle ICE). Audio arrives as raw Int16 PCM 24 kHz mono
  messages (~200 ms each) on the `audio` channel; transcripts return
  as JSON text on the same channel. The server auto-commits while
  audio flows (`WEBRTC_AUTO_COMMIT_SECONDS`) plus a final commit on
  channel close, since the extension never sends commits.

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
