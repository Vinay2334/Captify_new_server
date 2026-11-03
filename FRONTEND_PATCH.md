# Frontend patch: growing-window replace protocol

> Status: APPLIED to `extensionRTC/src/offscreen.js` (`handleServerMessage`
> + session-state comment). `content-script.js` needs no change (it already
> replaces line text and keys lines by the broadcast id).
> Required follow-up: rebuild the loadable extension — run `npm run build`
> in `extensionRTC/` so `dist/offscreen.js` picks up the `src/` change
> (`dist/` is what Chrome loads; `content-script.js` is copied verbatim).

The backend (`server_new`) now streams subtitle translations as a
**growing window**:

- Every transcriptor word re-translates the whole source window and is
  sent as `transcript.delta` with the **full translated snapshot** plus
  `seg_id` / `rev`.
- The frontend must **replace** the line for `seg_id` whenever `rev`
  bumps (token pieces *within* one `rev` still append).
- `transcript.completed` carries the same `seg_id` as the
  **segment-final identifier**: freeze that line, the next window uses
  the next `seg_id`.

The legacy `Extension/` build only logs socket messages, so only
`extensionRTC/src/offscreen.js` needs the change below
(`content-script.js` already replaces `currentLine.textContent`, keyed
by the id it is given — pass `seg_id` through as that id).

## 1. Session state (around line 92)

Replace:

```js
transcriptBuffers: new Map(),
currentItemId: null,
currentText: "",
```

with:

```js
transcriptBuffers: new Map(), // segKey -> { rev, text }
currentItemId: null, // now holds the active segKey
currentText: "",
```

`clearCaptions` (`.clear()`, `= null`, `= ""`) stays as-is.

## 2. `handleServerMessage` (around line 306)

Replace the `transcript.delta` / `transcript.completed` branches with:

```js
if (msg.type === "transcript.delta") {
  // seg_id is the replace key; item_id stays as the upstream turn id.
  // Missing seg_id/rev means an old backend: fall back to append.
  const seg = msg.seg_id || msg.item_id;
  const rev = msg.rev || 0;
  const prev = session.transcriptBuffers.get(seg);
  let next;
  if (!prev || prev.rev !== rev) {
    next = msg.delta || ""; // new window or new revision: replace
  } else {
    next = prev.text + (msg.delta || ""); // token piece: append
  }
  session.transcriptBuffers.set(seg, { rev, text: next });
  session.currentItemId = seg;
  session.currentText = next;
  broadcastCaption(session, session.currentText, session.currentItemId, false);
} else if (msg.type === "transcript.completed") {
  const seg = msg.seg_id || msg.item_id;
  const transcript = msg.transcript || "";
  if (transcript) {
    session.transcriptBuffers.set(seg, { rev: -1, text: transcript });
    session.currentItemId = seg;
    session.currentText = transcript;
    broadcastCaption(session, session.currentText, session.currentItemId, true);
    session.currentItemId = null;
    session.currentText = "";
  } else if (seg === session.currentItemId || !transcript) {
    // Empty completion (silence/VAD artifact): hide a stale line rather
    // than flashing an empty box.
    session.transcriptBuffers.delete(seg);
    if (seg === session.currentItemId) {
      session.currentItemId = null;
      session.currentText = "";
      broadcastCaption(session, "", seg, true);
    }
  }
}
```

Notes:

- Setting `currentItemId = null` after a final matches
  `content-script.js show(text, itemId, final=true)`, which already
  nulls its own key — the next `seg_id` always opens a fresh line.
- Old-backend fallback: without `seg_id`, `seg` is the upstream
  `item_id` and every message has `rev 0`, so the first delta replaces
  once and later pieces append — identical to the previous behavior.
