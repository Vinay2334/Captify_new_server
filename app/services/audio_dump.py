"""Opt-in recorder for inbound client audio (debugging only).

Saves exactly what the client sent as a standard WAV file (mono PCM16 at
24 kHz) so a silent-model session can be diagnosed by ear: silence, wrong
input, or resampling bugs are obvious on playback.

Disabled by default; enable with ``SAVE_RECEIVED_AUDIO=true``. Files land
in ``RECEIVED_AUDIO_DIR`` (default ``recordings/``). Never enable in
production: these files contain raw user audio. Stdlib only.
"""

import os
import time
import wave

SAMPLE_RATE_HZ = 24000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2

_counter = 0


class AudioDumpWriter:
    """Append-only WAV writer for one session. Creates the file lazily."""

    def __init__(self, directory: str, prefix: str = "webrtc") -> None:
        self._directory = directory
        self._prefix = prefix
        self._wav = None
        self.path: str | None = None
        self.bytes_written = 0
        self.chunks_written = 0

    def write(self, frames: bytes) -> None:
        """Append one validated PCM16 chunk. Skips empty input."""
        if not frames:
            return
        if self._wav is None:
            self._open()
        assert self._wav is not None
        self._wav.writeframes(frames)
        self.bytes_written += len(frames)
        self.chunks_written += 1

    def _open(self) -> None:
        global _counter
        os.makedirs(self._directory, exist_ok=True)
        _counter += 1
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        self.path = os.path.join(
            self._directory, f"{self._prefix}-{stamp}-{_counter}.wav"
        )
        self._wav = wave.open(self.path, "wb")
        self._wav.setnchannels(CHANNELS)
        self._wav.setsampwidth(SAMPLE_WIDTH_BYTES)
        self._wav.setframerate(SAMPLE_RATE_HZ)

    def close(self) -> "str | None":
        """Finalize the WAV file. Returns the path, or None if unused."""
        wav, self._wav = self._wav, None
        if wav is not None:
            try:
                wav.close()
            except Exception:
                pass
        return self.path
