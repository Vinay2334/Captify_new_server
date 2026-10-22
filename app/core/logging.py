"""Logging setup.

Never log audio payloads or API keys. Audio flow is visible through
metadata only: chunk counts, decoded byte sizes, bytes/s rates, commit
counts, and session summaries (see ``RelaySession``). Transcript text is
logged at INFO so it is visible in the terminal during local development.
"""

import logging
import sys

_configured = False


def configure_logging(level: str = "INFO") -> logging.Logger:
    global _configured
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    _configured = True
    return logging.getLogger("captify")


def get_logger(name: str) -> logging.Logger:
    if not _configured:
        configure_logging()
    return logging.getLogger(name)
