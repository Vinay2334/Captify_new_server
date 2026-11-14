
import logging
import sys

_configured = False

# Suppress noisy third-party networking logs
THIRD_PARTY_LOGGERS = [
    "aiortc",
    "aioice",
    "httpcore",
    "httpx",
    "websockets",
    "asyncio",
]

def configure_logging(level: str = "INFO") -> logging.Logger:
    global _configured

    log_level = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"
        )
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)

    # Explicitly control third-party loggers
    for logger_name in THIRD_PARTY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    _configured = True
    return logging.getLogger("captify")


def get_logger(name: str) -> logging.Logger:
    if not _configured:
        configure_logging()
    return logging.getLogger(name)
