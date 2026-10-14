"""Structured logging setup using structlog."""

from __future__ import annotations

import logging
import os
import sys

import structlog


def setup_logging(level: str | None = None) -> None:
    """Configure structlog + stdlib logging for the application."""
    log_level = getattr(logging, (level or os.environ.get("VTECH_LOG_LEVEL", "INFO")).upper())

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if sys.stderr.isatty() else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=log_level)
    _render_extras_on_stdlib_handlers()


# Attribute names present on every LogRecord. Derived from a probe record rather
# than hardcoded so it tracks the Python version (3.12 added ``taskName``).
_RESERVED_RECORD_ATTRS = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys()
) | {"message", "asctime"}


class ExtraFormatter(logging.Formatter):
    """Render a record's ``extra={}`` fields after the message.

    The stdlib handler was formatting with ``"%(message)s"`` alone, so every
    ``extra`` passed by the application was discarded before it reached the
    journal. That is why ``live_option_universe`` arrived as a bare event name
    with no symbol counts, and why ``reconnect_gap`` — which carries the exact
    boundaries of a live data gap handed to us by the Databento client — was
    computed and then thrown away.

    The message stays the leading token so existing ``journalctl | grep
    <event>`` usage keeps working; the fields are appended as ``key=value``.
    """

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {
            key: value
            for key, value in vars(record).items()
            if key not in _RESERVED_RECORD_ATTRS and not key.startswith("_")
        }
        if not extras:
            return base
        parts = []
        for key, value in sorted(extras.items()):
            text = str(value)
            parts.append(f"{key}={value!r}" if (" " in text or "=" in text) else f"{key}={text}")
        return f"{base} " + " ".join(parts)


def _render_extras_on_stdlib_handlers() -> None:
    """Swap the root handlers' formatter for one that keeps ``extra`` fields."""
    for handler in logging.getLogger().handlers:
        handler.setFormatter(ExtraFormatter("%(message)s"))
