"""Rotating DBN archive writer for a long-lived Databento live session.

Why this exists
---------------
``databento.Live.add_stream`` is set once and has no removal counterpart, so a
session that persists for a whole trading week would otherwise write one
archive file for the week. The archive contract in
``docs/architecture/DATA_PIPELINE.md`` is one directory per Chicago session
date, which the per-session teardown used to provide for free.

``RotatingDbnStream`` is a writable byte stream that satisfies
``add_stream``'s ``IO[bytes]`` form and rotates its output file whenever the
session date advances, keeping the archive layout unchanged while the live
session stays connected.

How it stays correct
--------------------
Two properties of the client make this safe, both verified against
``databento-python`` 0.74.1:

* The DBN metadata header is written exactly once, by
  ``DatabentoLiveProtocol.received_metadata``, and is **not** re-emitted after a
  reconnect (``received_metadata`` only writes when its metadata slot is
  empty). This class therefore captures the header and replays it at the head
  of every rotated file, so each file decodes standalone.
* ``_dispatch_writes`` calls ``stream.write(bytes(record))`` once per record,
  so every ``write`` carries one complete record and a rotation taken between
  calls can never split one.
* Live headers contain no symbol mappings. Mapping records received before a
  rotation are cached and replayed immediately after the new file's header,
  so retained subscriptions remain independently resolvable in every file.

Files open lazily, on the first *record* of a session date rather than at
rotation time. A rotation with no records behind it therefore creates nothing
— which is what stops the header-only archive stubs that reconnect churn
produced under the per-session writer (three 13-byte files on 2026-09-29).

Known limitation
----------------
The replayed header is verbatim, so its ``start`` field names the live
session's start rather than the rotated file's first record. Record timestamps
are unaffected, and both the ingest path (``src/ingestion/loader.py``,
``scripts/data/ingest_dbn.py``) and ``DBNStore`` read record timestamps rather
than the header hint, so this is cosmetic. Patching it would mean depending on
``databento_dbn.update_encoded_metadata``, whose signature is undocumented.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

from databento_dbn import DBNDecoder, RType, SymbolMappingMsg

logger = logging.getLogger("vtech.acquisition.rotating_stream")

# A DBN stream opens with the ASCII magic followed by a version byte.
_DBN_MAGIC = b"DBN"
# Highest DBN version this discriminator will accept. Checking the version byte
# as well as the magic makes a false positive effectively impossible: a record's
# first two bytes are its length and rtype, and 0x42 ("B") is not a defined
# rtype, so a record cannot open with "DBN" followed by a small integer.
_DBN_MAX_VERSION = 10


def _is_dbn_header(data: bytes) -> bool:
    return (
        len(data) > len(_DBN_MAGIC)
        and data[: len(_DBN_MAGIC)] == _DBN_MAGIC
        and data[len(_DBN_MAGIC)] <= _DBN_MAX_VERSION
    )


class RotatingDbnStream:
    """Writable byte stream for ``Live.add_stream`` that rotates per session date.

    Parameters
    ----------
    root:
        Directory that holds one subdirectory per session date. The caller
        passes a resolver-backed path (``raw_dir() / "live"``); this class never
        resolves paths itself, so ``src/common/data_paths.py`` stays the single
        path authority.
    session_date_fn:
        Returns the session date the next record belongs to. Called on every
        record write, so it must be cheap and must not raise. Injected rather
        than computed internally so rotation is testable without a clock.
    clock:
        Returns "now" for the ``stream_<HHMMSS>`` filename. Defaults to UTC now.
    suffix:
        File suffix. Defaults to ``.dbn`` — the daily
        ``scripts/ops/compress_raw_live.sh`` timer applies zstd afterwards.

    Notes
    -----
    ``ClientStream`` requires ``write``, ``writable()``, ``flush``, ``close``
    and a ``closed`` property, and reads ``__name__`` for its log messages.
    Exceptions raised from ``write`` are swallowed by ``ClientStream`` and
    turned into warnings, so failures here are logged rather than relying on
    the caller to see them.
    """

    __name__ = "RotatingDbnStream"

    def __init__(
        self,
        *,
        root: Path,
        session_date_fn: Callable[[], date],
        clock: Callable[[], datetime] | None = None,
        suffix: str = ".dbn",
    ) -> None:
        self._root = Path(root)
        self._session_date_fn = session_date_fn
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._suffix = suffix

        self._lock = threading.Lock()
        self._header: bytes | None = None
        # Full mapping identity, including validity and aliases. IDs alone are
        # insufficient: an ID can name different instruments over time. Keep
        # receipt order so overlapping updates replay in their original order.
        self._mappings: OrderedDict[tuple, bytes] = OrderedDict()
        self._fh = None
        self._current_date: date | None = None
        self._current_path: Path | None = None
        self._closed = False

        # Observability. The live streamer logs these on each rotation; without
        # them a week-long session gives no signal that rotation is happening.
        self.files_opened = 0
        self.rotations = 0
        self.records_written = 0
        self.bytes_written = 0
        self.mappings_replayed = 0
        self.paths: list[Path] = []

    # ── stream protocol ──────────────────────────────────────────

    def writable(self) -> bool:
        return not self._closed

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def current_path(self) -> Path | None:
        """Path currently being written, or None before the first record."""
        return self._current_path

    @property
    def header_captured(self) -> bool:
        return self._header is not None

    def write(self, data: bytes) -> int:
        """Write one DBN metadata header or one complete DBN record.

        The first call carries the metadata header, which is captured rather
        than written through; every later call carries exactly one record.
        """
        if not data:
            return 0
        with self._lock:
            if self._closed:
                raise ValueError("write to a closed RotatingDbnStream")

            if _is_dbn_header(data):
                # Metadata header. Capture it for replay into each rotated
                # file; never write it as a record. A second header would only
                # appear if the client re-sent metadata, in which case the newer
                # one is the correct thing to replay.
                self._header = bytes(data)
                return len(data)

            if self._header is None:
                raise ValueError(
                    "DBN record written before the metadata header — "
                    "cannot produce a decodable archive file",
                )

            session_date = self._session_date_fn()
            if self._fh is None or session_date != self._current_date:
                self._open(session_date)

            assert self._fh is not None  # _open guarantees this
            self._fh.write(data)
            self._cache_mapping(data)
            self.records_written += 1
            self.bytes_written += len(data)
            return len(data)

    def _cache_mapping(self, data: bytes) -> None:
        """Remember mapping records without decoding the market-data stream."""
        if len(data) < 2 or data[1] != int(RType.SYMBOL_MAPPING):
            return
        assert self._header is not None
        decoder = DBNDecoder(has_metadata=False, input_version=self._header[3])
        decoder.write(data)
        for record in decoder.decode():
            if not isinstance(record, SymbolMappingMsg):
                continue
            key = (
                record.publisher_id, record.instrument_id,
                record.stype_in, record.stype_in_symbol,
                record.stype_out, record.stype_out_symbol,
                record.start_ts, record.end_ts,
            )
            self._mappings[key] = bytes(data)
            # A reconnect can repeat a mapping with a new ts_event. Retain its
            # newest bytes and receipt position without growing the cache.
            self._mappings.move_to_end(key)

    def flush(self) -> None:
        with self._lock:
            if self._fh is not None and not self._fh.closed:
                self._fh.flush()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._close_handle()

    # ── internals ────────────────────────────────────────────────

    def _close_handle(self) -> None:
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:  # noqa: BLE001 — never let a close fault lose the stream
                logger.exception("dbn_archive_close_failed", extra={"path": str(self._current_path)})
            self._fh = None

    def _open(self, session_date: date) -> None:
        """Close any current file and open the one for ``session_date``."""
        rotating = self._fh is not None
        previous = self._current_path
        self._close_handle()

        directory = self._root / session_date.isoformat()
        directory.mkdir(parents=True, exist_ok=True)

        stamp = self._clock().strftime("%H%M%S")
        path = directory / f"stream_{stamp}{self._suffix}"
        collision = 1
        while path.exists():
            # Never clobber an existing archive file: a restart inside the same
            # second, or a re-run over an existing directory, gets its own name.
            path = directory / f"stream_{stamp}_{collision}{self._suffix}"
            collision += 1

        handle = path.open("xb")
        handle.write(self._header or b"")
        for mapping in self._mappings.values():
            handle.write(mapping)
        self.mappings_replayed += len(self._mappings)

        self._fh = handle
        self._current_date = session_date
        self._current_path = path
        self.files_opened += 1
        self.paths.append(path)
        if rotating:
            self.rotations += 1

        logger.info(
            "dbn_archive_opened",
            extra={
                "path": str(path),
                "session_date": session_date.isoformat(),
                "rotated_from": str(previous) if previous else None,
                "files_opened": self.files_opened,
                "records_written": self.records_written,
                "mappings_replayed": len(self._mappings),
            },
        )

    def __str__(self) -> str:
        return f"RotatingDbnStream({self._current_path or self._root})"
