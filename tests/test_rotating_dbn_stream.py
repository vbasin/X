"""Tests for :mod:`src.acquisition.rotating_stream`.

No database. The unit tests drive the stream with synthetic byte payloads that
reproduce the client's write pattern (one header, then one complete record per
``write``). The integration test replays a real ``.dbn.zst`` from the archive
through the stream and asserts every rotated file decodes standalone — it
skips when no archive file is available, so the suite still runs on a box
without market data.
"""

from __future__ import annotations

import functools
import glob
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.acquisition.rotating_stream import RotatingDbnStream

pytestmark = pytest.mark.acquisition

HEADER = b"DBN\x03" + b"\x00" * 60
D1 = date(2026, 9, 28)
D2 = date(2026, 9, 29)
D3 = date(2026, 9, 30)


def _record(n: int, size: int = 40) -> bytes:
    """A stand-in for one DBN record: fixed width, identifiable, not DBN magic."""
    body = n.to_bytes(4, "little") * (size // 4)
    assert body[:3] != b"DBN"
    return body


class _Clock:
    """Deterministic clock whose HHMMSS advances one second per call."""

    def __init__(self) -> None:
        self._t = datetime(2026, 9, 28, 17, 0, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        self._t = self._t.replace(second=(self._t.second + 1) % 60)
        return self._t


# ── header handling ──────────────────────────────────────────────


def test_header_is_captured_and_not_written_as_a_record(tmp_path: Path) -> None:
    stream = RotatingDbnStream(root=tmp_path, session_date_fn=lambda: D1, clock=_Clock())
    assert not stream.header_captured

    written = stream.write(HEADER)

    assert written == len(HEADER)
    assert stream.header_captured
    assert stream.records_written == 0
    # Nothing on disk yet: files open on the first record, not on the header.
    assert list(tmp_path.rglob("*.dbn")) == []
    assert stream.current_path is None
    stream.close()


def test_record_before_header_is_rejected(tmp_path: Path) -> None:
    stream = RotatingDbnStream(root=tmp_path, session_date_fn=lambda: D1, clock=_Clock())
    with pytest.raises(ValueError, match="before the metadata header"):
        stream.write(_record(1))
    stream.close()


def test_header_discriminator_checks_the_version_byte(tmp_path: Path) -> None:
    """A record is never mistaken for a header on the magic bytes alone.

    The magic alone is not a safe discriminator, so the version byte is checked
    too. A payload opening with ``DBN`` followed by an implausible version is a
    record and must be written through, not swallowed as metadata.
    """
    stream = RotatingDbnStream(root=tmp_path, session_date_fn=lambda: D1, clock=_Clock())
    stream.write(HEADER)
    look_alike = b"DBN\xff" + b"\x01" * 36
    stream.write(look_alike)
    stream.close()

    files = list(tmp_path.rglob("*.dbn"))
    assert len(files) == 1
    assert stream.records_written == 1
    assert files[0].read_bytes() == HEADER + look_alike


def test_every_file_starts_with_the_captured_header(tmp_path: Path) -> None:
    current = {"d": D1}
    stream = RotatingDbnStream(
        root=tmp_path, session_date_fn=lambda: current["d"], clock=_Clock()
    )
    stream.write(HEADER)
    stream.write(_record(1))
    current["d"] = D2
    stream.write(_record(2))
    stream.close()

    files = sorted(tmp_path.rglob("*.dbn"))
    assert len(files) == 2
    for f in files:
        assert f.read_bytes().startswith(HEADER)


# ── rotation ─────────────────────────────────────────────────────


def test_single_session_date_writes_one_file(tmp_path: Path) -> None:
    stream = RotatingDbnStream(root=tmp_path, session_date_fn=lambda: D1, clock=_Clock())
    stream.write(HEADER)
    for i in range(50):
        stream.write(_record(i))
    stream.close()

    files = list(tmp_path.rglob("*.dbn"))
    assert len(files) == 1
    assert files[0].parent.name == D1.isoformat()
    assert stream.files_opened == 1
    assert stream.rotations == 0
    assert stream.records_written == 50
    assert files[0].stat().st_size == len(HEADER) + 50 * 40


def test_rotation_splits_records_by_session_date(tmp_path: Path) -> None:
    current = {"d": D1}
    stream = RotatingDbnStream(
        root=tmp_path, session_date_fn=lambda: current["d"], clock=_Clock()
    )
    stream.write(HEADER)
    for i in range(10):
        stream.write(_record(i))
    current["d"] = D2
    for i in range(10, 25):
        stream.write(_record(i))
    current["d"] = D3
    for i in range(25, 30):
        stream.write(_record(i))
    stream.close()

    assert stream.files_opened == 3
    assert stream.rotations == 2
    assert stream.records_written == 30

    for d, expected in ((D1, 10), (D2, 15), (D3, 5)):
        files = list((tmp_path / d.isoformat()).glob("*.dbn"))
        assert len(files) == 1, f"{d} should have exactly one file"
        payload = files[0].read_bytes()[len(HEADER) :]
        assert len(payload) == expected * 40
        # Records land in the right file, in order.
        first = int.from_bytes(payload[:4], "little")
        assert first == {D1: 0, D2: 10, D3: 25}[d]


def test_large_header_is_replayed_into_every_file(tmp_path: Path) -> None:
    """A historical header can carry a large symbol map.

    Each rotated file must receive that header whole. Live symbol mappings
    instead arrive as records and are covered by the tests below.
    """
    big_header = b"DBN\x03" + b"\xab" * 300_000
    current = {"d": D1}
    stream = RotatingDbnStream(
        root=tmp_path, session_date_fn=lambda: current["d"], clock=_Clock()
    )
    stream.write(big_header)
    stream.write(_record(1))
    current["d"] = D2
    stream.write(_record(2))
    stream.close()

    files = sorted(tmp_path.rglob("*.dbn"))
    assert len(files) == 2
    for f in files:
        payload = f.read_bytes()
        assert payload[: len(big_header)] == big_header
        assert len(payload) == len(big_header) + 40


def test_full_week_of_rotations(tmp_path: Path) -> None:
    """Five session dates, as a Sunday-to-Friday live session would produce."""
    week = [date(2026, 9, 28) + timedelta(days=i) for i in range(5)]
    current = {"d": week[0]}
    stream = RotatingDbnStream(
        root=tmp_path, session_date_fn=lambda: current["d"], clock=_Clock()
    )
    stream.write(HEADER)
    for i, day in enumerate(week):
        current["d"] = day
        for _ in range(i + 1):  # 1, 2, 3, 4, 5 records per day
            stream.write(_record(i))
    stream.close()

    assert stream.files_opened == 5
    assert stream.rotations == 4
    assert stream.records_written == 15
    assert [p.parent.name for p in stream.paths] == [d.isoformat() for d in week]
    for i, day in enumerate(week):
        files = list((tmp_path / day.isoformat()).glob("*.dbn"))
        assert len(files) == 1
        assert files[0].stat().st_size == len(HEADER) + (i + 1) * 40


def test_rotation_without_records_creates_no_file(tmp_path: Path) -> None:
    """A session roll with no traffic behind it must not leave a header-only stub.

    This is the defect the per-session writer produced: three 13-byte files in
    65 seconds of reconnect churn on 2026-09-29.
    """
    current = {"d": D1}
    stream = RotatingDbnStream(
        root=tmp_path, session_date_fn=lambda: current["d"], clock=_Clock()
    )
    stream.write(HEADER)
    stream.write(_record(1))

    # Three rolls, no records in between.
    for d in (D2, D3, date(2026, 10, 1)):
        current["d"] = d

    stream.close()

    files = list(tmp_path.rglob("*.dbn"))
    assert len(files) == 1
    assert files[0].parent.name == D1.isoformat()
    assert stream.files_opened == 1
    assert stream.rotations == 0


def test_existing_file_is_never_clobbered(tmp_path: Path) -> None:
    clock = _Clock()
    first = RotatingDbnStream(root=tmp_path, session_date_fn=lambda: D1, clock=clock)
    first.write(HEADER)
    first.write(_record(1))
    original = first.current_path
    assert original is not None
    first.close()
    original_bytes = original.read_bytes()

    # A second stream in the same second-stamp must pick a new name.
    second = RotatingDbnStream(
        root=tmp_path, session_date_fn=lambda: D1, clock=lambda: datetime(
            2026, 9, 28, 17, 0, 1, tzinfo=timezone.utc
        )
    )
    second.write(HEADER)
    second.write(_record(2))
    assert second.current_path != original
    second.close()

    assert original.read_bytes() == original_bytes
    assert len(list((tmp_path / D1.isoformat()).glob("*.dbn"))) == 2


# ── stream protocol contract ─────────────────────────────────────


def test_satisfies_client_stream_contract(tmp_path: Path) -> None:
    """``ClientStream`` validates the object before the session accepts it."""
    from databento.common.types import ClientStream

    stream = RotatingDbnStream(root=tmp_path, session_date_fn=lambda: D1, clock=_Clock())
    wrapped = ClientStream(stream)  # raises ValueError if the contract is unmet

    wrapped.write(HEADER)
    wrapped.write(_record(1))
    wrapped.flush()
    assert not wrapped.is_closed
    assert wrapped.stream_name == "RotatingDbnStream"
    wrapped.close()
    assert wrapped.is_closed


def test_write_after_close_is_rejected(tmp_path: Path) -> None:
    stream = RotatingDbnStream(root=tmp_path, session_date_fn=lambda: D1, clock=_Clock())
    stream.write(HEADER)
    stream.write(_record(1))
    stream.close()

    assert stream.closed
    assert not stream.writable()
    with pytest.raises(ValueError, match="closed"):
        stream.write(_record(2))


def test_empty_write_is_a_noop(tmp_path: Path) -> None:
    stream = RotatingDbnStream(root=tmp_path, session_date_fn=lambda: D1, clock=_Clock())
    assert stream.write(b"") == 0
    assert stream.records_written == 0
    stream.close()


def test_concurrent_writes_lose_no_records(tmp_path: Path) -> None:
    """The client writes from its own thread; rotation must not race it."""
    current = {"d": D1}
    stream = RotatingDbnStream(
        root=tmp_path, session_date_fn=lambda: current["d"], clock=_Clock()
    )
    stream.write(HEADER)

    n_threads, per_thread = 8, 200

    def worker(tid: int) -> None:
        for i in range(per_thread):
            stream.write(_record(tid * per_thread + i))
            if tid == 0 and i == per_thread // 2:
                current["d"] = D2

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stream.close()

    assert stream.records_written == n_threads * per_thread
    total = sum(
        f.stat().st_size - len(HEADER) for f in tmp_path.rglob("*.dbn")
    )
    assert total == n_threads * per_thread * 40


# ── integration: real DBN bytes must decode after rotation ───────


def _timestamp(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, 12, tzinfo=timezone.utc).timestamp()) * 10**9


def _live_header() -> bytes:
    import databento_dbn as dbn

    return dbn.Metadata(
        dataset="GLBX.MDP3", start=_timestamp(D1), schema=None,
        stype_in=None, stype_out=dbn.SType.INSTRUMENT_ID, version=3,
    ).encode()


def _mapping(
    symbol: str, *, instrument_id: int = 42, alias: str | None = None,
    start: date = D1, end: date | None = None, event: int = 1,
) -> bytes:
    import databento_dbn as dbn

    return bytes(dbn.SymbolMappingMsg(
        publisher_id=0, instrument_id=instrument_id, ts_event=event,
        stype_in=dbn.SType.RAW_SYMBOL, stype_in_symbol=alias or symbol,
        stype_out=dbn.SType.RAW_SYMBOL, stype_out_symbol=symbol,
        start_ts=_timestamp(start),
        end_ts=dbn.UNDEF_TIMESTAMP if end is None else _timestamp(end),
    ))


def _quote(day: date, instrument_id: int = 42) -> bytes:
    import databento_dbn as dbn

    return bytes(dbn.BBOMsg(
        rtype=dbn.RType.BBO_1S, publisher_id=1, instrument_id=instrument_id,
        ts_event=_timestamp(day), ts_recv=_timestamp(day),
        price=25_000 * 10**9, size=1, side=dbn.Side.NONE,
    ))


def _symbols(path: Path) -> list[str | None]:
    import databento

    return databento.DBNStore.from_file(path).to_df(
        schema="bbo-1s", map_symbols=True,
    )["symbol"].tolist()


def test_rotated_file_retains_symbols_without_new_gateway_mappings(tmp_path: Path) -> None:
    """Retained chains and futures must decode from the rotated file alone."""
    current = {"d": D1}
    stream = RotatingDbnStream(root=tmp_path, session_date_fn=lambda: current["d"], clock=_Clock())
    stream.write(_live_header())
    stream.write(_mapping("NQZ6 C25000"))
    stream.write(_mapping("NQZ6", instrument_id=43, alias="NQ.FUT"))
    original_quotes = [_quote(D1), _quote(D2), _quote(D2, 43)]
    stream.write(original_quotes[0])
    current["d"] = D2
    for quote in original_quotes[1:]:
        stream.write(quote)
    stream.close()

    assert _symbols(stream.paths[1]) == ["NQZ6 C25000", "NQZ6"]
    import databento
    import databento_dbn as dbn

    retained_quotes = [
        bytes(record) for path in stream.paths
        for record in databento.DBNStore.from_file(path)
        if isinstance(record, dbn.BBOMsg)
    ]
    assert retained_quotes == original_quotes
    assert stream.records_written == 5
    assert stream.mappings_replayed == 2


def test_mapping_cache_preserves_aliases_updates_and_finite_validity(tmp_path: Path) -> None:
    """An ID can carry multiple intervals and input aliases over the week."""
    current = {"d": D1}
    stream = RotatingDbnStream(root=tmp_path, session_date_fn=lambda: current["d"], clock=_Clock())
    stream.write(_live_header())
    mappings = [
        _mapping("NQZ6", end=D2),
        _mapping("NQH7", start=D2),
        _mapping("NQH7", start=D2, alias="NQ.FUT"),
        _mapping("NQZ6 C25000", instrument_id=43, end=D2),
    ]
    for mapping in mappings:
        stream.write(mapping)
    current["d"] = D2
    stream.write(_quote(D2))
    stream.write(_quote(D2, 43))
    stream.close()

    assert _symbols(stream.paths[1]) == ["NQH7", None]
    import databento
    import databento_dbn as dbn

    cached = [bytes(r) for r in databento.DBNStore.from_file(stream.paths[1]) if isinstance(r, dbn.SymbolMappingMsg)]
    assert cached == mappings


def test_reconnect_mapping_duplicates_keep_latest_bytes_and_order(tmp_path: Path) -> None:
    """Reconnect repeats do not multiply the replayed map, or reorder updates."""
    current = {"d": D1}
    stream = RotatingDbnStream(root=tmp_path, session_date_fn=lambda: current["d"], clock=_Clock())
    stream.write(_live_header())
    old = _mapping("NQZ6", event=1)
    changed = _mapping("NQH7", event=2)
    repeated = _mapping("NQZ6", event=3)
    for mapping in (old, changed, repeated):
        stream.write(mapping)
    current["d"] = D2
    stream.write(_quote(D2))
    stream.close()

    import databento
    import databento_dbn as dbn

    cached = [bytes(r) for r in databento.DBNStore.from_file(stream.paths[1]) if isinstance(r, dbn.SymbolMappingMsg)]
    assert cached == [changed, repeated]
    assert _symbols(stream.paths[1]) == ["NQZ6"]
    assert stream.mappings_replayed == 2


def test_market_records_do_not_invoke_mapping_decoder(tmp_path: Path, monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("ordinary quotes must not construct a DBN decoder")

    monkeypatch.setattr("src.acquisition.rotating_stream.DBNDecoder", forbidden)
    stream = RotatingDbnStream(root=tmp_path, session_date_fn=lambda: D1, clock=_Clock())
    stream.write(_live_header())
    stream.write(_quote(D1))
    stream.close()


_ARCHIVE_ROOT = "/opt/vtech/data/sources/raw"
_MIN_RECORDS = 8


def _sized_candidates(pattern: str) -> list[Path]:
    """Readable ``.dbn.zst`` paths matching ``pattern``, smallest first.

    The archive tree contains dangling entries, so each candidate is stat'd
    defensively rather than trusting the glob.
    """
    sized: list[tuple[int, Path]] = []
    for raw in glob.glob(pattern, recursive=True):
        path = Path(raw)
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if 1_000 < size < 3_000_000:
            sized.append((size, path))
    sized.sort()
    return [p for _, p in sized]


def _first_decodable(paths: list[Path], limit: int = 25) -> Path | None:
    """First path that decodes and holds at least ``_MIN_RECORDS`` records."""
    databento = pytest.importorskip("databento")
    for path in paths[:limit]:
        try:
            if sum(1 for _ in databento.DBNStore.from_file(path)) >= _MIN_RECORDS:
                return path
        except Exception:  # noqa: BLE001 — a bad archive file is not a test failure
            continue
    return None


@functools.lru_cache(maxsize=None)
def _archive_sample(kind: str) -> Path | None:
    """A real archive file to replay, or None if this host has no market data.

    ``live`` is a mixed-schema file written by ``add_stream`` itself, which is
    exactly the byte pattern this class must survive in production; ``backfill``
    is a single-schema file with a larger metadata header.

    Resolved lazily and cached: the search globs the whole archive tree and
    decodes candidates, which is far too slow to run at collection time on
    every full-suite run.
    """
    return _first_decodable(_sized_candidates(f"{_ARCHIVE_ROOT}/{kind}/**/*.dbn.zst"))


@pytest.mark.parametrize("split_at", [1, 7])
@pytest.mark.parametrize("kind", ["live", "backfill"])
@pytest.mark.real_data  # Reads a live archive fixture; all outputs stay in tmp_path.
def test_real_dbn_survives_rotation(tmp_path: Path, kind: str, split_at: int) -> None:
    """Replay a real DBN stream the way the client writes it, then rotate.

    Original records remain byte-identical and in order; the new file also
    receives an explicit symbol-mapping prefix for standalone decoding.
    """
    sample = _archive_sample(kind)
    if sample is None:
        pytest.skip(f"no readable .dbn.zst under raw/{kind} on this host")

    databento = pytest.importorskip("databento")
    source = databento.DBNStore.from_file(sample)
    records = [bytes(r) for r in source]
    if len(records) < split_at + 1:
        pytest.skip(f"sample has only {len(records)} records")
    header = source.metadata.encode()

    current = {"d": D1}
    stream = RotatingDbnStream(
        root=tmp_path, session_date_fn=lambda: current["d"], clock=_Clock()
    )
    # Exactly the client's pattern: header once, then one record per write.
    stream.write(header)
    for i, rec in enumerate(records):
        if i == split_at:
            current["d"] = D2
        stream.write(rec)
    stream.close()

    files = sorted(tmp_path.rglob("*.dbn"))
    assert len(files) == 2, "expected one file either side of the rotation"

    decoded: list[bytes] = []
    for i, f in enumerate(files):
        store = databento.DBNStore.from_file(f)
        # Header replay must keep the metadata intact in every file.
        assert store.metadata.dataset == source.metadata.dataset
        assert store.metadata.schema == source.metadata.schema
        file_records = list(store)
        if i:
            replayed = file_records[:stream.mappings_replayed]
            from databento_dbn import SymbolMappingMsg
            assert all(isinstance(r, SymbolMappingMsg) for r in replayed)
            file_records = file_records[stream.mappings_replayed:]
        decoded.extend(bytes(r) for r in file_records)

    assert len(decoded) == len(records)
    assert decoded == records, "records must survive rotation byte-for-byte, in order"
