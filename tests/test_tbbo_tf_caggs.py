"""NQ timeframe cagg helpers (ADR-067)."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# Keep in lockstep with scripts/data/create_tbbo_tf_caggs.py
NQ_OUTRIGHT = r"^NQ[HMUZ][0-9]+$"
_NQ = re.compile(NQ_OUTRIGHT)
CHI = ZoneInfo("America/Chicago")


def test_nq_outright_regex() -> None:
    assert _NQ.match("NQU6")
    assert _NQ.match("NQH7")
    assert _NQ.match("NQM5")
    assert _NQ.match("NQZ26")
    assert not _NQ.match("NQU6-NQZ6")
    assert not _NQ.match("ESU6")
    assert not _NQ.match("MNQH6")
    assert not _NQ.match("NQ.c.0")


def test_clock_5m_left_label() -> None:
    """5m nested cagg is unix-epoch 5-minute buckets (same as 1min origin)."""
    ts = datetime(2026, 8, 25, 13, 34, tzinfo=timezone.utc)  # 08:34 CDT
    epoch = int(ts.timestamp())
    bucket_epoch = epoch - (epoch % 300)
    assert bucket_epoch % 300 == 0
    bucket = datetime.fromtimestamp(bucket_epoch, tz=timezone.utc)
    assert bucket.isoformat() == "2026-08-25T13:30:00+00:00"
    assert bucket.minute % 5 == 0 and bucket.second == 0


def test_globex_session_t_left_label() -> None:
    """Session T = [T-1 17:00, T 17:00) America/Chicago; last trade ≤ T 16:00."""
    summer_open = datetime(2026, 8, 24, 17, 0, tzinfo=CHI)  # T=2026-08-25
    summer_close = datetime(2026, 8, 25, 16, 0, tzinfo=CHI)
    summer_next = datetime(2026, 8, 25, 17, 0, tzinfo=CHI)
    assert summer_open.astimezone(timezone.utc).isoformat() == "2026-08-24T22:00:00+00:00"
    assert summer_close.astimezone(timezone.utc).isoformat() == "2026-08-25T21:00:00+00:00"
    assert summer_next.astimezone(timezone.utc).isoformat() == "2026-08-25T22:00:00+00:00"

    winter_open = datetime(2026, 1, 14, 17, 0, tzinfo=CHI)  # T=2026-01-15
    winter_close = datetime(2026, 1, 15, 16, 0, tzinfo=CHI)
    winter_next = datetime(2026, 1, 15, 17, 0, tzinfo=CHI)
    assert winter_open.astimezone(timezone.utc).isoformat() == "2026-01-14T23:00:00+00:00"
    assert winter_close.astimezone(timezone.utc).isoformat() == "2026-01-15T22:00:00+00:00"
    assert winter_next.astimezone(timezone.utc).isoformat() == "2026-01-15T23:00:00+00:00"
