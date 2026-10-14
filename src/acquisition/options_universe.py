"""NQ option parents and the live five-chain expiry universe.

Historical parent list
----------------------
``NQ_OPTION_PARENTS`` remains the complete Databento parent tuple used by
backfill / definition repair. It is **not** the live BBO subscription set.

Live subscription universe (ADR-034)
------------------------------------
Each Chicago session streams exactly five expiries:

* ``0dte`` — expires on the session date
* ``1dte`` — expires on the next CME trading day
* ``qne1`` / ``qne2`` — next two QNE month-end expiries *after* the session date
* ``qtr`` — closest NQ quarterly (3rd Friday of Mar/Jun/Sep/Dec) on or after
  the session date

Strike lists come from Databento ``symbology.resolve`` (free metadata), never
from the local ``definitions`` table. Live BBO/definition subscriptions use
``stype_in="raw_symbol"`` so other months, weeklies, and ``UD:`` combos never
hit the wire or the archive.
"""
from __future__ import annotations

import calendar
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

from src.common.cme_holidays import is_trading_day

logger = logging.getLogger("vtech.options_universe")

_CT = ZoneInfo("America/Chicago")

NQ_QUARTERLY_OPTION_PARENT = "NQ.OPT"

NQ_WEEKLY_OPTION_PARENTS = tuple(
    f"Q{week}{weekday}.OPT"
    for week in range(1, 6)
    for weekday in "ABCD"
) + tuple(f"QN{week}.OPT" for week in range(1, 5))

NQ_END_OF_MONTH_OPTION_PARENT = "QNE.OPT"

NQ_OPTION_PARENTS = (
    NQ_QUARTERLY_OPTION_PARENT,
    *NQ_WEEKLY_OPTION_PARENTS,
    NQ_END_OF_MONTH_OPTION_PARENT,
)

LIVE_OPTION_CATEGORIES = ("0dte", "1dte", "qne1", "qne2", "qtr")
MONTH_CODE = "FGHJKMNQUVXZ"
WEEKDAY_CODE = {0: "A", 1: "B", 2: "C", 3: "D"}
_QUARTERLY_MONTHS = frozenset({3, 6, 9, 12})
# CME equity-index sessions close at 16:00 CT and the next one opens at 17:00.
# The session date advances a few minutes *after* the close, for two reasons:
#   - the 16:00 CT minute carries the closing snapshot (one final quote per
#     instrument, ~1,300 rows), which belongs to the session that is ending;
#   - 16:01-16:44 CT is measurably dead on bbo_1s, so a boundary taken inside
#     it costs no data, where the old Chicago-midnight boundary cost 10-50s of
#     live quotes four times a week.
# With the refresh running every 300s the roll is detected between 16:05 and
# 16:10 CT, comfortably inside that window.
_SESSION_ROLL_MINUTE_CT = 16 * 60 + 5
_OUTRIGHT_OPTION_RE = re.compile(r"[A-Z0-9]{4,6} [CP]\d+")


# ── calendar ────────────────────────────────────────────────────────

def month_end_business(year: int, month: int) -> date:
    """Last weekday of the month — QNE (end-of-month) expiry."""
    day = date(year, month, calendar.monthrange(year, month)[1])
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def third_friday(year: int, month: int) -> date:
    fridays = [
        day for day in range(1, calendar.monthrange(year, month)[1] + 1)
        if date(year, month, day).weekday() == 4
    ]
    return date(year, month, fridays[2])


def next_trading_day(day: date) -> date:
    nxt = day + timedelta(days=1)
    while not is_trading_day(nxt):
        nxt += timedelta(days=1)
    return nxt


def globex_session_date(now: datetime | None = None) -> date:
    """Trading day that the current (or next) Globex session settles into.

    A CME equity-index session runs 17:00 CT to 16:00 CT the following day and
    is named for the day it *ends*. So the session date advances at the 16:00 CT
    close, not at Chicago midnight:

    * Mon 09:00 CT — inside the session that settles Mon 16:00 → ``Monday``
    * Mon 16:00 CT — the closing snapshot, still the ending session → ``Monday``
    * Mon 16:30 CT — the halt; the next session settles Tue 16:00 → ``Tuesday``
    * Mon 18:00 CT — inside that session → ``Tuesday``
    * Sun 18:00 CT — the week's first session settles Mon 16:00 → ``Monday``

    Keying on the session rather than the calendar date is what puts ``0dte`` on
    the chain the open session actually settles into. Under a calendar-date rule
    the 17:00–24:00 CT window still named the chain that expired at 15:00 CT
    that afternoon, and the chain expiring at the end of the *open* session was
    only subscribed as ``1dte``.

    The roll also lands inside the 16:01–16:44 CT window, where no quote is
    written at all, so a session boundary taken here costs no data.
    """
    ct = (now or datetime.now(timezone.utc)).astimezone(_CT)
    day = ct.date()
    if ct.hour * 60 + ct.minute >= _SESSION_ROLL_MINUTE_CT:
        # Past the close: the session being opened settles on the next
        # trading day, whatever today was.
        return next_trading_day(day)
    if is_trading_day(day):
        return day
    # Weekend or holiday before the close: the next session to open is the
    # next trading day's, so Sunday evening is already on Monday's set.
    return next_trading_day(day)


def live_option_session_date(now: datetime | None = None) -> date:
    """Session date used to pick the five chains and name the archive folder.

    Thin delegate to :func:`globex_session_date` so there is one implementation
    of the session calendar. Kept as a separate name because the streamer, the
    archive layout and the feature loader all refer to it.
    """
    return globex_session_date(now)


def target_roots(session_date: date) -> frozenset[str]:
    """CME roots of the five chains for ``session_date``.

    The membership test every subscription decision is gated on: a symbol is in
    scope only if its leading token is one of these. Deduplicated, because two
    categories can resolve to the same chain — on the session before a month
    end, ``1dte`` and ``qne1`` are the same root — so the set holds three to
    five entries, never a fixed five.
    """
    return frozenset(t.root for t in session_option_targets(session_date))


def root_for(expiry: date) -> str:
    """CME contract root of the chain expiring on ``expiry``.

    QNE (month-end) wins over the weekly series: on a month-end date the daily
    expiry *is* the QNE chain (2026-04-30 lists ``QNEJ6``, not ``Q5DJ6``).
    A quarterly 3rd Friday lists under ``NQ`` (e.g. ``NQU6``), not the Friday
    weekly root.
    """
    suffix = MONTH_CODE[expiry.month - 1] + str(expiry.year % 10)
    if expiry == month_end_business(expiry.year, expiry.month):
        return "QNE" + suffix
    if expiry.month in _QUARTERLY_MONTHS and expiry == third_friday(expiry.year, expiry.month):
        return "NQ" + suffix
    nth = (expiry.day - 1) // 7 + 1
    if expiry.weekday() == 4:
        return f"QN{nth}{suffix}"
    return f"Q{nth}{WEEKDAY_CODE[expiry.weekday()]}{suffix}"


def parent_for(root: str) -> str:
    """Databento parent smart symbol whose resolution contains ``root``."""
    if root.startswith("QNE"):
        return NQ_END_OF_MONTH_OPTION_PARENT
    if root.startswith("NQ"):
        return NQ_QUARTERLY_OPTION_PARENT
    if root.startswith("QN"):
        return f"QN{root[2]}.OPT"
    return f"{root[:3]}.OPT"


def _next_month_ends_after(day: date, n: int = 2) -> list[date]:
    """Next ``n`` month-end business days strictly after ``day``."""
    out: list[date] = []
    probe = day.replace(day=1)
    while len(out) < n:
        end = month_end_business(probe.year, probe.month)
        if end > day:
            out.append(end)
        probe = (probe + timedelta(days=32)).replace(day=1)
    return out


def _closest_quarterly_on_or_after(day: date) -> date:
    """First Mar/Jun/Sep/Dec 3rd Friday on or after ``day``."""
    year, month = day.year, day.month
    for _ in range(9):
        if month in _QUARTERLY_MONTHS:
            expiry = third_friday(year, month)
            if expiry >= day:
                return expiry
        month += 1
        if month > 12:
            month = 1
            year += 1
    raise RuntimeError(f"no quarterly expiry on or after {day}")


# ── targets + resolve ───────────────────────────────────────────────

@dataclass(frozen=True)
class OptionChainTarget:
    category: str
    expiry: date
    root: str
    parent: str


@dataclass(frozen=True)
class ResolvedOptionUniverse:
    session_date: date
    targets: tuple[OptionChainTarget, ...]
    symbols: tuple[str, ...]
    by_category: dict[str, tuple[str, ...]] = field(default_factory=dict)
    instrument_ids: dict[str, int] = field(default_factory=dict)
    missing: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.missing

    @property
    def parents(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(t.parent for t in self.targets))


def session_option_targets(session_date: date) -> tuple[OptionChainTarget, ...]:
    """Calendar identity of the five live chains for ``session_date``."""
    qne1, qne2 = _next_month_ends_after(session_date, 2)
    expiries = {
        "0dte": session_date,
        "1dte": next_trading_day(session_date),
        "qne1": qne1,
        "qne2": qne2,
        "qtr": _closest_quarterly_on_or_after(session_date),
    }
    targets = []
    for category in LIVE_OPTION_CATEGORIES:
        expiry = expiries[category]
        root = root_for(expiry)
        targets.append(OptionChainTarget(
            category=category,
            expiry=expiry,
            root=root,
            parent=parent_for(root),
        ))
    return tuple(targets)


ParentResolveFn = Callable[[date, str], Mapping[str, Mapping[str, int]]]


def _resolve_parent_via_historical(day: date, parent: str, client: Any) -> dict[str, dict[str, int]]:
    """Free ``symbology.resolve`` → ``{root: {raw_symbol: instrument_id}}``."""
    from src.acquisition.schemas import DATASET

    try:
        result = client.symbology.resolve(
            dataset=DATASET,
            symbols=[parent],
            stype_in="parent",
            stype_out="instrument_id",
            start_date=day.isoformat(),
            end_date=(day + timedelta(days=1)).isoformat(),
        )["result"]
    except Exception as exc:  # noqa: BLE001 — vendor 422 = no listing that day
        logger.info("no listing for %s on %s (%s)", parent, day, str(exc)[:80])
        return {}

    by_root: dict[str, dict[str, int]] = {}
    for symbol, intervals in (result or {}).items():
        if not _OUTRIGHT_OPTION_RE.fullmatch(str(symbol)) or not intervals:
            continue
        root = str(symbol).split(" ", 1)[0]
        by_root.setdefault(root, {})[str(symbol)] = int(intervals[0]["s"])
    return by_root


def resolve_session_option_universe(
    session_date: date | None = None,
    *,
    client: Any | None = None,
    resolve_parent: ParentResolveFn | None = None,
    now: datetime | None = None,
) -> ResolvedOptionUniverse:
    """Exact raw-symbol set for the five live chains.

    ``resolve_parent(day, parent)`` is injectable for tests. The default is a
    Databento Historical ``symbology.resolve`` call (metadata, not a download).
    """
    day = session_date or live_option_session_date(now)
    targets = session_option_targets(day)

    own_client = False
    if resolve_parent is None:
        if client is None:
            import databento as db
            client = db.Historical()
            own_client = True

        def resolve_parent(d: date, parent: str, _c=client) -> Mapping[str, Mapping[str, int]]:
            return _resolve_parent_via_historical(d, parent, _c)

    try:
        parent_cache: dict[str, Mapping[str, Mapping[str, int]]] = {}
        by_category: dict[str, tuple[str, ...]] = {}
        instrument_ids: dict[str, int] = {}
        missing: list[str] = []

        for target in targets:
            if target.parent not in parent_cache:
                parent_cache[target.parent] = resolve_parent(day, target.parent)
            chain = dict(parent_cache[target.parent].get(target.root, {}))
            if not chain:
                missing.append(target.category)
                by_category[target.category] = ()
                continue
            by_category[target.category] = tuple(sorted(chain))
            instrument_ids.update(chain)

        symbols = tuple(sorted(instrument_ids))
        if missing:
            logger.warning(
                "live_option_universe_incomplete",
                extra={"session_date": str(day), "missing": missing, "n_symbols": len(symbols)},
            )
        else:
            logger.info(
                "live_option_universe",
                extra={
                    "session_date": str(day),
                    "n_symbols": len(symbols),
                    "roots": {t.category: t.root for t in targets},
                },
            )
        return ResolvedOptionUniverse(
            session_date=day,
            targets=targets,
            symbols=symbols,
            by_category=by_category,
            instrument_ids=instrument_ids,
            missing=tuple(missing),
        )
    finally:
        if own_client and client is not None:
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass
