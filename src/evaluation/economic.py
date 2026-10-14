"""Canonical Economic Evaluator: filled trades -> cost-explicit PnL.

Status: PARTIAL. Real and tested; deliberately NOT wired into Track B/C
(``docs/MIGRATION_MANIFEST.md``). Pure functions over plain dataclasses -- no
DataFrame, no I/O except the explicit YAML loader, no dependency on any
existing Track A/B/C module.

Execution contract
==================

PnL is computed from **execution prices**, never from a caller-supplied
points figure. Each leg (entry, exit) is priced according to its order type:

MARKET
    Fills at the **executable** side of the book, never the midpoint::

        long  entry -> ask        short entry -> bid
        long  exit  -> bid        short exit  -> ask

    ``slippage_points_per_side`` is then applied *adversely* to that
    executable price. Slippage means ONLY additional adverse movement between
    the executable bid/ask when the order is ready and its eventual fill.

STOP
    ``stop_price`` is a **trigger**, not the fill price. Once triggered a
    plain STOP is a MARKET order: it fills at the executable bid/ask quoted
    at the trigger/execution timestamp, and market slippage applies.
    STOP_LIMIT is not modelled -- nothing in the repo requires it.

LIMIT
    Fill *determination* belongs to the upstream fill logic; this evaluator
    only prices a leg that is already known to have filled. It fills at the
    limit price. A better price is used **only** when the upstream fill logic
    explicitly reports one (``reported_fill_price``); a reported fill worse
    than the limit is rejected as impossible. **No slippage is applied.**

Spread is not slippage
----------------------
Using executable bid/ask prices *is* the spread cost -- a long bought at the
ask and sold at the bid has paid the spread with no further term. Folding
spread into ``slippage_points_per_side`` (as an earlier version of this
module did) double-counts it the moment real quotes are supplied, and makes
the slippage figure uninterpretable. The two are kept separate on purpose.

The current slippage assumption is **0**. It is persisted anyway, so every
result states its own assumption. Latency-driven slippage is not estimated.

Quantity and commission
-----------------------
For one completed trade::

    gross_dollars    = signed_execution_price_change * contract_multiplier * quantity
    total_commission = commission_per_trade * quantity
    net_dollars      = gross_dollars - total_commission

``commission_per_trade`` is ONE combined all-in figure per completed round
trip per unit (broker + exchange + clearing + other fees). It is not split
into categories.

Partial / one-sided fills
-------------------------
Not supported, and not invented. A :class:`Trade` is a *completed* round
trip: both legs filled, same quantity. An entry that never filled is not a
trade and must not be passed in; an entry whose exit never filled is an open
position, which this descriptive evaluator does not mark to market. Commission
is therefore always charged exactly once per completed trade.

Statistical inference remains DEFERRED -- see :class:`TradeLedger`.
"""
from __future__ import annotations

import enum
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


class OrderType(str, enum.Enum):
    """The order's INTENT. Never changed by the fact that its fill is known.

    A resolved STOP stays a STOP and a market entry stays MARKET even when an
    upstream simulator already determined the price. Rewriting a resolved
    MARKET/STOP fill as a LIMIT leg -- which an earlier adapter did, to stop
    slippage being applied twice -- loses the order semantics and makes
    slippage sensitivity silently ineffective.
    """

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class LegRole(str, enum.Enum):
    """Which leg of the trade this is, for per-leg slippage selection.

    A resting target and a swept stop are not the same execution assumption,
    so they are not given the same slippage. ``EXIT_TIME`` is a hold-limit or
    end-of-session flatten: a market order at an arbitrary moment, which gets
    the generic per-side assumption rather than the stop's.
    """

    ENTRY = "ENTRY"
    EXIT_TARGET = "EXIT_TARGET"
    EXIT_STOP = "EXIT_STOP"
    EXIT_TIME = "EXIT_TIME"


#: Where a leg's price came from. Free-text by design, but these are the values
#: this repository produces, and they are persisted with the ledger so a result
#: always states how its fills were obtained.
FILL_SOURCE_UNSPECIFIED = "unspecified"
FILL_SOURCE_PATH_SIM = "path_sim"        # 1s high/low first touch (portfolio_sim)
FILL_SOURCE_QUOTE = "quote"              # priced here from an executable bid/ask
FILL_SOURCE_TBBO = "tbbo"                # Databento top-of-book quote at the fill
FILL_SOURCE_REPORTED = "reported"        # a broker/paper fill report


@dataclass(frozen=True)
class Quote:
    """Top-of-book at one instant. Midpoint is intentionally not offered."""

    bid: float
    ask: float

    def __post_init__(self) -> None:
        if self.ask < self.bid:
            raise ValueError(f"crossed quote: bid={self.bid} > ask={self.ask}")


@dataclass(frozen=True)
class Leg:
    """One side (entry or exit) of a completed trade.

    Attributes:
        order_type: MARKET, LIMIT or STOP.
        quote: Executable book at order-ready time (MARKET) or at the
            trigger/execution timestamp (STOP). Required for both; unused
            for LIMIT.
        limit_price: Required for LIMIT.
        stop_price: Required for STOP. The trigger level -- recorded, but it
            is NOT the fill price.
        reported_fill_price: LIMIT only. Price improvement explicitly
            reported by the upstream fill logic. Never inferred.
        ts: ISO timestamp of the fill (or trigger, for STOP).
    """

    order_type: OrderType
    role: LegRole = LegRole.ENTRY
    quote: Quote | None = None
    limit_price: float | None = None
    stop_price: float | None = None
    #: A fill price already determined upstream -- by a path simulator, a TBBO
    #: replay, or a broker report. Valid for EVERY order type. When present it
    #: is used verbatim and NO slippage is added, because whatever produced it
    #: already accounted for execution. This is the seam a higher-fidelity
    #: execution source plugs into: supply the resolved price and name it in
    #: ``fill_source``; the order type and role stay what they were.
    resolved_fill_price: float | None = None
    fill_source: str = FILL_SOURCE_UNSPECIFIED
    ts: str | None = None

    def __post_init__(self) -> None:
        ot = OrderType(self.order_type)
        object.__setattr__(self, "order_type", ot)
        object.__setattr__(self, "role", LegRole(self.role))
        resolved = self.resolved_fill_price is not None
        # A quote is needed only when this leg must be PRICED here.
        if ot in (OrderType.MARKET, OrderType.STOP) and self.quote is None and not resolved:
            raise ValueError(
                f"{ot.value} leg needs either the executable quote or a "
                f"resolved_fill_price"
            )
        if ot is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("LIMIT leg requires limit_price")
        if ot is OrderType.STOP and self.stop_price is None:
            raise ValueError("STOP leg requires stop_price (the trigger)")
        if resolved and self.fill_source == FILL_SOURCE_UNSPECIFIED:
            raise ValueError(
                "a resolved_fill_price must name its fill_source -- an "
                "unattributed fill price is not auditable"
            )
        if self.role is LegRole.ENTRY:
            pass
        elif ot is OrderType.STOP and self.role is not LegRole.EXIT_STOP:
            raise ValueError("a STOP exit leg must have role EXIT_STOP")


@dataclass(frozen=True)
class EconomicAssumptions:
    """Explicit, persisted execution-cost assumptions for one evaluation.

    Attributes:
        contract_multiplier: Dollars per point per unit quantity.
        commission_per_trade: ONE combined all-in fee, in dollars, per
            completed round trip per unit quantity.
        slippage_points_per_side: Adverse movement beyond the executable
            bid/ask, applied to MARKET and STOP legs only. Currently 0.
        economic_contract_unit: The unit economic exposure is SIZED in --
            the profile name in ``config/execution_costs.yaml``, e.g.
            ``"NQ"`` or ``"MNQ_equivalent"``. Not a traded instrument:
            "MNQ_equivalent" is 0.1x NQ exposure and does NOT imply MNQ market
            data or that an MNQ contract is bought. (Renamed from
            ``instrument``, which wrongly implied exactly that.)
        underlying: The market whose data/prices the trades are computed
            from -- e.g. ``"NQ"``. Always recorded separately.
        fill_model: Free-text statement of the fill assumption, persisted
            with the result.
        label: Scenario key for :func:`evaluate_cost_sensitivity`.
    """

    contract_multiplier: float
    commission_per_trade: float
    slippage_points_per_side: float = 0.0
    economic_contract_unit: str | None = None
    underlying: str | None = None
    fill_model: str = "unspecified"
    label: str = "custom"
    #: Optional per-leg refinement of ``slippage_points_per_side``. A path
    #: simulator does not assume one number for every leg: a resting target is
    #: typically filled at its limit (no slippage) while a triggered stop
    #: sweeps the book. When a value here is None that leg falls back to
    #: ``slippage_points_per_side``, so the coarse single-number model is
    #: unchanged and remains the default. These exist so a simulator's finer
    #: assumption can be expressed HERE rather than kept privately, which is
    #: what made two independent cost models possible.
    entry_slippage_points: float | None = None
    stop_slippage_points: float | None = None
    target_slippage_points: float | None = None

    def __post_init__(self) -> None:
        if self.contract_multiplier <= 0:
            raise ValueError("contract_multiplier must be > 0")
        if self.commission_per_trade < 0:
            raise ValueError("commission_per_trade must be >= 0")
        if self.slippage_points_per_side < 0:
            raise ValueError("slippage_points_per_side must be >= 0")
        for name in ("entry_slippage_points", "stop_slippage_points",
                     "target_slippage_points"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be >= 0 when set")

    #: LegRole -> which per-leg field applies. EXIT_TIME has none: a flatten or
    #: hold-limit exit is a market order at an arbitrary moment, so it gets the
    #: generic per-side assumption rather than the stop's sweep assumption.
    _ROLE_FIELD = {
        "ENTRY": "entry_slippage_points",
        "EXIT_STOP": "stop_slippage_points",
        "EXIT_TARGET": "target_slippage_points",
        "EXIT_TIME": None,
    }

    def slippage_for(self, leg: "LegRole | str") -> float:
        """Slippage in points for one leg role.

        Accepts a :class:`LegRole`, or the short strings ``"entry"``/``"stop"``/
        ``"target"``. Falls back to ``slippage_points_per_side`` for a role with
        no explicit value, so an assumptions object that sets none behaves
        exactly as the single-number model did.
        """
        key = leg.value if isinstance(leg, LegRole) else {
            "entry": "ENTRY", "stop": "EXIT_STOP", "target": "EXIT_TARGET",
        }.get(str(leg), str(leg))
        if key not in self._ROLE_FIELD:
            raise ValueError(f"unknown leg role {leg!r}; have {sorted(self._ROLE_FIELD)}")
        field_name = self._ROLE_FIELD[key]
        specific = None if field_name is None else getattr(self, field_name)
        return float(self.slippage_points_per_side if specific is None else specific)

    def varies_execution_from(self, other: "EconomicAssumptions") -> tuple[str, ...]:
        """Fields that would change the FILLS, not merely their cost.

        Slippage and the fill model determine which price an order gets, so
        varying them cannot be evaluated against a fixed set of already-resolved
        fills -- that needs re-simulation. Multiplier and commission only
        reprice the same fills. :func:`evaluate_cost_sensitivity` uses this to
        refuse a scenario that silently would have no effect.
        """
        changed: list[str] = []
        for field_name in ("slippage_points_per_side", "entry_slippage_points",
                           "stop_slippage_points", "target_slippage_points",
                           "fill_model"):
            if getattr(self, field_name) != getattr(other, field_name):
                changed.append(field_name)
        return tuple(changed)


def zero_cost(
    contract_multiplier: float, *, fill_model: str = "unspecified"
) -> EconomicAssumptions:
    """Explicit zero-commission, zero-slippage scenario for a stated contract.

    A factory, not a constant: an earlier module-level ``ZERO_COST`` hardcoded
    ``contract_multiplier=1.0`` and so silently reported dollars equal to
    points. Zeroing costs must not also silently redefine the contract unit.
    """
    return EconomicAssumptions(
        contract_multiplier=contract_multiplier,
        commission_per_trade=0.0,
        slippage_points_per_side=0.0,
        fill_model=fill_model,
        label="zero_cost",
    )


def load_assumptions(
    path: str | Path, profile: str, *, label: str | None = None,
    fill_model: str = "unspecified",
) -> EconomicAssumptions:
    """Build assumptions from ``config/execution_costs.yaml``.

    A profile may state ``contract_multiplier`` directly, or state
    ``exposure_ratio_to_underlying`` against another profile, in which case
    only the MULTIPLIER is derived from it. Commission is never derived: each
    profile's ``commission_per_trade`` is its own independent assumption
    (MNQ-equivalent is ``$1.03``, not ``0.1 x $3.36``).
    """
    import yaml

    doc = yaml.safe_load(Path(path).read_text()) or {}
    profiles: Mapping[str, Any] = doc.get("profiles") or {}
    if profile not in profiles:
        raise KeyError(f"unknown cost profile {profile!r}; have {sorted(profiles)}")
    p = profiles[profile]

    if "contract_multiplier" in p:
        multiplier = float(p["contract_multiplier"])
    else:
        ref = profiles[p["exposure_relative_to"]]
        multiplier = float(ref["contract_multiplier"]) * float(p["exposure_ratio_to_underlying"])

    return EconomicAssumptions(
        contract_multiplier=multiplier,
        commission_per_trade=float(p["commission_per_trade"]),
        slippage_points_per_side=float(p.get("slippage_points_per_side", 0.0)),
        economic_contract_unit=str(profile),
        underlying=p.get("underlying"),
        fill_model=fill_model,
        label=label or str(profile),
    )


def execution_price(leg: Leg, *, direction: int, is_entry: bool,
                    slippage_points: float) -> float:
    """Price one leg. ``slippage_points`` applies only when pricing here.

    ``buying`` is true for a long entry and a short exit. Buyers lift the ask
    and slippage pushes them higher; sellers hit the bid and slippage pushes
    them lower.

    Precedence:

    1. ``resolved_fill_price`` -- an upstream execution source already
       determined this fill, so it is used verbatim and no slippage is added.
       Whatever produced it (a 1s path simulator, a TBBO replay, a broker
       report) already accounted for execution. The order type is unchanged: a
       resolved STOP is still a STOP.
    2. ``LIMIT`` without a resolved fill -- the limit price. No slippage, ever.
    3. ``MARKET``/``STOP`` without a resolved fill -- the executable side of the
       book, plus ``slippage_points`` adversely.
    """
    if direction not in (1, -1):
        raise ValueError("direction must be +1 (long) or -1 (short)")
    buying = (direction == 1) == is_entry

    if leg.resolved_fill_price is not None:
        fill = float(leg.resolved_fill_price)
        if leg.order_type is OrderType.LIMIT:
            limit = float(leg.limit_price)  # type: ignore[arg-type]
            worse = fill > limit if buying else fill < limit
            if worse:
                raise ValueError(
                    f"resolved LIMIT fill {fill} is worse than limit {limit} -- "
                    f"a limit order cannot fill through its limit. If the fill "
                    f"model genuinely fills through the level, the leg is not a "
                    f"LIMIT: use MARKET (or STOP) with role "
                    f"{leg.role.value}."
                )
        return fill

    if leg.order_type is OrderType.LIMIT:
        return float(leg.limit_price)  # type: ignore[arg-type]

    # MARKET, and STOP once triggered: executable side of the book.
    q = leg.quote
    assert q is not None
    base = q.ask if buying else q.bid
    return base + slippage_points if buying else base - slippage_points


@dataclass(frozen=True)
class Trade:
    """One COMPLETED round trip (both legs filled, same quantity).

    ``entry.ts`` / ``exit.ts`` bound the holding period. They are unused in
    every calculation here -- they are recorded because overlapping holding
    periods are exactly why ``t_stat_iid`` is biased, so persisting them is
    what makes the DEFERRED inference resolvable later.
    """

    trade_id: str
    direction: int
    entry: Leg
    exit: Leg
    quantity: float = 1.0
    sequence: float = 0.0

    def __post_init__(self) -> None:
        if self.direction not in (1, -1):
            raise ValueError("direction must be +1 (long) or -1 (short)")
        if self.quantity <= 0:
            raise ValueError("quantity must be > 0 for a completed trade")


@dataclass(frozen=True)
class TradeResult:
    trade_id: str
    direction: int
    quantity: float
    entry_order_type: str
    exit_order_type: str
    entry_price: float
    exit_price: float
    #: How each price was obtained, so a ledger states its own execution
    #: provenance. exit_role distinguishes a target from a stop from a flatten,
    #: which is what selected the slippage assumption.
    gross_points: float        # signed per-unit execution price change
    gross_dollars: float
    commission: float
    net_dollars: float
    sequence: float
    entry_ts: str | None = None
    exit_ts: str | None = None
    #: How each price was obtained, so a ledger states its own execution
    #: provenance. exit_role distinguishes a target from a stop from a flatten,
    #: which is what selected the slippage assumption. The slippage figures are
    #: what was APPLIED here: 0.0 whenever the fill arrived already resolved.
    entry_fill_source: str = FILL_SOURCE_UNSPECIFIED
    exit_fill_source: str = FILL_SOURCE_UNSPECIFIED
    exit_role: str = "EXIT_TIME"
    entry_slippage_points: float = 0.0
    exit_slippage_points: float = 0.0


@dataclass(frozen=True)
class TradeLedger:
    """Assumptions + per-trade results + descriptive aggregates, in dollars.

    NOTE ON STATISTICAL INFERENCE: ``t_stat_iid`` assumes independent trades.
    This system's trades overlap in time and are positively autocorrelated,
    so n_eff < n_trades and the statistic is BIASED UPWARD. No significance
    verdict is offered. Dependence-aware inference is DEFERRED.
    """

    assumptions: EconomicAssumptions
    trades: tuple[TradeResult, ...]
    n_trades: int
    gross_dollars: float
    total_commission: float
    net_dollars: float
    mean_net_dollars_per_trade: float
    hit_rate: float
    t_stat_iid: float
    max_drawdown_dollars: float


def compute_trade_ledger(
    trades: Sequence[Trade], assumptions: EconomicAssumptions
) -> TradeLedger:
    """Price every leg, net commission, aggregate. Pure; no I/O.

    Slippage is selected PER LEG from the leg's role
    (``EconomicAssumptions.slippage_for``), so ``entry_slippage_points``,
    ``stop_slippage_points`` and ``target_slippage_points`` genuinely
    participate in pricing rather than being recorded and ignored. A leg that
    arrived with a ``resolved_fill_price`` gets none: its execution already
    happened, and the applied figure is recorded as 0.0 so the ledger does not
    imply an assumption it did not use.
    """
    results: list[TradeResult] = []
    for t in sorted(trades, key=lambda t: t.sequence):
        slip_in = 0.0 if t.entry.resolved_fill_price is not None else \
            assumptions.slippage_for(t.entry.role)
        slip_out = 0.0 if t.exit.resolved_fill_price is not None else \
            assumptions.slippage_for(t.exit.role)
        p_in = execution_price(t.entry, direction=t.direction, is_entry=True,
                               slippage_points=slip_in)
        p_out = execution_price(t.exit, direction=t.direction, is_entry=False,
                                slippage_points=slip_out)
        gross_pts = (p_out - p_in) * t.direction
        gross_usd = gross_pts * assumptions.contract_multiplier * t.quantity
        commission = assumptions.commission_per_trade * t.quantity
        results.append(TradeResult(
            trade_id=t.trade_id, direction=t.direction, quantity=t.quantity,
            entry_order_type=t.entry.order_type.value,
            exit_order_type=t.exit.order_type.value,
            entry_price=p_in, exit_price=p_out, gross_points=gross_pts,
            gross_dollars=gross_usd, commission=commission,
            net_dollars=gross_usd - commission, sequence=t.sequence,
            entry_ts=t.entry.ts, exit_ts=t.exit.ts,
            entry_fill_source=t.entry.fill_source,
            exit_fill_source=t.exit.fill_source,
            exit_role=t.exit.role.value,
            entry_slippage_points=slip_in, exit_slippage_points=slip_out,
        ))

    n = len(results)
    if n == 0:
        nan = float("nan")
        return TradeLedger(assumptions, (), 0, 0.0, 0.0, 0.0, nan, nan, nan, 0.0)

    net = [r.net_dollars for r in results]
    mean = sum(net) / n
    if n >= 2:
        std = math.sqrt(sum((x - mean) ** 2 for x in net) / (n - 1))
        t_iid = mean / (std / math.sqrt(n)) if std > 0 else float("nan")
    else:
        t_iid = float("nan")

    cum = peak = max_dd = 0.0
    for x in net:
        cum += x
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)

    return TradeLedger(
        assumptions=assumptions, trades=tuple(results), n_trades=n,
        gross_dollars=sum(r.gross_dollars for r in results),
        total_commission=sum(r.commission for r in results),
        net_dollars=sum(net), mean_net_dollars_per_trade=mean,
        hit_rate=sum(1 for x in net if x > 0) / n,
        t_stat_iid=t_iid, max_drawdown_dollars=max_dd,
    )


def _all_fills_resolved(trades: Sequence[Trade]) -> bool:
    return bool(trades) and all(
        t.entry.resolved_fill_price is not None and t.exit.resolved_fill_price is not None
        for t in trades
    )


def evaluate_cost_sensitivity(
    trades: Sequence[Trade],
    scenarios: Sequence[EconomicAssumptions],
    *,
    strict: bool = True,
) -> Mapping[str, TradeLedger]:
    """FIXED-FILL sensitivity: the same fills, repriced.

    This answers "what would this book have earned at a different commission or
    contract size". The fills do not move, so only ``contract_multiplier`` and
    ``commission_per_trade`` may differ between scenarios.

    Varying a slippage term here is a mistake, not a variation: when every fill
    is already resolved, a different slippage assumption cannot change any
    price, so the scenario would return an identical ledger under a label that
    claims otherwise. With ``strict`` (the default) that raises and names the
    offending fields; use :func:`evaluate_execution_sensitivity`, which
    re-derives the fills, for that question.

    Duplicate labels raise rather than overwrite.
    """
    out: dict[str, TradeLedger] = {}
    baseline = scenarios[0] if scenarios else None
    fixed = _all_fills_resolved(trades)
    for scenario in scenarios:
        if scenario.label in out:
            raise ValueError(f"duplicate scenario label: {scenario.label!r}")
        if strict and fixed and baseline is not None:
            moved = scenario.varies_execution_from(baseline)
            if moved:
                raise ValueError(
                    f"scenario {scenario.label!r} varies {list(moved)}, which "
                    f"determines WHICH price an order gets -- but every fill in "
                    f"this book is already resolved, so the change cannot take "
                    f"effect and the ledger would be identical. Use "
                    f"evaluate_execution_sensitivity() to re-derive the fills, "
                    f"or pass strict=False if an identical ledger is genuinely "
                    f"what you want."
                )
        out[scenario.label] = compute_trade_ledger(trades, scenario)
    return out


def evaluate_execution_sensitivity(
    scenarios: Sequence[EconomicAssumptions],
    *,
    resimulate: "Callable[[EconomicAssumptions], Sequence[Trade]]",
) -> Mapping[str, TradeLedger]:
    """EXECUTION sensitivity: the fills themselves may change.

    Slippage and the fill model decide which price an order gets, and a
    different entry price changes which stops and targets are reached, so the
    book must be re-derived rather than repriced. ``resimulate`` is that
    re-derivation: given one set of assumptions it returns the trades that
    actually occur under them.

    ``src/backtest/portfolio_sim.py`` supplies it by re-running the path
    simulation. A higher-fidelity execution source (Databento TBBO
    top-of-book at each decision) plugs in at exactly the same point: it
    returns legs carrying ``quote`` and/or ``resolved_fill_price`` with
    ``fill_source=FILL_SOURCE_TBBO``, and nothing else in this module changes.
    """
    out: dict[str, TradeLedger] = {}
    for scenario in scenarios:
        if scenario.label in out:
            raise ValueError(f"duplicate scenario label: {scenario.label!r}")
        out[scenario.label] = compute_trade_ledger(resimulate(scenario), scenario)
    return out


#: Where the cost profiles live. A run records which profile it used, not a
#: copy of the file.
DEFAULT_COST_CONFIG = Path("config/execution_costs.yaml")


def _slippage_points_from_model(model: Any) -> tuple[float | None, str | None]:
    """Points per side implied by an objective's ``slippage_model`` string.

    ``fixed_1pt`` -> 1.0. Anything unrecognised returns ``(None, reason)``
    rather than 0.0: silently reading an unparsed model as zero slippage is
    the confirmed ``slippage_model`` failure mode (ADR-136), and an unknown
    assumption must stay visibly unknown.
    """
    text = str(model or "").strip().lower()
    if not text or text in ("none", "null"):
        return 0.0, None
    match = re.fullmatch(r"fixed_(\d+(?:\.\d+)?)(pt|pts|point|points)", text)
    if match:
        return float(match.group(1)), None
    return None, f"unrecognised slippage_model {model!r}"


def assumptions_record(
    objective: Mapping[str, Any] | None,
    *,
    costs_path: str | Path = DEFAULT_COST_CONFIG,
    applied_commission_per_trade: float | None = None,
    applied_slippage_points_per_side: float | None = None,
    pnl_unit: str | None = None,
) -> dict[str, Any]:
    """``EconomicAssumptions``-shaped record of what one run actually assumed.

    This is the single place a run's ``execution_assumptions`` is built, so a
    manifest never carries a second, differently-derived cost story. It returns
    a plain mapping rather than an :class:`EconomicAssumptions`, because a run
    may truthfully have applied NO contract multiplier at all (a points-only
    evaluation), which the dataclass correctly refuses to represent.

    Two sources, in order:

    * ``costs.execution_cost_profile`` names a profile in
      ``config/execution_costs.yaml`` -- the authoritative, fully specified
      case, resolved through :func:`load_assumptions`.
    * ``costs.cost_per_trade`` / ``costs.slippage_model`` -- the older inline
      form. ``slippage_model`` is recorded as ``None`` when it cannot be
      parsed, never as 0.0.

    ``applied_*`` are what the run DEMONSTRABLY used (e.g. the
    ``cost_per_trade`` echoed in a candidate's own pnl block). When an applied
    value differs from the declared one, BOTH are recorded and
    ``declared_not_applied`` names the divergence. An objective that declares a
    cost its evaluator never translated is exactly the failure this field
    exists to expose, so it must not be flattened to one number.
    """
    costs = dict((objective or {}).get("costs") or {})
    record: dict[str, Any] = {
        "contract_multiplier": None,
        "commission_per_trade": None,
        "slippage_points_per_side": None,
        "economic_contract_unit": None,
        "underlying": None,
        "fill_model": "unspecified",
        "label": "unspecified",
        "source": None,
        "notes": [],
    }

    profile = costs.get("execution_cost_profile")
    if profile:
        try:
            resolved = load_assumptions(costs_path, str(profile))
        except (OSError, KeyError, ValueError) as exc:
            record["notes"].append(
                f"cost profile {profile!r} declared but not resolvable: {exc}")
            record["label"] = str(profile)
            record["source"] = f"{costs_path}:{profile} (unresolved)"
        else:
            record.update({
                "contract_multiplier": resolved.contract_multiplier,
                "commission_per_trade": resolved.commission_per_trade,
                "slippage_points_per_side": resolved.slippage_points_per_side,
                "economic_contract_unit": resolved.economic_contract_unit,
                "underlying": resolved.underlying,
                "fill_model": resolved.fill_model,
                "label": resolved.label,
                "source": f"{costs_path}:{profile}",
            })
    elif "cost_per_trade" in costs or "slippage_model" in costs:
        slippage, why = _slippage_points_from_model(costs.get("slippage_model"))
        record.update({
            "commission_per_trade": (float(costs["cost_per_trade"])
                                     if costs.get("cost_per_trade") is not None else None),
            "slippage_points_per_side": slippage,
            "label": "objective_inline",
            "source": "objective costs block",
        })
        if why:
            record["notes"].append(why)
        record["notes"].append(
            "inline costs declare no contract multiplier, so this run's PnL is "
            "in the unit its evaluator used, not dollars")
    else:
        record["notes"].append("objective declares no costs block")

    declared_not_applied: dict[str, Any] = {}
    if applied_commission_per_trade is not None:
        if (record["commission_per_trade"] is not None
                and float(record["commission_per_trade"]) != float(applied_commission_per_trade)):
            declared_not_applied["commission_per_trade"] = record["commission_per_trade"]
        record["commission_per_trade"] = float(applied_commission_per_trade)
    if applied_slippage_points_per_side is not None:
        if (record["slippage_points_per_side"] is not None
                and float(record["slippage_points_per_side"])
                != float(applied_slippage_points_per_side)):
            declared_not_applied["slippage_points_per_side"] = record["slippage_points_per_side"]
        record["slippage_points_per_side"] = float(applied_slippage_points_per_side)
    if declared_not_applied:
        record["declared_not_applied"] = declared_not_applied
        record["notes"].append(
            "the objective declared a cost the run did not apply; both are kept "
            "so the divergence is visible rather than resolved")
    if pnl_unit:
        record["pnl_unit"] = str(pnl_unit)
    return record


# ══════════════════════════════════════════════════════════════════════════════
# The research workflow's economic artifacts
# ══════════════════════════════════════════════════════════════════════════════
#
# Every research run's economics comes through here, so future Track B and
# Track C runs do not each carry their own PnL/commission/multiplier arithmetic:
#
#   OOF / decision-policy output -> trade & path logic (which trades happen)
#     -> Trade objects -> this module -> trade_ledger.parquet
#                                       economic_metrics.json
#                                       cost_sensitivity.json
#
# The simulator/evaluator still decides WHICH trades happen and at what prices.
# This decides what they are worth.

def trades_from_book(
    book: "Any",
    *,
    entry_price_col: str,
    exit_price_col: str,
    side_col: str = "side",
    reached_col: str | None = None,
    stop_price_col: str | None = None,
    target_price_col: str | None = None,
    ts_col: str = "ts",
    exit_ts_col: str | None = None,
    quantity: float = 1.0,
    fill_source: str = FILL_SOURCE_PATH_SIM,
) -> list[Trade]:
    """Canonical ``Trade`` objects from a research book (one row per trade).

    Order intent is taken from HOW each trade ended, not asserted:

    * reached its target -> ``LIMIT`` at the target level (a resting exit);
    * otherwise, with a stop level -> ``STOP`` at that level (the trigger);
    * otherwise -> ``MARKET`` with role ``EXIT_TIME``.

    Every leg carries the book's own price as ``resolved_fill_price``, so this
    module prices it verbatim and adds no slippage -- whatever produced the book
    already resolved execution. That is what keeps a book's PnL unchanged while
    still recording true order semantics.
    """
    trades: list[Trade] = []
    rows = book.to_dict("records") if hasattr(book, "to_dict") else list(book)
    for i, row in enumerate(rows):
        side = row.get(side_col)
        direction = 1 if (side in (1, "1", "up", "long", True)) else -1
        entry_px = float(row[entry_price_col])
        exit_px = float(row[exit_price_col])
        reached = bool(row.get(reached_col)) if reached_col else None
        if reached and target_price_col and row.get(target_price_col) is not None:
            exit_leg = Leg(order_type=OrderType.LIMIT, role=LegRole.EXIT_TARGET,
                           limit_price=float(row[target_price_col]),
                           resolved_fill_price=exit_px, fill_source=fill_source,
                           ts=str(row.get(exit_ts_col or ts_col) or ""))
        elif stop_price_col and row.get(stop_price_col) is not None and reached is False:
            exit_leg = Leg(order_type=OrderType.STOP, role=LegRole.EXIT_STOP,
                           stop_price=float(row[stop_price_col]),
                           resolved_fill_price=exit_px, fill_source=fill_source,
                           ts=str(row.get(exit_ts_col or ts_col) or ""))
        else:
            exit_leg = Leg(order_type=OrderType.MARKET, role=LegRole.EXIT_TIME,
                           resolved_fill_price=exit_px, fill_source=fill_source,
                           ts=str(row.get(exit_ts_col or ts_col) or ""))
        trades.append(Trade(
            trade_id=str(row.get(ts_col) or i), direction=direction,
            quantity=float(row.get("contracts") or quantity), sequence=float(i),
            entry=Leg(order_type=OrderType.MARKET, role=LegRole.ENTRY,
                      resolved_fill_price=entry_px, fill_source=fill_source,
                      ts=str(row.get(ts_col) or "")),
            exit=exit_leg,
        ))
    return trades


def ledger_to_frame(ledger: TradeLedger) -> "Any":
    """``trade_ledger.parquet``'s rows: one per priced trade, with provenance."""
    import pandas as pd

    return pd.DataFrame([{
        "trade_id": r.trade_id, "direction": r.direction, "quantity": r.quantity,
        "entry_order_type": r.entry_order_type, "exit_order_type": r.exit_order_type,
        "exit_role": r.exit_role,
        "entry_price": r.entry_price, "exit_price": r.exit_price,
        "gross_points": r.gross_points, "gross_dollars": r.gross_dollars,
        "commission": r.commission, "net_dollars": r.net_dollars,
        "entry_ts": r.entry_ts, "exit_ts": r.exit_ts,
        "entry_fill_source": r.entry_fill_source,
        "exit_fill_source": r.exit_fill_source,
        "entry_slippage_points": r.entry_slippage_points,
        "exit_slippage_points": r.exit_slippage_points,
        "sequence": r.sequence,
    } for r in ledger.trades])


def ledger_summary(ledger: TradeLedger) -> dict[str, Any]:
    """``economic_metrics.json``: the aggregates, never separated from their assumptions."""
    from dataclasses import asdict

    return {
        "assumptions": asdict(ledger.assumptions),
        "n_trades": ledger.n_trades,
        "gross_dollars": ledger.gross_dollars,
        "total_commission": ledger.total_commission,
        "net_dollars": ledger.net_dollars,
        "mean_net_dollars_per_trade": ledger.mean_net_dollars_per_trade,
        "hit_rate": ledger.hit_rate,
        "max_drawdown_dollars": ledger.max_drawdown_dollars,
        "t_stat_iid": ledger.t_stat_iid,
        "inference_note": (
            "t_stat_iid assumes independent trades and is BIASED UPWARD for an "
            "overlapping book. No significance verdict is offered."
        ),
    }


def write_research_economics(
    model_dir: "Any",
    *,
    trades: Sequence[Trade],
    assumptions: EconomicAssumptions,
    scenarios: Sequence[EconomicAssumptions] = (),
) -> dict[str, Any]:
    """Write the three economic artifacts for one model, from one ledger.

    ``trade_ledger.parquet`` + ``economic_metrics.json`` always;
    ``cost_sensitivity.json`` when ``scenarios`` are given. Cost sensitivity here
    is FIXED-FILL by construction: it reprices these trades. A scenario that
    varies slippage is refused by :func:`evaluate_cost_sensitivity`, because with
    resolved fills it cannot take effect -- that question is
    :func:`evaluate_execution_sensitivity`, which needs the simulator.

    Returns the summary it wrote, so a caller can log or assert on it.
    """
    import json
    from dataclasses import asdict
    from pathlib import Path

    d = Path(model_dir)
    d.mkdir(parents=True, exist_ok=True)
    ledger = compute_trade_ledger(trades, assumptions)
    ledger_to_frame(ledger).to_parquet(
        d / "trade_ledger.parquet", index=False, compression="zstd")
    summary = ledger_summary(ledger)
    (d / "economic_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    if scenarios:
        graded = evaluate_cost_sensitivity(list(trades), list(scenarios))
        (d / "cost_sensitivity.json").write_text(json.dumps({
            "kind": "fixed_fill",
            "note": ("the same fills repriced. Varying slippage requires "
                     "re-simulation (evaluate_execution_sensitivity), not this."),
            "scenarios": {label: {
                "assumptions": asdict(lg.assumptions),
                "n_trades": lg.n_trades,
                "net_dollars": lg.net_dollars,
                "total_commission": lg.total_commission,
                "mean_net_dollars_per_trade": lg.mean_net_dollars_per_trade,
            } for label, lg in graded.items()},
        }, indent=2, sort_keys=True, default=str) + "\n")
    return summary
