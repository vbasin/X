#!/usr/bin/env python3
"""rd_dashboard.py — Research Director funnel + cost dashboard (W10 #20).

Read-only observability over the research journal + ledgers. Prints:
  - the idea funnel (counts per lifecycle state, via the funnel_v view),
  - run outcomes (succeeded / disproven / needs-refinement / inconclusive / failed),
  - LLM + GPU spend over the last 7 / 30 days (gpu_budget_ledger),
  - ideas awaiting promotion (STAGED_PENDING_PROMOTION),
  - NEEDS_REFINEMENT ideas with their tweaks.

No GPU contacted; pure SQL over the existing tables. Reuses common.db.get_pool.

Usage:
  python scripts/rd_dashboard.py [--program nq-50pt-08-15-cst]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from common.db import get_pool  # noqa: E402

_SCHEMA = "research_director"


def _rows(pool, sql: str, params: tuple = ()) -> list[tuple]:
    with pool.connection() as conn:
        return conn.execute(sql, params).fetchall()


def _print_table(title: str, header: tuple[str, ...], rows: list[tuple]) -> None:
    print(f"\n=== {title} ===")
    if not rows:
        print("  (none)")
        return
    widths = [len(h) for h in header]
    srows = [[str(c) for c in r] for r in rows]
    for r in srows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))
    print("  " + "  ".join(h.ljust(widths[i]) for i, h in enumerate(header)))
    print("  " + "  ".join("-" * widths[i] for i in range(len(header))))
    for r in srows:
        print("  " + "  ".join(c.ljust(widths[i]) for i, c in enumerate(r)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Research Director funnel + cost dashboard")
    parser.add_argument("--program", default=None, help="filter to a program_id")
    args = parser.parse_args(argv)
    pool = get_pool()

    prog_clause = "WHERE program_id = %s" if args.program else ""
    prog_params: tuple = (args.program,) if args.program else ()

    _print_table(
        "Idea funnel (state -> count)",
        ("program_id", "state", "n", "n_7d", "last_update"),
        _rows(pool, f"SELECT program_id, state, n, n_7d, last_update FROM {_SCHEMA}.funnel_v "
                    f"{prog_clause} ORDER BY n DESC", prog_params),
    )

    _print_table(
        "Run outcomes (terminal states)",
        ("state", "n"),
        _rows(pool, f"""
            SELECT state, count(*) FROM {_SCHEMA}.research_journal
            WHERE state IN ('SUCCEEDED_CLOSED','PROMOTED','DISPROVEN_CLOSED',
                            'NEEDS_REFINEMENT','INCONCLUSIVE','FAILED_RUN','NO_OP')
            {('AND program_id = %s' if args.program else '')}
            GROUP BY state ORDER BY count(*) DESC
        """, prog_params),
    )

    _print_table(
        "Spend by category (USD)",
        ("category", "usd_7d", "usd_30d"),
        _rows(pool, f"""
            SELECT category,
                   round(SUM(usd_debited) FILTER (WHERE ts > now() - interval '7 days')::numeric, 4),
                   round(SUM(usd_debited) FILTER (WHERE ts > now() - interval '30 days')::numeric, 4)
            FROM {_SCHEMA}.gpu_budget_ledger
            GROUP BY category ORDER BY category
        """),
    )

    _print_table(
        "LLM calls by agent (7d)",
        ("agent", "calls", "usd"),
        _rows(pool, f"""
            SELECT agent, count(*), round(COALESCE(SUM(usd_cost),0)::numeric, 4)
            FROM {_SCHEMA}.llm_call
            WHERE ts > now() - interval '7 days'
            GROUP BY agent ORDER BY count(*) DESC
        """),
    )

    _print_table(
        "Awaiting promotion (STAGED_PENDING_PROMOTION)",
        ("experiment_id", "scope", "title"),
        _rows(pool, f"""
            SELECT experiment_id, scope, left(title, 60)
            FROM {_SCHEMA}.research_journal
            WHERE state = 'STAGED_PENDING_PROMOTION'
            {('AND program_id = %s' if args.program else '')}
            ORDER BY updated_at DESC
        """, prog_params),
    )

    _print_table(
        "Needs refinement (with tweaks)",
        ("experiment_id", "title", "tweaks"),
        _rows(pool, f"""
            SELECT experiment_id, left(title, 50), COALESCE(tweaks::text, '')
            FROM {_SCHEMA}.research_journal
            WHERE state = 'NEEDS_REFINEMENT'
            {('AND program_id = %s' if args.program else '')}
            ORDER BY updated_at DESC LIMIT 20
        """, prog_params),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
