"""``python -m src.serving.promote`` entry (subcommands via promote module)."""

from __future__ import annotations

import sys


def main() -> int:
    # ``python -m src.serving.promote`` is preferred; this allows
    # ``python -m src.serving`` with promote as default if argv[1]==promote
    if len(sys.argv) > 1 and sys.argv[1] == "promote":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
    from src.serving.promote import promote_cli
    return promote_cli()


if __name__ == "__main__":
    raise SystemExit(main())
