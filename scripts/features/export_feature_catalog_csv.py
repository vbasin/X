"""Parse docs/reference/FEATURE_CATALOG.md and export all features to CSV.

Produces one row per feature with:
    module_num, prefix, module_name, source, module_file, cadence,
    feature, description, status

Usage:
    python scripts/features/export_feature_catalog_csv.py \
        --catalog docs/reference/FEATURE_CATALOG.md \
        --out data/exports/feature_catalog.csv
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


SECTION_HEADER_RE = re.compile(
    r"^##\s+(?P<num>\d+)\.\s+(?P<name>.+?)\s+—\s+`(?P<prefix>[a-z]+_)`\s+\((?P<count>\d+)\s+features\)"
)
META_RE = re.compile(r"^\*\*(?P<key>[A-Za-z ]+)\*\*:\s*(?P<val>.+?)\s*$")
# Table row: `feature_name` | description [ | status ]
FEATURE_ROW_RE = re.compile(
    r"^\|\s*`(?P<feat>[a-z]{2}_[A-Za-z0-9_]+)`\s*\|\s*(?P<desc>.+?)\s*(?:\|\s*(?P<status>[^|]*?)\s*)?\|\s*$"
)


def parse_catalog(path: Path) -> list[dict]:
    rows: list[dict] = []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    current: dict | None = None
    in_glossary = False

    for line in lines:
        # Section start
        m = SECTION_HEADER_RE.match(line)
        if m:
            current = {
                "module_num": int(m.group("num")),
                "module_name": m.group("name").strip(),
                "prefix": m.group("prefix"),
                "count_declared": int(m.group("count")),
                "source": "",
                "module_file": "",
                "cadence": "",
            }
            in_glossary = False
            continue

        if current is None:
            continue

        # Stop feature parsing once glossary section starts
        if line.strip().startswith("**Glossary**"):
            in_glossary = True
            continue

        # Metadata bullets (Source / Module / Cadence)
        mm = META_RE.match(line.strip())
        if mm:
            key = mm.group("key").strip().lower()
            val = mm.group("val").strip()
            # Strip trailing markdown-space markers used for line breaks
            val = re.sub(r"\s{2,}$", "", val)
            if key == "source":
                current["source"] = val
            elif key == "module":
                # value like `src/features/book_pressure.py`
                current["module_file"] = val.strip("`")
            elif key == "cadence":
                current["cadence"] = val

        # Feature table row (skip when we're inside a glossary)
        if in_glossary:
            continue

        fm = FEATURE_ROW_RE.match(line)
        if fm:
            feat = fm.group("feat")
            # Only accept features whose prefix matches the current section
            if not feat.startswith(current["prefix"]):
                continue
            desc = fm.group("desc").strip()
            status = (fm.group("status") or "").strip()
            rows.append(
                {
                    "module_num": current["module_num"],
                    "prefix": current["prefix"].rstrip("_"),
                    "module_name": current["module_name"],
                    "source": current["source"],
                    "module_file": current["module_file"],
                    "cadence": current["cadence"],
                    "feature": feat,
                    "description": desc,
                    "status": status,
                }
            )

    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", type=Path, default=Path("docs/reference/FEATURE_CATALOG.md"))
    ap.add_argument("--out", type=Path, default=Path("data/exports/feature_catalog.csv"))
    args = ap.parse_args()

    rows = parse_catalog(args.catalog)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "module_num",
        "prefix",
        "module_name",
        "feature",
        "description",
        "status",
        "source",
        "module_file",
        "cadence",
    ]
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    from collections import Counter
    counts = Counter((r["module_num"], r["prefix"]) for r in rows)
    print(f"Wrote {len(rows)} features to {args.out}")
    for (num, pfx), n in sorted(counts.items()):
        print(f"  {num:>2}. {pfx}_ : {n}")


if __name__ == "__main__":
    main()
