from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_ROOT = ROOT / "results" / "summaries"
RAW_ROOT = ROOT / "results" / "raw"
GENERATED_ROOT = ROOT / "results" / "generated"


def describe_csv(path: Path) -> None:
    df = pd.read_csv(path)
    print(f"- {path.relative_to(ROOT)}: {len(df):,} rows × {len(df.columns)} columns")
    print(f"  columns: {', '.join(df.columns[:12])}{' ...' if len(df.columns) > 12 else ''}")


def describe_json(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    keys = list(data.keys()) if isinstance(data, dict) else []
    print(f"- {path.relative_to(ROOT)}: JSON keys: {', '.join(keys[:12])}{' ...' if len(keys) > 12 else ''}")


def inspect_tree(title: str, root: Path) -> None:
    print(f"\n{title}:\n")
    if not root.exists():
        print(f"- {root.relative_to(ROOT)}: not present")
        return

    found = False
    for path in sorted(root.rglob("*")):
        if path.suffix == ".csv":
            describe_csv(path)
            found = True
        elif path.suffix == ".json":
            describe_json(path)
            found = True

    if not found:
        print("- no CSV/JSON files found")


def main() -> None:
    inspect_tree("Available summary outputs", SUMMARY_ROOT)
    inspect_tree("Available raw outputs", RAW_ROOT)
    inspect_tree("Available generated outputs", GENERATED_ROOT)


if __name__ == "__main__":
    main()
