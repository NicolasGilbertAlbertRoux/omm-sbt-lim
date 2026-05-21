from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_ROOT = ROOT / "results" / "summaries"
RAW_ROOT = ROOT / "results" / "raw"


def describe_csv(path: Path) -> None:
    df = pd.read_csv(path)
    print(f"- {path.relative_to(ROOT)}: {len(df):,} rows × {len(df.columns)} columns")
    print(f"  columns: {', '.join(df.columns[:12])}{' ...' if len(df.columns) > 12 else ''}")


def describe_json(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    keys = list(data.keys()) if isinstance(data, dict) else []
    print(f"- {path.relative_to(ROOT)}: JSON keys: {', '.join(keys[:12])}{' ...' if len(keys) > 12 else ''}")


def main() -> None:
    print("Available summary outputs:\n")
    for path in sorted(SUMMARY_ROOT.rglob("*")):
        if path.suffix == ".csv":
            describe_csv(path)
        elif path.suffix == ".json":
            describe_json(path)

    print("\nAvailable raw outputs:\n")
    for path in sorted(RAW_ROOT.rglob("*.csv")):
        describe_csv(path)


if __name__ == "__main__":
    main()
