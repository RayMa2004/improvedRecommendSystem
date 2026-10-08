"""Compare saved baseline and improved evaluation records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare(baseline_path: str | Path, improved_path: str | Path) -> dict:
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    improved = json.loads(Path(improved_path).read_text(encoding="utf-8"))
    result = {
        "baseline": str(baseline_path),
        "improved": str(improved_path),
        "metrics": {},
    }
    for metric in ("hit_rate", "recall", "ndcg"):
        before = float(baseline.get(metric, 0.0))
        after = float(improved.get(metric, 0.0))
        result["metrics"][metric] = {
            "baseline": before,
            "improved": after,
            "delta": after - before,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two evaluation.json files")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--improved", required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.baseline, args.improved), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
