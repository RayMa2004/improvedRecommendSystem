"""Search HistoryConfig on the validation split only.

The search reuses one fitted recommendation system and changes only the
history encoder configuration between trials.  The test split is deliberately
not accepted by the CLI, so the selected configuration remains untouched by
the final test evaluation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from evaluation.evaluate import evaluate_loaded_system
from history_features import HistoryConfig
from main import RecommenderSystem
from methods.registry import HISTORY_IMPROVEMENT_METHODS


def _numbers(value: str, cast):
    return [cast(part.strip()) for part in value.split(",") if part.strip()]


def build_candidates(args: argparse.Namespace) -> list[HistoryConfig]:
    candidates = []
    for mode, length, alpha, hard_topk, temperature, ranking_weight in itertools.product(
        _numbers(args.modes, str),
        _numbers(args.lengths, int),
        _numbers(args.alphas, float),
        _numbers(args.hard_topks, int),
        _numbers(args.temperatures, float),
        _numbers(args.ranking_weights, float),
    ):
        candidates.append(
            HistoryConfig(
                mode=mode,
                max_length=length,
                hard_threshold_alpha=alpha,
                hard_topk=hard_topk,
                temperature=temperature,
                ranking_weight=ranking_weight,
            )
        )
    return candidates


def search_history(
    project_root: str | Path,
    weights_dir: str | Path,
    split: str = "validation",
    metric: str = "ndcg",
    k: int = 10,
    max_users: int | None = None,
    candidates: list[HistoryConfig] | None = None,
    experiment_name: str = "history_search",
) -> dict:
    if split != "validation":
        raise ValueError("HistoryConfig search must use the validation split; test is reserved for final evaluation")
    if metric not in {"ndcg", "hit_rate", "recall"}:
        raise ValueError("metric must be one of: ndcg, hit_rate, recall")

    project_root = Path(project_root).resolve()
    split_path = project_root / "data" / "splits" / "interactions_validation.csv"
    validation_df = pd.read_csv(split_path, encoding="utf-8")
    weights_dir = Path(weights_dir)
    if not weights_dir.is_absolute():
        weights_dir = project_root / weights_dir
    if candidates is None:
        candidates = [HistoryConfig()]
    if not candidates:
        raise ValueError("At least one HistoryConfig candidate is required")

    previous_cwd = Path.cwd()
    os.chdir(project_root)
    try:
        first_config = candidates[0]
        system = RecommenderSystem(
            methods=HISTORY_IMPROVEMENT_METHODS,
            history_config=first_config,
        )
        system.fit_with_weight_dir(weights_dir)
        trials = []
        for index, config in enumerate(candidates, start=1):
            result = evaluate_loaded_system(
                system,
                validation_df,
                k=k,
                max_users=max_users,
                method_profile="history",
                history_config=config,
            )
            trials.append({
                "trial": index,
                "config": vars(config),
                "metrics": result,
            })
    finally:
        os.chdir(previous_cwd)

    best = max(trials, key=lambda row: row["metrics"].get(metric, float("-inf")))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = project_root / "experiments" / f"{timestamp}_{experiment_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_name": experiment_name,
        "created_at_utc": timestamp,
        "split": "validation",
        "weights_dir": str(weights_dir),
        "metric": metric,
        "k": k,
        "max_users": max_users,
        "candidate_count": len(candidates),
        "best_trial": best,
        "trials": trials,
        "test_policy": "test split is not read during HistoryConfig search",
    }
    (output_dir / "search_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "best_history_config.json").write_text(
        json.dumps({
            "config": best["config"],
            "validation_metrics": best["metrics"],
            "selection_metric": metric,
            "weights_dir": str(weights_dir),
            "split": "validation",
        }, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Select HistoryConfig using validation only")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--split", choices=["validation"], default="validation")
    parser.add_argument("--metric", choices=["ndcg", "hit_rate", "recall"], default="ndcg")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--modes", default="din,sim_soft,sim_hard,hybrid")
    parser.add_argument("--lengths", default="20,50")
    parser.add_argument("--alphas", default="1.5,2.0")
    parser.add_argument("--hard-topks", default="10")
    parser.add_argument("--temperatures", default="0.1")
    parser.add_argument("--ranking-weights", default="0.1,0.2")
    parser.add_argument("--experiment-name", default="history_search")
    args = parser.parse_args()
    result = search_history(
        project_root=args.project_root,
        weights_dir=args.weights_dir,
        split=args.split,
        metric=args.metric,
        k=args.k,
        max_users=args.max_users,
        candidates=build_candidates(args),
        experiment_name=args.experiment_name,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
