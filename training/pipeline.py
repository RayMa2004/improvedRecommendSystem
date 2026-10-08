"""Run the complete train/validation/final-train/test experiment protocol."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from evaluation.evaluate import evaluate_system
from evaluation.search_history import build_candidates, search_history
from history_features import HistoryConfig
from training.runner import run_training


def run_experiment_pipeline(
    project_root: str | Path,
    experiment_name: str = "history_protocol",
    k: int = 10,
    max_users: int | None = None,
    search_args: argparse.Namespace | None = None,
) -> dict:
    """Train baseline, select history parameters, retrain, then test once."""
    project_root = Path(project_root).resolve()
    baseline = run_training(
        project_root,
        stage="all",
        experiment_name=f"{experiment_name}_train",
        config={"method": "baseline", "protocol_stage": "train_only"},
        interaction_split="train",
    )
    baseline_weights = Path(baseline["outputs"][0]).parent
    baseline_validation = evaluate_system(
        project_root=project_root,
        weights_dir=baseline_weights,
        split="validation",
        k=k,
        max_users=max_users,
        method_profile="baseline",
        system_interaction_split="train",
    )

    if search_args is None:
        search_args = argparse.Namespace(
            modes="din,sim_soft,sim_hard,hybrid",
            lengths="20,50",
            alphas="1.5,2.0",
            hard_topks="10",
            temperatures="0.1",
            ranking_weights="0.1,0.2",
        )
    candidates = build_candidates(search_args)
    search = search_history(
        project_root=project_root,
        weights_dir=baseline_weights,
        split="validation",
        metric="ndcg",
        k=k,
        max_users=max_users,
        candidates=candidates,
        experiment_name=f"{experiment_name}_validation_search",
    )
    best_config = HistoryConfig(**search["best_trial"]["config"])

    final_train = run_training(
        project_root,
        stage="all",
        experiment_name=f"{experiment_name}_train_validation",
        config={
            "method": "history",
            "protocol_stage": "train_plus_validation",
            "history_config": vars(best_config),
        },
        interaction_split="train_validation",
    )
    final_weights = Path(final_train["outputs"][0]).parent

    # This is the only operation in the protocol that reads the test split.
    test_metrics = evaluate_system(
        project_root=project_root,
        weights_dir=final_weights,
        split="test",
        k=k,
        max_users=max_users,
        method_profile="history",
        history_config=best_config,
        system_interaction_split="train_validation",
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = project_root / "experiments" / f"{timestamp}_{experiment_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment_name": experiment_name,
        "created_at_utc": timestamp,
        "protocol": [
            "train on train split",
            "select HistoryConfig on validation split",
            "retrain on train plus validation split",
            "evaluate test split once",
        ],
        "baseline_training": baseline,
        "baseline_validation": baseline_validation,
        "validation_search": search,
        "final_training": final_train,
        "best_history_config": vars(best_config),
        "test_metrics": test_metrics,
        "test_read_policy": "test is read only by the final evaluation step",
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full recommendation experiment protocol")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--experiment-name", default="history_protocol")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--modes", default="din,sim_soft,sim_hard,hybrid")
    parser.add_argument("--lengths", default="20,50")
    parser.add_argument("--alphas", default="1.5,2.0")
    parser.add_argument("--hard-topks", default="10")
    parser.add_argument("--temperatures", default="0.1")
    parser.add_argument("--ranking-weights", default="0.1,0.2")
    args = parser.parse_args()
    result = run_experiment_pipeline(
        project_root=args.project_root,
        experiment_name=args.experiment_name,
        k=args.k,
        max_users=args.max_users,
        search_args=args,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
