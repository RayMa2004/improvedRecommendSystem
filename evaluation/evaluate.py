"""Evaluate the complete recommendation chain on a held-out split.

The test split is loaded only here, after the recommendation system has been
initialized from training data and exported weights.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from evaluation.metrics import evaluate_recommendations
from main import RecommenderSystem
from history_features import HistoryConfig, HistorySequenceEncoder
from methods.registry import ACTIVE_METHODS, HISTORY_IMPROVEMENT_METHODS, build_method_registry


def _select_split_rows(split_df: pd.DataFrame, max_users: int | None = None) -> pd.DataFrame:
    """Select complete user rows so labels and recommendations use the same population."""
    if max_users is None:
        return split_df.reset_index(drop=True)
    if max_users <= 0:
        return split_df.iloc[0:0].copy()
    user_ids = list(split_df["user_id"].drop_duplicates())[:max_users]
    return split_df[split_df["user_id"].isin(user_ids)].reset_index(drop=True)


def evaluate_loaded_system(
    system: RecommenderSystem,
    split_df: pd.DataFrame,
    k: int = 10,
    max_users: int | None = None,
    method_profile: str = "baseline",
    history_config: HistoryConfig | None = None,
) -> dict:
    """Evaluate an already fitted system on exactly the supplied split rows."""
    selected_df = _select_split_rows(split_df, max_users)
    if history_config is not None:
        system.history_config = history_config
        system.history_encoder = HistorySequenceEncoder(system.items, history_config)
    selected_methods = HISTORY_IMPROVEMENT_METHODS if method_profile == "history" else ACTIVE_METHODS
    system.methods = build_method_registry(selected_methods)

    recommendations: dict[str, list[str]] = {}
    failures: dict[str, str] = {}
    for user_id, user_df in selected_df.groupby("user_id", sort=False):
        user_id = str(user_id)
        row = user_df.iloc[0]
        try:
            result = system.recommend(
                user_id,
                int(row["hour"]),
                bool(row["is_weekend"]),
                bool(row["is_holiday"]),
            )
            recommendations[user_id] = [str(item_id) for item_id in result]
        except Exception as exc:  # keep a full evaluation run going
            failures[user_id] = f"{type(exc).__name__}: {exc}"

    metrics = evaluate_recommendations(recommendations, selected_df, k=k)
    metrics.update({
        "method_profile": method_profile,
        "method_selection": dict(selected_methods),
        "history_config": vars(history_config) if history_config else None,
        "failed_users": len(failures),
        "failure_examples": dict(list(failures.items())[:5]),
    })
    return metrics


def evaluate_system(
    project_root: str | Path,
    weights_dir: str | Path = "model_weights",
    split: str = "test",
    k: int = 10,
    max_users: int | None = None,
    method_profile: str = "baseline",
    history_config: HistoryConfig | None = None,
    system_interaction_split: str = "train",
) -> dict:
    project_root = Path(project_root).resolve()
    split_path = project_root / "data" / "splits" / f"interactions_{split}.csv"
    split_df = pd.read_csv(split_path, encoding="utf-8")
    if system_interaction_split not in {"train", "train_validation"}:
        raise ValueError("system_interaction_split must be 'train' or 'train_validation'")
    weights_dir = Path(weights_dir)
    if not weights_dir.is_absolute():
        weights_dir = project_root / weights_dir

    # The system itself reads only the training split. This evaluation split
    # is loaded above solely to create future-event requests and labels.
    selected_methods = HISTORY_IMPROVEMENT_METHODS if method_profile == "history" else ACTIVE_METHODS
    previous_cwd = Path.cwd()
    os.chdir(project_root)
    try:
        system = RecommenderSystem(
            interactions_path=project_root / "data" / "splits" / f"interactions_{system_interaction_split}.csv",
            methods=selected_methods,
            history_config=history_config,
        )
        system.fit_with_weight_dir(weights_dir)
    finally:
        os.chdir(previous_cwd)
    metrics = evaluate_loaded_system(
        system,
        split_df,
        k=k,
        max_users=max_users,
        method_profile=method_profile,
        history_config=history_config,
    )
    metrics.update({
        "split": split,
        "weights_dir": str(weights_dir.resolve()),
        "system_interaction_split": system_interaction_split,
    })
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the full recommendation chain")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--weights-dir", default="model_weights")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--system-interaction-split", choices=["train", "train_validation"], default="train")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--experiment-name", default="evaluation")
    parser.add_argument("--method-profile", choices=["baseline", "history"], default="baseline")
    parser.add_argument("--history-mode", choices=["din", "sim_soft", "sim_hard", "hybrid"], default="hybrid")
    parser.add_argument("--history-L", type=int, default=20)
    parser.add_argument("--history-alpha", type=float, default=2.0)
    parser.add_argument(
        "--history-config",
        default=None,
        help="JSON file produced by search_history (best_history_config.json)",
    )
    args = parser.parse_args()
    if args.history_config:
        config_payload = json.loads(Path(args.history_config).read_text(encoding="utf-8"))
        config_payload = config_payload.get("config", config_payload)
        history_config = HistoryConfig(**config_payload)
    else:
        history_config = HistoryConfig(
            mode=args.history_mode,
            max_length=args.history_L,
            hard_threshold_alpha=args.history_alpha,
        )
    result = evaluate_system(
        args.project_root,
        args.weights_dir,
        args.split,
        args.k,
        args.max_users,
        args.method_profile,
        history_config,
        args.system_interaction_split,
    )
    root = Path(args.project_root).resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = root / "experiments" / f"{timestamp}_{args.experiment_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
