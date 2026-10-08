"""Offline training board built around the repository's existing trainers."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from fine_ranking import FineRankingRecommender
from LightGCN import LightGCNRecommender
from rough_ranking import RoughRankingRecommender
from twin_towers_model import TwoTowersModelRecommender
from training.performance import training_loader_settings


def load_training_frames(project_root: Path, interaction_split: str = "train"):
    data_dir = project_root / "data"
    split_dir = data_dir / "splits"
    items = pd.read_csv(data_dir / "items_new.csv", encoding="utf-8")
    users = pd.read_csv(data_dir / "users_new.csv", encoding="utf-8")
    if interaction_split not in {"train", "train_validation"}:
        raise ValueError("interaction_split must be 'train' or 'train_validation'")
    interactions = pd.read_csv(split_dir / f"interactions_{interaction_split}.csv", encoding="utf-8")
    validation = pd.read_csv(split_dir / "interactions_validation.csv", encoding="utf-8")
    items["item_keywords"] = items["item_keywords"].fillna("").apply(lambda x: tuple(x.split(";")))
    users["user_categories"] = users["user_categories"].fillna("").apply(lambda x: tuple(x.split(";")))
    users["user_keywords"] = users["user_keywords"].fillna("").apply(lambda x: tuple(x.split(";")))
    merged = pd.merge(interactions, users, how="left", on="user_id")
    train = pd.merge(merged, items, how="left", on="item_id")
    return items, users, interactions, validation, train


def train_stage(stage: str, project_root: Path, experiment_dir: Path, interaction_split: str = "train") -> list[str]:
    items, users, interactions, validation, train = load_training_frames(project_root, interaction_split)
    if interactions.empty:
        raise ValueError(f"Training split is empty: {interaction_split}")
    weights_dir = project_root / "model_weights"
    output_dir = experiment_dir / "weights"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []

    previous_cwd = Path.cwd()
    os.chdir(project_root)
    try:
        if stage in {"recall", "all"}:
            two_tower = TwoTowersModelRecommender()
            two_tower.train_twin_towers_model(train)
            source = weights_dir / "improved_twin_towers_model.pth"
            target = output_dir / "improved_twin_towers_model.pth"
            shutil.copy2(source, target)
            outputs.append(str(target))

            item_idx2id = {idx: item_id for idx, item_id in enumerate(items["item_id"])}
            light_gcn = LightGCNRecommender(item_idx2id)
            light_gcn.train_light_gcn(2, list(users["user_id"]), list(items["item_id"]), 64, interactions)
            source = weights_dir / "lightgcn.pth"
            target = output_dir / "lightgcn.pth"
            shutil.copy2(source, target)
            outputs.append(str(target))

        if stage in {"rough_ranking", "all"}:
            rough = RoughRankingRecommender()
            rough.train_three_towers_model(train)
            source = weights_dir / "three_towers_model.pth"
            target = output_dir / "three_towers_model.pth"
            shutil.copy2(source, target)
            outputs.append(str(target))

        if stage in {"fine_ranking", "all"}:
            fine = FineRankingRecommender()
            fine.train_multi_task_model(train)
            source = weights_dir / "multi_task_model.pth"
            target = output_dir / "multi_task_model.pth"
            shutil.copy2(source, target)
            outputs.append(str(target))
    finally:
        os.chdir(previous_cwd)

    return outputs


def run_training(
    project_root: str | Path,
    stage: str = "all",
    experiment_name: str = "baseline",
    config: dict | None = None,
    interaction_split: str = "train",
) -> dict:
    project_root = Path(project_root).resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment_dir = project_root / "experiments" / f"{timestamp}_{experiment_name}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    config = config or {}
    training_device = "cuda" if torch.cuda.is_available() else "cpu"
    default_training_settings = {
        "recall": {
            "two_tower": {
                "epochs": 10, "negative_samples": 2,
                **training_loader_settings("TWO_TOWER", training_device),
            },
            "lightgcn": {"layers": 2, "embedding_dim": 64, "epochs": 50, "batch_size": 200, "batches_per_epoch": 10},
        },
        "rough_ranking": {
            "epochs": 10, "learning_rate": 0.001,
            **training_loader_settings("ROUGH_RANKING", training_device),
        },
        "fine_ranking": {
            "epochs": 1, "learning_rate": 0.01, "dcn_layers": 2,
            **training_loader_settings("FINE_RANKING", training_device),
        },
        "rearrangement": {
            "similarity_batch_size": int(os.getenv("MMR_SIMILARITY_BATCH_SIZE", "2048")),
            "device": training_device,
        },
    }
    record = {
        "experiment_name": experiment_name,
        "created_at_utc": timestamp,
        "stage": stage,
        "method": config.get("method", "baseline"),
        "project_root": str(project_root),
        "training_data": f"data/splits/interactions_{interaction_split}.csv",
        "validation_data": "data/splits/interactions_validation.csv",
        "test_data": "data/splits/interactions_test.csv (evaluation only)",
        "validation_policy": "held out from optimizer input; available to evaluation.evaluate --split validation",
        "default_training_settings": default_training_settings,
        "training_device": training_device,
        "config": config,
        "history_sequence": config.get("history_sequence", {
            "mode": "hybrid",
            "L": 20,
            "alpha": 2.0,
            "hard_topk": 10,
            "ranking_weight": 0.15,
        }),
    }
    record["interaction_split"] = interaction_split
    record["outputs"] = train_stage(stage, project_root, experiment_dir, interaction_split)
    (experiment_dir / "config.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Train recommendation models offline")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--stage", choices=["recall", "rough_ranking", "fine_ranking", "all"], default="all")
    parser.add_argument("--experiment-name", default="baseline")
    parser.add_argument("--interaction-split", choices=["train", "train_validation"], default="train")
    parser.add_argument("--config", default=None, help="Optional JSON experiment configuration")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    result = run_training(args.project_root, args.stage, args.experiment_name, config, args.interaction_split)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
