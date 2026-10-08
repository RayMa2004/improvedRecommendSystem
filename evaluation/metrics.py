"""Ranking metrics and test-only evaluation helpers."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import pandas as pd


def hit_rate_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    return float(bool(set(recommended[:k]) & relevant))


def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    return len(set(recommended[:k]) & relevant) / len(relevant) if relevant else 0.0


def ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 2) for rank, item in enumerate(recommended[:k]) if item in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def evaluate_recommendations(recommendations: dict[str, list[str]], test_df: pd.DataFrame, k: int = 10) -> dict:
    relevant_by_user = defaultdict(set)
    for row in test_df.itertuples(index=False):
        relevant_by_user[str(row.user_id)].add(str(row.item_id))
    rows = []
    for user_id, relevant in relevant_by_user.items():
        recommended = [str(item) for item in recommendations.get(user_id, [])]
        rows.append({
            "user_id": user_id,
            "hit_rate": hit_rate_at_k(recommended, relevant, k),
            "recall": recall_at_k(recommended, relevant, k),
            "ndcg": ndcg_at_k(recommended, relevant, k),
        })
    if not rows:
        return {"users": 0, "k": k, "hit_rate": 0.0, "recall": 0.0, "ndcg": 0.0}
    frame = pd.DataFrame(rows)
    return {
        "users": len(frame),
        "k": k,
        "hit_rate": float(frame["hit_rate"].mean()),
        "recall": float(frame["recall"].mean()),
        "ndcg": float(frame["ndcg"].mean()),
    }


def load_test_split(project_root: str | Path, split: str = "test") -> pd.DataFrame:
    path = Path(project_root) / "data" / "splits" / f"interactions_{split}.csv"
    return pd.read_csv(path, encoding="utf-8")
