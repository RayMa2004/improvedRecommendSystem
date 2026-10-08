"""Small adapters that make each recommendation stage replaceable."""

from __future__ import annotations

from typing import Any, Callable


BASELINE_METHODS = {
    "recall": "baseline",
    "rough_ranking": "baseline",
    "fine_ranking": "baseline",
    "rearrangement": "baseline",
}

# Switch this mapping into ACTIVE_METHODS when running the history-sequence
# experiment. Rearrangement intentionally remains unchanged.
HISTORY_IMPROVEMENT_METHODS = {
    "recall": "improved",
    "rough_ranking": "improved",
    "fine_ranking": "improved",
    "rearrangement": "baseline",
}

ACTIVE_METHODS = dict(BASELINE_METHODS)


def _unimplemented(stage: str) -> Callable[..., Any]:
    def method(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            f"The improved {stage} method is a reserved extension point. "
            "Implement and evaluate it before switching ACTIVE_METHODS."
        )

    return method


def _baseline_recall(recommender: Any, user_profile: Any, **kwargs: Any) -> Any:
    return recommender.recall(user_profile)


def _history_recall(recommender: Any, user_profile: Any, history_encoder: Any = None) -> Any:
    base = recommender.recall(user_profile)
    if history_encoder is None:
        return base
    return history_encoder.recall(
        user_profile.get_last_n_interactions(10_000),
        history_encoder.item_vectors.keys(),
        existing_ids=base,
    )


def _baseline_rough(
    recommender: Any, user_profile: Any, recall_items: Any, scene: Any, **kwargs: Any
) -> Any:
    return recommender.rough_ranking(user_profile, recall_items, scene)


def _history_rough(
    recommender: Any,
    user_profile: Any,
    recall_items: Any,
    scene: Any,
    history_encoder: Any = None,
    history_ids: Any = None,
) -> Any:
    if history_encoder is None or recall_items.empty:
        return recommender.rough_ranking(user_profile, recall_items, scene)
    history_ids = list(history_ids or [])
    if not history_ids:
        return recommender.rough_ranking(user_profile, recall_items, scene)
    candidate_ids = list(recall_items["item_id"])
    scores = history_encoder.score_candidates(history_ids, candidate_ids)
    return recommender.rough_ranking(
        user_profile,
        recall_items,
        scene,
        history_score_map=scores,
        history_weight=history_encoder.config.ranking_weight,
    )


def _baseline_fine(recommender: Any, features: Any, **kwargs: Any) -> Any:
    return recommender.fine_ranking(features)


def _history_fine(recommender: Any, features: Any, history_encoder: Any = None, history_ids: Any = None) -> Any:
    scores = recommender.fine_ranking(features)
    if history_encoder is None or not history_ids:
        return scores
    history_scores = history_encoder.score_candidates(history_ids, scores.keys())
    weight = history_encoder.config.ranking_weight
    return {
        item_id: float(score) + weight * max(0.0, history_scores.get(item_id, 0.0))
        for item_id, score in scores.items()
    }


def _baseline_rearrangement(recommender: Any, items: Any) -> Any:
    return recommender.mmr_diversity_selection(items)


def build_method_registry(methods: dict[str, str] | None = None) -> dict[str, Callable[..., Any]]:
    selected = dict(ACTIVE_METHODS)
    if methods:
        selected.update(methods)

    baseline = {
        "recall": _baseline_recall,
        "rough_ranking": _baseline_rough,
        "fine_ranking": _baseline_fine,
        "rearrangement": _baseline_rearrangement,
    }
    registry: dict[str, Callable[..., Any]] = {}
    for stage, method_name in selected.items():
        if method_name == "baseline":
            registry[stage] = baseline[stage]
        elif method_name == "improved":
            registry[stage] = {
                "recall": _history_recall,
                "rough_ranking": _history_rough,
                "fine_ranking": _history_fine,
                "rearrangement": _unimplemented(stage),
            }[stage]
        else:
            raise ValueError(f"Unknown {stage} method: {method_name}")
    return registry
