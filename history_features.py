"""History-sequence features for recall and ranking experiments.

The encoder uses the existing CLIP item content vectors.  DIN uses query
conditioned soft attention, SIM(soft) uses a soft similarity aggregation, and
SIM(hard) keeps the most similar historical items.  The hybrid policy selects
among them from the observed history length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F


@dataclass
class HistoryConfig:
    mode: str = "hybrid"
    max_length: int = 20  # L
    hard_threshold_alpha: float = 2.0  # alpha
    hard_topk: int = 10
    temperature: float = 0.1
    recall_topk: int = 50
    ranking_weight: float = 0.15

    def __post_init__(self) -> None:
        allowed = {"din", "sim_soft", "sim_hard", "hybrid"}
        if self.mode not in allowed:
            raise ValueError(f"Unknown history mode: {self.mode}")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.hard_threshold_alpha < 1:
            raise ValueError("hard_threshold_alpha must be >= 1")
        if self.hard_topk <= 0 or self.temperature <= 0:
            raise ValueError("hard_topk and temperature must be positive")


class HistorySequenceEncoder:
    """Encode a user's recent item IDs and score candidate items."""

    def __init__(self, items: Iterable, config: HistoryConfig | None = None):
        self.config = config or HistoryConfig()
        self.item_vectors = {}
        for item in items:
            if item.content_feature is None:
                continue
            vector = item.content_feature
            if not torch.is_tensor(vector):
                vector = torch.as_tensor(vector)
            self.item_vectors[item.item_id] = F.normalize(vector.float().reshape(-1), dim=0).cpu()

    def selected_mode(self, history_length: int) -> str:
        if self.config.mode != "hybrid":
            return self.config.mode
        if history_length > self.config.hard_threshold_alpha * self.config.max_length:
            return "sim_hard"
        if history_length > self.config.max_length:
            return "sim_soft"
        return "din"

    def _history_vectors(self, history_ids: Iterable[str]) -> torch.Tensor:
        vectors = [self.item_vectors[item_id] for item_id in history_ids if item_id in self.item_vectors]
        if not vectors:
            return torch.empty((0, 0), dtype=torch.float32)
        return torch.stack(vectors)

    def din_score(self, query: torch.Tensor, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """DIN-style query-conditioned attention over the full sequence."""
        similarities = history @ query
        weights = F.softmax(similarities / self.config.temperature, dim=0)
        return history, weights

    def sim_soft_score(self, query: torch.Tensor, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """SIM soft search: softmax over all searchable historical items."""
        similarities = history @ query
        weights = F.softmax(similarities / self.config.temperature, dim=0)
        return history, weights

    def sim_hard_score(self, query: torch.Tensor, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """SIM hard search: retain only the top similar historical items."""
        similarities = history @ query
        count = min(self.config.hard_topk, history.size(0))
        indices = torch.topk(similarities, count).indices
        selected = history[indices]
        weights = F.softmax(similarities[indices] / self.config.temperature, dim=0)
        return selected, weights

    def _weights(self, query: torch.Tensor, history: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
        if mode == "din":
            return self.din_score(query, history)
        if mode == "sim_soft":
            return self.sim_soft_score(query, history)
        return self.sim_hard_score(query, history)

    def encode(self, history_ids: Iterable[str], query_item_id: str | None = None) -> tuple[torch.Tensor | None, str]:
        history_ids = list(history_ids)
        history = self._history_vectors(history_ids)
        mode = self.selected_mode(len(history_ids))
        if history.numel() == 0:
            return None, mode
        query = self.item_vectors.get(query_item_id) if query_item_id else history[-1]
        if query is None:
            query = history.mean(dim=0)
        selected, weights = self._weights(query, history, mode)
        context = F.normalize((selected * weights.unsqueeze(1)).sum(dim=0), dim=0)
        return context, mode

    def score_item(self, history_ids: Iterable[str], item_id: str) -> float:
        history_ids = list(history_ids)
        query = self.item_vectors.get(item_id)
        if query is None:
            return 0.0
        history = self._history_vectors(history_ids)
        if history.numel() == 0:
            return 0.0
        mode = self.selected_mode(len(history_ids))
        selected, weights = self._weights(query, history, mode)
        context = F.normalize((selected * weights.unsqueeze(1)).sum(dim=0), dim=0)
        return float(torch.clamp(torch.dot(query, context), -1.0, 1.0).item())

    def score_candidates(self, history_ids: Iterable[str], candidate_ids: Iterable[str]) -> dict[str, float]:
        history_ids = list(history_ids)
        return {item_id: self.score_item(history_ids, item_id) for item_id in candidate_ids}

    def recall(self, history_ids: Iterable[str], all_item_ids: Iterable[str], existing_ids: Iterable[str] = ()) -> set[str]:
        history_ids = list(history_ids)
        existing_ids = set(existing_ids)
        if not history_ids:
            return set()
        scores = self.score_candidates(history_ids, all_item_ids)
        history_set = set(history_ids)
        ranked = sorted(
            ((item_id, score) for item_id, score in scores.items() if item_id not in history_set),
            key=lambda pair: pair[1], reverse=True,
        )
        return existing_ids.union(item_id for item_id, _ in ranked[: self.config.recall_topk])
