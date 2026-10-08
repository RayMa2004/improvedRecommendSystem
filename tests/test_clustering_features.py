"""Regression tests for clustering recall feature shapes."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from recall import ClusteringRecommender


class ClusteringFeatureShapeTests(unittest.TestCase):
    def test_online_cosine_similarity_accepts_batched_item_features(self):
        recommender = ClusteringRecommender(n_clusters=1)
        recommender.cluster_centers = np.array([[1.0, 0.0]], dtype=np.float32)

        seed = SimpleNamespace(
            item_id="seed",
            content_feature=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        )
        neighbor = SimpleNamespace(
            item_id="neighbor",
            content_feature=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        )
        recommender.items_by_cluster[0] = [seed, neighbor]

        self.assertEqual(recommender.get_nearest_cluster(seed), 0)
        self.assertEqual(
            recommender.find_similar_items_in_cluster(seed, cluster_idx=0, m=1),
            [neighbor],
        )


if __name__ == "__main__":
    unittest.main()
