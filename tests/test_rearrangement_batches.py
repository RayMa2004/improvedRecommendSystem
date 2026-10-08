"""Changing similarity block size must preserve MMR results."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import torch

import rearrangement
from rearrangement import MmrDiversity


class _Item:
    def __init__(self, item_id, feature, relevance):
        self.item_id = item_id
        self.content_feature = feature
        self.relevance_score = relevance


class RearrangementBatchTests(unittest.TestCase):
    def test_block_sizes_preserve_cosine_similarity_and_selection(self):
        items = [
            _Item("A", torch.tensor([1., 0.]), 0.9),
            _Item("B", np.array([0., 1.], dtype=np.float32), 0.8),
            _Item("C", torch.tensor([1., 1.]), 0.7),
        ]
        vectors = np.array([[1., 0.], [0., 1.], [1., 1.]], dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        selections = []
        for size in (1, 2, 4096):
            with patch.object(rearrangement, "device", "cpu"), patch.dict(
                os.environ, {"MMR_SIMILARITY_BATCH_SIZE": str(size)}
            ):
                recommender = MmrDiversity()
                recommender.build_cosine_similarity_matrix(items)
            np.testing.assert_allclose(recommender.cosine_similarity_matrix, vectors @ vectors.T, atol=1e-6)
            selections.append(recommender.mmr_diversity_selection(items))
        self.assertEqual(selections, [["A", "B", "C"]] * 3)

    def test_invalid_block_size_fails_before_matrix_allocation(self):
        with patch.dict(os.environ, {"MMR_SIMILARITY_BATCH_SIZE": "0"}):
            with self.assertRaises(ValueError):
                MmrDiversity().build_cosine_similarity_matrix([])


if __name__ == "__main__":
    unittest.main()
