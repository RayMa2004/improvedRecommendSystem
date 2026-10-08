"""CPU worker, batch shape, and ranking model regression checks."""

import os
import unittest
from unittest.mock import patch

import pandas as pd
import torch

import feature_processor
import fine_ranking
from dataset import ThreeTowerDataset
from feature_processor import FeatureProcessor
from fine_ranking import MultiTaskNet
from rough_ranking import ThreeTowerModel
from training.performance import build_training_loader, move_training_batch, training_loader_settings
from utils import collate_fn_three_towers


class RankingWorkerTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "user_id": ["U1", "U2", "U1", "U2"],
            "item_id": ["I1", "I2", "I2", "I1"],
            "gender": [0, 1, 0, 1], "age": [20, 30, 20, 30],
            "category": ["A", "B", "B", "A"], "price": [10, 20, 20, 10],
            "hour": [12, 13, 12, 13],
            "user_clicks": [0, 1, 2, 3], "item_clicks": [3, 2, 1, 0],
            "click": [1, 0, 1, 0], "cart": [0, 0, 1, 0],
            "forward": [0, 1, 0, 0], "buy": [0, 0, 0, 1],
        })
        self.processor = FeatureProcessor()
        self.processor.build_vocab_and_scale(
            self.frame, ["gender"], ["category"], ["age"], ["price"],
            ["hour"], ["user_clicks", "item_clicks"],
        )
        self.dataset = ThreeTowerDataset(
            self.frame, self.processor, ["gender"], ["category"], ["hour"],
            ["age"], ["price"], ["user_clicks", "item_clicks"],
            ["click", "cart", "forward", "buy"],
        )

    def test_scene_and_statistics_respect_explicit_cpu_device(self):
        with patch.object(feature_processor, "device", torch.device("meta")):
            scene = self.processor.transform_scene_features(self.frame, ["hour"], output_device="cpu")
            stats = self.processor.transform_stat_features(
                self.frame, ["user_clicks", "item_clicks"], output_device="cpu"
            )
            legacy_scene = self.processor.transform_scene_features(self.frame, ["hour"])
            legacy_stats = self.processor.transform_stat_features(self.frame, ["user_clicks", "item_clicks"])
        self.assertEqual(scene.device.type, "cpu")
        self.assertEqual(stats.device.type, "cpu")
        self.assertEqual(legacy_scene.device.type, "meta")
        self.assertEqual(legacy_stats.device.type, "meta")
        self.assertTrue(torch.equal(scene[:, 0], torch.tensor([0, 1, 0, 1])))
        self.assertTrue(torch.allclose(stats.mean(dim=0), torch.zeros(2), atol=1e-6))

    def test_dataset_returns_only_cpu_tensors_with_feature_axes_preserved(self):
        with patch.object(feature_processor, "device", torch.device("meta")):
            sample = self.dataset[0]
        self.assertTrue(all(value.device.type == "cpu" for value in sample.values()))
        self.assertEqual(sample["user_discrete"].shape, (1,))
        self.assertEqual(sample["item_discrete"].shape, (1,))
        self.assertEqual(sample["scene_discrete"].shape, (1,))
        self.assertEqual(sample["stat_continuous"].shape, (2,))
        self.assertTrue(torch.equal(sample["targets"], torch.tensor([1., 0., 0., 0.])))

    def test_spawned_loader_preserves_final_single_sample_batch(self):
        with patch.dict(os.environ, {"TEST_RANKING_BATCH_SIZE": "3", "TEST_RANKING_NUM_WORKERS": "2"}):
            loader = build_training_loader(self.dataset, collate_fn_three_towers, "TEST_RANKING", "cpu")
        sizes = []
        for batch in loader:
            size = batch["user_ids"].shape[0]
            sizes.append(size)
            self.assertTrue(all(value.device.type == "cpu" for value in batch.values()))
            self.assertEqual(batch["user_discrete"].shape, (size, 1))
            self.assertEqual(batch["scene_discrete"].shape, (size, 1))
            self.assertEqual(batch["targets"].shape, (size, 4))
        self.assertEqual(sizes, [3, 1])

    def test_loader_allows_zero_workers_and_rejects_invalid_settings(self):
        with patch.dict(os.environ, {"TEST_RANKING_BATCH_SIZE": "3", "TEST_RANKING_NUM_WORKERS": "0"}):
            settings = training_loader_settings("TEST_RANKING", "cpu")
            self.assertNotIn("persistent_workers", settings)
            self.assertNotIn("multiprocessing_context", settings)
            loader = build_training_loader(self.dataset, collate_fn_three_towers, "TEST_RANKING", "cpu")
            self.assertEqual(sum(batch["user_ids"].numel() for batch in loader), 4)
        with patch.dict(os.environ, {"TEST_RANKING_NUM_WORKERS": "-1"}):
            with self.assertRaises(ValueError):
                training_loader_settings("TEST_RANKING", "cpu")

    def test_batches_feed_both_ranking_models_including_single_sample(self):
        rough = ThreeTowerModel(
            2, [2], 1, [2], 2, [2], 1, 2,
            user_tower_hidden=[16], item_tower_hidden=[16], cross_tower_hidden=[16],
        )
        with patch.object(fine_ranking, "device", torch.device("cpu")):
            fine = MultiTaskNet(2, [2], 1, 2, [2], 1, [2], 2, hidden_dims=[16])
        for indices in ([0, 1, 2], [3]):
            batch = move_training_batch(collate_fn_three_towers([self.dataset[i] for i in indices]), "cpu")
            user_args = (batch["user_ids"], batch["user_discrete"], batch["user_continuous"].unsqueeze(1))
            item_args = (batch["item_ids"], batch["item_discrete"], batch["item_continuous"].unsqueeze(1))
            rough_outputs = rough(*user_args, batch["scene_discrete"], *item_args, batch["stat_continuous"])
            fine_outputs = fine(*user_args, *item_args, batch["scene_discrete"], batch["stat_continuous"])
            self.assertTrue(all(output.shape == (len(indices),) for output in rough_outputs))
            self.assertTrue(all(output.shape == (len(indices), 1) for output in fine_outputs))
            self.assertTrue(all(torch.isfinite(output).all() for output in (*rough_outputs, *fine_outputs)))


if __name__ == "__main__":
    unittest.main()
