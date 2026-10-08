"""Regression checks for CPU-only two-tower DataLoader workers."""

import unittest
from unittest.mock import patch

import pandas as pd
import torch
from torch.utils.data import DataLoader

import feature_processor
from dataset import TwoTowerDataset
from feature_processor import FeatureProcessor
from utils import collate_fn_two_towers


class TwoTowerWorkerTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "user_id": ["U1", "U2", "U1", "U2"],
            "item_id": ["I1", "I2", "I2", "I1"],
            "gender": [0, 1, 0, 1],
            "age": [20, 30, 20, 30],
            "category": ["A", "B", "B", "A"],
            "price": [10, 20, 20, 10],
        })
        self.processor = FeatureProcessor()
        self.processor.build_vocab_and_scale(
            self.frame, ["gender"], ["category"], ["age"], ["price"]
        )
        self.dataset = TwoTowerDataset(
            self.frame, self.processor, ["gender"], ["category"],
            ["age"], ["price"], n_neg=2,
        )

    def test_explicit_cpu_output_overrides_default_device(self):
        with patch.object(feature_processor, "device", torch.device("meta")):
            user = self.processor.transform_user_features(
                self.frame, ["gender"], ["age"], output_device="cpu"
            )
            item = self.processor.transform_item_features(
                self.frame, ["category"], ["price"], output_device="cpu"
            )
        self.assertTrue(all(t.device.type == "cpu" for t in (*user, *item)))
        self.assertEqual(user[1].shape, (4, 1))
        self.assertEqual(item[1].shape, (4, 1))
        self.assertTrue(torch.equal(user[0], torch.tensor([0, 1, 0, 1])))
        self.assertTrue(torch.equal(item[0], torch.tensor([0, 1, 1, 0])))
        self.assertTrue(torch.allclose(user[2].flatten(), torch.tensor([-1., 1., -1., 1.])))

    def test_default_output_device_is_preserved(self):
        with patch.object(feature_processor, "device", torch.device("meta")):
            user = self.processor.transform_user_features(self.frame, ["gender"], ["age"])
            item = self.processor.transform_item_features(self.frame, ["category"], ["price"])
        self.assertTrue(all(t.device.type == "meta" for t in (*user, *item)))

    def test_dataset_never_creates_accelerator_tensors(self):
        with patch.object(feature_processor, "device", torch.device("meta")):
            sample = self.dataset[0]
        tensors = list(sample["pos"])
        for negative in sample["neg"]:
            tensors.extend(negative)
        self.assertTrue(all(t.device.type == "cpu" for t in tensors))
        self.assertEqual(len(sample["neg"]), 2)
        self.assertEqual(sample["pos"][0].shape, (1,))

    def test_spawned_workers_produce_complete_cpu_batches(self):
        loader = DataLoader(
            self.dataset, batch_size=3, num_workers=2,
            multiprocessing_context="spawn", collate_fn=collate_fn_two_towers,
        )
        sizes = []
        for batch in loader:
            sizes.append(batch["user_feat"][0].shape[0])
            tensors = [*batch["user_feat"], *batch["pos_item_feat"]]
            for negative in batch["neg_item_feats"]:
                tensors.extend(negative)
            self.assertTrue(all(t.device.type == "cpu" for t in tensors))
            self.assertEqual(len(batch["neg_item_feats"]), 2)
        self.assertEqual(sizes, [3, 1])


if __name__ == "__main__":
    unittest.main()
