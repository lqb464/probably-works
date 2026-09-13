"""CPU numerical tests plus tiny real CLIP extraction and full pipeline smoke test."""
import json
import io
from contextlib import redirect_stdout
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from layer_probe.core import pool_tokens, partition, ridge, retrieval, paired_ci


class Numerics(unittest.TestCase):
    def test_mask_keeps_zero_content_and_excludes_after_eos(self):
        x = torch.arange(6.).reshape(1, 6, 1)
        glob, hidden = pool_tokens(x, "text", torch.tensor([[10, 0, 3, 11, 7, 0]]))
        self.assertEqual(glob.item(), 3)
        self.assertEqual(hidden.item(), 1.5)

    def test_image_pool_excludes_cls(self):
        g, h = pool_tokens(torch.tensor([[[100.], [1.], [3.]]]), "image")
        self.assertEqual(g.item(), 100)
        self.assertEqual(h.item(), 2)

    def test_partition(self):
        groups = partition(np.repeat(np.arange(30), 2), [.4, .2, .4], 42)
        self.assertEqual([len(x) for x in groups], [12, 6, 12])
        self.assertEqual(len(set(sum(groups, []))), 30)

    def test_multi_positive_metrics_and_chunk_invariance(self):
        q = torch.tensor([[1., 0.]])
        g = torch.tensor([[1., 0.], [.8, .2], [0., 1.]])
        rows = retrieval(q, g, torch.tensor([1]), torch.tensor([1, 2, 1]), chunk=1)
        self.assertAlmostEqual(rows[0, 3].item(), (1 + 2 / 3) / 2, places=6)
        self.assertAlmostEqual(rows[0, 4].item(), 2 / 3, places=6)
        self.assertEqual(paired_ci(rows, rows, torch.tensor([1]), repetitions=10)["r1"], [0., 0.])
        with self.assertRaises(ValueError):
            retrieval(q, g, torch.tensor([9]), torch.tensor([1, 2, 1]))
        torch.manual_seed(8)
        q, g = torch.randn(7, 5), torch.randn(9, 5)
        qi, gi = torch.arange(7) % 3, torch.arange(9) % 3
        torch.testing.assert_close(retrieval(q, g, qi, gi, chunk=1), retrieval(q, g, qi, gi, chunk=4))

    def test_ridge_identity_balance(self):
        x, y = torch.eye(2), torch.eye(2)
        a = ridge(x, y, torch.tensor([0, 1]), .01)
        b = ridge(x[[0, 0, 1]], y[[0, 0, 1]], torch.tensor([0, 0, 1]), .01)
        torch.testing.assert_close(a, b)

    def test_actual_clip_both_modalities_and_pipeline(self):
        from PIL import Image
        from model.clip_model import CLIP
        from layer_probe import run
        torch.set_num_threads(2)
        torch.manual_seed(7)
        model = CLIP(32, (32, 16), 2, 64, 16, 16, 8, 49408, 64, 1, 2).eval()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "tiny.pt"
            torch.save({"model": {"base_model." + k: v for k, v in model.state_dict().items()}}, checkpoint)
            paths = []
            for color in ("red", "blue", "black", "white"):
                image_path = root / (color + ".png")
                Image.new("RGB", (16, 32), color).save(image_path)
                paths.append(str(image_path))
            splits = {}
            for j, name in enumerate(("fit", "select", "test")):
                ids = [j * 4 + i for i in range(4)]
                splits[name] = dict(image_pids=ids, img_paths=paths,
                                    caption_pids=ids, captions=["red shirt", "blue coat", "black pants", "white hat"])
            for modality in ("text", "image"):
                args = Namespace(checkpoint=str(checkpoint), image_size=[32, 16], stride=16,
                                 device="cpu", modality=modality, batch_size=2, workers=0)
                loaded = run.load_clip(args)
                feats = run.extract(loaded, splits["fit"], args)
                self.assertEqual(feats[modality]["hidden_1"].shape, (4, 64))
                self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
                destination = root / modality
                argv = ["run", "--modality", modality, "--root", tmp, "--checkpoint", str(checkpoint),
                        "--output", str(destination), "--device", "cpu", "--workers", "0",
                        "--image-size", "32", "16", "--penalties", ".001", "--mix-seeds", "42", "--mix-steps", "2"]
                with patch.object(sys, "argv", argv), patch.object(run, "dataset_and_splits", return_value=(splits, "synthetic smoke")), redirect_stdout(io.StringIO()):
                    run.main()
                results = json.loads((destination / "test_results.json").read_text())
                self.assertTrue(all(0 <= r["r1"] <= 1 for r in results))
                self.assertTrue((destination / "selection.json").exists())


if __name__ == "__main__":
    unittest.main()
