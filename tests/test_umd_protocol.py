import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.io import savemat

from affordance_jev.datasets.umd import load_annotation, read_split, build_manifest
from affordance_jev.decision.regions import request_for, merge, decode
from affordance_jev.evaluation.metrics import Metrics, confusion_matrix
from affordance_jev.pipeline.run import sample_directory
from affordance_jev.segmentation.partition import partition


ROOT = Path(__file__).resolve().parents[1]
DEFINITIONS = json.loads((ROOT / "configs/umd_affordances.json").read_text())
CONFIG = json.loads((ROOT / "configs/umd_pipeline.json").read_text())


class UMDProtocolTests(unittest.TestCase):
    def test_manual_selection_uses_metadata_not_frame_modulo(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "tools/item_01"
            folder.mkdir(parents=True)
            (root / "novel_split.txt").write_text("2 item_01\n")
            for frame, kind in [(1, "automatic"), (2, "manual")]:
                stem = folder / f"item_01_{frame:08d}"
                Image.new("RGB", (4, 3)).save(str(stem) + "_rgb.jpg")
                savemat(str(stem) + "_label.mat", {"gt_label": np.ones((3, 4), np.uint8), "gt_type": kind})
            manifest = root / "index.jsonl"
            meta = build_manifest(root, manifest)
            row = json.loads(manifest.read_text())
            self.assertEqual(row["frame_index"], 2)
            self.assertEqual(meta["annotation_types_scanned"], {"automatic": 1, "manual": 1})

    def test_unknown_provenance_and_size_mismatch_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a.mat"
            savemat(p, {"gt_label": np.ones((2, 3), np.uint8)})
            with self.assertRaisesRegex(ValueError, "gt_type"):
                load_annotation(p)
            savemat(p, {"gt_label": np.ones((2, 3), np.uint8), "gt_type": "manual"})
            with self.assertRaisesRegex(ValueError, "dimension"):
                load_annotation(p, (3, 2))

    def test_original_split_formats_and_no_instance_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "split.txt"
            p.write_text("1 knife_01\n2 knife_02\n")
            self.assertEqual(read_split(p), {"knife_01": "train", "knife_02": "test"})
            p.write_text("folds: 0 1 0 0 0 0 0 0 0 0    object: knife_01\n")
            self.assertEqual(read_split(p, "fold", 1)["knife_01"], "train")
            self.assertEqual(read_split(p, "fold", 2)["knife_01"], "test")

    def test_incomplete_split_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tools").mkdir()
            (root / "novel_split.txt").write_text("2 missing_01\n")
            with self.assertRaises(FileNotFoundError):
                build_manifest(root, root / "index.jsonl")

    def test_umd_whole_object_survives_partition_and_merge(self):
        fg = np.ones((10, 10), bool)
        config = {**CONFIG["partition"], "minimum_component_pixels": 1}
        regions, _, _ = partition(np.array([fg]), [.9], fg, config)
        self.assertEqual(len(regions), 1)
        decision = [{"region_id": 1, "final_choice": "contain", "postprocess_reason": None}]
        merged, label = merge(regions, decision, fg, DEFINITIONS, whole_fraction=None)
        self.assertTrue(merged["contain"].all())
        self.assertTrue((label == 4).all())

    def test_request_uses_seven_native_roles_plus_rejection(self):
        request = request_for([{"id": 1}], DEFINITIONS, False)
        names = request["questions"]["region_01"]["criteria"]
        self.assertEqual(set(names), {"grasp", "cut", "scoop", "contain", "pound", "support", "wrap-grasp", "none"})
        self.assertNotIn("stick", names)
        self.assertIn("no whole-object size prohibition", request["state"]["messages"][0]["content"][0]["text"])

    def test_invalid_response_cannot_silently_become_a_label(self):
        with self.assertRaises(ValueError):
            decode({"answers": {}}, [{"id": 1}], DEFINITIONS)

    def test_void_is_ignored_and_background_false_positive_counts(self):
        gt = np.array([[1, 0], [-1, 2]], np.int16)
        pred = np.array([[1, 1], [7, 0]], np.uint8)
        matrix = confusion_matrix(gt, pred)
        self.assertEqual(matrix.sum(), 3)
        self.assertEqual(matrix[0, 1], 1)
        metric = Metrics()
        metric.add(gt, pred)
        result = metric.summary()
        self.assertAlmostEqual(result["per_class"]["grasp"]["iou"], .5)
        self.assertEqual(result["per_class"]["wrap-grasp"]["pred_pixels"], 0)
        self.assertAlmostEqual(result["gIoU_present_queries"], .25)
        self.assertAlmostEqual(result["P_at_50_present_queries"], .5)

    def test_failure_as_empty_prediction_scores_zero(self):
        metric = Metrics()
        metric.add(np.array([[1, 2], [4, 7]]), np.zeros((2, 2), np.uint8))
        self.assertEqual(metric.summary()["gIoU_present_queries"], 0)
        self.assertEqual(metric.summary()["foreground_mIoU"], 0)

    def test_missing_classes_do_not_inflate_macro_iou(self):
        metric = Metrics()
        metric.add(np.array([[1, 0]]), np.array([[1, 2]]))
        result = metric.summary()
        self.assertEqual(result["gIoU_present_queries"], 1)
        self.assertEqual(result["foreground_mIoU"], .5)
        self.assertIsNone(result["per_class"]["scoop"]["iou"])

    def test_overlap_and_unsafe_sample_id_rejected(self):
        fg = np.ones((3, 3), bool)
        with self.assertRaises(ValueError):
            merge([fg, fg], [{"final_choice": "grasp"}, {"final_choice": "cut"}], fg, DEFINITIONS)
        for sample_id in ["../escape", "/absolute", "a/../b", "a\\b"]:
            with self.assertRaises(ValueError):
                sample_directory("/tmp/results", sample_id)


if __name__ == "__main__":
    unittest.main()
