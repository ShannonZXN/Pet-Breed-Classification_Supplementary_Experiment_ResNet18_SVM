"""Functional tests using SYNTHETIC features, not pet-model performance.

Run `python self_test.py`. Everything is written to temporary directories.
No pretrained weights are downloaded. Torch structure checks are optional.
"""
import csv
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

import resnet18_svm as m
import merge_split_csvs as merge
import oxford_annotations as annotations
import build_manifest_from_ids as build_ids


class SetupTests(unittest.TestCase):
    def setUp(self):
        # Keep expected failures and synthetic model scores out of user logs.
        """Create tiny paired images, synthetic features, labels, and isolated output paths."""
        self.captured_output = io.StringIO()
        self.output_redirect = contextlib.redirect_stdout(self.captured_output)
        self.output_redirect.__enter__()
        self.addCleanup(self.output_redirect.__exit__, None, None, None)
        self.temp = tempfile.TemporaryDirectory(prefix="resnet_svm_selftest_")
        self.root = Path(self.temp.name)
        self.original = self.root / "original"
        self.foreground = self.root / "foreground"
        self.original.mkdir()
        self.foreground.mkdir()
        self.rows = []
        self.feature_rows = {}
        rng = np.random.default_rng(7)
        # Non-contiguous shared IDs deliberately test that labels are preserved.
        for category, cid in enumerate(["2", "10", "37"]):
            for number, split in enumerate(["train"] * 3 + ["val"] * 2 + ["test"] * 2):
                stem = f"breed_{cid}_{number}"
                path = self.original / f"{stem}.png"
                Image.new("RGB", (16, 16), (category * 60, number * 20, 30)).save(path)
                Image.new("RGB", (16, 16), (category * 60, number * 20, 0)).save(self.foreground / path.name)
                row = {"image_id": stem, "class_id": cid, "class_name": f"breed_{cid}", "split": split}
                self.rows.append(row)
                feature = rng.normal(0, 0.02, m.FEATURE_DIM).astype(np.float32)
                feature[category] += 10
                # Validation/test distributions differ: scaler must still use train only.
                if split != "train":
                    feature[-1] += 100
                self.feature_rows[stem] = feature
        self.manifest = self.root / "manifest.csv"
        m.write_csv(self.manifest, self.rows)
        self.annotation_dir = self.root / "annotations"
        self.annotation_dir.mkdir()
        for filename, included_splits in (
            ("list.txt", {"train", "val", "test"}),
            ("trainval.txt", {"train", "val"}), ("test.txt", {"test"}),
        ):
            (self.annotation_dir / filename).write_text("\n".join(
                f"{row['image_id']} {row['class_id']} 2 {['2','10','37'].index(row['class_id'])+1}"
                for row in self.rows if row["split"] in included_splits
            ) + "\n", encoding="utf-8")
        self.patches = mock.patch.multiple(
            m, MANIFEST_CSV=self.manifest, EXPECTED_CLASS_COUNT=3,
            ANNOTATIONS_PATH=self.annotation_dir,
            IMAGE_FOLDERS={"original": self.original, "foreground": self.foreground},
            OUTPUT_ROOT=self.root / "results", C_VALUES=[0.1, 1.0], KERNELS=["linear", "rbf"],
            RBF_GAMMAS=["scale"], MAX_ERROR_EXAMPLES=2,
        )
        self.patches.start()
        images = m.index_images(self.original)
        _, ids, hashes, signature = m.snapshot_images(images)
        features = np.stack([self.feature_rows[key] for key in ids])
        self.cache = m.OUTPUT_ROOT / "original" / "feature_cache" / "features.npz"
        self.cache.parent.mkdir(parents=True)
        np.savez_compressed(self.cache, features=features, image_ids=np.asarray(ids),
                            image_sha256=np.asarray(hashes), metadata_json=json.dumps({
                                "signature": signature, "extractor": m.EXTRACTOR_SPEC,
                                "source": "SYNTHETIC UNIT-TEST FEATURES; NOT IMAGENET RESULTS",
                            }))
        self.ids, self.features, self.signature = ids, features, signature

    def tearDown(self):
        """Restore module settings and remove all temporary test files."""
        self.patches.stop()
        self.temp.cleanup()

    def test_manifest_and_image_alignment(self):
        """Preserve noncontiguous labels and join shuffled feature rows to their image IDs."""
        rows, names = m.load_manifest()
        self.assertEqual(set(names), {"2", "10", "37"})
        selected, x, y = m.select_rows(rows, "val", self.ids[::-1], self.features[::-1])
        for row, feature, label in zip(selected, x, y):
            np.testing.assert_array_equal(feature, self.feature_rows[row["image_id"]])
            self.assertEqual(label, row["class_id"])
        # MAT files do not become images or class samples.
        (self.original / "metadata.mat").write_bytes(b"metadata")
        self.assertEqual(len(m.index_images(self.original)), 21)

    def test_preprocessing_must_preserve_image_dimensions(self):
        """Accept a native-size transform and reject one that changes the image grid."""
        path = self.root / "rectangular.png"
        Image.new("RGB", (137, 93)).save(path)
        items = [("rectangular", path)]
        dataset = m.ImageFeatureDataset(items, lambda image: np.zeros((3, image.height, image.width)))
        tensor, key = dataset[0]
        self.assertEqual(key, "rectangular")
        self.assertEqual(tensor.shape, (3, 93, 137))
        resized = m.ImageFeatureDataset(items, lambda image: np.zeros((3, 224, 224)))
        with self.assertRaisesRegex(RuntimeError, "Image dimensions changed"):
            resized[0]

    def test_smoke_uses_same_training_ids_and_keeps_full_cache(self):
        """Use identical training samples for both branches without altering images or full caches."""
        before_cache = self.cache.read_bytes()
        before_images = {p: p.read_bytes() for folder in (self.original, self.foreground)
                         for p in folder.iterdir()}
        calls = []

        def synthetic_extraction(dataset, refresh=False, selected_keys=None, cache_path=None):
            """Stand in for the extractor so smoke-report logic can be tested offline."""
            self.assertTrue(refresh)
            train_ids = {row["image_id"] for row in self.rows if row["split"] == "train"}
            self.assertTrue(set(selected_keys).issubset(train_ids))
            indexed = m.index_images(m.IMAGE_FOLDERS[dataset])
            ids = sorted(selected_keys)
            sizes = []
            for key in ids:
                with Image.open(indexed[key]) as image:
                    sizes.append(image.size)
            features = np.stack([self.feature_rows[key] for key in ids])
            metadata = {"model_eval": True, "parameters_frozen": True,
                        "native_resolution_checked": True, "source": "SYNTHETIC SMOKE TEST"}
            np.savez_compressed(cache_path, features=features, image_ids=np.asarray(ids),
                                input_sizes_wh=np.asarray(sizes), metadata_json=json.dumps(metadata))
            calls.append((dataset, ids))
            return indexed, ids, features, metadata

        with mock.patch.object(m, "SMOKE_SAMPLE_COUNT", 4), \
             mock.patch.object(m, "check_environment", return_value={"check_passed": True}), \
             mock.patch.object(m, "get_features", side_effect=synthetic_extraction):
            directories = m.run_smoke(["foreground", "original"])
        self.assertEqual(calls[0][1], calls[1][1])
        for dataset, directory in directories.items():
            self.assertTrue(directory.is_relative_to(m.OUTPUT_ROOT / dataset / "smoke_tests"))
            report = json.loads((directory / "step2_report.json").read_text())
            self.assertTrue(report["passed"])
            self.assertEqual(report["feature_shape"], [4, 512])
            self.assertTrue(report["checks"]["paired_dimensions_match"])
            with (directory / "sample_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
                samples = list(csv.DictReader(stream))
            self.assertEqual({row["split"] for row in samples}, {"train"})
            self.assertEqual([row["image_key"] for row in samples], calls[0][1])
        self.assertEqual(self.cache.read_bytes(), before_cache)
        self.assertTrue(all(p.read_bytes() == contents for p, contents in before_images.items()))

    def test_smoke_rejects_mismatched_pair_sizes_before_inference(self):
        """Reject paired-size disagreement before calling the feature extractor."""
        first_train = next(row for row in self.rows if row["split"] == "train")
        Image.new("RGB", (17, 16)).save(self.foreground / f"{first_train['image_id']}.png")
        with mock.patch.object(m, "SMOKE_SAMPLE_COUNT", 9), \
             mock.patch.object(m, "check_environment", return_value={"check_passed": True}), \
             mock.patch.object(m, "get_features") as extraction:
            with self.assertRaisesRegex(ValueError, "Paired image dimensions disagree"):
                m.run_smoke(["foreground", "original"])
            extraction.assert_not_called()
        reports = list(m.OUTPUT_ROOT.glob("*/smoke_tests/*/step2_report.json"))
        self.assertEqual(len(reports), 2)
        self.assertTrue(all(not json.loads(path.read_text())["passed"] for path in reports))

    def test_sample_extraction_cannot_replace_full_feature_cache(self):
        """Require a separate cache path for sample-only extraction."""
        before = self.cache.read_bytes()
        key = next(row["image_id"] for row in self.rows if row["split"] == "train")
        with self.assertRaisesRegex(ValueError, "separate cache path"):
            m.get_features("original", selected_keys=[key])
        with self.assertRaisesRegex(ValueError, "separate cache path"):
            m.get_features("original", selected_keys=[key], cache_path=self.cache)
        self.assertEqual(self.cache.read_bytes(), before)

    def test_environment_failure_is_saved(self):
        """Persist a useful failure report when extractor imports fail."""
        with mock.patch.object(m, "torch_modules", side_effect=RuntimeError("Synthetic import failure")):
            with self.assertRaisesRegex(RuntimeError, "Synthetic import failure"):
                m.check_environment()
        report = json.loads((m.OUTPUT_ROOT / "environment_report.json").read_text())
        self.assertFalse(report["check_passed"])
        self.assertIn("Synthetic import failure", report["error"])

    def test_directml_selection_and_report_do_not_require_cuda(self):
        """Report a mocked DirectML device correctly when CUDA is unavailable."""
        class SyntheticTensor:
            def __init__(self, values):
                self.values = np.asarray(values, dtype=np.float32)
            def to(self, device):
                return self
            def __add__(self, value):
                return SyntheticTensor(self.values + value)
            def cpu(self):
                return self
            def numpy(self):
                return self.values

        device = SimpleNamespace(type="privateuseone", index=0)
        directml = SimpleNamespace(device=mock.Mock(return_value=device))
        torch = SimpleNamespace(__version__="synthetic", tensor=SyntheticTensor,
                                cuda=SimpleNamespace(is_available=lambda: False))
        vision = SimpleNamespace(__version__="synthetic")
        with mock.patch.dict(sys.modules, {"torch_directml": directml}), \
             mock.patch.object(m, "DEVICE", "directml"), \
             mock.patch.object(m, "torch_modules", return_value=(torch, vision)):
            report = m.check_environment()
        directml.device.assert_called_once_with()
        self.assertTrue(report["check_passed"])
        self.assertTrue(report["device_tensor_test_passed"])
        self.assertFalse(report["cuda_available"])
        self.assertEqual(report["selected_backend"], "directml")

    def test_device_auto_keeps_cpu_and_explicit_directml_errors(self):
        """Select CPU under auto without CUDA, and surface explicit DirectML failures."""
        cpu = SimpleNamespace(type="cpu")
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False),
                                device=mock.Mock(return_value=cpu))
        with mock.patch.object(m, "DEVICE", "auto"):
            self.assertIs(m.device_for(torch), cpu)
        torch.device.assert_called_once_with("cpu")
        with mock.patch.dict(sys.modules, {"torch_directml": None}), \
             mock.patch.object(m, "DEVICE", "directml"):
            with self.assertRaisesRegex(RuntimeError, "DirectML was requested"):
                m.device_for(torch)

    def test_split_overlap_is_rejected(self):
        """Detect duplicate split IDs despite extension and case differences."""
        overlap = {**self.rows[0], "image_id": self.rows[0]["image_id"].upper() + ".JPG", "split": "test"}
        m.write_csv(self.manifest, self.rows + [overlap])
        with self.assertRaisesRegex(ValueError, "Repeated image"):
            m.load_manifest()

    def test_official_labels_and_test_membership_are_checked(self):
        """Reject altered breed labels and official test membership changes."""
        changed = [dict(row) for row in self.rows]
        changed[0]["class_id"] = "10"
        changed[0]["class_name"] = "breed_10"
        m.write_csv(self.manifest, changed)
        with self.assertRaisesRegex(ValueError, "Class ID disagrees"):
            m.load_manifest()
        changed = [dict(row) for row in self.rows]
        test_row = next(row for row in changed if row["split"] == "test")
        test_row["split"] = "val"
        m.write_csv(self.manifest, changed)
        with self.assertRaisesRegex(ValueError, "changes official test"):
            m.load_manifest()

    def test_explicit_team_label_mapping_preserves_team_ids(self):
        """Retain a complete explicit team label map and reject an incomplete map."""
        mapping = {"2": "0", "10": "1", "37": "2"}
        remapped = [{**row, "class_id": mapping[row["class_id"]]} for row in self.rows]
        m.write_csv(self.manifest, remapped)
        with mock.patch.object(m, "CLASS_ID_MAPPING", mapping):
            rows, names = m.load_manifest()
        self.assertEqual(set(names), {"0", "1", "2"})
        self.assertEqual([row["class_id"] for row in rows], [row["class_id"] for row in remapped])
        with mock.patch.object(m, "CLASS_ID_MAPPING", {"2": "0"}):
            with self.assertRaisesRegex(ValueError, "every official class"):
                m.load_manifest()

    def test_existing_validation_ids_get_official_labels(self):
        """Rebuild labels around existing validation IDs without changing their memberships."""
        val_file = self.root / "val_ids.txt"
        val_file.write_text("\n".join(row["image_id"].upper() + ".PNG" for row in self.rows
                                      if row["split"] == "val"), encoding="utf-8")
        destination = self.root / "from_ids.csv"
        with mock.patch.multiple(build_ids, ANNOTATIONS_PATH=self.annotation_dir,
                                 VALIDATION_IDS_TXT=val_file, OUTPUT_CSV=destination):
            build_ids.main()
        actual, _ = m.load_manifest(destination)
        self.assertEqual({row["image_id"]: (row["class_id"], row["split"]) for row in actual},
                         {row["image_id"]: (row["class_id"], row["split"]) for row in self.rows})

    def test_audit_before_shared_manifest_arrives(self):
        """Write a useful data audit while reporting that an empty manifest is not ready."""
        self.manifest.write_text("image_id,class_id,class_name,split\n", encoding="utf-8")
        summary = m.run_audit()
        self.assertFalse(summary["manifest_ready"])
        self.assertEqual(summary["image_counts"], {"original": 21, "foreground": 21})
        self.assertTrue(all(not absent for absent in summary["missing_in_variant"].values()))
        self.assertIn("Manifest is empty", summary["manifest_message"])

    def test_cache_rejects_changed_image_content(self):
        """Invalidate a cache when selected source image bytes change."""
        self.assertIsNotNone(m.read_feature_cache(self.cache, self.ids, self.signature))
        path = next(self.original.iterdir())
        path.write_bytes(path.read_bytes() + b"changed")
        _, ids, _, changed_signature = m.snapshot_images(m.index_images(self.original))
        self.assertNotEqual(changed_signature, self.signature)
        self.assertIsNone(m.read_feature_cache(self.cache, ids, changed_signature))

    def test_shared_csv_merge_preserves_assignments(self):
        """Merge three existing CSVs while preserving every image label and assignment."""
        split_files = {}
        for split in ("train", "val", "test"):
            path = self.root / f"{split}.csv"
            m.write_csv(path, [{key: row[key] for key in ("image_id", "class_id", "class_name")}
                               for row in self.rows if row["split"] == split])
            split_files[split] = path
        destination = self.root / "merged.csv"
        with mock.patch.multiple(merge, SPLIT_FILES=split_files, OUTPUT_CSV=destination):
            merge.main()
        actual, _ = m.load_manifest(destination)
        self.assertEqual({row["image_id"]: (row["class_id"], row["split"]) for row in actual},
                         {row["image_id"]: (row["class_id"], row["split"]) for row in self.rows})

    def test_training_scaler_and_final_evaluation(self):
        """Fit scaling on train only, export test results separately, and detect manifest edits."""
        model_path = m.train("original")
        _, bundle = m.load_model("original")
        rows, _ = m.load_manifest()
        _, x_train, _ = m.select_rows(rows, "train", self.ids, self.features)
        scaler = bundle["pipeline"].named_steps["scaler"]
        self.assertEqual(int(scaler.n_samples_seen_), 9)
        np.testing.assert_allclose(scaler.mean_, x_train.mean(axis=0, dtype=np.float64))
        self.assertLess(abs(scaler.mean_[-1]), 1)
        self.assertFalse(list(model_path.parent.glob("test_*")))
        result = m.test("original")
        self.assertEqual(set(result), {"accuracy", "macro_precision", "macro_recall", "macro_f1"})
        result_dir = next(model_path.parent.glob("test_*"))
        with (result_dir / "test_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
            predictions = list(csv.DictReader(stream))
        self.assertEqual({row["image_id"] for row in predictions},
                         {row["image_id"] for row in self.rows if row["split"] == "test"})
        self.assertEqual(len(predictions), 6)
        self.assertTrue((result_dir / "figures" / "confusion_matrix.png").is_file())
        self.manifest.write_bytes(self.manifest.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            m.test("original")

    def test_variants_keep_caches_models_and_results_separate(self):
        """Keep variant artifacts distinct and export their independently selected models."""
        foreground_images = m.index_images(self.foreground)
        _, ids, hashes, signature = m.snapshot_images(foreground_images)
        foreground_features = np.stack([self.feature_rows[key] for key in ids]).copy()
        foreground_features[:, :3] *= 0.8
        foreground_cache = m.OUTPUT_ROOT / "foreground" / "feature_cache" / "features.npz"
        foreground_cache.parent.mkdir(parents=True)
        np.savez_compressed(foreground_cache, features=foreground_features, image_ids=np.asarray(ids),
                            image_sha256=np.asarray(hashes), metadata_json=json.dumps({
                                "signature": signature, "extractor": m.EXTRACTOR_SPEC,
                                "source": "SYNTHETIC FOREGROUND TEST FEATURES; NOT IMAGENET RESULTS",
                            }))
        paths = {dataset: m.train(dataset) for dataset in ("original", "foreground")}
        self.assertNotEqual(paths["original"], paths["foreground"])
        for dataset, model_path in paths.items():
            self.assertTrue(model_path.is_relative_to(m.OUTPUT_ROOT / dataset / "runs"))
            _, bundle = m.load_model(dataset)
            settings = json.loads((model_path.parent / "final_model_settings.json").read_text())
            self.assertEqual(settings["dataset"], dataset)
            self.assertEqual(settings["search_space"]["C_VALUES"], [0.1, 1.0])
            self.assertEqual(settings["search_space"]["candidate_count"], 4)
            self.assertEqual(settings["svm_parameters"], bundle["pipeline"].named_steps["svm"].get_params())
            self.assertEqual(Path(settings["feature_cache_path"]),
                             m.OUTPUT_ROOT / dataset / "feature_cache" / "features.npz")
            with (model_path.parent / "validation_results.csv").open(encoding="utf-8-sig", newline="") as stream:
                results = list(csv.DictReader(stream))
            self.assertEqual(settings["selected"][m.SELECTION_METRIC],
                             max(float(row[m.SELECTION_METRIC]) for row in results))
            self.assertTrue((model_path.parent / "figures" / "validation_score_vs_C.png").is_file())
            self.assertFalse(list(model_path.parent.glob("test_*")))
        self.assertEqual(m.load_model("foreground")[1]["settings"]["feature_cache_signature"], signature)
        with self.assertRaisesRegex(ValueError, "another image variant"):
            m.load_model("foreground", paths["original"])
        m.test("foreground")
        foreground_test = next(paths["foreground"].parent.glob("test_*"))
        self.assertTrue((foreground_test / "test_predictions.csv").is_file())
        self.assertTrue(foreground_test.is_relative_to(m.OUTPUT_ROOT / "foreground"))

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision"),
                         "torch/torchvision unavailable; real extractor is not tested here")
    def test_extractor_structure_without_weight_download(self):
        """Check freezing, eval mode, native sizes, and 512 outputs using untrained weights."""
        import torch
        import torchvision
        original_constructor = torchvision.models.resnet18
        def no_download(**kwargs):
            """Check the requested weight enum while constructing an untrained model offline."""
            self.assertEqual(kwargs["weights"], torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            return original_constructor(weights=None)
        with mock.patch.object(torchvision.models, "resnet18", side_effect=no_download), mock.patch.object(m, "DEVICE", "cpu"):
            model, transform, device = m.build_extractor()
        self.assertFalse(model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        for width, height in ((300, 250), (193, 117)):
            pixels = transform(Image.new("RGB", (width, height))).unsqueeze(0).to(device)
            self.assertEqual(tuple(pixels.shape), (1, 3, height, width))
            with torch.inference_mode():
                output = model(pixels)
            self.assertEqual(tuple(output.shape), (1, 512))


if __name__ == "__main__":
    print("SYNTHETIC FUNCTIONAL TESTS ONLY — no pet-classification performance is measured.", flush=True)
    unittest.main(verbosity=2)
