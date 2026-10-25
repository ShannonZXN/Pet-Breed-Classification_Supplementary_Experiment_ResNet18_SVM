"""Regression checks for pixel preservation, matching, and processing reports.

All fixtures are small temporary files. No dataset download is required.
"""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import pet_background_utils as processing


class ProcessingChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pet-processing-tests-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.originals = self.root / "original"
        self.masks = self.root / "masks"
        self.originals.mkdir()
        self.masks.mkdir()
        self.labels = np.full((23, 27), 2, dtype=np.uint8)
        self.labels[3:20, 3:24] = 3
        self.labels[8:15, 8:19] = 1
        self.rgb = np.arange(23 * 27 * 3, dtype=np.uint16).reshape(23, 27, 3)
        self.rgb = (self.rgb % 256).astype(np.uint8)

    def pair(self, stem="Pet", labels=None):
        """Write one JPEG original and its label PNG; return both paths."""
        image_path = self.originals / f"{stem}.jpg"
        mask_path = self.masks / f"{stem}.png"
        Image.fromarray(self.rgb).save(image_path)
        Image.fromarray(self.labels if labels is None else labels).save(mask_path)
        return image_path, mask_path

    def quiet(self, function, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return function(*args, **kwargs)

    def rows(self, path):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            return reader.fieldnames, list(reader)

    def test_radius_zero_keeps_animal_and_edge_without_adding_background(self):
        labels = np.array([[1, 2, 3], [3, 2, 1]], dtype=np.uint8)
        expected = np.array([[True, False, True], [True, False, True]])
        for protection in (False, True):
            with self.subTest(protection=protection):
                np.testing.assert_array_equal(
                    processing.erode_mask(labels, 0, protection), expected
                )

    def test_disk_erosion_and_animal_restoration_have_explicit_expected_masks(self):
        labels = np.full((7, 7), 2, dtype=np.uint8)
        labels[1:6, 1:6] = 3
        labels[1, 1] = 1
        expected = np.zeros((7, 7), dtype=bool)
        expected[2:5, 2:5] = True
        np.testing.assert_array_equal(processing.erode_mask(labels, 1, False), expected)
        expected[1, 1] = True
        np.testing.assert_array_equal(processing.erode_mask(labels, 1, True), expected)
        # A radius larger than the foreground removes all unprotected pixels.
        self.assertFalse(processing.erode_mask(labels, 9, False).any())
        np.testing.assert_array_equal(processing.erode_mask(labels, 9, True), labels == 1)

    def test_image_border_is_not_an_extra_source_of_background(self):
        labels = np.full((3, 4), 3, dtype=np.uint8)
        # Even a large disk retains this entirely foreground image, because the
        # original implementation treats positions beyond its border as kept.
        self.assertTrue(processing.erode_mask(labels, 9, False).all())

    def test_rgb_invariants_across_radii_and_both_protection_modes(self):
        for radius in (0, 1, 9):
            for protection in (False, True):
                with self.subTest(radius=radius, protection=protection):
                    keep = processing.erode_mask(self.labels, radius, protection)
                    result = processing.apply_black_background(self.rgb, keep)
                    self.assertEqual(result.shape, self.rgb.shape)
                    self.assertEqual(result.dtype, np.uint8)
                    np.testing.assert_array_equal(result[keep], self.rgb[keep])
                    self.assertFalse(result[~keep].any())
                    self.assertFalse(keep[self.labels == 2].any())
                    if protection:
                        self.assertTrue(keep[self.labels == 1].all())

    def test_palette_masks_keep_their_indices_instead_of_palette_colors(self):
        image_path, mask_path = self.pair()
        palette = Image.fromarray(self.labels).convert("P")
        palette.putpalette([value for index in range(256)
                            for value in (255 - index, index, (index * 7) % 256)])
        palette.save(mask_path)
        original, labels = processing.load_pair(image_path, mask_path)
        np.testing.assert_array_equal(labels, self.labels)
        with Image.open(image_path) as decoded:
            np.testing.assert_array_equal(original, np.array(decoded.convert("RGB")))

    def test_invalid_mask_formats_labels_and_dimensions_are_rejected(self):
        image_path, mask_path = self.pair()
        invalid = {
            "color preview": self.rgb,
            "sixteen bit": self.labels.astype(np.uint16),
            "different size": self.labels[:-1],
            "unexpected label": np.zeros_like(self.labels),
        }
        for description, values in invalid.items():
            with self.subTest(description=description):
                Image.fromarray(values).save(mask_path)
                with self.assertRaises(ValueError):
                    processing.load_pair(image_path, mask_path)

    def test_radius_values_are_validated_before_sorting_and_deduplication(self):
        for invalid in (-1, True, False, "3", 1.5):
            with self.subTest(radius=invalid):
                with self.assertRaises(ValueError):
                    processing.erode_mask(self.labels, invalid)
        for comparisons in ([True], ["3"]):
            with self.subTest(comparisons=comparisons):
                with self.assertRaises(ValueError):
                    processing.run_sample(
                        self.originals, self.masks, self.root / "out",
                        radius=1, comparison_radii=comparisons,
                    )

    def test_discovery_is_nonrecursive_and_duplicate_stems_are_rejected(self):
        self.pair("Pet")
        (self.originals / "nested").mkdir()
        Image.fromarray(self.rgb).save(self.originals / "nested/Nested.jpg")
        Image.fromarray(self.rgb).save(self.originals / "Ignored.png")
        originals, lookup = processing.discover_inputs(self.originals, self.masks)
        self.assertEqual([path.name for path in originals], ["Pet.jpg"])
        self.assertEqual(set(lookup), {"pet"})
        Image.fromarray(self.rgb).save(self.originals / "Pet.jpeg")
        with self.assertRaisesRegex(ValueError, "Duplicate original"):
            processing.discover_inputs(self.originals, self.masks)
        (self.originals / "Pet.jpeg").unlink()
        Image.fromarray(self.labels).save(self.masks / "PET.PNG")
        if len(list(self.masks.iterdir())) == 2:
            with self.assertRaisesRegex(ValueError, "Duplicate mask"):
                processing.discover_inputs(self.originals, self.masks)

    def test_batch_continues_after_bad_pair_and_reports_missing_masks_as_skips(self):
        image_path, mask_path = self.pair("Valid")
        # Case-insensitive full-stem matching must still find this renamed mask.
        mask_path.rename(self.masks / "vALID.PNG")
        self.pair("Invalid", np.zeros_like(self.labels))
        Image.fromarray(self.rgb).save(self.originals / "Missing.jpg")
        Image.fromarray(self.labels).save(self.masks / "Orphan.png")
        output = self.root / "batch"
        self.assertEqual(self.quiet(processing.run_batch, self.originals,
                                   self.masks, output, radius=1), 1)
        summary = json.loads((output / "run_summary.json").read_text())
        self.assertEqual(summary["matched_images"], 2)
        self.assertEqual(summary["succeeded_images"], 1)
        self.assertEqual(summary["failed_images"], 1)
        self.assertEqual(summary["skipped_missing_masks"], 1)
        self.assertEqual(summary["masks_without_originals"], ["Orphan.png"])
        _, rows = self.rows(output / "processing_log.csv")
        self.assertEqual({row["image"]: row["status"] for row in rows}, {
            "Valid.jpg": "ok", "Invalid.jpg": "error",
            "Missing.jpg": "skipped_missing_mask",
        })
        with Image.open(output / "Valid.png") as saved:
            self.assertEqual(saved.mode, "RGB")
            self.assertEqual(saved.size, (27, 23))
            result = np.array(saved)
        with Image.open(image_path) as decoded:
            original = np.array(decoded.convert("RGB"))
        np.testing.assert_array_equal(result[self.labels == 1], original[self.labels == 1])
        self.assertFalse(result[self.labels == 2].any())

    def test_zero_matched_pairs_still_writes_reports_and_has_success_status(self):
        Image.fromarray(self.rgb).save(self.originals / "Missing.jpg")
        Image.fromarray(self.labels).save(self.masks / "Orphan.png")
        output = self.root / "batch"
        self.assertEqual(self.quiet(processing.run_batch, self.originals,
                                   self.masks, output), 0)
        self.assertEqual({path.name for path in output.iterdir()}, {
            "missing_masks.csv", "masks_without_originals.csv",
            "processing_log.csv", "run_summary.json",
        })
        summary = json.loads((output / "run_summary.json").read_text())
        self.assertEqual(summary["matched_images"], 0)
        self.assertEqual(summary["failed_images"], 0)
        self.assertEqual(summary["skipped_missing_masks"], 1)

    def test_seeded_sample_is_reproducible_and_never_replaces_a_missing_mask(self):
        self.pair("A")
        self.pair("B")
        Image.fromarray(self.rgb).save(self.originals / "C.jpg")
        # With this sorted three-file list and seed, C is the selected original.
        for name in ("first", "second"):
            output = self.root / name
            self.assertEqual(self.quiet(
                processing.run_sample, self.originals, self.masks, output,
                sample_size=1, random_seed=42, radius=1, comparison_radii=[]), 1)
            summary = json.loads((output / "run_summary.json").read_text())
            self.assertEqual(summary["selected_images"], ["C.jpg"])
            self.assertEqual(summary["succeeded_images"], 0)
            _, rows = self.rows(output / "processing_log.csv")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "missing_mask")
            self.assertFalse(list(output.rglob("*.png")))

    def test_legacy_run_keeps_unprotected_behavior_and_its_report_schema(self):
        labels = np.full_like(self.labels, 2)
        labels[11, 13] = 1
        self.pair(labels=labels)
        for legacy, protection in ((True, False), (False, True)):
            output = self.root / ("legacy" if legacy else "current")
            self.assertEqual(self.quiet(
                processing.run_sample, self.originals, self.masks, output,
                sample_size=1, radius=1, comparison_radii=[],
                protect_animal_region=protection, legacy_report=legacy), 0)
            fields, rows = self.rows(output / "processing_log.csv")
            self.assertEqual("protect_animal_region" in fields, not legacy)
            self.assertEqual(int(rows[0]["removed_label1_pixels"]), int(legacy))
            summary = json.loads((output / "run_summary.json").read_text())
            self.assertEqual(summary["erosion"]["protect_label1"], protection)
            with Image.open(output / "r_01/Pet.png") as saved:
                pixels = np.array(saved)
            self.assertEqual(bool(pixels[11, 13].any()), protection)

    def test_processing_output_cannot_overlap_inputs_in_either_direction(self):
        for output in (self.originals, self.originals / "result",
                       self.masks / "result", self.root):
            with self.subTest(output=output):
                with self.assertRaises(ValueError):
                    processing.resolve_folders(self.originals, self.masks, output)


if __name__ == "__main__":
    unittest.main()
