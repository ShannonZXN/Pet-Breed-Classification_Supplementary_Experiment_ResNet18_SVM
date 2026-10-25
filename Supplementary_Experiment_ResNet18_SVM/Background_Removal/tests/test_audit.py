"""Check filename audit semantics without requiring actual dataset images."""

import contextlib
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import audit_pet_outputs_new as audit_tool


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AuditTests(unittest.TestCase):
    """The audit uses filenames and logs, not image decoding or pixel checks."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pet-audit-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.originals = self.root / "originals"
        self.masks = self.root / "masks"
        self.output = self.root / "output"
        for folder in (self.originals, self.masks, self.output):
            folder.mkdir()

    def write_log(self, rows):
        with (self.output / "processing_log.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=["image", "status", "output"])
            writer.writeheader()
            writer.writerows(rows)

    def arguments(self, report="reports"):
        return [
            "--originals", str(self.originals), "--masks", str(self.masks),
            "--output", str(self.output), "--report-dir", str(self.root / report),
            "--unprocessed-dir", str(self.root / "copies"),
        ]

    def test_stale_png_is_unexpected_but_not_absent(self):
        for name in ("Good", "Stale", "Bad", "Missing", "NoMask"):
            (self.originals / f"{name}.jpg").write_bytes(b"original bytes")
            if name != "NoMask":
                (self.masks / f"{name}.png").write_bytes(b"trimap bytes")
        for name in ("Good.PNG", "Stale.png", "Orphan.png", "Extra.jpeg", "notes.txt"):
            (self.output / name).write_bytes(b"file bytes")
        (self.output / "nested").mkdir()
        (self.output / "nested" / "ignored.png").write_bytes(b"not top-level")
        self.write_log([
            {"image": "Good.jpg", "status": "ok", "output": "Good.png"},
            {"image": "Stale.jpg", "status": "error", "output": ""},
            {"image": "Bad.jpg", "status": "error", "output": ""},
        ])

        summary, issues = audit_tool.audit(self.originals, self.masks, self.output)

        self.assertEqual(summary["output_png_files"], 3)
        self.assertEqual(summary["output_other_image_files"], 1)
        self.assertEqual(summary["output_non_image_files"], 2)
        self.assertEqual(summary["output_subfolders"], 1)
        self.assertEqual(summary["output_total_items"], 7)
        self.assertEqual(set(summary["unexpected_output_images"]),
                         {"Stale.png", "Orphan.png", "Extra.jpeg"})
        self.assertEqual(set(summary["missing_paired_outputs"]),
                         {"Bad.png", "Missing.png"})
        reasons = {item["image"]: item["reason"]
                   for item in summary["unprocessed_originals"]}
        self.assertEqual(reasons, {"Bad.jpg": "processing_error",
                                  "Missing.jpg": "missing_output",
                                  "NoMask.jpg": "missing_mask"})
        stale_issue = next(item for item in issues if item["filename"] == "Stale.png")
        self.assertIn("not_listed_as_successful_in_last_run", stale_issue["detail"])

    def test_copying_is_fresh_and_findings_are_informational(self):
        for name in ("One", "Two"):
            (self.originals / f"{name}.jpg").write_bytes(name.encode("ascii"))
        (self.masks / "One.png").write_bytes(b"mask")
        before = {path.name: path.read_bytes() for path in self.originals.iterdir()}

        summaries = []
        for _ in range(2):
            with contextlib.redirect_stdout(io.StringIO()):
                status = audit_tool.main(self.arguments())
            self.assertEqual(status, 0)
            summary = json.loads((self.root / "reports/count_summary.json").read_text())
            self.assertEqual(summary["copied_original_count"], 2)
            self.assertEqual(summary["copy_errors"], 0)
            summaries.append(summary)
            for item in summary["unprocessed_originals"]:
                self.assertEqual(Path(item["copied_path"]).read_bytes(), before[item["image"]])
        self.assertNotEqual(summaries[0]["unprocessed_copy_folder"],
                            summaries[1]["unprocessed_copy_folder"])
        self.assertEqual(before, {path.name: path.read_bytes()
                                  for path in self.originals.iterdir()})

        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(audit_tool.main(self.arguments() + ["--no-copy-unprocessed"]), 0)
        summary = json.loads((self.root / "reports/count_summary.json").read_text())
        self.assertEqual(summary["copied_original_count"], 0)
        self.assertEqual(summary["unprocessed_copy_folder"], "")

        # A failed requested copy is a processing error, unlike audit findings.
        with patch.object(audit_tool.shutil, "copy2", side_effect=OSError("copy blocked")):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(audit_tool.main(self.arguments()), 1)
        summary = json.loads((self.root / "reports/count_summary.json").read_text())
        self.assertEqual(summary["copy_errors"], 2)

    def test_missing_or_malformed_log_falls_back_with_warning(self):
        examples = [
            None,
            "image,status\ncat.jpg,ok\n",
            "image,status,output\ncat.jpg,ok\n",
            "image,status,output\ncat.jpg,ok,cat.png,extra\n",
            "image,status,status,output\ncat.jpg,ok,ok,cat.png\n",
            "image,status,output\ncat.jpg,ok,\n",
        ]
        for contents in examples:
            with self.subTest(contents=contents):
                path = self.output / "processing_log.csv"
                if contents is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_text(contents, encoding="utf-8")
                successful, statuses, warnings = audit_tool.read_processing_log(self.output)
                self.assertIsNone(successful)
                self.assertEqual(statuses, {})
                self.assertTrue(warnings)
                self.assertIn("filenames", warnings[0])

    def test_sample_subfolder_log_is_rejected(self):
        for output_name in ("r_09/cat.png", "r_09\\cat.png"):
            with self.subTest(output_name=output_name):
                self.write_log([{"image": "cat.jpg", "status": "ok", "output": output_name}])
                with self.assertRaisesRegex(ValueError, "flat batch output"):
                    audit_tool.audit(self.originals, self.masks, self.output)

    def test_report_and_copy_destinations_cannot_be_inside_image_folders(self):
        for folder in (self.originals, self.masks, self.output):
            for destination in (folder, folder / "nested"):
                for option in ("--report-dir", "--unprocessed-dir"):
                    with self.subTest(option=option, destination=destination):
                        with self.assertRaisesRegex(ValueError, "outside all input and output"):
                            audit_tool.main(self.arguments() + [option, str(destination)])
                        self.assertFalse((folder / "nested").exists())

    def test_all_entry_scripts_explain_their_cli_without_input_files(self):
        for script in ("remove_pet_background.py", "remove_pet_background_BatchProcessing.py",
                       "remove_pet_background_old.py", "audit_pet_outputs_new.py"):
            with self.subTest(script=script):
                result = subprocess.run(
                    [sys.executable, "-S", str(PROJECT_ROOT / script), "--help"],
                    cwd=self.root, capture_output=True, text=True, timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--originals", result.stdout)
                self.assertIn("--masks", result.stdout)
                self.assertIn("--output", result.stdout)

        # Disabling site-packages proves the actual audit run, not only its help,
        # works without NumPy, Pillow, SciPy, or another third-party package.
        result = subprocess.run(
            [sys.executable, "-S", str(PROJECT_ROOT / "audit_pet_outputs_new.py")]
            + self.arguments() + ["--no-copy-unprocessed"],
            cwd=self.root, capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / "reports/count_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
