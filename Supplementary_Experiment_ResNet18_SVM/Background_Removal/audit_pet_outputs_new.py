"""Audit a flat batch-output folder and optionally copy unprocessed originals.

Run ``python audit_pet_outputs_new.py --help`` for command-line options, or edit
the settings below and run the script without arguments. This tool uses only
Python's standard library. It examines top-level filenames and the last batch
log; it does not open images, inspect pixels, or validate segmentation quality.

Audit findings are informational: missing or unexpected outputs do not make the
process exit with an error. Setup/read failures and copy failures produce exit
code 1. Source files are never moved or deleted.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


# -------------------------- Editable defaults ---------------------------
BASE_DIR = Path(__file__).resolve().parent
ORIGINAL_FOLDER = BASE_DIR / "original"
MASK_FOLDER = BASE_DIR / "masks"
OUTPUT_FOLDER = BASE_DIR / "background_removed_all"
AUDIT_REPORT_FOLDER = BASE_DIR / "image_count_audit"

# Copies go into a new run_* subfolder each time, keeping earlier copies separate.
COPY_UNPROCESSED_ORIGINALS = True
UNPROCESSED_FOLDER = BASE_DIR / "unprocessed_originals"
# Windows path example: ORIGINAL_FOLDER = Path(r"D:\Pets\images")
# ------------------------------------------------------------------------

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"
}


def read_processing_log(output_dir: Path):
    """Return successful flat filenames and image statuses, or warn on bad logs.

    An absent/unreadable log does not prevent counting files. In that case,
    membership in the latest recorded run cannot be checked. Nested output paths
    identify a sample/comparison workflow, which this flat-folder audit rejects.
    """
    log_path = output_dir / "processing_log.csv"
    warnings = []
    if not log_path.is_file():
        warnings.append(
            "processing_log.csv is absent; counts use filenames only and cannot "
            "verify membership in the latest recorded run."
        )
        return None, {}, warnings

    try:
        with log_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"status", "output"}.issubset(reader.fieldnames or []):
                raise ValueError("required columns status/output are absent")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValueError("duplicate CSV column names")
            rows = list(reader)

        # Ragged CSV rows can produce None cells or a list under the None key.
        # Reject the whole log instead of trusting only part of its run history.
        for number, row in enumerate(rows, start=2):
            if None in row or any(not isinstance(value, str) for value in row.values()):
                raise ValueError(f"malformed CSV cells on line {number}")
            if row["status"] == "ok" and not row["output"]:
                raise ValueError(f"successful row has no output on line {number}")
    except (OSError, UnicodeError, ValueError, csv.Error) as exc:
        warnings.append(
            f"Could not read processing_log.csv: {exc}. Counts use filenames "
            "only; last-run membership is unavailable."
        )
        return None, {}, warnings

    # Check this outside the fallback handler: sample outputs must not silently
    # become flat filenames and then be reported as missing batch outputs.
    if any("/" in row["output"].replace("\\", "/") for row in rows if row["output"]):
        raise ValueError(
            "This audit is for flat batch output, but processing_log.csv contains "
            "nested output paths. Select the folder produced by "
            "remove_pet_background_BatchProcessing.py, not a sample/comparison folder."
        )

    successful_outputs = {
        row["output"].casefold()
        for row in rows if row["status"] == "ok" and row["output"]
    }
    run_statuses = {
        row["image"].casefold(): row["status"]
        for row in rows if row.get("image")
    }
    return successful_outputs, run_statuses, warnings


def audit(original_dir: Path, mask_dir: Path, output_dir: Path):
    """Count top-level files and identify missing, unexpected, or stale outputs.

    An existing PNG satisfies the filename-existence check even if the last log
    does not list it as successful. Such a file is separately flagged as an
    unexpected output. It is not copied as an unprocessed original.
    """
    # Step 1: require the three image folders, then inventory direct children.
    for folder in (original_dir, mask_dir, output_dir):
        if not folder.is_dir():
            raise FileNotFoundError(f"Folder does not exist: {folder}")
    originals = sorted(
        path for path in original_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    )
    masks = sorted(
        path for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    )
    entries = sorted(output_dir.iterdir())
    images = [
        path for path in entries
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    pngs = [path for path in images if path.suffix.lower() == ".png"]
    other_files = [
        path for path in entries
        if path.is_file() and path.suffix.lower() not in IMAGE_EXTENSIONS
    ]
    folders = [path for path in entries if path.is_dir()]

    # Step 2: match complete stems and output names without considering case.
    # Lists preserve duplicates so ambiguous input stems can be reported.
    original_lookup = defaultdict(list)
    mask_lookup = defaultdict(list)
    for path in originals:
        original_lookup[path.stem.casefold()].append(path)
    for path in masks:
        mask_lookup[path.stem.casefold()].append(path)
    png_names = Counter(path.name.casefold() for path in pngs)
    successful_outputs, run_statuses, warnings = read_processing_log(output_dir)
    issues = []

    # Step 3: flag extra images, duplicate names, and files absent from the log.
    unexpected = []
    for path in images:
        key = path.stem.casefold()
        reasons = []
        if path.suffix.lower() != ".png":
            reasons.append("not_png_output")
        if key not in original_lookup:
            reasons.append("no_matching_original")
        elif key not in mask_lookup:
            reasons.append("original_has_no_mask")
        if len(original_lookup.get(key, [])) > 1:
            reasons.append("ambiguous_original_stem")
        if png_names[path.name.casefold()] > 1:
            reasons.append("duplicate_output_name_ignoring_case")
        if successful_outputs is not None and path.name.casefold() not in successful_outputs:
            reasons.append("not_listed_as_successful_in_last_run")
        if reasons:
            unexpected.append(path.name)
            issues.append({
                "category": "unexpected_output", "filename": path.name,
                "path": str(path), "detail": "; ".join(reasons),
            })

    # Step 4: find originals with no same-stem PNG at all, regardless of log status.
    missing = []
    unprocessed = []
    for path in originals:
        key = path.stem.casefold()
        expected = f"{path.stem}.png"
        if expected.casefold() not in png_names:
            has_mask = key in mask_lookup
            last_status = run_statuses.get(path.name.casefold(), "")
            reason = "missing_mask" if not has_mask else (
                "processing_error" if last_status == "error" else "missing_output"
            )
            unprocessed.append({
                "image": path.name, "original_path": str(path),
                "expected_output": str(output_dir / expected),
                "mask_exists": has_mask,
                "mask_path": str(mask_lookup[key][0]) if has_mask else "",
                "reason": reason, "last_run_status": last_status,
                "copied_path": "", "copy_error": "",
            })
            issues.append({
                "category": "unprocessed_original", "filename": path.name,
                "path": str(path), "detail": reason,
            })
            if has_mask:
                missing.append(expected)
                issues.append({
                    "category": "missing_paired_output", "filename": expected,
                    "path": str(output_dir / expected), "detail": f"original={path.name}",
                })

    # Step 5: explain why a folder's total item count differs from its PNG count.
    for path in other_files:
        issues.append({
            "category": "non_image_file", "filename": path.name,
            "path": str(path), "detail": "Counts as a folder item, not an image",
        })
    for path in folders:
        issues.append({
            "category": "subfolder", "filename": path.name, "path": str(path),
            "detail": "Contents are excluded from these top-level image counts",
        })
    for kind, lookup in (("original", original_lookup), ("mask", mask_lookup)):
        for paths in lookup.values():
            if len(paths) > 1:
                issues.append({
                    "category": f"duplicate_{kind}_stem", "filename": paths[0].stem,
                    "path": "", "detail": "; ".join(str(path) for path in paths),
                })

    # Step 6: include the recorded batch totals as context, not proof of validity.
    last_run = None
    summary_path = output_dir / "run_summary.json"
    if summary_path.is_file():
        try:
            value = json.loads(summary_path.read_text(encoding="utf-8-sig"))
            if not isinstance(value, dict):
                raise ValueError("run summary must be a JSON object")
            last_run = {key: value.get(key) for key in (
                "total_images", "matched_images", "succeeded_images", "failed_images",
                "skipped_missing_masks",
            )}
        except (OSError, UnicodeError, ValueError) as exc:
            warnings.append(f"Could not read run_summary.json: {exc}")
    summary = {
        "original_jpg_jpeg_files": len(originals), "png_mask_files": len(masks),
        "paired_originals": sum(path.stem.casefold() in mask_lookup for path in originals),
        "output_png_files": len(pngs), "output_other_image_files": len(images) - len(pngs),
        "output_all_image_files": len(images), "output_non_image_files": len(other_files),
        "output_subfolders": len(folders), "output_total_items": len(entries),
        "non_image_filenames": [path.name for path in other_files],
        "unexpected_output_images": unexpected, "missing_paired_outputs": missing,
        "unprocessed_original_count": len(unprocessed), "unprocessed_originals": unprocessed,
        "last_run": last_run, "warnings": warnings,
    }
    return summary, issues


def build_parser() -> argparse.ArgumentParser:
    """Expose folder overrides while retaining the editable defaults above."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--originals", type=Path, default=ORIGINAL_FOLDER,
                        help="Folder directly containing original JPG/JPEG images.")
    parser.add_argument("--masks", type=Path, default=MASK_FOLDER,
                        help="Folder directly containing PNG trimap masks.")
    parser.add_argument("--output", type=Path, default=OUTPUT_FOLDER,
                        help="Flat output folder produced by the batch script.")
    parser.add_argument("--report-dir", type=Path, default=AUDIT_REPORT_FOLDER,
                        help="Separate destination for JSON and CSV audit reports.")
    parser.add_argument("--unprocessed-dir", type=Path, default=UNPROCESSED_FOLDER,
                        help="Parent folder for fresh run_* folders of copied originals.")
    parser.add_argument("--no-copy-unprocessed", dest="copy_unprocessed",
                        action="store_false", default=COPY_UNPROCESSED_ORIGINALS,
                        help="Write reports without copying originals that lack a PNG.")
    return parser


def require_separate_destination(destination: Path, image_folders, setting: str):
    """Keep report/copy writes outside the input and output image folders."""
    if any(destination == folder or folder in destination.parents for folder in image_folders):
        raise ValueError(f"{setting} must be outside all input and output image folders")


def write_csv(path: Path, fields, records):
    """Write an Excel-friendly UTF-8 CSV report with a header even if empty."""
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main(argv=None) -> int:
    """Run the audit; main() uses defaults, and main([...]) applies CLI overrides."""
    args = build_parser().parse_args([] if argv is None else argv)

    # Step 1: resolve every path and validate destinations before writing anything.
    original_dir, mask_dir, output_dir, report_dir, copies_root = [
        Path(value).expanduser().resolve() for value in (
            args.originals, args.masks, args.output, args.report_dir, args.unprocessed_dir,
        )
    ]
    image_folders = (original_dir, mask_dir, output_dir)
    require_separate_destination(report_dir, image_folders, "AUDIT_REPORT_FOLDER")
    if type(args.copy_unprocessed) is not bool:
        raise ValueError("COPY_UNPROCESSED_ORIGINALS must be True or False")
    if args.copy_unprocessed:
        require_separate_destination(copies_root, image_folders, "UNPROCESSED_FOLDER")

    # Step 2: inventory filenames and last-run information before creating reports.
    summary, issues = audit(original_dir, mask_dir, output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    summary["unprocessed_copy_folder"] = ""
    summary["copied_original_count"] = 0
    copy_errors = 0

    # Step 3: copy only originals whose corresponding output PNG is absent.
    # A unique destination keeps earlier runs separate; originals stay in place.
    if args.copy_unprocessed and summary["unprocessed_originals"]:
        copies_root.mkdir(parents=True, exist_ok=True)
        copies_dir = Path(tempfile.mkdtemp(prefix="run_", dir=copies_root))
        summary["unprocessed_copy_folder"] = str(copies_dir)
        for item in summary["unprocessed_originals"]:
            destination = copies_dir / item["image"]
            try:
                shutil.copy2(item["original_path"], destination)
                item["copied_path"] = str(destination)
                summary["copied_original_count"] += 1
            except OSError as exc:
                copy_errors += 1
                item["copy_error"] = str(exc)
                print(f"COPY ERROR: {item['image']}: {exc}")
    summary["copy_errors"] = copy_errors

    # Step 4: save the same JSON and CSV report schemas as the original package.
    (report_dir / "count_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(report_dir / "file_audit.csv",
              ["category", "filename", "path", "detail"], issues)
    write_csv(report_dir / "unprocessed_originals.csv", [
        "image", "original_path", "expected_output", "mask_exists", "mask_path",
        "reason", "last_run_status", "copied_path", "copy_error",
    ], summary["unprocessed_originals"])

    # Step 5: print concise counts and the first 20 entries in each issue list.
    for key in ("original_jpg_jpeg_files", "png_mask_files", "paired_originals",
                "output_png_files", "output_other_image_files", "output_non_image_files",
                "output_subfolders", "output_total_items"):
        print(f"{key}: {summary[key]}")
    print("Non-image files:", ", ".join(summary["non_image_filenames"]) or "none")
    print("Unexpected output images:", len(summary["unexpected_output_images"]))
    for name in summary["unexpected_output_images"][:20]:
        print(f"  {name}")
    print("Missing paired outputs:", len(summary["missing_paired_outputs"]))
    for name in summary["missing_paired_outputs"][:20]:
        print(f"  {name}")
    print("Originals without an output PNG:", summary["unprocessed_original_count"])
    for item in summary["unprocessed_originals"][:20]:
        print(f"  {item['image']}  ({item['reason']})")
    print("Unprocessed original list:", report_dir / "unprocessed_originals.csv")
    if summary["unprocessed_copy_folder"]:
        print(f"Copied {summary['copied_original_count']} originals to: "
              f"{summary['unprocessed_copy_folder']}")
    if summary["last_run"] is not None:
        print("Last run:", json.dumps(summary["last_run"], ensure_ascii=False))
    for message in summary["warnings"]:
        print("WARNING:", message)
    print(f"Full reports: {report_dir}")

    # Findings are informational. Only failed copy operations fail a finished audit.
    return 1 if copy_errors else 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
