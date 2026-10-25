"""Shared image processing and report utilities for the three entry scripts.

The workflow uses supplied Oxford-IIIT Pet trimaps. It does not estimate masks,
train a model, resize images, or use a GPU. All model-ready outputs are opaque
RGB PNGs; removed pixels are black rather than transparent.
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from scipy.ndimage import binary_erosion
except ImportError as exc:
    raise ImportError(
        f"Missing dependency: {exc}. Install with: "
        "python -m pip install -r requirements.txt"
    ) from exc


PROCESSING_FIELDS = [
    "image", "mask", "status", "width", "height", "radius",
    "protect_animal_region", "initial_kept_pixels", "final_kept_pixels",
    "removed_label1_pixels", "removed_label3_pixels", "output", "error",
]


def validate_radius(radius):
    """Require a nonnegative integer radius; booleans are not valid radii."""
    if type(radius) is not int or radius < 0:
        raise ValueError("Radius must be a nonnegative integer")


def resolve_folders(original_folder, mask_folder, output_folder):
    """Resolve folders and prevent generated files from overlapping inputs."""
    original_dir, mask_dir, output_dir = [
        Path(value).expanduser().resolve()
        for value in (original_folder, mask_folder, output_folder)
    ]
    for folder in (original_dir, mask_dir):
        if not folder.is_dir():
            raise FileNotFoundError(f"Folder does not exist: {folder}")
        # Both nesting directions are rejected, including symlink-resolved paths.
        if (output_dir == folder or folder in output_dir.parents
                or output_dir in folder.parents):
            raise ValueError(
                "OUTPUT_FOLDER must be separate from the input folders; "
                "it cannot equal, contain, or be inside either input folder"
            )
    return original_dir, mask_dir, output_dir


def discover_inputs(original_dir, mask_dir):
    """List top-level JPG/JPEG originals and case-insensitive PNG mask stems."""
    originals = sorted(
        (path for path in original_dir.iterdir()
         if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg")),
        key=lambda path: (path.name.casefold(), path.name),
    )
    if not originals:
        raise ValueError(f"No JPG/JPEG images found in: {original_dir}")

    seen_stems = set()
    for path in originals:
        key = path.stem.casefold()
        if key in seen_stems:
            raise ValueError(f"Duplicate original basename: {path.stem}")
        seen_stems.add(key)

    mask_lookup = {}
    for path in sorted(mask_dir.iterdir(), key=lambda path: path.name):
        if path.is_file() and path.suffix.lower() == ".png":
            key = path.stem.casefold()
            if key in mask_lookup:
                raise ValueError(f"Duplicate mask basename: {path.stem}")
            mask_lookup[key] = path
    return originals, mask_lookup


def load_pair(image_path, mask_path):
    """Load original RGB pixels and a matching 2D uint8 trimap of labels 1/2/3.

    No EXIF rotation, resizing, cropping, padding, or normalization is applied.
    Palette PNGs retain their stored indices; rendered RGB mask previews are
    rejected because they do not reliably contain the original labels.
    """
    with Image.open(image_path) as image:
        original = np.array(image.convert("RGB"))
    with Image.open(mask_path) as mask:
        labels = np.array(mask)
    if labels.ndim != 2 or labels.dtype != np.uint8:
        raise ValueError(
            "Mask must contain 8-bit label indices, not a color preview; "
            f"got shape={labels.shape}, dtype={labels.dtype}"
        )
    if labels.shape != original.shape[:2]:
        raise ValueError(
            f"Image/mask size mismatch: image HxW={original.shape[:2]}, "
            f"mask HxW={labels.shape}"
        )
    values = np.unique(labels)
    if not np.isin(values, [1, 2, 3]).all():
        raise ValueError(f"Unexpected mask labels: {values.tolist()}; expected 1/2/3")
    return original, labels


def erode_mask(labels, radius, protect_animal_region=True):
    """Erode labels 1+3 once with a disk and optionally restore all label 1.

    The radius is measured in original-image pixels. With radius zero, all
    label-1 and label-3 pixels remain. The disk footprint spans (2r+1) pixels in
    each direction. Outside-image positions are not additional background.
    """
    validate_radius(radius)
    if type(protect_animal_region) is not bool:
        raise ValueError("PROTECT_ANIMAL_REGION must be True or False")
    keep = (labels == 1) | (labels == 3)
    if radius == 0:
        return keep
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    disk = (xx * xx + yy * yy) <= radius * radius
    eroded = binary_erosion(keep, structure=disk, iterations=1, border_value=1)
    if protect_animal_region:
        # Restore original animal pixels only; label 2 remains excluded.
        eroded |= labels == 1
    return eroded


def apply_black_background(original, keep):
    """Preserve retained decoded RGB pixels and replace every other pixel by 0."""
    removed = np.zeros_like(original)
    removed[keep] = original[keep]
    return removed


def write_csv(path, fields, records):
    """Write an Excel-friendly UTF-8 CSV, including headers for empty reports."""
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def write_summary(path, summary):
    """Save settings and counts as readable JSON with the original report schema."""
    with path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)


def erosion_details(protect_animal_region):
    """Describe the transformation applied after the initial label-1/3 mapping."""
    return {
        "probe_shape": "Euclidean disk", "iterations": 1,
        "protect_label1": protect_animal_region,
        "outside_image_border_value": 1,
    }


def image_record(image_path, mask_path, original, labels, keep, radius,
                 protect_animal_region, output_name):
    """Calculate the per-image audit statistics without changing image pixels."""
    initial = (labels == 1) | (labels == 3)
    return {
        "image": image_path.name, "mask": mask_path.name, "status": "ok",
        "width": original.shape[1], "height": original.shape[0],
        "radius": radius, "protect_animal_region": protect_animal_region,
        "initial_kept_pixels": int(initial.sum()),
        "final_kept_pixels": int(keep.sum()),
        "removed_label1_pixels": int(((labels == 1) & ~keep).sum()),
        "removed_label3_pixels": int(((labels == 3) & ~keep).sum()),
        "output": output_name,
    }


def save_comparison(original, results, destination, image_name):
    """Save a labeled sheet of full-size panels, separate from model inputs."""
    height, width = original.shape[:2]
    panels = [("Original", original)] + [
        (f"r = {radius}px", results[radius]) for radius in sorted(results)
    ]
    columns = min(3, len(panels))
    header_height = 32
    rows = math.ceil(len(panels) / columns)
    sheet = Image.new(
        "RGB", (columns * width, rows * (height + header_height)), (242, 244, 247)
    )
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    for index, (label, pixels) in enumerate(panels):
        x = (index % columns) * width
        y = (index // columns) * (height + header_height)
        draw.text((x + 10, y + 6), label, fill=(25, 35, 45), font=font)
        sheet.paste(Image.fromarray(pixels), (x, y + header_height))
    destination.mkdir(parents=True, exist_ok=True)
    sheet.save(destination / f"{image_name}_comparison.png")


def run_sample(original_folder, mask_folder, output_folder, *, radius=9,
               protect_animal_region=True, sample_size=35, random_seed=42,
               comparison_radii=(0, 3, 6, 9, 12), legacy_report=False):
    """Compare radii on a seeded sample drawn from originals before mask matching.

    Missing masks are not replaced by other samples. The exit status is 1 if any
    selected image fails, including a selected original without a mask. The
    legacy wrapper uses the same engine with protection disabled and omits the
    protection column, preserving its historical CSV schema.
    """
    # Step 1: Validate every supplied radius before deduplication. Otherwise a
    # boolean can disappear into the set because Python treats True as 1.
    comparison_radii = list(comparison_radii)
    requested_radii = [radius] + comparison_radii
    for value in requested_radii:
        validate_radius(value)
    radii = sorted(set(requested_radii))
    if type(protect_animal_region) is not bool:
        raise ValueError("PROTECT_ANIMAL_REGION must be True or False")
    if type(sample_size) is not int or sample_size < 1:
        raise ValueError("SAMPLE_SIZE must be a positive integer")
    if type(random_seed) is not int:
        raise ValueError("RANDOM_SEED must be an integer")
    original_dir, mask_dir, output_dir = resolve_folders(
        original_folder, mask_folder, output_folder
    )

    # Step 2: Discover deterministic filenames, then sample originals first.
    originals, mask_lookup = discover_inputs(original_dir, mask_dir)
    selected = random.Random(random_seed).sample(
        originals, min(sample_size, len(originals))
    )
    missing = [path.name for path in originals if path.stem.casefold() not in mask_lookup]
    print(f"Original JPG/JPEG files: {len(originals)}")
    print(f"Originals without a matching PNG mask: {len(missing)}")
    if len(originals) < sample_size:
        print(f"Requested {sample_size} images; only {len(originals)} available.")
    print(f"Random seed: {random_seed}; selected images: {len(selected)}")
    print(f"Main radius r: {radius}; all output radii: {radii}")
    print(f"Protect animal region (label 1): {protect_animal_region}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    succeeded = 0

    # Step 3: Process each selected image at every requested radius. Report a
    # bad pair and continue; do not silently select a replacement original.
    for index, image_path in enumerate(selected, start=1):
        mask_path = mask_lookup.get(image_path.stem.casefold())
        if mask_path is None:
            print(f"[{index}/{len(selected)}] MISSING MASK: {image_path.name}")
            records.append({
                "image": image_path.name, "status": "missing_mask",
                "error": f"No matching mask for {image_path.stem}.png",
            })
            continue
        try:
            original, labels = load_pair(image_path, mask_path)
            results = {}
            image_records = []
            for current_radius in radii:
                keep = erode_mask(labels, current_radius, protect_animal_region)
                removed = apply_black_background(original, keep)
                result_dir = output_dir / f"r_{current_radius:02d}"
                result_dir.mkdir(parents=True, exist_ok=True)
                result_path = result_dir / f"{image_path.stem}.png"
                Image.fromarray(removed).save(result_path)
                results[current_radius] = removed
                record = image_record(
                    image_path, mask_path, original, labels, keep, current_radius,
                    protect_animal_region, result_path.relative_to(output_dir).as_posix(),
                )
                if legacy_report:
                    record.pop("protect_animal_region")
                image_records.append(record)
            if comparison_radii:
                save_comparison(original, results, output_dir / "comparisons", image_path.stem)
            records.extend(image_records)
            succeeded += 1
            print(
                f"[{index}/{len(selected)}] OK: {image_path.name} "
                f"({original.shape[1]}x{original.shape[0]})"
            )
        except (OSError, ValueError) as exc:
            print(f"[{index}/{len(selected)}] ERROR: {image_path.name}: {exc}")
            records.append({
                "image": image_path.name, "mask": mask_path.name,
                "status": "error", "error": str(exc),
            })

    # Step 4: Save the historical report fields and the exact selected filenames.
    fields = [field for field in PROCESSING_FIELDS
              if not (legacy_report and field == "protect_animal_region")]
    write_csv(output_dir / "processing_log.csv", fields, records)
    write_summary(output_dir / "run_summary.json", {
        "original_folder": str(original_dir), "mask_folder": str(mask_dir),
        "output_folder": str(output_dir), "r": radius,
        "comparison_radii": comparison_radii, "all_output_radii": radii,
        "sample_size_requested": sample_size, "random_seed": random_seed,
        "selected_images": [path.name for path in selected],
        "succeeded_images": succeeded, "failed_images": len(selected) - succeeded,
        "originals_missing_masks": missing,
        "mask_mapping": {"1": "keep", "2": "remove", "3": "keep"},
        "erosion": erosion_details(protect_animal_region),
    })
    print(f"Completed: {succeeded}/{len(selected)} selected images succeeded.")
    print(f"Outputs: {output_dir}")
    return 0 if succeeded == len(selected) else 1


def run_batch(original_folder, mask_folder, output_folder, *, radius=9,
              protect_animal_region=True):
    """Process all matched originals and write filename audits before image I/O.

    Originals without a mask are intentional skips. Only per-pair processing
    failures cause exit status 1; missing masks alone do not fail a batch run.
    """
    # Step 1: Validate settings and discover all directly contained input files.
    validate_radius(radius)
    if type(protect_animal_region) is not bool:
        raise ValueError("PROTECT_ANIMAL_REGION must be True or False")
    original_dir, mask_dir, output_dir = resolve_folders(
        original_folder, mask_folder, output_folder
    )
    originals, mask_lookup = discover_inputs(original_dir, mask_dir)
    original_stems = {path.stem.casefold() for path in originals}
    pairs = [(path, mask_lookup[path.stem.casefold()])
             for path in originals if path.stem.casefold() in mask_lookup]
    unmatched_originals = [path for path in originals if path.stem.casefold() not in mask_lookup]
    unmatched_masks = [path for key, path in mask_lookup.items() if key not in original_stems]

    # Step 2: Write audits before opening any image or mask. Header-only reports
    # are still created when every filename has a match.
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "missing_masks.csv",
              ["image", "original_path", "expected_mask", "status"], [{
                  "image": path.name, "original_path": str(path),
                  "expected_mask": str(mask_dir / f"{path.stem}.png"),
                  "status": "skipped_missing_mask",
              } for path in unmatched_originals])
    write_csv(output_dir / "masks_without_originals.csv", ["mask", "mask_path", "status"], [{
        "mask": path.name, "mask_path": str(path), "status": "unused_mask",
    } for path in unmatched_masks])
    print(f"Original JPG/JPEG files: {len(originals)}")
    print(f"PNG mask files: {len(mask_lookup)}")
    print(f"Matched image/mask pairs: {len(pairs)}")
    print(f"Originals skipped (no matching mask): {len(unmatched_originals)}")
    print(f"Unused masks (no matching original): {len(unmatched_masks)}")
    print(f"Audit reports: {output_dir / 'missing_masks.csv'}")
    print(f"               {output_dir / 'masks_without_originals.csv'}")
    print(f"Processing ALL {len(pairs)} matched images with radius r = {radius}")
    print(f"Protect animal region (label 1): {protect_animal_region}")
    records = [{
        "image": path.name, "status": "skipped_missing_mask",
        "error": f"No matching mask for {path.stem}.png",
    } for path in unmatched_originals]
    succeeded = 0
    failed = 0

    # Step 3: Apply one common radius to each pair, preserving dimensions and
    # retained decoded RGB values. Each result is a separate opaque RGB PNG.
    for index, (image_path, mask_path) in enumerate(pairs, start=1):
        try:
            original, labels = load_pair(image_path, mask_path)
            keep = erode_mask(labels, radius, protect_animal_region)
            removed = apply_black_background(original, keep)
            result_path = output_dir / f"{image_path.stem}.png"
            Image.fromarray(removed).save(result_path)
            records.append(image_record(
                image_path, mask_path, original, labels, keep, radius,
                protect_animal_region, result_path.name,
            ))
            succeeded += 1
            print(
                f"[{index}/{len(pairs)}] OK: {image_path.name} "
                f"({original.shape[1]}x{original.shape[0]})"
            )
        except (OSError, ValueError) as exc:
            failed += 1
            print(f"[{index}/{len(pairs)}] ERROR: {image_path.name}: {exc}")
            records.append({
                "image": image_path.name, "mask": mask_path.name,
                "status": "error", "error": str(exc),
            })

    # Step 4: Save processing statistics and report success independently of skips.
    write_csv(output_dir / "processing_log.csv", PROCESSING_FIELDS, records)
    write_summary(output_dir / "run_summary.json", {
        "original_folder": str(original_dir), "mask_folder": str(mask_dir),
        "output_folder": str(output_dir), "r": radius,
        "total_images": len(originals), "total_masks": len(mask_lookup),
        "matched_images": len(pairs), "succeeded_images": succeeded,
        "failed_images": failed, "skipped_missing_masks": len(unmatched_originals),
        "originals_missing_masks": [path.name for path in unmatched_originals],
        "masks_without_originals": [path.name for path in unmatched_masks],
        "mask_mapping": {"1": "keep", "2": "remove", "3": "keep"},
        "erosion": erosion_details(protect_animal_region),
    })
    print(f"Completed: {succeeded} succeeded, {failed} errors, "
          f"{len(unmatched_originals)} skipped (no mask).")
    print(f"Outputs: {output_dir}")
    return 1 if failed else 0
