"""Reproduce the agreed stratified split; only write lists, never touch images.

Saved split lists were not included in the supplied ZIP. Prefer the exact
lists used in the reported experiment. Run this generator only when deliberately
creating a new shared split; share its saved outputs across all methods.
"""
from pathlib import Path

# Parameters: keep these unchanged once the group starts its experiments.
BASE_DIR = Path(__file__).resolve().parent
ANNOTATIONS_PATH = BASE_DIR / "annotations"
OUTPUT_FOLDER = BASE_DIR / "splits"
SEED = 42
TRAIN_COUNT = 2944
VALIDATION_COUNT = 736
EXPECTED_TEST_COUNT = 3669
EXPECTED_CLASS_COUNT = 37

import csv
import hashlib
import json
import platform
from collections import Counter

import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

from oxford_annotations import load_annotations, validate_manifest


FIELDS = ["image_id", "class_id", "class_name", "split"]


def write_csv(path, rows, fields):
    """Write a consistently ordered, Excel-friendly split table."""
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_splits(annotations):
    # Canonical input order makes the selection independent of TXT row order.
    """Generate a seeded stratified train/val split and retain official test IDs.

    This creates memberships; it is not a recovery of previously saved lists.
    Returns train, val, and test rows after overlap, coverage, and label checks.
    """
    pool = sorted(
        (row for row in annotations["records"] if row["official_split"] == "trainval"),
        key=lambda row: row["image_id"].casefold(),
    )
    test = [row for row in annotations["records"] if row["official_split"] == "test"]
    if len(pool) != TRAIN_COUNT + VALIDATION_COUNT or len(test) != EXPECTED_TEST_COUNT:
        raise ValueError("Annotation counts do not match the agreed dataset")
    classes = {row["class_id"] for row in annotations["classes"]}
    if len(classes) != EXPECTED_CLASS_COUNT:
        raise ValueError("Unexpected number of breeds")
    # Step 1: Stratify only official trainval using the configured sizes and seed.
    # Never redraw an existing experimental manifest just to prepare this repository.
    train, val = train_test_split(
        pool, train_size=TRAIN_COUNT, test_size=VALIDATION_COUNT,
        random_state=SEED, stratify=[row["class_id"] for row in pool],
    )
    # Step 2: Sort output rows for readability; row order is not the random selection order.
    splits = {}
    for name, selected in (("train", train), ("val", val), ("test", test)):
        selected = sorted(selected, key=lambda row: (
            int(row["class_id"]), int(row["image_id"].rpartition("_")[2]),
            row["image_id"].casefold(),
        ))
        splits[name] = [
            {**{field: row[field] for field in FIELDS[:-1]}, "split": name}
            for row in selected
        ]
        if {row["class_id"] for row in splits[name]} != classes:
            raise ValueError(f"A breed is missing from {name}")
    # Step 3: Verify duplicates, overlap, full coverage, and unchanged official test IDs.
    sets = {name: {row["image_id"].casefold() for row in rows}
            for name, rows in splits.items()}
    for name, rows in splits.items():
        if len(sets[name]) != len(rows):
            raise ValueError(f"Duplicate image IDs in {name}")
    if any(sets[a] & sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Split overlap")
    if sets["train"] | sets["val"] != {row["image_id"].casefold() for row in pool}:
        raise ValueError("Train/validation do not exactly cover official trainval")
    if sets["test"] != {row["image_id"].casefold() for row in test}:
        raise ValueError("Official test membership changed")
    if set().union(*sets.values()) != set(annotations["by_key"]):
        raise ValueError("Split union does not equal official classification list")
    validate_manifest([row for rows in splits.values() for row in rows], annotations)
    return splits


def main():
    """Create CSV/ID lists, class counts, and provenance for a newly generated split."""
    annotations = load_annotations(ANNOTATIONS_PATH)
    splits = build_splits(annotations)
    # Step 4: Save the new memberships as reusable CSVs and one-ID-per-line text files.
    # Existing files with these names are overwritten; archive agreed lists beforehand.
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    output_files = []
    for name, rows in splits.items():
        csv_path = OUTPUT_FOLDER / f"{name}.csv"
        ids_path = OUTPUT_FOLDER / f"{name}_ids.txt"
        write_csv(csv_path, rows, FIELDS)
        ids_path.write_text("".join(row["image_id"] + "\n" for row in rows), encoding="utf-8")
        output_files.extend((csv_path, ids_path))
    manifest = OUTPUT_FOLDER / "manifest.csv"
    write_csv(manifest, [row for rows in splits.values() for row in rows], FIELDS)
    output_files.append(manifest)
    counts = {name: Counter(row["class_id"] for row in rows) for name, rows in splits.items()}
    class_rows = []
    for breed in annotations["classes"]:
        cid = breed["class_id"]
        n = {name: counter[cid] for name, counter in counts.items()}
        class_rows.append({
            "class_id": cid, "class_name": breed["class_name"], "species": breed["species"],
            "official_trainval": n["train"] + n["val"], **n, "total": sum(n.values()),
        })
    class_path = OUTPUT_FOLDER / "class_counts.csv"
    write_csv(class_path, class_rows, [
        "class_id", "class_name", "species", "official_trainval", "train", "val", "test", "total",
    ])
    output_files.append(class_path)
    # Step 5: Save seed, package versions, source hashes, and output-file fingerprints.
    summary = {
        "seed": SEED,
        "method": "sklearn.model_selection.train_test_split; stratify=official CLASS-ID",
        "input_order": "official trainval image_id, casefold ascending",
        "counts": {name: len(rows) for name, rows in splits.items()},
        "total_labelled_images": sum(map(len, splits.values())),
        "class_counts": {name: len(counter) for name, counter in counts.items()},
        "checks": {
            "no_duplicates": True, "pairwise_disjoint": True,
            "train_plus_val_equals_official_trainval": True,
            "official_test_membership_unchanged": True,
            "union_equals_official_list": True, "labels_match_annotations": True,
            "all_breeds_present_in_every_split": True,
        },
        "source_annotation_sha256": annotations["summary"]["text_sha256"],
        "file_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output_files},
        "generated_with": {"python": platform.python_version(), "numpy": np.__version__,
                           "scikit-learn": sklearn.__version__},
        "notes": [
            "Use these saved lists for every method, original images, and foreground images.",
            "Images outside official list.txt are excluded. No image is copied, moved, edited or deleted.",
            "No feature extraction, model training, validation scoring or test evaluation has run.",
            "Saved lists are authoritative; do not regenerate independently for other experiments.",
        ],
    }
    (OUTPUT_FOLDER / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({"output_folder": str(OUTPUT_FOLDER), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
