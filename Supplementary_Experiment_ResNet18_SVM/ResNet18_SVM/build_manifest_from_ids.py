"""Fill breed labels from official annotations using the team's validation IDs.

This tool does not select validation samples. It requires the already agreed
validation-ID list. Train is the remaining official trainval; test stays fixed.
If the team supplies three complete CSVs instead, use merge_split_csvs.py.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ANNOTATIONS_PATH = BASE_DIR / "annotations"
VALIDATION_IDS_TXT = BASE_DIR / "splits" / "val_ids.txt"
OUTPUT_CSV = BASE_DIR / "splits" / "manifest.csv"
CLASS_ID_MAPPING = None  # Keep this complete ID map consistent with resnet18_svm.py.

import csv

from oxford_annotations import load_annotations, validate_manifest


def main():
    """Fill official labels around the supplied validation IDs without choosing new IDs."""
    # Step 1: Load official labels and validate any explicit shared class-ID mapping.
    annotations = load_annotations(ANNOTATIONS_PATH)
    validate_manifest([], annotations, class_id_mapping=CLASS_ID_MAPPING)
    if not VALIDATION_IDS_TXT.is_file():
        raise FileNotFoundError(f"Supply the team's agreed validation IDs first: {VALIDATION_IDS_TXT}")
    # Step 2: Read the already agreed validation IDs; reject duplicates and non-trainval IDs.
    validation = set()
    for line in VALIDATION_IDS_TXT.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key = Path(line.split()[0].replace("\\", "/")).stem.casefold()
        if key in validation:
            raise ValueError(f"Duplicate validation image: {key}")
        official = annotations["by_key"].get(key)
        if official is None or official["official_split"] != "trainval":
            raise ValueError(f"Validation ID must belong to official trainval: {key}")
        validation.add(key)
    if not validation:
        raise ValueError("Validation-ID file is empty")
    mapping = {str(k): str(v) for k, v in CLASS_ID_MAPPING.items()} if CLASS_ID_MAPPING is not None else None
    # Step 3: Keep official test fixed; remaining trainval rows become train.
    rows = []
    for record in annotations["records"]:
        split = "test" if record["official_split"] == "test" else (
            "val" if record["image_id"].casefold() in validation else "train"
        )
        rows.append({
            "image_id": record["image_id"],
            "class_id": mapping[record["class_id"]] if mapping is not None else record["class_id"],
            "class_name": record["class_name"], "split": split,
        })
    # Step 4: Check labels and retain at least one training example from every class.
    validate_manifest(rows, annotations, keep_official_test=True, class_id_mapping=CLASS_ID_MAPPING)
    train_classes = {row["class_id"] for row in rows if row["split"] == "train"}
    if train_classes != {row["class_id"] for row in rows}:
        raise ValueError("The agreed validation list removes all training samples from a class")
    # Step 5: Write a manifest without moving or modifying source images.
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_id", "class_id", "class_name", "split"])
        writer.writeheader()
        writer.writerows(rows)
    counts = {split: sum(row["split"] == split for row in rows) for split in ("train", "val", "test")}
    print(f"Saved official labels with the team's validation assignments: {OUTPUT_CSV}")
    print(counts)


if __name__ == "__main__":
    main()
