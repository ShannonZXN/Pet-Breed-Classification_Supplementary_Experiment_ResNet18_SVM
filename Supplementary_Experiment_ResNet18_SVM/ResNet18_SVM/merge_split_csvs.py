"""Merge the team's EXISTING split CSVs without changing any assignment/ID.

Edit the filenames and column names below. Requires only Python's stdlib.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SPLIT_FILES = {
    "train": BASE_DIR / "splits" / "train.csv",
    "val": BASE_DIR / "splits" / "val.csv",
    "test": BASE_DIR / "splits" / "test.csv",
}
# Left side is the required output column; right side is the team's CSV header.
COLUMNS = {"image_id": "image_id", "class_id": "class_id", "class_name": "class_name"}
OUTPUT_CSV = BASE_DIR / "splits" / "manifest.csv"

import csv


def main():
    """Merge three existing CSVs, reject duplicate IDs, and retain their assignments."""
    # Step 1: Read the three existing CSVs using the configured source-column mapping.
    rows, seen = [], set()
    for split, path in SPLIT_FILES.items():
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not set(COLUMNS.values()).issubset(reader.fieldnames or []):
                raise ValueError(f"Wrong columns in {path}: expected {COLUMNS}")
            for raw in reader:
                if not any(raw.values()):
                    continue
                row = {target: (raw.get(source) or "").strip() for target, source in COLUMNS.items()}
                if not all(row.values()):
                    raise ValueError(f"Empty field in {path}: {raw}")
                # Step 2: Detect cross-file and within-file repeats after normalizing the image ID.
                key = Path(row["image_id"].replace("\\", "/")).stem.casefold()
                if key in seen:
                    raise ValueError(f"Repeated image in shared splits: {row['image_id']}")
                seen.add(key)
                rows.append({**row, "split": split})
    # Step 3: Write unchanged assignments under the required manifest column names.
    # The main pipeline will then validate official labels and test membership.
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_id", "class_id", "class_name", "split"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} unchanged split assignments to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
