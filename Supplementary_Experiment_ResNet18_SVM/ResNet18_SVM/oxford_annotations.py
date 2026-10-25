"""Read and audit the uploaded Oxford-IIIT Pet annotation text files.

No split is created here. Official trainval remains trainval until the team
assigns its training/validation membership. Official test labels stay test.
Run `python oxford_annotations.py` to inspect the bundled annotation texts,
or `python oxford_annotations.py --annotations path/to/annotations.zip`.
"""
from pathlib import Path

ANNOTATIONS_PATH = Path(__file__).resolve().parent / "annotations"
REPORT_FOLDER = Path(__file__).resolve().parent / "annotation_audit"

import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter


def _contents(source):
    """Read the three annotation texts and optionally index trimap names without extraction."""
    names = ["list.txt", "trainval.txt", "test.txt"]
    if source.is_dir():
        texts = {name: (source / name).read_bytes() for name in names}
        folder = source / "trimaps"
        masks = {p.stem for p in folder.iterdir()
                 if p.is_file() and p.suffix.lower() == ".png"} if folder.is_dir() else None
    elif source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as z:
            texts = {}
            for name in names:
                matches = [item for item in z.infolist()
                           if not item.is_dir() and Path(item.filename).name == name
                           and not item.filename.startswith("__MACOSX/")]
                if len(matches) != 1:
                    raise ValueError(f"Expected one {name} in {source}; found {len(matches)}")
                texts[name] = z.read(matches[0])
            has_trimaps = any("/trimaps/" in "/" + item.filename for item in z.infolist())
            masks = {Path(item.filename).stem for item in z.infolist()
                     if not item.is_dir() and "/trimaps/" in "/" + item.filename
                     and Path(item.filename).suffix.lower() == ".png"} if has_trimaps else None
    else:
        raise FileNotFoundError(f"Expected annotation directory or ZIP: {source}")
    return texts, masks


def load_annotations(source=ANNOTATIONS_PATH):
    """Parse the supplied official labels and audit their internal consistency.

    Returns records, a normalized-key index, class descriptions, and a summary.
    Column 2 is the global breed target; column 4 is only species-local.
    """
    source = Path(source).expanduser().resolve()
    texts, masks = _contents(source)
    # Step 1: Read exactly four columns per label row, ignoring headers and blank lines.
    tables = {}
    for filename, data in texts.items():
        table = {}
        for line_number, line in enumerate(data.decode("utf-8-sig").splitlines(), start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"{filename}:{line_number}: expected 4 columns")
            image_id, category, species, local_breed = parts
            category, species, local_breed = int(category), int(species), int(local_breed)
            if category not in range(1, 38) or species not in (1, 2) or local_breed < 1:
                raise ValueError(f"Invalid annotation at {filename}:{line_number}")
            prefix, separator, suffix = image_id.rpartition("_")
            if not separator or not prefix or not suffix.isdigit():
                raise ValueError(f"Unrecognized Oxford image ID: {image_id}")
            key = image_id.casefold()
            if key in table:
                raise ValueError(f"Duplicate annotation ID in {filename}: {image_id}")
            table[key] = {
                "image_id": image_id, "class_id": str(category), "class_name": prefix,
                "species_id": str(species), "species": "cat" if species == 1 else "dog",
                "species_breed_id": str(local_breed),
            }
        tables[filename] = table
    # Step 2: Verify disjoint official splits, their union, and consistent row labels.
    full, trainval, test = (tables[name] for name in ("list.txt", "trainval.txt", "test.txt"))
    if set(trainval) & set(test):
        raise ValueError("Official trainval/test files overlap")
    if set(full) != set(trainval) | set(test):
        raise ValueError("list.txt is not exactly the union of trainval.txt and test.txt")
    for filename in ("trainval.txt", "test.txt"):
        for key, row in tables[filename].items():
            if row != full[key]:
                raise ValueError(f"Conflicting labels in {filename}: {row['image_id']}")
    # Step 3: Build the global breed map and distinguish it from species-local IDs.
    classes, prefix_ids = {}, {}
    for row in full.values():
        cid = row["class_id"]
        description = {key: row[key] for key in (
            "class_id", "class_name", "species_id", "species", "species_breed_id"
        )}
        if cid in classes and description != classes[cid]:
            raise ValueError(f"Conflicting class mapping: {cid}")
        if row["class_name"] in prefix_ids and prefix_ids[row["class_name"]] != cid:
            raise ValueError(f"Breed maps to multiple class IDs: {row['class_name']}")
        classes[cid] = description
        prefix_ids[row["class_name"]] = cid
        row["official_split"] = "trainval" if row["image_id"].casefold() in trainval else "test"
    records = list(full.values())
    # Step 4: Report trimap coverage only when masks were actually supplied.
    # The repository bundles text annotations, so mask inventory is normally unknown.
    mask_keys = {name.casefold() for name in masks} if masks is not None else None
    extra_masks = sorted(name for name in masks if name.casefold() not in full) if masks is not None else None
    summary = {
        "source": str(source), "labelled_images": len(full),
        "official_trainval": len(trainval), "official_test": len(test),
        "trainval_test_overlap": 0, "list_equals_split_union": True,
        "class_count": len(classes), "class_ids": sorted(classes, key=int),
        "species_class_counts": dict(Counter(row["species"] for row in classes.values())),
        "species_image_counts": dict(Counter(row["species"] for row in records)),
        "trimap_count": len(masks) if masks is not None else None,
        "labelled_images_without_trimaps": sorted(row["image_id"] for key, row in full.items()
                                                   if mask_keys is not None and key not in mask_keys),
        "trimaps_outside_classification_lists": extra_masks,
        "species_local_breed_ids": {
            species: sorted({int(row["species_breed_id"]) for row in records if row["species"] == species})
            for species in ("cat", "dog")
        },
        "text_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in texts.items()},
        "notes": [
            "Use CLASS-ID (column 2) as the 37-class classification target.",
            "BREED ID (column 4) is species-local and is not a unique global breed target.",
            "The supplied README reverses the cat/dog BREED-ID ranges: actual records have 12 cat and 25 dog breeds.",
            "No independent validation split is provided by these two official split files.",
            "Do not infer why an image is outside the lists; the uploaded README does not explain the exclusions.",
        ],
    }
    return {"records": records, "by_key": full,
            "classes": [classes[cid] for cid in sorted(classes, key=int)], "summary": summary}


def validate_manifest(rows, annotations, keep_official_test=True, class_id_mapping=None):
    """Check each supplied row against official breed labels and split membership."""
    def normalized_name(name):
        """Compare breed names consistently across spaces, underscores, and case."""
        return "_".join(name.strip().lower().replace("_", " ").split())
    # Step 1: An alternative class numbering must be complete and one-to-one.
    mapping = None
    if class_id_mapping is not None:
        mapping = {str(key): str(value) for key, value in class_id_mapping.items()}
        official_ids = {row["class_id"] for row in annotations["classes"]}
        if set(mapping) != official_ids or len(set(mapping.values())) != len(mapping) or "" in mapping.values():
            raise ValueError("CLASS_ID_MAPPING must map every official class ID to a distinct team ID")
    # Step 2: Check each listed image against its official label and allowed split.
    for row in rows:
        key = Path(row["image_id"].replace("\\", "/")).stem.casefold()
        official = annotations["by_key"].get(key)
        if official is None:
            raise ValueError(f"Shared image is outside official labelled lists: {row['image_id']}")
        expected_id = mapping[official["class_id"]] if mapping is not None else official["class_id"]
        if row["class_id"] != expected_id:
            raise ValueError(f"Class ID disagrees with annotation for {row['image_id']}: "
                             f"shared={row['class_id']}, expected={expected_id}. "
                             "Set the agreed CLASS_ID_MAPPING explicitly if the team renumbered labels.")
        if normalized_name(row["class_name"]) != normalized_name(official["class_name"]):
            raise ValueError(f"Breed name disagrees with annotation for {row['image_id']}")
        if keep_official_test:
            expected = "test" if row["split"] == "test" else "trainval"
            if official["official_split"] != expected:
                raise ValueError(f"Shared split changes official test membership: {row['image_id']}")


def write_reports(annotations, folder):
    """Save official labels, class mappings, hashes, and any extra-trimap findings."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "annotation_summary.json").write_text(
        json.dumps(annotations["summary"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for filename, rows, fields in (
        ("official_labels.csv", annotations["records"], [
            "image_id", "class_id", "class_name", "species_id", "species", "species_breed_id", "official_split"]),
        ("official_class_mapping.csv", annotations["classes"], [
            "class_id", "class_name", "species_id", "species", "species_breed_id"]),
        ("trimaps_outside_classification_lists.csv", [
            {"image_id": image, "status": "not_in_official_classification_lists"}
            for image in annotations["summary"]["trimaps_outside_classification_lists"] or []
        ], ["image_id", "status"]),
    ):
        with (folder / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def main():
    """Read annotations from the selected directory/ZIP and export their audit reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS_PATH)
    parser.add_argument("--report-folder", type=Path, default=REPORT_FOLDER)
    args = parser.parse_args()
    annotations = load_annotations(args.annotations)
    write_reports(annotations, args.report_folder)
    print(json.dumps(annotations["summary"], ensure_ascii=False, indent=2))
    print(f"Annotation reports: {args.report_folder}")


if __name__ == "__main__":
    main()
