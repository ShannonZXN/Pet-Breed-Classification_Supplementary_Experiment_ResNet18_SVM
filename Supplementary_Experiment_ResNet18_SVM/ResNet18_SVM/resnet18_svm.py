"""Frozen ImageNet ResNet-18 features + SVM for the team's pet-breed task.

Edit SETTINGS, then run `python resnet18_svm.py --help`.
Stages are separate so selecting SVM settings never evaluates the test set.
The team supplies the split and class IDs; this program never makes a split.
"""

from pathlib import Path

# ================== EXPERIMENT SETTINGS ==================
# Keep these defaults when reproducing the supplied experiment.
# For another computer, edit paths and select an installed extraction backend.
BASE_DIR = Path(__file__).resolve().parent
IMAGE_FOLDERS = {
    "original": BASE_DIR / "original",
    "foreground": BASE_DIR / "background_removed_all",
}
MANIFEST_CSV = BASE_DIR / "splits" / "manifest.csv"
ANNOTATIONS_PATH = BASE_DIR / "annotations"  # Directory or ZIP containing the official text annotations.
KEEP_OFFICIAL_TEST = True        # Preserve official test membership unless the agreed protocol changes.
CLASS_ID_MAPPING = None           # None keeps IDs 1-37; otherwise provide a complete official-to-team ID map.
OUTPUT_ROOT = BASE_DIR / "resnet18_svm_outputs_available"  # Separate output root retained from the supplied experiment.
EXPECTED_CLASS_COUNT = 37
# Missing files are skipped separately in each variant.
# Use the availability reports to identify which train/val/test samples were used.
SKIP_MISSING_IMAGES = True      # Skip absent files, retaining each available row's original split and label.
BATCH_SIZE = 1                  # One image at a time preserves variable native widths and heights.
NUM_WORKERS = 0                 # Start with 0 on Windows; increase only after checking worker behavior.
# DirectML is the original Windows/AMD choice, not an automatic GPU detector.
# For a portable CPU/CUDA setup, deliberately change this to "auto" or "cpu".
DEVICE = "directml"                # auto / cpu / cuda / directml; explicit DirectML setting is retained.
SEED = 42
SMOKE_SAMPLE_COUNT = 16         # Sample these same training IDs for the two-variant smoke check.
C_LOG10_MIN = -3                # Lower search bound: C = 10^(-3) = 0.001.
C_LOG10_MAX = 3                 # Upper search bound: C = 10^3 = 1000.
C_STEPS_PER_DECADE = 4          # Four intervals per decade; adjacent C values differ by about 1.778.
C_VALUES = [
    10.0 ** (step / C_STEPS_PER_DECADE)
    for step in range(C_LOG10_MIN * C_STEPS_PER_DECADE,
                      C_LOG10_MAX * C_STEPS_PER_DECADE + 1)
]                              # 25 values by default; an explicit positive-value list can also be used.
KERNELS = ["linear", "rbf"]
RBF_GAMMAS = ["scale"]
SELECTION_METRIC = "accuracy"  # accuracy / macro_f1; agree on the selection metric before experiments.
MAX_ERROR_EXAMPLES = 12
# ==========================================================

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone

import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from oxford_annotations import load_annotations, validate_manifest, write_reports

FEATURE_DIM = 512
# This record is saved with feature caches and model settings.
# Its values also participate in source-image/cache fingerprints.
EXTRACTOR_SPEC = {
    "implementation": "torchvision_resnet18_fc_identity_native_v2",
    "weights": "ResNet18_Weights.IMAGENET1K_V1",
    "feature_layer": "avgpool -> flatten (fc replaced by Identity)",
    "feature_dim": FEATURE_DIM,
    "rgb": True,
    "preserve_native_resolution": True,
    "resize": None,
    "crop": None,
    "padding": None,
    "pixel_value_scaling": "uint8 RGB / 255 -> float32",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "augmentation": False,
    "frozen": True,
}


def write_json(path, value):
    """Write an indented UTF-8 JSON report, creating its parent folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path, rows, fields=None):
    """Write a table with an explicit header and an Excel-friendly UTF-8 BOM."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        if not rows:
            raise ValueError("Supply CSV fields for an empty table")
        fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path):
    """Hash file bytes in chunks without decoding images or loading the whole file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_key(value):
    # A shared image_id may be a basename, a JPG/PNG name, or a relative path.
    """Normalize a filename or path to a case-insensitive image basename."""
    return Path(str(value).strip().replace("\\", "/")).stem.casefold()


def index_images(folder):
    """Index only top-level JPG/JPEG/PNG files; reject ambiguous basenames."""
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder does not exist: {folder}")
    result = {}
    for path in sorted(folder.iterdir(), key=lambda p: (p.name.casefold(), p.name)):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            key = image_key(path.name)
            if key in result:
                raise ValueError(f"Ambiguous image ID: {result[key]} and {path}")
            result[key] = path
    if not result:
        raise ValueError(f"No JPG/JPEG/PNG images in: {folder}")
    return result


def load_manifest(path=None, expected_classes=None):
    """Load the agreed manifest and validate IDs, labels, and split membership.

    Returns the manifest rows (with normalized keys) and class-ID/name mapping.
    Validation preserves the supplied class IDs and never creates a split.
    """
    path = Path(path or MANIFEST_CSV)
    if not path.is_file():
        raise FileNotFoundError(f"Shared manifest is required for training: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = ["image_id", "class_id", "class_name", "split"]
        if not set(fields).issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest needs columns: {', '.join(fields)}")
        rows = []
        for line, raw in enumerate(reader, start=2):
            if not any(raw.values()):
                continue
            row = {field: (raw.get(field) or "").strip() for field in fields}
            if not all(row.values()):
                raise ValueError(f"Empty required value in manifest line {line}")
            row["split"] = {"validation": "val", "valid": "val"}.get(
                row["split"].lower(), row["split"].lower()
            )
            if row["split"] not in {"train", "val", "test"}:
                raise ValueError(f"Unknown split in line {line}: {row['split']}")
            row["key"] = image_key(row["image_id"])
            rows.append(row)
    if not rows:
        raise ValueError("Manifest is empty: insert the team's agreed rows first")
    # Reject repeated images even if extensions, path forms, or letter case differ.
    # Require a one-to-one relationship between class IDs and breed names.
    seen, class_names, name_ids = set(), {}, {}
    for row in rows:
        if row["key"] in seen:
            raise ValueError(f"Repeated image across/within splits: {row['image_id']}")
        seen.add(row["key"])
        cid, name = row["class_id"], row["class_name"]
        if cid in class_names and class_names[cid] != name:
            raise ValueError(f"Class ID {cid} has conflicting names")
        if name in name_ids and name_ids[name] != cid:
            raise ValueError(f"Class {name} has conflicting IDs")
        class_names[cid], name_ids[name] = name, cid
    expected_classes = EXPECTED_CLASS_COUNT if expected_classes is None else expected_classes
    if len(class_names) != expected_classes:
        raise ValueError(f"Expected {expected_classes} breeds, found {len(class_names)}")
    for split in ("train", "val", "test"):
        if not any(row["split"] == split for row in rows):
            raise ValueError(f"Shared manifest has no {split} rows")
    train_classes = {row["class_id"] for row in rows if row["split"] == "train"}
    if train_classes != set(class_names):
        raise ValueError("Training split must contain every class used in validation/test")
    # Check actual official labels; preserve IDs instead of inferring a new encoding.
    validate_manifest(rows, load_annotations(ANNOTATIONS_PATH), KEEP_OFFICIAL_TEST, CLASS_ID_MAPPING)
    return rows, class_names


def environment_info():
    """Record installed package versions without importing PyTorch."""
    versions = {}
    for name in ("torch", "torchvision", "torch-directml", "numpy", "Pillow", "scikit-learn", "joblib", "matplotlib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": versions}


def image_availability(rows, available_keys):
    """Partition listed rows into available and absent files/features.

    The returned summary records per-split counts under the current missing-file
    policy. It does not change any image label or train/validation/test assignment.
    """
    available_keys = set(available_keys)
    available = [row for row in rows if row["key"] in available_keys]
    missing = [row for row in rows if row["key"] not in available_keys]
    summary = {
        "missing_image_policy": "skip" if SKIP_MISSING_IMAGES else "error",
        "listed_count": len(rows), "available_count": len(available),
        "missing_count": len(missing),
        "missing_image_ids": [row["image_id"] for row in missing],
        "split_counts": {
            split: {"listed": sum(row["split"] == split for row in rows),
                    "available": sum(row["split"] == split for row in available),
                    "missing": sum(row["split"] == split for row in missing)}
            for split in ("train", "val", "test")
        },
    }
    return available, missing, summary


def write_availability_report(directory, dataset, rows, available_keys):
    """Save the available manifest, missing-image list, and per-split counts."""
    available, missing, summary = image_availability(rows, available_keys)
    summary["dataset"] = dataset
    summary["image_folder"] = str(Path(IMAGE_FOLDERS[dataset]).resolve())
    fields = ["image_id", "class_id", "class_name", "split"]
    write_csv(directory / "available_manifest.csv", [
        {field: row[field] for field in fields} for row in available
    ], fields)
    write_csv(directory / "missing_images.csv", [
        {**{field: row[field] for field in fields}, "reason": "image_file_missing"}
        for row in missing
    ], fields + ["reason"])
    write_json(directory / "availability_summary.json", summary)
    return summary


def torch_modules():
    """Import torch/torchvision only when a stage actually needs the extractor."""
    try:
        import torch
        import torchvision
    except (ImportError, RuntimeError, OSError) as exc:
        raise RuntimeError(
            "Cannot import a compatible torch/torchvision pair. Follow README.md, "
            "then run the check stage. Original error: " + str(exc)
        ) from exc
    return torch, torchvision


def device_for(torch):
    """Resolve the requested extraction backend and fail if it cannot initialize.

    The auto option selects CUDA when available, otherwise CPU. DirectML must
    be requested explicitly; it is the supplied configuration's default.
    """
    if DEVICE.lower() in {"directml", "dml"}:
        try:
            import torch_directml
            return torch_directml.device()
        except (ImportError, OSError, RuntimeError) as exc:
            raise RuntimeError(
                "DirectML was requested but could not initialize. Use a separate Python 3.12 "
                "environment with requirements-directml.txt; see docs/DIRECTML.md. "
                "Original error: " + str(exc)
            ) from exc
    if DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("DEVICE requests CUDA, but torch.cuda.is_available() is False")
    return device


def build_extractor():
    """Build the frozen ImageNet ResNet-18 and native-resolution transform.

    Returns (model, transform, device). Replacing fc with Identity exposes the
    512 values after global average pooling and flattening. eval() fixes
    BatchNorm behavior; requires_grad_(False) disables parameter gradients.
    """
    torch, torchvision = torch_modules()
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = False
    device = device_for(torch)
    # Load published ImageNet weights (downloaded once if not already cached).
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights=weights)
    if model.fc.in_features != FEATURE_DIM:
        raise RuntimeError(f"Unexpected ResNet feature width: {model.fc.in_features}")
    # Remove only the classifier head; retain adaptive pooling and the flatten step.
    model.fc = torch.nn.Identity()
    # Freeze parameters and BatchNorm state; no CNN fine-tuning occurs.
    model.requires_grad_(False)
    model.eval().to(device)
    # Keep the original H/W: do not call weights.transforms(), which resizes
    # and center-crops. Fixed channel normalization does not change dimensions.
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=EXTRACTOR_SPEC["mean"], std=EXTRACTOR_SPEC["std"]),
    ])
    return model, transform, device


class ImageFeatureDataset:
    """Map-style dataset; global definition also works with Windows workers."""
    def __init__(self, items, transform):
        """Store the ordered (image key, path) pairs and the in-memory transform."""
        self.items = items
        self.transform = transform

    def __len__(self):
        """Return the number of indexed image pairs supplied to this dataset."""
        return len(self.items)

    def __getitem__(self, index):
        """Decode one RGB image and confirm that the transform preserves its pixel grid."""
        key, path = self.items[index]
        try:
            with Image.open(path) as image:
                expected_shape = (3, image.height, image.width)
                # Decode stored orientation as RGB; no resize, crop, padding, or EXIF transpose.
                tensor = self.transform(image.convert("RGB"))
                if tuple(tensor.shape) != expected_shape:
                    raise ValueError(f"Image dimensions changed: expected {expected_shape}, got {tuple(tensor.shape)}")
        except (OSError, ValueError) as exc:
            # Stop instead of silently changing the team's shared sample set.
            raise RuntimeError(f"Cannot preprocess {path}: {exc}") from exc
        return tensor, key


def snapshot_images(images):
    """Sort image IDs, hash source bytes, and fingerprint the extractor/input pair.

    The signature covers EXTRACTOR_SPEC and the selected source image hashes.
    Software versions and the device are recorded separately, not in this hash.
    """
    items = sorted(images.items())
    ids = [key for key, _ in items]
    hashes = [sha256_file(path) for _, path in items]
    signature = hashlib.sha256(json.dumps(
        {"extractor": EXTRACTOR_SPEC, "images": list(zip(ids, hashes))},
        sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    return items, ids, hashes, signature


def read_feature_cache(path, ids, signature):
    """Reuse a cache only when its signature, row IDs, shape, and values are valid."""
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as cache:
            meta = json.loads(str(cache["metadata_json"].item()))
            features = cache["features"]
            cache_ids = cache["image_ids"]
        if (meta["signature"] != signature or features.shape != (len(ids), FEATURE_DIM)
                or not np.array_equal(cache_ids, np.asarray(ids))
                or not np.isfinite(features).all()):
            return None
        return features, meta
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def feature_cache_path(dataset):
    """Return the independent full-cache path for one input variant."""
    return OUTPUT_ROOT / dataset / "feature_cache" / "features.npz"


def get_features(dataset, refresh=False, selected_keys=None, cache_path=None):
    """Return aligned frozen features for the available requested manifest images.

    Full extraction uses every available listed split, including test images,
    without using labels to learn extractor parameters. selected_keys is for
    smoke checks and must use a separate cache. refresh bypasses cache reuse.
    Returns (image index, ordered IDs, N-by-512 float32 features, metadata).
    """
    # Step 1: Validate the agreed manifest before inspecting the chosen input variant.
    rows, _ = load_manifest()
    indexed = index_images(IMAGE_FOLDERS[dataset])
    required = {row["key"] for row in rows}
    if selected_keys is not None:
        requested = set(selected_keys)
        if not requested or not requested.issubset(required):
            raise ValueError("Sample feature IDs must be a nonempty subset of the manifest")
        if cache_path is None or Path(cache_path).resolve() == feature_cache_path(dataset).resolve():
            raise ValueError("Sample extraction needs a separate cache path; keep the full cache intact")
        required = requested
    # Step 2: Apply the configured missing-file policy and report the actual subset.
    missing = sorted(required - set(indexed))
    if missing and not SKIP_MISSING_IMAGES:
        raise ValueError(f"{dataset}: {len(missing)} listed images missing; first: {missing[:8]}")
    # Exclude all unlisted files before hashing, decoding, or inference.
    # The count depends on the actual image folders; it is not assumed to be 41.
    images = {key: path for key, path in indexed.items() if key in required}
    report_dir = (OUTPUT_ROOT / dataset / "data_audit" if selected_keys is None
                  else Path(cache_path).parent)
    availability = write_availability_report(
        report_dir, dataset, [row for row in rows if row["key"] in required], images)
    counts = availability["split_counts"]
    print(f"{dataset}: available {len(images)}/{len(required)}, skipped missing {len(missing)}; "
          + ", ".join(f"{split} {counts[split]['available']}/{counts[split]['listed']}"
                      for split in ("train", "val", "test")))
    if missing:
        print(f"Missing-image list: {report_dir / 'missing_images.csv'}")
    if not images:
        raise ValueError(f"{dataset}: none of the requested listed images are available")
    print(f"Selected {len(images)} listed {dataset} images from {len(indexed)} image files")
    # Step 3: Hash only selected image files; unrelated images do not invalidate the cache.
    items, ids, hashes, signature = snapshot_images(images)
    current_metadata = {"data_availability": availability,
                        "source_image_sha256": dict(zip(ids, hashes))}
    cache_path = feature_cache_path(dataset) if cache_path is None else Path(cache_path)
    # Step 4: Reuse matching finite N-by-512 features, unless refresh is requested.
    cached = None if refresh else read_feature_cache(cache_path, ids, signature)
    if cached is not None:
        features, metadata = cached
        metadata = {**metadata, **current_metadata}
        print(f"Reusing {dataset} feature cache: {features.shape} ({cache_path})")
        return images, ids, features, metadata
    if BATCH_SIZE != 1:
        raise ValueError("Keep BATCH_SIZE=1 for native-resolution images; no resizing or padding is used")
    print(f"Extracting {dataset}: {len(items)} native-resolution images -> {FEATURE_DIM}-D features")
    started = time.perf_counter()
    model, transform, device = build_extractor()
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("The feature extractor must be frozen and in evaluation mode")
    load_seconds = time.perf_counter() - started
    torch, _ = torch_modules()
    # Step 5: Preserve sorted ID order. Variable native dimensions require batch size 1.
    loader = torch.utils.data.DataLoader(
        ImageFeatureDataset(items, transform), batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=device.type == "cuda",
    )
    chunks, observed_ids, input_sizes_wh = [], [], []
    started = time.perf_counter()
    # Step 6: Run forward passes without autograd and move each feature row to CPU.
    with torch.no_grad():
        for batch_number, (pixels, batch_ids) in enumerate(loader, start=1):
            feature_batch = model(pixels.to(device, non_blocking=device.type == "cuda"))
            if (feature_batch.ndim != 2 or feature_batch.shape[1] != FEATURE_DIM
                    or feature_batch.shape[0] != len(batch_ids)):
                raise RuntimeError(f"Unexpected feature shape: {tuple(feature_batch.shape)}")
            chunks.append(feature_batch.cpu().numpy().astype(np.float32, copy=False))
            observed_ids.extend(batch_ids)
            input_sizes_wh.extend([[int(pixels.shape[-1]), int(pixels.shape[-2])]
                                   for _ in batch_ids])
            if batch_number % 10 == 0 or batch_number == len(loader):
                print(f"  {min(batch_number * BATCH_SIZE, len(items))}/{len(items)}")
    # Step 7: Confirm every output row still belongs to the expected source image.
    if observed_ids != ids:
        raise RuntimeError("Feature rows lost their image-ID alignment")
    features = np.concatenate(chunks, axis=0)
    if not np.isfinite(features).all():
        raise ValueError("Non-finite feature values")
    metadata = {
        **current_metadata,
        "signature": signature, "extractor": EXTRACTOR_SPEC,
        "image_count": len(ids), "dataset": dataset, "device": str(device),
        "model_eval": True, "parameters_frozen": True,
        "autograd_mode": "no_grad",
        "native_resolution_checked": True,
        "weights_load_seconds": load_seconds,
        "inference_seconds": time.perf_counter() - started,
        "environment": environment_info(),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Step 8: Write a complete temporary NPZ before replacing the canonical cache.
    temporary = cache_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, features=features, image_ids=np.asarray(ids),
                            image_sha256=np.asarray(hashes),
                            input_sizes_wh=np.asarray(input_sizes_wh, dtype=np.int64),
                            metadata_json=json.dumps(metadata))
    temporary.replace(cache_path)
    print(f"Saved feature cache: {cache_path}")
    return images, ids, features, metadata


def select_rows(rows, split, ids, features):
    """Align one split to cached feature rows by image key, never by row position."""
    positions = {key: i for i, key in enumerate(ids)}
    selected = [row for row in rows if row["split"] == split]
    absent = [row["image_id"] for row in selected if row["key"] not in positions]
    if absent and not SKIP_MISSING_IMAGES:
        raise ValueError(f"{split}: {len(absent)} shared images missing; first: {absent[:8]}")
    selected = [row for row in selected if row["key"] in positions]
    if not selected:
        raise ValueError(f"{split}: no available listed images; this stage needs a nonempty {split} split")
    # A manifest may be ordered differently from the cache; join by ID explicitly.
    indices = [positions[row["key"]] for row in selected]
    return selected, features[indices], np.asarray([row["class_id"] for row in selected])


def metrics(y_true, y_pred, labels):
    """Compute accuracy and macro metrics over the explicit full label universe."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(precision), "macro_recall": float(recall), "macro_f1": float(f1)}


def fit_candidates(x_train, y_train, x_val, y_val, class_ids):
    """Fit StandardScaler + SVC candidates on train and select by validation only.

    Each candidate gets a fresh pipeline. Exact score ties retain the first
    candidate in declared grid order. Returns the fitted winner, its result,
    and all validation records; the winner is not refitted on train + val.
    """
    if SELECTION_METRIC not in {"accuracy", "macro_f1"}:
        raise ValueError("SELECTION_METRIC must be accuracy or macro_f1")
    if not C_VALUES or any(not np.isfinite(value) or value <= 0 for value in C_VALUES):
        raise ValueError("C_VALUES must contain finite positive numbers")
    if not KERNELS or any(kernel not in {"linear", "rbf"} for kernel in KERNELS):
        raise ValueError("KERNELS must contain linear and/or rbf")
    candidates, best, best_score = [], None, -np.inf
    # Step 1: Visit kernels, C values, and gammas in the declared deterministic order.
    for kernel in KERNELS:
        gammas = ["scale"] if kernel == "linear" else RBF_GAMMAS
        for c in C_VALUES:
            for gamma in gammas:
                # The scaler's mean/std are learned from x_train only.
                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("svm", SVC(C=c, kernel=kernel, gamma=gamma,
                                decision_function_shape="ovr", break_ties=True,
                                cache_size=512, random_state=SEED)),
                ])
                # Step 2: Fit the scaler and SVM together on training features only.
                started = time.perf_counter()
                pipeline.fit(x_train, y_train)
                seconds = time.perf_counter() - started
                # Step 3: Score this candidate on validation; test arrays are never passed here.
                result = {"kernel": kernel, "C": float(c),
                          "gamma": gamma if kernel == "rbf" else "unused",
                          **metrics(y_val, pipeline.predict(x_val), class_ids),
                          "train_seconds": seconds}
                candidates.append(result)
                print(f"  {kernel}, C={c:.6g}, gamma={gamma}: val {SELECTION_METRIC}={result[SELECTION_METRIC]:.4f}")
                # Exact ties keep the first candidate in the declared grid.
                if result[SELECTION_METRIC] > best_score:
                    best_score = result[SELECTION_METRIC]
                    best = (pipeline, dict(result))
    if best is None:
        raise ValueError("No SVM candidates; check RBF_GAMMAS")
    return best[0], best[1], candidates


def export_validation_curve(path, candidates):
    """Plot all validation scores against C on a logarithmic axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = {}
    for result in candidates:
        groups.setdefault((result["kernel"], str(result["gamma"])), []).append(result)
    fig, ax = plt.subplots(figsize=(8, 5))
    for (kernel, gamma), group in groups.items():
        group = sorted(group, key=lambda result: result["C"])
        label = kernel if gamma == "unused" else f"{kernel}, gamma={gamma}"
        ax.plot([row["C"] for row in group], [row[SELECTION_METRIC] for row in group],
                marker="o", markersize=4, label=label)
    selected = max(candidates, key=lambda result: result[SELECTION_METRIC])
    ax.scatter([selected["C"]], [selected[SELECTION_METRIC]], s=100,
               facecolors="none", edgecolors="black", linewidths=1.5,
               label="Selected configuration", zorder=3)
    ax.set(xscale="log", xlabel="C (log scale)",
           ylabel=f"Validation {SELECTION_METRIC.replace('_', ' ')}",
           title="SVM parameter search")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def export_predictions(path, rows, x, y, pipeline, class_names):
    """Write per-image predictions and the score for each predicted class.

    Decision scores are uncalibrated SVM outputs, not probabilities. Return
    predictions, all score columns, their class order, and the CSV records.
    """
    predictions = pipeline.predict(x)
    scores = np.asarray(pipeline.decision_function(x))
    classes = list(pipeline.named_steps["svm"].classes_)
    if scores.ndim == 1:  # Binary fixtures/other use; the project has 37 breeds.
        scores = np.column_stack([-scores, scores])
    class_columns = {cid: i for i, cid in enumerate(classes)}
    result = []
    for i, (row, truth, prediction) in enumerate(zip(rows, y, predictions)):
        result.append({
            "image_id": row["image_id"], "true_label": str(truth),
            "true_breed": class_names[str(truth)], "predicted_label": str(prediction),
            "predicted_breed": class_names[str(prediction)],
            "decision_score": float(scores[i, class_columns[prediction]]),
            "correct": bool(truth == prediction),
        })
    write_csv(path, result)
    return predictions, scores, classes, result


def run_audit(datasets=None):
    """Report image pairing, official annotations, and manifest availability.

    The direct Python call checks both folders unless datasets is provided.
    The CLI defaults to foreground; use --dataset both for a paired audit.
    manifest_ready requires valid metadata and nonempty available splits,
    but does not certify complete class coverage after missing-file skips.
    """
    datasets = list(IMAGE_FOLDERS) if datasets is None else datasets
    indexes = {name: index_images(IMAGE_FOLDERS[name]) for name in datasets}
    all_ids = sorted(set().union(*(set(index) for index in indexes.values())))
    table = [{"image_key": key, **{name: str(index.get(key, ""))
              for name, index in indexes.items()}} for key in all_ids]
    report_dir = OUTPUT_ROOT / "data_audit"
    write_csv(report_dir / "paired_images.csv", table)
    annotations = load_annotations(ANNOTATIONS_PATH)
    write_reports(annotations, report_dir)
    labelled_ids = set(annotations["by_key"])
    outside = [{"image_key": key, **{name: str(index.get(key, ""))
                for name, index in indexes.items()}}
               for key in all_ids if key not in labelled_ids]
    write_csv(report_dir / "images_outside_official_lists.csv", outside,
              ["image_key", *indexes])
    missing = {name: [key for key in all_ids if key not in index]
               for name, index in indexes.items()}
    summary = {"image_counts": {name: len(index) for name, index in indexes.items()},
               "missing_in_variant": missing, "manifest_ready": False,
               "official_labelled_count": len(labelled_ids),
               "official_trainval": annotations["summary"]["official_trainval"],
               "official_test": annotations["summary"]["official_test"],
               "images_outside_official_lists": len(outside),
               "official_images_missing_in_variant": {
                   name: sorted(labelled_ids - set(index)) for name, index in indexes.items()
               }}
    if MANIFEST_CSV.is_file() and MANIFEST_CSV.stat().st_size:
        try:
            rows, names = load_manifest()
        except ValueError as exc:
            summary["manifest_message"] = str(exc)
        else:
            required = {row["key"] for row in rows}
            absent = {name: sorted(required - set(index)) for name, index in indexes.items()}
            availability = {name: write_availability_report(
                OUTPUT_ROOT / name / "data_audit", name, rows, index)
                for name, index in indexes.items()}
            usable = all(all(details["split_counts"][split]["available"] > 0
                             for split in ("train", "val", "test"))
                         for details in availability.values())
            summary.update({"manifest_ready": usable and (SKIP_MISSING_IMAGES or not any(absent.values())),
                            "all_listed_images_present": not any(absent.values()),
                            "missing_image_policy": "skip" if SKIP_MISSING_IMAGES else "error",
                            "availability_by_variant": availability,
                            "split_counts": dict(Counter(row["split"] for row in rows)),
                            "class_count": len(names), "manifest_missing_images": absent,
                            "unlisted_images": {name: sorted(set(index) - required)
                                                for name, index in indexes.items()}})
            write_csv(report_dir / "class_mapping.csv", [
                {"class_id": cid, "class_name": name} for cid, name in names.items()
            ])
    else:
        summary["manifest_message"] = f"Awaiting the team's shared manifest: {MANIFEST_CSV}"
    write_json(report_dir / "audit_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Audit files: {report_dir}")
    return summary


def train(dataset, refresh=False):
    """Select and save one train-fitted SVM using the validation split.

    Test features may already be cached, but no test labels or predictions
    enter fit_candidates. The output records the exact available sample counts.
    """
    # Step 1: Load aligned features, then separate train and validation by saved membership.
    rows, names = load_manifest()
    images, ids, features, cache_meta = get_features(dataset, refresh)
    train_rows, x_train, y_train = select_rows(rows, "train", ids, features)
    val_rows, x_val, y_val = select_rows(rows, "val", ids, features)
    if len(set(y_train)) < 2:
        raise ValueError("SVM training needs available images from at least two classes")
    class_ids = sorted(names)
    # Step 2: Select the validation winner, retaining its train-only fitted parameters.
    pipeline, selected, candidates = fit_candidates(x_train, y_train, x_val, y_val, class_ids)
    # Step 3: Give each successful training result its own UTC timestamped directory.
    run_name = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    run_dir = OUTPUT_ROOT / dataset / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    availability = write_availability_report(run_dir, dataset, rows, ids)
    write_csv(run_dir / "validation_results.csv", candidates)
    export_validation_curve(run_dir / "figures" / "validation_score_vs_C.png", candidates)
    export_predictions(run_dir / "validation_predictions.csv", val_rows, x_val, y_val, pipeline, names)
    # Step 4: Record sample counts, full SVM parameters, search space, and fingerprints.
    settings = {
        "dataset": dataset, "selected": selected, "selection_metric": SELECTION_METRIC,
        "svm_parameters": pipeline.named_steps["svm"].get_params(deep=False),
        "search_space": {"C_VALUES": [float(value) for value in C_VALUES],
                         "KERNELS": list(KERNELS), "RBF_GAMMAS": list(RBF_GAMMAS),
                         "candidate_count": len(candidates)},
        "selected_C_at_grid_boundary": {"lower": selected["C"] == min(C_VALUES),
                                        "upper": selected["C"] == max(C_VALUES)},
        "tie_rule": "first candidate in grid order", "class_names": names,
        "train_count": len(train_rows), "validation_count": len(val_rows),
        "data_availability": availability,
        "trained_class_ids": list(pipeline.named_steps["svm"].classes_),
        "manifest_sha256": sha256_file(MANIFEST_CSV), "extractor": EXTRACTOR_SPEC,
        "annotation_text_sha256": load_annotations(ANNOTATIONS_PATH)["summary"]["text_sha256"],
        "keep_official_test": KEEP_OFFICIAL_TEST, "class_id_mapping": CLASS_ID_MAPPING,
        "feature_cache_signature": cache_meta["signature"], "feature_extraction": cache_meta,
        "feature_cache_path": str(feature_cache_path(dataset).resolve()),
        "final_fit_data": "train only; validation is not refitted",
        "environment": environment_info(), "script_sha256": sha256_file(Path(__file__)),
        "decision_scores": "uncalibrated SVM decision scores, not probabilities",
    }
    write_json(run_dir / "final_model_settings.json", settings)
    (run_dir / "final_model_settings.txt").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Step 5: Save the fitted scaler/SVM together; update the latest-model pointer last.
    bundle = {"pipeline": pipeline, "settings": settings}
    model_path = run_dir / "best_model.joblib"
    joblib.dump(bundle, model_path)
    write_json(OUTPUT_ROOT / dataset / "latest_model.json", {"model_path": str(model_path.resolve())})
    print(f"Selected on validation: {selected}")
    print(f"Model and validation records: {run_dir}")
    return model_path


def load_model(dataset, model_path=None):
    """Load a saved variant-matched bundle and verify its extractor specification."""
    if model_path is None:
        pointer = OUTPUT_ROOT / dataset / "latest_model.json"
        if not pointer.is_file():
            raise FileNotFoundError(f"Run train for {dataset} first, or supply --model")
        model_path = Path(json.loads(pointer.read_text(encoding="utf-8"))["model_path"])
    model_path = Path(model_path).expanduser().resolve()
    # Load bundles produced by this project from a trusted source.
    bundle = joblib.load(model_path)
    if bundle["settings"]["dataset"] != dataset:
        raise ValueError("The selected model belongs to another image variant")
    if bundle["settings"]["extractor"] != EXTRACTOR_SPEC:
        raise ValueError("Model and current feature-extraction settings differ")
    return model_path, bundle


def export_test_analysis(run_dir, selected_rows, y, predictions, names, images, pred_records):
    """Export the confusion matrix, per-class metrics, directed errors, and examples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Use an explicit shared label order for matrices and macro/per-class metrics.
    # IDs are strings here: the order is lexical, and the CSV/figure labels make it explicit.
    labels = sorted(names)
    matrix = confusion_matrix(y, predictions, labels=labels)
    write_csv(run_dir / "confusion_matrix.csv", [
        {"true_class_id": cid, **{pid: int(matrix[i, j]) for j, pid in enumerate(labels)}}
        for i, cid in enumerate(labels)
    ])
    precision, recall, f1, support = precision_recall_fscore_support(
        y, predictions, labels=labels, zero_division=0
    )
    write_csv(run_dir / "per_class_metrics.csv", [
        {"class_id": cid, "breed": names[cid], "precision": float(precision[i]),
         "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
        for i, cid in enumerate(labels)
    ])
    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(15, 13))
    display = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(display, ax=ax, fraction=0.04)
    tick_names = [names[cid].replace("_", " ") for cid in labels]
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)),
           xticklabels=tick_names, yticklabels=tick_names,
           xlabel="Predicted breed", ylabel="True breed", title="ResNet-18 features + SVM: test")
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "confusion_matrix.png", dpi=180)
    plt.close(fig)
    # Count directed errors (true breed -> predicted breed); reverse pairs stay separate.
    confused = sorted([(int(matrix[i, j]), labels[i], labels[j])
                       for i in range(len(labels)) for j in range(len(labels))
                       if i != j and matrix[i, j] > 0], reverse=True)
    write_csv(run_dir / "most_confused_pairs.csv", [
        {"count": count, "true_breed": names[truth], "predicted_breed": names[pred]}
        for count, truth, pred in confused
    ], ["count", "true_breed", "predicted_breed"])
    errors = []
    error_dir = run_dir / "error_cases"
    error_dir.mkdir(exist_ok=True)
    # Copy the first configured number of errors in manifest order for inspection.
    for row, record in zip(selected_rows, pred_records):
        if not record["correct"] and len(errors) < MAX_ERROR_EXAMPLES:
            source = images[row["key"]]
            destination = error_dir / f"{len(errors)+1:02d}_{source.name}"
            shutil.copy2(source, destination)
            errors.append({**record, "example_file": destination.name})
    write_csv(error_dir / "index.csv", errors, [
        "image_id", "true_label", "true_breed", "predicted_label", "predicted_breed",
        "decision_score", "correct", "example_file",
    ])


def test(dataset, model_path=None, refresh=False):
    """Evaluate a saved model without refitting or selecting new hyperparameters.

    Require the same manifest bytes. In skip mode, added/absent image files may
    change availability, but edits to source images shared with training fail.
    Reports describe the test subset actually evaluated.
    """
    # Step 1: Load the chosen fitted bundle and require the same manifest file bytes.
    model_path, bundle = load_model(dataset, model_path)
    settings = bundle["settings"]
    if sha256_file(MANIFEST_CSV) != settings["manifest_sha256"]:
        raise ValueError("Shared manifest changed since training; reconcile it before testing")
    rows, names = load_manifest()
    images, ids, features, cache_meta = get_features(dataset, refresh)
    if cache_meta["signature"] != settings["feature_cache_signature"]:
        # Added/removed files change the cache signature. In skip mode this is
        # allowed, while an edited image already seen in this run still fails.
        previous_sources = settings.get("feature_extraction", {}).get("source_image_sha256")
        current_sources = cache_meta.get("source_image_sha256", {})
        if not SKIP_MISSING_IMAGES or not previous_sources:
            raise ValueError("Image content/extractor changed since training; train a consistent run first")
        changed = [key for key in set(previous_sources) & set(current_sources)
                   if previous_sources[key] != current_sources[key]]
        if changed:
            raise ValueError(f"Image content changed since training for {len(changed)} images; "
                             f"train a consistent run first. First: {sorted(changed)[:8]}")
        print("Image availability changed since training; evaluating the available test images.")
    # Step 2: Evaluate only the currently available test rows; never call fit here.
    selected, x_test, y_test = select_rows(rows, "test", ids, features)
    result_dir = model_path.parent / datetime.now(timezone.utc).strftime("test_%Y%m%dT%H%M%S_%fZ")
    result_dir.mkdir(parents=True, exist_ok=False)
    availability = write_availability_report(result_dir, dataset, rows, ids)
    predictions, scores, classes, records = export_predictions(
        result_dir / "test_predictions.csv", selected, x_test, y_test, bundle["pipeline"], names
    )
    # Step 3: Export aggregate metrics, all decision scores, and detailed error analysis.
    result = metrics(y_test, predictions, sorted(names))
    write_csv(result_dir / "test_metrics.csv", [{"dataset": dataset, "test_count": len(y_test), **result}])
    write_csv(result_dir / "test_decision_scores.csv", [
        {"image_id": row["image_id"], **{f"score_{cid}": float(scores[i, j])
         for j, cid in enumerate(classes)}} for i, row in enumerate(selected)
    ])
    export_test_analysis(result_dir, selected, y_test, predictions, names, images, records)
    # Step 4: Preserve both training-time and current availability in the test report.
    write_json(result_dir / "evaluation_metadata.json", {
        "model_path": str(model_path), "manifest_sha256": settings["manifest_sha256"],
        "dataset": dataset, "test_count": len(y_test), "metrics": result,
        "listed_test_count": availability["split_counts"]["test"]["listed"],
        "skipped_test_count": availability["split_counts"]["test"]["missing"],
        "data_availability": availability,
        "training_data_availability": settings.get("data_availability"),
        "feature_cache_signature": cache_meta["signature"],
    })
    print(f"Final test {dataset}: {result}")
    print(f"Predictions, metrics, figures and errors: {result_dir}")
    return result


def predict(dataset, image_path, model_path=None):
    """Extract one image with the same transform and apply the saved SVM pipeline."""
    _, bundle = load_model(dataset, model_path)
    if bundle["settings"]["extractor"] != EXTRACTOR_SPEC:
        raise ValueError("Model preprocessing differs from this extractor; use a consistent trained model")
    model, transform, device = build_extractor()
    torch, _ = torch_modules()
    with Image.open(image_path) as image:
        pixels = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(pixels).cpu().numpy()
    label = str(bundle["pipeline"].predict(features)[0])
    result = {"image": str(image_path), "input_variant": dataset,
              "predicted_label": label, "predicted_breed": bundle["settings"]["class_names"][label]}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def check_environment():
    """Probe the chosen device and save a versioned success or failure report."""
    info = environment_info()
    destination = OUTPUT_ROOT / "environment_report.json"
    try:
        torch, torchvision = torch_modules()
        device = device_for(torch)
        backend = "directml" if device.type == "privateuseone" else device.type
        # Check a real device operation and the CPU/NumPy transfer used by extraction.
        probe = (torch.tensor([1.0]).to(device) + 2.0).cpu().numpy()
        if not np.array_equal(probe, np.asarray([3.0])):
            raise RuntimeError("The selected device failed the tensor/NumPy check")
        info.update({"torch": torch.__version__, "torchvision": torchvision.__version__,
                     "cuda_available": torch.cuda.is_available(),
                     "selected_device": str(device), "selected_backend": backend,
                     "device_tensor_test_passed": True, "check_passed": True})
        if device.type == "cuda":
            info["gpu"] = torch.cuda.get_device_name(0)
    except (OSError, ValueError, RuntimeError) as exc:
        info.update({"check_passed": False, "error": str(exc)})
        write_json(destination, info)
        print(f"Environment check failed; report: {destination}")
        raise
    write_json(destination, info)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    print(f"Environment report: {destination}")
    return info


def run_smoke(datasets):
    """Check matching training IDs, native dimensions, and frozen feature extraction.

    In skip mode the sample pool is the available training-ID intersection of
    the requested variants. Each branch uses a separate temporary-size cache
    under its smoke run, leaving the full feature cache untouched.
    """
    rows, _ = load_manifest()
    indexes = {dataset: index_images(IMAGE_FOLDERS[dataset]) for dataset in datasets}
    # Step 1: Build a training-only pool; skip mode intersects the requested variants.
    pool = sorted((row for row in rows if row["split"] == "train"), key=lambda row: row["key"])
    if SKIP_MISSING_IMAGES:
        common_keys = set.intersection(*(set(index) for index in indexes.values()))
        pool = [row for row in pool if row["key"] in common_keys]
    if SMOKE_SAMPLE_COUNT < 1 or not pool or (not SKIP_MISSING_IMAGES and SMOKE_SAMPLE_COUNT > len(pool)):
        raise ValueError(f"SMOKE_SAMPLE_COUNT must be between 1 and {len(pool)}")
    sample_count = min(SMOKE_SAMPLE_COUNT, len(pool))
    # Step 2: Select the same seeded training IDs for each branch and sort by key.
    selected = sorted(random.Random(SEED).sample(pool, sample_count), key=lambda row: row["key"])
    expected_ids = [row["key"] for row in selected]
    environment = check_environment()
    run_name = datetime.now(timezone.utc).strftime("smoke_%Y%m%dT%H%M%S_%fZ")
    directories = {dataset: OUTPUT_ROOT / dataset / "smoke_tests" / run_name for dataset in datasets}
    reports = {dataset: {
        "passed": False, "dataset": dataset, "sample_count": len(selected), "seed": SEED,
        "source_split": "train", "manifest_sha256": sha256_file(MANIFEST_CSV),
        "image_ids": [row["image_id"] for row in selected], "extractor": EXTRACTOR_SPEC,
        "environment": environment,
        "requested_sample_count": SMOKE_SAMPLE_COUNT,
        "available_train_pool_count": len(pool),
        "data_availability": image_availability(rows, indexes[dataset])[2],
    } for dataset in datasets}
    for dataset, directory in directories.items():
        write_json(directory / "step2_report.json", reports[dataset])
    try:
        records_by_variant, sizes_by_variant = {}, {}
        for dataset in datasets:
            indexed = indexes[dataset]
            absent = [row["image_id"] for row in selected if row["key"] not in indexed]
            if absent:
                raise ValueError(f"{dataset}: sampled images missing: {absent[:8]}")
            records, sizes = [], {}
            for row in selected:
                path = indexed[row["key"]]
                with Image.open(path) as image:
                    width, height = image.size
                sizes[row["key"]] = (width, height)
                records.append({
                    **{field: row[field] for field in ("image_id", "class_id", "class_name", "split")},
                    "image_key": row["key"], "image_path": str(path),
                    "width": width, "height": height,
                })
            records_by_variant[dataset], sizes_by_variant[dataset] = records, sizes
        # Step 3: Require paired image dimensions to match before feature inference.
        reference_sizes = sizes_by_variant[datasets[0]]
        for dataset, sizes in sizes_by_variant.items():
            if sizes != reference_sizes:
                bad = [key for key in expected_ids if sizes[key] != reference_sizes[key]]
                raise ValueError(f"Paired image dimensions disagree for {dataset}: {bad[:8]}")
        for dataset, directory in directories.items():
            write_csv(directory / "sample_manifest.csv", records_by_variant[dataset])
            # Step 4: Force a fresh sample cache in this smoke run, separate from full features.
            _, ids, features, metadata = get_features(
                dataset, refresh=True, selected_keys=expected_ids,
                cache_path=directory / "sample_features.npz",
            )
            if ids != expected_ids or features.shape != (len(selected), FEATURE_DIM):
                raise RuntimeError("Smoke-test feature rows do not match the sampled image IDs")
            if not np.isfinite(features).all():
                raise RuntimeError("Smoke-test features contain non-finite values")
            # Step 5: Check actual network-input sizes, finite features, and frozen/eval metadata.
            with np.load(directory / "sample_features.npz", allow_pickle=False) as cached:
                actual_sizes = cached["input_sizes_wh"]
            expected_sizes = np.asarray([reference_sizes[key] for key in ids], dtype=np.int64)
            if not np.array_equal(actual_sizes, expected_sizes):
                raise RuntimeError("Network inputs do not preserve the original image dimensions")
            if not all(metadata.get(key) is True for key in
                       ("model_eval", "parameters_frozen", "native_resolution_checked")):
                raise RuntimeError("The extractor did not confirm frozen native-resolution inference")
            reports[dataset].update({
                "passed": True, "feature_shape": list(features.shape),
                "checks": {"train_samples_only": True, "image_id_alignment": True,
                           "finite_features": True, "native_dimensions_preserved": True,
                           "model_eval": True, "parameters_frozen": True,
                           "paired_dimensions_match": True if len(datasets) > 1 else None},
                "feature_extraction": metadata,
            })
            write_json(directory / "step2_report.json", reports[dataset])
            print(f"Smoke test PASSED for {dataset}: {directory}")
    except Exception as exc:
        for dataset, directory in directories.items():
            if not reports[dataset]["passed"]:
                reports[dataset]["error"] = f"{type(exc).__name__}: {exc}"
                write_json(directory / "step2_report.json", reports[dataset])
        raise
    return directories


def main(argv=None):
    """Parse the requested stage and dispatch it for one variant or both sequentially."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["check", "audit", "smoke", "extract", "train", "test", "predict"])
    parser.add_argument("--dataset", choices=["original", "foreground", "both"], default="foreground")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--model", type=Path, help="Explicit best_model.joblib; default is latest model")
    parser.add_argument("--image", type=Path, help="Image for the predict stage")
    # Parse only the supported stage options. Paths and experiment settings live above.
    args = parser.parse_args(argv)
    if BATCH_SIZE < 1 or NUM_WORKERS < 0:
        raise ValueError("BATCH_SIZE must be positive and NUM_WORKERS nonnegative")
    if args.dataset == "both" and (args.model or args.stage == "predict"):
        raise ValueError("Choose one dataset for --model or predict")
    if args.stage == "check":
        check_environment()
    elif args.stage == "audit":
        if not run_audit(["foreground", "original"] if args.dataset == "both" else [args.dataset])["manifest_ready"]:
            raise ValueError(f"Data audit failed; inspect {OUTPUT_ROOT / 'data_audit' / 'audit_summary.json'}")
    elif args.stage == "smoke":
        run_smoke(["foreground", "original"] if args.dataset == "both" else [args.dataset])
    else:
        # Both runs foreground first, then original, with separate caches and fitted models.
        for dataset in (["foreground", "original"] if args.dataset == "both" else [args.dataset]):
            if args.stage == "extract":
                get_features(dataset, args.refresh_cache)
            elif args.stage == "train":
                train(dataset, args.refresh_cache)
            elif args.stage == "test":
                test(dataset, args.model, args.refresh_cache)
            elif args.stage == "predict":
                if not args.image:
                    raise ValueError("predict requires --image")
                predict(dataset, args.image, args.model)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
