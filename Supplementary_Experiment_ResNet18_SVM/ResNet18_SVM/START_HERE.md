# Run the experiment step by step

Open a terminal in the directory containing `resnet18_svm.py`. Keep the whole
package together so its helper imports and relative paths work.

This guide follows the supplied code's native-resolution, frozen-feature
experiment. It uses existing original and foreground images; it does not create
segmentations, resized variants, or a new model architecture.

## Step 0 — Prepare the environment and paths

Choose one extraction backend:

| Backend | Setup |
| --- | --- |
| CPU | Use the commands below and set `DEVICE = "cpu"`. |
| NVIDIA CUDA | Install a compatible torch/torchvision pair from the [official installer](https://pytorch.org/get-started/locally/); set `DEVICE = "cuda"` or `"auto"`. |
| Windows DirectML | Use [docs/DIRECTML.md](docs/DIRECTML.md); retain `DEVICE = "directml"`. |

For Windows CPU, install Python 3.12 and run in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

These commands call the environment's interpreter directly; activation is
optional. Subsequent Windows commands use this same interpreter. For DirectML,
use `.\.venv-dml\Scripts\python.exe` instead.

For macOS/Linux CPU, create and activate an environment, then install:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

If the CPU wheel index does not provide your platform's wheels, use the
official PyTorch installer for that platform. This code's `auto` selects CUDA or
CPU; it does not select Apple's MPS backend.

Edit the top of `resnet18_svm.py`:

```python
IMAGE_FOLDERS = {
    "original": Path(r"D:\Pets\original"),
    "foreground": Path(r"D:\Pets\background_removed_all"),
}
DEVICE = "cpu"  # Use the backend installed above.
BATCH_SIZE = 1
NUM_WORKERS = 0
```

Use actual paths directly containing the images. `Abyssinian_1.jpg` and
`Abyssinian_1.png` map to the same ID. Subfolders are not searched. Two supported
files with the same normalized basename in one folder are an error.

Other supplied defaults are retained:

```python
OUTPUT_ROOT = BASE_DIR / "resnet18_svm_outputs_available"
SKIP_MISSING_IMAGES = True
KEEP_OFFICIAL_TEST = True
CLASS_ID_MAPPING = None
```

The missing-image policy allows partial variants. Choose the intended policy
before running, and review actual counts rather than assuming complete input.

## Step 1 — Supply the exact split membership

The source ZIP includes official annotation texts, but **no saved manifest or
validation-ID file**. Choose the route matching the files you actually have.

### A. You already have the experiment's manifest

Copy it to `splits/manifest.csv`, or edit `MANIFEST_CSV` to its actual path.
Keep the assignments used by the reported results.

Required schema:

```csv
image_id,class_id,class_name,split
```

Each image occurs once, with `split` equal to `train`, `val`, or `test`.
Official class IDs are 1–37 by default. `validation` and `valid` are accepted
as aliases of `val`. The validator checks official labels and membership;
see [the split README](splits/README.md).

### B. You have three existing split CSVs

Edit `SPLIT_FILES` and `COLUMNS` in `merge_split_csvs.py`, then run:

```powershell
.\.venv\Scripts\python.exe merge_split_csvs.py
```

This preserves assignments and merges the required columns. It does not sample
or choose validation IDs. The main pipeline subsequently checks official labels.

### C. You have the agreed validation IDs

Place one ID per line in `splits/val_ids.txt`, or edit `VALIDATION_IDS_TXT` in
`build_manifest_from_ids.py`, then run:

```powershell
.\.venv\Scripts\python.exe build_manifest_from_ids.py
```

Official trainval IDs in that list become validation; remaining trainval IDs
become training. Official test remains test. Labels come from the annotation
texts. If class IDs were remapped, keep the mapping consistent in both scripts.

### D. You deliberately want a new split

Only when starting a new experiment without an existing agreed allocation:

```powershell
.\.venv\Scripts\python.exe generate_shared_splits.py
```

This generates 2,944 train / 736 validation / 3,669 official test rows, stratified
by breed with seed 42. It writes CSVs, ID lists, class counts, and
`split_summary.json`. It **overwrites** files of the same names in `splits/`.
Keep and share those generated memberships for every method and image variant.

A matching seed/count is not proof of membership equality with a historical
experiment: input order, library versions, and any earlier selection procedure
matter. This archive does not claim to recover missing historical split files.

## Step 2 — Check the environment, data, and a small shared sample

Run each command after the previous one succeeds:

```powershell
.\.venv\Scripts\python.exe resnet18_svm.py check
.\.venv\Scripts\python.exe self_test.py
.\.venv\Scripts\python.exe resnet18_svm.py audit --dataset both
.\.venv\Scripts\python.exe resnet18_svm.py smoke --dataset both
```

| Command | What to inspect |
| --- | --- |
| `check` | `environment_report.json`: `check_passed: true`, correct selected backend, passing tensor probe. |
| `self_test.py` | Functional-test summary; its synthetic scores are not pet-classification results. |
| `audit --dataset both` | `data_audit/audit_summary.json` and each variant's available/missing manifests. |
| `smoke --dataset both` | Each `smoke_tests/smoke_.../step2_report.json`: `passed: true`, matching IDs and dimensions. |

The smoke check uses up to 16 common available **training** IDs under the default
skip policy. It generates `N × 512` features for the same N images in each
variant. It does not fit an SVM, and it cannot overwrite the full feature cache.
If fewer than 16 common training images exist, N is smaller; an empty pool fails.

The environment check only tests imports and a simple device operation. The
smoke check is needed to exercise actual pretrained ResNet operators.

The audit can pass when some listed images are absent because skip mode is
enabled. It requires a valid manifest and nonempty available train/val/test
splits; it does not guarantee all 37 classes remain represented after skipping.
Inspect per-variant sample counts before proceeding.

## Step 3 — Train and select the foreground model

```powershell
.\.venv\Scripts\python.exe resnet18_svm.py train --dataset foreground
```

On its first run, this stage extracts 512 features for every available listed
foreground image, including validation and test images, and caches them. The
extractor is frozen and learns no parameters from those images. Only training
features/labels fit the scaler and SVM; only validation determines the winner.

The default search compares 25 C values from 0.001 to 1000 for each of linear
and RBF kernels (RBF gamma `scale`): 50 candidates. Exact ties keep the first
candidate. The selected pipeline remains fitted on training only.

Review that run's:

- `validation_results.csv` and `figures/validation_score_vs_C.png`.
- `validation_predictions.csv`.
- `best_model.joblib` and `final_model_settings.json`.
- `available_manifest.csv`, `missing_images.csv`, and `availability_summary.json`.

## Step 4 — Train and select the original-image model

```powershell
.\.venv\Scripts\python.exe resnet18_svm.py train --dataset original
```

Use the same saved manifest and search grid. This branch extracts its own
features, fits its own scaler, and independently selects its own SVM. The
winning configurations may differ. No additional split generation is needed.

For a single extraction stage before both training commands, you may instead
run `resnet18_svm.py extract --dataset both`. Training then reuses valid caches.

## Step 5 — Evaluate the two saved models

After fixing both models and settings:

```powershell
.\.venv\Scripts\python.exe resnet18_svm.py test --dataset both
```

This command does not fit or reselect the models. Each model receives its own
available test rows. It exports accuracy, macro precision/recall/F1, individual
predictions, uncalibrated decision scores, per-class metrics, a confusion
matrix, directed confusion counts, and up to 12 error examples.

Do not assume the two evaluated test sets are identical under skip mode.
`test_count`, `listed_test_count`, `skipped_test_count`, and the available
manifests describe what was evaluated. This package does not automatically
recompute metrics on a common test intersection.

## Optional — Classify one new image

```powershell
.\.venv\Scripts\python.exe resnet18_svm.py predict --dataset original --image "D:\Pets\example.jpg"
```

For an earlier model or a moved project, supply its actual path:

```powershell
.\.venv\Scripts\python.exe resnet18_svm.py predict --dataset original --image "D:\Pets\example.jpg" --model "D:\Results\best_model.joblib"
```

The foreground model expects an input prepared using the same foreground
procedure. These scripts do not generate a mask for an arbitrary new photo.

## If something fails

| Symptom | Action |
| --- | --- |
| Manifest not found | Supply existing memberships using Step 1. |
| DirectML import fails | Install its separate environment, or deliberately choose CPU/auto. |
| No images found | Point to the folder directly containing JPG/JPEG/PNG files. |
| Duplicate image ID | Remove ambiguity such as both `cat_1.jpg` and `cat_1.png` in one variant. |
| Audit succeeds with missing files | This is permitted by `SKIP_MISSING_IMAGES = True`; inspect availability reports. |
| Paired dimensions disagree | Check the original/foreground pairing and their preparation. |
| Out of memory | Try CPU extraction while retaining native resolution and batch size 1. |
| Cache appears unchanged after a software upgrade | Use `--refresh-cache`; versions/device are not part of the signature. |
| Manifest changed since training | Use the original exact manifest or train a new consistent run. |
| Latest model path no longer exists | Use `--model` with the relocated bundle's path. |

Source images are read only. Audit/cache paths can be updated; train/test/smoke
runs create new directories. Keep the exact manifest and run outputs with the
results you report.
