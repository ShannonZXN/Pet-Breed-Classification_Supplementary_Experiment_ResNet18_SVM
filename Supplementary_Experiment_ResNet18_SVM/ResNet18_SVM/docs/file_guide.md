# Code guide

All original Python filenames are retained. The main file keeps experiment
settings near its beginning so readers can compare them with recorded runs.

## Execution order

| Order | File / stage | Inputs | Outputs |
| --- | --- | --- | --- |
| 1 | One manifest preparation route, if needed | Existing memberships, or an explicitly chosen new split | `splits/manifest.csv` |
| 2 | `resnet18_svm.py check` | Installed packages, `DEVICE` | Environment/device report |
| 3 | `self_test.py` | Temporary synthetic fixtures | Functional-test summary |
| 4 | `resnet18_svm.py audit --dataset both` | Images, manifest, annotations | Pairing and availability reports |
| 5 | `resnet18_svm.py smoke --dataset both` | Shared available training IDs, weights | Sample features and checks |
| 6 | `resnet18_svm.py train --dataset foreground` | Available foreground images / cache | Validation-selected fitted pipeline |
| 7 | `resnet18_svm.py train --dataset original` | Available originals / cache | Independent fitted pipeline |
| 8 | `resnet18_svm.py test --dataset both` | Saved models and available test rows | Test metrics, predictions, analysis |

`extract` is an optional explicit cache-building step; `train` also builds or
reuses that cache. `predict` is an optional single-image inference stage.

## resnet18_svm.py

| Functions / class | Responsibility |
| --- | --- |
| `write_json`, `write_csv`, `sha256_file` | Consistent report writing and byte fingerprints. |
| `image_key`, `index_images` | Normalize image IDs and discover supported files. |
| `load_manifest` | Validate shared labels and memberships against official annotations. |
| `image_availability`, `write_availability_report` | Describe exact listed/available/missing subsets. |
| `environment_info`, `torch_modules`, `device_for` | Record software and initialize the chosen backend. |
| `build_extractor`, `ImageFeatureDataset` | Frozen ResNet-18 and dimension-preserving image decoding. |
| `snapshot_images`, `read_feature_cache`, `feature_cache_path`, `get_features` | Validate/reuse or extract aligned 512-dimensional features. |
| `select_rows` | Join each manifest split to cache rows by ID. |
| `metrics`, `fit_candidates` | Training-only fitting and validation-only model selection. |
| `export_validation_curve`, `export_predictions` | Validation plots, per-image labels, and decision scores. |
| `run_audit` | Data readiness and cross-variant availability audit. |
| `train`, `load_model` | Save and restore a variant-matched fitted pipeline with provenance. |
| `test`, `export_test_analysis` | Final evaluation without fitting and detailed error exports. |
| `predict` | Extract one new image and apply the fitted pipeline. |
| `check_environment`, `run_smoke` | Device checks and a shared training sample check. |
| `main` | CLI validation and sequential stage dispatch. |

NumPy, Pillow, scikit-learn, and joblib are used by the main module. PyTorch and
torchvision are imported when extractor/device stages need them. Matplotlib is
imported when exporting figures and uses the noninteractive Agg backend.

## Annotation and split utilities

| File | Dependencies | Behavior |
| --- | --- | --- |
| `oxford_annotations.py` | Python standard library | Read a directory or ZIP; validate official lists and class mappings; write annotation reports. |
| `generate_shared_splits.py` | NumPy, scikit-learn, annotation helper | Generate a new stratified train/val allocation while preserving official test. |
| `build_manifest_from_ids.py` | Standard library, annotation helper | Use existing validation IDs, fill official labels, assign the remaining trainval to train. |
| `merge_split_csvs.py` | Python standard library | Merge existing train/val/test tables and reject repeated normalized IDs. |

Choose one appropriate preparation route. These scripts are alternatives, not
three consecutive steps. They overwrite their configured outputs and never
modify image files. `generate_shared_splits.py` creates memberships; the other
two preserve supplied memberships within their documented schemas.

`oxford_annotations.py --help` and `resnet18_svm.py --help` expose CLI options.
The three split-preparation helpers use editable settings near their tops and
have no argument parser; do not use `--help` to inspect them because executing
them runs their configured task.

## Tests and GitHub configuration

`self_test.py` uses `unittest`, temporary folders, synthetic features, and mock
devices. It verifies program behavior, not pet-breed accuracy. The optional
extractor structure test constructs a network with `weights=None` after checking
the requested pretrained weight enum, so it needs no weight download.

`.github/workflows/checks.yml` compiles the modules, installs a CPU
torch/torchvision pair and dependencies, runs the suite, and checks the bundled
annotation counts. It requires neither pet images nor a saved manifest and
does not run the real smoke/train/test experiment.
