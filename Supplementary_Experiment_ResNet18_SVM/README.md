# Cat and Dog Breed Classification
### UTS 31256 · Background Removal and Frozen ResNet18–SVM

This repository brings together two Python packages supporting the supplementary experiments in the UTS 31256 project. The work investigates **image resolution**, **background removal**, and classification using **fixed pretrained ResNet18 features and an SVM**.

**[Background removal](Background_Removal/README.md)** · **[ResNet18–SVM](ResNet18_SVM/README.md)** · **[Run the classifier](ResNet18_SVM/START_HERE.md)** · **[Reported results](ResNet18_SVM/docs/RESULTS.md)**

## 1. Project workflow

The packages have different roles and retain their original code and documentation.

| Package | What it does | Main entry points |
| --- | --- | --- |
| [`Background_Removal/`](Background_Removal/) | Uses **supplied Oxford-IIIT Pet trimaps** to prepare black-background foreground PNGs at native resolution; provides radius previews, batch processing, and output checks. It **does not train an automatic segmentation model**. | `remove_pet_background.py`, `remove_pet_background_BatchProcessing.py`, `audit_pet_outputs_new.py` |
| [`ResNet18_SVM/`](ResNet18_SVM/) | Reads prepared images, extracts **512 frozen ResNet18 features**, trains and selects an SVM on training/validation data, and reports test metrics. | `resnet18_svm.py`, `self_test.py` |

```text
Oxford-IIIT Pet images + supplied trimaps
                  |
                  v
   Background_Removal (native-resolution foreground PNGs)
                  |
                  +------------------------+
                  |                        |
          Original image          Prepared foreground image
                  |                        |
                  +------------+-----------+
                               v
                    Frozen ResNet18
                       512 features
                               v
                  StandardScaler + SVM
                               v
             Validation selection -> Test metrics
```

**Important:** The team's 224 × 224 Original and Foreground images were prepared separately. The background-removal package does **not** reproduce every team resizing or grey-background preparation step, and the classifier does **not** resize images or create masks.

## 2. Repository layout

```text
.
├── README.md                       # This overview
├── Background_Removal/
│   ├── README.md                   # Input, masks, radius controls, commands
│   ├── remove_pet_background.py    # Compare erosion radii on sample images
│   ├── remove_pet_background_BatchProcessing.py
│   ├── audit_pet_outputs_new.py
│   ├── pet_background_utils.py     # Shared implementation functions
│   ├── remove_pet_background_old.py # Legacy/optional script
│   ├── requirements.txt
│   ├── assets/  docs/  tests/
│   └── CHANGELOG.md
└── ResNet18_SVM/
    ├── README.md                   # Method and configuration details
    ├── START_HERE.md               # Detailed execution guide
    ├── resnet18_svm.py             # Extract, select, test, report
    ├── oxford_annotations.py       # Check official labels and splits
    ├── build_manifest_from_ids.py  # Optional existing-ID route
    ├── merge_split_csvs.py         # Optional existing-CSV route
    ├── generate_shared_splits.py   # Optional NEW allocation only
    ├── self_test.py
    ├── requirements.txt
    ├── requirements-directml.txt
    ├── annotations/                # Official text annotation files if supplied
    ├── splits/                     # SAVED allocation files
    │   ├── README.md
    │   ├── manifest.csv
    │   ├── train.csv  val.csv  test.csv
    │   ├── train_ids.txt  val_ids.txt  test_ids.txt
    │   ├── class_counts.csv
    │   └── split_summary.json
    ├── docs/
    └── CHANGELOG.md
```

Keep the contents of each package together. The ResNet18–SVM Python files use local imports and paths relative to their package directory. Large image collections and saved model weights do not need to be stored in GitHub.

## 3. Dataset and saved split files

The **Oxford-IIIT Pet Dataset** contains **7,349 labelled images** from **37 breeds**: **3,680 official trainval** images and **3,669 official test** images.

The repository now includes `ResNet18_SVM/splits/manifest.csv` and its supporting CSV, TXT, and JSON files. The saved allocation records:

| Split | Listed images | Purpose |
| --- | ---: | --- |
| Training | 2,944 | Fit the feature scaler and SVM |
| Validation | 736 | Select SVM hyperparameters |
| Official test | 3,669 | Evaluate the selected classifier |

`split_summary.json` records **stratified sampling by official breed class**, **seed 42**, checks for duplicate/overlapping IDs, unchanged official test membership, and file checksums. The main program reads `splits/manifest.csv` rather than creating a split.

**Reproducibility qualification:** These files are a saved, internally documented **80:20 allocation**. Their presence does not by itself prove that their exact image IDs match the manifest used in every historical reported run. For that, compare the manifest SHA-256 against the original run's `final_model_settings.json` or other saved training record. **Do not regenerate the allocation and assume it reproduces the original experiment.**

The dataset lists contain 3,669 test IDs, but some prepared-image folders contain fewer available files. The reported **reduced-resolution Foreground** condition used **3,662** test images.

## 4. Four input conditions

| Input condition | Source | Background |
| --- | --- | --- |
| Original — native resolution | Original Oxford image dimensions | Natural background |
| Foreground — native resolution | Native-resolution images processed using supplied trimaps | Black replacement background |
| Original — reduced resolution | Team-prepared 224 × 224 images | Original background retained |
| Foreground — reduced resolution | Team-prepared 224 × 224 images | Team-prepared foreground/background treatment |

The classifier processes **one Original/Foreground pair per configured resolution**. Run the native-resolution pair and the reduced-resolution pair separately, using **different output directories**. Each input condition fits its own scaler and SVM while using the same model architecture and parameter-search procedure.

## 5. Quick start

### Step A — Prepare native-resolution foreground images

Follow [Background_Removal/README.md](Background_Removal/README.md) for its input layout and command options. From inside `Background_Removal/`:

```bash
python -m pip install -r requirements.txt
python remove_pet_background.py
python remove_pet_background_BatchProcessing.py
python audit_pet_outputs_new.py --no-copy-unprocessed
```

The preview helps inspect erosion radius. The supplied package defaults to **radius 9** with animal-region protection enabled; verify the options used for the specific experiment. The batch script matches JPG/JPEG images to supplied PNG trimaps and outputs native-size, black-background RGB PNGs in `background_removed_all/` by default. The audit examines filename/count/log consistency, not mask accuracy.

The older `remove_pet_background_old.py` is **optional**, not a required extra stage. `pet_background_utils.py` is imported by the processing scripts.

### Step B — Prepare ResNet18–SVM inputs

Use the instructions in [ResNet18_SVM/START_HERE.md](ResNet18_SVM/START_HERE.md) to install compatible PyTorch, torchvision and scikit-learn dependencies. The original Windows/AMD configuration uses **DirectML** for feature extraction; scikit-learn fits SVMs on the **CPU**. To run on a different machine, deliberately select a compatible backend (CPU or CUDA) in the main script.

Make sure the official annotation texts are accessible to the classifier:

```text
ResNet18_SVM/annotations/list.txt
ResNet18_SVM/annotations/trainval.txt
ResNet18_SVM/annotations/test.txt
```

The saved manifest is already documented at `ResNet18_SVM/splits/manifest.csv`. In `ResNet18_SVM/resnet18_svm.py`, set the image directories for **one resolution pair**:

```python
IMAGE_FOLDERS = {
    "original": BASE_DIR.parent / "Background_Removal" / "original",
    "foreground": BASE_DIR.parent / "Background_Removal" / "background_removed_all",
}
OUTPUT_ROOT = BASE_DIR / "results_native"  # Keep resolution runs separate.
DEVICE = "cpu"                            # Example only; original setup used DirectML.
```

Use real image paths rather than these examples. The input folders must directly contain image files. For the reduced-resolution pair, use the **team-prepared** folders and a different `OUTPUT_ROOT`, such as `results_reduced`. The classifier preserves input dimensions.

### Step C — Check, train, and evaluate

From within `ResNet18_SVM/`, after installing dependencies:

```bash
python self_test.py
python resnet18_svm.py check
python resnet18_svm.py audit --dataset both
python resnet18_svm.py smoke --dataset both
python resnet18_svm.py train --dataset both
python resnet18_svm.py test --dataset both
```

Here, `--dataset both` means the **Original and Foreground** variants of the **currently configured resolution**, **not** all four experimental conditions. The full experiment needs two separate configured runs.

`train` uses frozen ResNet18 to extract or reuse features, fits a `StandardScaler` and SVM candidates using training samples, and selects on validation accuracy. `test` evaluates the saved model **without refitting**. Missing image files are logged, and available sample counts should be checked before comparing conditions.

## 6. Classification settings

| Setting | Implementation |
| --- | --- |
| Feature model | `torchvision` ResNet18; `ResNet18_Weights.IMAGENET1K_V1` |
| Weights | Frozen parameters, evaluation mode, `Identity` replacing final classifier |
| Feature output | 512 values, after global average pooling |
| Input processing | RGB, scale to [0, 1], ImageNet channel normalization; **no resize** |
| Feature scaling | Training-only `StandardScaler` |
| Classifier | scikit-learn `SVC`; linear and RBF kernels |
| Search grid | 25 log-spaced C values from 0.001 to 1000 per kernel; RBF gamma = `scale` |
| Selection | Highest validation accuracy; first candidate wins exact ties |
| Final testing | No train + validation refit; each condition has its own fitted model |

## 7. Reported test results

These values are **from the UTS 31256 project report**, not newly measured by this README or a fresh GitHub run.

| Condition | Test images | Accuracy (%) | Macro F1 (%) |
| --- | ---: | ---: | ---: |
| Original — native | 3,669 | 86.02 | 85.75 |
| Foreground — native | 3,669 | 86.26 | 85.98 |
| **Original — 224 × 224** | **3,669** | **87.57** | **87.48** |
| Foreground — 224 × 224 | 3,662 | 86.18 | 86.17 |

The reduced-resolution Original condition had the highest reported test accuracy. Background removal had **different effects** across resolutions. Because input preparation, sample availability and separately selected SVMs varied, the results should not be used to attribute performance changes to one factor alone.

The 80:20 training–validation allocation was selected to use more training images while retaining validation data. It was **not** compared with a 50:50 split using the same ResNet18–SVM model under an otherwise matched protocol; therefore, its independent performance effect was not established.

## 8. Tests and reproducibility

```bash
# Run within Background_Removal/
python -m unittest discover -s tests -v

# Run within ResNet18_SVM/
python self_test.py
```

These are implementation/functional tests, **not** reproductions of the reported pet classification metrics. Full reproduction also requires prepared image folders, annotation texts, the exact run's manifest, and the matching environment. Do not upload complete image datasets, feature caches, trained weights, or private local paths without checking size and rights.

For more detail, see the [background package guide](Background_Removal/README.md), [classification setup guide](ResNet18_SVM/START_HERE.md), and [reproducibility notes](ResNet18_SVM/docs/reproducibility.md).

## 9. Dataset reference

Parkhi, O. M., Vedaldi, A., Zisserman, A., & Jawahar, C. V. (2012). *Cats and Dogs*. CVPR.

[Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)

**Scope:** This is a code-and-documentation repository for the supplementary image-preparation and frozen-feature classification work. It does not replace the group report or constitute a new result benchmark. Both packages retain their own documentation and revision history.
