# Pet Breed Classification
## Background Removal and Frozen ResNet18–SVM

**UTS 31256 · Supplementary experiments on image resolution, background preparation, and an alternative training–validation allocation**

[Background removal guide](Background_Removal/README.md) · [Classification guide](ResNet18_SVM/README.md) · [Classification setup](ResNet18_SVM/START_HERE.md) · [Reported results](ResNet18_SVM/docs/RESULTS.md)

## Overview

This repository brings together two Python packages used for pet-image preparation and breed classification:

| Component | Purpose |
| --- | --- |
| **Background_Removal** | Prepare foreground images from original photographs and supplied trimap annotations; preview erosion settings, process matched images, and audit output files. |
| **ResNet18_SVM** | Extract frozen pretrained ResNet18 features, select an SVM using validation accuracy, and evaluate predictions for 37 pet breeds. |

The packages remain separate. Background processing produces image files; the classification program reads those files through configured input paths. The background package does not train a segmentation model, and the classification package does not resize images or remove their backgrounds.

```text
Original photographs + supplied trimaps
                 ↓
       Background preparation
                 ↓
Original images / prepared Foreground images
                 ↓
       Frozen pretrained ResNet18
                 ↓
       512-dimensional features
                 ↓
    StandardScaler → SVM selection
                 ↓
      Test predictions and metrics
```

## Repository structure

```text
.
├── README.md                         # This combined overview
├── Background_Removal/
│   ├── README.md
│   ├── requirements.txt
│   ├── remove_pet_background.py
│   ├── remove_pet_background_BatchProcessing.py
│   ├── audit_pet_outputs_new.py
│   ├── pet_background_utils.py
│   ├── remove_pet_background_old.py
│   ├── CHANGELOG.md
│   ├── assets/
│   ├── docs/
│   └── tests/
└── ResNet18_SVM/
    ├── README.md
    ├── START_HERE.md
    ├── requirements.txt
    ├── requirements-directml.txt
    ├── resnet18_svm.py
    ├── oxford_annotations.py
    ├── build_manifest_from_ids.py
    ├── merge_split_csvs.py
    ├── generate_shared_splits.py
    ├── self_test.py
    ├── CHANGELOG.md
    ├── annotations/
    ├── splits/
    └── docs/
```

Keep each package's internal structure unchanged. Links in this README assume the folder names shown above. Large datasets, model files, and feature caches can remain outside the repository.

## Experimental configurations

The supplementary classification experiments covered four input conditions:

| Condition | Image source | Background |
| --- | --- | --- |
| Original, native resolution | Original-resolution dataset images | Retained |
| Foreground, native resolution | Prepared original-resolution images | Replaced with black in the reported experiment |
| Original, reduced resolution | Team-prepared 224 × 224 images | Retained |
| Foreground, reduced resolution | Team-prepared 224 × 224 foreground images | Team-prepared replacement background |

The reduced-resolution inputs were prepared separately by the team. They should not be assumed to use the same mask processing or replacement colour as the native-resolution foreground images.

The reported split list contained **2,944 training**, **736 validation**, and **3,669 test** images. This 80:20 training–validation allocation reserved more examples for SVM fitting than the approximately 50:50 allocation used by the other four project pipelines. It was not a controlled comparison of split ratios using the same model.

## Running the workflow

### 1. Prepare the environments and data

Follow each package's own installation guide and requirements file.

| Package | Environment notes |
| --- | --- |
| Background removal | Python 3.10 or later; NumPy, Pillow, and SciPy; CPU processing. |
| ResNet18–SVM | Python 3.12 setup described in its guide; compatible PyTorch/torchvision plus `requirements.txt`. |
| Windows/AMD extraction | The supplied classifier defaults to DirectML. Follow its [DirectML guide](ResNet18_SVM/docs/DIRECTML.md) in a separate environment. |

For a different machine, deliberately select an installed classification backend. `DEVICE = "cpu"` uses CPU; `"auto"` selects CUDA when available, otherwise CPU. The code does not automatically select Apple's MPS backend. SVM fitting remains on CPU.

Provide the original images, matching trimaps, and the four prepared classification inputs as needed. For the classifier, also supply:

```text
ResNet18_SVM/annotations/list.txt
ResNet18_SVM/annotations/trainval.txt
ResNet18_SVM/annotations/test.txt
ResNet18_SVM/splits/manifest.csv
```

**The exact historical manifest is still required for reproducing the reported allocation.** A `splits/README.md` is an instruction file, not the manifest. The required CSV columns are `image_id,class_id,class_name,split`. Do not generate a new split and present it as the original one. See the [split instructions](ResNet18_SVM/splits/README.md).

### 2. Prepare and check native-resolution foreground images

Open a terminal inside `Background_Removal`. With the documented default layout, place JPG/JPEG originals in `original/` and matching PNG trimaps in `masks/`. Filenames must have matching stems.

```bash
python -m pip install -r requirements.txt
python remove_pet_background.py
```

Inspect the sample outputs before running the batch step. The current package defaults to radius **9** and animal-region protection **enabled**; use the radius and protection setting corresponding to the experiment being reproduced.

Then process the dataset and audit its output:

```bash
python remove_pet_background_BatchProcessing.py
python audit_pet_outputs_new.py --no-copy-unprocessed
```

The batch script writes opaque, black-background RGB PNGs into `background_removed_all/`, preserving the original dimensions. Its input trimaps use labels 1 (animal), 2 (background), and 3 (uncertain boundary). The audit checks filenames and processing-log membership, not segmentation accuracy.

`remove_pet_background_old.py` is an optional legacy workflow, not another required step. `pet_background_utils.py` contains shared functions and is imported by the processing scripts. Detailed options and external-folder examples are in the [background removal guide](Background_Removal/README.md).

### 3. Connect the prepared images to the classifier

In `ResNet18_SVM/resnet18_svm.py`, set `IMAGE_FOLDERS` to the actual input directories. For the default background-package layout, a native-resolution CPU example is:

```python
IMAGE_FOLDERS = {
    "original": BASE_DIR.parent / "Background_Removal" / "original",
    "foreground": BASE_DIR.parent / "Background_Removal" / "background_removed_all",
}
OUTPUT_ROOT = BASE_DIR / "results_native"
DEVICE = "cpu"
```

Use the flat batch-output folder, not the sample comparison sheets. These path/backend settings are an example for running on another machine, not a claim that the original experiments used CPU.

For reduced-resolution experiments, point both inputs to the team's existing 224 × 224 folders and use a different output root, such as `results_reduced`. Retain the same saved split assignments and selection procedure. The program reads stored image dimensions without resizing them.

### 4. Validate, train, and test

After installing the classification environment and supplying the required inputs, open a terminal inside `ResNet18_SVM`. Run each stage after the preceding checks succeed:

```bash
python self_test.py
python resnet18_svm.py check
python resnet18_svm.py audit --dataset both
python resnet18_svm.py smoke --dataset both
python resnet18_svm.py train --dataset both
python resnet18_svm.py test --dataset both
```

Here, `both` means **Original and Foreground for the currently configured resolution pair**, not all four conditions at once. Repeat the workflow with the reduced-resolution paths and separate output root to cover the other pair. Review available sample counts even when the audit succeeds: missing files are skipped and recorded.

Training extracts or reuses frozen features, fits candidate scalers and SVMs on training data only, and selects by validation accuracy. Testing evaluates the selected models without further fitting. Exact commands, output filenames, and troubleshooting are in [START_HERE.md](ResNet18_SVM/START_HERE.md).

## Core classification settings

| Setting | Value |
| --- | --- |
| Pretrained weights | `ResNet18_Weights.IMAGENET1K_V1` |
| Feature extractor | Frozen ResNet18 in evaluation mode; final classifier replaced with `Identity` |
| Features | 512 values per image |
| Image preparation | RGB conversion, pixel scaling to [0, 1], ImageNet channel normalisation; dimensions unchanged |
| Feature scaling | `StandardScaler`, fitted on training features only |
| SVM search | Linear and RBF; 25 logarithmically spaced C values from 0.001 to 1000 per kernel |
| RBF gamma | `scale` |
| Model selection | Highest validation accuracy; first candidate wins exact ties |
| Final models | Separate scaler and SVM for each condition; no train–validation refit |

The classification program reads an existing manifest. The split helpers are alternative preparation routes, not a sequence: `merge_split_csvs.py` preserves existing CSV assignments; `build_manifest_from_ids.py` uses existing validation IDs; `generate_shared_splits.py` deliberately creates a new allocation.

## Reported results

The following values are transcribed from the project report, Table 5.5. They are **not results from a new run of this combined repository**.

| Input condition | Test images | Accuracy (%) | Macro F1 (%) |
| --- | ---: | ---: | ---: |
| Original, native resolution | 3,669 | 86.02 | 85.75 |
| Foreground, native resolution | 3,669 | 86.26 | 85.98 |
| Original, reduced resolution | 3,669 | **87.57** | **87.48** |
| Foreground, reduced resolution | 3,662 | 86.18 | 86.17 |

Reduced-resolution Original achieved the highest scores among these four conditions. Background removal did not consistently improve performance. See the [reported-results notes](ResNet18_SVM/docs/RESULTS.md) for the source and interpretation.

## Reproducibility and limitations

Exact reproduction requires the historical manifest, prepared images, preprocessing settings, and recorded environment—not just matching split counts or a random seed. Keep the background-processing logs and classification run records with the corresponding experiment.

The reduced Foreground group had seven fewer test images. Resolution, padding, background colour, mask processing, and independently selected SVM settings also differed across conditions. Consequently, the results do not isolate the effect of resolution, background removal, or the 80:20 allocation alone. Supplied trimaps also do not establish performance with automatic segmentation.

Both packages provide synthetic tests, which check implementation behaviour rather than the reported pet-classification accuracy:

```bash
# From Background_Removal/
python -m unittest discover -s tests -v

# From ResNet18_SVM/
python self_test.py
```

This README combines navigation and execution guidance; it does not merge or replace the two implementations. Refer to each package's README and changelog for its detailed behaviour and revision history.
