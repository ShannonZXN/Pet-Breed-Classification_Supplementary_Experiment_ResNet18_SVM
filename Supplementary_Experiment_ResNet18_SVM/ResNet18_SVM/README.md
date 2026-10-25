<p align="center">
  <img src="assets/project-banner.svg" alt="ResNet18 Feature Extraction with SVM Classification" width="100%">
</p>

<p align="center">
  <strong>Frozen ResNet18 Features · SVM Classification · Image Resolution and Background Comparison</strong>
</p>

<p align="center">
  <a href="START_HERE.md">Start Here</a> ·
  <a href="docs/EXPERIMENT_SETUP.md">Experiment Setup</a> ·
  <a href="docs/file_guide.md">Code Guide</a> ·
  <a href="docs/RESULTS.md">Results</a> ·
  <a href="docs/reproducibility.md">Reproducibility</a>
</p>

# Pretrained ResNet18 Feature Extraction with SVM

### UTS 31256 — Cat and Dog Breed Classification

This project uses a pretrained ResNet18 network to extract image features and a Support Vector Machine (SVM) to classify 37 cat and dog breeds from the Oxford-IIIT Pet Dataset.

The experiments investigate how image resolution and background removal affect classification performance. An alternative 80:20 training–validation split was also adopted to provide more training samples while retaining a validation set for model selection.

The implementation uses Python, PyTorch, torchvision and scikit-learn.

---

## 1. Project Overview

The classification method consists of two main stages:

```text
Input Image
     |
     v
Pretrained ResNet18
(Frozen Parameters)
     |
     v
512-Dimensional Features
     |
     v
StandardScaler
     |
     v
SVM Classifier
(Linear / RBF)
     |
     v
Predicted Pet Breed
```

ResNet18 uses pretrained ImageNet-1K weights and remains frozen throughout feature extraction.

Its original classification layer is replaced with an Identity layer, allowing the network to output 512-dimensional feature vectors.

These features are standardised and passed to an SVM classifier trained using labelled pet images.

Different SVM configurations are evaluated on the validation set before the selected model is tested on the independent test set.

---

## 2. Experimental Design

### 2.1 Image Resolution and Background Comparison

Four image conditions were investigated:

| Image Condition | Resolution | Background |
|---|---|---|
| Original — Native | Original image dimensions | Original background |
| Foreground — Native | Original image dimensions | Background removed |
| Original — Reduced | 224 × 224 | Original background |
| Foreground — Reduced | 224 × 224 | Background removed |

The native-resolution images were obtained from the Oxford-IIIT Pet Dataset.

The reduced-resolution images were prepared separately by the project team.

The native-resolution foreground images were generated using the supplied trimap annotations, binary masks and mask processing.

The background removal implementation is provided separately in the `Background_Removal` package.

The reduced-resolution foreground images were supplied by the team and used as prepared inputs.

**Important:** The classification program does not resize images or generate foreground masks. It reads the prepared images while preserving their existing dimensions.

### 2.2 Alternative Training–Validation Split

The experiments used an 80:20 training–validation allocation.

The purpose was to increase the number of images available for SVM training while retaining sufficient validation samples for model selection.

The Oxford-IIIT Pet Dataset contains 7,349 labelled images across 37 breeds.

The official dataset provides 3,680 trainval images and 3,669 test images.

The saved manifest divides the official trainval set into:

| Dataset Split | Images |
|---|---:|
| Training | 2,944 |
| Validation | 736 |
| Test | 3,669 |
| **Total** | **7,349** |

The saved split was generated using stratified sampling with random seed 42.

The official test membership remains unchanged.

The dataset allocation files are provided in the [`splits/`](splits/) directory.

These include:

- `manifest.csv`
- `train.csv`
- `val.csv`
- `test.csv`
- `train_ids.txt`
- `val_ids.txt`
- `test_ids.txt`
- `class_counts.csv`
- `split_summary.json`

The split files record image identifiers, breed labels and dataset assignments.

The saved manifest should be used when reproducing the experiment. Generating a new split is unnecessary and may produce different image assignments.

**Note:** The 80:20 allocation was an alternative experimental configuration, not a controlled comparison against a 50:50 split using the same model. Therefore, the results do not independently establish whether 80:20 improves classification accuracy.

---

## 3. Classification Implementation

### 3.1 Pretrained ResNet18 Feature Extraction

The implementation uses:

- Architecture: ResNet18
- Pretrained weights: `ResNet18_Weights.IMAGENET1K_V1`
- Framework: PyTorch and torchvision
- Network parameters: Frozen
- Network mode: Evaluation
- Output dimension: 512 features

Input images are converted to RGB and normalised using the ImageNet parameters:

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

The input images retain their stored dimensions.

Global average pooling allows the network to produce 512-dimensional feature vectors from images of different supported sizes.

### 3.2 Feature Standardisation

The extracted features are standardised using scikit-learn's `StandardScaler`.

The scaler is fitted using training features only.

The same fitted scaler is applied to validation and test features to avoid using evaluation data when calculating standardisation parameters.

### 3.3 SVM Classification and Model Selection

The implementation uses scikit-learn's `SVC`.

Two SVM kernels were evaluated:

- Linear
- Radial Basis Function (RBF)

For each kernel, 25 logarithmically spaced C values between 0.001 and 1000 were tested.

The RBF kernel used:

```python
gamma = "scale"
```

This produced 50 candidate configurations for each image condition.

The model with the highest validation accuracy was selected.

When multiple configurations achieved the same accuracy, the first configuration in the search order was retained.

The selected scaler and SVM were used for final testing without further training.

Each image condition had its own fitted scaler and SVM.

---

## 4. Repository Structure

```text
ResNet18_SVM/
│
├── README.md
├── START_HERE.md
├── CHANGELOG.md
│
├── requirements.txt
├── requirements-directml.txt
│
├── resnet18_svm.py
├── oxford_annotations.py
├── build_manifest_from_ids.py
├── generate_shared_splits.py
├── merge_split_csvs.py
├── self_test.py
│
├── annotations/
│   └── README.md
│
├── splits/
│   ├── README.md
│   ├── manifest.csv
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   ├── train_ids.txt
│   ├── val_ids.txt
│   ├── test_ids.txt
│   ├── class_counts.csv
│   └── split_summary.json
│
├── docs/
│   ├── EXPERIMENT_SETUP.md
│   ├── file_guide.md
│   ├── DIRECTML.md
│   ├── RESULTS.md
│   ├── reproducibility.md
│   └── REVIEW_NOTES.md
│
└── assets/
    └── project-banner.svg
```

### Main Python Files

| File | Purpose |
|---|---|
| `resnet18_svm.py` | Main classification pipeline, feature extraction, SVM training, model selection and evaluation |
| `oxford_annotations.py` | Load and validate Oxford-IIIT Pet annotations |
| `build_manifest_from_ids.py` | Build a manifest from existing validation image IDs |
| `generate_shared_splits.py` | Generate a new stratified dataset split when explicitly required |
| `merge_split_csvs.py` | Combine existing training, validation and test CSV files |
| `self_test.py` | Run functional tests using synthetic data |

The helper scripts are provided for dataset preparation and verification.

The existing saved manifest should be used for reproducing the reported data allocation.

---

## 5. Computing Environment

The reported experiments used a Windows-based Python environment.

### Main Environment

| Component | Configuration |
|---|---|
| Operating System | Windows 10, 64-bit |
| Python | 3.12 |
| Deep Learning Framework | PyTorch |
| Feature Extraction | Pretrained ResNet18 |
| GPU | AMD Radeon RX 5700 XT |
| GPU Backend | DirectML |
| SVM Training | CPU, using scikit-learn |

The ResNet18 feature extraction stage used GPU acceleration through DirectML.

SVM training, validation and prediction were performed using scikit-learn on the CPU.

### Required Dependencies

Main Python packages include:

- PyTorch
- torchvision
- NumPy
- Pillow
- scikit-learn
- joblib
- matplotlib

The dependency files are:

- `requirements.txt`
- `requirements-directml.txt`

The exact setup instructions are provided in [`START_HERE.md`](START_HERE.md).

DirectML configuration instructions are available in [`docs/DIRECTML.md`](docs/DIRECTML.md).

The program also supports CPU and CUDA configurations when the required dependencies are installed and the device setting is changed.

---

## 6. How to Run

### Step 1 — Prepare the Dataset

Download the Oxford-IIIT Pet Dataset from:

https://www.robots.ox.ac.uk/~vgg/data/pets/

Prepare the required image folders and official annotation files.

The classification program expects the input images to have already been prepared.

For foreground processing, refer to the separate `Background_Removal` package.

### Step 2 — Confirm the Dataset Split

The repository includes the saved split files in:

```text
splits/
```

The main classification program reads:

```text
splits/manifest.csv
```

Do not regenerate the dataset split when using the saved allocation.

The manifest contains the image IDs, labels and dataset assignments required for training, validation and testing.

### Step 3 — Configure Input Paths

Open:

```text
resnet18_svm.py
```

Update the image folder paths:

```python
IMAGE_FOLDERS = {
    "original": Path(r"D:\Pets\original"),
    "foreground": Path(r"D:\Pets\background_removed_all"),
}
```

Replace these example paths with the locations of your actual images.

Choose the appropriate device:

```python
DEVICE = "directml"
```

Other supported settings include `"cpu"`, `"cuda"` and `"auto"`.

The default DirectML setting matches the original Windows/AMD configuration.

### Step 4 — Check the Environment

Run:

```bash
python resnet18_svm.py check
```

Then run the functional tests:

```bash
python self_test.py
```

### Step 5 — Check the Dataset

```bash
python resnet18_svm.py audit --dataset both
```

A small feature-extraction test can be performed using:

```bash
python resnet18_svm.py smoke --dataset both
```

### Step 6 — Train the SVM Models

```bash
python resnet18_svm.py train --dataset both
```

This stage extracts or reuses ResNet18 features, fits the SVM candidates and selects the best configuration using validation accuracy.

Original and Foreground models are trained separately.

### Step 7 — Evaluate the Models

```bash
python resnet18_svm.py test --dataset both
```

The selected models are evaluated using their corresponding available test images.

The test stage does not retrain or reselect the SVM.

### Step 8 — Repeat for the Alternative Resolution

To evaluate both native-resolution and reduced-resolution inputs, configure the corresponding image folders and run the pipeline separately.

Use a different `OUTPUT_ROOT` for each resolution so that the saved feature caches, selected models and evaluation results remain separate.

The program does not automatically switch between all four image conditions.

For detailed execution instructions, see:

**[START_HERE.md](START_HERE.md)**

---

## 7. Experimental Results

The reported test results for the four image conditions were:

| Input Condition | Test Images | Accuracy (%) | Macro F1 (%) |
|---|---:|---:|---:|
| Original — Native | 3,669 | 86.02 | 85.75 |
| Foreground — Native | 3,669 | 86.26 | 85.98 |
| **Original — Reduced** | **3,669** | **87.57** | **87.48** |
| Foreground — Reduced | 3,662 | 86.18 | 86.17 |

The best reported performance was achieved using reduced-resolution Original images:

**Test Accuracy: 87.57%**

**Macro F1: 87.48%**

Reducing resolution improved the test accuracy of Original images by 1.55 percentage points.

However, background removal did not consistently improve classification accuracy.

At native resolution, removing the background slightly increased accuracy.

At reduced resolution, background removal decreased accuracy by approximately 1.39 percentage points.

These results indicate that image resolution and background processing can affect the classification performance of fixed pretrained features.

The values above are taken from the project report. They are not newly generated benchmark results from this repository documentation.

Further results and explanations are provided in:

**[docs/RESULTS.md](docs/RESULTS.md)**

---

## 8. Limitations and Reproducibility

Several factors should be considered when interpreting the results.

### Image Preparation

Native-resolution and reduced-resolution images were prepared using different processing methods.

Differences in resizing, padding, background replacement and mask processing may affect the features extracted by ResNet18.

The observed differences cannot be attributed to image resolution alone.

### Available Test Images

The reduced-resolution Foreground condition contained seven fewer test images than the other three conditions.

The classification program records missing images and evaluates the available samples.

### Training–Validation Allocation

The supplementary ResNet18–SVM experiments used an 80:20 split, while the other classification pipelines in the group project used approximately 50:50.

Although the main experiments used the official test image set, differences in training allocations and other experimental settings limit direct comparisons between methods.

The reported results do not independently demonstrate that the 80:20 split is better than 50:50.

### Saved Experiment Records

The repository contains dataset allocation files and documented implementation settings.

However, the saved split files alone do not prove that their exact image memberships match every historical training run.

For exact reproduction of a reported model, the saved manifest should also be checked against the manifest hash stored in its original training records.

The complete prepared image datasets, pretrained model weights, trained classifiers and historical experiment outputs are not included in this source-code repository.

Additional reproducibility information is provided in:

**[docs/reproducibility.md](docs/reproducibility.md)**

---

## 9. Relationship to Background Removal

This repository contains the ResNet18–SVM classification implementation.

The related `Background_Removal` package provides the foreground image preparation procedures.

The combined workflow is:

```text
Oxford-IIIT Pet Dataset
          |
          v
Original Images + Trimap Annotations
          |
          v
Background Removal
(Binary Mask + Erosion)
          |
          v
Prepared Foreground Images
          |
          v
Frozen ResNet18 Feature Extraction
          |
          v
512-Dimensional Features
          |
          v
SVM Training and Model Selection
          |
          v
Test Evaluation and Classification Results
```

The reduced-resolution Original and Foreground images were supplied separately by the project team.

Background removal and classification remain separate programs, allowing each stage to be examined independently.

---

## 10. References

**Dataset**

Parkhi, O. M., Vedaldi, A., Zisserman, A., & Jawahar, C. V. (2012). *Cats and Dogs*. IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Oxford-IIIT Pet Dataset:

https://www.robots.ox.ac.uk/~vgg/data/pets/

**ResNet**

He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

**Software Documentation**

PyTorch:

https://pytorch.org/

Torchvision ResNet18:

https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html

Scikit-learn SVM:

https://scikit-learn.org/stable/modules/svm.html

---

## 11. Project Notes

This repository documents the supplementary experiments conducted for UTS 31256 — Image Processing and Pattern Recognition.

It includes the Python implementation of frozen ResNet18 feature extraction, SVM model selection and evaluation, together with the saved dataset split files and supporting documentation.

The code is provided for academic review and reproducibility.

The implementation, experimental configurations and reported findings should be interpreted together with the accompanying project report.

No separate software licence has been assigned to this repository.
