# Dataset Split Files

This directory contains the saved dataset allocation files used to organise the Oxford-IIIT Pet Dataset for the ResNet18–SVM classification experiments.

The dataset contains 7,349 labelled images across 37 pet breeds. The official dataset provides 3,680 trainval images and 3,669 test images.

The saved allocation divides the official trainval set into 80% training and 20% validation, while keeping the official test set unchanged.

## 1. Dataset Allocation

| Dataset Split | Number of Images | Purpose |
|---|---:|---|
| Training | 2,944 | Train the SVM classifier and fit StandardScaler |
| Validation | 736 | Select SVM parameters using validation accuracy |
| Test | 3,669 | Evaluate the selected model |
| Total | 7,349 | 37 pet breeds |

The training and validation sets were generated using stratified sampling with random seed 42.

Stratification was based on the official breed class IDs to maintain breed representation across the two sets.

All 37 breeds are represented in each dataset split.

## 2. Directory Contents

The following files are provided:

| Filename | Description |
|---|---|
| manifest.csv | Complete dataset allocation, including image IDs, breed labels and assigned splits |
| train.csv | Training image IDs and breed labels |
| val.csv | Validation image IDs and breed labels |
| test.csv | Official test image IDs and breed labels |
| train_ids.txt | Training image ID list |
| val_ids.txt | Validation image ID list |
| test_ids.txt | Test image ID list |
| class_counts.csv | Number of images per breed in each split |
| split_summary.json | Split generation settings, dataset statistics, integrity checks and file hashes |

The CSV and TXT files provide different representations of the saved dataset allocation.

The classification program primarily uses manifest.csv.

## 3. Manifest Format

The manifest.csv file contains four required columns:

| Column | Description |
|---|---|
| image_id | Image identifier from the Oxford-IIIT Pet Dataset |
| class_id | Official breed class ID (1–37) |
| class_name | Corresponding breed name |
| split | Dataset assignment: train, val or test |

Each image appears only once in the manifest.

The program uses these records to match image features with the correct breed labels and dataset splits.

Image IDs are matched without requiring the same filename extension or letter case.

## 4. Split Generation and Validation

The saved allocation was generated using:

- Method: sklearn.model_selection.train_test_split
- Random seed: 42
- Stratification: Official breed class IDs
- Training images: 2,944
- Validation images: 736
- Official test images: 3,669

According to split_summary.json, the following checks passed:

- No duplicate image IDs
- No overlap between training, validation and test sets
- Training and validation together cover the official trainval set
- Official test membership remains unchanged
- All 37 breeds are represented in each split
- Image labels match the official annotations

The summary also records SHA-256 hashes for the source annotation files and generated split files.

## 5. Using the Split Files

The main classification program reads:

text splits/manifest.csv 

To use this allocation, keep the saved files in the splits/ directory and run the classification program from the ResNet18_SVM project folder.

The main program does not create a new training–validation split.

It reads the existing manifest and checks image IDs, labels and dataset assignments against the official annotations.

### Important: Do Not Regenerate the Split

The saved split files should be retained when reproducing the reported experiments.

The script generate_shared_splits.py can create a new stratified allocation, but it should not be run unnecessarily.

Even when the same split ratio and random seed are used, regenerating the allocation does not automatically guarantee identical image assignments under different software versions or preparation procedures.

Use the saved manifest.csv as the reference for this allocation.

## 6. Image Availability

The manifest contains 7,349 listed images, but the number of images actually processed depends on file availability in each input folder.

The classification program can skip missing image files while retaining the original dataset assignments.

For the reported experiments:

| Image Condition | Available Test Images |
|---|---:|
| Original — Native Resolution | 3,669 |
| Foreground — Native Resolution | 3,669 |
| Original — Reduced Resolution | 3,669 |
| Foreground — Reduced Resolution | 3,662 |

The reduced-resolution Foreground condition contained seven fewer test images.

This difference should be considered when comparing classification performance between image conditions.

The program records available and missing images in its experiment outputs.

## 7. Reproducibility Notes

The saved manifest provides a consistent dataset allocation for the classification pipeline.

However, the split files alone do not confirm that every historical experiment used exactly the same image memberships.

For exact reproduction of a previously reported result, the manifest should be checked against the file hash recorded in the corresponding original model settings or training records.

The training–validation allocation was not designed as a controlled comparison between 80:20 and 50:50 splits.

The 80:20 allocation was selected to provide more training images while retaining a validation set for SVM model selection.

Differences in classification performance between models using different allocations cannot be attributed to the split ratio alone.

---

For installation, input preparation and execution instructions, see the main project README and START_HERE.md.
