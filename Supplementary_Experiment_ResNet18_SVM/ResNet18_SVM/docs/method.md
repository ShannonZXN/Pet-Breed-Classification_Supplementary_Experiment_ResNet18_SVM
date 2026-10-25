# How the experiment works

This describes the supplied implementation, including its current defaults.
The source files contain comments at the corresponding stages.

## 1. Establish image identities and labels

`image_key()` normalizes basenames without their extensions and ignores letter
case. `index_images()` scans only one folder level for JPG/JPEG/PNG files and
rejects duplicate normalized IDs. Metadata files are not treated as images.

`load_annotations()` checks that official trainval and test are disjoint and
their union is exactly `list.txt`. Column 2 (`CLASS-ID`) supplies the global
breed target. Column 4 (`BREED ID`) is species-local and is not a 37-class label.
The supplied records contain 12 cat breeds and 25 dog breeds. The original
annotation README reverses those species-local ranges; it is preserved unchanged.

`load_manifest()` validates the four-column shared manifest. It rejects repeated
IDs, missing required values, inconsistent class names, unexpected class counts,
invalid splits, and official label disagreements. With `KEEP_OFFICIAL_TEST=True`,
each listed row must retain its official test/non-test membership. It does not
require that a supplied manifest contain every official image.

The manifest must include train, val, and test, and its listed training classes
must cover every class used anywhere in that manifest. An alternative class
numbering requires a complete explicit `CLASS_ID_MAPPING`. No IDs are silently
renumbered.

## 2. Determine the available samples

The default `SKIP_MISSING_IMAGES=True` intersects each variant's files with its
listed IDs. Rows keep their labels and split assignments. This policy handles
absent files, not unreadable or corrupt images: decode failures still stop
feature extraction.

Availability is recorded in three files:

| File | Purpose |
| --- | --- |
| `available_manifest.csv` | Exact available rows and their original split assignments. |
| `missing_images.csv` | Listed rows skipped because their files are absent. |
| `availability_summary.json` | Listed, available, and missing counts by split. |

Each variant is processed independently for full extraction/training/testing.
The paired smoke check instead uses common available training IDs. A smoke pass
therefore does not prove identical full-dataset sample membership.

## 3. Preserve the native image grid

Each stored image is decoded as RGB. The transform converts pixels into a
`3 × H × W` float tensor and applies channel normalization:

```text
scaled pixel = RGB value / 255
normalized channel = (scaled channel - ImageNet mean) / ImageNet standard deviation
```

Means are `[0.485, 0.456, 0.406]`; standard deviations are
`[0.229, 0.224, 0.225]`. No resize, crop, padding, augmentation, or EXIF transpose
is performed. `ImageFeatureDataset` checks the transformed shape against the
stored dimensions. Batch size 1 supports different image sizes without padding.

This is a deliberate project-specific departure from the weight enum's default
transform, which resizes and centrally crops to 224 × 224. The classifier's
behavior at native resolution must be measured using the project's validation
data; it is not inferred from the published ImageNet score.

## 4. Extract fixed features

`build_extractor()` loads `ResNet18_Weights.IMAGENET1K_V1`, replaces only `fc`
with `torch.nn.Identity()`, and keeps the convolutional backbone, adaptive global
average pooling, and flattening. Each image produces a 512-value feature vector.

The model uses `requires_grad_(False)` and `eval()`. Forward passes run under
`torch.no_grad()`. No CNN optimizer, backpropagation, or fine-tuning is used.
The selected CPU/CUDA/DirectML backend is used for extraction; feature vectors
are transferred to CPU for storage and scikit-learn.

Extraction uses all currently available listed images, including test images.
This caches a fixed transformation: it does not use test labels to fit the
extractor or select the SVM. Feature rows retain sorted normalized IDs, input
sizes, and source hashes in `features.npz`.

## 5. Join features to the saved splits

`select_rows()` maps manifest IDs to feature-row positions explicitly. It does
not assume manifest order and cache order are identical. Under skip mode, it
returns only available rows; an empty requested split fails.

The manifest's class IDs remain strings. Explicit label lists are sorted
lexically in evaluation, so IDs can appear as `1, 10, 11, ..., 2, ...` in matrices.
This is a display order, not a class renumbering. CSV headers and figure labels
state which classes each row/column represents.

## 6. Fit candidates and select on validation

Each candidate creates a fresh scikit-learn pipeline:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(...)),
])
```

Only `x_train` and `y_train` are passed to `pipeline.fit()`. The scaler's
per-feature mean and standard deviation are learned from training features,
separately from the fixed pixel normalization in Step 3. Validation/test data
receive the fitted transform.

The default C grid is `10 ** (k / 4)` for integer k from −12 through 12. Kernels
are visited in `linear`, then `rbf` order; C values increase within each kernel.
Linear gamma is unused; RBF uses `gamma="scale"`. There are 50 candidates per
variant with the default configuration.

`SVC` uses `decision_function_shape="ovr"`, `break_ties=True`, a 512 MB kernel
cache, and seed 42. SVC's multiclass training uses one-versus-one internally;
the configured decision function exposes one score per trained class.

Validation accuracy is the default selection metric; `macro_f1` is also
supported. Exact ties keep the first candidate. There is no cross-validation
loop and no refit on train plus validation. The saved winner retains its
training-only scaler and SVM parameters.

## 7. Save the experiment

Each successful training result receives a new UTC timestamped run directory.
The code saves every candidate's validation metrics and fitting time, the
selection curve, validation predictions, a fitted model bundle, and final
settings. Those settings record sample availability, class IDs, the full search
space, selected C boundary status, source/manifest hashes, and environment.

`best_model.joblib` contains the fitted scaler/SVM and settings. It does not
contain the ResNet weights; the torchvision weight cache supplies those for new
extraction. `latest_model.json` records the bundle's absolute path.

## 8. Evaluate the saved model

`test()` loads a variant-matched bundle and checks manifest bytes and source
consistency. It applies the saved pipeline to available test rows without
refitting. Under skip mode, file additions/removals can change availability;
changed bytes for images common to the saved/current snapshots are rejected.

Aggregate outputs are accuracy, macro precision, macro recall, and macro F1.
Macro metrics use the manifest's full class universe with `zero_division=0`.
Thus a class with no available evaluation support can contribute zero rather
than being silently removed from the macro label list.

Further outputs include individual predictions, all class decision-score
columns, per-class metrics, a labelled confusion matrix, and directed
true-breed → predicted-breed confusion counts. Up to 12 error images are copied
in test-manifest order; they are not selected as the most confident errors.
SVM decision scores are not calibrated probabilities or percentage confidence.

## Official implementation references

- [Torchvision ResNet-18 and weight transforms](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
- [Torchvision ResNet implementation](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
- [scikit-learn SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [scikit-learn data leakage and pipelines](https://scikit-learn.org/stable/common_pitfalls.html)

These describe the underlying libraries. Experiment settings and local behavior
above come from the supplied source, rather than from library defaults alone.
