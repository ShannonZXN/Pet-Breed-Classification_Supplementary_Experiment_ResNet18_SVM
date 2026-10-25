# Reproducibility and verification

## Keep the exact memberships

The source ZIP did not contain `splits/`, despite the older README saying it
did. The revised package therefore contains a split README rather than
invented historical memberships. Use the exact manifest behind reported results.

The generator's target is 2,944 train / 736 validation / 3,669 test, with 37
classes and seed 42. A newly generated allocation is a new saved artifact until
its membership is checked against the original experiment. Save its CSV/ID lists
and `split_summary.json` and reuse them for every variant/method.

The validator checks the official status of each listed row but accepts
manifests that are subsets of the official lists. A count of 7,349 is therefore
not enforced by `load_manifest()`; inspect your manifest and run metadata.

`test()` hashes the exact manifest bytes, not just its semantic memberships.
Changing row order, line endings, or a trailing newline can trigger the
changed-manifest check even when the same images are listed.

## Missing images and comparability

The supplied code has `SKIP_MISSING_IMAGES=True`. The documentation revision
retains this setting rather than changing the sample policy.

| Condition | Current behavior |
| --- | --- |
| A listed image file is absent | Skip it in that variant and record the missing row. |
| An unlisted image exists | Exclude it from features and learning; report it in audits. |
| A file is present but cannot decode | Fail extraction; this is not a missing-file skip. |
| Two supported files share a normalized ID | Fail indexing because the ID is ambiguous. |
| A selected split has no available feature rows | Fail that stage. |
| Fewer than two training classes remain | Fail SVM training. |
| Some of the 37 training classes are missing after skips | Training may proceed with the remaining classes if at least two remain. |

The manifest checks class coverage before file availability is applied. The
audit's readiness flag requires nonempty available splits, not complete
post-skip class coverage. Use available manifests to inspect class coverage as
well as counts.

Full extraction, training, and evaluation intersect separately with each
variant's files. They do not automatically restrict both variants to a common
image intersection. Two test accuracies can consequently describe different
test subsets. Equalizing only evaluation IDs would still leave any differences
in available training/validation samples to assess separately.

`run_smoke()` uses common available training IDs only for its sample check.
Its paired-dimension check applies to that sample; full extraction checks each
image's own native input dimensions and does not pair every image across variants.

For a completeness requirement, deliberately set `SKIP_MISSING_IMAGES=False`
before a new consistent run and fix absent inputs. Do not relabel or randomly
replace missing images to make counts match. This edition does not add a new
common-subset metric computation.

## What the feature cache identifies

The fingerprint is SHA-256 over the sorted selected image IDs and their
source-byte hashes, plus `EXTRACTOR_SPEC`. Only available requested manifest
images participate; unrelated files are excluded before hashing or decoding.

`read_feature_cache()` also checks exact image ID order, an `N × 512` shape,
and finite values. Invalid caches are re-extracted. Full caches use temporary
writes before replacing the canonical file. Sample caches must use a different
path from the full cache.

The fingerprint does **not** include package versions, device, manifest split
assignments, or labels. Versions and device are recorded as metadata. Changing
which split an image belongs to does not require recomputing its frozen features,
but it does require training an appropriate new model. When changing extractor
software or backend and wishing to recompute features, use `--refresh-cache`
and keep the resulting run's environment records.

Cache shape and ID checks ensure consistency with the expected inputs; they do
not prove that an externally edited NPZ contains authentic ResNet features.

## Source changes after fitting

The saved model records its manifest hash and extraction snapshot. A changed
manifest is rejected at test time. If the feature signature changes:

- Strict missing-file mode rejects the mismatch.
- Skip mode compares hashes of source images present in both snapshots.
- Changed bytes for a shared image reject testing.
- Added/removed files can be allowed under skip mode; the test report preserves
  the training-time and current availability.

An added training file at test time does not refit the saved classifier. The
model remains trained on its earlier recorded training subset. Interpret the
training-time availability as authoritative for how the saved model was fitted.

## Preserve an experiment record

Keep the exact manifest, annotation text hashes, available/missing manifests,
source image checksums, feature settings/environment, fitted model bundle,
validation table, and evaluation outputs with the report. Record upstream image
preparation separately, including background-removal radius or pre-made input
resolution; the classifier hashes image content but does not infer those steps.

`requirements.txt` is a compatibility range, not a lockfile for a historical run.
Use the run's `environment` records and, when sharing a newly verified setup,
save its package list with `python -m pip freeze`. Seed 42 does not guarantee
bit-identical results across different software/hardware stacks.

## Verification of this edition

The six supplied Python files were compared before/after using their parsed
executable statements. Differences allowed by that check were docstrings and
the two references to renamed English documentation in error messages.
All experiment constants, branches, algorithms, model settings, and file
formats remain as supplied. Annotation texts and their original README were
preserved byte-for-byte. Input source hashes are recorded in
[`source_integrity.json`](source_integrity.json).

The existing functional suite was run locally on both source and revised code:
**18 tests discovered, 17 passed, 1 skipped**. The optional untrained ResNet
structure test was skipped because torch/torchvision were not installed.
Compilation, CLI help, annotation counts, and local documentation links were
also checked. A temporary generation exercise checked the configured split
counts and invariants; generated memberships from that exercise are not
distributed as the historical experiment's manifest.

| Package | Local functional-check version |
| --- | --- |
| Python | 3.12.14 |
| NumPy | 2.3.5 |
| Pillow | 12.3.0 |
| scikit-learn | 1.8.0 |
| joblib | 1.5.3 |
| Matplotlib | 3.10.8 |
| torch / torchvision / torch-directml | Not installed |

Synthetic fixtures check behavior and result exports; their scores are not
dataset outcomes. No real pretrained extraction, full pet-image experiment,
CUDA/DirectML hardware run, or hosted GitHub Actions run was performed here.
Run `check`, `audit --dataset both`, and `smoke --dataset both` on the target
machine before full training.
