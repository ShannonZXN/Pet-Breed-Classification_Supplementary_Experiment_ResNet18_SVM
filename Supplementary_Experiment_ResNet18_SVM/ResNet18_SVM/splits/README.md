# Supply the experiment's shared manifest

The original experiment manifest is provided as manifest.csv. It contains the image identifiers, breed labels and dataset assignments used for the reported ResNet18–SVM experiments.

For reported results, place the exact experiment manifest here as `manifest.csv`.
The four required columns are:

| Column | Meaning |
| --- | --- |
| `image_id` | Official image basename, filename, or path; matching ignores extension/case. |
| `class_id` | Official global breed ID, 1–37, unless an explicit complete mapping is configured. |
| `class_name` | Corresponding official breed name. |
| `split` | `train`, `val`, or `test`; `validation`/`valid` are accepted val aliases. |

One image may appear only once. The manifest must contain all three splits and
37 breeds by default, and its listed training rows must cover every listed
class. Each row is checked against the official annotations. Official test
membership is retained when `KEEP_OFFICIAL_TEST=True`.

Alternatively, preserve memberships using three existing CSVs with
`merge_split_csvs.py`, or use an existing `val_ids.txt` with
`build_manifest_from_ids.py`. Both helpers have editable paths at their tops.

Only when deliberately creating a new allocation, run
`generate_shared_splits.py`. It targets 2,944 training, 736 validation, and 3,669
test IDs with stratification and seed 42. Its saved outputs include the
manifest, CSV/ID lists, class counts, package versions, and checksums. Keep those
outputs fixed across all variants and methods. Generation overwrites files
with the same names and does not prove equality with an absent historical list.

See [the run guide](../START_HERE.md#step-1--supply-the-exact-split-membership)
before selecting a preparation route.
