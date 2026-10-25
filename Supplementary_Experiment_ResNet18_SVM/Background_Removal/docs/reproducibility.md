# Reproducibility and execution guide

[← Back to README](../README.md)

## Execution order

1. Install `requirements.txt` and provide the original images and label masks.
2. Run `remove_pet_background.py` to inspect a seeded sample at several radii.
3. Choose and record the radius and protection mode used for the experiment.
4. Run `remove_pet_background_BatchProcessing.py` with the same settings on the
   complete input folders.
5. Run `audit_pet_outputs_new.py` against that flat batch output directory.
6. Preserve the processing/audit reports alongside the experiment results.

`remove_pet_background_old.py` is an alternative historical workflow, not a
required step. It always uses unprotected erosion and preserves its earlier
report schema. Its default output directory is now `background_removed_legacy`.

## Record the actual configuration

| Item | Why it matters |
| --- | --- |
| Input folder and original filenames | Different available files can change the processed dataset. |
| Mask folder and labels | A different mask version can change which pixels survive. |
| Radius | Measured in original-image pixels; keep the value used in the experiment. |
| Animal protection | Protected and unprotected erosion are distinct conditions. |
| Sample size and random seed | Define the sample drawn from the sorted original list. |
| Comparison radii | Define additional sample outputs, not batch processing. |
| Missing masks and errors | Determine which originals actually produced outputs. |
| Python/package versions | Help reproduce decoding, image writing, and morphology. |

For an environment record, save the following terminal output with your
experiment materials:

```bash
python --version
python -m pip show numpy Pillow scipy
```

`requirements.txt` specifies minimum dependencies for installation. It is not
an exact environment lockfile or a claim that every possible version
combination has been tested.

## Deterministic sampling

The sample workflows sort originals by case-insensitive name, with the original
name as a tie-breaker, then use `random.Random(seed).sample(...)`. The same
available filename list, seed, sample size, and Python sampling behavior give
the same selected files and order. Adding or removing originals can change the
sample even when the seed stays fixed.

Sampling happens before checking masks. A missing selected mask is logged as a
failure; another image is not silently substituted. The batch workflow instead
processes all matched pairs and records originals without masks as skips.

## Existing outputs and interruptions

Use a separate output folder for each experimental condition. Rerunning a
script overwrites outputs with the same names but does not clean other files.
An output left from an older run is not evidence of success in the latest run.

Batch `missing_masks.csv` and `masks_without_originals.csv` are written before
images are opened. `processing_log.csv` and `run_summary.json` are written after
the processing loop. If execution is interrupted, final reports may be absent
or belong to an earlier run, while some new PNGs may already exist.

The audit lists stale outputs through last-run log membership when a readable
log exists. Its absent-output copy list is based on whether a PNG currently
exists, so an old PNG can prevent an original from being copied even when that
PNG was not successful in the latest run. Without a usable log, the audit warns
that it cannot verify latest-run membership.

## Relation to the classification experiment

This utility does not alter the dataset split or perform model selection. Keep
the train/validation/test assignments in the downstream classification pipeline
and check availability in each split. Do not infer a common evaluation set
solely from a total output count.

The default values in this repository reflect the supplied scripts, except for
the isolated legacy output directory. They are not a recommendation to change
the preprocessing of an experiment that has already been run and reported.

## Preparing the GitHub repository

Upload the **contents** of this project folder to the root of your repository,
including `.github`, `.gitignore`, and `.gitattributes`. The README references
its banner and documentation using relative paths. No account-specific links,
repository owner names, or invented contributor names are embedded.

The ignore rules exclude the default dataset/result folders, virtual
environments, Python caches, and archives. Check custom data/output folders
before committing them. The GitHub Actions workflow uses the same synthetic
checks as the local test command; a successful local run is not a hosted CI run.
