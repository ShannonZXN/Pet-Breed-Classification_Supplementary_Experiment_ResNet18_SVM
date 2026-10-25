# Revision notes

## GitHub documentation edition — 24 October 2026

- Added an English README with a vector banner, quick-start commands, settings,
  CLI reference, outputs, and links to detailed guides.
- Replaced `README_zh.md`, `START_HERE_zh.md`, and `AMD_RX5700XT_zh.md` with
  English documentation. Error-message references now point to the new names.
- Translated Chinese source comments and added function docstrings and
  explanatory comments around extraction, caching, split validation, model
  selection, evaluation, and reporting.
- Retained all six Python filenames, executable statements, experiment defaults,
  dependency version constraints, CSV/JSON schemas, and command options.
- Documented the actual supplied settings: explicit DirectML, skipping missing
  images, and the `resnet18_svm_outputs_available` output root.
- Corrected stale claims that saved splits and prior audit outputs were bundled.
  Added `splits/README.md`; no missing historical IDs or results were invented.
- Explained independent available subsets, frozen test-feature caching,
  train-only standardization/fitting, exact tie rules, and uncalibrated scores.
- Preserved the original annotation texts and dataset README byte-for-byte.
- Added GitHub Actions configuration, repository ignore rules, and line-ending
  conventions. No CI pass badge or hosted run is claimed.

The source-comparison check excludes docstrings and the two renamed-document
references in error strings. It found no other executable changes. Source
hashes and the comparison scope are recorded in `docs/source_integrity.json`.
See `docs/reproducibility.md` for local test coverage and untested hardware.
