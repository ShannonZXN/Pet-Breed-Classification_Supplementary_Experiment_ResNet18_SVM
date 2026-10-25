# Revision notes

## GitHub preparation

- Replaced the outdated Chinese `README.txt` with an English Markdown README,
  navigation, a local SVG banner, and dedicated method/reproducibility guides.
- Retained the four supplied entry-script filenames for traceability.
- Added English instructions, function descriptions, configuration comments,
  and comments explaining the processing stages and report meanings.
- Consolidated repeated processing utilities into `pet_background_utils.py`.
- Added command-line options while keeping editable per-script settings.
- Preserved current radius/protection/sample defaults and the original erosion,
  black-fill, image-dimension, seeded-sampling, and missing-mask behavior.
- Separated the legacy workflow's output folder as `background_removed_legacy`
  so it does not overwrite protected sample results.
- Validated raw radii before sorting/deduplication, preventing mixed-type errors
  and Boolean values from hiding behind equal integer values.
- Added input/output path separation checks, defensive audit-log parsing, and
  a clear rejection of sample-subfolder logs in the flat batch audit.
- Added synthetic regression tests and a GitHub Actions workflow, plus ignore
  and text-formatting rules for repository use.

The supplied archive did not contain dataset images or sample results. The
revised archive also contains no dataset images, trained models, or generated
experiment results. Regression fixtures are generated temporarily by the tests.
