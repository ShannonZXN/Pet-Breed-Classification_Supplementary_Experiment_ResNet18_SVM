"""Reproduce the legacy sampled workflow without protecting animal pixels.

Run: python remove_pet_background_old.py
Help: python remove_pet_background_old.py --help
This historical variant is optional. Its default output folder is now separate
so running it cannot overwrite results from the protected sample workflow.
"""

import sys
from pathlib import Path

from remove_pet_background import build_parser


# SETTINGS: preserve the legacy experiment's unprotected erosion and sampling.
BASE_DIR = Path(__file__).resolve().parent
ORIGINAL_FOLDER = BASE_DIR / "original"
MASK_FOLDER = BASE_DIR / "masks"
OUTPUT_FOLDER = BASE_DIR / "background_removed_legacy"  # Separate from protected results.
# Windows example: ORIGINAL_FOLDER = Path(r"D:\Pets\images")
r = 3                         # Historical trial radius; not an established optimum.
SAMPLE_SIZE = 10               # Original legacy default, unlike the current sample's 35.
RANDOM_SEED = 42               # Same seed selects the same sample for unchanged inputs.
COMPARE_RADII = [0, 3, 6, 9, 12]  # Use [] to save only r and omit comparison sheets.


def main(argv=None):
    """Run the legacy settings through the shared sample engine, always unprotected."""
    # Step 1: Reuse the sample argument definitions, without protection switches.
    parser = build_parser(
        originals=ORIGINAL_FOLDER, masks=MASK_FOLDER, output=OUTPUT_FOLDER,
        radius=r, sample_size=SAMPLE_SIZE, seed=RANDOM_SEED,
        compare_radii=COMPARE_RADII, legacy=True,
    )
    args = parser.parse_args([] if argv is None else argv)

    # Step 2: Disable animal-region restoration and preserve legacy CSV fields.
    from pet_background_utils import run_sample
    return run_sample(
        args.originals, args.masks, args.output, radius=args.radius,
        protect_animal_region=False, sample_size=args.sample_size,
        random_seed=args.seed, comparison_radii=args.compare_radii,
        legacy_report=True,
    )


if __name__ == "__main__":
    # Step 3: A missing selected mask or an image error returns status 1.
    try:
        sys.exit(main(sys.argv[1:]))
    except (OSError, ValueError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
