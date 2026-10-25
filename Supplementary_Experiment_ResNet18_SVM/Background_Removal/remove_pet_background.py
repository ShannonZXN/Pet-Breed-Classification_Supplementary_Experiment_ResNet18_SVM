"""Preview background removal on a seeded sample and compare erosion radii.

Run: python remove_pet_background.py
Help: python remove_pet_background.py --help
Shared processing functions are defined once in pet_background_utils.py.
"""

import argparse
import sys
from pathlib import Path


# SETTINGS: edit these defaults, or override them with command-line arguments.
BASE_DIR = Path(__file__).resolve().parent
ORIGINAL_FOLDER = BASE_DIR / "original"
MASK_FOLDER = BASE_DIR / "masks"
OUTPUT_FOLDER = BASE_DIR / "background_removed"
# Windows example: ORIGINAL_FOLDER = Path(r"D:\Pets\images")
r = 9                         # Erosion radius in original-image pixels; 0 disables it.
PROTECT_ANIMAL_REGION = True   # Restore all original label-1 pixels after erosion.
SAMPLE_SIZE = 35               # Select up to this many originals before mask matching.
RANDOM_SEED = 42               # Reuse this seed to compare the same input sample.
COMPARE_RADII = [0, 3, 6, 9, 12]  # Use [] to save only r and omit comparison sheets.


def build_parser(*, originals, masks, output, radius, sample_size, seed,
                 compare_radii, protect=True, legacy=False):
    """Build the current/legacy sample CLI without importing image dependencies."""
    description = (
        "Compare erosion radii without animal-region protection (legacy workflow)."
        if legacy else "Preview original-size pet background removal on a seeded sample."
    )
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--originals", type=Path, default=originals, help="JPG/JPEG input folder")
    parser.add_argument("--masks", type=Path, default=masks, help="PNG trimap input folder")
    parser.add_argument("--output", type=Path, default=output, help="Separate results folder")
    parser.add_argument("--radius", type=int, default=radius, help="Main erosion radius in pixels")
    parser.add_argument("--sample-size", type=int, default=sample_size, help="Number of originals to sample")
    parser.add_argument("--seed", type=int, default=seed, help="Seed for reproducible selection")
    parser.add_argument(
        "--compare-radii", type=int, nargs="*", default=list(compare_radii), metavar="R",
        help="Additional radii; specify the flag with no values to disable comparison sheets",
    )
    if not legacy:
        parser.add_argument(
            "--protect-animal-region", action=argparse.BooleanOptionalAction,
            default=protect, help="Restore all trimap label-1 animal pixels after erosion",
        )
    else:
        parser.set_defaults(protect_animal_region=False)
    return parser


def main(argv=None):
    """Run with editable defaults, or with an explicit list of CLI arguments.

    Calling main() from Python uses only the settings above. The CLI wrapper
    supplies sys.argv[1:] so unrelated host-program arguments cannot leak in.
    """
    # Step 1: Parse optional overrides; --help works before packages are imported.
    parser = build_parser(
        originals=ORIGINAL_FOLDER, masks=MASK_FOLDER, output=OUTPUT_FOLDER,
        radius=r, sample_size=SAMPLE_SIZE, seed=RANDOM_SEED,
        compare_radii=COMPARE_RADII, protect=PROTECT_ANIMAL_REGION,
    )
    args = parser.parse_args([] if argv is None else argv)

    # Step 2: Run the shared validate -> sample -> process -> report workflow.
    from pet_background_utils import run_sample
    return run_sample(
        args.originals, args.masks, args.output, radius=args.radius,
        protect_animal_region=args.protect_animal_region,
        sample_size=args.sample_size, random_seed=args.seed,
        comparison_radii=args.compare_radii,
    )


if __name__ == "__main__":
    # Step 3: Return a useful shell status; image-level errors are logged inside.
    try:
        sys.exit(main(sys.argv[1:]))
    except (OSError, ValueError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
