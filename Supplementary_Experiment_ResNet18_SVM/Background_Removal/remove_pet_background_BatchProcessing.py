"""Remove backgrounds from every matched original using one erosion radius.

Run: python remove_pet_background_BatchProcessing.py
Help: python remove_pet_background_BatchProcessing.py --help
The filename is retained from the supplied experiment package.
"""

import argparse
import sys
from pathlib import Path


# SETTINGS: defaults match the supplied batch script.
r = 9                         # Common radius in original-image pixels; 0 disables erosion.
PROTECT_ANIMAL_REGION = True   # Keep every original animal pixel labeled 1.
BASE_DIR = Path(__file__).resolve().parent
ORIGINAL_FOLDER = BASE_DIR / "original"
MASK_FOLDER = BASE_DIR / "masks"
OUTPUT_FOLDER = BASE_DIR / "background_removed_all"
# Windows example: ORIGINAL_FOLDER = Path(r"D:\Pets\images")


def main(argv=None):
    """Run with settings above or explicit CLI arguments; return shell status."""
    # Step 1: Accept reproducible overrides without requiring source-code edits.
    parser = argparse.ArgumentParser(
        description="Remove backgrounds from all matched JPG/JPEG and PNG trimap pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--originals", type=Path, default=ORIGINAL_FOLDER, help="JPG/JPEG input folder")
    parser.add_argument("--masks", type=Path, default=MASK_FOLDER, help="PNG trimap input folder")
    parser.add_argument("--output", type=Path, default=OUTPUT_FOLDER, help="Separate results folder")
    parser.add_argument("--radius", type=int, default=r, help="Erosion radius in original-image pixels")
    parser.add_argument(
        "--protect-animal-region", action=argparse.BooleanOptionalAction,
        default=PROTECT_ANIMAL_REGION, help="Restore all label-1 animal pixels after erosion",
    )
    args = parser.parse_args([] if argv is None else argv)

    # Step 2: The common engine validates files, writes pre-processing audits,
    # processes all pairs, and writes the final CSV/JSON reports.
    from pet_background_utils import run_batch
    return run_batch(
        args.originals, args.masks, args.output, radius=args.radius,
        protect_animal_region=args.protect_animal_region,
    )


if __name__ == "__main__":
    # Step 3: Missing masks are intentional skips; actual processing errors fail.
    try:
        sys.exit(main(sys.argv[1:]))
    except (OSError, ValueError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
