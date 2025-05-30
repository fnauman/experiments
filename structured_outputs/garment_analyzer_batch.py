"""
garment_analyzer_batch.py – Batch runner that stores classification results in a pandas DataFrame
==============================================================================================

This script wraps **garment_analyzer_strict.py** and adds :

* directory traversal (optionally recursive) so you can pass whole folders;
* collection of every successful response into a pandas **DataFrame**;
* auto‑save of that DataFrame to CSV / Parquet.

────────────────────────── Example CLI invocations ──────────────────────────

1. Single image
   -------------

    python garment_analyzer_strict.py data/images/folder1/test1.jpg

   (This is exactly the same as before – you still get JSON in stdout.)

2. Many images / folders with a DataFrame output
   ---------------------------------------------

    # Non‑recursive – only top‑level files in the two folders
    python garment_analyzer_batch.py data/images/folder1 data/images/folder2 \
        --output labels.csv

    # Recursive walk through *one* directory tree, saving to Parquet
    python garment_analyzer_batch.py data/images/folder1 -r -o labels.parquet

The resulting file contains columns:  `image`, `color`, `trend`, `category`,
`price`  (plus an `error` column if any files failed).
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAIError
from pydantic import ValidationError

# We import our strict analyser helpers (async) from the sibling module
import garment_analyzer_strict as ga

load_dotenv()

# ─────────────────────────────── helper utils ──────────────────────────────

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def iter_image_paths(inputs: Iterable[Path], recursive: bool) -> List[Path]:
    """Yield unique, sorted image paths from files / directories given on CLI."""
    seen = set()
    images: List[Path] = []
    for p in inputs:
        if p.is_dir():
            glob_pattern = "**/*" if recursive else "*"
            for file in p.glob(glob_pattern):
                if file.suffix.lower() in VALID_EXTS and file not in seen:
                    images.append(file)
                    seen.add(file)
        elif p.is_file() and p.suffix.lower() in VALID_EXTS:
            if p not in seen:
                images.append(p)
                seen.add(p)
    return sorted(images)


async def analyse_to_records(paths: List[Path]):
    """Return a list of dicts suitable for a pandas DataFrame."""
    records = []

    async def _analyse(path: Path):
        try:
            b64 = ga.image_to_base64(path)
            parsed = await ga._call_openai(b64)  # type: ignore (private helper is fine here)
            records.append({"image": str(path), **parsed.model_dump()})
        except (OpenAIError, ValidationError) as err:
            # Store the error alongside the image so we can inspect later
            records.append({"image": str(path), "error": str(err)})

    await asyncio.gather(*[_analyse(p) for p in paths])
    return records


# ──────────────────────────────── main routine ─────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify garment images in bulk and save results to a DataFrame.")
    parser.add_argument("inputs", nargs="+", type=Path,
                        help="Paths to image files or directories containing images")
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="Recursively search directories for images")
    parser.add_argument("-o", "--output", default="labels.csv",
                        help="Output file name (extension .csv or .parquet determines format)")

    args = parser.parse_args()

    # Resolve paths and gather images
    img_paths = iter_image_paths(args.inputs, args.recursive)
    if not img_paths:
        sys.exit("❌ No images found matching the provided inputs.")

    print(f"🔎 Found {len(img_paths)} image(s). Submitting to OpenAI…")

    # Run analysis concurrently
    records = asyncio.run(analyse_to_records(img_paths))

    # Convert to DataFrame
    df = pd.DataFrame.from_records(records)
    print("\n📊 DataFrame preview:\n", df.head())

    # Persist to disk
    out_path = Path(args.output)
    if out_path.suffix.lower() == ".parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
    print(f"\n✅ Saved {len(df)} rows to {out_path.resolve()}")


if __name__ == "__main__":
    main()
