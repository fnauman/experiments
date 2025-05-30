#!/usr/bin/env python
"""
garment_batch_job.py – submit a vision-classification Batch job

Usage:
    python garment_batch_job.py /home/nauman/data/wargon/test_images/ 
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

# --- reuse the “strict” script so the prompt & helpers stay in one place ----------
from garment_analyzer_strict import (
    SYSTEM_PROMPT,               # full controlled-vocabulary prompt  :contentReference[oaicite:0]{index=0}
    image_to_base64,             # down-size & encode helper          :contentReference[oaicite:1]{index=1}
)
# -------------------------------------------------------------------------------

load_dotenv("/home/nauman/.env")                    # so OPENAI_API_KEY is picked up by the SDK
client = OpenAI()                # ← synchronous client is fine for Batch

MODEL          = "gpt-4o-2024-08-06"   # keep in sync with garment_analyzer_strict.py
TEMPERATURE    = 0
IMAGE_DETAIL   = "auto"
MAX_TOKENS     = 256
OUTFILE        = Path("garment_batch_tasks.jsonl")


def build_tasks(image_dir: Path) -> list[dict]:
    """Create one Batch-API task per image (Base64 inlined)."""
    img_paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    tasks: list[dict] = []

    for idx, path in enumerate(img_paths):
        b64 = image_to_base64(path)                # identical resize/quality path
        task = {
            "custom_id": path.stem,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Analyse this garment."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64}",
                                    "detail": IMAGE_DETAIL,
                                },
                            },
                        ],
                    },
                ],
            },
        }
        tasks.append(task)

    # --- sanity check ----------------------------------------------------------
    if len(tasks) != len(img_paths):           # should never happen, but be safe
        raise RuntimeError(
            f"Found {len(img_paths)} images but built {len(tasks)} tasks."
        )
    return tasks


def write_jsonl(tasks: list[dict], outfile: Path) -> None:
    with outfile.open("w") as f:
        for obj in tasks:
            f.write(json.dumps(obj) + "\n")


def submit_batch(jsonl_file: Path, window: str):
    """Upload JSONL & create Batch job."""
    try:
        batch_file = client.files.create(file=jsonl_file.open("rb"), purpose="batch")
        print(f"📤  Uploaded tasks file – id: {batch_file.id}")

        job = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window=window, 
        )
        print("🚀  Batch job submitted successfully!\n")
        print(job)                              # prints id, status, etc.
    except OpenAIError as err:
        print(f"❌  Batch submission failed: {err}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create and submit a Batch job for garment vision analysis."
    )
    parser.add_argument("folder", type=Path, help="Folder with images to classify")
    parser.add_argument(
        "--window", default="24h", help="Batch completion window (e.g. 1h, 24h)"
    )
    args = parser.parse_args()

    tasks = build_tasks(args.folder)
    write_jsonl(tasks, OUTFILE)
    print(f"📝  Wrote {len(tasks)} tasks to {OUTFILE}")

    submit_batch(OUTFILE, args.window)


if __name__ == "__main__":
    main()
