"""
garment_analyzer_batch.py – Flexible runner that chooses between online calls and the Batch API
============================================================================================

• **Online mode** (default):
  *Runs the same concurrent, low‑latency calls you already had – perfect for a few dozen images.*

• **Batch API mode**: 50 % cheaper, massively higher throughput, but asynchronous (jobs finish within 24 h).

  ```bash
  # Explicitly force the Batch API
  python garment_analyzer_batch.py data/images/folder1 --batch-api -o labels.parquet

  # Or let the script decide when #images ≥ threshold (default 1000)
  python garment_analyzer_batch.py data/images/folder1 data/images/folder2 \
      --batch-threshold 500 -r -o labels.csv
  ```

The resulting file always contains the columns:  `image`, `color`, `trend`, `category`, `price` (plus `error` on failures).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pydantic import ValidationError

# Strict analyser helpers (vision + schema‑strict) --------------------------
import garment_analyzer_strict as ga  # re‑use prompt + schema + helpers

load_dotenv()
SYNC_CLIENT = OpenAI(api_key=ga.API_KEY)  # synchronous client for batch endpoints

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

# ───────────────────────── file & path utilities ──────────────────────────

def iter_image_paths(inputs: Iterable[Path], recursive: bool) -> List[Path]:
    """Gather unique image paths from files and/or directories."""
    seen, images = set(), []
    for p in inputs:
        if p.is_dir():
            glob = "**/*" if recursive else "*"
            for f in p.glob(glob):
                if f.suffix.lower() in VALID_EXTS and f not in seen:
                    images.append(f)
                    seen.add(f)
        elif p.is_file() and p.suffix.lower() in VALID_EXTS and p not in seen:
            images.append(p)
            seen.add(p)
    return sorted(images)

# ───────────────────── online (low‑latency) execution ─────────────────────

async def _analyse_one(path: Path) -> dict:
    try:
        b64 = ga.image_to_base64(path)
        parsed = await ga._call_openai(b64)  # type: ignore (private but fine)
        return {"image": str(path), **parsed.model_dump()}
    except (OpenAIError, ValidationError) as err:
        return {"image": str(path), "error": str(err)}

async def analyse_online(paths: List[Path]) -> List[dict]:
    return await asyncio.gather(*[_analyse_one(p) for p in paths])

# ─────────────────────── helper: build batch tasks ────────────────────────

def build_batch_tasks(paths: List[Path]) -> List[dict]:
    schema = ga.GarmentAnalysis.model_json_schema(ref_template="#/$defs/{model}")
    tasks = []
    for p in paths:
        b64 = ga.image_to_base64(p)
        task = {
            "custom_id": str(p),  # use path as deterministic id
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": ga.MODEL,
                "temperature": ga.TEMPERATURE,
                "max_tokens": ga.MAX_TOKENS,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": schema,
                    "strict": True,
                },
                "messages": [
                    {"role": "system", "content": ga.SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Analyse this garment."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64}",
                                    "detail": ga.IMAGE_DETAIL,
                                },
                            },
                        ],
                    },
                ],
            },
        }
        tasks.append(task)
    return tasks

# ─────────────────────────── Batch API execution ──────────────────────────

def run_batch_api(paths: List[Path], out_path: Path, wait: bool = True) -> None:
    """Submit a Batch API job and optionally wait for completion → DataFrame."""
    # 1️⃣ write JSONL
    tasks = build_batch_tasks(paths)
    jsonl_file = Path("batch_tasks.jsonl")
    with jsonl_file.open("w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")

    # 2️⃣ upload + create job (24 h completion window = 50 % discount)  ([cookbook.openai.com](https://cookbook.openai.com/examples/batch_processing))
    file_obj = SYNC_CLIENT.files.create(file=jsonl_file.open("rb"), purpose="batch")
    job = SYNC_CLIENT.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"🆔 Batch job created: {job.id}\n   Status: {job.status}\n")

    if not wait:
        print("💤  Exiting without waiting (use `OPENAI_API_KEY` + job id to poll later).")
        return

    # 3️⃣ poll until done (could take minutes↔hours)
    print("⏳  Waiting for job to finish … (poll every 30 s)")
    while True:
        job = SYNC_CLIENT.batches.retrieve(job.id)
        if job.status == "completed":
            break
        if job.status in {"failed", "expired"}:
            sys.exit(f"❌ Batch ended with status {job.status}")
        time.sleep(30)

    print("✅ Batch completed. Downloading results…")
    result_bytes = SYNC_CLIENT.files.content(job.output_file_id).content

    # 4️⃣ parse
    records = []
    for line in result_bytes.splitlines():
        obj = json.loads(line)
        path = obj.get("custom_id", "<unknown>")
        try:
            ana = ga.GarmentAnalysis.model_validate_json(
                obj["response"]["body"]["choices"][0]["message"]["content"]
            )
            records.append({"image": path, **ana.model_dump()})
        except ValidationError as err:
            records.append({"image": path, "error": str(err)})

    df = pd.DataFrame.from_records(records)
    if out_path.suffix.lower() == ".parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)

    print(f"📊  Saved {len(df)} rows → {out_path.resolve()}")

# ───────────────────────────────── CLI ─────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Garment classifier with optional Batch API.")
    parser.add_argument("inputs", nargs="+", type=Path,
                        help="Image file(s) or directory path(s)")
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="Recursively search directories")
    parser.add_argument("-o", "--output", default="labels.csv",
                        help="Output file (.csv or .parquet)")
    parser.add_argument("--batch-api", action="store_true",
                        help="Force use of the Batch API (cheaper but async)")
    parser.add_argument("--batch-threshold", type=int, default=1000,
                        help="#images ≥ threshold → switch to Batch API automatically")
    parser.add_argument("--no-wait", action="store_true",
                        help="Submit batch job and exit without waiting for results")

    args = parser.parse_args()

    paths = iter_image_paths(args.inputs, args.recursive)
    if not paths:
        sys.exit("❌ No valid images found.")

    use_batch = args.batch_api or len(paths) >= args.batch_threshold

    out_path = Path(args.output)

    if use_batch:
        run_batch_api(paths, out_path, wait=not args.no_wait)
    else:
        print(f"⚡  Using online mode for {len(paths)} image(s)…")
        records = asyncio.run(analyse_online(paths))
        df = pd.DataFrame.from_records(records)
        if out_path.suffix.lower() == ".parquet":
            df.to_parquet(out_path, index=False)
        else:
            df.to_csv(out_path, index=False)
        print(f"📊  Saved {len(df)} rows → {out_path.resolve()}")


if __name__ == "__main__":
    main()
