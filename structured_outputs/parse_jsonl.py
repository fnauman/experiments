# garment_results_to_dataframe.py
"""
Extract structured garment classifications from an OpenAI Batch‑API result JSONL into a Pandas DataFrame.

Usage:
    python garment_results_to_dataframe.py --job_id <BATCH_JOB_ID> [--output results.jsonl]  # download & parse
    python garment_results_to_dataframe.py --input <LOCAL_JSONL_PATH>                  # parse existing file

Examples:
    python garment_results_to_dataframe.py --job_id abcd1234 --output batch_out.jsonl
    python garment_results_to_dataframe.py --input batch_out.jsonl
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from openai import OpenAI, OpenAIError


def download_results(job_id: str, output: Path) -> Path:
    """
    Download batch job result file by job_id and save to output path.
    Returns the path to the downloaded JSONL file.
    """
    client = OpenAI()
    try:
        job = client.batches.get(job_id=job_id)
        file_id = job.result_file_id
        raw = client.files.download(file_id=file_id)
    except OpenAIError as e:
        raise RuntimeError(f"Failed to download batch results: {e}")

    output.write_bytes(raw)
    print(f"Downloaded batch results to {output}")
    return output


def parse_jsonl(path: Path) -> pd.DataFrame:
    """
    Parse the batch JSONL file and return a DataFrame with columns:
    custom_id, color, trend, category, price
    """
    records = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            cid = entry.get('custom_id')
            # Navigate into body choices
            try:
                content = entry['response']['body']['choices'][0]['message']['content']
                # content is a JSON string; parse it
                data = json.loads(content)
            except Exception as e:
                print(f"Warning: failed to parse entry {cid}: {e}")
                continue

            # Collect fields
            records.append({
                'custom_id': cid,
                'color': data.get('color'),
                'trend': data.get('trend'),
                'category': data.get('category'),
                'price': data.get('price'),
            })

    df = pd.DataFrame.from_records(records)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Load and parse OpenAI Batch JSONL into a DataFrame."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--job_id', type=str, help='Batch job ID to download results from')
    group.add_argument('--input', type=Path, help='Path to local JSONL results file')
    parser.add_argument('--output', type=Path, default=Path('batch_results.jsonl'),
                        help='Path to save downloaded JSONL')
    parser.add_argument('--csv', type=Path, default=None,
                        help='Optional path to save DataFrame as CSV')
    args = parser.parse_args()

    if args.job_id:
        jsonl_path = download_results(args.job_id, args.output)
    else:
        jsonl_path = args.input

    df = parse_jsonl(jsonl_path)
    print(df)

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"Saved DataFrame to {args.csv}")


if __name__ == '__main__':
    main()
