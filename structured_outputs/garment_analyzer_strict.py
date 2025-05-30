"""
garment_analyzer_strict.py – Vision classification with OpenAI Structured Outputs (strict=True)
Requires: openai>=1.16.0, pydantic>=2.7, pillow, python-dotenv

Usage:
    python garment_analyzer_strict.py path/to/image1.jpg path/to/image2.png
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel, Field, ValidationError
from PIL import Image

# ─────────────────────────────────────── configuration ──
load_dotenv("/home/nauman/.env")
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-2024-08-06")  # vision-enabled model (e.g. gpt-4.1, gpt-4o-mini)
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0))  # deterministic classification
IMAGE_DETAIL = os.getenv("OPENAI_IMAGE_DETAIL", "auto")   # "low"|"auto"|"high"
MAX_TOKENS = 256

client = AsyncOpenAI(api_key=API_KEY)

# ─────────────────────────────── controlled vocabularies ──
class Color(str, Enum):
    BLACK = "black"; WHITE = "white"; RED = "red"; GREEN = "green"
    BLUE = "blue"; YELLOW = "yellow"; BROWN = "brown"; ORANGE = "orange"
    PURPLE = "purple"; PINK = "pink"; GRAY = "gray"; BEIGE = "beige"
    METALLIC = "metallic"; MULTICOLOR = "multicolor"

class Trend(str, Enum):
    ATHLETIC = "athletic"; CASUAL = "casual"; FORMAL = "formal"
    STREETWEAR = "streetwear"; VINTAGE = "vintage"; CLASSIC = "classic"
    TRADITIONAL = "traditional"

class Category(str, Enum):
    MENS = "men's"; WOMENS = "women's"; KIDS = "kid's"; UNISEX = "unisex"

class Price(str, Enum):
    BUDGET = "budget"; MID_RANGE = "mid-range"; PREMIUM = "premium"

class GarmentAnalysis(BaseModel):
    """Structured classification of a single garment image."""
    color: Color = Field(..., description="Primary color of the garment")
    trend: Trend = Field(..., description="Style trend category")
    category: Category = Field(..., description="Target demographic")
    price: Price = Field(..., description="Estimated price range")

# ──────────────────────────────────────── prompts ──
ALLOWED_VALUES = (
    f"- color: {{{', '.join(c.value for c in Color)}}}\n"
    f"- trend: {{{', '.join(t.value for t in Trend)}}}\n"
    f"- category: {{{', '.join(c.value for c in Category)}}}\n"
    f"- price: {{{', '.join(p.value for p in Price)}}}"
)
SYSTEM_PROMPT = (
    "You are a senior fashion merchandiser.\n"
    "Classify the garment in the image into **exactly** the controlled vocabularies below. "
    "Return *only* a JSON object with the keys `color`, `trend`, `category`, `price`. "
    "Do **not** include any additional keys or explanatory text.\n\n"
    "Controlled vocabularies:\n"
    f"{ALLOWED_VALUES}"
)

# ───────────────────────────── helper functions ──
def image_to_base64(path: Path, max_side: int = 512, quality: int = 88) -> str:
    """Open, RGB‑convert, thumbnail and base64‑encode an image."""
    with Image.open(path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode()

async def _call_openai(b64: str) -> GarmentAnalysis:
    """Single request to OpenAI with strict JSON‑schema output."""
    # schema = GarmentAnalysis.model_json_schema(ref_template="#/$defs/{model}")

    response = await client.beta.chat.completions.parse(
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        response_format=GarmentAnalysis, 
        messages=[
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
    )

    return response.choices[0].message.parsed

async def analyse_paths(paths: Iterable[Path]) -> None:
    """Analyse many images concurrently and pretty‑print results."""
    async def _analyse(path: Path):
        try:
            b64 = image_to_base64(path)
            parsed = await _call_openai(b64)
            print(f"\n✅ {path.name}\n{parsed.model_dump_json(indent=2)}")
        except (OpenAIError, ValidationError) as err:
            print(f"\n❌ {path.name} – {err}")

    await asyncio.gather(*[_analyse(p) for p in paths])

# ────────────────────────────── CLI entry‑point ──
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classify garment images with OpenAI Vision models (strict outputs).")
    parser.add_argument("images", nargs="+", type=Path, help="Path(s) to image file(s)")
    args = parser.parse_args()

    asyncio.run(analyse_paths(args.images))
