"""Rate non-diarized classroom transcripts with the Gemini API.

Requires:
    pip install -U google-genai
    export GEMINI_API_KEY="your-key"

The output structure remains compatible with the Claude/GPT rating runs:
three arrays (runs), each containing one {"rating": ...} object per segment.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path

from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parent

INPUT_DIR = ROOT / "ground_truth"
PROMPT_PATH = ROOT / "prompts" / "prompt_questioning.txt"
OUTPUT_DIR = ROOT / "gemini_nod" 

# Stable Gemini model suited to high-volume structured classification.
MODEL_NAME = "gemini-3.6-flash"
N_RUNS = 3
START_ID = 1
MAX_API_ATTEMPTS = 6

SYSTEM_PROMPT = (
    "You are an expert in evaluating the quality of classroom "
    "interactions based on lesson transcripts. Follow the supplied rubric "
    "carefully and return only the requested structured rating."
)

RATING_SCHEMA = {
    "type": "object",
    "properties": {
        "rating": {
            "type": "integer",
            "enum": [1, 2, 3, 4],
            "description": "The rubric rating from 1 through 4.",
        },
    },
    "required": ["rating"],
    "additionalProperties": False,
}

if not os.environ.get("GEMINI_API_KEY"):
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Run: "
        "export GEMINI_API_KEY='your-key'"
    )

client = genai.Client()


def parse_rating(response) -> int | None:
    """Extract and validate a structured 1–4 rating."""
    parsed = getattr(response, "parsed", None)

    if isinstance(parsed, dict):
        value = parsed.get("rating")
    else:
        try:
            text = response.text or ""
        except Exception:
            text = ""

        try:
            value = json.loads(text).get("rating")
        except (json.JSONDecodeError, AttributeError, TypeError):
            # Defensive fallback if a provider returns surrounding text.
            match = re.search(r'["\']rating["\']\s*:\s*([1-4])', text)
            value = int(match.group(1)) if match else None

    try:
        rating = int(value)
    except (TypeError, ValueError):
        return None

    return rating if rating in {1, 2, 3, 4} else None


def rate_segment(prompt: str) -> int | None:
    """Return a 1–4 Gemini rating, retrying transient/API failures."""
    for attempt in range(1, MAX_API_ATTEMPTS + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_json_schema=RATING_SCHEMA,
                    max_output_tokens=128,
                    thinking_config=types.ThinkingConfig(
                        thinking_level="minimal"
                    ),
                ),
            )
        except Exception as error:
            if attempt == MAX_API_ATTEMPTS:
                print(f"Gemini API failed after {attempt} attempts: {error}")
                return None

            # Exponential backoff with jitter helps with 429 and 5xx errors.
            wait_seconds = min(2 ** (attempt - 1), 60) + random.random()
            print(
                f"Gemini API error: {error}; retrying in "
                f"{wait_seconds:.1f}s ({attempt}/{MAX_API_ATTEMPTS})..."
            )
            time.sleep(wait_seconds)
            continue

        rating = parse_rating(response)
        if rating is not None:
            return rating

        finish_reasons = []
        for candidate in getattr(response, "candidates", None) or []:
            finish_reasons.append(str(getattr(candidate, "finish_reason", None)))

        if attempt == MAX_API_ATTEMPTS:
            print(
                "Gemini returned no valid rating after all attempts; "
                f"finish reasons: {finish_reasons}"
            )
            return None

        wait_seconds = min(2 ** (attempt - 1), 60) + random.random()
        print(
            "Gemini returned an invalid/blocked response; "
            f"finish reasons: {finish_reasons}; retrying in "
            f"{wait_seconds:.1f}s ({attempt}/{MAX_API_ATTEMPTS})..."
        )
        time.sleep(wait_seconds)

    return None


def build_transcript(items: list[dict]) -> str:
    """Join non-diarized utterances in their original order."""
    parts = []
    for item in items:
        text = item.get("text", "")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return " ".join(parts)


def get_items_from_segment(segment: dict) -> list[dict]:
    if isinstance(segment.get("items"), list):
        return segment["items"]
    if "text" in segment:
        return [segment]
    return []


def process_file(file_path: Path, prompt_template: str) -> bool:
    output_stem = file_path.stem.replace("_segmented", "")
    output_path = OUTPUT_DIR / f"{output_stem}_output.json"

    if output_path.exists():
        print(f"Skipping {file_path.name}: output already exists")
        return False

    with file_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        print(f"Skipping {file_path.name}: expected a list of segments")
        return False

    results = []

    for run_index in range(N_RUNS):
        run_results = []

        for segment_index, segment in enumerate(data):
            # Append exactly one result per segment to preserve alignment.
            if not isinstance(segment, dict):
                run_results.append({"rating": None})
                continue

            items = get_items_from_segment(segment)
            transcript = build_transcript(items) if items else ""
            if not transcript:
                run_results.append({"rating": None})
                continue

            prompt = prompt_template.replace("{transcript}", transcript)
            print(
                f"Processing {file_path.stem}, "
                f"run {run_index + 1}/{N_RUNS}, "
                f"segment {segment_index + 1}/{len(data)}..."
            )
            run_results.append({"rating": rate_segment(prompt)})

        results.append(run_results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    print(f"Completed {file_path.stem}")
    return True


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_PATH}")

    with PROMPT_PATH.open("r", encoding="utf-8") as handle:
        prompt_template = handle.read()

    files = []
    for file_path in sorted(INPUT_DIR.glob("*_segmented.json")):
        match = re.match(r"(\d+)", file_path.stem)
        if match is None:
            print(f"Skipping {file_path.name}: no leading lesson number")
            continue
        if int(match.group(1)) < START_ID:
            continue
        files.append(file_path)

    if not files:
        print(
            f"No JSON files at or above lesson {START_ID:03d} "
            f"in {INPUT_DIR}"
        )
        return

    processed = 0
    for file_path in files:
        try:
            if process_file(file_path, prompt_template):
                processed += 1
        except Exception as error:
            print(f"Error processing {file_path.name}: {error}")

    print(f"All processing complete! Processed {processed} file(s).")


if __name__ == "__main__":
    main()
