from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score classroom lesson transcripts with Qwen via the questioning prompt."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "medium_all",
        help="Folder of '<lesson>_segmented.json' files to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "qwen_largest" / "medium",
        help="Folder to write per-lesson result JSON to.",
    )
    parser.add_argument(
        "--api-url",
        default=os.getenv("REMOTE_API_URL"),
        required=os.getenv("REMOTE_API_URL") is None,
        help="Completions endpoint of the remote inference server (OpenAI-compatible /v1/completions). "
        "Defaults to $REMOTE_API_URL if set.",
    )
    return parser.parse_args()


args = parse_args()

INPUT_DIR = args.input_dir
PROMPT_PATH = ROOT / "prompts" / "prompt_questioning.txt"
OUTPUT_DIR = args.output_dir
ERROR_DIR = OUTPUT_DIR / "errors"

API_URL = args.api_url
API_KEY = os.getenv("REMOTE_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing REMOTE_API_KEY. Run: export REMOTE_API_KEY='your_key_here'")

MODEL_NAME = "qwen3.5-397b-a17b"
# MODEL_NAME = "openai-gpt-oss-120b"
# MODEL_NAME = "qwen3.6-27b"
# qwen3.6-27b  qwen3.5-397b-a17b


def call_qwen_with_retry(
    prompt: str,
    max_tokens: int = 4000,
    temperature: float = 0.3,
    max_retries: int = 8,
    initial_delay: int = 2,
) -> str | None:
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=120)
            response.raise_for_status()
            result = response.json()

            return result["choices"][0]["text"]
        except requests.exceptions.RequestException as error:
            if attempt == max_retries - 1:
                print(f"Max retries reached. Error: {error}")
                raise

            wait_time = initial_delay * (2 ** attempt)
            print(f"Request error, retrying in {wait_time}s... ({attempt + 1}/{max_retries})")
            time.sleep(wait_time)

    return None


def clean_response(response_text: str) -> str:
    cleaned = response_text.strip()

    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json") :].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```") :].strip()

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    json_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if json_match:
        cleaned = json_match.group(0).strip()

    return cleaned


def parse_rating_response(response_text: str) -> dict | None:
    try:
        parsed = json.loads(clean_response(response_text))
    except json.JSONDecodeError:
        return None

    if isinstance(parsed, dict) and "rating" in parsed:
        return parsed

    return None


def call_qwen_until_valid_rating(
    prompt: str,
    max_tokens: int = 4000,
    temperature: float = 0.3,
    max_retries: int = 8,
    initial_delay: int = 2,
) -> dict:
    attempt = 0

    while True:
        attempt += 1
        result_text = call_qwen_with_retry(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
            initial_delay=initial_delay,
        )

        if not result_text:
            print(f"Empty response, retrying prompt... attempt {attempt}")
            time.sleep(initial_delay)
            continue

        parsed_result = parse_rating_response(result_text)
        if parsed_result is not None:
            return parsed_result

        print(f"Missing rating or invalid JSON, retrying prompt... attempt {attempt}")
        time.sleep(initial_delay)


def get_valid_segment_rating(
    prompt: str,
    output_stem: str,
    run_index: int,
    segment_index: int,
) -> dict:
    while True:
        result_text = call_qwen_with_retry(prompt)

        if not result_text:
            print(
                f"Empty response for {output_stem}, run {run_index + 1}, segment {segment_index + 1}; retrying..."
            )
            continue

        parsed_result = parse_rating_response(result_text)
        if parsed_result is not None:
            return parsed_result

        print(
            f"Invalid or missing rating for {output_stem}, run {run_index + 1}, segment {segment_index + 1}; retrying..."
        )
        ERROR_DIR.mkdir(parents=True, exist_ok=True)
        error_path = ERROR_DIR / f"{output_stem}_run_{run_index + 1}_segment_{segment_index + 1}_invalid.txt"
        with error_path.open("a", encoding="utf-8") as handle:
            handle.write(result_text + "\n\n")

        time.sleep(2)


def build_transcript(items: list[dict]) -> str:
    parts = []

    for item in items:
        speaker = str(item.get("speaker", ""))

        text = item.get("text", "")
        if not isinstance(text, str):
            continue

        text = re.sub(r"[.,!?]", "", text).strip()

        if text:
            parts.append(f"({speaker}: {text})")

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

    for run_index in range(3):
        result_c = []

        for idx, segment in enumerate(data):
            if not isinstance(segment, dict):
                continue

            items = get_items_from_segment(segment)
            if not items:
                continue

            transcript = build_transcript(items)
            if not transcript:
                continue

            prompt = prompt_template.replace("{transcript}", transcript)

            print(f"Processing {file_path.stem}, run {run_index + 1}/3, segment {idx + 1}/{len(data)}...")
            parsed_result = get_valid_segment_rating(prompt, output_stem, run_index, idx)
            result_c.append(parsed_result)

            time.sleep(1)

        results.append(result_c)

        if run_index < 2:
            time.sleep(1)

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

    files = sorted(INPUT_DIR.glob("*.json"))

    if not files:
        print(f"No JSON files found in {INPUT_DIR}")
        return

    processed = 0

    for file_path in files:
        try:
            if process_file(file_path, prompt_template):
                processed += 1
        except Exception as error:
            print(f"Error processing {file_path.name}: {error}")
            continue

    print(f"All processing complete! Processed {processed} file(s).")


if __name__ == "__main__":
    main()
