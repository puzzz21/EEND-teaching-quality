"""Rate Nature of Discourse on NON-diarized ground-truth transcripts with Claude.

Paired with claude_nod_diarized.py the same way gpt5_nod.py pairs with
gpt5_nod_diarized.py: same rating pipeline, but input comes from the plain
`ground_truth` folder (no per-item speaker labels) and the prompt is the
non-diarized variant. 3 runs per lesson, one rating per segment,
index-aligned placeholders on failure, resume-safe (skips existing outputs).
Output format matches the GPT-5 runs so the same merge/QWK scripts work.

Requires: pip install anthropic, and ANTHROPIC_API_KEY in the environment
(or an `ant auth login` profile).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parent

INPUT_DIR = ROOT / "ground_truth" 
PROMPT_PATH = ROOT / "prompts" / "prompt_questioning.txt"
OUTPUT_DIR = ROOT / "claude" / "ground_truth"

MODEL_NAME = "claude-opus-4-8"
N_RUNS = 3

# Only lessons with this id or higher are processed (1 = all lessons).
START_ID = 1

SYSTEM_PROMPT = (
    "You are an expert in evaluating the quality of classroom "
    "interactions based on lesson transcripts."
)

RATING_SCHEMA = {
    "type": "object",
    "properties": {
        "rating": {"type": "integer", "enum": [1, 2, 3, 4]},
    },
    "required": ["rating"],
    "additionalProperties": False,
}

client = anthropic.Anthropic(max_retries=5)


def rate_segment(prompt: str) -> int | None:
    """Return the 1-4 rating for one segment, or None on failure."""
    try:
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            output_config={
                "effort": "medium",
                "format": {"type": "json_schema", "schema": RATING_SCHEMA},
            },
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.RateLimitError as error:
        print(f"Rate limited beyond SDK retries: {error}")
        return None
    except anthropic.APIStatusError as error:
        print(f"API error {error.status_code}: {error.message}")
        return None
    except anthropic.APIConnectionError as error:
        print(f"Connection error: {error}")
        return None

    if response.stop_reason == "refusal":
        print("Request refused; recording None")
        return None

    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        return int(json.loads(text)["rating"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        print(f"Unparseable response ({error}): {text[:200]}")
        return None


def build_transcript(items: list[dict]) -> str:
    # Non-diarized: no speaker field on these items, just plain utterance
    # text in order. Joined with spaces (unlike gpt5_nod.py's original
    # no-separator concatenation, which glues consecutive utterances
    # together and erases turn boundaries -- a bug identified earlier).
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
        result_c = []
        for idx, segment in enumerate(data):
            # Always append exactly one entry per segment so ratings stay
            # index-aligned with the segments when merged later.
            if not isinstance(segment, dict):
                result_c.append({"rating": None})
                continue

            items = get_items_from_segment(segment)
            transcript = build_transcript(items) if items else ""
            if not transcript:
                result_c.append({"rating": None})
                continue

            prompt = prompt_template.replace("{transcript}", transcript)
            print(f"Processing {file_path.stem}, run {run_index + 1}/{N_RUNS}, segment {idx + 1}/{len(data)}...")
            result_c.append({"rating": rate_segment(prompt)})

        results.append(result_c)

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
        print(f"No JSON files at or above lesson {START_ID:03d} in {INPUT_DIR}")
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
