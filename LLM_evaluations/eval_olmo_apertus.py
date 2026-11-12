"""Rate Questioning on ground-truth transcripts with a local Olmo/Apertus model.

Same pipeline as apertus_cot.py / qwen_cot.py (same input, prompt,
resume-skip, 3 runs per lesson, retry-until-valid-JSON), run locally via
`transformers` against one of the models in MODEL_PRESETS below, chosen
with --model.

Tokenization deliberately follows each model's own usage pattern (plain
string -> tokenizer(...), NOT apply_chat_template()) rather than the
chat-template pattern used in apertus_cot.py -- the Olmo-3 quickstart
example tokenizes the raw prompt directly and the "Think" variant still
produces a <think>...</think> block on its own, so no chat wrapper is
applied here. If you find responses don't follow the rating-JSON format
reliably, try switching to tokenizer.apply_chat_template(...) instead.

Requires: pip install torch transformers accelerate
A CUDA GPU with ~16GB+ VRAM is strongly recommended (bf16 weights); falls
back to CPU automatically if none is available, but that will be very slow
across ~1,100 segment-ratings (92 lessons x 3 runs x ~4 segments).

Run from the project root: python3 eval_olmo_apertus.py --model olmo-3.1-32b-instruct
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent

# "Think" models write an explicit reasoning trace before the final answer,
# so they need more headroom than a plain instruct model -- 1024 (enough for
# the Apertus/instruct models) risks truncating a Think model mid-thought
# before it ever emits the rating JSON.
MODEL_PRESETS = {
    "olmo-3-7b-think": {
        "repo_id": "allenai/Olmo-3-7B-Think"
    },
    "olmo-3.1-32b-instruct": {
        "repo_id": "allenai/Olmo-3.1-32B-Instruct"
    },
    "apertus-8b-instruct": {
        "repo_id": "swiss-ai/Apertus-8B-Instruct-2509"
    },
    "apertus-70b-instruct": {
        "repo_id": "swiss-ai/Apertus-70B-Instruct-2509"
    },
}

N_RUNS = 3
MAX_RETRIES_PER_SEGMENT = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rate Questioning on transcripts with a local Olmo or Apertus model."
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_PRESETS),
        default="olmo-3.1-32b-instruct",
        help="Which local model to load and run.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "ground_truth",
        help="Folder of '<lesson>_segmented.json' files to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Folder to write per-lesson result JSON to. Defaults to results/<model>/ground_truth.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override the preset generation budget for the chosen model.",
    )
    return parser.parse_args()


args = parse_args()
preset = MODEL_PRESETS[args.model]

INPUT_DIR = args.input_dir
PROMPT_PATH = ROOT / "prompts" / "prompt_questioning.txt"
OUTPUT_DIR = args.output_dir if args.output_dir is not None else ROOT / "results" / args.model / "ground_truth"
ERROR_DIR = OUTPUT_DIR / "errors"

MODEL_NAME = preset["repo_id"]
MAX_NEW_TOKENS = 258

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

print(f"Loading {MODEL_NAME} on {DEVICE} ({DTYPE}), max_new_tokens={MAX_NEW_TOKENS}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE).to(DEVICE)
model.eval()
print("Model loaded.")


def call_olmo(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Run one generation and return only the newly generated text (decoded).

    Tokenizes the raw prompt string directly (matching the model's own
    quickstart example) rather than via a chat template.
    """
    inputs = tokenizer([prompt], return_tensors="pt", return_token_type_ids=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_k=50,
            top_p=0.95,
        )

    output_ids = generated_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(output_ids, skip_special_tokens=True)


def clean_response(response_text: str) -> str:
    cleaned = response_text.strip()

    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```"):].strip()

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


def get_valid_segment_rating(
    prompt: str,
    output_stem: str,
    run_index: int,
    segment_index: int,
    max_retries: int = MAX_RETRIES_PER_SEGMENT,
) -> dict:
    for attempt in range(max_retries):
        result_text = call_olmo(prompt)

        if not result_text:
            print(f"Empty response for {output_stem}, run {run_index + 1}, "
                  f"segment {segment_index + 1}; retrying (attempt {attempt + 1})...")
            continue

        parsed_result = parse_rating_response(result_text)
        if parsed_result is not None:
            return parsed_result

        print(f"Invalid or missing rating for {output_stem}, run {run_index + 1}, "
              f"segment {segment_index + 1}; retrying (attempt {attempt + 1})...")
        ERROR_DIR.mkdir(parents=True, exist_ok=True)
        error_path = ERROR_DIR / f"{output_stem}_run_{run_index + 1}_segment_{segment_index + 1}_invalid.txt"
        with error_path.open("a", encoding="utf-8") as handle:
            handle.write(result_text + "\n\n")

    print(f"Giving up on {output_stem}, run {run_index + 1}, segment {segment_index + 1} "
          f"after {max_retries} attempts; recording None")
    return {"rating": None}


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

    for run_index in range(N_RUNS):
        result_c = []

        for idx, segment in enumerate(data):
            if not isinstance(segment, dict):
                result_c.append({"rating": None})
                continue

            items = get_items_from_segment(segment)
            transcript = build_transcript(items) if items else ""
            if not transcript:
                result_c.append({"rating": None})
                continue

            prompt = prompt_template.replace("{transcript}", transcript)

            print(f"Processing {file_path.stem}, run {run_index + 1}/{N_RUNS}, "
                  f"segment {idx + 1}/{len(data)}...")
            result_c.append(get_valid_segment_rating(prompt, output_stem, run_index, idx))

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

    files = sorted(INPUT_DIR.glob("*.json"))

    if not files:
        print(f"No JSON files found in {INPUT_DIR}")
        return

    processed = 0
    start = time.time()

    for file_path in files:
        try:
            if process_file(file_path, prompt_template):
                processed += 1
        except Exception as error:
            print(f"Error processing {file_path.name}: {error}")
            continue

    elapsed = time.time() - start
    print(f"All processing complete! Processed {processed} file(s) in {elapsed / 60:.1f} min.")


if __name__ == "__main__":
    main()
