import argparse
import json
import os
import time
from pathlib import Path

from openai import OpenAI, RateLimitError

ROOT = Path(__file__).resolve().parent

PROMPT_PATH = ROOT / "prompts" / "prompt_questioning.txt"
OUTPUT_DIR = ROOT / "gpt5" 

MODEL_NAME = "gpt-5.5"

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY. Run: export OPENAI_API_KEY='your_key_here'")

client = OpenAI(api_key=API_KEY)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score classroom lesson transcripts with GPT via the questioning prompt."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Folder containing '<NNN>_segmented.json' lesson files.",
    )
    parser.add_argument(
        "--num-lessons",
        type=int,
        default=138,
        help="Number of lesson IDs to process (IDs run 1..num-lessons).",
    )
    return parser.parse_args()


def call_openai_with_retry(prompt, max_retries=8, initial_delay=1):
    """Call OpenAI API with exponential backoff retry logic."""
    delay = initial_delay

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an expert in evaluating the quality of classroom interactions based on lesson transcripts."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content

        except RateLimitError as e:
            if attempt == max_retries - 1:
                print(f"Max retries reached. Error: {e}")
                raise

            wait_time = delay * (2 ** attempt)
            print(f"Rate limit hit. Waiting {wait_time} seconds... (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait_time)

        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

    return None


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with PROMPT_PATH.open("r", encoding="utf-8") as f:
        prompt_template = f.read()

    # Lesson IDs are not filtered here: an earlier hardcoded exclusion list
    # silently dropped lesson 55 from every run. The FileNotFoundError and
    # "scores" checks below already skip lessons that shouldn't be processed.
    for i in range(args.num_lessons):
        id = i + 1
        num = f"{id:03d}"
        results = []

        try:
            with (args.input_dir / f"{num}_segmented.json").open("r", encoding="utf-8") as f:
                arr = json.load(f)

            if "scores" not in arr[0]:
                continue

            for c in range(3):
                result_c = []

                for j in arr:
                    transcript = ""

                    for k in j["items"]:
                        transcript = transcript + k["text"]

                    prompt = prompt_template.replace("{transcript}", transcript)

                    print(f"Processing {num}, iteration {c + 1}/3...")

                    result = call_openai_with_retry(prompt)

                    if result:
                        try:
                            cleaned_result = result.replace("```json", "").replace("```", "").strip()
                            parsed_result = json.loads(cleaned_result)
                            result_c.append(parsed_result)
                        except json.JSONDecodeError as e:
                            print(f"Failed to parse JSON for {num}: {e}")
                            print(f"Raw response: {result[:200]}...")
                            with (OUTPUT_DIR / f"{num}_error_{c}.txt").open("a", encoding="utf-8") as f:
                                f.write(result)
                            continue
                    else:
                        print(f"Failed to get result for {num}")

                results.append(result_c)

            with (OUTPUT_DIR / f"{num}_output.json").open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            print(f"Completed {num}")

        except json.JSONDecodeError as e:
            print(f"JSON decode error for {num}: {e}")
            continue
        except FileNotFoundError as e:
            print(f"File not found for {num}: {e}")
            continue
        except Exception as e:
            print(f"Error processing {num}: {e}")
            continue

    print("All processing complete!")


if __name__ == "__main__":
    main()
