import argparse
import json
import os
import re
import sys
import wave

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from utils.maths_normalizer import NormalizeGermanMathExpression

math_normalizer = NormalizeGermanMathExpression()

TALIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "talis")


def clean_transcript(text):
    # Remove speaker tokens like s09, s1, l, k (case insensitive)
    text = re.sub(r'\b(?:s\d+|[lk])\b', '', text, flags=re.IGNORECASE)

    # Remove things like (S09: ...), (L: ...), etc.
    text = re.sub(r'\(\s*[a-z]\d*\s*:.*?\)', '', text, flags=re.IGNORECASE)

    # Remove empty or short parentheses (e.g. (), (word))
    text = re.sub(r'\(\s*\w*\s*\)', '', text)

    # Remove trailing dashes (broken words)
    text = re.sub(r'\b\w+-\s*', '', text)

    # Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def parse_args():
    parser = argparse.ArgumentParser(description="Build the TALIS training jsonl from chunked transcripts.")
    parser.add_argument("--output-path", default="data/train.jsonl", help="Where to write the training jsonl.")
    return parser.parse_args()


def main():
    args = parse_args()
    excluded_files = [52, 53, 54]
    indices = [i for i in range(139) if i not in excluded_files]
    train_files = indices[1:86]

    print(train_files)
    train_results = []
    all_seconds = 0

    for index in train_files:
        print(f"Processing file index: {index}")

        if index < 10:
            num = f"00{index}"
        elif index < 100:
            num = f"0{index}"
        else:
            num = f"{index}"

        with open(os.path.join(TALIS_DIR, "translate", f"{num}.json"), "r", encoding="utf-8") as f:
            data = json.load(f)

        chunk_folder = os.path.join(TALIS_DIR, "chunked", f"chunks_{num}")

        for fname in os.listdir(chunk_folder):
            if fname.endswith('.wav'):
                with wave.open(os.path.join(chunk_folder, fname), 'r') as wf:
                    all_seconds += wf.getnframes() / wf.getframerate()

        num_files = len([f for f in os.listdir(chunk_folder) if os.path.isfile(os.path.join(chunk_folder, f))])

        for ind in range(num_files):
            ref = data[ind]["text"]
            clean_text = math_normalizer(clean_transcript(ref))

            train_results.append({
                "chunk": num,
                "file_index": ind,
                "transcript": clean_text,
            })

    print(len(train_results))
    print(all_seconds)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        for item in train_results:
            f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")


if __name__ == "__main__":
    main()