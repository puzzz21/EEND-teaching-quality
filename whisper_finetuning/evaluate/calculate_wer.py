import argparse
import json
import os
import sys
from pathlib import Path

from evaluate import load as load_metric

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from utils.maths_normalizer import NormalizeGermanMathExpression
from utils.whisper_normalizer import BasicTextNormalizer


def parse_args():
    parser = argparse.ArgumentParser(description="Recalculate WER over a merged result JSON file.")
    parser.add_argument("--result-path", type=Path, default=Path("result.json"), help="Merged result JSON produced by process_results.py.")
    return parser.parse_args()


args = parse_args()

print("Loading WER metric...")

file_path = args.result_path

with file_path.open("r", encoding="utf-8") as file:
    result = json.load(file)

normalizer = BasicTextNormalizer(True)
math_normalizer = NormalizeGermanMathExpression()
wer_metric = load_metric("wer")


def remove_repeated_sequences(
    text,
    min_ngram=1,
    max_ngram=10,
    max_repeats=2,
):
    words = text.split()
    i = 0
    output = []

    while i < len(words):
        found_repeat = False
        max_possible_ngram = min(
            max_ngram,
            (len(words) - i) // 2,
        )

        for n in range(min_ngram, max_possible_ngram + 1):
            phrase = words[i:i + n]
            repeat_count = 1
            j = i + n

            while (
                j + n <= len(words)
                and words[j:j + n] == phrase
            ):
                repeat_count += 1
                j += n

            if repeat_count > max_repeats:
                output.extend(phrase * max_repeats)
                i = j
                found_repeat = True
                break

        if not found_repeat:
            output.append(words[i])
            i += 1

    return " ".join(output)


for entry in result:
    for data in entry:
        print(data["transcription"])
        if data["transcription"] is None or data["reference"] is None:
            print("Skipping entry due to missing transcription or reference.")
            continue
        prediction = normalizer(
            math_normalizer(data["transcription"])
        )
        reference = normalizer(
            math_normalizer(data["reference"])
        )



        cleaned = remove_repeated_sequences(prediction)
        print(f"Prediction: {cleaned}")
        print(f"Reference: {reference}")

        if cleaned is None or reference is None or cleaned.strip() == "" or reference.strip() == "":
            print("Skipping entry due to missing transcription or reference.")
            continue

        wer = wer_metric.compute(
            predictions=[cleaned],
            references=[reference],
        )

        data["normalized"] = prediction
        data["normalized_reference"] = reference
        data["cleaned"] = cleaned
        data["wer"] = wer


# Save all added fields back to the same file
with file_path.open("w", encoding="utf-8") as file:
    json.dump(
        result,
        file,
        ensure_ascii=False,
        indent=2,
    )

print(f"Updated results saved to {file_path}")