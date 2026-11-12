import argparse
import json
import re
from pathlib import Path
from statistics import mean

from evaluate import load as load_metric


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recalculate ASR WER for a folder of JSON files."
    )
    parser.add_argument(
        "--json-folder",
        type=Path,
        required=True,
        help="Folder containing the per-chunk JSON files.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=89,
        help="Only process files with an index >= this value.",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help=(
            "Path to write the summary JSON to. Defaults to "
            "'primeline_whisper_large_v3_turbo_german_recalculated_wer.json' "
            "inside --json-folder."
        ),
    )
    return parser.parse_args()


args = parse_args()

json_folder = args.json_folder

start_index = args.start_index

summary_file = (
    args.summary_file
    if args.summary_file is not None
    else json_folder / "primeline_whisper_large_v3_turbo_german_recalculated_wer.json"
)

file_pattern = re.compile(
    r"^primeline_whisper-large-v3-turbo-german__(\d+)\.json$"
)

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


def iterate_objects(value):
    """Recursively find objects inside JSON arrays/objects."""
    if isinstance(value, dict):
        yield value

        for nested_value in value.values():
            if isinstance(nested_value, (dict, list)):
                yield from iterate_objects(nested_value)

    elif isinstance(value, list):
        for item in value:
            yield from iterate_objects(item)


def average_or_none(values):
    return mean(values) if values else None


# Find only matching files numbered 089 and higher
json_files = []

for file_path in json_folder.iterdir():
    if not file_path.is_file():
        continue

    match = file_pattern.fullmatch(file_path.name)

    if not match:
        continue

    file_index = int(match.group(1))

    if file_index >= start_index:
        json_files.append((file_index, file_path))

json_files.sort(key=lambda item: item[0])

all_teacher_wers = []
all_student_wers = []
per_file_results = []

for file_index, file_path in json_files:
    print(f"Processing: {file_path.name}")

    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    file_teacher_wers = []
    file_student_wers = []
    skipped = 0

    for item in iterate_objects(data):
        required_keys = {
            "speaker",
            "normalized_reference",
            "nomalized_predictions_no_repe",
        }

        missing_keys = required_keys - item.keys()

        if missing_keys:
            # Ignore parent objects and unrelated nested objects
            continue

        speaker = str(item["speaker"]).strip().upper()
        reference = item["normalized_reference"]
        prediction = remove_repeated_sequences(item["nomalized_predictions"])

        if not isinstance(reference, str):
            skipped += 1
            continue

        if not isinstance(prediction, str):
            skipped += 1
            continue

        reference = reference.strip()
        prediction = prediction.strip()

        # The WER metric cannot meaningfully evaluate an empty reference
        if not reference:
            print(
                f"Skipping object in {file_path.name}: "
                "normalized_reference is empty"
            )
            skipped += 1
            continue

        recalculated_wer = wer_metric.compute(
            predictions=[prediction],
            references=[reference],
        )

        # Overwrite the previous WER
        item["wer"] = float(recalculated_wer)

        if speaker == "L":
            file_teacher_wers.append(recalculated_wer)
            all_teacher_wers.append(recalculated_wer)
        else:
            file_student_wers.append(recalculated_wer)
            all_student_wers.append(recalculated_wer)

    # Save recalculated WER values back to the same JSON file
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    file_result = {
        "index": file_index,
        "file_name": file_path.name,
        "teacher": {
            "speaker": "L",
            "count": len(file_teacher_wers),
            "average_wer": average_or_none(
                file_teacher_wers
            ),
        },
        "students": {
            "speaker": "all except L",
            "count": len(file_student_wers),
            "average_wer": average_or_none(
                file_student_wers
            ),
        },
        "skipped_objects": skipped,
    }

    per_file_results.append(file_result)

    print(
        f"  Teacher: count={len(file_teacher_wers)}, "
        f"average={average_or_none(file_teacher_wers)}"
    )
    print(
        f"  Students: count={len(file_student_wers)}, "
        f"average={average_or_none(file_student_wers)}"
    )


summary = {
    "model": "openai_whisper-large-v3-turbo",
    "start_index": start_index,
    "reference_key": "normalized_reference",
    "prediction_key": "nomalized_predictions_no_repe",
    "files_processed": len(json_files),
    "overall": {
        "teacher": {
            "speaker": "L",
            "count": len(all_teacher_wers),
            "average_wer": average_or_none(
                all_teacher_wers
            ),
        },
        "students": {
            "speaker": "all except L",
            "count": len(all_student_wers),
            "average_wer": average_or_none(
                all_student_wers
            ),
        },
    },
    "per_file": per_file_results,
}

with summary_file.open("w", encoding="utf-8") as file:
    json.dump(summary, file, ensure_ascii=False, indent=2)

print("\nFinished")
print("Files processed:", len(json_files))
print(
    "Overall teacher average WER:",
    summary["overall"]["teacher"]["average_wer"],
)
print(
    "Overall student average WER:",
    summary["overall"]["students"]["average_wer"],
)
print("Summary saved to:", summary_file)