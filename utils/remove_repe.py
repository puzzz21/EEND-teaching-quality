import argparse
import json
import os

MIN_NGRAM = 1
MAX_NGRAM = 10
MAX_REPEATS = 2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Strip repeated word sequences from transcript JSON files in place."
    )
    parser.add_argument(
        "--base-dir",
        required=True,
        help="Folder of '*.json' transcript files to clean in place.",
    )
    return parser.parse_args()


def remove_repeated_sequences(text, min_ngram=1, max_ngram=10, max_repeats=2):
    words = text.split()
    i = 0
    output = []

    while i < len(words):
        found_repeat = False
        max_possible_ngram = min(max_ngram, (len(words) - i) // 2)

        # IMPORTANT: check smallest phrase first
        for n in range(min_ngram, max_possible_ngram + 1):
            phrase = words[i:i + n]

            repeat_count = 1
            j = i + n

            while j + n <= len(words) and words[j:j + n] == phrase:
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


def main():
    args = parse_args()
    base_dir = args.base_dir

    files = [
        f for f in os.listdir(base_dir)
        if f.endswith(".json") and os.path.isfile(os.path.join(base_dir, f))
    ]

    for filename in files:
        file_path = os.path.join(base_dir, filename)
        result = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            if isinstance(raw, dict) and "data" in raw:
                data = raw["data"]
                keep_dict_structure = True
            elif isinstance(raw, list):
                data = raw
                keep_dict_structure = False
            else:
                raise ValueError("Unsupported JSON structure")

            for d in data:
                cleaned_items = []

                for item in d.get("items", []):
                    text = item.get("text", "")
                    text = text.replace("\n", " ").strip()

                    cleaned_text = remove_repeated_sequences(
                        text,
                        min_ngram=MIN_NGRAM,
                        max_ngram=MAX_NGRAM,
                        max_repeats=MAX_REPEATS
                    )

                    cleaned_items.append({
                        "time": item.get("time"),
                        "text": cleaned_text
                    })

                d["cleaned_items"] = cleaned_items
                result.append(d)

            output_json = {"data": result} if keep_dict_structure else result

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(output_json, f, indent=2, ensure_ascii=False)

            print(f"Processed: {filename}")

        except Exception as e:
            print(f"Failed processing file: {filename}")
            print(f"Reason: {e}")


if __name__ == "__main__":
    main()
