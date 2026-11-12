import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge ASR result files with their reference transcripts."
    )
    parser.add_argument("--results-dir", type=Path, required=True, help="Folder of raw per-run result JSON files.")
    parser.add_argument("--translate-dir", type=Path, required=True, help="Folder of '<num>.json' reference transcripts.")
    parser.add_argument("--output-path", type=Path, default=Path("result.json"), help="Where to write the merged JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    merged = []

    for file_path in args.results_dir.rglob("*"):
        if not file_path.is_file():
            continue

        with file_path.open("r", encoding="utf-8") as file:
            result_file = json.load(file)

        file_name = file_path.name.split(".")[0].split("_")[-1] + ".json"
        with (args.translate_dir / file_name).open("r", encoding="utf-8") as file:
            data = json.load(file)

        for indx, i in enumerate(result_file):
            id = i["audio_file"].split(".")[0]
            obj = data[int(id)]

            result_file[indx]["reference"] = obj["text"]
            result_file[indx]["speaker"] = obj["speaker"]
            result_file[indx]["time"] = obj["time"]
        merged.append(result_file)

    with args.output_path.open("w", encoding="utf-8") as file:
        json.dump(merged, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
