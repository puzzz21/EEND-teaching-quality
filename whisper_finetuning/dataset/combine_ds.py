import argparse
import json
import os

TALIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "talis")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resolve TALIS chunk audio paths into a training jsonl."
    )
    parser.add_argument("--input-path", default="train.jsonl", help="Input jsonl with 'chunk', 'file_index', 'transcript' fields.")
    parser.add_argument("--output-path", default="train_talis.jsonl", help="Where to write the resolved training jsonl.")
    return parser.parse_args()


def get_training_data(input_path, output_path):
    data = []

    with open(input_path, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line)
            audio_file = os.path.join(TALIS_DIR, "chunked", f"chunks_{record['chunk']}", f"{record['file_index']}.wav")
            data.append({
                "file_audio": audio_file,
                "transcript": record["transcript"]
            })

    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')


if __name__ == "__main__":
    args = parse_args()
    get_training_data(args.input_path, args.output_path)
