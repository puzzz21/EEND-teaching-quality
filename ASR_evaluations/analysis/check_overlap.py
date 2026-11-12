import argparse
import json
import os
from pathlib import Path

import torchaudio
from huggingface_hub import hf_hub_download
from pyannote.audio import Model, Pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect overlapped speech duration over folders of WAV chunks."
    )
    parser.add_argument("--input-folder", type=Path, required=True, help="Folder containing per-lesson subfolders of WAV chunks.")
    parser.add_argument("--output-folder", type=Path, required=True, help="Folder to write per-lesson overlap JSON results to.")
    parser.add_argument("--max-chunk-id", type=int, default=60, help="Skip chunk folders with an index beyond this (0-based).")
    return parser.parse_args()


args = parse_args()

token = os.getenv("HF_TOKEN")
if not token:
    raise RuntimeError("Missing HF_TOKEN. Run: export HF_TOKEN='your_token_here'")

model = Model.from_pretrained('pyannote/segmentation', use_auth_token=token)

config_path = hf_hub_download(
                repo_id="pyannote/overlapped-speech-detection",
                filename="config.yaml",
                token=token,
                force_download=True,
            )
pipeline = Pipeline.from_pretrained(
    config_path,
    use_auth_token=token,
)

folder = args.input_folder
args.output_folder.mkdir(parents=True, exist_ok=True)

chunk_folders = sorted(path for path in folder.iterdir() if path.is_dir())

for chunk_id, chunk_folder in enumerate(chunk_folders):
    if chunk_id > args.max_chunk_id:
        continue
    json_file = args.output_folder / f"{chunk_id + 1}.json"

    obj = []
    total_overlap_duration = 0.0

    wav_files = sorted(
        path
        for path in chunk_folder.iterdir()
        if path.is_file() and path.suffix.lower() == ".wav"
    )

    for file_id, wav_file in enumerate(wav_files):
        print(f"Processing: {wav_file}")

        try:
            waveform, sample_rate = torchaudio.load(str(wav_file))
        except Exception as error:
            print(f"Skipping unreadable audio: {wav_file}")
            print(f"Reason: {error}")
            continue

        if waveform.numel() == 0 or waveform.shape[-1] == 0:
            print(f"Empty audio: {wav_file}")
            overlap_duration = 0

        else:

            output = pipeline(str(wav_file))
            overlap_timeline = output.get_timeline().support()
            overlap_duration = overlap_timeline.duration()

            total_overlap_duration += overlap_duration

        obj.append(
            {
                "id": file_id,
                "file": wav_file.name,
                "overlap_duration": round(overlap_duration, 3),
            }
        )

        print(f"File overlap: {overlap_duration:.2f} seconds")

    with json_file.open("w", encoding="utf-8") as file:
        json.dump(obj, file, indent=2)

    print(f"Total overlap: {total_overlap_duration:.2f} seconds")
    print(f"Saved results to: {json_file}")
