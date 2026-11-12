import argparse
import json
import os
from pathlib import Path
from statistics import mean, median

import torchaudio
from pyannote.audio import Inference, Model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute per-chunk SNR/C50 stats over folders of WAV chunks."
    )
    parser.add_argument("--input-folder", type=Path, required=True, help="Folder containing per-lesson subfolders of WAV chunks.")
    parser.add_argument("--output-folder", type=Path, required=True, help="Folder to write per-lesson SNR JSON results to.")
    parser.add_argument("--max-chunk-id", type=int, default=117, help="Skip chunk folders with an index beyond this (0-based).")
    return parser.parse_args()


args = parse_args()

token = os.getenv("HF_TOKEN")
if not token:
    raise RuntimeError("Missing HF_TOKEN. Run: export HF_TOKEN='your_token_here'")

model = Model.from_pretrained(
    "pyannote/brouhaha",
    use_auth_token=token,
)

inference = Inference(model)

input_folder = args.input_folder
output_folder = args.output_folder
output_folder.mkdir(parents=True, exist_ok=True)

chunk_folders = sorted(
    path for path in input_folder.iterdir() if path.is_dir()
)

for chunk_id, chunk_folder in enumerate(chunk_folders):
    if chunk_id > args.max_chunk_id:
        continue
    json_file = output_folder / f"{chunk_id + 1}.json"
    results = []

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
            print(f"Skipping unreadable audio: {error}")

            results.append(
                {
                    "id": file_id,
                    "file": wav_file.name,
                    "status": "decode_error",
                    "error": str(error),
                }
            )
            continue

        if waveform.numel() == 0 or waveform.shape[-1] == 0:
            print(f"Skipping empty audio: {wav_file}")

            results.append(
                {
                    "id": file_id,
                    "file": wav_file.name,
                    "status": "empty",
                }
            )
            continue

        audio_duration = waveform.shape[-1] / sample_rate

        try:
            output = inference(
                {
                    "waveform": waveform,
                    "sample_rate": sample_rate,
                }
            )
        except Exception as error:
            print(f"Inference failed: {error}")

            results.append(
                {
                    "id": file_id,
                    "file": wav_file.name,
                    "status": "inference_error",
                    "error": str(error),
                }
            )
            continue

        snr_values = []
        c50_values = []
        speech_frames = 0
        far_field_frames = 0

        # Duration represented by each prediction step.
        frame_step = float(output.sliding_window.step)

        for _, (vad, snr, c50) in output:
            vad = float(vad)
            snr = float(snr)
            c50 = float(c50)

            # Ignore non-speech frames.
            if vad < 0.5:
                continue

            speech_frames += 1
            snr_values.append(snr)
            c50_values.append(c50)

            # Example far-field/reverberant threshold.
            if c50 < 0.0:
                far_field_frames += 1

        speech_duration = min(
            speech_frames * frame_step,
            audio_duration,
        )

        far_field_duration = min(
            far_field_frames * frame_step,
            speech_duration,
        )

        if snr_values:
            result = {
                "id": file_id,
                "file": wav_file.name,
                "status": "success",
                "audio_duration": round(audio_duration, 3),
                "speech_duration": round(speech_duration, 3),
                "mean_snr": round(mean(snr_values), 3),
                "median_snr": round(median(snr_values), 3),
                "mean_c50": round(mean(c50_values), 3),
                "median_c50": round(median(c50_values), 3),
                "far_field_duration": round(far_field_duration, 3),
                "far_field_percentage_of_speech": round(
                    100 * far_field_duration / speech_duration, 2
                )
                if speech_duration > 0
                else 0.0,
            }
        else:
            result = {
                "id": file_id,
                "file": wav_file.name,
                "status": "no_speech",
                "audio_duration": round(audio_duration, 3),
                "speech_duration": 0.0,
                "mean_snr": None,
                "median_snr": None,
                "mean_c50": None,
                "median_c50": None,
                "far_field_duration": 0.0,
                "far_field_percentage_of_speech": 0.0,
            }

        results.append(result)
        print(result)

    with json_file.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"Saved results to: {json_file}")