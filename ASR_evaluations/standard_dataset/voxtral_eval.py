from transformers import VoxtralForConditionalGeneration, AutoProcessor
import torch
from datasets import load_dataset
import tempfile
import soundfile as sf
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from utils.whisper_normalizer import BasicTextNormalizer
import argparse
import json
import warnings
from evaluate import load as load_metric
from tqdm import tqdm
import time
import numpy as np

warnings.filterwarnings("ignore")

dataset_definition = {
    "uhhlt/Tuda-De": {
        "dataset_language": "",
        "reference": "sentence"
    },
    "mozilla-foundation/common_voice_9_0": {
        "dataset_language": "de",
        "reference": "sentence",
        "use_auth_token": os.getenv("HF_TOKEN")
    },
    "facebook/multilingual_librispeech": {
        "dataset_language": "german",
        "reference": "transcript"
    },
    "facebook/voxpopuli": {
        "dataset_language": "de",
        "reference": "normalized_text"
    },
    "google/fleurs": {
        "dataset_language": "de_de",
        "reference": "transcription"
    }
}

models = [
    # "mistralai/Voxtral-Mini-3B-2507",
    "mistralai/Voxtral-Small-24B-2507"
]

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Metrics
wer_metric = load_metric("wer")

# Tracking lists
predictions = []
references = []
latency = []
durations = []


def main():
    for repo_id in models:
        print("*" * 80)
        print(f"Evaluating model: {repo_id}")
        evaluate(repo_id)

    print("\nAll models evaluated successfully!")


def evaluate(repo_id: str):
    # Load processor + model
    processor = AutoProcessor.from_pretrained(repo_id)
    model = VoxtralForConditionalGeneration.from_pretrained(
        repo_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # Load dataset
    lang = dataset_definition[args.dataset]["dataset_language"]
    data = load_dataset(args.dataset, lang, split="test")

    # Filter out entries with empty or None audio
    data = data.filter(lambda x: x["audio"] is not None and len(x["audio"]["array"]) > 0)
    
    # tqdm progress bar
    pbar = tqdm(
        data,
        desc=f"Evaluating {repo_id}",
        unit="sample",
        dynamic_ncols=True,  # adjusts width to terminal
        smoothing=0.1        # smoother ETA updates
    )

    for d in pbar:
        audio_input_dict = d["audio"]

        # Skip any 0-length audio just in case
        if len(audio_input_dict["array"]) == 0:
            continue

        # Convert to WAV PCM_16 for safety
        temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_file_path = temp_audio_file.name
    
        try:
            sf.write(
                temp_file_path,
                audio_input_dict["array"],
                audio_input_dict["sampling_rate"],
                format="WAV",
                subtype="PCM_16"
            )

            start_time = time.time()

            inputs = processor.apply_transcription_request(
                language="de",
                audio=temp_file_path,
                model_id=repo_id,
            )

            input_ids = inputs.pop("input_ids").to(device)
            inputs = {k: v.to(device, dtype=torch.bfloat16) for k, v in inputs.items()}
            inputs["input_ids"] = input_ids

            outputs = model.generate(**inputs, max_new_tokens=500)
            decoded_outputs = processor.batch_decode(
                outputs[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

            elapsed_time = time.time() - start_time

            predictions.append(normalizer(decoded_outputs[0]))
            references.append(normalizer(d[dataset_definition[args.dataset]["reference"]]))
            latency.append(elapsed_time)
            durations.append(len(audio_input_dict["array"]) / audio_input_dict["sampling_rate"])
            
            avg_time = sum(latency) / len(latency)
            pbar.set_postfix({
                "last_sample": f"{elapsed_time:.2f}s",
                "avg_sample": f"{avg_time:.2f}s",
                "WER_so_far": f"{(wer_metric.compute(predictions=predictions, references=references)):.3f}"
            })

        finally:
            os.remove(temp_file_path)

    wer = wer_metric.compute(predictions=predictions, references=references)
    total_duration = sum(durations)
    total_latency = sum(latency)
    rtf = total_latency / total_duration

    print("\n🤖 Model Evaluation Results:")
    print(f"✅ Model: {repo_id}, Dataset: {args.dataset}, WER: {wer:.3f}, latency: {total_latency:.3f}, rtf: {rtf:.3f}, total_duration: {total_duration:.3f} seconds")

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"output_{repo_id.replace('/', '_')}_{args.dataset.replace('/', '_')}.json")

    with open(output_file, "a") as f:
        f.write(json.dumps({
            "model": repo_id,
            "dataset": args.dataset,
            "wer": wer,
            "rtf": rtf,
            "total_duration": total_duration,
            "latency": total_latency
        }) + "\n")

    print(f"\n✅ Done! WER: {wer:.3f}")
    print(f"\n🎉 Output saved to: {output_file}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="</> Voxtral ASR Benchmark Script")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "uhhlt/Tuda-De",
            "facebook/multilingual_librispeech",
            "facebook/voxpopuli",
            "google/fleurs"
        ],
        default="uhhlt/Tuda-De",
        help="🤗 Hugging Face dataset name"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Folder to write per-model/dataset result JSON lines to.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    normalizer = BasicTextNormalizer(remove_diacritics=True)
    main()
