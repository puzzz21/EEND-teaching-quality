from transformers import pipeline
from evaluate import load as load_metric
from itertools import islice
import time
import json
from datasets import load_dataset, Dataset
import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from utils.whisper_normalizer import BasicTextNormalizer

models = [
    "openai/whisper-tiny",
    "openai/whisper-base",
    "openai/whisper-small",
    "openai/whisper-medium",
    "openai/whisper-large",
    "openai/whisper-large-v2",
    "openai/whisper-large-v3",
    "openai/whisper-large-v3-turbo",
    "primeline/whisper-large-v3-turbo-german",
    "primeline/distil-whisper-large-v3-german",
    "primeline/whisper-large-v3-german",
    "primeline/whisper-tiny-german-1224",
    "primeline/whisper-tiny-german"
]

dataset_definition = {
    "uhhlt/Tuda-De": {
        "dataset_language": "",
        "reference" : "sentence"
    },
    "mozilla-foundation/common_voice_9_0": {
        "dataset_language": "de",
        "reference" : "sentence",
        "use_auth_token" : os.getenv("HF_TOKEN")
    },
    "facebook/multilingual_librispeech": {
        "dataset_language": "german",
        "reference" : "transcript"
    },
    "facebook/voxpopuli": {
        "dataset_language": "de",
        "reference" : "normalized_text"
    },
    "google/fleurs": {
        "dataset_language": "de_de",
        "reference" : "transcription"
    }
}

wer_metric = load_metric("wer")

def evaluation_pipeline(model: str, data: Dataset) -> dict:

    whisper_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        device_map="auto",
        ignore_warning=True
    )
    
    # Transcription function
    def transcribe(batch):
        audio_input = {
            "array": batch["audio"]["array"],
            "sampling_rate": batch["audio"]["sampling_rate"]
        }
        try:
            result = whisper_pipeline(audio_input, chunk_length_s=30)
        except StopIteration as e:
            # Handle end of iteration gracefully
            result = {"text": ""} 

        normalized_result = normalizer.__call__(result["text"])
        return {"prediction": normalized_result}

    # Start timer and process
    start_time = time.time()
    ds = data.map(transcribe)
    end_time = time.time()
    elapsed_time = end_time - start_time

    ds = ds.filter(lambda x: x["prediction"] is not None and str(x["prediction"]).strip() != "")

    total_duration = sum(len(audio["array"]) / audio["sampling_rate"] for audio in data["audio"])

    # Evaluate WER
    references = [normalizer.__call__(ref) for ref in ds[dataset_definition[args.dataset]["reference"]]]
    predictions = ds["prediction"]
    wer = wer_metric.compute(predictions=predictions, references=references)
    return { "wer": wer, "latency": elapsed_time, "total_duration": total_duration }

def write_result(model: str, dataset_name: str, wer: float, latency: float, duration: float, output_dir: str) -> None:
    rtf = latency / duration
    print("\n🤖 Model Evaluation Results:")
    print(f"✅ Model: {model}, Dataset: {dataset_name}, WER: {wer:.3f}, latency: {latency}, rtf: {rtf:.3f}, total_duration: {duration:.3f} seconds")
    result = {
        "model": model,
        "dataset": dataset_name,
        "wer": wer,
        "rtf": rtf,
        "total_duration": duration,
        "latency": latency
    }

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"output_{model.replace('/', '_')}_{dataset_name.replace('/', '_')}.json")

    # Append result to the file as a JSON lines file
    with open(output_file, "a") as f:
        f.write(json.dumps(result))
        f.write("\n")
    
    print(f"\n✅ Done! WER: {wer:.3f}")
    print(f"\n🎉 Output saved to: {output_file}")

def main() -> None:
    print(f"🤫 Using model: {args.model}")
    print(f"📚 Using dataset: {args.dataset}")

    if "use_auth_token" in dataset_definition[args.dataset]:
        auth_token = dataset_definition[args.dataset]["use_auth_token"]
        if not auth_token:
            raise RuntimeError("Missing HF_TOKEN. Run: export HF_TOKEN='your_token_here'")
        data = load_dataset(args.dataset, dataset_definition[args.dataset]["dataset_language"], split="test", use_auth_token=auth_token)
    else:
        data = load_dataset(args.dataset, dataset_definition[args.dataset]["dataset_language"], split="test")

    data = data.filter(
        lambda x: (
            x.get("audio") is not None
            and len(x["audio"]["array"]) > 0
            and x.get(dataset_definition[args.dataset]["reference"]) is not None
            and str(x.get(dataset_definition[args.dataset]["reference"])).strip() != ""
        )
    )
    
    print(f"📚 Loaded dataset: {args.dataset} with {len(data)} samples")


    if args.model == "all":
        
        print("🤖 Evaluating all models...")
        for m in models:
            print(f"\n🔍 Evaluating model: {m}")
            result = evaluation_pipeline(m, data)
            write_result(m, args.dataset, result["wer"], result["latency"], result["total_duration"], args.output_dir)
    else:

        print(f"\n🔍 Evaluating model: {args.model}")
        result = evaluation_pipeline(args.model, data)
        write_result(args.model, args.dataset, result["wer"], result["latency"], result["total_duration"], args.output_dir)
    
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="</> Whisper ASR Benchmark Script"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices= models + ["all"],
        default="openai/whisper-small",
        help="🤗 Hugging Face model name (e.g., openai/whisper-large, primeline/whisper-large-v3-german) or 'all' for all the models"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
        "uhhlt/Tuda-De",
        "mozilla-foundation/common_voice_9_0",
        "facebook/multilingual_librispeech",
        "facebook/voxpopuli",
        "google/fleurs"
        ],
        default="facebook/multilingual_librispeech",
        help="🤗 Hugging Face dataset name (e.g., mozilla-foundation/common_voice_9_0, uhhlt/Tuda-De, google/fleurs)"
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