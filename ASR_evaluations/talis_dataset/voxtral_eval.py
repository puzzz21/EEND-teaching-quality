import argparse
import json
import os
import re
import sys
import time

import soundfile as sf
import spacy
import torch
from evaluate import load as load_metric
from transformers import AutoProcessor, VoxtralForConditionalGeneration

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import utils.eval_og as eval_og
from utils.maths_normalizer import NormalizeGermanMathExpression

nlp = spacy.load("de_core_news_lg")

wer_metric = load_metric("wer")

models = [
    "mistralai/Voxtral-Mini-3B-2507"
]

device = "cuda"

# Load processor + model
processor = AutoProcessor.from_pretrained("mistralai/Voxtral-Mini-3B-2507")
model = VoxtralForConditionalGeneration.from_pretrained(
        "mistralai/Voxtral-Mini-3B-2507", torch_dtype=torch.bfloat16, device_map="auto"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Voxtral models on the GTI teacher-focus dataset.")
    parser.add_argument("--translate-dir", default="translate", help="Folder of '<num>.json' reference transcripts.")
    parser.add_argument("--chunk-dir", default="chunked", help="Folder of 'chunks_<num>' audio-chunk subfolders.")
    parser.add_argument("--output-dir", default="teacher_focus_result_vox", help="Folder to write per-model result JSON to.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    for model_name in models:
        evaluate_model(model_name, args.translate_dir, args.chunk_dir, args.output_dir)


def transcribe(model_name, file_name):
    try:
        inputs = processor.apply_transcription_request(
                language="de",
                audio=file_name,
                model_id=model_name,
            )

        input_ids = inputs.pop("input_ids").to(device)
        inputs = {k: v.to(device, dtype=torch.bfloat16) for k, v in inputs.items()}
        inputs["input_ids"] = input_ids

        outputs = model.generate(**inputs, max_new_tokens=500)
        decoded_outputs = processor.batch_decode(
                outputs[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
        )
    except StopIteration:
        # Handle end of iteration gracefully
        decoded_outputs = [""]

    result_text = remove_names(decoded_outputs[0])
    return {"prediction": result_text}


def evaluation_pipeline(model_name: str, file_name):
    start_time = time.time()
    result = transcribe(model_name, file_name)
    elapsed_time = time.time() - start_time
    result["elapsed_time"] = elapsed_time
    return result


def build_repetition_stats(text: str) -> dict:
    stats = eval_og.compute_repetition_ratio([text])
    stats["rep_w"] = eval_og.calculate_rep_w([text])
    stats["rep_r"] = eval_og.calculate_rep_r([text])
    return stats


def evaluate_model(model_name: str, translate_dir: str, chunk_dir: str, output_dir: str):
    for index in [i for i in range(42, 139) if i not in [52, 53, 54]]:
        results = []
        if index < 10:
            num = f"00{index}"
        elif index < 100:
            num = f"0{index}"
        else:
            num = f"{index}"

        file_name = model_name.replace("/", "_") + "__" + str(num)

        output_path = os.path.join(output_dir, f"{file_name}.json")
        if os.path.exists(output_path):
            continue

        with open(os.path.join(translate_dir, f"{num}.json"), "r", encoding="utf-8") as f:
            data = json.load(f)

        chunk_folder = os.path.join(chunk_dir, f"chunks_{num}")

        num_files = len([f for f in os.listdir(chunk_folder) if os.path.isfile(os.path.join(chunk_folder, f))])

        speakers = get_speakers(data)

        for ind in range(num_files):
            chunk_path = os.path.join(chunk_folder, f"{ind}.wav")
            audio_array, _ = sf.read(chunk_path)
            if len(audio_array) == 0:
                continue

            try:
                result = evaluation_pipeline(model_name, chunk_path)
            except ValueError as e:
                print(f"Value error: {e}")
                print("Please check if the file is a valid audio file and not corrupted.")
                result = {"prediction": "", "elapsed_time": 0}

            prediction = result["prediction"]
            pred = math_normalizer(prediction)

            ref = data[ind]["text"]
            reference = math_normalizer(ref)

            ref_text = [entry for entry in reference.split() if entry not in speakers]
            reference = " ".join(ref_text)

            pred_words = pred.split()
            ref_words = reference.split()

            if len(pred_words) > 0 and len(ref_words) > 0:
                wer = wer_metric.compute(
                        predictions=[pred],
                        references=[reference]
                )
            else:
                wer = None
                print("WER: N/A (empty prediction or reference)")

            elapsed_time = result["elapsed_time"]

            if data[ind]["time"] is None or data[ind + 1]["time"] is None:
                dur = None
            else:
                start = get_seconds(data[ind]["time"])
                end = get_seconds(data[ind + 1]["time"])
                dur = end - start

            unnorm_pred = build_repetition_stats(prediction)
            norm_pred = build_repetition_stats(pred)
            norm_ref = build_repetition_stats(reference)
            unnorm_ref = build_repetition_stats(ref)

            if data[ind]["speaker"] is None:
                percent_teacher_speaking_value = None
            else:
                percent_teacher_speaking_value = percent_teacher_speaking(data[ind]["text"], data[ind]["speaker"])
            has_math_eq = has_math_equation(data[ind]["text"])

            results.append({
                    "chunk_index": ind,
                    "speaker": data[ind]["speaker"],
                    "time": data[ind]["time"],
                    "wer": wer,
                    "elapsed_time": elapsed_time,
                    "duration": dur,
                    "nomalized_predictions": pred,
                    "unnormalized_predictions": prediction,
                    "normalized_reference": reference,
                    "unnormalized_reference": ref,
                    "unnormalized_prediction_rep": unnorm_pred,
                    "normalized_prediction_rep": norm_pred,
                    "normalized_reference_rep": norm_ref,
                    "unnormalized_reference_rep": unnorm_ref,
                    "percent_teacher_speaking": percent_teacher_speaking_value,
                    "has_math_equation": has_math_eq
                })

            print("chunk_id done: ", ind)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)


def has_math_equation(text: str) -> bool:
    """
    Detects if a text likely contains a mathematical expression or equation.
    Looks for numbers, operators, or math-related symbols.
    """
    math_pattern = re.compile(
        r'(\d+\s*[\+\-\*/^=]\s*\d+|'     # e.g. 3 + 4, 5*2, 10/3
        r'=\s*\d+|'                      # e.g. = 5
        r'[A-Za-z]\s*=\s*\d+|'           # e.g. x = 4
        r'[\+\-\*/^=<>≈]|'               # math operators
        r'\b(halb|viertel|drittel|hoch|geteilt|mal|prozent)\b)',  # German math words
        re.IGNORECASE
    )

    return bool(math_pattern.search(text))


def percent_teacher_speaking(text, speaker):
    og_total_text = len(text.split())
    total_text = 0

    speaker = re.sub(r'[^\w\s]', '', speaker).strip()

    if speaker.lower() == "l":
        cleaned_text, bracketed = remove_and_list_bracketed(text)
        total_text = len(cleaned_text.split())
        for bracket in bracketed:
            if bracket.startswith("(L:"):
                bracket_content = bracket[3:-1].strip()  # Remove "(L:" and ")"
                total_text += len(bracket_content.split())
    else:
        cleaned_text, bracketed = remove_and_list_bracketed(text)
        for bracket in bracketed:
            if bracket.startswith("(L:"):
                bracket_content = bracket[3:-1].strip()  # Remove "(L:" and ")"
                total_text += len(bracket_content.split())

    return (total_text / og_total_text) * 100 if og_total_text > 0 else 0


def remove_and_list_bracketed(text: str):
    """
    Removes bracketed text (e.g. '(L: ... )') and returns:
      - the cleaned text (without brackets)
      - a list of removed bracket contents (including brackets)
    """
    bracketed = re.findall(r'\([^)]*\)', text)
    cleaned = re.sub(r'\([^)]*\)', '', text)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()

    return cleaned, bracketed


def remove_names(sent: str) -> str:
    doc = nlp(sent)
    person_tokens = {token.i for ent in doc.ents if ent.label_ == "PER" for token in ent}
    filtered_tokens = [token.text_with_ws for token in doc if token.i not in person_tokens]
    return "".join(filtered_tokens).strip()


def get_speakers(data):
    speakers = []

    for entry in data:
        if "speaker" in entry and entry["speaker"] is not None and entry["speaker"] != "":
            text = re.sub(r"[?!.]", "", entry["speaker"].lower())
            speakers.append(text)
    return list(set(speakers))


def get_seconds(time_str):
    parts = time_str.split(':')
    parts = [float(p) for p in parts]

    # Handle H:MM:SS or HH:MM:SS format
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    else:
        raise ValueError("Invalid time format")

    total_seconds = hours * 3600 + minutes * 60 + seconds
    return total_seconds


if __name__ == "__main__":
    math_normalizer = NormalizeGermanMathExpression()
    main()
