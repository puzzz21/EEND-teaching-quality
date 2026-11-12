import argparse
import json
import os
import re
import sys
import time

import librosa
import spacy
import torch
from evaluate import load as load_metric
from transformers import AutoProcessor, Wav2Vec2ForCTC

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import utils.eval_og as eval_og
from utils.maths_normalizer import NormalizeGermanMathExpression
from utils.whisper_normalizer import BasicTextNormalizer

nlp = spacy.load("de_core_news_lg")

wer_metric = load_metric("wer")

models = [
    "facebook/mms-1b-all"
]

device = "cuda"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MMS models on the GTI teacher-focus dataset.")
    parser.add_argument("--translate-dir", default="translate", help="Folder of '<num>.json' reference transcripts.")
    parser.add_argument("--chunk-dir", default="chunked", help="Folder of 'chunks_<num>' audio-chunk subfolders.")
    parser.add_argument("--output-dir", default="teacher_focus_result_mms", help="Folder to write per-model result JSON to.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    for model_name in models:
        evaluate_model(model_name, args.translate_dir, args.chunk_dir, args.output_dir)


def build_repetition_stats(text: str) -> dict:
    stats = eval_og.compute_repetition_ratio([text])
    stats["rep_w"] = eval_og.calculate_rep_w([text])
    stats["rep_r"] = eval_og.calculate_rep_r([text])
    return stats


def evaluate_model(model_name: str, translate_dir: str, chunk_dir: str, output_dir: str):
    processor = AutoProcessor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)

    processor.tokenizer.set_target_lang("deu")
    model.load_adapter("deu")

    for index in [i for i in range(1, 139) if i not in [52, 53, 54]]:
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
            try:
                start_time = time.time()
                audio_array, sr = librosa.load(os.path.join(chunk_folder, f"{ind}.wav"))  # resamples automatically

                inputs = processor(audio_array, sampling_rate=sr, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs).logits

                ids = torch.argmax(outputs, dim=-1)[0]
                transcription = processor.decode(ids)

                elapsed_time = time.time() - start_time

                result = {"prediction": remove_names(transcription), "elapsed_time": elapsed_time}

            except ValueError as e:
                print(f"Value error: {e}")
                print("Please check if the file is a valid audio file and not corrupted.")
                result = {"prediction": "", "elapsed_time": 0}

            prediction = result["prediction"]
            pred = math_normalizer(prediction)

            ref = data[ind]["text"]
            ref_text = [entry for entry in ref.split() if entry not in speakers]

            reference = " ".join(ref_text)
            reference = math_normalizer(reference)

            norm_pred_text = normalizer(pred)
            norm_ref_text = normalizer(reference)

            pred_words = pred.split()
            ref_words = reference.split()

            if len(pred_words) > 0 and len(ref_words) > 0:
                wer = wer_metric.compute(
                        predictions=[norm_pred_text],
                        references=[norm_ref_text]
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
            norm_pred = build_repetition_stats(norm_pred_text)
            norm_ref = build_repetition_stats(norm_ref_text)
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
                    "nomalized_predictions": norm_pred_text,
                    "unnormalized_predictions": prediction,
                    "normalized_reference": norm_ref_text,
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
    normalizer = BasicTextNormalizer(True)
    math_normalizer = NormalizeGermanMathExpression()
    main()
