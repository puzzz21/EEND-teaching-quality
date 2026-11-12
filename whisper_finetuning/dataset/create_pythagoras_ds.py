import json
import os
import re
import sys
from pathlib import Path

from pydub import AudioSegment

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from utils.maths_normalizer import NormalizeGermanMathExpression

math_normalizer = NormalizeGermanMathExpression()

PYTHAGORAS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pythagoras")


def time_to_ms(t):
    """Convert HH:MM:SS or MM:SS to milliseconds"""
    parts = list(map(float, t.split(":")))

    if len(parts) == 3:
        h, m, s = parts
    else:
        h = 0
        m, s = parts

    return int((h * 3600 + m * 60 + s) * 1000)


def chunk_audio(json_path, audio_path, output_base):
    # Load JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Load audio
    audio = AudioSegment.from_wav(audio_path)
    audio_length = len(audio)  # in ms

    for i, entry in enumerate(data):
        start = time_to_ms(entry["time"])

        if i < len(data) - 1:
            end = time_to_ms(data[i + 1]["time"])
        else:
            end = audio_length  # last chunk

        chunk = audio[start:end]

        out_path = os.path.join(output_base, f"{i}.wav")
        chunk.export(out_path, format="wav")

    print(f"Done. Saved {len(data)} chunks to {output_base}")


def clean_transcript(text):
    text = re.sub(r'(\/\s*){1,}', ' ', text)

    # Remove speaker tokens like s09, s1, l, k (case insensitive)
    text = re.sub(r'\b(?:s\d+|[lk])\b', '', text, flags=re.IGNORECASE)

    # Remove things like (S09: ...), (L: ...), etc.
    text = re.sub(r'\(\s*[a-z]\d*\s*:.*?\)', '', text, flags=re.IGNORECASE)

    # Remove empty or short parentheses (e.g. (), (word))
    text = re.sub(r'\(\s*\w*\s*\)', '', text)

    # Remove trailing dashes (broken words)
    text = re.sub(r'\b\w+-\s*', '', text)

    # Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def save_jsonl(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    folder = Path(os.path.join(PYTHAGORAS_DIR, "json_transcripts"))
    result = []

    for file in folder.iterdir():
        wav_file = file.name.replace("-Transkr-Video.json", ".wav")

        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for ind, entry in enumerate(data):
            clean_text = math_normalizer(clean_transcript(entry["text"]))
            result.append({
                "file_audio": wav_file,
                "chunk": ind,
                "transcript": clean_text,
            })

    save_jsonl(result, os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "train_pythagoras.jsonl"))


main()
