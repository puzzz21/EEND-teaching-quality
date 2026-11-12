#!/usr/bin/env python3
"""Conservative fine-tuning of Whisper Large V3 Turbo for German classrooms.

Expected JSONL fields:
    file_audio: path to an audio file
    transcript: matching reference text

Recommended optional fields:
    speaker: "L" for teacher; every other value is treated as student
    recording_id: lesson/recording identifier used for leakage-safe splitting

Install:
    pip install torch transformers datasets accelerate evaluate jiwer soundfile librosa
"""

from __future__ import annotations

import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import datasets
import whisper_fientuning.evaluate.evaluate as evaluate
import librosa
import numpy as np
import soundfile as sf
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer


# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
JSONL_PATH = str(SCRIPT_DIR / "data" / "train_data.jsonl")
OUTPUT_DIR = str(SCRIPT_DIR / "whisper-turbo-german-classroom-improved")
HF_CACHE = os.environ.get("HF_HOME", str(Path.home() / "hf_cache"))

# Model/data configuration
MODEL_ID = "openai/whisper-large-v3-turbo"
LANGUAGE = "german"
TASK = "transcribe"
SAMPLING_RATE = 16_000
MIN_AUDIO_SECS = 0.5
MAX_AUDIO_SECS = 30.0
MAX_LABEL_TOKENS = 448
VALIDATION_FRACTION = 0.10
SEED = 42

# Training configuration chosen to reduce catastrophic forgetting
BATCH_SIZE = 4
GRAD_ACCUM = 4                 # Effective batch size: 16 per process
LEARNING_RATE = 3e-6
WARMUP_RATIO = 0.05
MAX_STEPS = 2500
EVAL_STEPS = 200
SAVE_STEPS = 200
FREEZE_ENCODER = True
DATALOADER_WORKERS = 4

# Balance teacher/student duration in the training data. This requires a
# `speaker` field. Teacher is speaker L; everything else is a student.
BALANCE_TRAIN_BY_SPEAKER_DURATION = True

datasets.config.HF_DATASETS_CACHE = HF_CACHE
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def speaker_group(record: dict) -> str | None:
    """Return teacher/student, or None when no speaker metadata is present."""
    if "speaker" not in record or record["speaker"] is None:
        return None
    return "teacher" if str(record["speaker"]).strip().upper() == "L" else "student"


def recording_group(record: dict) -> str | None:
    """Find a recording-level grouping key without using utterance identity."""
    for key in ("recording_id", "lesson_id", "source_file", "file_number"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()

    # This works when each lesson/recording has its own audio directory.
    parent = Path(record["file_audio"]).parent.name
    return parent or None


def audio_duration(path: str) -> float:
    """Read audio duration from its header without decoding the full waveform."""
    info = sf.info(path)
    return float(info.frames) / float(info.samplerate)


def load_and_validate_jsonl(path: str) -> list[dict]:
    """Load samples and remove audio/text pairs that would be mismatched."""
    records = []
    rejection_counts = defaultdict(int)

    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                rejection_counts["invalid_json"] += 1
                continue

            audio_path = str(record.get("file_audio", "")).strip()
            transcript = str(record.get("transcript", "")).strip()

            if not audio_path or not transcript:
                rejection_counts["missing_audio_or_text"] += 1
                continue
            if not Path(audio_path).is_file():
                rejection_counts["missing_audio_file"] += 1
                continue

            try:
                duration = audio_duration(audio_path)
            except Exception as error:
                print(f"Unreadable audio on JSONL line {line_number}: {error}")
                rejection_counts["unreadable_audio"] += 1
                continue

            # Never truncate audio while keeping the full reference. Long
            # examples must be segmented with matching transcript boundaries.
            if duration < MIN_AUDIO_SECS:
                rejection_counts["too_short"] += 1
                continue
            if duration > MAX_AUDIO_SECS:
                rejection_counts["too_long"] += 1
                continue

            record = dict(record)
            record["file_audio"] = audio_path
            record["transcript"] = transcript
            record["duration"] = duration
            record["speaker_group"] = speaker_group(record)
            record["recording_group"] = recording_group(record)
            records.append(record)

    print(f"Accepted {len(records)} samples from {path}")
    print("Rejected samples:", dict(rejection_counts))
    return records


def split_by_recording(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split complete recordings when possible; otherwise use seeded shuffle."""
    groups = defaultdict(list)
    missing_group = False

    for record in records:
        group = record.get("recording_group")
        if group is None:
            missing_group = True
            break
        groups[group].append(record)

    # A single shared parent directory is not a meaningful recording split.
    if not missing_group and len(groups) >= 5:
        group_names = sorted(groups)
        rng = random.Random(SEED)
        rng.shuffle(group_names)
        number_eval_groups = max(1, round(len(group_names) * VALIDATION_FRACTION))
        eval_groups = set(group_names[:number_eval_groups])

        train_records = []
        eval_records = []
        for group, group_records in groups.items():
            target = eval_records if group in eval_groups else train_records
            target.extend(group_records)

        print(
            f"Recording-level split: {len(groups) - len(eval_groups)} train "
            f"recordings, {len(eval_groups)} validation recordings"
        )
        return train_records, eval_records

    print(
        "WARNING: no usable recording_id grouping was found. Falling back to "
        "a seeded utterance split. Add recording_id to the JSONL to prevent "
        "recording leakage."
    )
    shuffled = list(records)
    random.Random(SEED).shuffle(shuffled)
    eval_size = max(1, round(len(shuffled) * VALIDATION_FRACTION))
    return shuffled[eval_size:], shuffled[:eval_size]


def report_speaker_distribution(name: str, records: list[dict]) -> None:
    stats = defaultdict(lambda: {"samples": 0, "seconds": 0.0, "words": 0})
    for record in records:
        group = record.get("speaker_group") or "unknown"
        stats[group]["samples"] += 1
        stats[group]["seconds"] += float(record["duration"])
        stats[group]["words"] += len(record["transcript"].split())

    print(f"{name} distribution:")
    for group, values in sorted(stats.items()):
        print(
            f"  {group}: {values['samples']} samples, "
            f"{values['seconds'] / 3600:.2f} hours, {values['words']} words"
        )


def balance_by_speaker_duration(records: list[dict]) -> list[dict]:
    """Oversample the smaller speaker group until durations are comparable."""
    grouped = {"teacher": [], "student": []}
    for record in records:
        group = record.get("speaker_group")
        if group in grouped:
            grouped[group].append(record)

    if not grouped["teacher"] or not grouped["student"]:
        print("Speaker balancing skipped: both teacher and student labels are required")
        return records

    durations = {
        group: sum(float(record["duration"]) for record in group_records)
        for group, group_records in grouped.items()
    }
    target_duration = max(durations.values())
    rng = random.Random(SEED)
    balanced = []

    for group, group_records in grouped.items():
        balanced.extend(group_records)
        added_duration = durations[group]
        candidates = list(group_records)
        rng.shuffle(candidates)
        candidate_index = 0

        while added_duration < target_duration:
            record = candidates[candidate_index % len(candidates)]
            balanced.append(record)
            added_duration += float(record["duration"])
            candidate_index += 1

    # Preserve any records with unknown speaker labels once, without using
    # them to define the teacher/student balancing target.
    balanced.extend(record for record in records if record.get("speaker_group") is None)
    rng.shuffle(balanced)
    print(f"Speaker-duration balancing: {len(records)} -> {len(balanced)} samples")
    return balanced


def records_to_dataset(records: list[dict]) -> Dataset:
    return Dataset.from_dict(
        {
            "audio": [record["file_audio"] for record in records],
            "transcript": [record["transcript"] for record in records],
            "duration": [float(record["duration"]) for record in records],
        }
    )


def build_datasets(records: list[dict]) -> DatasetDict:
    train_records, eval_records = split_by_recording(records)
    report_speaker_distribution("Train before balancing", train_records)
    report_speaker_distribution("Validation", eval_records)

    if BALANCE_TRAIN_BY_SPEAKER_DURATION:
        train_records = balance_by_speaker_duration(train_records)

    teacher_eval = [r for r in eval_records if r.get("speaker_group") == "teacher"]
    student_eval = [r for r in eval_records if r.get("speaker_group") == "student"]

    # A duration-balanced validation set supplies the checkpoint-selection
    # metric. Original teacher/student sets remain available for diagnostics.
    balanced_eval = balance_by_speaker_duration(eval_records)

    result = {
        "train": records_to_dataset(train_records),
        "balanced": records_to_dataset(balanced_eval),
    }
    if teacher_eval:
        result["teacher"] = records_to_dataset(teacher_eval)
    if student_eval:
        result["student"] = records_to_dataset(student_eval)

    print("Dataset sizes:", {key: len(value) for key, value in result.items()})
    return DatasetDict(result)


def tokenize_labels(batch: dict, tokenizer: WhisperTokenizer) -> dict:
    batch["labels"] = tokenizer(
        batch["transcript"],
        truncation=False,
    ).input_ids
    return batch


def labels_fit(example: dict) -> bool:
    return len(example["labels"]) <= MAX_LABEL_TOKENS


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int
    sampling_rate: int = SAMPLING_RATE

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        audio_arrays = []
        for feature in features:
            array, _ = librosa.load(
                feature["audio"],
                sr=self.sampling_rate,
                mono=True,
            )
            audio_arrays.append(array)

        batch = self.processor.feature_extractor(
            audio_arrays,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1),
            -100,
        )

        if (labels[:, 0] == self.decoder_start_token_id).all().item():
            labels = labels[:, 1:]

        result = {
            "input_features": batch.input_features,
            "labels": labels,
        }
        if "attention_mask" in batch:
            result["attention_mask"] = batch.attention_mask
        return result


def build_compute_metrics(tokenizer: WhisperTokenizer):
    wer_metric = evaluate.load("wer")
    normalizer = BasicTextNormalizer()

    def compute_metrics(prediction):
        prediction_ids = prediction.predictions
        label_ids = prediction.label_ids.copy()
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        prediction_text = tokenizer.batch_decode(
            prediction_ids,
            skip_special_tokens=True,
        )
        reference_text = tokenizer.batch_decode(
            label_ids,
            skip_special_tokens=True,
        )

        normalized_predictions = [normalizer(text).strip() for text in prediction_text]
        normalized_references = [normalizer(text).strip() for text in reference_text]

        # Empty references cannot contribute a meaningful WER denominator.
        valid_pairs = [
            (predicted, reference)
            for predicted, reference in zip(
                normalized_predictions,
                normalized_references,
            )
            if reference
        ]
        if not valid_pairs:
            return {"wer": math.inf}

        predictions, references = zip(*valid_pairs)
        wer = wer_metric.compute(
            predictions=list(predictions),
            references=list(references),
        )
        return {"wer": float(wer)}

    return compute_metrics


def main() -> None:
    print("=" * 70)
    print(f"Model: {MODEL_ID}")
    print(f"Language: {LANGUAGE}; task: {TASK}")
    print(f"CUDA devices: {torch.cuda.device_count()}")
    print("=" * 70)

    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        MODEL_ID,
        return_attention_mask=True,
    )
    tokenizer = WhisperTokenizer.from_pretrained(
        MODEL_ID,
        language=LANGUAGE,
        task=TASK,
    )
    processor = WhisperProcessor.from_pretrained(
        MODEL_ID,
        language=LANGUAGE,
        task=TASK,
    )

    records = load_and_validate_jsonl(JSONL_PATH)
    dataset = build_datasets(records)

    dataset = dataset.map(
        lambda batch: tokenize_labels(batch, tokenizer),
        batched=True,
        batch_size=256,
        num_proc=4,
        desc="Tokenizing labels",
    )
    dataset = dataset.filter(
        labels_fit,
        num_proc=4,
        desc="Removing labels longer than Whisper's decoder limit",
    )

    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    model.config.use_cache = False
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None
    model.generation_config.do_sample = False
    model.generation_config.num_beams = 1

    if FREEZE_ENCODER:
        model.freeze_encoder()
        print("Whisper encoder frozen; decoder remains trainable")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    bf16_enabled = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16_enabled = torch.cuda.is_available() and not bf16_enabled

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        remove_unused_columns=False,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="linear",
        max_steps=MAX_STEPS,
        gradient_checkpointing=True,
        bf16=bf16_enabled,
        fp16=fp16_enabled,
        tf32=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_balanced_wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        generation_num_beams=1,
        logging_steps=25,
        report_to=["tensorboard"],
        push_to_hub=False,
        dataloader_num_workers=DATALOADER_WORKERS,
        dataloader_persistent_workers=DATALOADER_WORKERS > 0,
        seed=SEED,
        data_seed=SEED,
    )

    eval_datasets = {
        key: value
        for key, value in dataset.items()
        if key != "train"
    }

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=eval_datasets,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
        processing_class=processor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("Starting training...")
    trainer.train()

    best_dir = os.path.join(OUTPUT_DIR, "best_model")
    trainer.save_model(best_dir)
    processor.save_pretrained(best_dir)

    print("Best checkpoint:", trainer.state.best_model_checkpoint)
    print("Best validation metric:", trainer.state.best_metric)
    print("Saved best model to:", best_dir)


if __name__ == "__main__":
    main()
