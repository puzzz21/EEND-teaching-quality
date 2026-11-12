# code

Research code for ASR, speaker diarization, and LLM-based evaluation of German classroom lesson recordings (student/teacher transcripts, math content). Five independent projects live side by side here.

## Layout

```
ASR_evaluations/     Benchmark ASR models on classroom + public German datasets
LLM_evaluations/     Use LLMs as judges to rate transcript teaching quality
diarization/         Speaker diarization pipeline + evaluation
utils/               Shared normalization/repetition helpers used across projects
whisper_finetuning/  Fine-tune Whisper on German classroom audio
```

## ASR_evaluations/

Benchmarks ASR models — Whisper (`openai/whisper-*`, `primeline/whisper-*-german`), NVIDIA NeMo (Canary, Parakeet), Meta MMS, Voxtral, and Meta Omnilingual ASR (LLM + CTC variants) — on German classroom recordings, computing WER, then analyzes what drives WER.

- `gti_dataset/` — one eval script per model family, run against the "GTI" classroom dataset (`translate/<num>.json` references + `chunked*/chunks_<num>/*.wav` audio). Strips speaker names via spaCy NER, normalizes German math expressions, computes WER, repetition stats, and teacher-speaking percentage per chunk. `segmented.py` groups results into ~16-minute segments for downstream rating.
- `standard_dataset/` — same models against public benchmarks (Tuda-De, Common Voice, MLS, VoxPopuli, FLEURS German splits), computing WER and RTF.
- `analysis/` — statistical drivers of WER: overlapped-speech detection and SNR/C50 acoustic quality (via pyannote), and an OLS regression (`wer_regression.py`) of WER against these features, exported as a LaTeX table for a paper.

Requires `HF_TOKEN` for gated pyannote models and gated HF datasets.

## LLM_evaluations/

Uses LLMs (Claude, GPT-5, Gemini, Qwen, Olmo/Apertus) as judges to rate classroom transcripts on two rubrics — Nature of Discourse (NOD) and Questioning — on a 1–4 scale, then scores agreement with two human raters via Quadratic Weighted Kappa (QWK).

- `eval_claude.py`, `eval_gpt5.py`, `eval_gemini.py` — hosted-API judges (need `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY` respectively). 3 runs per lesson, resume-safe.
- `eval_qwen.py` — hits a self-hosted OpenAI-compatible endpoint (`REMOTE_API_URL`).
- `eval_olmo_apertus.py` — local HF `transformers` inference (GPU recommended).
- `calculate_qwk_ref.py` — core scoring: human-rater reliability QWK, and per-model QWK across 5 transcript-quality conditions (ground truth vs. Whisper turbo/large/medium/small).
- `calculate_all.py` — same as above plus stddev across runs; prints data for bar-chart plotting.
- `prompts/` — rubric prompt templates (diarized and non-diarized variants).

## diarization/

- `diarization_pipeline.py` — end-to-end pipeline: `pyannote/speaker-diarization-3.1` + local Whisper transcription, aligned by segment overlap, then a heuristic multi-signal scorer (`EnhancedTeacherIdentifier`) classifies the highest-scoring speaker as "Teacher". Requires `HF_TOKEN`.
- `der_eval.py` — evaluates diarization output by collapsing consecutive same-speaker entries into turns and scoring turn-sequence similarity (`difflib.SequenceMatcher`) against a reference. Despite the name, this is a turn-sequence match score, not true pyannote DER.

## utils/

Shared helpers imported across the other projects:

- `whisper_normalizer.py` — Whisper's basic text normalizer, used for WER normalization.
- `maths_normalizer.py` — normalizes spoken German math (number words, fractions, decimals, operators) into symbolic form.
- `eval_og.py` — repetition-degeneration metrics (rep-w, rep-r, n-gram duplicate ratio) for flagging ASR hallucination loops.
- `remove_repe.py` — CLI that strips repeated n-gram sequences from transcript JSON files in place.

## whisper_finetuning/

Fine-tunes `openai/whisper-large-v3-turbo` on German classroom audio.

- `finetune.py` — loads a JSONL dataset (`file_audio`, `transcript`, optional `speaker`/`recording_id`), splits train/eval by recording to avoid leakage, optionally balances teacher/student speaking duration, and trains with HF `Seq2SeqTrainer` (frozen encoder, early stopping on eval WER).
- `dataset/` — builds the training JSONL from the raw TALIS classroom corpus (`create_talis_ds.py`, `combine_ds.py`, `create_pythagoras_ds.py`).
- `evaluate/` — WER evaluation/merge tooling for fine-tuned model outputs. Note: `evaluate/evaluate.py` is a near-duplicate of the top-level `finetune.py` (imported by it for the `evaluate.load("wer")` helper) and is likely a stale earlier version.

## Notes

-  `requirements.txt` currently exists for any project; dependencies include `transformers`, `evaluate`, `datasets`, `nemo_toolkit`, `omnilingual_asr`, `pyannote.audio`, `openai-whisper`, `torch`, `anthropic`, `openai`, `google-genai`, `spacy` (`de_core_news_lg`), `statsmodels`, `text_to_num`, and `nltk`.
- Several scripts require API keys or tokens via environment variables: `HF_TOKEN`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `REMOTE_API_URL`.
