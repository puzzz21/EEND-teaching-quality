"""
Fit the exact additive model used in the paper table:

    log(1 + WER) ~ student speech + student microphone
                   + median SNR + median C50
                   + log utterance duration + overlap

There is NO speaker-by-microphone interaction.

Input
-----
snr_wer_results_all.json, produced by the merge script. It must contain:

    {
      "data": {
        "teacher_mic_teacher": [...],
        "teacher_mic_student": [...],
        "student_mic_teacher": [...],
        "student_mic_student": [...]
      }
    }

Outputs
-------
regression_exact_table.tex
regression_exact_results.csv
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm


# Use the same minimum duration as the analysis script.
MIN_DURATION_SECONDS = 1.0

PREDICTORS = [
    "student_speech",
    "student_microphone",
    "median_snr",
    "median_c50",
    "log_utterance_duration",
    "overlap",
]

DISPLAY_NAMES = {
    "student_speech": "Student speech",
    "student_microphone": "Student microphone",
    "median_snr": "Median SNR",
    "median_c50": "Median C50",
    "log_utterance_duration": "Log utterance duration",
    "overlap": r"Overlap (\% of audio)",
}


def finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_results(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    groups = payload.get("data")
    if not isinstance(groups, dict):
        raise ValueError("Input JSON must contain a dictionary under 'data'.")

    results = [
        record
        for group in groups.values()
        if isinstance(group, list)
        for record in group
        if isinstance(record, dict)
    ]
    if not results:
        raise ValueError("No result records were found in the input JSON.")
    return results


def build_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for result in results:
        wer = finite_float(result.get("wer"))
        duration = finite_float(
            result.get("audio_duration", result.get("duration"))
        )
        snr = finite_float(result.get("median_snr"))
        c50 = finite_float(result.get("median_c50"))
        overlap = finite_float(result.get("overlap_percentage_of_audio"))

        microphone = result.get("microphone")
        speaker_role = result.get("speaker_role")

        # Support older merged files that encoded group membership but did
        # not store microphone and speaker_role inside each record.
        if microphone not in {"teacher", "student"}:
            microphone = result.get("_microphone")
        if speaker_role not in {"teacher", "student"}:
            speaker = result.get("speaker")
            if speaker == "L":
                speaker_role = "teacher"
            elif speaker is not None:
                # L is the teacher; every other speaker code (S01-S29, Ss,
                # S?, K, TL, TL2, O, ...) counts as a student.
                speaker_role = "student"

        cluster_key = result.get("cluster_key")
        if cluster_key is None:
            cluster_key = result.get("file_number")

        if (
            wer is None
            or wer < 0
            or duration is None
            or duration < MIN_DURATION_SECONDS
            or microphone not in {"teacher", "student"}
            or speaker_role not in {"teacher", "student"}
            or cluster_key is None
        ):
            continue

        rows.append(
            {
                "log1p_wer": math.log1p(wer),
                "student_speech": int(speaker_role == "student"),
                "student_microphone": int(microphone == "student"),
                "median_snr": snr,
                "median_c50": c50,
                "log_utterance_duration": math.log(duration),
                # Percentage points: 1 unit means one percentage point.
                "overlap": overlap,
                "cluster_key": str(cluster_key),
            }
        )

    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        raise ValueError("No usable rows could be constructed.")

    before = len(dataframe)
    dataframe = (
        dataframe.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["log1p_wer", "cluster_key", *PREDICTORS])
        .copy()
    )

    print(f"Rows before complete-case filtering: {before}")
    print(f"Rows used in regression:             {len(dataframe)}")
    print(
        f"Recording clusters:                  "
        f"{dataframe['cluster_key'].nunique()}"
    )

    if len(dataframe) < 30:
        raise ValueError("Fewer than 30 complete observations remain.")
    if dataframe["cluster_key"].nunique() < 2:
        raise ValueError("At least two recording clusters are required.")

    return dataframe


def fit_model(dataframe: pd.DataFrame) -> Any:
    x = sm.add_constant(
        dataframe[PREDICTORS].astype(float),
        has_constant="add",
    )
    y = dataframe["log1p_wer"].astype(float)

    # OLS coefficients with recording-clustered standard errors.
    return sm.OLS(y, x).fit(
        cov_type="cluster",
        cov_kwds={
            "groups": dataframe["cluster_key"].to_numpy(),
            "use_correction": True,
        },
    )


def result_table(model: Any) -> pd.DataFrame:
    intervals = model.conf_int(alpha=0.05)
    records = []

    for predictor in PREDICTORS:
        records.append(
            {
                "predictor": predictor,
                "display_name": DISPLAY_NAMES[predictor],
                "beta": float(model.params[predictor]),
                "standard_error": float(model.bse[predictor]),
                "ci_95_lower": float(intervals.loc[predictor, 0]),
                "ci_95_upper": float(intervals.loc[predictor, 1]),
                "p_value": float(model.pvalues[predictor]),
            }
        )

    return pd.DataFrame(records)


def latex_number(value: float, digits: int = 3) -> str:
    rounded = f"{value:.{digits}f}"
    if rounded in ("-0." + "0" * digits,):
        rounded = rounded[1:]  # avoid printing negative zero, e.g. "-0.000"
    return f"${rounded}$" if rounded.startswith("-") else rounded


def latex_p_value(value: float) -> str:
    if value < 0.001:
        return "$<0.001$"
    return f"{value:.3f}"


def write_latex(
    table: pd.DataFrame,
    model: Any,
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    rows = []
    for _, result in table.iterrows():
        ci = (
            f"[{latex_number(float(result['ci_95_lower']))}, "
            f"{latex_number(float(result['ci_95_upper']))}]"
        )
        rows.append(
            f"{result['display_name']} & "
            f"{latex_number(float(result['beta']))} & "
            f"{ci} & "
            f"{latex_p_value(float(result['p_value']))} \\\\"
        )

    latex = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Regression coefficients for predicting "
        "$\\log(1+\\mathrm{WER})$. Standard errors are clustered by "
        "recording.}\n"
        "\\label{tab:regression}\n"
        "\\begin{tabular}{lccc}\n"
        "\\toprule\n"
        "\\textbf{Predictor} & \\textbf{$\\beta$} & \\textbf{95\\% CI} & "
        "\\textbf{$p$} \\\\\n"
        "\\midrule\n"
        + "\n".join(rows)
        + "\n\\midrule\n"
        f"Observations & \\multicolumn{{3}}{{c}}{{{int(model.nobs)}}} \\\\\n"
        f"Recording clusters & \\multicolumn{{3}}{{c}}{{"
        f"{dataframe['cluster_key'].nunique()}}} \\\\\n"
        f"$R^2$ & \\multicolumn{{3}}{{c}}{{{model.rsquared:.3f}}} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    path.write_text(latex, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the log(1+WER) additive regression model and export a LaTeX table + CSV."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Merged SNR/WER results JSON (must contain a 'data' dict of groups).",
    )
    parser.add_argument(
        "--latex-path",
        type=Path,
        default=Path("regression_exact_table.tex"),
        help="Where to write the LaTeX table.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("regression_exact_results.csv"),
        help="Where to write the coefficient CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input_path
    latex_path = args.latex_path
    csv_path = args.csv_path

    results = load_results(input_path)

    # Recover microphone/speaker role from group names when older records
    # do not contain those fields.
    with input_path.open("r", encoding="utf-8") as file:
        groups = json.load(file)["data"]
    enriched: list[dict[str, Any]] = []
    for group_name, group in groups.items():
        if not isinstance(group, list):
            continue
        microphone = "student" if group_name.startswith("student_mic") else "teacher"
        speaker_role = "student" if group_name.endswith("_student") else "teacher"
        for record in group:
            if isinstance(record, dict):
                copy = dict(record)
                copy.setdefault("_microphone", microphone)
                copy.setdefault("speaker_role", speaker_role)
                enriched.append(copy)

    # Use enriched records when available; otherwise retain the ordinary load.
    dataframe = build_dataframe(enriched or results)
    model = fit_model(dataframe)
    table = result_table(model)

    table.to_csv(csv_path, index=False)
    write_latex(table, model, dataframe, latex_path)

    print(model.summary())
    print("\nTable values:")
    print(table.to_string(index=False))
    print(f"\nLaTeX table: {latex_path.resolve()}")
    print(f"CSV results: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
