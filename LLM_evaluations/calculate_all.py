"""Per-model QWK avg + SD across the 3 prompting runs, for both conditions.

Reuses calculate_qwk_ref.py's qwk()/load_reference()/load_model_runs() so
the numbers here are always computed the exact same way as that script's
"MODEL vs HUMAN" table -- this just adds the population standard deviation
across the 3 per-run QWK values (statistics.pstdev) as the error-bar source
for plot_qwk_bar_chart.py, and prints a DATA dict ready to paste there.

calculate_qwk_ref.py currently hardcodes its own PREPROCESSED_DIR/RESULT_DIRS
to the diarized folders, so this script does NOT import those constants --
it defines its own non-diarized and diarized folder maps explicitly and
calls the reusable functions directly, to get both conditions in one run
regardless of whatever calculate_qwk_ref.py's current hardcoded defaults are.

Usage:
    python calculate_run_sd.py
"""

import os
import statistics

import calculate_qwk_ref as c

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_ORDER = ["ground_truth", "turbo", "large", "medium", "small"]

NON_DIARIZED = {
    "preprocessed_dir": os.path.join(ROOT, "preprocessed"),
    "result_dirs": {name: os.path.join(ROOT, name) for name in MODEL_ORDER},
}
DIARIZED = {
    "preprocessed_dir": os.path.join(ROOT, "preprocessed_diarized"),
    "result_dirs": {name: os.path.join(ROOT, f"{name}_diarized") for name in MODEL_ORDER},
}


def compute_condition(preprocessed_dir, result_dirs):
    """model -> (avg_qwk, population_sd, [run1, run2, run3])."""
    by_lesson, ref1, ref2, skipped = c.load_reference(preprocessed_dir)
    print(f"  n reference segments: {len(ref1)}  (skipped {skipped})")

    results = {}
    for model in MODEL_ORDER:
        result_dir = result_dirs[model]
        if not os.path.isdir(result_dir):
            print(f"  {model:<14} SKIP: results folder not found: {result_dir}")
            continue

        run_pairs = c.load_model_runs(result_dir, by_lesson)
        run_qwks = []
        for pairs in run_pairs:
            if len(pairs) < 2:
                run_qwks.append(float("nan"))
                continue
            refs, preds = zip(*pairs)
            run_qwks.append(c.qwk(list(refs), list(preds)))

        avg = c.mean(run_qwks)
        valid = [v for v in run_qwks if not c.is_nan(v)]
        sd = statistics.pstdev(valid) if len(valid) == len(run_qwks) and valid else float("nan")
        results[model] = (avg, sd, run_qwks)

        runs_str = ", ".join(f"{v:.4f}" for v in run_qwks)
        print(f"  {model:<14} runs=[{runs_str}]  avg={avg:.4f}  sd={sd:.4f}")

    return results


def main():
    print("=" * 70)
    print("NON-DIARIZED (W/O Speaker Annotations)")
    print("=" * 70)
    wo = compute_condition(NON_DIARIZED["preprocessed_dir"], NON_DIARIZED["result_dirs"])

    print()
    print("=" * 70)
    print("DIARIZED (W/ Speaker Annotations)")
    print("=" * 70)
    w = compute_condition(DIARIZED["preprocessed_dir"], DIARIZED["result_dirs"])

    print()
    print("=" * 70)
    print("DATA dict for plot_qwk_bar_chart.py")
    print("=" * 70)
    display_labels = {
        "ground_truth": "Ground Truth",
        "turbo": "Turbo",
        "large": "Large",
        "medium": "Medium",
        "small": "Small",
    }
    print("DATA = {")
    for model in MODEL_ORDER:
        if model not in wo or model not in w:
            continue
        wo_avg, wo_sd, _ = wo[model]
        w_avg, w_sd, _ = w[model]
        label = display_labels[model]
        print(
            f'    "{label}":'.ljust(20)
            + f'{{"wo": {wo_avg:.4f}, "wo_sd": {wo_sd:.4f}, '
            f'"w": {w_avg:.4f}, "w_sd": {w_sd:.4f}}},'
        )
    print("}")


if __name__ == "__main__":
    main()
