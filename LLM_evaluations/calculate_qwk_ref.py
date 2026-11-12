"""Calculate quadratic weighted kappa (QWK) over the preprocessed segmented files.

Methodology (matches the paper writeup this script is scored against):

  Model performance was computed by comparing the model predictions with the
  average score assigned by the two human raters for each segment. As a
  human baseline, we computed QWK between the scores assigned independently
  by the two human raters. To account for LLM stochasticity, each model was
  prompted three times, and the average QWK across the three runs was
  reported.

Concretely:
  1. HUMAN RELIABILITY: a single QWK(rater 1, rater 2) -- plain pairwise
     kappa between the two human raters' scores, no averaging or
     generalization across a larger rater pool.
  2. MODEL vs HUMAN: for each model, QWK is computed three times -- once per
     prompting run -- between that run's raw per-segment ratings and the
     rounded average of the two human raters, then the three run-level QWK
     values are averaged and reported as the model's score. This is NOT the
     same as averaging the three runs' ratings first and computing one QWK
     on the average (that collapses run-to-run variance before scoring
     instead of after, which is a different -- usually higher -- number).

Raw per-run ratings are read directly from RESULT_DIRS below (each
<lesson>_output.json holds 3 runs x per-segment {"rating": n}), not from the
pre-averaged model columns in preprocessed/*.json. Those columns (written by
set_ratings_plain copy.py) exist for other tooling but are not used here,
since they've already averaged away the run-to-run variance this script is
supposed to report separately.

Reference scores (both raters' scores per segment) still come from
preprocessed/*.json's "reference" field -- that part is unaffected by the
run-averaging question, since it's human data, not model output.

Scores are on a 1-4 scale; fractional values are rounded to the nearest
integer category (half-up: 2.5 -> 3) before computing kappa. Segments with a
missing/NaN reference, or a missing/NaN rating in a given run, are skipped
for that comparison.
"""

import argparse
import glob
import json
import math
import os
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED_DIR = os.path.join(ROOT, "preprocessed")
MIN_RATING = 1
MAX_RATING = 4

MODEL_ORDER = ["ground_truth", "turbo", "large", "medium", "small"]


def build_result_dirs(results_root):
    """model key -> folder of <lesson>_output.json raw per-run ratings.

    Matches the mapping in set_ratings_plain copy.py, which populates the
    pre-averaged columns in preprocessed/*.json from these same folders.
    """
    return {model: os.path.join(results_root, model) for model in MODEL_ORDER}

ROUNDING = "half_up"  # "half_up" or "bankers"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def round_rating(value, mode=None):
    """Round a score to the nearest integer category.

    half_up: 2.5 -> 3, 3.5 -> 4 (consistent direction)
    bankers: Python's built-in round(); 2.5 -> 2, 3.5 -> 4
    """
    mode = mode or ROUNDING
    if mode == "bankers":
        return int(round(value))
    return int(math.floor(value + 0.5))


def is_nan(value):
    return isinstance(value, float) and math.isnan(value)


def valid_rating(value):
    return isinstance(value, (int, float)) and not is_nan(value)


def clamp(value, lo=MIN_RATING, hi=MAX_RATING):
    return max(lo, min(hi, value))


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def qwk(a, b, min_rating=MIN_RATING, max_rating=MAX_RATING):
    """Quadratic weighted kappa between two lists of integer ratings.

    Returns float('nan') when the expected disagreement is zero, which happens
    if either rater used only a single category. Reporting NaN is safer than
    dividing by zero or silently returning 0.
    """
    cats = list(range(min_rating, max_rating + 1))
    k = len(cats)
    idx = {c: i for i, c in enumerate(cats)}

    observed = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        observed[idx[x]][idx[y]] += 1

    n = len(a)
    hist_a = Counter(a)
    hist_b = Counter(b)

    numerator = 0.0
    denominator = 0.0
    for i in range(k):
        for j in range(k):
            weight = (i - j) ** 2 / (k - 1) ** 2
            expected = hist_a[cats[i]] * hist_b[cats[j]] / n
            numerator += weight * observed[i][j]
            denominator += weight * expected

    if denominator == 0:
        return float("nan")
    return 1 - numerator / denominator


def mean(values):
    values = [v for v in values if not is_nan(v)]
    return sum(values) / len(values) if values else float("nan")


def fmt(value):
    return "  n/a " if isinstance(value, float) and math.isnan(value) else f"{value:.4f}"


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #

def load_reference(directory=PREPROCESSED_DIR, rounding=None):
    """lesson number -> list of (rater1, rater2, avg) per segment, in order.

    Also returns the flat ref1/ref2 lists (over every segment with a valid
    reference, across all lessons) for the human-reliability QWK.
    """
    by_lesson = {}
    ref1, ref2 = [], []
    skipped = 0

    for path in sorted(glob.glob(os.path.join(directory, "*_segmented.json"))):
        lesson = os.path.basename(path).split("_")[0]
        with open(path) as f:
            objects = json.load(f)

        lesson_refs = []
        for obj in objects:
            reference = obj.get("reference")
            if (
                not isinstance(reference, list)
                or len(reference) < 2
                or not all(valid_rating(v) for v in reference[:2])
            ):
                skipped += 1
                lesson_refs.append(None)
                continue
            r1 = clamp(round_rating(reference[0], rounding))
            r2 = clamp(round_rating(reference[1], rounding))
            avg = clamp(round_rating(sum(reference[:2]) / 2, rounding))
            lesson_refs.append((r1, r2, avg))
            ref1.append(r1)
            ref2.append(r2)

        by_lesson[lesson] = lesson_refs

    return by_lesson, ref1, ref2, skipped


def load_model_runs(result_dir, by_lesson, rounding=None):
    """3 lists of (ref_avg, pred) pairs, one per prompting run, pooled across lessons.

    Segments are matched to their reference by (lesson, index). A run
    contributes a pair only where both the reference and that run's rating
    for the segment are valid -- runs are allowed to have gaps (e.g. a
    shorter trailing segment) independently of each other.
    """
    run_pairs = [[], [], []]

    for lesson, lesson_refs in by_lesson.items():
        out_path = os.path.join(result_dir, f"{lesson}_output.json")
        if not os.path.exists(out_path):
            continue
        try:
            with open(out_path) as f:
                runs = json.load(f)
        except json.JSONDecodeError:
            continue

        for run_idx, run in enumerate(runs[:3]):
            for seg_idx, ref in enumerate(lesson_refs):
                if ref is None or seg_idx >= len(run):
                    continue
                rating = run[seg_idx].get("rating")
                if not valid_rating(rating):
                    continue
                _, _, ref_avg = ref
                pred = clamp(round_rating(rating, rounding))
                run_pairs[run_idx].append((ref_avg, pred))

    return run_pairs


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def report_human(ref1, ref2):
    print("=" * 78)
    print("HUMAN RELIABILITY")
    print("=" * 78)
    print(f"  n segments        : {len(ref1)}")
    print(f"  QWK (rater 1 vs rater 2) : {fmt(qwk(ref1, ref2))}")
    print()
    return qwk(ref1, ref2)


def report_models(by_lesson, human_baseline, result_dirs, rounding=None):
    print("=" * 78)
    print("MODEL vs HUMAN (average QWK across 3 prompting runs)")
    print("=" * 78)
    header = f"  {'model':<16}{'n (run1/2/3)':>16}{'run1':>9}{'run2':>9}{'run3':>9}{'avg QWK':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    print(f"  {'human baseline':<16}{'':>16}{'':>9}{'':>9}{'':>9}{fmt(human_baseline):>10}")
    print("  " + "-" * (len(header) - 2))

    qwk_avg_by_model = {}
    for model in MODEL_ORDER:
        result_dir = result_dirs[model]
        if not os.path.isdir(result_dir):
            print(f"  {model:<16}  results folder not found: {result_dir}")
            continue

        run_pairs = load_model_runs(result_dir, by_lesson, rounding)
        run_qwks = []
        ns = []
        for pairs in run_pairs:
            ns.append(len(pairs))
            if len(pairs) < 2:
                run_qwks.append(float("nan"))
                continue
            refs, preds = zip(*pairs)
            run_qwks.append(qwk(list(refs), list(preds)))

        avg_qwk = mean(run_qwks)
        qwk_avg_by_model[model] = avg_qwk
        n_str = "/".join(str(n) for n in ns)
        print(
            f"  {model:<16}{n_str:>16}"
            f"{fmt(run_qwks[0]):>9}{fmt(run_qwks[1]):>9}{fmt(run_qwks[2]):>9}"
            f"{fmt(avg_qwk):>10}"
        )

    print()
    ordered_values = [round(qwk_avg_by_model.get(name, float("nan")), 4) for name in MODEL_ORDER]
    print(f"  QWK avg list ({', '.join(m.upper() for m in MODEL_ORDER)}): {ordered_values}")
    print()
    print("  Each run's QWK is computed against the rounded average of the two")
    print("  human raters, on that run's own per-segment ratings (not an average")
    print("  of ratings across runs). 'avg QWK' is the mean of the three")
    print("  per-run QWK values -- report this column, and compare it against")
    print("  the human baseline row above.")
    print()
    return qwk_avg_by_model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default=PREPROCESSED_DIR, help="directory of *_segmented.json (reference source)")
    parser.add_argument(
        "--results-root",
        default=ROOT,
        help="directory containing the per-model result folders (ground_truth/large/medium/small/turbo)",
    )
    parser.add_argument(
        "--rounding",
        choices=["half_up", "bankers"],
        default=ROUNDING,
        help="tie-breaking rule for .5 values (default: half_up)",
    )
    args = parser.parse_args()

    by_lesson, ref1, ref2, skipped = load_reference(args.dir, args.rounding)

    if not ref1:
        print(f"No usable segments found in {args.dir}")
        return

    print()
    print(f"files read     : {len(by_lesson)}")
    print(f"objects used   : {len(ref1)}  (skipped {skipped} with missing/NaN reference)")
    print(f"rounding rule  : {args.rounding}")
    print()

    human_baseline = report_human(ref1, ref2)
    result_dirs = build_result_dirs(args.results_root)
    report_models(by_lesson, human_baseline, result_dirs, rounding=args.rounding)


if __name__ == "__main__":
    main()
