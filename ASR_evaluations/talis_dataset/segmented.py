from datetime import timedelta
import argparse
import json
from pathlib import Path

def minutes(td: timedelta):
    return td.total_seconds() / 60

def hms_to_timedelta(time_str):
    h, m, s = map(int, time_str.split(":"))
    return timedelta(hours=h, minutes=m, seconds=s)

def timedelta_to_hms(td: timedelta):
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes_ = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours}:{minutes_:02}:{seconds:02}"

def segment_transcript(rows):

    SEGMENT_LENGTH = 16
    MIN_LAST = 8
    
    segments = []
    current_segment = []

    segment_start_time = rows[0]["time"]
    if segment_start_time is None:
        segment_start_time = timedelta(seconds=0)

    for row in rows:
        diff = minutes(row["time"] - segment_start_time)

        if diff < SEGMENT_LENGTH:
            current_segment.append(row)
        else:
            segments.append(current_segment)
            current_segment = [row]
            segment_start_time = row["time"]

    if current_segment:
        segments.append(current_segment)

    # merge last segment if needed
    if len(segments) > 1:
        last_segment = segments[-1]
        first_time = last_segment[0]["time"]
        last_time = last_segment[-1]["time"]

        last_minutes = minutes(last_time - first_time)

        if last_minutes <= MIN_LAST:
            segments[-2].extend(last_segment)
            segments.pop()

    return segments


def build_output(segments):
    output = []
    for seg_idx, seg in enumerate(segments, start=1):
        seg_start = seg[0]["time"]
        seg_end = seg[-1]["time"]
        seg_duration = seg_end - seg_start

        output.append({
            "segment": seg_idx,
            "start": timedelta_to_hms(seg_start),
            "end": timedelta_to_hms(seg_end),
            "duration_minutes": round(minutes(seg_duration), 3),
            "item_count": len(seg),
            "items": [
                {
                    "time": timedelta_to_hms(item["time"]),
                    "preprocessed_text": item["preprocessed_reference"],
                }
                for item in seg
            ]
        })
    return output


def process_directory(results_dir: Path):
    for input_path in sorted(results_dir.glob("*ground_truth*.json")):
        file_name_model = input_path.stem
        print(file_name_model)

        with open(input_path, "r") as f:
            data = json.load(f)

        for row in data:
            row["time"] = hms_to_timedelta(row["time"]) if row["time"] is not None else timedelta(seconds=0)

        segments = segment_transcript(data)
        output = build_output(segments)

        output_path = results_dir / f"{file_name_model}_segmented.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment ground-truth transcripts into fixed-length chunks.")
    parser.add_argument(
        "results_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent / "results_preprocess",
        help="Directory containing the ground_truth JSON files (default: results_preprocess next to this script).",
    )
    args = parser.parse_args()
    process_directory(args.results_dir)

