from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


OUTPUT_COLUMNS = [
    "video_name",
    "frame_index",
    "timestamp_ms",
    "track_id",
    "x",
    "y",
    "width",
    "height",
    "visibility",
    "occluded",
    "outside",
    "occlusion_state",
]

ISSUE_COLUMNS = ["source", "line", "severity", "message", "raw_row"]

SUMMARY_COLUMNS = [
    "video_name",
    "minimum_frame_index",
    "maximum_frame_index",
    "unique_tracks",
    "annotation_rows",
    "inferred_outside_rows",
    "visible_rows",
    "partially_occluded_rows",
    "fully_occluded_rows",
    "outside_rows",
]


@dataclass(frozen=True)
class MotRecord:
    mot_frame: int
    track_id: int
    x: float
    y: float
    width: float
    height: float
    confidence: float
    class_id: int
    visibility: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert validated CVAT MOT ZIPs into canonical CSVs."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="A MOT ZIP file or directory containing MOT ZIP files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ground_truth_csv"),
        help="Output directory. Default: ground_truth_csv",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frame rate used to calculate timestamp_ms. Default: 30",
    )
    parser.add_argument(
        "--visible-threshold",
        type=float,
        default=0.99,
        help="Visibility at or above this value is VISIBLE. Default: 0.99",
    )
    parser.add_argument(
        "--zero-threshold",
        type=float,
        default=0.001,
        help="Visibility at or below this value is FULLY_OCCLUDED. Default: 0.001",
    )
    parser.add_argument(
        "--keep-mot-frame-index",
        action="store_true",
        help="Keep MOT's 1-based frame indices instead of converting to 0-based.",
    )
    parser.add_argument(
        "--infer-outside",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Infer outside rows from gaps inside tracks. Enabled by default.",
    )
    parser.add_argument(
        "--round-coordinates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Round x, y, width, and height to integers.",
    )
    parser.add_argument(
        "--video-name-regex",
        default=r"(?i)(?:_validated)?(?:_mot)?$",
        help="Regex removed from ZIP stems to derive video_name.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop on malformed MOT rows instead of recording and skipping them.",
    )
    return parser.parse_args()


def find_zip_files(input_path: Path) -> list[Path]:
    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        if input_path.suffix.lower() != ".zip":
            raise ValueError(f"Input is not a ZIP archive: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    files = sorted(path for path in input_path.rglob("*.zip") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No ZIP files found under: {input_path}")
    return files


def normalize_video_name(zip_path: Path, suffix_regex: str) -> str:
    name = re.sub(suffix_regex, "", zip_path.stem).strip("_- ")
    return name or zip_path.stem


def find_gt_txt(archive: zipfile.ZipFile) -> str:
    candidates = [
        name
        for name in archive.namelist()
        if not name.endswith("/") and Path(name).name.lower() == "gt.txt"
    ]
    if not candidates:
        raise FileNotFoundError("No gt.txt file found inside MOT ZIP.")
    preferred = [name for name in candidates if name.lower().endswith("gt/gt.txt")]
    return sorted(preferred or candidates, key=len)[0]


def parse_int(value: str, field: str) -> int:
    try:
        return int(float(value.strip()))
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {field}: {value!r}") from exc


def parse_float(value: str, field: str) -> float:
    try:
        return float(value.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid number for {field}: {value!r}") from exc


def parse_mot_text(
    text: str,
    source_name: str,
    strict: bool,
) -> tuple[list[MotRecord], list[dict[str, Any]]]:
    records: list[MotRecord] = []
    issues: list[dict[str, Any]] = []

    for line_number, row in enumerate(csv.reader(io.StringIO(text)), start=1):
        if not row or all(not cell.strip() for cell in row):
            continue

        if len(row) < 6:
            issue = {
                "source": source_name,
                "line": line_number,
                "severity": "error",
                "message": f"Expected at least 6 columns, found {len(row)}.",
                "raw_row": ",".join(row),
            }
            if strict:
                raise ValueError(issue["message"])
            issues.append(issue)
            continue

        try:
            frame = parse_int(row[0], "frame")
            track_id = parse_int(row[1], "track_id")
            x = parse_float(row[2], "x")
            y = parse_float(row[3], "y")
            width = parse_float(row[4], "width")
            height = parse_float(row[5], "height")
            confidence = parse_float(row[6], "confidence") if len(row) > 6 and row[6].strip() else 1.0
            class_id = parse_int(row[7], "class_id") if len(row) > 7 and row[7].strip() else 1
            visibility = parse_float(row[8], "visibility") if len(row) > 8 and row[8].strip() else 1.0
        except ValueError as exc:
            issue = {
                "source": source_name,
                "line": line_number,
                "severity": "error",
                "message": str(exc),
                "raw_row": ",".join(row),
            }
            if strict:
                raise
            issues.append(issue)
            continue

        if width <= 0 or height <= 0:
            issue = {
                "source": source_name,
                "line": line_number,
                "severity": "error",
                "message": f"Non-positive box dimensions: width={width}, height={height}.",
                "raw_row": ",".join(row),
            }
            if strict:
                raise ValueError(issue["message"])
            issues.append(issue)
            continue

        if not 0.0 <= visibility <= 1.0:
            issues.append(
                {
                    "source": source_name,
                    "line": line_number,
                    "severity": "warning",
                    "message": f"Visibility {visibility} was clamped to [0, 1].",
                    "raw_row": ",".join(row),
                }
            )
            visibility = max(0.0, min(1.0, visibility))

        records.append(
            MotRecord(
                mot_frame=frame,
                track_id=track_id,
                x=x,
                y=y,
                width=width,
                height=height,
                confidence=confidence,
                class_id=class_id,
                visibility=visibility,
            )
        )

    return records, issues


def classify_visibility(
    visibility: float,
    visible_threshold: float,
    zero_threshold: float,
) -> tuple[int, str]:
    if visibility <= zero_threshold:
        return 1, "FULLY_OCCLUDED"
    if visibility < visible_threshold:
        return 1, "PARTIALLY_OCCLUDED"
    return 0, "VISIBLE"


def output_frame_index(mot_frame: int, keep_mot_index: bool) -> int:
    return mot_frame if keep_mot_index else mot_frame - 1


def timestamp_ms(mot_frame: int, fps: float) -> float:
    return (mot_frame - 1) * 1000.0 / fps


def coordinate_value(value: float, round_coordinates: bool) -> int | str:
    if round_coordinates:
        return int(round(value))
    formatted = f"{value:.6f}".rstrip("0").rstrip(".")
    return formatted or "0"


def records_to_rows(
    video_name: str,
    records: list[MotRecord],
    fps: float,
    visible_threshold: float,
    zero_threshold: float,
    keep_mot_index: bool,
    infer_outside: bool,
    round_coordinates: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    by_track: dict[int, list[MotRecord]] = defaultdict(list)
    seen_pairs: set[tuple[int, int]] = set()

    for record in records:
        pair = (record.mot_frame, record.track_id)
        if pair in seen_pairs:
            issues.append(
                {
                    "source": video_name,
                    "line": "",
                    "severity": "warning",
                    "message": f"Duplicate frame/track pair: {pair}.",
                    "raw_row": "",
                }
            )
        seen_pairs.add(pair)
        by_track[record.track_id].append(record)

        occluded, state = classify_visibility(
            record.visibility,
            visible_threshold,
            zero_threshold,
        )
        rows.append(
            {
                "video_name": video_name,
                "frame_index": output_frame_index(record.mot_frame, keep_mot_index),
                "timestamp_ms": f"{timestamp_ms(record.mot_frame, fps):.3f}",
                "track_id": record.track_id,
                "x": coordinate_value(record.x, round_coordinates),
                "y": coordinate_value(record.y, round_coordinates),
                "width": coordinate_value(record.width, round_coordinates),
                "height": coordinate_value(record.height, round_coordinates),
                "visibility": f"{record.visibility:.6f}",
                "occluded": occluded,
                "outside": 0,
                "occlusion_state": state,
            }
        )

    if infer_outside:
        for track_id, track_records in by_track.items():
            annotated_frames = {record.mot_frame for record in track_records}
            first_frame = min(annotated_frames)
            last_frame = max(annotated_frames)
            for missing_frame in range(first_frame, last_frame + 1):
                if missing_frame in annotated_frames:
                    continue
                rows.append(
                    {
                        "video_name": video_name,
                        "frame_index": output_frame_index(missing_frame, keep_mot_index),
                        "timestamp_ms": f"{timestamp_ms(missing_frame, fps):.3f}",
                        "track_id": track_id,
                        "x": "",
                        "y": "",
                        "width": "",
                        "height": "",
                        "visibility": "0.000000",
                        "occluded": 1,
                        "outside": 1,
                        "occlusion_state": "OUTSIDE",
                    }
                )

    rows.sort(
        key=lambda row: (
            str(row["video_name"]),
            int(row["frame_index"]),
            int(row["track_id"]),
            int(row["outside"]),
        )
    )
    return rows, issues


def write_csv_file(
    path: Path,
    rows: Iterable[dict[str, Any]],
    columns: list[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def make_summary(video_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    non_outside = [row for row in rows if int(row["outside"]) == 0]
    outside = [row for row in rows if int(row["outside"]) == 1]
    states: dict[str, int] = defaultdict(int)
    for row in rows:
        states[str(row["occlusion_state"])] += 1
    frame_indices = [int(row["frame_index"]) for row in rows]
    track_ids = {int(row["track_id"]) for row in rows}
    return {
        "video_name": video_name,
        "minimum_frame_index": min(frame_indices) if frame_indices else "",
        "maximum_frame_index": max(frame_indices) if frame_indices else "",
        "unique_tracks": len(track_ids),
        "annotation_rows": len(non_outside),
        "inferred_outside_rows": len(outside),
        "visible_rows": states["VISIBLE"],
        "partially_occluded_rows": states["PARTIALLY_OCCLUDED"],
        "fully_occluded_rows": states["FULLY_OCCLUDED"],
        "outside_rows": states["OUTSIDE"],
    }


def main() -> int:
    args = parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be greater than zero.")
    if not 0.0 <= args.zero_threshold <= 1.0:
        raise ValueError("--zero-threshold must be between 0 and 1.")
    if not 0.0 <= args.visible_threshold <= 1.0:
        raise ValueError("--visible-threshold must be between 0 and 1.")
    if args.zero_threshold >= args.visible_threshold:
        raise ValueError("--zero-threshold must be lower than --visible-threshold.")

    zip_files = find_zip_files(args.input)
    output_dir = args.output.expanduser().resolve()
    per_video_dir = output_dir / "per_video"
    per_video_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    all_issues: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    seen_video_names: set[str] = set()

    for zip_path in zip_files:
        video_name = normalize_video_name(zip_path, args.video_name_regex)
        if video_name in seen_video_names:
            raise ValueError(
                f"Multiple ZIP files resolve to video_name={video_name!r}. "
                "Rename them or change --video-name-regex."
            )
        seen_video_names.add(video_name)

        print(f"[convert] {zip_path.name} -> {video_name}")
        with zipfile.ZipFile(zip_path, "r") as archive:
            gt_name = find_gt_txt(archive)
            mot_text = archive.read(gt_name).decode("utf-8-sig", errors="replace")

        records, parse_issues = parse_mot_text(
            mot_text,
            source_name=zip_path.name,
            strict=args.strict,
        )
        rows, conversion_issues = records_to_rows(
            video_name=video_name,
            records=records,
            fps=args.fps,
            visible_threshold=args.visible_threshold,
            zero_threshold=args.zero_threshold,
            keep_mot_index=args.keep_mot_frame_index,
            infer_outside=args.infer_outside,
            round_coordinates=args.round_coordinates,
        )

        write_csv_file(
            per_video_dir / f"{video_name}_ground_truth.csv",
            rows,
            OUTPUT_COLUMNS,
        )
        all_rows.extend(rows)
        all_issues.extend(parse_issues)
        all_issues.extend(conversion_issues)
        summaries.append(make_summary(video_name, rows))

        print(
            f"          {len(records)} MOT rows, {len(rows)} CSV rows, "
            f"{len(parse_issues) + len(conversion_issues)} issue(s)"
        )

    all_rows.sort(
        key=lambda row: (
            str(row["video_name"]),
            int(row["frame_index"]),
            int(row["track_id"]),
            int(row["outside"]),
        )
    )

    write_csv_file(
        output_dir / "ground_truth_all_videos.csv",
        all_rows,
        OUTPUT_COLUMNS,
    )
    write_csv_file(
        output_dir / "dataset_summary.csv",
        summaries,
        SUMMARY_COLUMNS,
    )
    write_csv_file(
        output_dir / "validation_issues.csv",
        all_issues,
        ISSUE_COLUMNS,
    )

    settings = {
        "fps": args.fps,
        "frame_indexing": "MOT 1-based" if args.keep_mot_frame_index else "OpenCV 0-based",
        "visible_threshold": args.visible_threshold,
        "zero_threshold": args.zero_threshold,
        "infer_outside": args.infer_outside,
        "round_coordinates": args.round_coordinates,
        "input_zip_count": len(zip_files),
    }
    (output_dir / "conversion_settings.json").write_text(
        json.dumps(settings, indent=2),
        encoding="utf-8",
    )

    print(f"\nComplete. Results written to: {output_dir}")
    print("Review validation_issues.csv and all inferred OUTSIDE rows.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
