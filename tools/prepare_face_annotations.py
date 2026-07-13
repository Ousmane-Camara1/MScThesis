#!/usr/bin/env python3
"""
prepare_face_annotations.py

Creates ten standardized 15-second clips, runs an Ultralytics-compatible
face detector plus Deep SORT tracking, and exports:

- canonical CSV annotations
- CVAT-compatible MOT ZIP files
- annotated preview videos
- automatic occlusion-candidate intervals

The generated output is model-assisted pre-annotation, not final ground truth.
Manually inspect and correct it in CVAT before using it for thesis evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO


@dataclass(frozen=True)
class ClipSpec:
    clip_name: str
    source_path: Path
    start_seconds: float
    category: str
    notes: str


CSV_FIELDS = [
    "clip_name",
    "source_video",
    "clip_start_seconds",
    "category",
    "frame_index_0",
    "mot_frame_1",
    "timestamp_ms",
    "track_id",
    "x",
    "y",
    "width",
    "height",
    "detector_confidence",
    "annotation_source",
    "visibility",
    "occlusion_candidate",
    "prediction_gap_frames",
    "manual_occlusion_state",
    "manually_verified",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare fixed video clips and CVAT face-track pre-annotations."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Ultralytics-compatible face detector .pt file.",
    )
    parser.add_argument("--output", type=Path, default=Path("prepared_dataset"))
    parser.add_argument("--expected-clips", type=int, default=10)
    parser.add_argument("--clip-seconds", type=float, default=15.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--face-class-id", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-age", type=int, default=30)
    parser.add_argument("--n-init", type=int, default=1)
    parser.add_argument("--max-predicted-gap", type=int, default=15)
    parser.add_argument("--predicted-visibility", type=float, default=0.50)
    parser.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--trim",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--annotate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def require_command(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(
            f"'{name}' was not found. On macOS install it with: brew install ffmpeg"
        )
    return path


def normalize_name(value: str) -> str:
    value = value.strip()
    cleaned = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value
    ).strip("_")
    if not cleaned:
        raise ValueError("clip_name cannot be empty.")
    return cleaned


def read_manifest(path: Path, expected_clips: int) -> list[ClipSpec]:
    required = {"clip_name", "source_path", "start_seconds"}
    clips: list[ClipSpec] = []
    seen: set[str] = set()

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Manifest has no header.")
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing manifest columns: {sorted(missing)}")

        for row_number, row in enumerate(reader, start=2):
            clip_name = normalize_name(row["clip_name"])
            if clip_name in seen:
                raise ValueError(f"Duplicate clip_name on row {row_number}: {clip_name}")
            seen.add(clip_name)

            source = Path(row["source_path"]).expanduser()
            if not source.is_absolute():
                source = (path.parent / source).resolve()
            if not source.exists():
                raise FileNotFoundError(f"Missing source video: {source}")

            start = float(row["start_seconds"])
            if start < 0:
                raise ValueError(f"Negative start_seconds on row {row_number}.")

            clips.append(
                ClipSpec(
                    clip_name=clip_name,
                    source_path=source,
                    start_seconds=start,
                    category=(row.get("category") or "").strip(),
                    notes=(row.get("notes") or "").strip(),
                )
            )

    if len(clips) != expected_clips:
        raise ValueError(
            f"Expected {expected_clips} clips, but manifest contains {len(clips)}."
        )
    return clips


def run_checked(command: list[str]) -> None:
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(command)
            + "\n\n"
            + result.stderr[-4000:]
        )


def trim_clip(
    spec: ClipSpec,
    destination: Path,
    seconds: float,
    fps: int,
    width: int,
    height: int,
    overwrite: bool,
) -> None:
    if destination.exists() and not overwrite:
        print(f"[trim] Reusing {destination.name}")
        return

    ffmpeg = require_command("ffmpeg")
    frame_count = int(round(seconds * fps))
    video_filter = (
        f"fps={fps},"
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,"
        "setsar=1"
    )

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{spec.start_seconds:.6f}",
        "-i",
        str(spec.source_path),
        "-vf",
        video_filter,
        "-frames:v",
        str(frame_count),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    run_checked(command)


def inspect_video(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    metadata = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return metadata


def validate_clip(
    path: Path,
    seconds: float,
    fps: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    metadata = inspect_video(path)
    expected_frames = int(round(seconds * fps))

    if metadata["frame_count"] != expected_frames:
        raise RuntimeError(
            f"{path.name}: expected {expected_frames} frames, "
            f"found {metadata['frame_count']}."
        )
    if metadata["width"] != width or metadata["height"] != height:
        raise RuntimeError(
            f"{path.name}: expected {width}x{height}, "
            f"found {metadata['width']}x{metadata['height']}."
        )
    if not math.isclose(metadata["fps"], fps, rel_tol=0.01, abs_tol=0.05):
        raise RuntimeError(
            f"{path.name}: expected {fps} FPS, found {metadata['fps']:.3f}."
        )
    return metadata


def clamp_box(
    box: list[float] | tuple[float, ...],
    frame_width: int,
    frame_height: int,
) -> Optional[tuple[int, int, int, int]]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0, min(frame_width - 1, int(round(x1))))
    y1 = max(0, min(frame_height - 1, int(round(y1))))
    x2 = max(0, min(frame_width, int(round(x2))))
    y2 = max(0, min(frame_height, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def detect_faces(
    model: YOLO,
    frame: np.ndarray,
    args: argparse.Namespace,
    device: str,
) -> list[tuple[list[float], float, str]]:
    classes = None if args.face_class_id < 0 else [args.face_class_id]
    results = model.predict(
        source=frame,
        imgsz=args.imgsz,
        conf=args.confidence,
        iou=args.iou,
        device=device,
        classes=classes,
        verbose=False,
    )

    detections: list[tuple[list[float], float, str]] = []
    if not results or results[0].boxes is None:
        return detections

    boxes = results[0].boxes
    xyxy = boxes.xyxy.detach().cpu().numpy()
    scores = boxes.conf.detach().cpu().numpy()

    for box, score in zip(xyxy, scores):
        x1, y1, x2, y2 = [float(v) for v in box]
        w = x2 - x1
        h = y2 - y1
        if w > 1 and h > 1:
            detections.append(([x1, y1, w, h], float(score), "face"))

    return detections


def track_box(track: Any) -> tuple[Optional[np.ndarray], bool]:
    try:
        original = track.to_ltrb(orig=True, orig_strict=True)
    except TypeError:
        original = None

    if original is not None:
        return original, True
    return track.to_ltrb(), False


def numeric_track_id(raw_id: Any) -> int:
    text = str(raw_id)
    try:
        return int(text.split("_")[-1])
    except ValueError:
        return abs(hash(text)) % 2_000_000_000 + 1


def tracker_confidence(track: Any) -> Optional[float]:
    getter = getattr(track, "get_det_conf", None)
    if callable(getter):
        value = getter()
        return None if value is None else float(value)
    value = getattr(track, "det_conf", None)
    return None if value is None else float(value)


def make_preview_writer(
    path: Path,
    fps: int,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create preview: {path}")
    return writer


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_occlusion_intervals(
    rows: list[dict[str, Any]],
    fps: int,
) -> list[dict[str, Any]]:
    by_track: dict[int, list[int]] = defaultdict(list)

    for row in rows:
        if int(row["occlusion_candidate"]) == 1:
            by_track[int(row["track_id"])].append(int(row["mot_frame_1"]))

    intervals: list[dict[str, Any]] = []

    for track_id, frames in by_track.items():
        frames = sorted(set(frames))
        if not frames:
            continue

        start = frames[0]
        previous = frames[0]

        for frame_number in frames[1:] + [None]:
            if frame_number is not None and frame_number == previous + 1:
                previous = frame_number
                continue

            intervals.append(
                {
                    "track_id": track_id,
                    "start_mot_frame_1": start,
                    "end_mot_frame_1": previous,
                    "duration_frames": previous - start + 1,
                    "start_ms": round((start - 1) * 1000 / fps, 3),
                    "end_ms": round(previous * 1000 / fps, 3),
                    "automatic_reason": (
                        "Deep SORT prediction without a current detector match"
                    ),
                    "manual_review": "",
                    "manual_occlusion_state": "",
                }
            )

            if frame_number is None:
                break
            start = frame_number
            previous = frame_number

    return intervals


def write_occlusion_csv(
    path: Path,
    intervals: list[dict[str, Any]],
) -> None:
    fields = [
        "track_id",
        "start_mot_frame_1",
        "end_mot_frame_1",
        "duration_frames",
        "start_ms",
        "end_ms",
        "automatic_reason",
        "manual_review",
        "manual_occlusion_state",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(intervals)


def write_mot_zip(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    for row in rows:
        lines.append(
            ",".join(
                [
                    str(int(row["mot_frame_1"])),
                    str(int(row["track_id"])),
                    str(int(row["x"])),
                    str(int(row["y"])),
                    str(int(row["width"])),
                    str(int(row["height"])),
                    "1",
                    "1",
                    f"{float(row['visibility']):.4f}",
                ]
            )
        )

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("gt/labels.txt", "face\n")
        archive.writestr("gt/gt.txt", "\n".join(lines) + "\n")


def prepare_directories(root: Path) -> dict[str, Path]:
    directories = {
        "root": root,
        "clips": root / "clips",
        "csv": root / "canonical_csv",
        "mot": root / "cvat_mot",
        "previews": root / "previews",
        "occlusion": root / "occlusion_candidates",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def process_clip(
    spec: ClipSpec,
    clip_path: Path,
    model: YOLO,
    args: argparse.Namespace,
    device: str,
    directories: dict[str, Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata = validate_clip(
        clip_path,
        args.clip_seconds,
        args.fps,
        args.width,
        args.height,
    )

    tracker = DeepSort(
        max_age=args.max_age,
        n_init=args.n_init,
        max_cosine_distance=0.25,
        nn_budget=100,
        embedder="mobilenet",
        half=False,
        bgr=True,
        embedder_gpu=False,
    )

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open clip: {clip_path}")

    preview_writer: Optional[cv2.VideoWriter] = None
    if args.preview:
        preview_writer = make_preview_writer(
            directories["previews"] / f"{spec.clip_name}_preview.mp4",
            args.fps,
            args.width,
            args.height,
        )

    rows: list[dict[str, Any]] = []
    unique_tracks: set[int] = set()
    frame_index = 0
    start_time = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections = detect_faces(model, frame, args, device)
        tracks = tracker.update_tracks(detections, frame=frame)

        mot_frame = frame_index + 1
        timestamp_ms = frame_index * 1000.0 / args.fps

        for track in tracks:
            if not track.is_confirmed():
                continue

            gap = int(getattr(track, "time_since_update", 0))
            raw_box, has_detection = track_box(track)

            if raw_box is None:
                continue
            if not has_detection and gap > args.max_predicted_gap:
                continue

            box = clamp_box(raw_box, args.width, args.height)
            if box is None:
                continue

            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            track_id = numeric_track_id(track.track_id)
            unique_tracks.add(track_id)

            confidence = tracker_confidence(track)
            source = "detector_matched" if has_detection else "tracker_prediction"
            occlusion_candidate = 0 if has_detection else 1
            visibility = 1.0 if has_detection else args.predicted_visibility

            rows.append(
                {
                    "clip_name": spec.clip_name,
                    "source_video": str(spec.source_path),
                    "clip_start_seconds": f"{spec.start_seconds:.6f}",
                    "category": spec.category,
                    "frame_index_0": frame_index,
                    "mot_frame_1": mot_frame,
                    "timestamp_ms": f"{timestamp_ms:.3f}",
                    "track_id": track_id,
                    "x": x1,
                    "y": y1,
                    "width": width,
                    "height": height,
                    "detector_confidence": (
                        "" if confidence is None else f"{confidence:.6f}"
                    ),
                    "annotation_source": source,
                    "visibility": f"{visibility:.4f}",
                    "occlusion_candidate": occlusion_candidate,
                    "prediction_gap_frames": gap,
                    "manual_occlusion_state": "",
                    "manually_verified": "",
                    "notes": spec.notes,
                }
            )

            if preview_writer is not None:
                color = (0, 255, 0) if has_detection else (0, 165, 255)
                label = (
                    f"ID {track_id} DETECTED"
                    if has_detection
                    else f"ID {track_id} OCCLUSION? gap={gap}"
                )
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        if preview_writer is not None:
            cv2.putText(
                frame,
                f"{spec.clip_name} | frame {mot_frame}",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            preview_writer.write(frame)

        frame_index += 1
        if frame_index % 50 == 0:
            print(
                f"[annotate] {spec.clip_name}: "
                f"{frame_index}/{metadata['frame_count']} frames"
            )

    elapsed = time.perf_counter() - start_time
    cap.release()
    if preview_writer is not None:
        preview_writer.release()

    canonical_path = directories["csv"] / f"{spec.clip_name}.csv"
    mot_path = directories["mot"] / f"{spec.clip_name}_mot.zip"
    occlusion_path = (
        directories["occlusion"]
        / f"{spec.clip_name}_occlusion_candidates.csv"
    )

    write_csv(canonical_path, rows)
    write_mot_zip(mot_path, rows)
    intervals = build_occlusion_intervals(rows, args.fps)
    write_occlusion_csv(occlusion_path, intervals)

    summary = {
        "clip_name": spec.clip_name,
        "category": spec.category,
        "frames": frame_index,
        "duration_seconds": frame_index / args.fps,
        "preliminary_track_count": len(unique_tracks),
        "annotation_rows": len(rows),
        "occlusion_candidate_intervals": len(intervals),
        "processing_seconds": round(elapsed, 3),
        "offline_processing_fps": (
            round(frame_index / elapsed, 3) if elapsed > 0 else 0.0
        ),
        "canonical_csv": str(canonical_path),
        "cvat_mot_zip": str(mot_path),
    }
    return rows, summary


def write_summary(
    path: Path,
    summaries: list[dict[str, Any]],
) -> None:
    if not summaries:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)


def main() -> int:
    args = parse_args()
    args.output = args.output.resolve()
    args.manifest = args.manifest.resolve()
    args.weights = args.weights.resolve()

    directories = prepare_directories(args.output)
    clips = read_manifest(args.manifest, args.expected_clips)

    if not args.weights.exists():
        raise FileNotFoundError(f"Detector weights not found: {args.weights}")

    device = select_device(args.device)
    print(f"[environment] Detector device: {device}")
    print("[environment] Deep SORT appearance embedder: CPU")

    shutil.copy2(
        args.manifest,
        directories["root"] / "clips_manifest_used.csv",
    )

    run_metadata = {
        "manifest": str(args.manifest),
        "weights": str(args.weights),
        "device": device,
        "clip_seconds": args.clip_seconds,
        "fps": args.fps,
        "resolution": [args.width, args.height],
        "imgsz": args.imgsz,
        "confidence": args.confidence,
        "iou": args.iou,
        "max_age": args.max_age,
        "max_predicted_gap": args.max_predicted_gap,
        "python": sys.version,
        "torch": torch.__version__,
        "opencv": cv2.__version__,
    }
    (directories["root"] / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2),
        encoding="utf-8",
    )

    if args.trim:
        for spec in clips:
            destination = directories["clips"] / f"{spec.clip_name}.mp4"
            trim_clip(
                spec,
                destination,
                args.clip_seconds,
                args.fps,
                args.width,
                args.height,
                args.overwrite,
            )
            validate_clip(
                destination,
                args.clip_seconds,
                args.fps,
                args.width,
                args.height,
            )

    if not args.annotate:
        print("Clip creation complete.")
        return 0

    model = YOLO(str(args.weights))
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for index, spec in enumerate(clips, start=1):
        clip_path = directories["clips"] / f"{spec.clip_name}.mp4"
        if not clip_path.exists():
            raise FileNotFoundError(f"Missing standardized clip: {clip_path}")

        print(f"\n[annotate] {index}/{len(clips)}: {spec.clip_name}")
        rows, summary = process_clip(
            spec,
            clip_path,
            model,
            args,
            device,
            directories,
        )
        all_rows.extend(rows)
        summaries.append(summary)

    write_csv(directories["root"] / "all_preannotations.csv", all_rows)
    write_summary(directories["root"] / "dataset_summary.csv", summaries)

    instructions = f"""CVAT REVIEW CHECKLIST

1. Create one CVAT video task per clip in:
   {directories["clips"]}

2. Create one label named:
   face

3. Import the matching MOT archive from:
   {directories["mot"]}

4. Use preview videos from:
   {directories["previews"]}

5. Green boxes are current detector matches.
   Orange boxes are Deep SORT predictions and possible occlusions.

6. Review every track and correct:
   - missed faces
   - false positives
   - inaccurate boxes
   - identity switches
   - tracks continuing after exit
   - partial and full occlusion intervals
   - the first visible frame after reappearance

7. Candidate intervals are listed in:
   {directories["occlusion"]}

8. Export corrected annotations before using them as thesis ground truth.

Recommended thesis wording:
"manually validated model-assisted annotations"
"""
    (directories["root"] / "CVAT_REVIEW_INSTRUCTIONS.txt").write_text(
        instructions,
        encoding="utf-8",
    )

    print(f"\nComplete. Output: {directories['root']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
