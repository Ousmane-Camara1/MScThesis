#!/usr/bin/env python3
"""
benchmark_v1_v2.py

Unified Raspberry Pi benchmark for the thesis privacy pipeline.

It provides:
- Pipeline V1: preserved global-state behavior based on the original script.
- Pipeline V2: per-target state, clarified CNN interval semantics,
  detection-to-track association, and aggressive CNN reacquisition.
- Ground-truth evaluation using validated canonical CSV annotations.
- Privacy-mask coverage classification per visible target-frame.
- Frame-level, target-level, output-event, and run-summary CSV files.
- Correct end-to-end throughput calculation.
- Stage-level latency logging.
- CPU, temperature, frequency, and throttling telemetry.
- Optional headless execution.
- Configurable look-ahead/startup buffer for RQ3.

Required ground-truth columns:
    video_name
    frame_index
    timestamp_ms
    track_id
    x
    y
    width
    height
    visibility
    occluded
    outside
    occlusion_state

Important experimental separation:
- Instrumentation and ground-truth evaluation are shared by V1 and V2.
- V1 preserves the original global state and all-tracker replacement behavior.
- V2 contains the algorithmic improvements.
- Use buffer_size=0 for V1 versus V2 and CNN-interval experiments.
- Use V2 with buffer sizes 0, 5, 15, and 30 for the startup-buffer experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np

try:
    import psutil
except ImportError:
    psutil = None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

BBox = tuple[int, int, int, int]


@dataclass
class GroundTruthTarget:
    video_name: str
    frame_index: int
    timestamp_ms: float
    track_id: int
    bbox: BBox
    visibility: float
    occluded: bool
    outside: bool
    occlusion_state: str


@dataclass
class StageTimes:
    decode_ms: float = 0.0
    resize_ms: float = 0.0
    detection_ms: float = 0.0
    tracking_ms: float = 0.0
    association_ms: float = 0.0
    coasting_ms: float = 0.0
    masking_ms: float = 0.0
    buffer_ms: float = 0.0
    evaluation_ms: float = 0.0
    telemetry_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class PipelineResult:
    boxes: list[BBox]
    state_summary: str
    cnn_executed: bool
    stage_times: StageTimes
    active_tracks: int


@dataclass
class V1State:
    consecutive_misses: int = 0
    cached_faces: list[BBox] = field(default_factory=list)
    trackers: list[Any] = field(default_factory=list)
    current_mode: str = "INITIALIZING"


@dataclass
class TargetTrack:
    track_id: int
    bbox: BBox
    tracker: Any
    state: str = "TRACKING"
    missed_frames: int = 0
    last_detection_frame: int = -1


@dataclass
class TargetEvaluationState:
    ever_visible: bool = False
    previously_visible: bool = False
    leak_streak: int = 0
    leak_event_count: int = 0
    leak_event_durations: list[int] = field(default_factory=list)
    max_leak_streak: int = 0
    pending_reappearance_frame: Optional[int] = None
    reacquisition_delays: list[int] = field(default_factory=list)
    unresolved_reappearances: int = 0


@dataclass
class BufferItem:
    frame_index: int
    frame: Optional[np.ndarray]
    has_any_mask: bool
    visible_target_count: int
    protected_target_count: int


@dataclass
class BufferOutput:
    released: list[BufferItem] = field(default_factory=list)
    dropped: list[BufferItem] = field(default_factory=list)
    gate_open: bool = False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Pipeline V1 or V2 against validated ground truth."
    )

    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--video-name", default=None)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--pipeline-version",
        choices=["v1", "v2"],
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--run-id", default=None)

    parser.add_argument(
        "--cnn-interval",
        type=int,
        default=4,
        help=(
            "Actual interval between CNN frames. "
            "1 means CNN every frame; 4 means one CNN frame followed by "
            "three tracker-only frames."
        ),
    )
    parser.add_argument("--max-lost-frames", type=int, default=15)
    parser.add_argument("--expansion-margin", type=int, default=10)
    parser.add_argument("--association-iou", type=float, default=0.20)
    parser.add_argument("--tracker-reinit-iou", type=float, default=0.45)

    parser.add_argument("--detect-width", type=int, default=480)
    parser.add_argument("--detect-height", type=int, default=360)
    parser.add_argument("--score-threshold", type=float, default=0.60)
    parser.add_argument("--nms-threshold", type=float, default=0.30)
    parser.add_argument("--top-k", type=int, default=50)

    parser.add_argument("--mask-padding", type=int, default=30)
    parser.add_argument("--protected-threshold", type=float, default=0.90)
    parser.add_argument("--partial-threshold", type=float, default=0.50)

    parser.add_argument(
        "--buffer-size",
        type=int,
        default=0,
        help="Look-ahead/startup buffer capacity in frames.",
    )
    parser.add_argument(
        "--startup-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For buffer_size > 0, hold output until a privacy mask exists and "
            "drop the unprotected startup prefix."
        ),
    )

    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument(
        "--include-decode-in-throughput",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--save-output-video", type=Path, default=None)
    parser.add_argument("--telemetry-every", type=int, default=30)
    parser.add_argument("--repeat-number", type=int, default=1)
    parser.add_argument(
        "--stop-after-frames",
        type=int,
        default=0,
        help="0 processes the complete video.",
    )
    parser.add_argument(
        "--strict-ground-truth",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    args = parser.parse_args()

    if args.cnn_interval < 1:
        parser.error("--cnn-interval must be at least 1.")
    if args.max_lost_frames < 0:
        parser.error("--max-lost-frames cannot be negative.")
    if args.expansion_margin < 0:
        parser.error("--expansion-margin cannot be negative.")
    if args.buffer_size < 0:
        parser.error("--buffer-size cannot be negative.")
    if not 0.0 <= args.partial_threshold < args.protected_threshold <= 1.0:
        parser.error(
            "Require 0 <= partial-threshold < protected-threshold <= 1."
        )
    if args.telemetry_every < 1:
        parser.error("--telemetry-every must be at least 1.")

    return args


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def now_ms() -> float:
    return time.perf_counter_ns() / 1_000_000.0


def normalize_name(value: str) -> str:
    stem = Path(value).stem.lower()
    for suffix in (
        "_ground_truth",
        "_validated_mot",
        "_validated",
        "_mot",
    ):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return "".join(ch for ch in stem if ch.isalnum())


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    array = np.asarray(values, dtype=np.float64)
    return float(np.percentile(array, q))


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(statistics.fmean(values)) if values else 0.0


def clamp_bbox(box: BBox, width: int, height: int) -> Optional[BBox]:
    x, y, w, h = box
    x1 = max(0, min(width - 1, int(round(x))))
    y1 = max(0, min(height - 1, int(round(y))))
    x2 = max(0, min(width, int(round(x + w))))
    y2 = max(0, min(height, int(round(y + h))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


def expand_bbox(
    box: BBox,
    margin: int,
    frame_width: int,
    frame_height: int,
) -> BBox:
    x, y, w, h = box
    expanded = (
        x - margin,
        y - margin,
        w + 2 * margin,
        h + 2 * margin,
    )
    clamped = clamp_bbox(expanded, frame_width, frame_height)
    return clamped if clamped is not None else (0, 0, 1, 1)


def bbox_to_xyxy(box: BBox) -> tuple[int, int, int, int]:
    x, y, w, h = box
    return x, y, x + w, y + h


def bbox_iou(first: BBox, second: BBox) -> float:
    ax1, ay1, ax2, ay2 = bbox_to_xyxy(first)
    bx1, by1, bx2, by2 = bbox_to_xyxy(second)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def greedy_iou_match(
    track_boxes: dict[int, BBox],
    detections: list[BBox],
    minimum_iou: float,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    candidates: list[tuple[float, int, int]] = []

    for track_id, track_box in track_boxes.items():
        for detection_index, detection in enumerate(detections):
            score = bbox_iou(track_box, detection)
            if score >= minimum_iou:
                candidates.append((score, track_id, detection_index))

    candidates.sort(reverse=True)

    matched_tracks: set[int] = set()
    matched_detections: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for score, track_id, detection_index in candidates:
        if track_id in matched_tracks or detection_index in matched_detections:
            continue
        matched_tracks.add(track_id)
        matched_detections.add(detection_index)
        matches.append((track_id, detection_index, score))

    unmatched_tracks = set(track_boxes).difference(matched_tracks)
    unmatched_detections = set(range(len(detections))).difference(
        matched_detections
    )

    return matches, unmatched_tracks, unmatched_detections


def create_tracker() -> Any:
    factories = [
        lambda: cv2.legacy.TrackerMOSSE_create(),
        lambda: cv2.legacy.TrackerKCF_create(),
        lambda: cv2.TrackerKCF_create(),
    ]

    for factory in factories:
        try:
            return factory()
        except (AttributeError, cv2.error):
            continue

    raise RuntimeError(
        "No MOSSE or KCF tracker is available. Install an OpenCV contrib build."
    )


def initialize_tracker(frame: np.ndarray, box: BBox) -> Any:
    tracker = create_tracker()
    result = tracker.init(frame, tuple(float(value) for value in box))
    if result is False:
        raise RuntimeError(f"Tracker initialization failed for box {box}.")
    return tracker


def update_tracker(tracker: Any, frame: np.ndarray) -> tuple[bool, Optional[BBox]]:
    try:
        success, raw_box = tracker.update(frame)
    except cv2.error:
        return False, None

    if not success:
        return False, None

    x, y, w, h = [int(round(value)) for value in raw_box]
    clamped = clamp_bbox((x, y, w, h), frame.shape[1], frame.shape[0])
    return (clamped is not None), clamped


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def parse_bool(value: Any) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def load_ground_truth(
    path: Path,
    selected_video_name: str,
    strict: bool,
) -> tuple[dict[int, list[GroundTruthTarget]], set[int], list[str]]:
    required = {
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
    }

    by_frame: dict[int, list[GroundTruthTarget]] = defaultdict(list)
    all_track_ids: set[int] = set()
    issues: list[str] = []
    normalized_selected = normalize_name(selected_video_name)
    available_names: set[str] = set()

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Ground-truth CSV has no header.")

        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Ground-truth CSV is missing columns: {sorted(missing)}"
            )

        for line_number, row in enumerate(reader, start=2):
            row_video = row["video_name"]
            available_names.add(row_video)

            if normalize_name(row_video) != normalized_selected:
                continue

            try:
                frame_index = int(float(row["frame_index"]))
                track_id = int(float(row["track_id"]))
                outside = parse_bool(row["outside"])
                visibility = float(row["visibility"])
                state = row["occlusion_state"].strip().upper()

                if outside:
                    bbox = (0, 0, 0, 0)
                else:
                    x = int(round(float(row["x"])))
                    y = int(round(float(row["y"])))
                    width = int(round(float(row["width"])))
                    height = int(round(float(row["height"])))
                    bbox = (x, y, width, height)
                    if width <= 0 or height <= 0:
                        raise ValueError("Non-positive width or height.")

                target = GroundTruthTarget(
                    video_name=row_video,
                    frame_index=frame_index,
                    timestamp_ms=float(row["timestamp_ms"]),
                    track_id=track_id,
                    bbox=bbox,
                    visibility=visibility,
                    occluded=parse_bool(row["occluded"]),
                    outside=outside,
                    occlusion_state=state,
                )
            except (TypeError, ValueError) as exc:
                message = f"Line {line_number}: {exc}"
                if strict:
                    raise ValueError(message) from exc
                issues.append(message)
                continue

            by_frame[frame_index].append(target)
            all_track_ids.add(track_id)

    if not by_frame:
        raise ValueError(
            f"No rows matched video name {selected_video_name!r}. "
            f"Available video_name values: {sorted(available_names)}"
        )

    return dict(by_frame), all_track_ids, issues


def is_visible_target(target: GroundTruthTarget) -> bool:
    return (
        not target.outside
        and target.occlusion_state not in {"FULLY_OCCLUDED", "OUTSIDE"}
        and target.bbox[2] > 0
        and target.bbox[3] > 0
    )


# ---------------------------------------------------------------------------
# YuNet detector
# ---------------------------------------------------------------------------

class YuNetDetector:
    def __init__(self, args: argparse.Namespace) -> None:
        if not args.model_path.exists():
            raise FileNotFoundError(f"YuNet model not found: {args.model_path}")

        self.input_size = (args.detect_width, args.detect_height)
        self.detector = cv2.FaceDetectorYN.create(
            model=str(args.model_path),
            config="",
            input_size=self.input_size,
            score_threshold=args.score_threshold,
            nms_threshold=args.nms_threshold,
            top_k=args.top_k,
        )

    def detect(self, frame: np.ndarray) -> tuple[list[BBox], float, float]:
        resize_start = now_ms()
        small_frame = cv2.resize(frame, self.input_size)
        resize_ms = now_ms() - resize_start

        detection_start = now_ms()
        _, faces = self.detector.detect(small_frame)
        detection_ms = now_ms() - detection_start

        if faces is None:
            return [], resize_ms, detection_ms

        frame_height, frame_width = frame.shape[:2]
        scale_x = frame_width / self.input_size[0]
        scale_y = frame_height / self.input_size[1]

        boxes: list[BBox] = []
        for face in faces:
            raw = face[:4]
            box = (
                int(round(raw[0] * scale_x)),
                int(round(raw[1] * scale_y)),
                int(round(raw[2] * scale_x)),
                int(round(raw[3] * scale_y)),
            )
            clamped = clamp_bbox(box, frame_width, frame_height)
            if clamped is not None:
                boxes.append(clamped)

        return boxes, resize_ms, detection_ms


def warm_up_detector(
    detector: YuNetDetector,
    sample_frame: np.ndarray,
    runs: int,
) -> None:
    for _ in range(max(0, runs)):
        detector.detect(sample_frame)


# ---------------------------------------------------------------------------
# Pipeline V1
# ---------------------------------------------------------------------------

class PipelineV1:
    """
    Preserves the original script's global-state behavior:
    - one global miss counter
    - one global cached-face list
    - all trackers are replaced after every successful CNN frame
    - a failure of one target can be hidden by another successful target
    """

    def __init__(
        self,
        detector: YuNetDetector,
        args: argparse.Namespace,
    ) -> None:
        self.detector = detector
        self.args = args
        self.state = V1State()

    def process(self, frame: np.ndarray, frame_index: int) -> PipelineResult:
        times = StageTimes()
        frame_height, frame_width = frame.shape[:2]
        current_faces: list[BBox] = []
        tracking_successful = False

        run_cnn = (
            frame_index % self.args.cnn_interval == 0
            or (
                len(self.state.trackers) == 0
                and self.state.consecutive_misses == 0
            )
        )

        if run_cnn:
            detections, resize_ms, detection_ms = self.detector.detect(frame)
            times.resize_ms = resize_ms
            times.detection_ms = detection_ms

            if detections:
                self.state.trackers = []
                tracker_start = now_ms()

                for box in detections:
                    current_faces.append(box)
                    self.state.trackers.append(
                        initialize_tracker(frame, box)
                    )

                times.tracking_ms += now_ms() - tracker_start
                self.state.cached_faces = list(current_faces)
                self.state.consecutive_misses = 0
                tracking_successful = True
                self.state.current_mode = "DETECTION"
        else:
            tracker_start = now_ms()
            trackers_to_keep: list[Any] = []

            for tracker in self.state.trackers:
                success, box = update_tracker(tracker, frame)
                if success and box is not None:
                    current_faces.append(box)
                    trackers_to_keep.append(tracker)

            times.tracking_ms = now_ms() - tracker_start
            self.state.trackers = trackers_to_keep

            if current_faces:
                self.state.cached_faces = list(current_faces)
                self.state.consecutive_misses = 0
                tracking_successful = True
                self.state.current_mode = "TRACKING"

        if not tracking_successful:
            coast_start = now_ms()
            self.state.consecutive_misses += 1
            self.state.trackers = []

            if self.state.consecutive_misses <= self.args.max_lost_frames:
                self.state.cached_faces = [
                    expand_bbox(
                        box,
                        self.args.expansion_margin,
                        frame_width,
                        frame_height,
                    )
                    for box in self.state.cached_faces
                ]
                self.state.current_mode = (
                    f"MEMORY_COAST_{self.state.consecutive_misses}"
                )
            else:
                self.state.cached_faces = []
                self.state.current_mode = "SEARCHING"

            times.coasting_ms = now_ms() - coast_start

        return PipelineResult(
            boxes=list(self.state.cached_faces),
            state_summary=self.state.current_mode,
            cnn_executed=run_cnn,
            stage_times=times,
            active_tracks=len(self.state.cached_faces),
        )


# ---------------------------------------------------------------------------
# Pipeline V2
# ---------------------------------------------------------------------------

class PipelineV2:
    """
    Per-target implementation:
    - one tracker, state, and miss counter per target
    - actual CNN interval semantics
    - aggressive CNN execution while any target is coasting
    - IoU association between detections and existing targets
    - tracker reuse when detection correction remains sufficiently close
    """

    def __init__(
        self,
        detector: YuNetDetector,
        args: argparse.Namespace,
    ) -> None:
        self.detector = detector
        self.args = args
        self.tracks: dict[int, TargetTrack] = {}
        self.next_track_id = 1

    def _create_track(
        self,
        frame: np.ndarray,
        box: BBox,
        frame_index: int,
    ) -> TargetTrack:
        track = TargetTrack(
            track_id=self.next_track_id,
            bbox=box,
            tracker=initialize_tracker(frame, box),
            state="TRACKING",
            missed_frames=0,
            last_detection_frame=frame_index,
        )
        self.next_track_id += 1
        return track

    def _coast_track(
        self,
        track: TargetTrack,
        frame_width: int,
        frame_height: int,
    ) -> bool:
        track.missed_frames += 1

        if track.missed_frames > self.args.max_lost_frames:
            track.state = "SEARCHING"
            return False

        track.bbox = expand_bbox(
            track.bbox,
            self.args.expansion_margin,
            frame_width,
            frame_height,
        )
        track.state = "COASTING"
        track.tracker = None
        return True

    def process(self, frame: np.ndarray, frame_index: int) -> PipelineResult:
        times = StageTimes()
        frame_height, frame_width = frame.shape[:2]

        aggressive_reacquisition = (
            not self.tracks
            or any(
                track.state in {"COASTING", "SEARCHING"}
                for track in self.tracks.values()
            )
        )
        scheduled_cnn = frame_index % self.args.cnn_interval == 0
        run_cnn = scheduled_cnn or aggressive_reacquisition

        if run_cnn:
            detections, resize_ms, detection_ms = self.detector.detect(frame)
            times.resize_ms = resize_ms
            times.detection_ms = detection_ms

            association_start = now_ms()
            current_boxes = {
                track_id: track.bbox
                for track_id, track in self.tracks.items()
            }
            matches, unmatched_tracks, unmatched_detections = greedy_iou_match(
                current_boxes,
                detections,
                self.args.association_iou,
            )
            times.association_ms = now_ms() - association_start

            tracker_start = now_ms()

            for track_id, detection_index, score in matches:
                track = self.tracks[track_id]
                detection_box = detections[detection_index]

                if (
                    track.tracker is None
                    or score < self.args.tracker_reinit_iou
                ):
                    track.tracker = initialize_tracker(
                        frame,
                        detection_box,
                    )

                track.bbox = detection_box
                track.state = "TRACKING"
                track.missed_frames = 0
                track.last_detection_frame = frame_index

            for detection_index in unmatched_detections:
                new_track = self._create_track(
                    frame,
                    detections[detection_index],
                    frame_index,
                )
                self.tracks[new_track.track_id] = new_track

            tracker_fallbacks: dict[int, tuple[bool, Optional[BBox]]] = {}
            for track_id in unmatched_tracks:
                track = self.tracks[track_id]
                if track.tracker is None:
                    tracker_fallbacks[track_id] = (False, None)
                else:
                    tracker_fallbacks[track_id] = update_tracker(
                        track.tracker,
                        frame,
                    )

            times.tracking_ms += now_ms() - tracker_start

            coast_start = now_ms()
            tracks_to_remove: list[int] = []

            for track_id in unmatched_tracks:
                track = self.tracks[track_id]
                tracker_success, tracker_box = tracker_fallbacks[track_id]

                if tracker_success and tracker_box is not None:
                    track.bbox = tracker_box
                    track.state = "TRACKING"
                    track.missed_frames = 0
                elif not self._coast_track(
                    track,
                    frame_width,
                    frame_height,
                ):
                    tracks_to_remove.append(track_id)

            for track_id in tracks_to_remove:
                del self.tracks[track_id]

            times.coasting_ms = now_ms() - coast_start

        else:
            tracker_start = now_ms()
            tracks_to_remove: list[int] = []

            for track_id, track in self.tracks.items():
                if track.tracker is None:
                    success, box = False, None
                else:
                    success, box = update_tracker(track.tracker, frame)

                if success and box is not None:
                    track.bbox = box
                    track.state = "TRACKING"
                    track.missed_frames = 0
                else:
                    if not self._coast_track(
                        track,
                        frame_width,
                        frame_height,
                    ):
                        tracks_to_remove.append(track_id)

            for track_id in tracks_to_remove:
                del self.tracks[track_id]

            times.tracking_ms = now_ms() - tracker_start

        state_counts: dict[str, int] = defaultdict(int)
        for track in self.tracks.values():
            state_counts[track.state] += 1

        if not state_counts:
            state_summary = "SEARCHING:0"
        else:
            state_summary = ";".join(
                f"{state}:{state_counts[state]}"
                for state in sorted(state_counts)
            )

        return PipelineResult(
            boxes=[track.bbox for track in self.tracks.values()],
            state_summary=state_summary,
            cnn_executed=run_cnn,
            stage_times=times,
            active_tracks=len(self.tracks),
        )


# ---------------------------------------------------------------------------
# Reversible masking and exact privacy-mask geometry
# ---------------------------------------------------------------------------

class ReversibleMasker:
    def __init__(self, maximum_width: int, maximum_height: int) -> None:
        rng = np.random.default_rng(seed=42)
        self.key_matrix = rng.integers(
            0,
            256,
            size=(maximum_height, maximum_width, 3),
            dtype=np.uint8,
        )
        self.ellipse_cache: dict[tuple[int, int], np.ndarray] = {}

    def ellipse_mask(self, width: int, height: int) -> np.ndarray:
        key = (width, height)
        cached = self.ellipse_cache.get(key)
        if cached is not None:
            return cached

        mask = np.zeros((height, width), dtype=np.uint8)
        center = (width // 2, height // 2)
        axes = (max(1, width // 2), max(1, height // 2))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        self.ellipse_cache[key] = mask
        return mask

    def render(
        self,
        frame: np.ndarray,
        boxes: list[BBox],
        padding: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = frame.shape[:2]
        union_mask = np.zeros((height, width), dtype=np.uint8)

        for box in boxes:
            x, y, w, h = box
            padded = clamp_bbox(
                (
                    x - padding,
                    y - padding,
                    w + 2 * padding,
                    h + 2 * padding,
                ),
                width,
                height,
            )
            if padded is None:
                continue

            x1, y1, roi_width, roi_height = padded
            x2 = x1 + roi_width
            y2 = y1 + roi_height

            roi = frame[y1:y2, x1:x2]
            key_roi = self.key_matrix[:roi_height, :roi_width]
            ellipse = self.ellipse_mask(roi_width, roi_height)

            encrypted = cv2.bitwise_xor(roi, key_roi)
            cv2.copyTo(encrypted, ellipse, roi)
            cv2.bitwise_or(
                union_mask[y1:y2, x1:x2],
                ellipse,
                dst=union_mask[y1:y2, x1:x2],
            )

        return frame, union_mask


# ---------------------------------------------------------------------------
# Privacy evaluator
# ---------------------------------------------------------------------------

class PrivacyEvaluator:
    def __init__(
        self,
        ground_truth_by_frame: dict[int, list[GroundTruthTarget]],
        all_track_ids: set[int],
        protected_threshold: float,
        partial_threshold: float,
    ) -> None:
        self.gt = ground_truth_by_frame
        self.protected_threshold = protected_threshold
        self.partial_threshold = partial_threshold
        self.states = {
            track_id: TargetEvaluationState()
            for track_id in all_track_ids
        }
        self.target_rows: list[dict[str, Any]] = []

        self.total_visible_target_frames = 0
        self.total_protected = 0
        self.total_partial = 0
        self.total_leaked = 0

    @staticmethod
    def coverage(
        privacy_mask: np.ndarray,
        box: BBox,
    ) -> float:
        frame_height, frame_width = privacy_mask.shape[:2]
        clamped = clamp_bbox(box, frame_width, frame_height)
        if clamped is None:
            return 0.0

        x, y, w, h = clamped
        region = privacy_mask[y : y + h, x : x + w]
        if region.size == 0:
            return 0.0

        covered_pixels = int(np.count_nonzero(region))
        return covered_pixels / float(w * h)

    def classify(self, coverage: float) -> str:
        if coverage >= self.protected_threshold:
            return "PROTECTED"
        if coverage >= self.partial_threshold:
            return "PARTIALLY_PROTECTED"
        return "LEAKED"

    def _close_leak_event(self, state: TargetEvaluationState) -> None:
        if state.leak_streak > 0:
            state.leak_event_durations.append(state.leak_streak)
            state.leak_streak = 0

    def evaluate_frame(
        self,
        run_id: str,
        frame_index: int,
        privacy_mask: np.ndarray,
    ) -> dict[str, Any]:
        targets = [
            target
            for target in self.gt.get(frame_index, [])
            if is_visible_target(target)
        ]
        current_visible_ids = {target.track_id for target in targets}

        for track_id, state in self.states.items():
            if track_id not in current_visible_ids:
                self._close_leak_event(state)
                state.previously_visible = False

        protected_count = 0
        partial_count = 0
        leaked_count = 0
        reappearance_count = 0
        acquired_reappearance_count = 0

        for target in targets:
            state = self.states[target.track_id]
            target_coverage = self.coverage(
                privacy_mask,
                target.bbox,
            )
            status = self.classify(target_coverage)

            reappearance = (
                state.ever_visible and not state.previously_visible
            )
            reacquisition_delay: Optional[int] = None

            if reappearance:
                reappearance_count += 1
                if status == "PROTECTED":
                    reacquisition_delay = 0
                    state.reacquisition_delays.append(0)
                    acquired_reappearance_count += 1
                else:
                    state.pending_reappearance_frame = frame_index

            if (
                state.pending_reappearance_frame is not None
                and status == "PROTECTED"
            ):
                reacquisition_delay = (
                    frame_index - state.pending_reappearance_frame
                )
                state.reacquisition_delays.append(reacquisition_delay)
                state.pending_reappearance_frame = None
                acquired_reappearance_count += 1

            if status == "LEAKED":
                leaked_count += 1
                self.total_leaked += 1

                if state.leak_streak == 0:
                    state.leak_event_count += 1

                state.leak_streak += 1
                state.max_leak_streak = max(
                    state.max_leak_streak,
                    state.leak_streak,
                )
            else:
                self._close_leak_event(state)

                if status == "PROTECTED":
                    protected_count += 1
                    self.total_protected += 1
                else:
                    partial_count += 1
                    self.total_partial += 1

            self.total_visible_target_frames += 1
            state.ever_visible = True
            state.previously_visible = True

            self.target_rows.append(
                {
                    "run_id": run_id,
                    "frame_index": frame_index,
                    "track_id": target.track_id,
                    "ground_truth_occlusion_state": target.occlusion_state,
                    "ground_truth_visibility": target.visibility,
                    "x": target.bbox[0],
                    "y": target.bbox[1],
                    "width": target.bbox[2],
                    "height": target.bbox[3],
                    "coverage": round(target_coverage, 6),
                    "privacy_status": status,
                    "reappearance_event": int(reappearance),
                    "reacquisition_delay_frames": (
                        ""
                        if reacquisition_delay is None
                        else reacquisition_delay
                    ),
                }
            )

        return {
            "visible_gt_targets": len(targets),
            "protected_targets": protected_count,
            "partially_protected_targets": partial_count,
            "leaked_targets": leaked_count,
            "reappearance_events": reappearance_count,
            "acquired_reappearances": acquired_reappearance_count,
        }

    def finalize(self) -> dict[str, Any]:
        all_event_durations: list[int] = []
        all_reacquisition_delays: list[int] = []
        leak_event_count = 0
        maximum_consecutive = 0
        unresolved_reappearances = 0

        for state in self.states.values():
            self._close_leak_event(state)

            if state.pending_reappearance_frame is not None:
                state.unresolved_reappearances += 1
                state.pending_reappearance_frame = None

            all_event_durations.extend(state.leak_event_durations)
            all_reacquisition_delays.extend(state.reacquisition_delays)
            leak_event_count += state.leak_event_count
            maximum_consecutive = max(
                maximum_consecutive,
                state.max_leak_streak,
            )
            unresolved_reappearances += state.unresolved_reappearances

        leakage_rate = (
            100.0
            * self.total_leaked
            / self.total_visible_target_frames
            if self.total_visible_target_frames
            else 0.0
        )

        return {
            "visible_target_frames": self.total_visible_target_frames,
            "protected_target_frames": self.total_protected,
            "partially_protected_target_frames": self.total_partial,
            "leaked_target_frames": self.total_leaked,
            "target_leakage_rate_pct": leakage_rate,
            "leak_event_count": leak_event_count,
            "maximum_consecutive_leaked_target_frames": maximum_consecutive,
            "mean_leak_event_duration_frames": safe_mean(
                all_event_durations
            ),
            "mean_reacquisition_delay_frames": safe_mean(
                all_reacquisition_delays
            ),
            "maximum_reacquisition_delay_frames": (
                max(all_reacquisition_delays)
                if all_reacquisition_delays
                else 0
            ),
            "resolved_reappearance_count": len(
                all_reacquisition_delays
            ),
            "unresolved_reappearance_count": unresolved_reappearances,
        }


# ---------------------------------------------------------------------------
# Look-ahead/startup output buffer
# ---------------------------------------------------------------------------

class LookAheadBuffer:
    def __init__(
        self,
        capacity: int,
        startup_gate: bool,
    ) -> None:
        self.capacity = capacity
        self.startup_gate = startup_gate and capacity > 0
        self.queue: deque[BufferItem] = deque()
        self.gate_open = not self.startup_gate
        self.total_startup_dropped = 0

    def push(self, item: BufferItem) -> BufferOutput:
        result = BufferOutput(gate_open=self.gate_open)

        if self.capacity == 0:
            result.released.append(item)
            return result

        self.queue.append(item)

        if not self.gate_open:
            if item.has_any_mask:
                self.gate_open = True

                while self.queue and not self.queue[0].has_any_mask:
                    dropped = self.queue.popleft()
                    result.dropped.append(dropped)
                    self.total_startup_dropped += 1
            else:
                result.gate_open = False
                return result

        if len(self.queue) > self.capacity:
            result.released.append(self.queue.popleft())

        result.gate_open = self.gate_open
        return result

    def flush(self) -> list[BufferItem]:
        if not self.gate_open:
            self.total_startup_dropped += len(self.queue)
            self.queue.clear()
            return []

        released = list(self.queue)
        self.queue.clear()
        return released


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

class TelemetrySampler:
    def __init__(self, every_frames: int) -> None:
        self.every_frames = every_frames
        self.last = {
            "cpu_percent": "",
            "temperature_c": "",
            "cpu_frequency_mhz": "",
            "throttled_hex": "",
        }

        if psutil is not None:
            psutil.cpu_percent(interval=None)

    @staticmethod
    def read_temperature() -> Any:
        thermal_paths = [
            Path("/sys/class/thermal/thermal_zone0/temp"),
            Path("/sys/class/thermal/thermal_zone1/temp"),
        ]

        for path in thermal_paths:
            try:
                raw = float(path.read_text().strip())
                return raw / 1000.0 if raw > 1000 else raw
            except (OSError, ValueError):
                continue

        try:
            result = subprocess.run(
                ["vcgencmd", "measure_temp"],
                capture_output=True,
                text=True,
                timeout=1,
                check=False,
            )
            text = result.stdout.strip()
            if "=" in text:
                return float(
                    text.split("=", 1)[1]
                    .replace("'C", "")
                    .strip()
                )
        except (OSError, ValueError, subprocess.SubprocessError):
            pass

        return ""

    @staticmethod
    def read_throttled() -> Any:
        try:
            result = subprocess.run(
                ["vcgencmd", "get_throttled"],
                capture_output=True,
                text=True,
                timeout=1,
                check=False,
            )
            text = result.stdout.strip()
            if "=" in text:
                return text.split("=", 1)[1]
        except (OSError, subprocess.SubprocessError):
            pass

        return ""

    @staticmethod
    def read_frequency() -> Any:
        if psutil is not None:
            try:
                frequency = psutil.cpu_freq()
                if frequency is not None:
                    return frequency.current
            except (OSError, AttributeError):
                pass

        path = Path(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
        )
        try:
            return float(path.read_text().strip()) / 1000.0
        except (OSError, ValueError):
            return ""

    def sample(self, frame_index: int) -> tuple[dict[str, Any], float]:
        start = now_ms()

        if frame_index % self.every_frames == 0:
            if psutil is not None:
                try:
                    self.last["cpu_percent"] = psutil.cpu_percent(
                        interval=None
                    )
                except OSError:
                    self.last["cpu_percent"] = ""

            self.last["temperature_c"] = self.read_temperature()
            self.last["cpu_frequency_mhz"] = self.read_frequency()
            self.last["throttled_hex"] = self.read_throttled()

        return dict(self.last), now_ms() - start


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

FRAME_FIELDS = [
    "run_id",
    "video_name",
    "pipeline_version",
    "repeat_number",
    "frame_index",
    "cnn_interval",
    "buffer_size",
    "cnn_executed",
    "state_summary",
    "active_tracks",
    "predicted_mask_count",
    "visible_gt_targets",
    "protected_targets",
    "partially_protected_targets",
    "leaked_targets",
    "reappearance_events",
    "acquired_reappearances",
    "decode_ms",
    "resize_ms",
    "detection_ms",
    "tracking_ms",
    "association_ms",
    "coasting_ms",
    "masking_ms",
    "buffer_ms",
    "evaluation_ms",
    "telemetry_ms",
    "total_ms",
    "instantaneous_fps",
    "cpu_percent",
    "temperature_c",
    "cpu_frequency_mhz",
    "throttled_hex",
    "buffer_gate_open",
    "released_frame_index",
    "startup_frames_dropped_this_step",
]

TARGET_FIELDS = [
    "run_id",
    "frame_index",
    "track_id",
    "ground_truth_occlusion_state",
    "ground_truth_visibility",
    "x",
    "y",
    "width",
    "height",
    "coverage",
    "privacy_status",
    "reappearance_event",
    "reacquisition_delay_frames",
]

OUTPUT_FIELDS = [
    "run_id",
    "event_type",
    "processing_frame_index",
    "output_frame_index",
    "visible_target_count",
    "protected_target_count",
    "unprotected_target_count",
    "has_any_mask",
]

SUMMARY_FIELDS = [
    "run_id",
    "video_name",
    "video_path",
    "pipeline_version",
    "repeat_number",
    "cnn_interval",
    "buffer_size",
    "startup_gate",
    "processed_frames",
    "output_released_frames",
    "output_unprotected_frames",
    "output_unprotected_target_frames",
    "startup_dropped_frames",
    "total_processing_seconds",
    "mean_fps",
    "median_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "frames_within_40ms_pct",
    "mean_cpu_percent",
    "peak_cpu_percent",
    "initial_temperature_c",
    "mean_temperature_c",
    "peak_temperature_c",
    "throttling_samples",
    "visible_target_frames",
    "protected_target_frames",
    "partially_protected_target_frames",
    "leaked_target_frames",
    "target_leakage_rate_pct",
    "leak_event_count",
    "maximum_consecutive_leaked_target_frames",
    "mean_leak_event_duration_frames",
    "mean_reacquisition_delay_frames",
    "maximum_reacquisition_delay_frames",
    "resolved_reappearance_count",
    "unresolved_reappearance_count",
]


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def append_summary(
    path: Path,
    row: dict[str, Any],
) -> None:
    exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=SUMMARY_FIELDS,
            extrasaction="ignore",
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def create_output_writer(
    path: Path,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {path}")
    return writer


def display_standby(width: int, height: int) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        "PRIVACY INITIALIZATION",
        (max(10, width // 2 - 220), height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def main() -> int:
    args = parse_args()

    args.video = args.video.expanduser().resolve()
    args.ground_truth = args.ground_truth.expanduser().resolve()
    args.model_path = args.model_path.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.ground_truth.exists():
        raise FileNotFoundError(
            f"Ground-truth CSV not found: {args.ground_truth}"
        )

    video_name = args.video_name or args.video.stem
    run_id = args.run_id or (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + f"_{normalize_name(video_name)}"
        + f"_{args.pipeline_version}"
        + f"_r{args.repeat_number}"
    )

    ground_truth_by_frame, all_track_ids, gt_issues = load_ground_truth(
        args.ground_truth,
        video_name,
        args.strict_ground_truth,
    )

    if gt_issues:
        issue_path = args.output_dir / f"{run_id}_ground_truth_issues.txt"
        issue_path.write_text("\n".join(gt_issues), encoding="utf-8")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ok, sample_frame = capture.read()
    if not ok or sample_frame is None:
        raise RuntimeError("Video contains no readable frames.")

    detector = YuNetDetector(args)
    warm_up_detector(detector, sample_frame, args.warmup_runs)
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if args.pipeline_version == "v1":
        pipeline: Any = PipelineV1(detector, args)
    else:
        pipeline = PipelineV2(detector, args)

    masker = ReversibleMasker(frame_width, frame_height)
    evaluator = PrivacyEvaluator(
        ground_truth_by_frame,
        all_track_ids,
        args.protected_threshold,
        args.partial_threshold,
    )
    output_buffer = LookAheadBuffer(
        args.buffer_size,
        args.startup_gate,
    )
    telemetry = TelemetrySampler(args.telemetry_every)

    output_writer: Optional[cv2.VideoWriter] = None
    if args.save_output_video is not None:
        output_path = args.save_output_video.expanduser().resolve()
        output_writer = create_output_writer(
            output_path,
            fps,
            frame_width,
            frame_height,
        )

    standby_frame = (
        display_standby(frame_width, frame_height)
        if args.display
        else None
    )

    frame_rows: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    total_latencies: list[float] = []
    cpu_values: list[float] = []
    temperature_values: list[float] = []

    initial_temperature: Any = TelemetrySampler.read_temperature()
    throttling_samples = 0
    output_released_frames = 0
    output_unprotected_frames = 0
    output_unprotected_target_frames = 0
    processed_frames = 0

    print(
        f"Running {args.pipeline_version.upper()} on {args.video.name} "
        f"with CNN interval={args.cnn_interval}, "
        f"buffer={args.buffer_size}, display={args.display}"
    )

    try:
        frame_index = 0

        while True:
            if (
                args.stop_after_frames > 0
                and frame_index >= args.stop_after_frames
            ):
                break

            loop_start = now_ms()
            decode_start = now_ms()
            ok, frame = capture.read()
            decode_ms = now_ms() - decode_start

            if not ok or frame is None:
                break

            pipeline_result = pipeline.process(frame, frame_index)

            mask_start = now_ms()
            rendered_frame, privacy_mask = masker.render(
                frame,
                pipeline_result.boxes,
                args.mask_padding,
            )
            masking_ms = now_ms() - mask_start
            core_processing_ms = now_ms() - loop_start

            evaluation_start = now_ms()
            evaluation = evaluator.evaluate_frame(
                run_id,
                frame_index,
                privacy_mask,
            )
            evaluation_ms = now_ms() - evaluation_start

            buffer_start = now_ms()
            buffer_result = output_buffer.push(
                BufferItem(
                    frame_index=frame_index,
                    frame=(
                        rendered_frame.copy()
                        if args.display or output_writer is not None
                        else None
                    ),
                    has_any_mask=bool(pipeline_result.boxes),
                    visible_target_count=evaluation[
                        "visible_gt_targets"
                    ],
                    protected_target_count=evaluation[
                        "protected_targets"
                    ],
                )
            )
            buffer_ms = now_ms() - buffer_start

            released_frame_index: Any = ""
            for released in buffer_result.released:
                output_released_frames += 1
                released_frame_index = released.frame_index
                unprotected_target_count = max(
                    0,
                    released.visible_target_count
                    - released.protected_target_count,
                )
                if unprotected_target_count > 0:
                    output_unprotected_frames += 1
                    output_unprotected_target_frames += (
                        unprotected_target_count
                    )

                output_rows.append(
                    {
                        "run_id": run_id,
                        "event_type": "RELEASED",
                        "processing_frame_index": frame_index,
                        "output_frame_index": released.frame_index,
                        "visible_target_count": released.visible_target_count,
                        "protected_target_count": released.protected_target_count,
                        "unprotected_target_count": unprotected_target_count,
                        "has_any_mask": int(released.has_any_mask),
                    }
                )

                if released.frame is not None:
                    if output_writer is not None:
                        output_writer.write(released.frame)

                    if args.display:
                        cv2.imshow("Thesis Pipeline", released.frame)

            for dropped in buffer_result.dropped:
                output_rows.append(
                    {
                        "run_id": run_id,
                        "event_type": "STARTUP_DROPPED",
                        "processing_frame_index": frame_index,
                        "output_frame_index": dropped.frame_index,
                        "visible_target_count": dropped.visible_target_count,
                        "protected_target_count": dropped.protected_target_count,
                        "unprotected_target_count": max(
                            0,
                            dropped.visible_target_count
                            - dropped.protected_target_count,
                        ),
                        "has_any_mask": int(dropped.has_any_mask),
                    }
                )

            if args.display and not buffer_result.released:
                assert standby_frame is not None
                cv2.imshow("Thesis Pipeline", standby_frame)

            telemetry_values, telemetry_ms = telemetry.sample(frame_index)

            if telemetry_values["cpu_percent"] != "":
                cpu_values.append(
                    float(telemetry_values["cpu_percent"])
                )

            if telemetry_values["temperature_c"] != "":
                temperature_values.append(
                    float(telemetry_values["temperature_c"])
                )

            throttled_value = str(
                telemetry_values["throttled_hex"]
            ).strip().lower()

            if throttled_value not in {"", "0x0", "0"}:
                throttling_samples += 1

            # Ground-truth evaluation and telemetry are instrumentation, not
            # deployed pipeline work, so they are logged but excluded from FPS.
            total_ms = core_processing_ms + buffer_ms
            instantaneous_fps = (
                1000.0 / total_ms if total_ms > 0 else 0.0
            )

            pipeline_result.stage_times.decode_ms = decode_ms
            pipeline_result.stage_times.masking_ms = masking_ms
            pipeline_result.stage_times.buffer_ms = buffer_ms
            pipeline_result.stage_times.evaluation_ms = evaluation_ms
            pipeline_result.stage_times.telemetry_ms = telemetry_ms
            pipeline_result.stage_times.total_ms = total_ms

            latency_for_throughput = total_ms
            if not args.include_decode_in_throughput:
                latency_for_throughput = max(0.0, total_ms - decode_ms)

            total_latencies.append(latency_for_throughput)

            frame_rows.append(
                {
                    "run_id": run_id,
                    "video_name": video_name,
                    "pipeline_version": args.pipeline_version,
                    "repeat_number": args.repeat_number,
                    "frame_index": frame_index,
                    "cnn_interval": args.cnn_interval,
                    "buffer_size": args.buffer_size,
                    "cnn_executed": int(
                        pipeline_result.cnn_executed
                    ),
                    "state_summary": pipeline_result.state_summary,
                    "active_tracks": pipeline_result.active_tracks,
                    "predicted_mask_count": len(
                        pipeline_result.boxes
                    ),
                    **evaluation,
                    **asdict(pipeline_result.stage_times),
                    "instantaneous_fps": instantaneous_fps,
                    **telemetry_values,
                    "buffer_gate_open": int(
                        buffer_result.gate_open
                    ),
                    "released_frame_index": released_frame_index,
                    "startup_frames_dropped_this_step": len(
                        buffer_result.dropped
                    ),
                }
            )

            processed_frames += 1
            frame_index += 1

            if args.display:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

    finally:
        capture.release()

        for released in output_buffer.flush():
            output_released_frames += 1
            unprotected_target_count = max(
                0,
                released.visible_target_count
                - released.protected_target_count,
            )
            if unprotected_target_count > 0:
                output_unprotected_frames += 1
                output_unprotected_target_frames += (
                    unprotected_target_count
                )

            output_rows.append(
                {
                    "run_id": run_id,
                    "event_type": "FLUSHED",
                    "processing_frame_index": processed_frames,
                    "output_frame_index": released.frame_index,
                    "visible_target_count": released.visible_target_count,
                    "protected_target_count": released.protected_target_count,
                    "unprotected_target_count": unprotected_target_count,
                    "has_any_mask": int(released.has_any_mask),
                }
            )

            if released.frame is not None and output_writer is not None:
                output_writer.write(released.frame)

        if output_writer is not None:
            output_writer.release()

        if args.display:
            cv2.destroyAllWindows()

    privacy_summary = evaluator.finalize()
    total_processing_seconds = sum(total_latencies) / 1000.0
    mean_fps = (
        processed_frames / total_processing_seconds
        if total_processing_seconds > 0
        else 0.0
    )

    summary = {
        "run_id": run_id,
        "video_name": video_name,
        "video_path": str(args.video),
        "pipeline_version": args.pipeline_version,
        "repeat_number": args.repeat_number,
        "cnn_interval": args.cnn_interval,
        "buffer_size": args.buffer_size,
        "startup_gate": int(args.startup_gate),
        "processed_frames": processed_frames,
        "output_released_frames": output_released_frames,
        "output_unprotected_frames": output_unprotected_frames,
        "output_unprotected_target_frames": (
            output_unprotected_target_frames
        ),
        "startup_dropped_frames": (
            output_buffer.total_startup_dropped
        ),
        "total_processing_seconds": total_processing_seconds,
        "mean_fps": mean_fps,
        "median_latency_ms": percentile(total_latencies, 50),
        "p95_latency_ms": percentile(total_latencies, 95),
        "p99_latency_ms": percentile(total_latencies, 99),
        "frames_within_40ms_pct": (
            100.0
            * sum(value <= 40.0 for value in total_latencies)
            / len(total_latencies)
            if total_latencies
            else 0.0
        ),
        "mean_cpu_percent": safe_mean(cpu_values),
        "peak_cpu_percent": max(cpu_values) if cpu_values else 0.0,
        "initial_temperature_c": initial_temperature,
        "mean_temperature_c": safe_mean(temperature_values),
        "peak_temperature_c": (
            max(temperature_values)
            if temperature_values
            else 0.0
        ),
        "throttling_samples": throttling_samples,
        **privacy_summary,
    }

    frame_path = args.output_dir / f"{run_id}_frame_metrics.csv"
    target_path = args.output_dir / f"{run_id}_target_metrics.csv"
    output_path = args.output_dir / f"{run_id}_output_events.csv"
    summary_path = args.output_dir / "run_summary.csv"
    settings_path = args.output_dir / f"{run_id}_settings.json"

    write_csv(frame_path, FRAME_FIELDS, frame_rows)
    write_csv(target_path, TARGET_FIELDS, evaluator.target_rows)
    write_csv(output_path, OUTPUT_FIELDS, output_rows)
    append_summary(summary_path, summary)

    settings = {
        **vars(args),
        "video": str(args.video),
        "ground_truth": str(args.ground_truth),
        "model_path": str(args.model_path),
        "output_dir": str(args.output_dir),
        "run_id": run_id,
        "video_fps": fps,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "opencv_version": cv2.__version__,
        "python_version": sys.version,
    }
    settings_path.write_text(
        json.dumps(settings, indent=2, default=str),
        encoding="utf-8",
    )

    print("\nRun complete")
    print(f"  Run ID: {run_id}")
    print(f"  Processed frames: {processed_frames}")
    print(f"  Mean FPS: {mean_fps:.3f}")
    print(
        f"  Target leakage rate: "
        f"{privacy_summary['target_leakage_rate_pct']:.3f}%"
    )
    print(
        f"  Maximum consecutive leaked target-frames: "
        f"{privacy_summary['maximum_consecutive_leaked_target_frames']}"
    )
    print(
        f"  Mean reacquisition delay: "
        f"{privacy_summary['mean_reacquisition_delay_frames']:.3f} frames"
    )
    print(f"  Frame metrics: {frame_path}")
    print(f"  Target metrics: {target_path}")
    print(f"  Summary: {summary_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
        cv2.error,
    ) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
