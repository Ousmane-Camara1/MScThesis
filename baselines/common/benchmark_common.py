#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, statistics, subprocess, sys, time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Protocol

import cv2
import numpy as np

try:
    import psutil
except ImportError:
    psutil = None

BBox = tuple[int, int, int, int]

@dataclass
class DetectionResult:
    boxes: list[BBox]
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0

class DetectorProtocol(Protocol):
    name: str
    def warmup(self, frame: np.ndarray, runs: int) -> None: ...
    def detect(self, frame: np.ndarray) -> DetectionResult: ...

@dataclass
class GroundTruthTarget:
    frame_index: int
    track_id: int
    bbox: BBox
    visibility: float
    outside: bool
    occlusion_state: str

@dataclass
class TargetState:
    ever_visible: bool = False
    previously_visible: bool = False
    leak_streak: int = 0
    leak_event_count: int = 0
    leak_event_durations: list[int] = field(default_factory=list)
    max_leak_streak: int = 0
    pending_reappearance_frame: Optional[int] = None
    reacquisition_delays: list[int] = field(default_factory=list)
    unresolved_reappearances: int = 0

FRAME_FIELDS = [
    "run_id", "video_name", "model_name", "repeat_number", "frame_index",
    "detected_faces", "visible_gt_targets", "protected_targets",
    "partially_protected_targets", "leaked_targets", "reappearance_events",
    "acquired_reappearances", "decode_ms", "preprocess_ms", "inference_ms",
    "postprocess_ms", "masking_ms", "measured_total_ms", "instantaneous_fps",
    "cpu_percent", "temperature_c", "cpu_frequency_mhz", "throttled_hex",
]
TARGET_FIELDS = [
    "run_id", "video_name", "model_name", "frame_index", "track_id",
    "ground_truth_occlusion_state", "ground_truth_visibility", "x", "y",
    "width", "height", "coverage", "privacy_status", "reappearance_event",
    "reacquisition_delay_frames",
]
SUMMARY_FIELDS = [
    "run_id", "video_name", "video_path", "model_name", "repeat_number",
    "processed_frames", "include_decode_in_throughput", "total_measured_seconds",
    "mean_fps", "median_latency_ms", "p95_latency_ms", "p99_latency_ms",
    "frames_within_40ms_pct", "mean_preprocess_ms", "mean_inference_ms",
    "mean_postprocess_ms", "mean_masking_ms", "mean_cpu_percent",
    "peak_cpu_percent", "initial_temperature_c", "mean_temperature_c",
    "peak_temperature_c", "throttling_samples", "visible_target_frames",
    "protected_target_frames", "partially_protected_target_frames",
    "leaked_target_frames", "target_leakage_rate_pct", "leak_event_count",
    "maximum_consecutive_leaked_target_frames", "mean_leak_event_duration_frames",
    "mean_reacquisition_delay_frames", "maximum_reacquisition_delay_frames",
    "resolved_reappearance_count", "unresolved_reappearance_count",
]

def now_ms() -> float:
    return time.perf_counter_ns() / 1_000_000.0

def safe_mean(values) -> float:
    values = list(values)
    return float(statistics.fmean(values)) if values else 0.0

def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q)) if values else 0.0

def normalize_name(value: str) -> str:
    stem = Path(value).stem.lower()
    for suffix in ("_ground_truth", "_validated_mot", "_validated", "_mot"):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
    return "".join(ch for ch in stem if ch.isalnum())

def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}

def clamp_bbox(box: BBox, frame_width: int, frame_height: int) -> Optional[BBox]:
    x, y, w, h = box
    x1 = max(0, min(frame_width - 1, int(round(x))))
    y1 = max(0, min(frame_height - 1, int(round(y))))
    x2 = max(0, min(frame_width, int(round(x + w))))
    y2 = max(0, min(frame_height, int(round(y + h))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1

def load_ground_truth(csv_path: Path, video_name: str):
    required = {
        "video_name", "frame_index", "track_id", "x", "y", "width", "height",
        "visibility", "outside", "occlusion_state",
    }
    by_frame: dict[int, list[GroundTruthTarget]] = defaultdict(list)
    track_ids: set[int] = set()
    available: set[str] = set()
    selected = normalize_name(video_name)
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Ground-truth CSV has no header.")
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing ground-truth columns: {sorted(missing)}")
        for line, row in enumerate(reader, 2):
            available.add(row["video_name"])
            if normalize_name(row["video_name"]) != selected:
                continue
            try:
                outside = parse_bool(row["outside"])
                bbox = (0, 0, 0, 0) if outside else (
                    int(round(float(row["x"]))), int(round(float(row["y"]))),
                    int(round(float(row["width"]))), int(round(float(row["height"]))),
                )
                if not outside and (bbox[2] <= 0 or bbox[3] <= 0):
                    raise ValueError("non-positive box")
                target = GroundTruthTarget(
                    frame_index=int(float(row["frame_index"])),
                    track_id=int(float(row["track_id"])), bbox=bbox,
                    visibility=float(row["visibility"]), outside=outside,
                    occlusion_state=row["occlusion_state"].strip().upper(),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid ground-truth row {line}: {exc}") from exc
            by_frame[target.frame_index].append(target)
            track_ids.add(target.track_id)
    if not by_frame:
        raise ValueError(f"No ground truth matched {video_name!r}. Available: {sorted(available)}")
    return dict(by_frame), track_ids

def is_visible(target: GroundTruthTarget) -> bool:
    return (
        not target.outside
        and target.occlusion_state not in {"FULLY_OCCLUDED", "OUTSIDE"}
        and target.bbox[2] > 0 and target.bbox[3] > 0
    )

class ReversibleMasker:
    def __init__(self, width: int, height: int) -> None:
        rng = np.random.default_rng(42)
        self.key = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        self.cache: dict[tuple[int, int], np.ndarray] = {}
    def ellipse(self, width: int, height: int) -> np.ndarray:
        key = (width, height)
        if key not in self.cache:
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.ellipse(mask, (width // 2, height // 2),
                        (max(1, width // 2), max(1, height // 2)),
                        0, 0, 360, 255, -1)
            self.cache[key] = mask
        return self.cache[key]
    def render(self, frame: np.ndarray, boxes: list[BBox], padding: int):
        height, width = frame.shape[:2]
        union = np.zeros((height, width), dtype=np.uint8)
        for x, y, w, h in boxes:
            padded = clamp_bbox((x-padding, y-padding, w+2*padding, h+2*padding), width, height)
            if padded is None:
                continue
            x, y, w, h = padded
            roi = frame[y:y+h, x:x+w]
            mask = self.ellipse(w, h)
            encrypted = cv2.bitwise_xor(roi, self.key[:h, :w])
            cv2.copyTo(encrypted, mask, roi)
            cv2.bitwise_or(union[y:y+h, x:x+w], mask, dst=union[y:y+h, x:x+w])
        return frame, union

class PrivacyEvaluator:
    def __init__(self, gt, track_ids, protected: float, partial: float) -> None:
        self.gt = gt
        self.protected_threshold = protected
        self.partial_threshold = partial
        self.states = {track_id: TargetState() for track_id in track_ids}
        self.target_rows: list[dict[str, Any]] = []
        self.visible = self.protected = self.partial = self.leaked = 0
    @staticmethod
    def coverage(mask: np.ndarray, bbox: BBox) -> float:
        clamped = clamp_bbox(bbox, mask.shape[1], mask.shape[0])
        if clamped is None:
            return 0.0
        x, y, w, h = clamped
        roi = mask[y:y+h, x:x+w]
        return float(np.count_nonzero(roi)) / float(w*h) if roi.size else 0.0
    def classify(self, coverage: float) -> str:
        if coverage >= self.protected_threshold:
            return "PROTECTED"
        if coverage >= self.partial_threshold:
            return "PARTIALLY_PROTECTED"
        return "LEAKED"
    @staticmethod
    def close_event(state: TargetState) -> None:
        if state.leak_streak:
            state.leak_event_durations.append(state.leak_streak)
            state.leak_streak = 0
    def evaluate(self, run_id, video_name, model_name, frame_index, mask):
        targets = [t for t in self.gt.get(frame_index, []) if is_visible(t)]
        visible_ids = {t.track_id for t in targets}
        for track_id, state in self.states.items():
            if track_id not in visible_ids:
                self.close_event(state)
                state.previously_visible = False
        counts = {"visible_gt_targets": len(targets), "protected_targets": 0,
                  "partially_protected_targets": 0, "leaked_targets": 0,
                  "reappearance_events": 0, "acquired_reappearances": 0}
        for target in targets:
            state = self.states[target.track_id]
            coverage = self.coverage(mask, target.bbox)
            status = self.classify(coverage)
            reappearance = state.ever_visible and not state.previously_visible
            delay: Optional[int] = None
            if reappearance:
                counts["reappearance_events"] += 1
                if status == "PROTECTED":
                    delay = 0
                    state.reacquisition_delays.append(0)
                    counts["acquired_reappearances"] += 1
                else:
                    state.pending_reappearance_frame = frame_index
            if state.pending_reappearance_frame is not None and status == "PROTECTED":
                delay = frame_index - state.pending_reappearance_frame
                state.reacquisition_delays.append(delay)
                state.pending_reappearance_frame = None
                counts["acquired_reappearances"] += 1
            if status == "LEAKED":
                counts["leaked_targets"] += 1; self.leaked += 1
                if state.leak_streak == 0:
                    state.leak_event_count += 1
                state.leak_streak += 1
                state.max_leak_streak = max(state.max_leak_streak, state.leak_streak)
            else:
                self.close_event(state)
                if status == "PROTECTED":
                    counts["protected_targets"] += 1; self.protected += 1
                else:
                    counts["partially_protected_targets"] += 1; self.partial += 1
            self.visible += 1
            state.ever_visible = state.previously_visible = True
            self.target_rows.append({
                "run_id": run_id, "video_name": video_name, "model_name": model_name,
                "frame_index": frame_index, "track_id": target.track_id,
                "ground_truth_occlusion_state": target.occlusion_state,
                "ground_truth_visibility": target.visibility,
                "x": target.bbox[0], "y": target.bbox[1], "width": target.bbox[2],
                "height": target.bbox[3], "coverage": round(coverage, 6),
                "privacy_status": status, "reappearance_event": int(reappearance),
                "reacquisition_delay_frames": "" if delay is None else delay,
            })
        return counts
    def finalize(self):
        durations, delays = [], []
        events = max_streak = unresolved = 0
        for state in self.states.values():
            self.close_event(state)
            if state.pending_reappearance_frame is not None:
                state.unresolved_reappearances += 1
            durations.extend(state.leak_event_durations)
            delays.extend(state.reacquisition_delays)
            events += state.leak_event_count
            max_streak = max(max_streak, state.max_leak_streak)
            unresolved += state.unresolved_reappearances
        return {
            "visible_target_frames": self.visible,
            "protected_target_frames": self.protected,
            "partially_protected_target_frames": self.partial,
            "leaked_target_frames": self.leaked,
            "target_leakage_rate_pct": 100*self.leaked/self.visible if self.visible else 0.0,
            "leak_event_count": events,
            "maximum_consecutive_leaked_target_frames": max_streak,
            "mean_leak_event_duration_frames": safe_mean(durations),
            "mean_reacquisition_delay_frames": safe_mean(delays),
            "maximum_reacquisition_delay_frames": max(delays) if delays else 0,
            "resolved_reappearance_count": len(delays),
            "unresolved_reappearance_count": unresolved,
        }

class TelemetrySampler:
    def __init__(self, every: int) -> None:
        self.every = every
        self.last = {"cpu_percent": "", "temperature_c": "", "cpu_frequency_mhz": "", "throttled_hex": ""}
        if psutil is not None:
            psutil.cpu_percent(None)
    @staticmethod
    def temperature():
        for path in (Path("/sys/class/thermal/thermal_zone0/temp"), Path("/sys/class/thermal/thermal_zone1/temp")):
            try:
                raw = float(path.read_text().strip()); return raw/1000 if raw > 1000 else raw
            except (OSError, ValueError):
                pass
        try:
            result = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True, text=True, timeout=1)
            return float(result.stdout.split("=",1)[1].replace("'C", "").strip())
        except Exception:
            return ""
    @staticmethod
    def frequency():
        if psutil is not None:
            try:
                value = psutil.cpu_freq(); return value.current if value else ""
            except Exception:
                pass
        try:
            return float(Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq").read_text())/1000
        except Exception:
            return ""
    @staticmethod
    def throttled():
        try:
            result = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=1)
            return result.stdout.strip().split("=",1)[1]
        except Exception:
            return ""
    def sample(self, frame_index: int):
        if frame_index % self.every == 0:
            if psutil is not None:
                try: self.last["cpu_percent"] = psutil.cpu_percent(None)
                except Exception: pass
            self.last["temperature_c"] = self.temperature()
            self.last["cpu_frequency_mhz"] = self.frequency()
            self.last["throttled_hex"] = self.throttled()
        return dict(self.last)

def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)

def append_summary(path: Path, row: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        if not exists: writer.writeheader()
        writer.writerow(row)

def create_common_parser(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video", default="all")
    parser.add_argument("--repeat-number", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--mask-padding", type=int, default=30)
    parser.add_argument("--protected-threshold", type=float, default=0.90)
    parser.add_argument("--partial-threshold", type=float, default=0.50)
    parser.add_argument("--include-decode-in-throughput", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--display", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-output-videos", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--telemetry-every", type=int, default=30)
    parser.add_argument("--stop-after-frames", type=int, default=0)
    return parser

def discover_videos(clips_dir: Path, selection: str):
    if selection.lower() == "all":
        videos = sorted(clips_dir.glob("*.mp4"))
    else:
        candidate = clips_dir / (selection if Path(selection).suffix else f"{selection}.mp4")
        videos = [candidate]
    missing = [path for path in videos if not path.exists()]
    if not videos or missing:
        raise FileNotFoundError(f"Missing videos: {missing or clips_dir}")
    return videos

def draw_boxes(frame, boxes, name):
    for x, y, w, h in boxes:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.putText(frame, name, (12,30), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 2)

def run_single_video(detector: DetectorProtocol, args, video_path: Path):
    video_name = video_path.stem
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{normalize_name(video_name)}_{normalize_name(detector.name)}_r{args.repeat_number}"
    gt, track_ids = load_ground_truth(args.ground_truth, video_name)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise RuntimeError(f"Cannot open {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)); width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, sample = cap.read()
    if not ok: raise RuntimeError(f"No frames in {video_path}")
    detector.warmup(sample, args.warmup_runs); cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    masker = ReversibleMasker(width, height)
    evaluator = PrivacyEvaluator(gt, track_ids, args.protected_threshold, args.partial_threshold)
    telemetry = TelemetrySampler(args.telemetry_every)
    writer = None
    if args.save_output_videos:
        out = args.output_dir / detector.name / "output_videos" / f"{video_name}_{detector.name}_r{args.repeat_number}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height))
    initial_temp = TelemetrySampler.temperature()
    rows, latencies, prep, infer, post, mask_values, cpu_values, temps = [], [], [], [], [], [], [], []
    throttling_samples = frame_index = 0
    try:
        while True:
            if args.stop_after_frames and frame_index >= args.stop_after_frames: break
            start = now_ms(); ok, frame = cap.read(); decode_ms = now_ms() - start
            if not ok: break
            result = detector.detect(frame)
            boxes = [c for b in result.boxes if (c := clamp_bbox(b, width, height)) is not None]
            start = now_ms(); rendered, privacy_mask = masker.render(frame, boxes, args.mask_padding); masking_ms = now_ms() - start
            measured = result.preprocess_ms + result.inference_ms + result.postprocess_ms + masking_ms + (decode_ms if args.include_decode_in_throughput else 0)
            evaluation = evaluator.evaluate(run_id, video_name, detector.name, frame_index, privacy_mask)
            telem = telemetry.sample(frame_index)
            if telem["cpu_percent"] != "": cpu_values.append(float(telem["cpu_percent"]))
            if telem["temperature_c"] != "": temps.append(float(telem["temperature_c"]))
            if str(telem["throttled_hex"]).strip().lower() not in {"", "0", "0x0"}: throttling_samples += 1
            latencies.append(measured); prep.append(result.preprocess_ms); infer.append(result.inference_ms); post.append(result.postprocess_ms); mask_values.append(masking_ms)
            rows.append({"run_id": run_id, "video_name": video_name, "model_name": detector.name, "repeat_number": args.repeat_number,
                         "frame_index": frame_index, "detected_faces": len(boxes), **evaluation, "decode_ms": decode_ms,
                         "preprocess_ms": result.preprocess_ms, "inference_ms": result.inference_ms,
                         "postprocess_ms": result.postprocess_ms, "masking_ms": masking_ms, "measured_total_ms": measured,
                         "instantaneous_fps": 1000/measured if measured > 0 else 0, **telem})
            if args.display or writer is not None: draw_boxes(rendered, boxes, detector.name)
            if writer is not None: writer.write(rendered)
            if args.display:
                cv2.imshow(f"{detector.name} Benchmark", rendered)
                if cv2.waitKey(1) & 0xFF == ord("q"): break
            frame_index += 1
    finally:
        cap.release()
        if writer is not None: writer.release()
        if args.display: cv2.destroyAllWindows()
    privacy = evaluator.finalize(); total_seconds = sum(latencies)/1000
    summary = {
        "run_id": run_id, "video_name": video_name, "video_path": str(video_path), "model_name": detector.name,
        "repeat_number": args.repeat_number, "processed_frames": frame_index,
        "include_decode_in_throughput": int(args.include_decode_in_throughput), "total_measured_seconds": total_seconds,
        "mean_fps": frame_index/total_seconds if total_seconds else 0, "median_latency_ms": percentile(latencies,50),
        "p95_latency_ms": percentile(latencies,95), "p99_latency_ms": percentile(latencies,99),
        "frames_within_40ms_pct": 100*sum(v<=40 for v in latencies)/len(latencies) if latencies else 0,
        "mean_preprocess_ms": safe_mean(prep), "mean_inference_ms": safe_mean(infer),
        "mean_postprocess_ms": safe_mean(post), "mean_masking_ms": safe_mean(mask_values),
        "mean_cpu_percent": safe_mean(cpu_values), "peak_cpu_percent": max(cpu_values) if cpu_values else 0,
        "initial_temperature_c": initial_temp, "mean_temperature_c": safe_mean(temps),
        "peak_temperature_c": max(temps) if temps else 0, "throttling_samples": throttling_samples, **privacy,
    }
    base = args.output_dir / detector.name
    write_csv(base/"frame_metrics"/f"{run_id}.csv", FRAME_FIELDS, rows)
    write_csv(base/"target_metrics"/f"{run_id}.csv", TARGET_FIELDS, evaluator.target_rows)
    append_summary(args.output_dir/"baseline_run_summary.csv", summary)
    settings = {**vars(args), "clips_dir": str(args.clips_dir), "ground_truth": str(args.ground_truth), "output_dir": str(args.output_dir),
                "video_path": str(video_path), "video_fps": fps, "frame_width": width, "frame_height": height,
                "model_name": detector.name, "opencv_version": cv2.__version__, "python_version": sys.version}
    settings_path = base/"settings"/f"{run_id}.json"; settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=2, default=str), encoding="utf-8")
    print(f"[{detector.name}] {video_name}: {summary['mean_fps']:.3f} FPS, {privacy['target_leakage_rate_pct']:.3f}% leakage")

def run_benchmark(detector: DetectorProtocol, args):
    args.clips_dir = args.clips_dir.expanduser().resolve(); args.ground_truth = args.ground_truth.expanduser().resolve(); args.output_dir = args.output_dir.expanduser().resolve()
    if not args.clips_dir.exists(): raise FileNotFoundError(args.clips_dir)
    if not args.ground_truth.exists(): raise FileNotFoundError(args.ground_truth)
    if not 0 <= args.partial_threshold < args.protected_threshold <= 1: raise ValueError("Invalid privacy thresholds")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for video in discover_videos(args.clips_dir, args.video):
        run_single_video(detector, args, video)
