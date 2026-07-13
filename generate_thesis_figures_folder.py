#!/usr/bin/env python3
"""Generate Chapter 5 figures and tables from a smart_cam results folder.

Place this file in the smart_cam project root and run:

    python generate_thesis_figures.py --clean

Defaults:
    results directory: ./results
    output directory:  ./thesis_results_outputs

The script discovers both descriptive experiment folders, such as
``experiment_1_baselines``, and compact folders, such as ``e1``. It uses
frame-level, target-level, telemetry, and output-event CSV files whenever they
exist. When only run summaries exist, it generates the figures supported by
those summaries and documents the missing raw data in the quality report.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REAL_TIME_FPS = 25.0
INPUT_FPS = 30.0
DEFAULT_TELEMETRY_EVERY = 30

BASELINE_NAME_MAP = {
    "HaarCascade": "Haar Cascade",
    "Haar Cascade": "Haar Cascade",
    "MTCNN": "MTCNN",
    "UltraFace": "UltraFace",
    "BlazeFace": "BlazeFace",
    "MediaPipe": "BlazeFace",
    "PureYuNet": "Pure YuNet",
    "Pure YuNet": "Pure YuNet",
}

E1_ORDER = [
    "Haar Cascade",
    "MTCNN",
    "UltraFace",
    "BlazeFace",
    "Pure YuNet",
    "Pipeline V2",
]

E2_ORDER = ["Pipeline V1", "Pipeline V2"]

E4_ORDER = ["Pure YuNet", "BlazeFace", "Pipeline V1", "Pipeline V2"]

E5_ORDER = [
    "A: Pure YuNet",
    "B: Tracker only",
    "C: FSM, no expansion",
    "D: Full Pipeline V2",
]

NUMERIC_COLUMNS = {
    "repeat_number",
    "processed_frames",
    "include_decode_in_throughput",
    "cnn_interval",
    "buffer_size",
    "startup_gate",
    "frame_index",
    "output_released_frames",
    "output_unprotected_frames",
    "output_unprotected_target_frames",
    "startup_dropped_frames",
    "total_measured_seconds",
    "total_processing_seconds",
    "mean_fps",
    "median_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "frames_within_40ms_pct",
    "mean_preprocess_ms",
    "mean_inference_ms",
    "mean_postprocess_ms",
    "mean_masking_ms",
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
    "detected_faces",
    "visible_gt_targets",
    "protected_targets",
    "partially_protected_targets",
    "leaked_targets",
    "reappearance_events",
    "acquired_reappearances",
    "decode_ms",
    "preprocess_ms",
    "inference_ms",
    "postprocess_ms",
    "resize_ms",
    "detection_ms",
    "tracking_ms",
    "association_ms",
    "coasting_ms",
    "masking_ms",
    "buffer_ms",
    "evaluation_ms",
    "telemetry_ms",
    "measured_total_ms",
    "total_ms",
    "instantaneous_fps",
    "cpu_percent",
    "temperature_c",
    "cpu_frequency_mhz",
    "cnn_executed",
    "active_tracks",
    "predicted_mask_count",
    "buffer_gate_open",
    "released_frame_index",
    "startup_frames_dropped_this_step",
    "coverage",
    "ground_truth_visibility",
    "reacquisition_delay_frames",
    "processing_frame_index",
    "output_frame_index",
    "visible_target_count",
    "protected_target_count",
    "unprotected_target_count",
    "has_any_mask",
}


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    figures: Path
    tables: Path
    reports: Path


@dataclass
class BuildLog:
    figures: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def add_figure(self, experiment: str, stem: str, description: str, source: str) -> None:
        self.figures.append(
            {
                "experiment": experiment,
                "figure_stem": stem,
                "description": description,
                "data_source": source,
            }
        )

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"[warning] {message}")

    def fail(self, message: str) -> None:
        self.failures.append(message)
        print(f"[error] {message}", file=sys.stderr)


@dataclass(frozen=True)
class ExperimentRoots:
    e1: tuple[Path, ...]
    e2: tuple[Path, ...]
    e3: tuple[Path, ...]
    e3_confirmation: tuple[Path, ...]
    e4: tuple[Path, ...]
    e5: tuple[Path, ...]
    e6: tuple[Path, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate thesis figures from smart_cam/results."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Results directory. Default: ./results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("thesis_results_outputs"),
        help="Output directory. Default: ./thesis_results_outputs",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("pdf", "png"),
        default=("pdf", "png"),
        help="Figure formats. Default: pdf png",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the existing output directory before generating figures.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution. Default: 300",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop immediately when an experiment cannot be processed.",
    )
    return parser.parse_args()


def prepare_output(path: Path, clean: bool) -> OutputPaths:
    root = path.expanduser().resolve()
    if clean and root.exists():
        shutil.rmtree(root)
    figures = root / "figures"
    tables = root / "tables"
    reports = root / "reports"
    for directory in (root, figures, tables, reports):
        directory.mkdir(parents=True, exist_ok=True)
    return OutputPaths(root, figures, tables, reports)


def unique_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    seen: set[Path] = set()
    output: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen and resolved.exists():
            seen.add(resolved)
            output.append(resolved)
    return tuple(output)


def direct_or_recursive(results_root: Path, names: Sequence[str]) -> tuple[Path, ...]:
    found: list[Path] = []
    for name in names:
        direct = results_root / name
        if direct.is_dir():
            found.append(direct)
    if found:
        return unique_paths(found)

    lowered = {name.lower() for name in names}
    for path in results_root.rglob("*"):
        if path.is_dir() and path.name.lower() in lowered:
            found.append(path)
    return unique_paths(found)


def resolve_experiment_roots(results_root: Path) -> ExperimentRoots:
    root = results_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(
            f"Results directory not found: {root}\n"
            "Run the script from the smart_cam root or pass --results-dir explicitly."
        )

    e1 = direct_or_recursive(root, ("experiment_1_baselines", "e1"))
    e2 = direct_or_recursive(root, ("experiment_2_v1_vs_v2", "e2"))
    e3 = direct_or_recursive(root, ("experiment_3_cnn_interval", "e3"))
    e3_confirmation = direct_or_recursive(
        root,
        (
            "experiment_3_cnn_interval_confirmation",
            "e3_confirmation",
            "experiment_3_confirmation",
        ),
    )
    e4 = direct_or_recursive(root, ("experiment_4_thermal", "e4"))
    e5 = direct_or_recursive(root, ("experiment_5_privacy", "e5"))
    e6 = direct_or_recursive(root, ("experiment_6_startup_buffer", "e6"))

    return ExperimentRoots(e1, e2, e3, e3_confirmation, e4, e5, e6)


def all_csv_files(roots: Sequence[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        files.extend(root.rglob("*.csv"))
    return sorted(set(path.resolve() for path in files))


def find_csvs(
    roots: Sequence[Path],
    predicate: Callable[[Path], bool],
) -> list[Path]:
    return [path for path in all_csv_files(roots) if predicate(path)]


def is_baseline_summary(path: Path) -> bool:
    return path.name == "baseline_run_summary.csv"


def is_pipeline_summary(path: Path) -> bool:
    return path.name.startswith("run_summary") and path.suffix.lower() == ".csv"


def is_frame_metrics(path: Path) -> bool:
    lower = path.name.lower()
    return "frame_metrics" in lower or path.parent.name.lower() == "frame_metrics"


def is_target_metrics(path: Path) -> bool:
    lower = path.name.lower()
    return "target_metrics" in lower or path.parent.name.lower() == "target_metrics"


def is_output_events(path: Path) -> bool:
    lower = path.name.lower()
    return "output_events" in lower or path.parent.name.lower() == "output_events"


def read_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    for column in df.columns:
        if column in NUMERIC_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["_source_file"] = str(path)
    return df


def load_many(paths: Sequence[Path], dedupe: Sequence[str] | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        df = read_csv(path)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if dedupe:
        columns = [column for column in dedupe if column in combined.columns]
        if columns:
            combined = combined.drop_duplicates(subset=columns, keep="last")
    return combined.reset_index(drop=True)


def natural_video_key(value: object) -> tuple[int, str]:
    text = str(value)
    match = re.search(r"(\d+)", text)
    return (int(match.group(1)) if match else 10**9, text)


def ordered_videos(values: Iterable[object]) -> list[str]:
    return sorted({str(value) for value in values if pd.notna(value)}, key=natural_video_key)


def architecture_label(row: pd.Series) -> str:
    model = str(row.get("model_name", "")).strip()
    if model and model.lower() != "nan":
        return BASELINE_NAME_MAP.get(model, model)
    pipeline = str(row.get("pipeline_version", "")).strip().lower()
    if pipeline in {"v1", "1"}:
        return "Pipeline V1"
    if pipeline in {"v2", "2"}:
        return "Pipeline V2"
    return "Unknown"


def attach_architecture(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    output = df.copy()
    output["architecture"] = output.apply(architecture_label, axis=1)
    return output


def e5_configuration_from_run_id(run_id: object, model_name: object = "") -> str:
    text = str(run_id)
    model = str(model_name)
    if model in {"PureYuNet", "Pure YuNet"} or "pureyunet" in text.lower():
        return "A: Pure YuNet"
    if re.search(r"exp5[_-]?B", text, flags=re.IGNORECASE) or "tracker_only" in text.lower():
        return "B: Tracker only"
    if re.search(r"exp5[_-]?C", text, flags=re.IGNORECASE) or "no_expansion" in text.lower():
        return "C: FSM, no expansion"
    if re.search(r"exp5[_-]?D", text, flags=re.IGNORECASE) or "full_v2" in text.lower():
        return "D: Full Pipeline V2"
    return "Unclassified"


def attach_e5_configuration(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    output = df.copy()
    output["configuration"] = [
        e5_configuration_from_run_id(run_id, model)
        for run_id, model in zip(
            output.get("run_id", pd.Series("", index=output.index)),
            output.get("model_name", pd.Series("", index=output.index)),
        )
    ]
    return output


def weighted_leakage(group: pd.DataFrame) -> float:
    if "visible_target_frames" not in group or "leaked_target_frames" not in group:
        return float("nan")
    visible = pd.to_numeric(group["visible_target_frames"], errors="coerce").sum(min_count=1)
    leaked = pd.to_numeric(group["leaked_target_frames"], errors="coerce").sum(min_count=1)
    if pd.isna(visible) or visible <= 0:
        return float("nan")
    return float(100.0 * leaked / visible)


def safe_mean(group: pd.DataFrame, column: str) -> float:
    if column not in group:
        return float("nan")
    return float(pd.to_numeric(group[column], errors="coerce").mean())


def safe_std(group: pd.DataFrame, column: str) -> float:
    if column not in group:
        return float("nan")
    return float(pd.to_numeric(group[column], errors="coerce").std(ddof=1))


def safe_sum(group: pd.DataFrame, column: str) -> float:
    if column not in group:
        return float("nan")
    return float(pd.to_numeric(group[column], errors="coerce").sum(min_count=1))


def safe_max(group: pd.DataFrame, column: str) -> float:
    if column not in group:
        return float("nan")
    series = pd.to_numeric(group[column], errors="coerce")
    return float(series.max()) if series.notna().any() else float("nan")


def summarize(df: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    records: list[dict[str, object]] = []
    for group_key, group in df.groupby(list(keys), dropna=False, sort=False):
        values = group_key if isinstance(group_key, tuple) else (group_key,)
        record: dict[str, object] = dict(zip(keys, values))
        record.update(
            {
                "runs": int(len(group)),
                "mean_fps": safe_mean(group, "mean_fps"),
                "fps_sd": safe_std(group, "mean_fps"),
                "median_latency_ms": safe_mean(group, "median_latency_ms"),
                "p95_latency_ms": safe_mean(group, "p95_latency_ms"),
                "p99_latency_ms": safe_mean(group, "p99_latency_ms"),
                "frames_within_40ms_pct": safe_mean(group, "frames_within_40ms_pct"),
                "mean_cpu_percent": safe_mean(group, "mean_cpu_percent"),
                "peak_cpu_percent": safe_max(group, "peak_cpu_percent"),
                "initial_temperature_c": safe_mean(group, "initial_temperature_c"),
                "mean_temperature_c": safe_mean(group, "mean_temperature_c"),
                "peak_temperature_c": safe_max(group, "peak_temperature_c"),
                "throttling_samples": safe_sum(group, "throttling_samples"),
                "visible_target_frames": safe_sum(group, "visible_target_frames"),
                "protected_target_frames": safe_sum(group, "protected_target_frames"),
                "partially_protected_target_frames": safe_sum(
                    group, "partially_protected_target_frames"
                ),
                "leaked_target_frames": safe_sum(group, "leaked_target_frames"),
                "target_leakage_rate_pct": weighted_leakage(group),
                "leak_event_count": safe_sum(group, "leak_event_count"),
                "mean_max_consecutive_leak_frames": safe_mean(
                    group, "maximum_consecutive_leaked_target_frames"
                ),
                "maximum_consecutive_leak_frames": safe_max(
                    group, "maximum_consecutive_leaked_target_frames"
                ),
                "mean_leak_event_duration_frames": safe_mean(
                    group, "mean_leak_event_duration_frames"
                ),
                "mean_reacquisition_delay_frames": safe_mean(
                    group, "mean_reacquisition_delay_frames"
                ),
                "maximum_reacquisition_delay_frames": safe_max(
                    group, "maximum_reacquisition_delay_frames"
                ),
                "resolved_reappearance_count": safe_sum(
                    group, "resolved_reappearance_count"
                ),
                "unresolved_reappearance_count": safe_sum(
                    group, "unresolved_reappearance_count"
                ),
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def save_table(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    cleaned = df.drop(columns=["_source_file"], errors="ignore")
    cleaned.to_csv(path, index=False, float_format="%.6f")


def save_json(data: dict[str, object], path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, default=json_default), encoding="utf-8")


def json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def save_figure(
    fig: plt.Figure,
    stem: str,
    outputs: OutputPaths,
    formats: Sequence[str],
    dpi: int,
    log: BuildLog,
    experiment: str,
    description: str,
    source: str,
) -> None:
    fig.tight_layout()
    for extension in formats:
        fig.savefig(
            outputs.figures / f"{stem}.{extension}",
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(fig)
    log.add_figure(experiment, stem, description, source)


def bar_labels(ax: plt.Axes, decimals: int = 1) -> None:
    for container in ax.containers:
        try:
            ax.bar_label(container, fmt=f"%.{decimals}f", padding=2, fontsize=8)
        except (AttributeError, ValueError, TypeError):
            continue


def ordered_summary(df: pd.DataFrame, column: str, order: Sequence[str]) -> pd.DataFrame:
    output = df.copy()
    output[column] = pd.Categorical(output[column], categories=order, ordered=True)
    return output.sort_values(column).reset_index(drop=True)


def latency_column(df: pd.DataFrame) -> str | None:
    for column in ("measured_total_ms", "total_ms"):
        if column in df.columns:
            return column
    return None


def sample_per_group(
    df: pd.DataFrame,
    group_column: str,
    max_per_group: int = 6000,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, group in df.groupby(group_column, sort=False):
        if len(group) > max_per_group:
            group = group.sample(max_per_group, random_state=42)
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def add_elapsed_time(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    column = latency_column(output)
    if column is None:
        if "frame_index" in output:
            output["elapsed_seconds"] = output["frame_index"] / INPUT_FPS
        return output
    output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0.0)
    output = output.sort_values("frame_index")
    output["elapsed_seconds"] = output[column].cumsum() / 1000.0
    return output


def telemetry_points(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "frame_index" not in df:
        return df
    selected = df[pd.to_numeric(df["frame_index"], errors="coerce") % DEFAULT_TELEMETRY_EVERY == 0]
    return selected if not selected.empty else df


def run_guarded(
    name: str,
    function: Callable[[], None],
    log: BuildLog,
    strict: bool,
) -> None:
    print(f"\n[{name}]")
    try:
        function()
    except Exception as exc:  # noqa: BLE001 - report and continue by design
        message = f"{name} failed: {exc}"
        log.fail(message)
        if strict:
            raise
        traceback.print_exc()


def load_experiment_summaries(roots: Sequence[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = load_many(find_csvs(roots, is_baseline_summary), dedupe=("run_id",))
    pipeline = load_many(find_csvs(roots, is_pipeline_summary), dedupe=("run_id",))
    return baseline, pipeline


def load_raw(roots: Sequence[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = load_many(
        find_csvs(roots, is_frame_metrics),
        dedupe=("run_id", "frame_index"),
    )
    targets = load_many(
        find_csvs(roots, is_target_metrics),
        dedupe=("run_id", "frame_index", "track_id"),
    )
    outputs = load_many(
        find_csvs(roots, is_output_events),
        dedupe=("run_id", "event_type", "processing_frame_index", "output_frame_index"),
    )
    return frames, targets, outputs


def experiment_1(
    roots: Sequence[Path],
    outputs: OutputPaths,
    formats: Sequence[str],
    dpi: int,
    log: BuildLog,
) -> dict[str, object]:
    if not roots:
        raise FileNotFoundError("Experiment 1 folder was not found.")

    baseline, pipeline = load_experiment_summaries(roots)
    if baseline.empty and pipeline.empty:
        raise FileNotFoundError("Experiment 1 summaries were not found.")

    combined = pd.concat(
        [attach_architecture(baseline), attach_architecture(pipeline)],
        ignore_index=True,
        sort=False,
    )
    combined = combined[combined["architecture"] != "Unknown"]

    overall = ordered_summary(summarize(combined, ["architecture"]), "architecture", E1_ORDER)
    by_video = summarize(combined, ["video_name", "architecture"])
    by_video["video_number"] = by_video["video_name"].map(lambda value: natural_video_key(value)[0])
    by_video["architecture"] = pd.Categorical(
        by_video["architecture"], categories=E1_ORDER, ordered=True
    )
    by_video = by_video.sort_values(["video_number", "architecture"]).drop(columns="video_number")

    save_table(overall, outputs.tables / "e1_overall_architecture_summary.csv")
    save_table(by_video, outputs.tables / "e1_per_video_architecture_summary.csv")

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.bar(
        overall["architecture"].astype(str),
        overall["mean_fps"],
        yerr=overall["fps_sd"].fillna(0),
        capsize=4,
    )
    ax.axhline(REAL_TIME_FPS, linestyle="--", linewidth=1.2, label="25 FPS threshold")
    ax.set_xlabel("Architecture")
    ax.set_ylabel("Mean throughput (FPS)")
    ax.set_title("Experiment 1: Overall processing throughput")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    bar_labels(ax, 1)
    save_figure(
        fig,
        "e1_overall_mean_fps",
        outputs,
        formats,
        dpi,
        log,
        "E1",
        "Overall mean FPS with run-to-run standard deviation.",
        "run summaries",
    )

    videos = ordered_videos(by_video["video_name"])
    x = np.arange(len(videos))
    width = 0.125
    fig, ax = plt.subplots(figsize=(15.5, 7.0))
    for index, architecture in enumerate(E1_ORDER):
        subset = by_video[by_video["architecture"].astype(str) == architecture].set_index(
            "video_name"
        )
        values = [
            subset.loc[video, "mean_fps"] if video in subset.index else np.nan
            for video in videos
        ]
        ax.bar(
            x + (index - (len(E1_ORDER) - 1) / 2) * width,
            values,
            width,
            label=architecture,
        )
    ax.axhline(REAL_TIME_FPS, linestyle="--", linewidth=1.2, label="25 FPS threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(videos)
    ax.set_xlabel("Video sequence")
    ax.set_ylabel("Mean throughput (FPS)")
    ax.set_title("Experiment 1: Throughput across the evaluation dataset")
    ax.legend(ncol=3, fontsize=9)
    save_figure(
        fig,
        "e1_fps_all_sequences",
        outputs,
        formats,
        dpi,
        log,
        "E1",
        "Per-video mean FPS for all baseline architectures and Pipeline V2.",
        "run summaries",
    )

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.bar(overall["architecture"].astype(str), overall["target_leakage_rate_pct"])
    ax.set_xlabel("Architecture")
    ax.set_ylabel("Target leakage rate (%)")
    ax.set_title("Experiment 1: Aggregate target-level privacy leakage")
    ax.tick_params(axis="x", rotation=25)
    bar_labels(ax, 2)
    save_figure(
        fig,
        "e1_overall_target_leakage",
        outputs,
        formats,
        dpi,
        log,
        "E1",
        "Target-frame-weighted leakage rate for all architectures.",
        "run summaries",
    )

    frames, _, _ = load_raw(roots)
    if not frames.empty:
        frames = attach_architecture(frames)
        latency = latency_column(frames)
        if latency:
            frames = frames[frames["architecture"].isin(E1_ORDER)]
            frames = sample_per_group(frames.dropna(subset=[latency]), "architecture")
            data = [
                frames.loc[frames["architecture"] == architecture, latency].to_numpy()
                for architecture in E1_ORDER
                if (frames["architecture"] == architecture).any()
            ]
            labels = [
                architecture
                for architecture in E1_ORDER
                if (frames["architecture"] == architecture).any()
            ]
            fig, ax = plt.subplots(figsize=(11.0, 6.4))
            ax.boxplot(data, tick_labels=labels, showfliers=False)
            ax.axhline(40.0, linestyle="--", linewidth=1.2, label="40 ms frame budget")
            ax.set_xlabel("Architecture")
            ax.set_ylabel("Frame-processing latency (ms)")
            ax.set_title("Experiment 1: Frame-level latency distribution")
            ax.tick_params(axis="x", rotation=25)
            ax.legend()
            save_figure(
                fig,
                "e1_frame_latency_distribution",
                outputs,
                formats,
                dpi,
                log,
                "E1",
                "Distribution of frame-level processing latency.",
                "frame metrics",
            )
    else:
        fig, ax = plt.subplots(figsize=(11.0, 6.3))
        data = [
            combined.loc[
                combined["architecture"] == architecture, "median_latency_ms"
            ].dropna().to_numpy()
            for architecture in E1_ORDER
            if (combined["architecture"] == architecture).any()
        ]
        labels = [
            architecture
            for architecture in E1_ORDER
            if (combined["architecture"] == architecture).any()
        ]
        ax.boxplot(data, tick_labels=labels, showfliers=True)
        ax.set_xlabel("Architecture")
        ax.set_ylabel("Run-level median latency (ms)")
        ax.set_title("Experiment 1: Distribution of run-level median latency")
        ax.tick_params(axis="x", rotation=25)
        save_figure(
            fig,
            "e1_run_median_latency_distribution",
            outputs,
            formats,
            dpi,
            log,
            "E1",
            "Fallback distribution of run-level median latency.",
            "run summaries",
        )
        log.warn("E1 frame metrics were absent; generated a run-level latency distribution.")

    return {
        "architectures": int(overall["architecture"].notna().sum()),
        "runs": int(len(combined)),
    }


def experiment_2(
    roots: Sequence[Path],
    outputs: OutputPaths,
    formats: Sequence[str],
    dpi: int,
    log: BuildLog,
) -> dict[str, object]:
    if not roots:
        raise FileNotFoundError("Experiment 2 folder was not found.")
    _, pipeline = load_experiment_summaries(roots)
    if pipeline.empty:
        raise FileNotFoundError("Experiment 2 run_summary.csv was not found.")

    df = attach_architecture(pipeline)
    df = df[df["architecture"].isin(E2_ORDER)]
    overall = ordered_summary(summarize(df, ["architecture"]), "architecture", E2_ORDER)
    by_video = summarize(df, ["video_name", "architecture"])
    save_table(overall, outputs.tables / "e2_pipeline_overall_summary.csv")
    save_table(by_video, outputs.tables / "e2_pipeline_per_video_summary.csv")

    pivot = by_video.pivot(index="video_name", columns="architecture", values="mean_fps")
    videos = ordered_videos(pivot.index)
    pivot = pivot.reindex(videos)
    if set(E2_ORDER).issubset(pivot.columns):
        fig, ax = plt.subplots(figsize=(10.5, 6.2))
        y = np.arange(len(videos))
        for index, video in enumerate(videos):
            v1 = pivot.loc[video, "Pipeline V1"]
            v2 = pivot.loc[video, "Pipeline V2"]
            ax.hlines(index, min(v1, v2), max(v1, v2), linewidth=1.5)
        ax.scatter(pivot["Pipeline V1"], y, marker="o", label="Pipeline V1")
        ax.scatter(pivot["Pipeline V2"], y, marker="s", label="Pipeline V2")
        ax.axvline(REAL_TIME_FPS, linestyle="--", linewidth=1.2, label="25 FPS threshold")
        ax.set_yticks(y)
        ax.set_yticklabels(videos)
        ax.set_xlabel("Mean throughput (FPS)")
        ax.set_ylabel("Video sequence")
        ax.set_title("Experiment 2: Pipeline V1 versus Pipeline V2")
        ax.legend()
        save_figure(
            fig,
            "e2_pipeline_v1_v2_dumbbell",
            outputs,
            formats,
            dpi,
            log,
            "E2",
            "Paired per-video throughput comparison.",
            "run summaries",
        )

        privacy = by_video.pivot(
            index="video_name", columns="architecture", values="target_leakage_rate_pct"
        ).reindex(videos)
        x = np.arange(len(videos))
        width = 0.36
        fig, ax = plt.subplots(figsize=(12.5, 6.4))
        ax.bar(x - width / 2, privacy["Pipeline V1"], width, label="Pipeline V1")
        ax.bar(x + width / 2, privacy["Pipeline V2"], width, label="Pipeline V2")
        ax.set_xticks(x)
        ax.set_xticklabels(videos)
        ax.set_xlabel("Video sequence")
        ax.set_ylabel("Target leakage rate (%)")
        ax.set_title("Experiment 2: Target-level privacy leakage")
        ax.legend()
        save_figure(
            fig,
            "e2_pipeline_v1_v2_leakage",
            outputs,
            formats,
            dpi,
            log,
            "E2",
            "Per-video target leakage for Pipeline V1 and Pipeline V2.",
            "run summaries",
        )

    frames, _, _ = load_raw(roots)
    if not frames.empty:
        frames = attach_architecture(frames)
        stage_candidates = [
            "decode_ms",
            "resize_ms",
            "detection_ms",
            "tracking_ms",
            "association_ms",
            "coasting_ms",
            "masking_ms",
        ]
        stages = [column for column in stage_candidates if column in frames.columns]
        if stages:
            stage_summary = (
                frames[frames["architecture"].isin(E2_ORDER)]
                .groupby("architecture")[stages]
                .mean(numeric_only=True)
                .reindex(E2_ORDER)
            )
            save_table(
                stage_summary.reset_index(),
                outputs.tables / "e2_mean_stage_latency.csv",
            )
            fig, ax = plt.subplots(figsize=(9.5, 6.0))
            bottom = np.zeros(len(stage_summary))
            for stage in stages:
                values = stage_summary[stage].fillna(0).to_numpy()
                ax.bar(stage_summary.index, values, bottom=bottom, label=stage.replace("_ms", ""))
                bottom += values
            ax.set_xlabel("Pipeline version")
            ax.set_ylabel("Mean stage latency (ms per frame)")
            ax.set_title("Experiment 2: Mean processing-stage latency")
            ax.legend(ncol=2, fontsize=9)
            save_figure(
                fig,
                "e2_stage_latency_comparison",
                outputs,
                formats,
                dpi,
                log,
                "E2",
                "Mean stage-level processing latency for V1 and V2.",
                "frame metrics",
            )

    return {"runs": int(len(df))}


def split_e3_summaries(
    phase_a_roots: Sequence[Path],
    confirmation_roots: Sequence[Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    phase_a_files: list[Path] = []
    phase_b_files: list[Path] = []

    for path in find_csvs(phase_a_roots, is_pipeline_summary):
        lower = str(path).lower()
        if "confirmation" in lower:
            phase_b_files.append(path)
        else:
            phase_a_files.append(path)

    for path in find_csvs(confirmation_roots, is_pipeline_summary):
        phase_b_files.append(path)

    phase_a = load_many(sorted(set(phase_a_files)), dedupe=("run_id",))
    phase_b = load_many(sorted(set(phase_b_files)), dedupe=("run_id",))

    if phase_b.empty and not phase_a.empty and "cnn_interval" in phase_a:
        unique_intervals = sorted(phase_a["cnn_interval"].dropna().unique())
        if set(unique_intervals) == {5, 6} and phase_a["video_name"].nunique() >= 10:
            phase_b = phase_a.copy()
            phase_a = pd.DataFrame()
    return phase_a, phase_b


def experiment_3(
    roots: Sequence[Path],
    confirmation_roots: Sequence[Path],
    outputs: OutputPaths,
    formats: Sequence[str],
    dpi: int,
    log: BuildLog,
) -> dict[str, object]:
    if not roots and not confirmation_roots:
        raise FileNotFoundError("Experiment 3 folders were not found.")

    phase_a, phase_b = split_e3_summaries(roots, confirmation_roots)
    if phase_a.empty:
        raise FileNotFoundError("Experiment 3 Phase A summary was not found.")

    interval_summary = summarize(phase_a, ["cnn_interval"]).sort_values("cnn_interval")
    save_table(interval_summary, outputs.tables / "e3_phase_a_interval_summary.csv")

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    ax.errorbar(
        interval_summary["cnn_interval"],
        interval_summary["mean_fps"],
        yerr=interval_summary["fps_sd"].fillna(0),
        marker="o",
        capsize=4,
    )
    ax.axhline(REAL_TIME_FPS, linestyle="--", linewidth=1.2, label="25 FPS threshold")
    ax.set_xticks(interval_summary["cnn_interval"])
    ax.set_xlabel("CNN execution interval, $I_{cnn}$")
    ax.set_ylabel("Mean throughput (FPS)")
    ax.set_title("Experiment 3: Throughput by CNN execution interval")
    ax.legend()
    save_figure(
        fig,
        "e3_fps_by_cnn_interval",
        outputs,
        formats,
        dpi,
        log,
        "E3",
        "Mean FPS by scheduled CNN interval.",
        "run summaries",
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    ax.plot(interval_summary["cnn_interval"], interval_summary["p95_latency_ms"], marker="o")
    ax.axhline(40.0, linestyle="--", linewidth=1.2, label="40 ms frame budget")
    ax.set_xticks(interval_summary["cnn_interval"])
    ax.set_xlabel("CNN execution interval, $I_{cnn}$")
    ax.set_ylabel("Mean 95th-percentile latency (ms)")
    ax.set_title("Experiment 3: Tail latency by CNN execution interval")
    ax.legend()
    save_figure(
        fig,
        "e3_p95_latency_by_cnn_interval",
        outputs,
        formats,
        dpi,
        log,
        "E3",
        "Mean run-level 95th-percentile latency by interval.",
        "run summaries",
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    ax.plot(
        interval_summary["cnn_interval"],
        interval_summary["target_leakage_rate_pct"],
        marker="o",
    )
    ax.set_xticks(interval_summary["cnn_interval"])
    ax.set_xlabel("CNN execution interval, $I_{cnn}$")
    ax.set_ylabel("Target leakage rate (%)")
    ax.set_title("Experiment 3: Privacy leakage by CNN execution interval")
    save_figure(
        fig,
        "e3_leakage_by_cnn_interval",
        outputs,
        formats,
        dpi,
        log,
        "E3",
        "Target-frame-weighted leakage by interval.",
        "run summaries",
    )

    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    ax.scatter(interval_summary["mean_fps"], interval_summary["target_leakage_rate_pct"], s=70)
    for row in interval_summary.itertuples(index=False):
        ax.annotate(
            f"I={int(row.cnn_interval)}",
            (row.mean_fps, row.target_leakage_rate_pct),
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.axvline(REAL_TIME_FPS, linestyle="--", linewidth=1.2, label="25 FPS threshold")
    ax.set_xlabel("Mean throughput (FPS)")
    ax.set_ylabel("Target leakage rate (%)")
    ax.set_title("Experiment 3: Throughput-privacy trade-off")
    ax.legend()
    save_figure(
        fig,
        "e3_interval_tradeoff",
        outputs,
        formats,
        dpi,
        log,
        "E3",
        "Throughput versus target leakage for each interval.",
        "run summaries",
    )

    if not phase_b.empty:
        confirmation = summarize(phase_b, ["cnn_interval"]).sort_values("cnn_interval")
        confirmation_video = summarize(phase_b, ["video_name", "cnn_interval"])
        save_table(confirmation, outputs.tables / "e3_phase_b_confirmation_summary.csv")
        save_table(
            confirmation_video,
            outputs.tables / "e3_phase_b_per_video_summary.csv",
        )

        pivot = confirmation_video.pivot(
            index="video_name", columns="cnn_interval", values="mean_fps"
        )
        videos = ordered_videos(pivot.index)
        pivot = pivot.reindex(videos)
        intervals = sorted(pivot.columns)
        if len(intervals) >= 2:
            fig, ax = plt.subplots(figsize=(10.5, 6.2))
            y = np.arange(len(videos))
            for index, video in enumerate(videos):
                values = [pivot.loc[video, interval] for interval in intervals]
                ax.hlines(index, min(values), max(values), linewidth=1.5)
            for marker, interval in zip(("o", "s", "^", "D"), intervals):
                ax.scatter(pivot[interval], y, marker=marker, label=f"Interval {int(interval)}")
            ax.axvline(REAL_TIME_FPS, linestyle="--", linewidth=1.2, label="25 FPS threshold")
            ax.set_yticks(y)
            ax.set_yticklabels(videos)
            ax.set_xlabel("Mean throughput (FPS)")
            ax.set_ylabel("Video sequence")
            ax.set_title("Experiment 3: Full-dataset interval confirmation")
            ax.legend()
            save_figure(
                fig,
                "e3_interval_5_6_confirmation",
                outputs,
                formats,
                dpi,
                log,
                "E3",
                "Paired full-dataset comparison of the selected intervals.",
                "run summaries",
            )

    all_roots = unique_paths((*roots, *confirmation_roots))
    frames, _, _ = load_raw(all_roots)
    if not frames.empty and "cnn_interval" in frames:
        if "cnn_executed" in frames:
            observed = (
                frames.groupby("cnn_interval", as_index=False)["cnn_executed"]
                .mean()
                .sort_values("cnn_interval")
            )
            observed["observed_cnn_execution_pct"] = 100.0 * observed["cnn_executed"]
            save_table(observed, outputs.tables / "e3_observed_cnn_execution_rate.csv")
            fig, ax = plt.subplots(figsize=(8.8, 5.8))
            ax.plot(
                observed["cnn_interval"],
                observed["observed_cnn_execution_pct"],
                marker="o",
                label="Observed execution rate",
            )
            scheduled = 100.0 / observed["cnn_interval"]
            ax.plot(
                observed["cnn_interval"],
                scheduled,
                marker="s",
                linestyle="--",
                label="Nominal scheduled rate",
            )
            ax.set_xticks(observed["cnn_interval"])
            ax.set_xlabel("CNN execution interval, $I_{cnn}$")
            ax.set_ylabel("Frames executing CNN (%)")
            ax.set_title("Experiment 3: Scheduled and observed CNN execution")
            ax.legend()
            save_figure(
                fig,
                "e3_observed_cnn_execution_rate",
                outputs,
                formats,
                dpi,
                log,
                "E3",
                "Observed CNN execution rate, including aggressive reacquisition.",
                "frame metrics",
            )

        latency = latency_column(frames)
        if latency and "cnn_executed" in frames:
            selected = frames.copy()
            if (selected["cnn_interval"] == 5).any():
                selected = selected[selected["cnn_interval"] == 5]
            if "video_name" in selected and (selected["video_name"].astype(str) == "video3").any():
                selected = selected[selected["video_name"].astype(str) == "video3"]
            if "repeat_number" in selected and (selected["repeat_number"] == 1).any():
                selected = selected[selected["repeat_number"] == 1]
            if "run_id" in selected and not selected.empty:
                selected = selected[selected["run_id"] == selected["run_id"].iloc[0]]
            selected = selected.sort_values("frame_index").head(60)
            if not selected.empty:
                fig, ax = plt.subplots(figsize=(11.5, 5.8))
                ax.plot(selected["frame_index"], selected[latency], linewidth=1.2, label="Frame latency")
                cnn = selected[selected["cnn_executed"] == 1]
                tracker = selected[selected["cnn_executed"] == 0]
                ax.scatter(cnn["frame_index"], cnn[latency], marker="o", label="CNN frame")
                ax.scatter(tracker["frame_index"], tracker[latency], marker=".", label="Tracker-only frame")
                ax.axhline(40.0, linestyle="--", linewidth=1.2, label="40 ms frame budget")
                ax.set_xlabel("Frame index")
                ax.set_ylabel("Processing latency (ms)")
                ax.set_title("Experiment 3: Interleaved frame-processing latency")
                ax.legend(ncol=2)
                save_figure(
                    fig,
                    "e3_frame_latency_trace",
                    outputs,
                    formats,
                    dpi,
                    log,
                    "E3",
                    "Representative frame-level latency trace with CNN frames marked.",
                    "frame metrics",
                )
    else:
        log.warn("E3 frame metrics were absent; CNN execution-rate and latency-trace figures were skipped.")

    return {"phase_a_runs": int(len(phase_a)), "phase_b_runs": int(len(phase_b))}


def experiment_4(
    roots: Sequence[Path],
    outputs: OutputPaths,
    formats: Sequence[str],
    dpi: int,
    log: BuildLog,
) -> dict[str, object]:
    if not roots:
        raise FileNotFoundError("Experiment 4 folder was not found.")

    baseline, pipeline = load_experiment_summaries(roots)
    combined = pd.concat(
        [attach_architecture(baseline), attach_architecture(pipeline)],
        ignore_index=True,
        sort=False,
    )
    combined = combined[combined["architecture"].isin(E4_ORDER)]
    if combined.empty:
        raise FileNotFoundError("Experiment 4 summaries were not found.")

    summary = ordered_summary(summarize(combined, ["architecture"]), "architecture", E4_ORDER)
    save_table(summary, outputs.tables / "e4_thermal_summary.csv")

    frames, _, _ = load_raw(roots)
    if frames.empty:
        for metric, ylabel, title, stem in (
            ("peak_temperature_c", "Peak temperature (°C)", "Experiment 4: Peak temperature", "e4_peak_temperature"),
            ("mean_cpu_percent", "Mean CPU utilization (%)", "Experiment 4: Mean CPU utilization", "e4_mean_cpu"),
            ("mean_fps", "Mean throughput (FPS)", "Experiment 4: Sustained throughput", "e4_sustained_fps"),
        ):
            fig, ax = plt.subplots(figsize=(9.5, 5.8))
            ax.bar(summary["architecture"].astype(str), summary[metric])
            ax.set_xlabel("Architecture")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=20)
            bar_labels(ax, 1)
            save_figure(
                fig,
                stem,
                outputs,
                formats,
                dpi,
                log,
                "E4",
                title,
                "run summaries",
            )
        log.warn("E4 frame metrics were absent; generated summary bars instead of time-series plots.")
        return {"runs": int(len(combined)), "raw_frame_rows": 0}

    frames = attach_architecture(frames)
    frames = frames[frames["architecture"].isin(E4_ORDER)]
    latency = latency_column(frames)
    run_records: list[dict[str, object]] = []

    for architecture in E4_ORDER:
        arch = frames[frames["architecture"] == architecture]
        if arch.empty:
            continue
        for run_id, run in arch.groupby("run_id", sort=False):
            run = add_elapsed_time(run)
            lat_col = latency_column(run)
            elapsed_max = float(run["elapsed_seconds"].max()) if "elapsed_seconds" in run else 0.0
            first = run[run["elapsed_seconds"] <= 60.0] if "elapsed_seconds" in run else run.head(0)
            last = run[run["elapsed_seconds"] >= max(0.0, elapsed_max - 60.0)] if "elapsed_seconds" in run else run.head(0)

            def fps_for(part: pd.DataFrame) -> float:
                if part.empty or lat_col is None:
                    return float("nan")
                seconds = pd.to_numeric(part[lat_col], errors="coerce").sum() / 1000.0
                return float(len(part) / seconds) if seconds > 0 else float("nan")

            run_records.append(
                {
                    "architecture": architecture,
                    "run_id": run_id,
                    "duration_seconds": elapsed_max,
                    "first_minute_fps": fps_for(first),
                    "last_minute_fps": fps_for(last),
                    "fps_change_pct": (
                        100.0 * (fps_for(last) - fps_for(first)) / fps_for(first)
                        if pd.notna(fps_for(first)) and fps_for(first) > 0 and pd.notna(fps_for(last))
                        else float("nan")
                    ),
                    "mean_cpu_percent": safe_mean(run, "cpu_percent"),
                    "peak_temperature_c": safe_max(run, "temperature_c"),
                    "minimum_cpu_frequency_mhz": (
                        float(pd.to_numeric(run["cpu_frequency_mhz"], errors="coerce").min())
                        if "cpu_frequency_mhz" in run and pd.to_numeric(run["cpu_frequency_mhz"], errors="coerce").notna().any()
                        else float("nan")
                    ),
                    "throttling_samples": (
                        int(
                            (~run.get("throttled_hex", pd.Series("0x0", index=run.index))
                             .astype(str)
                             .str.lower()
                             .isin({"", "0", "0x0", "nan"}))
                            .sum()
                        )
                    ),
                }
            )

    thermal_runs = pd.DataFrame(run_records)
    save_table(thermal_runs, outputs.tables / "e4_sustained_run_metrics.csv")

    def plot_time_series(column: str, ylabel: str, title: str, stem: str) -> None:
        if column not in frames:
            log.warn(f"E4 {column} was absent; skipped {stem}.")
            return
        fig, ax = plt.subplots(figsize=(11.5, 6.0))
        plotted = False
        for architecture in E4_ORDER:
            arch = frames[frames["architecture"] == architecture]
            if arch.empty:
                continue
            run_id = arch["run_id"].iloc[0]
            run = add_elapsed_time(arch[arch["run_id"] == run_id])
            run = telemetry_points(run)
            values = pd.to_numeric(run[column], errors="coerce")
            valid = values.notna()
            if valid.any():
                ax.plot(run.loc[valid, "elapsed_seconds"] / 60.0, values[valid], label=architecture)
                plotted = True
        if not plotted:
            plt.close(fig)
            log.warn(f"E4 {column} contained no usable values; skipped {stem}.")
            return
        ax.set_xlabel("Elapsed processing time (minutes)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        save_figure(
            fig,
            stem,
            outputs,
            formats,
            dpi,
            log,
            "E4",
            title,
            "frame metrics",
        )

    plot_time_series(
        "temperature_c",
        "SoC temperature (°C)",
        "Experiment 4: Temperature during sustained processing",
        "e4_temperature_over_time",
    )
    plot_time_series(
        "cpu_percent",
        "Aggregate CPU utilization (%)",
        "Experiment 4: CPU utilization during sustained processing",
        "e4_cpu_over_time",
    )
    plot_time_series(
        "cpu_frequency_mhz",
        "CPU frequency (MHz)",
        "Experiment 4: CPU frequency during sustained processing",
        "e4_cpu_frequency_over_time",
    )

    if latency:
        fig, ax = plt.subplots(figsize=(11.5, 6.0))
        for architecture in E4_ORDER:
            arch = frames[frames["architecture"] == architecture]
            if arch.empty:
                continue
            run_id = arch["run_id"].iloc[0]
            run = add_elapsed_time(arch[arch["run_id"] == run_id])
            rolling_latency = pd.to_numeric(run[latency], errors="coerce").rolling(
                window=300, min_periods=60
            ).mean()
            rolling_fps = 1000.0 / rolling_latency
            ax.plot(run["elapsed_seconds"] / 60.0, rolling_fps, label=architecture)
        ax.axhline(REAL_TIME_FPS, linestyle="--", linewidth=1.2, label="25 FPS threshold")
        ax.set_xlabel("Elapsed processing time (minutes)")
        ax.set_ylabel("Rolling throughput (FPS)")
        ax.set_title("Experiment 4: Rolling throughput during sustained processing")
        ax.legend()
        save_figure(
            fig,
            "e4_rolling_fps_over_time",
            outputs,
            formats,
            dpi,
            log,
            "E4",
            "Rolling 300-frame throughput over the thermal run.",
            "frame metrics",
        )

    return {"runs": int(len(combined)), "raw_frame_rows": int(len(frames))}


def experiment_5(
    roots: Sequence[Path],
    outputs: OutputPaths,
    formats: Sequence[str],
    dpi: int,
    log: BuildLog,
) -> dict[str, object]:
    if not roots:
        raise FileNotFoundError("Experiment 5 folder was not found.")

    baseline, pipeline = load_experiment_summaries(roots)
    combined = pd.concat(
        [attach_e5_configuration(baseline), attach_e5_configuration(pipeline)],
        ignore_index=True,
        sort=False,
    )
    combined = combined[combined["configuration"].isin(E5_ORDER)]
    if combined.empty:
        raise FileNotFoundError("Experiment 5 summaries were not found.")

    overall = ordered_summary(summarize(combined, ["configuration"]), "configuration", E5_ORDER)
    by_video = summarize(combined, ["video_name", "configuration"])
    save_table(overall, outputs.tables / "e5_privacy_ablation_overall.csv")
    save_table(by_video, outputs.tables / "e5_privacy_ablation_per_video.csv")

    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    ax.bar(overall["configuration"].astype(str), overall["target_leakage_rate_pct"])
    ax.set_xlabel("Ablation configuration")
    ax.set_ylabel("Target leakage rate (%)")
    ax.set_title("Experiment 5: Aggregate temporal privacy leakage")
    ax.tick_params(axis="x", rotation=18)
    bar_labels(ax, 2)
    save_figure(
        fig,
        "e5_aggregate_leakage",
        outputs,
        formats,
        dpi,
        log,
        "E5",
        "Target-frame-weighted leakage across the privacy ablation.",
        "run summaries",
    )

    videos = ordered_videos(by_video["video_name"])
    x = np.arange(len(videos))
    width = 0.2
    fig, ax = plt.subplots(figsize=(15.0, 7.0))
    for index, configuration in enumerate(E5_ORDER):
        subset = by_video[by_video["configuration"] == configuration].set_index("video_name")
        values = [
            subset.loc[video, "target_leakage_rate_pct"] if video in subset.index else np.nan
            for video in videos
        ]
        ax.bar(
            x + (index - (len(E5_ORDER) - 1) / 2) * width,
            values,
            width,
            label=configuration,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(videos)
    ax.set_xlabel("Video sequence")
    ax.set_ylabel("Target leakage rate (%)")
    ax.set_title("Experiment 5: Privacy leakage across all sequences")
    ax.legend(ncol=2)
    save_figure(
        fig,
        "e5_leakage_all_sequences",
        outputs,
        formats,
        dpi,
        log,
        "E5",
        "Per-video leakage across all ablation configurations.",
        "run summaries",
    )

    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    ax.bar(
        overall["configuration"].astype(str),
        overall["maximum_consecutive_leak_frames"],
    )
    ax.set_xlabel("Ablation configuration")
    ax.set_ylabel("Maximum consecutive leaked target-frames")
    ax.set_title("Experiment 5: Longest observed privacy failure")
    ax.tick_params(axis="x", rotation=18)
    bar_labels(ax, 0)
    save_figure(
        fig,
        "e5_max_consecutive_leakage",
        outputs,
        formats,
        dpi,
        log,
        "E5",
        "Maximum consecutive target-frame leakage by configuration.",
        "run summaries",
    )

    visible = overall["visible_target_frames"].replace(0, np.nan)
    composition = pd.DataFrame(
        {
            "configuration": overall["configuration"].astype(str),
            "Protected": 100.0 * overall["protected_target_frames"] / visible,
            "Partially protected": 100.0
            * overall["partially_protected_target_frames"]
            / visible,
            "Leaked": 100.0 * overall["leaked_target_frames"] / visible,
        }
    )
    save_table(composition, outputs.tables / "e5_target_frame_composition.csv")
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    bottom = np.zeros(len(composition))
    for column in ("Protected", "Partially protected", "Leaked"):
        values = composition[column].fillna(0).to_numpy()
        ax.bar(composition["configuration"], values, bottom=bottom, label=column)
        bottom += values
    ax.set_xlabel("Ablation configuration")
    ax.set_ylabel("Visible target-frames (%)")
    ax.set_title("Experiment 5: Privacy-status composition")
    ax.tick_params(axis="x", rotation=18)
    ax.legend()
    save_figure(
        fig,
        "e5_target_frame_composition",
        outputs,
        formats,
        dpi,
        log,
        "E5",
        "Protected, partially protected, and leaked target-frame composition.",
        "run summaries",
    )

    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    ax.scatter(overall["mean_fps"], overall["target_leakage_rate_pct"], s=75)
    for row in overall.itertuples(index=False):
        ax.annotate(
            str(row.configuration).split(":")[0],
            (row.mean_fps, row.target_leakage_rate_pct),
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.axvline(REAL_TIME_FPS, linestyle="--", linewidth=1.2, label="25 FPS threshold")
    ax.set_xlabel("Mean throughput (FPS)")
    ax.set_ylabel("Target leakage rate (%)")
    ax.set_title("Experiment 5: Throughput-privacy trade-off")
    ax.legend()
    save_figure(
        fig,
        "e5_throughput_privacy_tradeoff",
        outputs,
        formats,
        dpi,
        log,
        "E5",
        "Throughput versus leakage for each temporal-memory ablation.",
        "run summaries",
    )

    frames, targets, _ = load_raw(roots)
    if not frames.empty:
        frames = attach_e5_configuration(frames)
        full = frames[frames["configuration"] == "D: Full Pipeline V2"]
        if not full.empty and {"video_name", "visible_gt_targets"}.issubset(full.columns):
            density = (
                full.groupby("video_name", as_index=False)
                .agg(
                    mean_visible_targets=("visible_gt_targets", "mean"),
                    mean_fps=("instantaneous_fps", "mean"),
                )
            )
            d_summary = by_video[by_video["configuration"] == "D: Full Pipeline V2"][
                ["video_name", "target_leakage_rate_pct"]
            ]
            density = density.merge(d_summary, on="video_name", how="left")
            density = density.sort_values(
                "video_name", key=lambda series: series.map(lambda value: natural_video_key(value)[0])
            )
            save_table(density, outputs.tables / "e5_target_density_scaling.csv")

            fig, ax = plt.subplots(figsize=(8.8, 6.0))
            ax.scatter(density["mean_visible_targets"], density["mean_fps"], s=70)
            for row in density.itertuples(index=False):
                ax.annotate(str(row.video_name), (row.mean_visible_targets, row.mean_fps), xytext=(5, 5), textcoords="offset points")
            ax.axhline(REAL_TIME_FPS, linestyle="--", linewidth=1.2, label="25 FPS threshold")
            ax.set_xlabel("Mean simultaneously visible targets")
            ax.set_ylabel("Mean instantaneous throughput (FPS)")
            ax.set_title("Experiment 5: Throughput versus target density")
            ax.legend()
            save_figure(
                fig,
                "e5_throughput_by_target_density",
                outputs,
                formats,
                dpi,
                log,
                "E5",
                "Observed Pipeline V2 throughput versus mean visible target count.",
                "frame metrics and run summaries",
            )

            fig, ax = plt.subplots(figsize=(8.8, 6.0))
            ax.scatter(
                density["mean_visible_targets"],
                density["target_leakage_rate_pct"],
                s=70,
            )
            for row in density.itertuples(index=False):
                ax.annotate(str(row.video_name), (row.mean_visible_targets, row.target_leakage_rate_pct), xytext=(5, 5), textcoords="offset points")
            ax.set_xlabel("Mean simultaneously visible targets")
            ax.set_ylabel("Target leakage rate (%)")
            ax.set_title("Experiment 5: Privacy leakage versus target density")
            save_figure(
                fig,
                "e5_leakage_by_target_density",
                outputs,
                formats,
                dpi,
                log,
                "E5",
                "Observed Pipeline V2 leakage versus mean visible target count.",
                "frame metrics and run summaries",
            )

    if not targets.empty:
        targets = attach_e5_configuration(targets)
        if "video_name" not in targets.columns:
            run_to_video = combined[["run_id", "video_name"]].drop_duplicates()
            targets = targets.merge(run_to_video, on="run_id", how="left")
        video_choice = "video3" if (targets.get("video_name", pd.Series(dtype=str)).astype(str) == "video3").any() else None
        selected = targets.copy()
        if video_choice:
            selected = selected[selected["video_name"].astype(str) == video_choice]
        if "run_id" in selected:
            pieces: list[pd.DataFrame] = []
            for configuration in E5_ORDER:
                config = selected[selected["configuration"] == configuration]
                if config.empty:
                    continue
                run_id = config["run_id"].iloc[0]
                pieces.append(config[config["run_id"] == run_id])
            selected = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
        if not selected.empty and {"frame_index", "privacy_status", "configuration"}.issubset(selected.columns):
            status_rank = {
                "PROTECTED": 0,
                "Protected": 0,
                "PARTIALLY_PROTECTED": 1,
                "PARTIALLY PROTECTED": 1,
                "Partially Protected": 1,
                "LEAKED": 2,
                "Leaked": 2,
            }
            selected["status_rank"] = selected["privacy_status"].map(status_rank)
            timeline = (
                selected.dropna(subset=["status_rank"])
                .groupby(["configuration", "frame_index"], as_index=False)["status_rank"]
                .max()
            )
            if not timeline.empty:
                fig, ax = plt.subplots(figsize=(12.0, 6.0))
                for configuration in E5_ORDER:
                    subset = timeline[timeline["configuration"] == configuration]
                    if not subset.empty:
                        ax.step(subset["frame_index"], subset["status_rank"], where="post", label=configuration)
                ax.set_yticks([0, 1, 2])
                ax.set_yticklabels(["Protected", "Partial", "Leaked"])
                ax.set_xlabel("Frame index")
                ax.set_ylabel("Worst visible-target privacy status")
                ax.set_title(f"Experiment 5: Frame-level privacy timeline for {video_choice or 'selected video'}")
                ax.legend(ncol=2)
                save_figure(
                    fig,
                    "e5_video3_privacy_timeline",
                    outputs,
                    formats,
                    dpi,
                    log,
                    "E5",
                    "Frame-level worst-target privacy status during the selected occlusion sequence.",
                    "target metrics",
                )
    else:
        log.warn("E5 target metrics were absent; the frame-level privacy timeline was skipped.")

    return {"runs": int(len(combined))}


def infer_buffer_from_run_id(run_id: object) -> float:
    match = re.search(r"(?:^|_)M(\d+)(?:_|$)", str(run_id), flags=re.IGNORECASE)
    return float(match.group(1)) if match else float("nan")


def experiment_6(
    roots: Sequence[Path],
    outputs: OutputPaths,
    formats: Sequence[str],
    dpi: int,
    log: BuildLog,
) -> dict[str, object]:
    if not roots:
        raise FileNotFoundError("Experiment 6 folder was not found.")
    _, summary_df = load_experiment_summaries(roots)
    if summary_df.empty:
        raise FileNotFoundError("Experiment 6 run_summary.csv was not found.")

    df = summary_df.copy()
    df["nominal_output_delay_ms"] = 1000.0 * df["buffer_size"] / INPUT_FPS
    grouped = (
        df.groupby(["video_name", "buffer_size"], as_index=False)
        .agg(
            runs=("run_id", "size"),
            nominal_output_delay_ms=("nominal_output_delay_ms", "mean"),
            mean_fps=("mean_fps", "mean"),
            median_latency_ms=("median_latency_ms", "mean"),
            p95_latency_ms=("p95_latency_ms", "mean"),
            output_released_frames=("output_released_frames", "mean"),
            output_unprotected_frames=("output_unprotected_frames", "mean"),
            output_unprotected_target_frames=("output_unprotected_target_frames", "mean"),
            startup_dropped_frames=("startup_dropped_frames", "mean"),
        )
        .sort_values(["video_name", "buffer_size"])
    )
    leakage = (
        df.groupby(["video_name", "buffer_size"], as_index=False)
        .apply(lambda group: pd.Series({"target_leakage_rate_pct": weighted_leakage(group)}), include_groups=False)
        .reset_index(drop=True)
    )
    grouped = grouped.merge(leakage, on=["video_name", "buffer_size"], how="left")
    save_table(grouped, outputs.tables / "e6_startup_buffer_summary.csv")

    videos = ordered_videos(grouped["video_name"])
    for metric, ylabel, title, stem in (
        (
            "output_unprotected_frames",
            "Mean unprotected output frames",
            "Experiment 6: Startup exposure by buffer capacity",
            "e6_unprotected_frames_by_buffer",
        ),
        (
            "startup_dropped_frames",
            "Mean startup-dropped frames",
            "Experiment 6: Frames withheld during startup",
            "e6_startup_dropped_frames",
        ),
    ):
        fig, ax = plt.subplots(figsize=(9.2, 6.0))
        for video in videos:
            subset = grouped[grouped["video_name"] == video].sort_values("buffer_size")
            ax.plot(subset["buffer_size"], subset[metric], marker="o", label=video)
        ax.set_xticks(sorted(grouped["buffer_size"].dropna().unique()))
        ax.set_xlabel("Startup buffer capacity (frames)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        save_figure(
            fig,
            stem,
            outputs,
            formats,
            dpi,
            log,
            "E6",
            title,
            "run summaries",
        )

    delay = grouped[["buffer_size", "nominal_output_delay_ms"]].drop_duplicates().sort_values("buffer_size")
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    ax.bar(delay["buffer_size"].astype(int).astype(str), delay["nominal_output_delay_ms"])
    ax.set_xlabel("Startup buffer capacity (frames)")
    ax.set_ylabel("Nominal output delay (ms at 30 FPS)")
    ax.set_title("Experiment 6: Nominal buffer-induced output delay")
    bar_labels(ax, 0)
    save_figure(
        fig,
        "e6_nominal_output_delay",
        outputs,
        formats,
        dpi,
        log,
        "E6",
        "Nominal delay implied by each buffer capacity.",
        "configured buffer sizes",
    )

    frames, _, event_rows = load_raw(roots)
    event_metrics: list[dict[str, object]] = []
    if not event_rows.empty:
        summary_map = df[["run_id", "video_name", "buffer_size"]].drop_duplicates()
        duplicate_ids = summary_map["run_id"].duplicated(keep=False)
        unique_summary_map = summary_map[~duplicate_ids]
        event_rows = event_rows.merge(unique_summary_map, on="run_id", how="left")
        event_rows["buffer_size"] = event_rows["buffer_size"].fillna(
            event_rows["run_id"].map(infer_buffer_from_run_id)
        )

        frame_lookup: dict[str, pd.DataFrame] = {}
        if not frames.empty and "run_id" in frames:
            frame_lookup = {str(run_id): group for run_id, group in frames.groupby("run_id")}

        for run_id, events in event_rows.groupby("run_id", sort=False):
            events = events.sort_values("processing_frame_index")
            released = events[events["event_type"].astype(str).str.upper().isin({"RELEASED", "FLUSHED"})]
            if released.empty:
                continue
            first_release = released.iloc[0]
            protected_release = released[
                (pd.to_numeric(released["visible_target_count"], errors="coerce") > 0)
                & (pd.to_numeric(released["unprotected_target_count"], errors="coerce") == 0)
            ]
            first_protected_release = protected_release.iloc[0] if not protected_release.empty else None
            delay_frames = float(first_release["processing_frame_index"] - first_release["output_frame_index"])
            record: dict[str, object] = {
                "run_id": run_id,
                "video_name": first_release.get("video_name", np.nan),
                "buffer_size": first_release.get("buffer_size", infer_buffer_from_run_id(run_id)),
                "first_release_processing_frame": first_release["processing_frame_index"],
                "first_release_source_frame": first_release["output_frame_index"],
                "measured_delay_frames": delay_frames,
                "measured_delay_ms_at_30fps": 1000.0 * delay_frames / INPUT_FPS,
                "first_protected_release_processing_frame": (
                    first_protected_release["processing_frame_index"] if first_protected_release is not None else np.nan
                ),
                "first_protected_release_source_frame": (
                    first_protected_release["output_frame_index"] if first_protected_release is not None else np.nan
                ),
            }
            run_frames = frame_lookup.get(str(run_id))
            if run_frames is not None and not run_frames.empty:
                run_frames = add_elapsed_time(run_frames)
                protected = run_frames[pd.to_numeric(run_frames.get("protected_targets", 0), errors="coerce") > 0]
                record["first_protected_processing_frame"] = (
                    float(protected["frame_index"].iloc[0]) if not protected.empty else np.nan
                )
                release_index = int(first_release["processing_frame_index"])
                at_release = run_frames[pd.to_numeric(run_frames["frame_index"], errors="coerce") <= release_index]
                record["first_release_elapsed_ms"] = (
                    1000.0 * float(at_release["elapsed_seconds"].iloc[-1]) if not at_release.empty else np.nan
                )
            event_metrics.append(record)

    event_df = pd.DataFrame(event_metrics)
    if not event_df.empty:
        save_table(event_df, outputs.tables / "e6_output_event_metrics.csv")
        event_summary = (
            event_df.groupby("buffer_size", as_index=False)
            .agg(
                runs=("run_id", "size"),
                measured_delay_frames=("measured_delay_frames", "mean"),
                measured_delay_ms_at_30fps=("measured_delay_ms_at_30fps", "mean"),
                first_release_processing_frame=("first_release_processing_frame", "mean"),
                first_protected_processing_frame=("first_protected_processing_frame", "mean"),
                first_protected_release_processing_frame=("first_protected_release_processing_frame", "mean"),
                first_release_elapsed_ms=("first_release_elapsed_ms", "mean"),
            )
            .sort_values("buffer_size")
        )
        save_table(event_summary, outputs.tables / "e6_output_event_summary.csv")

        fig, ax = plt.subplots(figsize=(8.8, 5.8))
        ax.plot(event_summary["buffer_size"], event_summary["measured_delay_ms_at_30fps"], marker="o")
        ax.set_xticks(event_summary["buffer_size"])
        ax.set_xlabel("Startup buffer capacity (frames)")
        ax.set_ylabel("Measured source-to-release delay (ms at 30 FPS)")
        ax.set_title("Experiment 6: Measured output delay")
        save_figure(
            fig,
            "e6_measured_output_delay",
            outputs,
            formats,
            dpi,
            log,
            "E6",
            "Measured frame-index delay between processing and release.",
            "output-event logs",
        )

        timeline_columns = [
            ("first_protected_processing_frame", "First protected processing frame", "o"),
            ("first_release_processing_frame", "First released output", "s"),
            ("first_protected_release_processing_frame", "First protected released output", "^"),
        ]
        fig, ax = plt.subplots(figsize=(10.0, 6.0))
        for column, label, marker in timeline_columns:
            if column in event_summary and event_summary[column].notna().any():
                ax.plot(event_summary["buffer_size"], event_summary[column], marker=marker, label=label)
        ax.set_xticks(event_summary["buffer_size"])
        ax.set_xlabel("Startup buffer capacity (frames)")
        ax.set_ylabel("Mean processing-frame index")
        ax.set_title("Experiment 6: Cold-start processing and output timeline")
        ax.legend()
        save_figure(
            fig,
            "e6_startup_timeline",
            outputs,
            formats,
            dpi,
            log,
            "E6",
            "Mean first-protection and first-release events by buffer capacity.",
            "frame metrics and output-event logs",
        )

        tradeoff = grouped.groupby("buffer_size", as_index=False)["output_unprotected_frames"].mean().merge(
            event_summary[["buffer_size", "measured_delay_ms_at_30fps"]],
            on="buffer_size",
            how="inner",
        )
        fig, ax = plt.subplots(figsize=(8.8, 6.0))
        ax.scatter(tradeoff["measured_delay_ms_at_30fps"], tradeoff["output_unprotected_frames"], s=75)
        for row in tradeoff.itertuples(index=False):
            ax.annotate(
                f"M={int(row.buffer_size)}",
                (row.measured_delay_ms_at_30fps, row.output_unprotected_frames),
                xytext=(5, 5),
                textcoords="offset points",
            )
        ax.set_xlabel("Measured output delay (ms at 30 FPS)")
        ax.set_ylabel("Mean unprotected released frames")
        ax.set_title("Experiment 6: Startup privacy-latency trade-off")
        save_figure(
            fig,
            "e6_buffer_tradeoff",
            outputs,
            formats,
            dpi,
            log,
            "E6",
            "Measured delay versus unprotected output exposure.",
            "run summaries and output-event logs",
        )
    else:
        log.warn("E6 output-event logs were absent; measured-delay and startup-timeline figures were skipped.")

    return {"runs": int(len(df)), "videos": int(df["video_name"].nunique())}


def write_manifest(log: BuildLog, outputs: OutputPaths) -> None:
    manifest = pd.DataFrame(log.figures)
    if not manifest.empty:
        manifest.to_csv(outputs.reports / "figure_manifest.csv", index=False)


def describe_roots(roots: ExperimentRoots, results_root: Path) -> list[str]:
    lines = ["DISCOVERED EXPERIMENT DIRECTORIES", "=================================", ""]
    for name in ("e1", "e2", "e3", "e3_confirmation", "e4", "e5", "e6"):
        paths = getattr(roots, name)
        if not paths:
            lines.append(f"{name}: NOT FOUND")
        else:
            for index, path in enumerate(paths, start=1):
                try:
                    display = path.relative_to(results_root)
                except ValueError:
                    display = path
                lines.append(f"{name}[{index}]: {display}")
    lines.append("")
    return lines


def write_quality_report(
    roots: ExperimentRoots,
    results_root: Path,
    outputs: OutputPaths,
    log: BuildLog,
) -> None:
    lines = [
        "THESIS FIGURE DATA-QUALITY REPORT",
        "=================================",
        "",
        f"Results root: {results_root}",
        f"Generated figures: {len(log.figures)}",
        f"Warnings: {len(log.warnings)}",
        f"Failures: {len(log.failures)}",
        "",
    ]
    lines.extend(describe_roots(roots, results_root))

    lines.extend(["RAW FILE INVENTORY", "------------------", ""])
    for label, experiment_roots in (
        ("E1", roots.e1),
        ("E2", roots.e2),
        ("E3", unique_paths((*roots.e3, *roots.e3_confirmation))),
        ("E4", roots.e4),
        ("E5", roots.e5),
        ("E6", roots.e6),
    ):
        csvs = all_csv_files(experiment_roots)
        lines.append(
            f"{label}: summaries={sum(is_baseline_summary(p) or is_pipeline_summary(p) for p in csvs)}, "
            f"frame_metrics={sum(is_frame_metrics(p) for p in csvs)}, "
            f"target_metrics={sum(is_target_metrics(p) for p in csvs)}, "
            f"output_events={sum(is_output_events(p) for p in csvs)}"
        )
    lines.append("")

    if log.warnings:
        lines.extend(["WARNINGS", "--------"])
        lines.extend(f"- {warning}" for warning in log.warnings)
        lines.append("")
    if log.failures:
        lines.extend(["FAILURES", "--------"])
        lines.extend(f"- {failure}" for failure in log.failures)
        lines.append("")

    lines.extend(
        [
            "INTERPRETATION NOTES",
            "--------------------",
            "Experiment 1 retains the timing-boundary limitation documented in Chapter 4.",
            "Use its cross-architecture throughput comparison as preliminary until the",
            "baseline and pipeline timing boundaries are standardized and rerun.",
            "",
            "Experiment 4 contains one sustained run per architecture unless additional",
            "runs exist in the results folder. Describe a single-run dataset as a",
            "preliminary sustained thermal characterization.",
            "",
            "The script computes aggregate leakage as total leaked visible target-frames",
            "divided by total visible target-frames. It does not average run-level",
            "percentage values.",
        ]
    )
    (outputs.reports / "data_quality_report.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_readme(outputs: OutputPaths) -> None:
    text = """# Thesis results outputs

This directory was generated from the `smart_cam/results` tree.

- `figures/` contains PDF and PNG figures. Use the PDF files in LaTeX.
- `tables/` contains aggregated CSV tables and `chapter5_key_metrics.json`.
- `reports/figure_manifest.csv` maps each figure to its experiment and source data.
- `reports/data_quality_report.txt` records missing raw files, skipped figures, and known limitations.

The script calculates aggregate privacy leakage from target-frame counts:

`100 * sum(leaked_target_frames) / sum(visible_target_frames)`

It uses frame-level and output-event files when available and falls back to run summaries when necessary.
"""
    (outputs.root / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_root = args.results_dir.expanduser().resolve()
    outputs = prepare_output(args.output_dir, args.clean)
    log = BuildLog()
    roots = resolve_experiment_roots(results_root)
    metrics: dict[str, object] = {}

    run_guarded(
        "Experiment 1",
        lambda: metrics.update(
            e1=experiment_1(roots.e1, outputs, args.formats, args.dpi, log)
        ),
        log,
        args.strict,
    )
    run_guarded(
        "Experiment 2",
        lambda: metrics.update(
            e2=experiment_2(roots.e2, outputs, args.formats, args.dpi, log)
        ),
        log,
        args.strict,
    )
    run_guarded(
        "Experiment 3",
        lambda: metrics.update(
            e3=experiment_3(
                roots.e3,
                roots.e3_confirmation,
                outputs,
                args.formats,
                args.dpi,
                log,
            )
        ),
        log,
        args.strict,
    )
    run_guarded(
        "Experiment 4",
        lambda: metrics.update(
            e4=experiment_4(roots.e4, outputs, args.formats, args.dpi, log)
        ),
        log,
        args.strict,
    )
    run_guarded(
        "Experiment 5",
        lambda: metrics.update(
            e5=experiment_5(roots.e5, outputs, args.formats, args.dpi, log)
        ),
        log,
        args.strict,
    )
    run_guarded(
        "Experiment 6",
        lambda: metrics.update(
            e6=experiment_6(roots.e6, outputs, args.formats, args.dpi, log)
        ),
        log,
        args.strict,
    )

    write_manifest(log, outputs)
    write_quality_report(roots, results_root, outputs, log)
    write_readme(outputs)
    save_json(metrics, outputs.tables / "chapter5_key_metrics.json")

    pdf_count = len(list(outputs.figures.glob("*.pdf")))
    png_count = len(list(outputs.figures.glob("*.png")))
    csv_count = len(list(outputs.tables.glob("*.csv")))

    print("\nGeneration complete")
    print(f"  Results directory: {results_root}")
    print(f"  Output directory:  {outputs.root}")
    print(f"  Figures: {len(log.figures)} ({pdf_count} PDF, {png_count} PNG)")
    print(f"  Tables:  {csv_count} CSV")
    print(f"  Warnings: {len(log.warnings)}")
    print(f"  Failures: {len(log.failures)}")
    print("\nReview reports/data_quality_report.txt before using the figures in Chapter 5.")


if __name__ == "__main__":
    main()
