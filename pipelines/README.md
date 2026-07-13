# Raspberry Pi Pipeline V1/V2 Benchmark

This package contains one unified benchmark script:

```text
benchmark_v1_v2.py
```

It evaluates Pipeline V1 and Pipeline V2 using the same validated ground truth,
privacy-mask evaluator, timing boundaries, telemetry collection, and CSV output.

## Install dependencies

On Raspberry Pi OS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The tracker requires an OpenCV contrib build. Depending on your Raspberry Pi
installation, you may already have OpenCV from the system repository. Do not
install a second conflicting OpenCV build without checking `cv2.__version__`.

## Required inputs

- One standardized 15-second video
- Its validated canonical ground-truth CSV, or the combined CSV containing all videos
- YuNet ONNX model

The ground-truth `video_name` must match the video filename stem after basic
normalization. Use `--video-name` to override the name.

## Pipeline V1

V1 keeps the original global-state behavior:

```bash
python benchmark_v1_v2.py \
  --video clips/video_03.mp4 \
  --ground-truth ground_truth_all_videos.csv \
  --model-path face_detection_yunet_2023mar.onnx \
  --pipeline-version v1 \
  --cnn-interval 4 \
  --buffer-size 0 \
  --output-dir results
```

## Pipeline V2

V2 uses per-target state, IoU association, and aggressive reacquisition:

```bash
python benchmark_v1_v2.py \
  --video clips/video_03.mp4 \
  --ground-truth ground_truth_all_videos.csv \
  --model-path face_detection_yunet_2023mar.onnx \
  --pipeline-version v2 \
  --cnn-interval 4 \
  --buffer-size 0 \
  --output-dir results
```

## CNN interval meaning

The CLI uses the actual interval between CNN frames:

```text
--cnn-interval 1  CNN every frame
--cnn-interval 2  one CNN frame, then one tracker-only frame
--cnn-interval 4  one CNN frame, then three tracker-only frames
--cnn-interval 6  one CNN frame, then five tracker-only frames
```

## RQ3 buffer runs

Use V2 for the buffer experiment:

```bash
python benchmark_v1_v2.py \
  --video clips/video_01.mp4 \
  --ground-truth ground_truth_all_videos.csv \
  --model-path face_detection_yunet_2023mar.onnx \
  --pipeline-version v2 \
  --cnn-interval 4 \
  --buffer-size 30 \
  --warmup-runs 0 \
  --output-dir results
```

For a true cold-start trial, launch a fresh Python process for every run and
use `--warmup-runs 0`.

Run buffer capacities:

```text
0
5
15
30
```

## Headless benchmarks

The default is headless. Keep it this way for final throughput measurements.

To inspect output:

```bash
--display
```

To save the masked stream:

```bash
--save-output-video results/video_03_v2.mp4
```

Do not compare a headless run against a displayed run.

## Repeated runs

Run each configuration as a separate process:

```bash
for repeat in 1 2 3; do
  python benchmark_v1_v2.py \
    --video clips/video_03.mp4 \
    --ground-truth ground_truth_all_videos.csv \
    --model-path face_detection_yunet_2023mar.onnx \
    --pipeline-version v2 \
    --cnn-interval 4 \
    --buffer-size 0 \
    --repeat-number "$repeat" \
    --output-dir results
done
```

## Output files

```text
results/
├── <run_id>_frame_metrics.csv
├── <run_id>_target_metrics.csv
├── <run_id>_output_events.csv
├── <run_id>_settings.json
└── run_summary.csv
```

### Frame metrics

Includes:

- decode latency
- resize latency
- CNN latency
- tracker latency
- association latency
- coasting latency
- masking latency
- buffer latency
- total latency
- CPU utilization
- temperature
- CPU frequency
- throttling state
- target privacy counts

### Target metrics

For every visible ground-truth target-frame:

- actual privacy-mask coverage
- protected, partially protected, or leaked status
- reappearance event
- reacquisition delay

### Run summary

Includes:

- mean FPS calculated as frames divided by total measured time
- median, p95, and p99 latency
- percentage of frames completed within 40 ms
- target leakage rate
- leakage-event statistics
- consecutive leakage
- reacquisition latency
- CPU and thermal summaries

## Privacy classification

Coverage is the fraction of the ground-truth facial rectangle covered by the
actual union of elliptical privacy masks:

```text
Protected:             coverage >= 0.90
Partially protected:   0.50 <= coverage < 0.90
Leaked:                coverage < 0.50
```

Fully occluded and outside ground-truth rows are excluded from the visible
target-frame denominator.

## Experimental rules

For V1 versus V2:

```text
buffer_size = 0
same cnn_interval
same video
same ground truth
same thresholds
same display setting
```

For CNN-interval ablation:

```text
pipeline_version = v2
buffer_size = 0
cnn_interval = 1, 2, 3, 4, 5, 6
```

For startup-buffer evaluation:

```text
pipeline_version = v2
buffer_size = 0, 5, 15, 30
warmup_runs = 0
fresh process per trial
```

## Important implementation note

The buffer prevents display-stage startup leakage by withholding output until
the first protected frame and dropping the unprotected startup prefix. It does
not claim to retroactively reconstruct or secure frames that were already
captured internally.
