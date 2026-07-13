# Corrected stateless baseline benchmarks

Copy the contents into the existing `smart_cam` project so the structure becomes:

```text
smart_cam/
├── baselines/
│   ├── common/
│   │   └── benchmark_common.py
│   ├── 1_MTCNN/
│   │   ├── mtcnn_trial.py
│   │   └── mtcnnenv/
│   ├── 2_UltraFace/
│   │   ├── benchmark_ultraface.py
│   │   ├── ultrafaceenv/
│   │   └── Ultra-Light-Fast-Generic-Face-Detector-1MB/
│   ├── 3_BlazeFace/
│   │   ├── benchmark_mediapipe.py
│   │   └── blazefaceenv/
│   ├── 4_HaarCascadeBaseline/
│   │   ├── benchmark_baseline.py
│   │   └── baselineenv/
│   └── 5_PureYuNet/
│       └── benchmark_yunet.py
├── data/
│   ├── clips/
│   │   ├── video_01.mp4
│   │   └── ... video_10.mp4
│   └── annotations/ground_truth_csv/
│       └── ground_truth_all_videos.csv
├── models/yunet/
│   └── face_detection_yunet_2023mar.onnx
└── results/experiment_1_baselines/
```

Run every command from the `smart_cam` root.

## What changed

Every baseline now uses the same shared evaluator and therefore:

- compares privacy masks against validated visible target-frames;
- excludes `FULLY_OCCLUDED` and `OUTSIDE` targets;
- uses ground-truth face-area coverage, not `detections == 0`;
- classifies `>= 90%` as protected, `50% to < 90%` as partially protected, and `< 50%` as leaked;
- calculates FPS as processed frames divided by total measured time;
- logs decode, preprocessing, inference, post-processing, masking, p50, p95, and p99 latency;
- runs headlessly unless `--display` is explicitly supplied;
- records CPU utilization, temperature, frequency, and throttling;
- writes per-frame, per-target, settings, and run-summary CSV files.

The shared module must be copied to `baselines/common/benchmark_common.py`.

## MTCNN

Activate `baselines/1_MTCNN/mtcnnenv`, then:

```bash
python baselines/1_MTCNN/mtcnn_trial.py \
  --clips-dir data/clips \
  --ground-truth data/annotations/ground_truth_csv/ground_truth_all_videos.csv \
  --output-dir results/experiment_1_baselines \
  --video all \
  --repeat-number 1
```

Required packages include `mtcnn`, its TensorFlow backend, `opencv-python`, `numpy`, and `psutil`.

## UltraFace

The default repository and model paths are:

```text
baselines/2_UltraFace/Ultra-Light-Fast-Generic-Face-Detector-1MB
baselines/2_UltraFace/Ultra-Light-Fast-Generic-Face-Detector-1MB/models/onnx/version-RFB-320.onnx
```

Activate `ultrafaceenv`, then:

```bash
python baselines/2_UltraFace/benchmark_ultraface.py \
  --clips-dir data/clips \
  --ground-truth data/annotations/ground_truth_csv/ground_truth_all_videos.csv \
  --output-dir results/experiment_1_baselines \
  --video all \
  --repeat-number 1
```

Override the defaults with `--ultraface-repo` or `--model-path` when necessary.

## MediaPipe BlazeFace

Activate `blazefaceenv`, then:

```bash
python baselines/3_BlazeFace/benchmark_mediapipe.py \
  --clips-dir data/clips \
  --ground-truth data/annotations/ground_truth_csv/ground_truth_all_videos.csv \
  --output-dir results/experiment_1_baselines \
  --video all \
  --repeat-number 1
```

## Haar Cascade

Activate `baselineenv`, then:

```bash
python baselines/4_HaarCascadeBaseline/benchmark_baseline.py \
  --clips-dir data/clips \
  --ground-truth data/annotations/ground_truth_csv/ground_truth_all_videos.csv \
  --output-dir results/experiment_1_baselines \
  --video all \
  --repeat-number 1
```

## Pure YuNet

The default model is `models/yunet/face_detection_yunet_2023mar.onnx`.

```bash
python baselines/5_PureYuNet/benchmark_yunet.py \
  --clips-dir data/clips \
  --ground-truth data/annotations/ground_truth_csv/ground_truth_all_videos.csv \
  --output-dir results/experiment_1_baselines \
  --video all \
  --repeat-number 1
```

## One-video smoke test

Before running all ten clips, test each script on one video and 30 frames:

```bash
--video video_01 --stop-after-frames 30
```

For example:

```bash
python baselines/4_HaarCascadeBaseline/benchmark_baseline.py \
  --clips-dir data/clips \
  --ground-truth data/annotations/ground_truth_csv/ground_truth_all_videos.csv \
  --output-dir results/smoke_tests \
  --video video_01 \
  --stop-after-frames 30
```

## Three final repeats

Start a new process for every repeat:

```bash
for repeat in 1 2 3; do
  python baselines/5_PureYuNet/benchmark_yunet.py \
    --clips-dir data/clips \
    --ground-truth data/annotations/ground_truth_csv/ground_truth_all_videos.csv \
    --output-dir results/experiment_1_baselines \
    --video all \
    --repeat-number "$repeat"
done
```

## Outputs

```text
results/experiment_1_baselines/
├── baseline_run_summary.csv
├── MTCNN/
├── UltraFace/
├── BlazeFace/
├── HaarCascade/
└── PureYuNet/
```

Each model directory contains `frame_metrics/`, `target_metrics/`, and `settings/`.

Keep `--include-decode-in-throughput` enabled for all baselines if it is also enabled for Pipeline V1 and V2. Do not use `--display` during final throughput runs.
