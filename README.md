# MScThesis: Stateful Edge Vision for Real-Time Privacy
Stateful Edge Vision: Temporal Tracking and Reversible Anonymization for Real-Time Privacy

This is a privacy-preserving edge-computer-vision prototype for real-time facial anonymization on a Raspberry Pi 5.

The project combines periodic YuNet face detection with lightweight MOSSE/KCF correlation-filter tracking, per-target temporal state, conservative privacy-mask continuation, reversible XOR-based region transformation, and optional startup output gating.

The implementation was developed as part of my Master's thesis:

> **Stateful Edge Vision: Temporal Tracking and Reversible Anonymization for Real-Time Privacy**  
> *Bridging the Gap Between Real-Time Inference and Temporal Security on Single-Board Computers*

## Overview

Frame-independent face anonymization can expose visible faces whenever the detector misses a frame. This project introduces a stateful pipeline that maintains an independent record for each target and continues privacy rendering during short detector or tracker failures.

The repository includes:

- stateless detector baselines;
- Pipeline V1 with global tracking state;
- Pipeline V2 with independent per-target state;
- CNN interval ablation experiments;
- temporal privacy ablation experiments;
- thermal and CPU telemetry collection;
- startup output-gating experiments;
- scripts for aggregating results and producing thesis figures.

## Main Features

### Hybrid detection and tracking

Pipeline V2 interleaves YuNet face detection with MOSSE tracking when available and KCF tracking as an OpenCV fallback. It also increases detector frequency when tracking becomes uncertain.

### Per-target temporal memory

Each active target maintains its own identifier, bounding box, tracker, operational state, consecutive-miss counter, and most recent detector-update frame.

### Temporal coasting

When both detection association and tracker updating fail, the system temporarily retains the latest target box and expands it conservatively. This is a spatial uncertainty mechanism rather than a motion-prediction model.

### Reversible privacy rendering

The prototype applies an elliptical XOR transformation to active facial regions. The transformation is lightweight and exactly reversible when the same key matrix is reapplied.

It is intended for experimental evaluation only and does **not** provide production-grade confidentiality, authenticated encryption, or secure key management.

### Startup output gating

An optional FIFO-based startup gate can withhold frames while the pipeline initializes.

The current implementation should be treated as an experimental fail-closed mechanism. In the evaluated cold-start trials, non-zero buffer capacities prevented unprotected output but did not produce a usable delayed protected stream.

## Repository Structure

```text
MScThesis/
├── data/
│   ├── clips/
│   ├── thermal/
│   └── annotations/
│       ├── validated_mot/
│       └── ground_truth_csv/
├── models/
│   ├── yunet/
│   └── ultraface/
├── baselines/
│   ├── common/
│   ├── 1_MTCNN/
│   ├── 2_UltraFace/
│   ├── 3_BlazeFace/
│   ├── 4_HaarCascadeBaseline/
│   └── 5_PureYuNet/
├── pipelines/
│   ├── benchmark_v1_v2.py
│   └── legacy/
├── results/
│   ├── experiment_1_baselines/
│   ├── experiment_2_v1_vs_v2/
│   ├── experiment_3_cnn_interval/
│   ├── experiment_4_thermal/
│   ├── experiment_5_privacy/
│   └── experiment_6_startup_buffer/
└── thesis_results_outputs/
    └── figures/
```

Some directories are intentionally excluded from version control because they contain large videos, model weights, generated results, or thesis figures.

## Hardware and Software

The primary benchmark platform was:

- Raspberry Pi 5;
- Broadcom BCM2712;
- four ARM Cortex-A76 CPU cores;
- 8 GB LPDDR4X RAM;
- active cooling;
- 64-bit Raspberry Pi OS;
- Python 3.11.

Separate virtual environments may be required because TensorFlow, MediaPipe, ONNX Runtime, OpenCV, and OpenCV contrib can require different dependency combinations.

## Installation

Clone the repository:

```bash
git clone https://github.com/Ousmane-Camara1/MScThesis.git
cd MScThesis
```

Create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Upgrade the packaging tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Some detector directories may contain their own dependency or lock files. Install those inside separate virtual environments when needed.

## Model Files

ONNX model files are excluded from Git.

Place the required model weights in the corresponding directories:

```text
models/
├── yunet/
│   └── face_detection_yunet.onnx
└── ultraface/
    └── version-RFB-320.onnx
```

Update the local configuration when your filenames differ.

## Data and Annotations

Videos, annotations, and generated results are excluded from version control.

Expected local layout:

```text
data/
├── clips/
├── thermal/
└── annotations/
    ├── validated_mot/
    └── ground_truth_csv/
```

The evaluation dataset used ten standardized 15-second videos at:

- 1920 × 1080 resolution;
- 30 FPS;
- 450 frames per video.

The final annotations were manually validated after model-assisted pre-annotation and exported into a canonical target-level CSV format.

## Running the Pipeline

The main V1/V2 benchmark entry point is:

```bash
python pipelines/benchmark_v1_v2.py --help
```

Use the command-line help to inspect the options available in the current version of the script.

A typical run should specify an input video, pipeline version, CNN interval, maximum coasting duration, spatial expansion margin, mask padding, and startup-buffer capacity.

Example template:

```bash
python pipelines/benchmark_v1_v2.py \
  --input data/clips/video_01.mp4 \
  --pipeline v2 \
  --cnn-interval 5 \
  --max-lost-frames 15 \
  --expansion-margin 10 \
  --mask-padding 30 \
  --startup-buffer 0
```

The exact option names may differ. Confirm them with `--help` before execution.

## Experimental Evaluation

The repository supports six experimental groups:

1. comparative baseline performance;
2. Pipeline V1 versus Pipeline V2;
3. CNN execution-interval ablation;
4. sustained CPU and thermal behaviour;
5. temporal privacy ablation;
6. startup output gating.

The principal privacy metric is target-level facial-region coverage:

```text
Protected:           coverage >= 0.90
Partially protected: 0.50 <= coverage < 0.90
Leaked:              coverage < 0.50
```

## Selected Results

In the controlled Pipeline V1 versus Pipeline V2 comparison:

- Pipeline V1: 41.04 FPS;
- Pipeline V2: 40.95 FPS;
- P95 latency decreased from 86.70 ms to 44.72 ms;
- target leakage decreased from 24.16% to 15.79%.

In the temporal privacy ablation:

- Pure YuNet leakage: 24.096%;
- full Pipeline V2 leakage: 15.655%;
- leakage events decreased from 201 to 90;
- mean reacquisition delay decreased from 9.30 to 4.63 frames.

The results show that per-target temporal state improves privacy continuity, but does not eliminate failures under rapid motion, prolonged detector loss, or incomplete startup control.

## Generated Outputs

Depending on the experiment, a run may generate:

- frame-level metric CSV files;
- target-level privacy CSV files;
- run-summary CSV files;
- telemetry CSV files;
- JSON configuration files;
- startup output-event logs;
- aggregated plots and thesis figures.

Generated outputs are excluded from version control by default.

## Reproducibility Notes

For reproducible comparisons:

- use the same standardized input videos;
- retain the same ground-truth annotations;
- disable graphical display during timed runs;
- disable output-video encoding during benchmarks;
- execute systems sequentially rather than in parallel;
- use fresh Python processes for repeated cold-start trials;
- preserve raw outputs without manual editing;
- record exact dependency versions for each environment.

## Security Limitations

This repository is a research prototype.

It does not currently provide production-grade encryption, authenticated key exchange, secure key storage, resistance to a compromised operating system, formal anonymity guarantees, guaranteed zero leakage, or a complete asynchronous startup architecture.

A production deployment should replace the experimental XOR transformation with authenticated encryption such as AES-GCM and implement secure nonce, key-generation, storage, rotation, and access-control procedures.

## Citation

```bibtex
@mastersthesis{camara2026stateful,
  author = {Ousmane Camara},
  title = {Stateful Edge Vision: Temporal Tracking and Reversible Anonymization for Real-Time Privacy},
  school = {University of Geneva},
  year = {2026},
  month = {July}
}
```

## License

Add the intended software license before distributing or reusing the project.

Dataset and model files may be governed by separate third-party licences and should not be redistributed without confirming their terms.

## Author

**Ousmane Camara**  
Master of Science in Computer Science  
University of Geneva
