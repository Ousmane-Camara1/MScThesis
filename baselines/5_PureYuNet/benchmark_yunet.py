#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
BASELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASELINE_DIR.parent.parent
sys.path.insert(0, str(BASELINE_DIR.parent / 'common'))
import cv2
import numpy as np
from benchmark_common import DetectionResult, create_common_parser, now_ms, run_benchmark

class PureYuNetDetector:
    name = 'PureYuNet'
    def __init__(self, model_path: Path, input_width: int, input_height: int, score: float, nms: float, top_k: int) -> None:
        if not model_path.exists(): raise FileNotFoundError(model_path)
        self.input_size = (input_width,input_height)
        self.detector = cv2.FaceDetectorYN.create(str(model_path), '', self.input_size, score, nms, top_k)
    def warmup(self, frame: np.ndarray, runs: int) -> None:
        for _ in range(runs): self.detect(frame)
    def detect(self, frame: np.ndarray) -> DetectionResult:
        start = now_ms(); resized = cv2.resize(frame, self.input_size); preprocess = now_ms()-start
        start = now_ms(); _, faces = self.detector.detect(resized); inference = now_ms()-start
        start = now_ms(); boxes = []
        if faces is not None:
            sx, sy = frame.shape[1]/self.input_size[0], frame.shape[0]/self.input_size[1]
            for face in faces:
                x,y,w,h = face[:4]; boxes.append((int(round(x*sx)), int(round(y*sy)), int(round(w*sx)), int(round(h*sy))))
        return DetectionResult(boxes, preprocess, inference, now_ms()-start)

def main() -> None:
    parser = create_common_parser('Pure YuNet stateless baseline benchmark')
    parser.add_argument('--model-path', type=Path, default=PROJECT_ROOT/'models'/'yunet'/'face_detection_yunet_2023mar.onnx')
    parser.add_argument('--input-width', type=int, default=480); parser.add_argument('--input-height', type=int, default=360)
    parser.add_argument('--score-threshold', type=float, default=0.60); parser.add_argument('--nms-threshold', type=float, default=0.30); parser.add_argument('--top-k', type=int, default=50)
    args = parser.parse_args(); args.model_path = args.model_path.expanduser().resolve()
    run_benchmark(PureYuNetDetector(args.model_path, args.input_width, args.input_height, args.score_threshold, args.nms_threshold, args.top_k), args)

if __name__ == '__main__': main()
