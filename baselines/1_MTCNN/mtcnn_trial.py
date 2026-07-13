#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
BASELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASELINE_DIR.parent / 'common'))
import cv2
import numpy as np
from mtcnn import MTCNN
from benchmark_common import DetectionResult, create_common_parser, now_ms, run_benchmark

class MTCNNDetector:
    name = 'MTCNN'
    def __init__(self, min_confidence: float) -> None:
        self.detector = MTCNN(); self.min_confidence = min_confidence
    def warmup(self, frame: np.ndarray, runs: int) -> None:
        for _ in range(runs): self.detect(frame)
    def detect(self, frame: np.ndarray) -> DetectionResult:
        start = now_ms(); rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); preprocess = now_ms()-start
        start = now_ms(); detections = self.detector.detect_faces(rgb); inference = now_ms()-start
        start = now_ms(); boxes = []
        for detection in detections:
            if float(detection.get('confidence', 0.0)) < self.min_confidence: continue
            x, y, w, h = detection['box']; boxes.append((int(round(x)), int(round(y)), int(round(w)), int(round(h))))
        return DetectionResult(boxes, preprocess, inference, now_ms()-start)

def main() -> None:
    parser = create_common_parser('MTCNN stateless baseline benchmark')
    parser.add_argument('--min-confidence', type=float, default=0.70)
    args = parser.parse_args(); run_benchmark(MTCNNDetector(args.min_confidence), args)

if __name__ == '__main__': main()
