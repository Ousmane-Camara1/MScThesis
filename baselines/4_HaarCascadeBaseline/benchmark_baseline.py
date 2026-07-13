#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
BASELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASELINE_DIR.parent / 'common'))
import cv2
import numpy as np
from benchmark_common import DetectionResult, create_common_parser, now_ms, run_benchmark

class HaarCascadeDetector:
    name = 'HaarCascade'
    def __init__(self, scale_factor: float, min_neighbors: int, minimum_size: int) -> None:
        path = Path(cv2.data.haarcascades)/'haarcascade_frontalface_default.xml'; self.detector = cv2.CascadeClassifier(str(path))
        if self.detector.empty(): raise RuntimeError(f'Could not load {path}')
        self.scale_factor = scale_factor; self.min_neighbors = min_neighbors; self.minimum_size = minimum_size
    def warmup(self, frame: np.ndarray, runs: int) -> None:
        for _ in range(runs): self.detect(frame)
    def detect(self, frame: np.ndarray) -> DetectionResult:
        start = now_ms(); gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY); preprocess = now_ms()-start
        start = now_ms(); faces = self.detector.detectMultiScale(gray, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors, minSize=(self.minimum_size,self.minimum_size)); inference = now_ms()-start
        start = now_ms(); boxes = [(int(x),int(y),int(w),int(h)) for x,y,w,h in faces]
        return DetectionResult(boxes, preprocess, inference, now_ms()-start)

def main() -> None:
    parser = create_common_parser('Haar Cascade stateless baseline benchmark')
    parser.add_argument('--scale-factor', type=float, default=1.10); parser.add_argument('--min-neighbors', type=int, default=5); parser.add_argument('--minimum-size', type=int, default=20)
    args = parser.parse_args(); run_benchmark(HaarCascadeDetector(args.scale_factor, args.min_neighbors, args.minimum_size), args)

if __name__ == '__main__': main()
