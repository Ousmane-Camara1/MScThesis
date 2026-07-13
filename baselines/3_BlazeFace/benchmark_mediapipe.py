#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
BASELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASELINE_DIR.parent / 'common'))
import cv2
import mediapipe as mp
import numpy as np
from benchmark_common import DetectionResult, create_common_parser, now_ms, run_benchmark

class BlazeFaceDetector:
    name = 'BlazeFace'
    def __init__(self, model_selection: int, confidence: float) -> None:
        self.detector = mp.solutions.face_detection.FaceDetection(model_selection=model_selection, min_detection_confidence=confidence)
    def warmup(self, frame: np.ndarray, runs: int) -> None:
        for _ in range(runs): self.detect(frame)
    def detect(self, frame: np.ndarray) -> DetectionResult:
        start = now_ms(); rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); preprocess = now_ms()-start
        start = now_ms(); results = self.detector.process(rgb); inference = now_ms()-start
        start = now_ms(); h, w = frame.shape[:2]; boxes = []
        if results.detections:
            for detection in results.detections:
                box = detection.location_data.relative_bounding_box
                boxes.append((int(round(box.xmin*w)), int(round(box.ymin*h)), int(round(box.width*w)), int(round(box.height*h))))
        return DetectionResult(boxes, preprocess, inference, now_ms()-start)
    def close(self): self.detector.close()

def main() -> None:
    parser = create_common_parser('MediaPipe BlazeFace stateless baseline benchmark')
    parser.add_argument('--model-selection', type=int, choices=[0,1], default=0)
    parser.add_argument('--min-detection-confidence', type=float, default=0.50)
    args = parser.parse_args(); detector = BlazeFaceDetector(args.model_selection, args.min_detection_confidence)
    try: run_benchmark(detector, args)
    finally: detector.close()

if __name__ == '__main__': main()
