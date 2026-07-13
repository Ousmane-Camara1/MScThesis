#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
BASELINE_DIR = Path(__file__).resolve().parent
DEFAULT_REPO = BASELINE_DIR / 'Ultra-Light-Fast-Generic-Face-Detector-1MB'
sys.path.insert(0, str(BASELINE_DIR.parent / 'common'))
sys.path.insert(0, str(DEFAULT_REPO))
import cv2
import numpy as np
import onnxruntime as ort
from benchmark_common import DetectionResult, create_common_parser, now_ms, run_benchmark

class UltraFaceDetector:
    name = 'UltraFace'
    def __init__(self, model_path: Path, repo_path: Path, probability_threshold: float, iou_threshold: float) -> None:
        if not model_path.exists(): raise FileNotFoundError(model_path)
        if not repo_path.exists(): raise FileNotFoundError(repo_path)
        if str(repo_path) not in sys.path: sys.path.insert(0, str(repo_path))
        try:
            import vision.utils.box_utils_numpy as box_utils
        except ImportError as exc:
            raise ImportError(f'Could not import UltraFace vision utilities from {repo_path}') from exc
        self.box_utils = box_utils; self.threshold = probability_threshold; self.iou = iou_threshold
        self.session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
    def warmup(self, frame: np.ndarray, runs: int) -> None:
        for _ in range(runs): self.detect(frame)
    def postprocess(self, frame_width, frame_height, confidences, boxes):
        boxes = boxes[0]; confidences = confidences[0]; picked = []
        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]; mask = probs > self.threshold; probs = probs[mask]
            if probs.shape[0] == 0: continue
            box_probs = np.concatenate([boxes[mask, :], probs.reshape(-1,1)], axis=1)
            picked.append(self.box_utils.hard_nms(box_probs, iou_threshold=self.iou, top_k=-1))
        if not picked: return []
        picked = np.vstack(picked); picked[:, [0,2]] *= frame_width; picked[:, [1,3]] *= frame_height
        return [(int(round(x1)), int(round(y1)), int(round(x2-x1)), int(round(y2-y1))) for x1,y1,x2,y2,_ in picked]
    def detect(self, frame: np.ndarray) -> DetectionResult:
        start = now_ms(); image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); image = cv2.resize(image, (320,240)); image = (image.astype(np.float32)-127.0)/128.0; image = np.transpose(image, (2,0,1))[None, ...].astype(np.float32); preprocess = now_ms()-start
        start = now_ms(); outputs = self.session.run(None, {self.input_name: image}); inference = now_ms()-start
        if len(outputs) != 2: raise RuntimeError(f'Expected 2 UltraFace outputs, got {len(outputs)}')
        start = now_ms(); boxes = self.postprocess(frame.shape[1], frame.shape[0], outputs[0], outputs[1]); postprocess = now_ms()-start
        return DetectionResult(boxes, preprocess, inference, postprocess)

def main() -> None:
    parser = create_common_parser('UltraFace stateless baseline benchmark')
    parser.add_argument('--ultraface-repo', type=Path, default=DEFAULT_REPO)
    parser.add_argument('--model-path', type=Path, default=DEFAULT_REPO/'models'/'onnx'/'version-RFB-320.onnx')
    parser.add_argument('--probability-threshold', type=float, default=0.70)
    parser.add_argument('--iou-threshold', type=float, default=0.30)
    args = parser.parse_args(); args.ultraface_repo = args.ultraface_repo.expanduser().resolve(); args.model_path = args.model_path.expanduser().resolve()
    run_benchmark(UltraFaceDetector(args.model_path, args.ultraface_repo, args.probability_threshold, args.iou_threshold), args)

if __name__ == '__main__': main()
