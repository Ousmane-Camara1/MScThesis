import cv2
import numpy as np
import time
import argparse
import os
import csv

# --- CLI ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description="Privacy-Preserving Vision: Skip-Frame Pipeline")
parser.add_argument('video_source', nargs='?', default='pi', help="Use 'pi' for live camera, 'benchmark' for batch testing, or 'video.mp4'.")
args = parser.parse_args()

# --- LOGGING FUNCTION ---
def log_experiment_to_csv(video_path, model_name, frame_count, missed_frames, fps_list):
    csv_output_path = "evaluation_benchmarks.csv" 
    file_exists = os.path.exists(csv_output_path)
    
    leaked_frames = len(missed_frames)
    failure_rate_pct = round((leaked_frames / frame_count) * 100, 2) if frame_count > 0 else 0.0
    mean_fps = round(sum(fps_list) / len(fps_list), 2) if fps_list else 0.0
    
    row_data = {
        "video_name": os.path.basename(video_path),
        "model_name": model_name,  
        "total_frames": frame_count,
        "leaked_frames": leaked_frames,
        "failure_rate_pct": failure_rate_pct,
        "mean_fps": mean_fps
    }
    
    with open(csv_output_path, mode='a', newline='') as csv_file:
        fieldnames = ["video_name", "model_name", "total_frames", "leaked_frames", "failure_rate_pct", "mean_fps"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)
        
    print(f"Logged {model_name} performance on {row_data['video_name']} successfully.")

# --- CONFIGURATION & CONSTANTS ---
YUNET_MODEL_PATH = "face_detection_yunet_2023mar.onnx"
if not os.path.exists(YUNET_MODEL_PATH):
    print("Error: YuNet model not found.")
    exit(1)

# Detection Resolution 
DETECT_WIDTH = 480 
DETECT_HEIGHT = 360

# Hybrid Tracking Heuristics
SKIP_FRAMES = 3         # Run CNN 1 frame, use Optical Flow for the next 3 frames
MAX_LOST_FRAMES = 15    # Frames to wait before completely dropping the privacy mask
EXPANSION_MARGIN = 10   # Pixels to grow the mask per frame when target is lost

# Cryptographic Key Matrix
np.random.seed(42) 
CIPHER_KEY_MATRIX = np.random.randint(0, 256, (4000, 4000, 3), dtype=np.uint8)

# Initialize Edge-Optimized CNN
yunet = cv2.FaceDetectorYN.create(
    model=YUNET_MODEL_PATH, config="", input_size=(DETECT_WIDTH, DETECT_HEIGHT), 
    score_threshold=0.6, nms_threshold=0.3, top_k=50
)

# --- DYNAMIC REVERSIBLE MASK ---
def apply_dynamic_reversible_mask(frame, x, y, w, h, padding=30):
    ih, iw = frame.shape[:2]
    
    x1, y1 = max(0, x - padding), max(0, y - padding)
    x2, y2 = min(iw, x + w + padding), min(ih, y + h + padding)
    
    new_w, new_h = x2 - x1, y2 - y1
    if new_w <= 0 or new_h <= 0: return frame

    face_roi = frame[y1:y2, x1:x2]
    key_roi = CIPHER_KEY_MATRIX[:new_h, :new_w] 
    
    shape_mask = np.zeros((new_h, new_w), dtype=np.uint8)
    center, axes = (new_w // 2, new_h // 2), (new_w // 2, new_h // 2)
    cv2.ellipse(shape_mask, center, axes, 0, 0, 360, 255, -1)
    
    encrypted_roi = cv2.bitwise_xor(face_roi, key_roi)
    condition = shape_mask[:, :, None] == 255
    frame[y1:y2, x1:x2] = np.where(condition, encrypted_roi, face_roi)
    
    return frame

# --- TRACKER SELECTION FACTORY ---
def create_tracker():
    try:
        return cv2.legacy.TrackerMOSSE_create()
    except AttributeError:
        return cv2.TrackerKCF_create()

# --- SOURCE ROUTING LOGIC ---
is_live_pi = (args.video_source.lower() == 'pi')
is_benchmark = (args.video_source.lower() == 'benchmark')

if is_benchmark:
    video_sources = [
        "1stvideo.mp4", "2ndvideo.mp4", "3rdvideo.mp4", 
        "video4.mp4", "video5.mp4", "video6.mp4", 
        "video7.mp4", "video8.mp4", "video9.mp4", 
        "video10.mp4", "video11trial.mp4", "video12.mp4", "video13.mp4"
    ]
    print("Initializing Automated Batch Benchmark Mode...")
else:
    video_sources = [args.video_source]

for video_target in video_sources:
    if is_live_pi:
        from picamera2 import Picamera2
        picam2 = Picamera2()
        config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
        picam2.configure(config)
        picam2.start()
    else:
        if not os.path.exists(video_target) and is_benchmark:
            print(f"Skipping {video_target} - File not found.")
            continue
        cap = cv2.VideoCapture(video_target)
        if is_benchmark:
            print(f"\nProcessing Benchmark: {video_target}")

    # --- PIPELINE STATE VARIABLES ---
    reconstruct_mode = False
    prev_time = time.time()
    consecutive_misses = 0
    cached_faces = []
    trackers = [] 
    frame_count = 0
    current_mode = "Initializing"
    
    # Audit Trackers
    missed_frames = []
    fps_list = []

    print("Pipeline Active. Press 'r' to toggle Decryption. Press 'q' to quit (or skip to next in benchmark).")

    try:
        while True:
            # 1. ACQUIRE FRAME
            if is_live_pi:
                frame_rgb = picam2.capture_array()
                if frame_rgb is None: break
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                frame = cv2.flip(frame, 1)
            else:
                ret, frame = cap.read()
                if not ret: break

            ih, iw = frame.shape[:2]
            frame_count += 1
            current_faces = []
            tracking_successful = False

            # 2. DECISION GATE: CNN vs TRACKER
            # Aggressive Re-acquisition Logic applied here
            run_cnn = (frame_count % SKIP_FRAMES == 0) or (len(trackers) == 0 and consecutive_misses == 0)

            if run_cnn:
                # STATE A: CNN DETECTION 
                small_frame = cv2.resize(frame, (DETECT_WIDTH, DETECT_HEIGHT))
                scale_x, scale_y = iw / DETECT_WIDTH, ih / DETECT_HEIGHT
                
                _, faces = yunet.detect(small_frame)
                
                if faces is not None:
                    trackers = [] 
                    for face in faces:
                        box = face[:4]
                        x, y = int(box[0] * scale_x), int(box[1] * scale_y)
                        w, h = int(box[2] * scale_x), int(box[3] * scale_y)
                        current_faces.append((x, y, w, h))
                        
                        tracker = create_tracker()
                        tracker.init(frame, (x, y, w, h))
                        trackers.append(tracker)
                        
                    cached_faces = current_faces
                    consecutive_misses = 0
                    tracking_successful = True
                    current_mode = "Detection (Pure CNN)"
            else:
                # STATE B: OPTICAL FLOW TRACKING 
                trackers_to_keep = []
                for tracker in trackers:
                    success, box = tracker.update(frame)
                    if success:
                        x, y, w, h = [int(v) for v in box]
                        current_faces.append((x, y, w, h))
                        trackers_to_keep.append(tracker)
                
                trackers = trackers_to_keep 
                
                if len(current_faces) > 0:
                    cached_faces = current_faces
                    consecutive_misses = 0
                    tracking_successful = True
                    current_mode = "Tracking (Optical Flow)"

            # 3. MEMORY EXTENSION (If both CNN and Tracker fail)
            if not tracking_successful:
                # STATE C: TARGET LOST (Coasting)
                consecutive_misses += 1
                trackers = [] 
                
                if consecutive_misses <= MAX_LOST_FRAMES:
                    current_mode = f"Memory Coast ({consecutive_misses})"
                    expanded_faces = []
                    for (x, y, w, h) in cached_faces:
                        new_x = max(0, x - EXPANSION_MARGIN)
                        new_y = max(0, y - EXPANSION_MARGIN)
                        new_w = w + (EXPANSION_MARGIN * 2)
                        new_h = h + (EXPANSION_MARGIN * 2)
                        expanded_faces.append((new_x, new_y, new_w, new_h))
                    cached_faces = expanded_faces
                else:
                    current_mode = "Searching..."
                    cached_faces = []

            # 4. PRIVACY VULNERABILITY AUDIT CHECK
            # If cached_faces is empty, the mask has dropped, meaning biometric data leaked.
            if len(cached_faces) == 0:
                missed_frames.append(frame_count)

            # 5. APPLY CRYPTOGRAPHY
            for (x, y, w, h) in cached_faces:
                frame = apply_dynamic_reversible_mask(frame, x, y, w, h)
                if reconstruct_mode:
                    frame = apply_dynamic_reversible_mask(frame, x, y, w, h)

            # 6. METRICS & UI
            curr_time = time.time()
            process_time = curr_time - prev_time
            fps = 1 / process_time if process_time > 0 else 0
            prev_time = curr_time
            fps_list.append(fps)

            status_color = (0, 255, 0) if reconstruct_mode else (0, 0, 255)
            cv2.putText(frame, f"FPS: {int(fps)} | {current_mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "AUTHORIZED" if reconstruct_mode else "SECURE", (10, ih - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            cv2.imshow('Thesis Pipeline', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): 
                print("Aborted by user.")
                break
            elif key == ord('r'): 
                reconstruct_mode = not reconstruct_mode
                
    except KeyboardInterrupt:
        pass
    finally:
        if is_live_pi: 
            picam2.stop()
        else: 
            cap.release()
            if is_benchmark:
                log_experiment_to_csv(video_target, "Proposed_Framework", frame_count, missed_frames, fps_list)
        cv2.destroyAllWindows()

print("Execution Complete.")
