import cv2
import numpy as np
import time
import os

# --- CONFIGURATION & CONSTANTS ---
YUNET_MODEL_PATH = "face_detection_yunet_2023mar.onnx"
video_path = "video11trial.mp4" # Update to your Kaggle video

if not os.path.exists(YUNET_MODEL_PATH):
    print("Error: YuNet model not found.")
    exit(1)

# Detection Resolution 
DETECT_WIDTH = 480 
DETECT_HEIGHT = 360

# Hybrid Tracking Heuristics
SKIP_FRAMES = 3         
MAX_LOST_FRAMES = 15    
EXPANSION_MARGIN = 10   

# Initialize Edge-Optimized CNN
yunet = cv2.FaceDetectorYN.create(
    model=YUNET_MODEL_PATH, config="", input_size=(DETECT_WIDTH, DETECT_HEIGHT), 
    score_threshold=0.6, nms_threshold=0.3, top_k=50
)

# --- TRACKER SELECTION FACTORY ---
def create_tracker():
    try:
        return cv2.legacy.TrackerMOSSE_create()
    except AttributeError:
        return cv2.TrackerKCF_create()

# --- VIDEO SOURCE INITIALIZATION ---
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Starting Visual Audit on: {video_path}")
print("The video will pause ANY TIME the Coasting Memory fails and drops the mask.")
print("Press 'n' to advance to the next dropped frame.")
print("Press 'q' to quit the audit early.\n")

# --- PIPELINE STATE VARIABLES ---
consecutive_misses = 0
cached_faces = []
trackers = [] 
frame_count = 0
missed_frames = []
multiple_faces_frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    ih, iw = frame.shape[:2]
    frame_count += 1
    current_faces = []
    tracking_successful = False

    # 1. DECISION GATE: CNN vs TRACKER
    # DECISION GATE: Run CNN on the Nth frame, OR aggressively scan every frame if the target is currently lost
    run_cnn = (frame_count % SKIP_FRAMES == 0) or (len(trackers) == 0)

    if run_cnn:
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
    else:
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

    # 2. MEMORY EXTENSION (Coasting Phase)
    if not tracking_successful:
        consecutive_misses += 1
        trackers = [] 
        
        if consecutive_misses <= MAX_LOST_FRAMES:
            expanded_faces = []
            for (x, y, w, h) in cached_faces:
                new_x = max(0, x - EXPANSION_MARGIN)
                new_y = max(0, y - EXPANSION_MARGIN)
                new_w = w + (EXPANSION_MARGIN * 2)
                new_h = h + (EXPANSION_MARGIN * 2)
                expanded_faces.append((new_x, new_y, new_w, new_h))
            cached_faces = expanded_faces
        else:
            cached_faces = []

    # 3. AUDIT LOGGING & VISUAL DISPLAY
    # A failure only occurs if cached_faces is completely empty
    face_count = len(cached_faces)
    
    if face_count == 0:
        missed_frames.append(frame_count)
        
        display_frame = frame.copy()
        cv2.putText(display_frame, f"PRIVACY LEAK: Frame {frame_count}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(display_frame, f"Failed after coasting {MAX_LOST_FRAMES} frames", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        cv2.putText(display_frame, "Press 'n' for next, 'q' to quit", (20, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Apply the scaling fix for large surveillance videos
        cv2.namedWindow("Proposed Framework - Audit", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Proposed Framework - Audit", 1280, 720)
        cv2.imshow("Proposed Framework - Audit", display_frame)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            print("Audit manually terminated by user.")
            break
            
    elif face_count > 1:
        multiple_faces_frames.append(frame_count)

    # Print a progress update every 100 frames
    if frame_count % 100 == 0:
        print(f"Scanned {frame_count} / {total_frames} frames...")

cap.release()
cv2.destroyAllWindows()

# --- TERMINAL REPORT GENERATION ---
print("\n" + "="*45)
print("PROPOSED FRAMEWORK AUDIT REPORT")
print("="*45)
print(f"Total Frames Processed: {frame_count}")
print(f"Frames with ZERO active masks (Leaks): {len(missed_frames)}")
print(f"Frames with MULTIPLE masked targets: {len(multiple_faces_frames)}")
print(f"Failure Rate: {round((len(missed_frames) / frame_count) * 100, 2)}%")
print("="*45)
