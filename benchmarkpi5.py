import cv2
import time
import numpy as np
import mediapipe as mp
from picamera2 import Picamera2

# --- CONFIGURATION ---
MAX_NUM_FACES = 1
VISUALIZE = True  # Set to False for pure AI speed testing

# 1. Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=MAX_NUM_FACES,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 2. Initialize Pi Camera 5 (Native Mode)
print("Initializing Pi Camera 5...")
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

print(f"Starting Benchmark... (Max Faces: {MAX_NUM_FACES})")

try:
    while True:
        # A. Capture frame directly as Numpy Array (Zero-Copy)
        # This is much faster than OpenCV's capture
        image_rgb = picam2.capture_array()

        if image_rgb is None:
            break

        # --- TIMER START ---
        start_t = time.time()
        
        # B. Perform inference (MediaPipe)
        # Note: Image is already RGB, so no conversion needed!
        image_rgb.flags.writeable = False
        results = face_mesh.process(image_rgb)
        
        # --- TIMER END ---
        end_t = time.time()
        fps = 1 / (end_t - start_t)

        # C. Visualization
        if VISUALIZE:
            # Convert back to BGR for OpenCV display
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image_bgr,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
                    )

            cv2.putText(image_bgr, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Pi 5 Native Benchmark', image_bgr)

        if cv2.waitKey(1) & 0xFF == 27: # ESC to quit
            break
            
except KeyboardInterrupt:
    print("Stopped by user")
finally:
    picam2.stop()
    cv2.destroyAllWindows()
