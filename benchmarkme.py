import cv2
import time
import mediapipe as mp

# --- CONFIGURATION ---
VIDEO_SOURCE = 0  # 0 = Live Camera
MAX_NUM_FACES = 1
VISUALIZE = True

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=MAX_NUM_FACES,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize Video Capture (Standard OpenCV)
cap = cv2.VideoCapture(VIDEO_SOURCE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print(f"Starting Benchmark... (Max Faces: {MAX_NUM_FACES})")

try:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Video finished or failed to load.")
            break

        # Convert the BGR image to RGB
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # --- TIMER START ---
        start_t = time.time()
        
        # Perform inference
        results = face_mesh.process(image_rgb)
        
        # --- TIMER END ---
        end_t = time.time()
        
        # Calculate FPS
        diff = end_t - start_t
        fps = 1 / diff if diff > 0 else 0

        # Visualization
        image.flags.writeable = True
        if VISUALIZE and results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

        cv2.putText(image, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Pi 5 Face Mesh', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break
            
except KeyboardInterrupt:
    print("Stopped by user")
finally:
    cap.release()
    cv2.destroyAllWindows()