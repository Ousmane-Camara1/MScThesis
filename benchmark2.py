import cv2
import time
import mediapipe as mp

# --- CONFIGURATION ---
# Replace '0' with 'video_filename.mp4' to test on recorded videos
VIDEO_SOURCE = '2ndvideo.mp4'
# Set to 1 for your first test, then increase for the multi-person test
MAX_NUM_FACES = 5
# Set to False to measure raw processing speed without drawing overhead
VISUALIZE = True  

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=5,              # Ensure this is > 1 for the crowd video!
    refine_landmarks=False,       # Turn OFF iris tracking (it fails on small faces and slows you down)
    min_detection_confidence=0.2, # Drop from 0.5 to 0.2 to catch "fainter" faces
    min_tracking_confidence=0.2
)

# Initialize Video
cap = cv2.VideoCapture(VIDEO_SOURCE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # Lower resolution = Higher FPS
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_frame_time = 0
new_frame_time = 0

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
        
        # Calculate FPS (Instantaneous)
        fps = 1 / (end_t - start_t)

        # Draw the precise mesh landmarks
        image.flags.writeable = True
        if VISUALIZE and results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

        # Display FPS on screen
        cv2.putText(image, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Pi 5 Face Mesh', image)

        if cv2.waitKey(5) & 0xFF == 27: # Press ESC to exit
            break
            
except KeyboardInterrupt:
    print("Stopped by user")
finally:
    cap.release()
    cv2.destroyAllWindows()