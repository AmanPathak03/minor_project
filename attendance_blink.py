import cv2
import face_recognition
import numpy as np
import pickle
import os
import random
import mediapipe as mp
import time

# --- CONFIGURATION ---
DATA_FILE = '/Users/amanpathak/Downloads/attendance_data_fixed.pkl'
THRESHOLD_FACE = 0.5
MAR_THRESH = 0.4  # Mouth Aspect Ratio for Smiling (Adjust if needed)

# --- MEDIAPIPE SETUP (HANDS) ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# --- LOAD DATA ---
def load_data():
    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: '{DATA_FILE}' not found.")
        return None, None
    with open(DATA_FILE, 'rb') as f:
        embeddings, labels = pickle.load(f)
    return embeddings, labels

# --- MATH HELPERS ---
def calculate_mar(mouth):
    # Mouth Aspect Ratio (for Smile)
    A = np.linalg.norm(np.array(mouth[2]) - np.array(mouth[10])) # Vertical
    B = np.linalg.norm(np.array(mouth[4]) - np.array(mouth[8]))  # Vertical
    C = np.linalg.norm(np.array(mouth[0]) - np.array(mouth[6]))  # Horizontal
    mar = (A + B) / (2.0 * C)
    return mar

def count_fingers(hand_landmarks):
    # Tips IDs: Thumb=4, Index=8, Middle=12, Ring=16, Pinky=20
    # PIP IDs (Knuckles): Thumb=3, Index=6, Middle=10, Ring=14, Pinky=18
    
    cnt = 0
    # 1. Check Thumb (x-axis comparison depends on hand side, simplified here)
    # Assuming right hand facing camera: Tip x > IP x
    if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
        cnt += 1
    
    # 2. Check 4 Fingers (y-axis comparison: Tip should be higher than PIP)
    # Note: In image coordinates, 'higher' means a LOWER y value (0 is top)
    fingers = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    
    for tip, pip in zip(fingers, pips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            cnt += 1
            
    return cnt

def run_challenge_system():
    known_embeddings, known_names = load_data()
    if known_embeddings is None: return

    cap = cv2.VideoCapture(0)
    
    # State Machine Variables
    current_state = "SEARCHING" # SEARCHING -> CHALLENGING -> MARKED
    current_user = None
    assigned_task = None
    task_success_time = 0
    
    # List of possible tasks
    TASKS = ["SMILE", "SHOW 2 FINGERS", "SHOW 5 FINGERS"]
    
    print("📷 System Active.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Flip frame for mirror view (easier for user to interact)
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- 1. ALWAYS RUN HAND DETECTION ---
        hand_results = hands.process(rgb_frame)
        finger_count = 0
        if hand_results.multi_hand_landmarks:
            for hand_lms in hand_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                finger_count = count_fingers(hand_lms)

        # --- 2. RUN FACE DETECTION ---
        # Optimization: Run on smaller frame
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        face_locs = face_recognition.face_locations(rgb_small)
        
        if len(face_locs) == 0:
            current_state = "SEARCHING"
            current_user = None
            assigned_task = None
        else:
            # We found a face
            top, right, bottom, left = face_locs[0]
            # Scale up
            top *= 4; right *= 4; bottom *= 4; left *= 4
            
            # --- STATE: SEARCHING FOR IDENTITY ---
            if current_state == "SEARCHING":
                face_enc = face_recognition.face_encodings(rgb_small, [face_locs[0]])[0]
                matches = face_recognition.compare_faces(known_embeddings, face_enc, tolerance=THRESHOLD_FACE)
                dists = face_recognition.face_distance(known_embeddings, face_enc)
                
                best_idx = np.argmin(dists)
                
                if matches[best_idx]:
                    current_user = known_names[best_idx]
                    # Assign a random task!
                    assigned_task = random.choice(TASKS)
                    current_state = "CHALLENGING"
                    print(f"Identity Verified: {current_user}. Assigned Task: {assigned_task}")
                else:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            # --- STATE: DOING THE TASK ---
            elif current_state == "CHALLENGING":
                # Draw Orange Box (Pending)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 165, 255), 2)
                
                msg = ""
                task_passed = False
                
                # 1. CHECK FINGER TASKS
                if assigned_task == "SHOW 2 FINGERS":
                    msg = "Please Show 2 Fingers!"
                    if finger_count == 2: task_passed = True
                        
                elif assigned_task == "SHOW 5 FINGERS":
                    msg = "Show Open Palm (5 Fingers)!"
                    if finger_count == 5: task_passed = True
                
                # 2. CHECK SMILE TASK
                elif assigned_task == "SMILE":
                    msg = "Please Smile!"
                    # Get landmarks for smile check
                    landmarks = face_recognition.face_landmarks(rgb_small, [face_locs[0]])[0]
                    mouth = landmarks['top_lip'] + landmarks['bottom_lip']
                    mar = calculate_mar(mouth)
                    # cv2.putText(frame, f"MAR: {mar:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2) # Debug
                    if mar > MAR_THRESH: task_passed = True

                # Display Instruction
                cv2.putText(frame, msg, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
                
                if task_passed:
                    current_state = "MARKED"
                    task_success_time = time.time()

            # --- STATE: SUCCESS ---
            elif current_state == "MARKED":
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom-30), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, f"Present: {current_user}", (left+5, bottom-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                
                cv2.putText(frame, "✅ ATTENDANCE MARKED", (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
                
                # Reset after 3 seconds so the next student can come
                if time.time() - task_success_time > 3:
                    current_state = "SEARCHING"
                    current_user = None
                    assigned_task = None

        cv2.imshow('Random Challenge Attendance', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_challenge_system()