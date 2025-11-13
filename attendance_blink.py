import cv2
import face_recognition
import numpy as np
import pickle
import os
from datetime import datetime

# --- CONFIGURATION ---
THRESHOLD = 0.5
REQUIRED_BLINKS = 2        # How many times to blink
EYE_AR_THRESH = 0.23       # Threshold to decide if eye is closed (Adjust if needed)
DATA_FILE = '/Users/amanpathak/Downloads/attendance_data_fixed.pkl'

# --- LOAD DATA ---
def load_data():
    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: '{DATA_FILE}' not found.")
        return None, None
    with open(DATA_FILE, 'rb') as f:
        embeddings, labels = pickle.load(f)
    return embeddings, labels

# --- MATH: CALCULATE EYE ASPECT RATIO (EAR) ---
def eye_aspect_ratio(eye):
    # Calculate vertical distances
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    # Calculate horizontal distance
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    # EAR Formula
    ear = (A + B) / (2.0 * C)
    return ear

def run_system():
    known_embeddings, known_names = load_data()
    if known_embeddings is None: return

    video_capture = cv2.VideoCapture(0)
    
    # Blink State Variables
    blink_count = 0
    eyes_closed = False
    attendance_marked = False
    current_user_name = "Unknown"

    print(f"📷 System Active. Please Blink {REQUIRED_BLINKS} times to mark attendance.")

    while True:
        ret, frame = video_capture.read()
        if not ret: break

        # Resize for speed
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # 1. Detect Face Locations
        face_locations = face_recognition.face_locations(rgb_small_frame)
        
        # If no face is found, reset everything
        if len(face_locations) == 0:
            blink_count = 0
            attendance_marked = False
            current_user_name = "Unknown"
        
        else:
            # 2. Get Facial Landmarks (Points on eyes, nose, mouth)
            # We use the full frame for better landmark accuracy, or scale up the locations
            # For speed, let's compute landmarks on the small frame
            face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame, face_locations)

            for face_landmarks, face_location in zip(face_landmarks_list, face_locations):
                
                # --- BLINK DETECTION LOGIC ---
                left_eye = face_landmarks['left_eye']
                right_eye = face_landmarks['right_eye']

                ear_left = eye_aspect_ratio(left_eye)
                ear_right = eye_aspect_ratio(right_eye)
                avg_ear = (ear_left + ear_right) / 2.0

                # Check if eyes are closed
                if avg_ear < EYE_AR_THRESH:
                    eyes_closed = True
                # Check if eyes opened again (A blink happened)
                elif avg_ear >= EYE_AR_THRESH and eyes_closed:
                    blink_count += 1
                    eyes_closed = False # Reset state
                
                # --- VISUAL FEEDBACK ---
                top, right, bottom, left = face_location
                top *= 4; right *= 4; bottom *= 4; left *= 4 # Scale back up
                
                color = (0, 165, 255) # Orange (Waiting)
                msg = f"Blinks: {blink_count}/{REQUIRED_BLINKS}"

                # --- ATTENDANCE LOGIC ---
                if blink_count >= REQUIRED_BLINKS:
                    if not attendance_marked:
                        # Perform Face Recognition ONLY after blinks are done
                        face_encoding = face_recognition.face_encodings(rgb_small_frame, [face_location])[0]
                        
                        matches = face_recognition.compare_faces(known_embeddings, face_encoding, tolerance=THRESHOLD)
                        face_distances = face_recognition.face_distance(known_embeddings, face_encoding)
                        
                        best_match_index = np.argmin(face_distances)

                        if matches[best_match_index]:
                            current_user_name = known_names[best_match_index]
                            attendance_marked = True
                            print(f"✅ Attendance Marked: {current_user_name}")
                        else:
                            current_user_name = "Unknown Student"
                            # Reset blinks if unknown so they can try again
                            blink_count = 0 
                    
                    # Change UI to Green (Success)
                    color = (0, 255, 0)
                    msg = f"Present: {current_user_name}"

                # Draw Box and Text
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, msg, (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)

                # Optional: Display EAR value for debugging
                cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Liveness Attendance System', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_system()