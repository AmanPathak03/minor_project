import cv2
import face_recognition
import numpy as np
import pickle
from datetime import datetime
import csv
import os

# --- CONFIGURATION ---
THRESHOLD = 0.5  # 0.5 is standard. Decrease to 0.4 if you get false positives.
DATA_FILE = '/Users/amanpathak/Downloads/attendance_data_fixed.pkl'
LOG_FILE = '/Users/amanpathak/Downloads/attendance_log.csv'

def load_data():
    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: '{DATA_FILE}' not found. Please download it from Kaggle.")
        return None, None
    
    print("🔄 Loading student database...")
    with open(DATA_FILE, 'rb') as f:
        embeddings, labels = pickle.load(f)
    print(f"✅ Loaded {len(embeddings)} records.")
    return embeddings, labels

def mark_in_csv(name):
    """Records attendance in a CSV file with the current time."""
    now = datetime.now()
    date_string = now.strftime('%Y-%m-%d')
    time_string = now.strftime('%H:%M:%S')
    
    # Check if we already marked this person today (optional optimization)
    # For now, we just append to the file
    if not os.path.isfile(LOG_FILE):
        with open(LOG_FILE, 'w') as f:
            f.write('Name,Date,Time\n')
            
    with open(LOG_FILE, 'a') as f:
        f.write(f'{name},{date_string},{time_string}\n')

def run_camera():
    known_embeddings, known_names = load_data()
    if known_embeddings is None: return

    video_capture = cv2.VideoCapture(0) # 0 = Default Webcam
    
    print("📷 Camera Started. Press 'q' to Quit.")

    while True:
        ret, frame = video_capture.read()
        if not ret: break

        # 1. Resize for speed (1/4th size)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # 2. Detect faces
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            # 3. Compare faces
            matches = face_recognition.compare_faces(known_embeddings, face_encoding, tolerance=THRESHOLD)
            face_distances = face_recognition.face_distance(known_embeddings, face_encoding)

            name = "Unknown"
            color = (0, 0, 255) # Red

            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index]:
                name = known_names[best_match_index]
                color = (0, 255, 0) # Green
                
                # Optional: Auto-mark attendance
                # mark_in_csv(name) 

            # 4. Draw Box (Scale coordinates back up by 4)
            top, right, bottom, left = face_location
            top *= 4; right *= 4; bottom *= 4; left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Attendance System', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_camera()