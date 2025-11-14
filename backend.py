import pickle
import cv2
import face_recognition
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
import os
import csv
from datetime import datetime
from pydantic import BaseModel

# --- Global Variables ---
known_embeddings_global = None
known_names_global = None
THRESHOLD = 0.5
LOG_FILE = '/Users/amanpathak/Downloads/attendance_log.csv'

# --- Pydantic Model for Request Body ---
# This ensures the app sends a JSON with a "name" field
class AttendanceRecord(BaseModel):
    name: str

# --- Load Model on Startup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global known_embeddings_global, known_names_global
    print("Loading attendance database...")
    try:
        with open('/Users/amanpathak/Downloads/attendance_data_fixed.pkl', 'rb') as f:
            known_embeddings_global, known_names_global = pickle.load(f)
        print(f"✅ Database loaded with {len(known_names_global)} records.")
    except FileNotFoundError:
        print("❌ CRITICAL: '/Users/amanpathak/Downloads/attendance_data_fixed.pkl' not found.")
    yield
    print("Server shutting down.")

# Initialize the FastAPI app
app = FastAPI(lifespan=lifespan)

# --- Endpoint 1: Identity Check ---
@app.post("/check_attendance")
async def check_attendance(file: UploadFile = File(...)):
    """
    Receives an image file, finds a face, and checks for a match.
    """
    if known_embeddings_global is None:
        raise HTTPException(status_code=500, detail="Server is not ready. Model data not loaded.")

    # 1. Read and decode image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 2. Find face
    face_locations = face_recognition.face_locations(rgb_img)
    if not face_locations:
        raise HTTPException(status_code=404, detail="No face found in the image.")
        
    face_encoding = face_recognition.face_encodings(rgb_img, face_locations)[0]
    
    # 3. Compare with database
    face_distances = face_recognition.face_distance(known_embeddings_global, face_encoding)
    best_match_index = np.argmin(face_distances)
    best_distance = face_distances[best_match_index]

    if best_distance < THRESHOLD:
        name = known_names_global[best_match_index]
        return {
            "status": "Present",
            "name": name,
            "confidence_score": (1.0 - best_distance)
        }
    else:
        return {
            "status": "Absent",
            "name": "Unknown",
            "confidence_score": (1.0 - best_distance)
        }

# --- Endpoint 2: Log Attendance (NEW) ---
@app.post("/mark_present")
async def mark_present(record: AttendanceRecord):
    """
    Receives a student's name (post-liveness) and logs it to a CSV.
    """
    try:
        name = record.name
        now = datetime.now()
        date_string = now.strftime('%Y-%m-%d')
        time_string = now.strftime('%H:%M:%S')
        
        # Check if file needs a header
        file_exists = os.path.isfile(LOG_FILE)
        
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Write header if it's a new file
            if not file_exists:
                writer.writerow(['Name', 'Date', 'Time'])
                
            # Write the attendance record
            writer.writerow([name, date_string, time_string])
            
        print(f"Logged: {name} at {time_string}")
        return {
            "status": "success",
            "message": f"Attendance logged for {name}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error logging attendance: {e}")

# --- Main block to run the server ---
if __name__ == "__main__":
    # Runs the server on http://127.0.0.1:8000
    # Use host="0.0.0.0" to make it accessible on your network
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)