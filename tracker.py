import os
import cv2
import requests
from ultralytics import YOLO

# 1. Initialize YOLOv8 Model
print("Loading YOLOv8 model...")
model = YOLO('best.pt')

# 2. Interactive Route Setup (Initial & Final Destination)
print("--- BEL Urban Intelligence Route Setup ---")
try:
    start_lat = float(input("Enter Starting Latitude (default 28.6139): ") or 28.6139)
    start_lng = float(input("Enter Starting Longitude (default 77.2090): ") or 77.2090)
    end_lat = float(input("Enter Final Latitude (default 28.6500): ") or 28.6500)
    end_lng = float(input("Enter Final Longitude (default 77.2300): ") or 77.2300)
except ValueError:
    print("Invalid input. Using default New Delhi coordinates.")
    start_lat, start_lng, end_lat, end_lng = 28.6139, 77.2090, 28.6500, 77.2300

current_lat, current_lng = start_lat, start_lng
total_steps = 1000
lat_step = (end_lat - start_lat) / total_steps
lng_step = (end_lng - start_lng) / total_steps

# 3. Check for Video Source (Webcam or File)
# Change video_file to a file name like "test_traffic.mp4" if using recorded footage, or keep 0 for webcam
video_file = 0 
if os.path.exists(str(video_file)) and not isinstance(video_file, int):
    print(f"🎬 Using video file: {video_file}")
    cap = cv2.VideoCapture(video_file)
else:
    print("⚠️ Using laptop webcam (0)...")
    cap = cv2.VideoCapture(0)

API_URL = "http://127.0.0.1:8000/api/detections"
processed_ids = set()

print("\n🚀 CV Tracker started! Press 'q' on the video window to quit.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or cannot read frame.")
        break

    # 4. Run Inference & Tracking using ByteTrack
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml", 
        verbose=False
    )

    # 5. Process Detection Results
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, track_id, conf, cls in zip(boxes, ids, confs, classes):
            cls_name = model.names[cls]
            
            # Only send HTTP request if it's new and confidence is above 60% (filters out false triggers/faces)
            if track_id not in processed_ids and conf > 0.60:
                payload = {
                    "lat": current_lat,
                    "lng": current_lng,
                    "cls_name": cls_name,
                    "confidence": float(conf),
                    "tracker_id": int(track_id)
                }
                
                try:
                    res = requests.post(API_URL, json=payload, timeout=1)
                    if res.status_code == 200:
                        processed_ids.add(track_id)
                        print(f"📡 Detected & Sent: {cls_name.upper()} (ID #{track_id}, Conf: {conf:.2f}) -> Dashboard")
                except Exception:
                    print("⚠️ Could not reach Backend. Make sure backend.py is running on port 8000!")

            # Draw visual bounding box and label
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, 
                f"{cls_name} #{track_id} ({conf:.2f})", 
                (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 0), 
                2
            )

    # Progress GPS coordinates smoothly toward the final destination
    if current_lat < end_lat:
        current_lat += lat_step
        current_lng += lng_step

    # Display video window
    cv2.imshow("Bus Onboard Camera Feed - BEL Urban Intel", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()