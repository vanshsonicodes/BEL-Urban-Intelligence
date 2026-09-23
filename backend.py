import threading
import uvicorn
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from pydantic import BaseModel
from datetime import datetime
import requests
import time

# 1. Database Setup (SQLite)
# This creates a local file named 'urban_intel.db' to store the detections.
DATABASE_URL = "sqlite:///./urban_intel.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class DetectionDB(Base):
    __tablename__ = "detections"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    lat = Column(Float)
    lng = Column(Float)
    cls_name = Column(String, index=True) 
    confidence = Column(Float)
    tracker_id = Column(Integer)

# Create the tables in the database
Base.metadata.create_all(bind=engine)

# 2. Pydantic Schema (API Input Validation)
# This enforces the structure of the JSON data coming from Gaytri's CV pipeline.
class DetectionPayload(BaseModel):
    lat: float
    lng: float
    cls_name: str
    confidence: float
    tracker_id: int

# 3. FastAPI Initialization
app = FastAPI(title="BEL Urban Intel API")

# Allow the frontend dashboard to fetch data without CORS blocks
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# Database session dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 4. API Endpoints
@app.post("/api/detections")
def add_detection(payload: DetectionPayload, db: Session = Depends(get_db)):
    """Receives JSON from the CV tracker and saves it to SQLite."""
    new_det = DetectionDB(**payload.model_dump())
    db.add(new_det)
    db.commit()
    return {"status": "success", "id": new_det.id}

@app.get("/api/detections")
def get_detections(db: Session = Depends(get_db)):
    """Returns all detections to render on Vansh's map dashboard."""
    return db.query(DetectionDB).all()

# 5. Helper function to inject dummy data automatically on launch
def insert_dummy_data():
    """Runs in the background on startup to populate the database for frontend testing."""
    time.sleep(2) # Give the server 2 seconds to boot up
    url = "http://127.0.0.1:8000/api/detections"
    
    # Fake detections for Vansh to test his map pins
    dummy_records = [
        {"lat": 28.6139, "lng": 77.2090, "cls_name": "pothole", "confidence": 0.91, "tracker_id": 101},
        {"lat": 28.6150, "lng": 77.2100, "cls_name": "bus", "confidence": 0.85, "tracker_id": 102},
        {"lat": 28.6120, "lng": 77.2050, "cls_name": "illegal_dumping", "confidence": 0.78, "tracker_id": 103}
    ]
    
    for item in dummy_records:
        try:
            requests.post(url, json=item)
        except Exception:
            pass
            
    print("\n✅ Dummy data inserted! Check http://127.0.0.1:8000/api/detections\n")

if __name__ == "__main__":
    # Start the dummy data injector in a separate thread so it doesn't block the server
    threading.Thread(target=insert_dummy_data, daemon=True).start()
    
    print("🚀 Starting Server on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)