from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import tempfile
import shutil
from pathlib import Path
import json
from typing import Dict, Any, List, Optional
import logging
from PIL import Image
import io

# Import our inference system
from inference import CarDefectInference

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RichardsDrive Car Defect Detection API",
    description="AI-powered car defect detection and analysis",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference engine
inference_engine = None

def find_latest_checkpoint() -> str:
    """Find the latest trained model checkpoint"""
    # Use the specific trained model path from the latest training run
    best_model_path = Path("C:\\Users\\aidyn\\Desktop\\PR\\Decentraton\\decentrathon-indrive-richards-case1\\runs\\car_defect_detection3\\weights\\best.pt")
    
    if best_model_path.exists():
        return str(best_model_path)
    
    # Fallback to relative path search
    runs_dir = Path("runs")
    if not runs_dir.exists():
        raise FileNotFoundError("No runs directory found")
    
    # Look for the latest car_defect_detection run
    detection_dirs = list(runs_dir.glob("car_defect_detection*"))
    if not detection_dirs:
        raise FileNotFoundError("No car_defect_detection runs found")
    
    # Sort by modification time and get the latest
    latest_dir = max(detection_dirs, key=lambda x: x.stat().st_mtime)
    
    # Check for best.pt first, then last.pt
    weights_dir = latest_dir / "weights"
    if not weights_dir.exists():
        raise FileNotFoundError(f"No weights directory found in {latest_dir}")
    
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"
    
    if best_pt.exists():
        return str(best_pt)
    elif last_pt.exists():
        return str(last_pt)
    else:
        raise FileNotFoundError(f"No model weights found in {weights_dir}")

def initialize_inference_engine():
    """Initialize the inference engine with the latest model"""
    global inference_engine
    try:
        model_path = find_latest_checkpoint()
        logger.info(f"Loading model from: {model_path}")
        inference_engine = CarDefectInference(model_path)
        inference_engine.load_model()
        logger.info("Inference engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}")
        raise

def convert_analysis_to_frontend_format(analysis, img_width: int, img_height: int) -> Dict[str, Any]:
    """Convert our analysis format to match frontend expectations"""
    
    # Calculate cleanliness score based on defects
    if not analysis.car_detected:
        cleanliness_score = 0
        cleanliness_status = "poor"
        cleanliness_description = "No car detected in the image"
    elif not analysis.defect_groups:
        cleanliness_score = 95
        cleanliness_status = "excellent"
        cleanliness_description = "Vehicle appears to be in excellent condition with no visible defects"
    else:
        # Calculate score based on severity and number of defects
        # Convert severity strings to numeric scores
        severity_map = {'slight': 1, 'moderate': 3, 'severe': 5}
        total_severity = sum(severity_map.get(group.severity, 1) for group in analysis.defect_groups)
        num_defects = len(analysis.defect_groups)
        
        # Score calculation (higher severity and more defects = lower score)
        base_score = 100
        severity_penalty = min(total_severity * 10, 60)  # Max 60 points for severity
        count_penalty = min(num_defects * 5, 20)  # Max 20 points for count
        
        cleanliness_score = max(base_score - severity_penalty - count_penalty, 10)
        
        if cleanliness_score >= 80:
            cleanliness_status = "good"
            cleanliness_description = "Vehicle appears well-maintained with minor issues detected"
        elif cleanliness_score >= 60:
            cleanliness_status = "fair"
            cleanliness_description = "Vehicle shows moderate wear and defects that may need attention"
        else:
            cleanliness_status = "poor"
            cleanliness_description = "Vehicle has significant defects that require immediate attention"
    
    # Determine damage level
    damaged = len(analysis.defect_groups) > 0
    if not damaged:
        damage_level = "none"
        damage_description = "No structural damage detected"
    else:
        # Convert severity strings to numeric scores for comparison
        severity_map = {'slight': 1, 'moderate': 3, 'severe': 5}
        max_severity = max(severity_map.get(group.severity, 1) for group in analysis.defect_groups)
        if max_severity <= 2:
            damage_level = "minor"
            damage_description = f"Minor defects detected: {', '.join(set(group.defect_type for group in analysis.defect_groups))}"
        elif max_severity <= 4:
            damage_level = "moderate"
            damage_description = f"Moderate damage detected: {', '.join(set(group.defect_type for group in analysis.defect_groups))}"
        else:
            damage_level = "severe"
            damage_description = f"Severe damage detected: {', '.join(set(group.defect_type for group in analysis.defect_groups))}"
    
    # Create heatmap areas from defect groups
    heatmap_areas = []
    
    for group in analysis.defect_groups:
        # Convert bbox center to percentage coordinates
        if group.bbox:
            # Normalize pixel coordinates to 0-1, then convert to percentage
            x1_norm = float(group.bbox[0]) / img_width
            y1_norm = float(group.bbox[1]) / img_height
            x2_norm = float(group.bbox[2]) / img_width
            y2_norm = float(group.bbox[3]) / img_height
            
            x_center = ((x1_norm + x2_norm) / 2) * 100
            y_center = ((y1_norm + y2_norm) / 2) * 100
            
            # Map severity string to frontend severity levels
            severity_map = {'slight': 1, 'moderate': 3, 'severe': 5}
            severity_score = severity_map.get(group.severity, 1)
            if severity_score <= 2:
                severity = "low"
            elif severity_score <= 4:
                severity = "medium"
            else:
                severity = "high"
            
            heatmap_areas.append({
                "x": round(x_center, 1),
                "y": round(y_center, 1),
                "severity": severity,
                "description": f"{group.defect_type.title()} (confidence: {float(group.confidence):.2f})",
                "bbox": {
                    "x1": round(x1_norm * 100, 1),
                    "y1": round(y1_norm * 100, 1),
                    "x2": round(x2_norm * 100, 1),
                    "y2": round(y2_norm * 100, 1)
                },
                "defect_type": group.defect_type,
                "confidence": round(float(group.confidence), 3)
            })
    
    return {
        "cleanliness": {
            "score": int(cleanliness_score),
            "status": cleanliness_status,
            "description": cleanliness_description
        },
        "integrity": {
            "damaged": damaged,
            "damageLevel": damage_level,
            "description": damage_description
        },
        "heatmap": {
            "areas": heatmap_areas
        }
    }

@app.on_event("startup")
async def startup_event():
    """Initialize the inference engine on startup"""
    initialize_inference_engine()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "RichardsDrive Car Defect Detection API is running"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    global inference_engine
    return {
        "status": "healthy" if inference_engine is not None else "unhealthy",
        "model_loaded": inference_engine is not None,
        "version": "1.0.0"
    }

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Analyze uploaded car image for defects"""
    global inference_engine
    
    if inference_engine is None:
        raise HTTPException(status_code=500, detail="Inference engine not initialized")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Validate file size (10MB limit)
    max_size = 10 * 1024 * 1024  # 10MB
    file_content = await file.read()
    if len(file_content) > max_size:
        raise HTTPException(status_code=400, detail="File size exceeds 10MB limit")
    
    try:
        # Validate image format
        image = Image.open(io.BytesIO(file_content))
        image.verify()  # Verify it's a valid image
        
        # Reset file pointer and save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        try:
            # Get image dimensions
            import cv2
            img = cv2.imread(temp_file_path)
            img_height, img_width = img.shape[:2]
            
            # Run inference
            logger.info(f"Analyzing image: {file.filename}")
            analysis = inference_engine.predict_image(
                temp_file_path,
                conf_threshold=0.3,
                iou_threshold=0.5
            )
            
            # Convert to frontend format
            result = convert_analysis_to_frontend_format(analysis, img_width, img_height)
            
            logger.info(f"Analysis completed for {file.filename}")
            return JSONResponse(content=result)
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
            
    except Exception as e:
        logger.error(f"Error analyzing image {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    global inference_engine
    
    if inference_engine is None:
        raise HTTPException(status_code=500, detail="Inference engine not initialized")
    
    try:
        model_path = find_latest_checkpoint()
        return {
            "model_path": model_path,
            "class_names": inference_engine.class_names,
            "model_loaded": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )