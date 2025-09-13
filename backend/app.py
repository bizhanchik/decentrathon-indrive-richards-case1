from fastapi import FastAPI, File, UploadFile, HTTPException, status, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import os
import tempfile
import shutil
from pathlib import Path
import json
from typing import Dict, Any, List, Optional
import logging
import time
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import google.generativeai as genai
from dotenv import load_dotenv

# Import our inference system
from inference import CarDefectInference

# Load environment variables
load_dotenv()

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

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info("Gemini API configured successfully")
    except Exception as e:
        logger.error(f"Failed to configure Gemini API: {e}")
else:
    logger.warning("GEMINI_API_KEY not found in environment variables")

# Nano Banana model (Gemini 2.5 Flash Image Preview)
NANO_BANANA_MODEL = "gemini-2.5-flash-image-preview"

# Pydantic models for image generation
class TextToImageRequest(BaseModel):
    prompt: str
    image_size: str = "1024x1024"

class ImageToImageRequest(BaseModel):
    image_base64: str
    prompt: str = ""

class GeneratedImageResponse(BaseModel):
    image_base64: str
    message: str = "Image successfully generated"

def find_latest_checkpoint() -> str:
    """Find the latest trained model checkpoint"""
    # Look for runs directory in parent directory (since we're in backend/)
    runs_dir = Path("../runs")
    if not runs_dir.exists():
        # Try current directory as fallback
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

async def analyze_and_repair_with_gemini(image_path: str) -> tuple[int, Optional[str]]:
    """
    Use Gemini to check for damage and provide a mock repaired image.
    Note: Gemini 2.5 Flash Image Preview doesn't support image generation,
    so we'll detect damage and return a mock repair for demonstration.
    
    Args:
        image_path: Path to the car image
        
    Returns:
        Tuple of (damage_detected, repaired_image_base64)
        damage_detected: 0 if no damage, 1 if damage detected
        repaired_image_base64: Base64 string of mock repaired image if damage found, None otherwise
    """
    try:
        if not GEMINI_API_KEY or GEMINI_API_KEY == 'your-gemini-api-key-here':
            logger.warning("Gemini API key not configured, returning mock result")
            # Mock response for testing - assume damage detected and return original image
            with open(image_path, 'rb') as img_file:
                return 1, base64.b64encode(img_file.read()).decode('utf-8')
            
        # Load and encode image for Gemini
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
            
        # Create Gemini model for damage detection only
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Prepare the image for Gemini
        image_parts = [
            {
                "mime_type": "image/jpeg",
                "data": image_data
            }
        ]
        
        # Create prompt for damage detection only
        prompt = """
        Carefully examine this car image for ANY visible damage or imperfections including:
        - Dents (even small ones)
        - Scratches (any visible marks on paint)
        - Rust or corrosion
        - Cracks in body panels or windows
        - Missing parts or trim
        - Paint damage, chips, or discoloration
        - Worn or damaged bumpers
        - Any other visible defects or wear
        
        Look closely at all areas of the vehicle. Even minor damage should be detected.
        
        Respond with ONLY a single number:
        0 = Absolutely no visible damage or wear detected
        1 = Any visible damage, wear, or imperfections detected
        
        Be thorough in your assessment - if you see ANY imperfection, return 1.
        """
        
        # Generate response for damage detection
        response = model.generate_content([prompt] + image_parts)
        
        # Parse response for damage status
        damage_detected = 0
        repaired_image = None
        
        result_text = response.text.strip()
        if '1' in result_text:
            damage_detected = 1
            # Generate actual repaired image using Nano Banana
            repaired_image = await generate_repaired_car_with_nanobanana(image_path)
            logger.info("Damage detected - generating repaired image with Nano Banana")
        else:
            logger.info("No damage detected")
        
        return damage_detected, repaired_image
            
    except Exception as e:
        logger.error(f"Error in Gemini analysis and repair: {e}")
        # Return 0 on error to avoid issues
        return 0, None

async def generate_repaired_car_with_nanobanana(image_path: str) -> Optional[str]:
    """
    Generate a repaired car image using the Nano Banana model (Gemini 2.5 Flash Image Preview).
    """
    try:
        if not GEMINI_API_KEY:
            logger.error("Gemini API key not available")
            return None
            
        # Load the image
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # Prepare image for Gemini
        image_parts = [
            {
                "mime_type": "image/jpeg",
                "data": image_data
            }
        ]
        
        # Create a detailed repair prompt for Nano Banana
        repair_prompt = """
        You are an expert automotive repair AI. Looking at this damaged car image, generate a high-quality repaired version that shows:
        
        1. All visible damage completely fixed (dents removed, scratches eliminated, rust cleaned)
        2. Perfect paint finish with matching colors
        3. Restored body panels to original condition
        4. Clean, professional automotive appearance
        5. Maintain the same car model, color scheme, and lighting conditions
        6. Keep the same camera angle and background
        
        Generate a photorealistic image of this exact car but in perfect, like-new condition. The repair should look professional and seamless, as if done by expert automotive technicians.
        """
        
        # Use Nano Banana model for image generation
        model = genai.GenerativeModel(NANO_BANANA_MODEL)
        
        # Generate the repaired image
        response = model.generate_content([repair_prompt] + image_parts)
        
        # Extract the generated image from response
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        # Get the generated image data
                        generated_image_data = part.inline_data.data
                        return base64.b64encode(generated_image_data).decode('utf-8')
        
        logger.warning("No image generated by Nano Banana model")
        return None
        
    except Exception as e:
        logger.error(f"Error generating repaired image with Nano Banana: {e}")
        return None

# This function is now replaced by analyze_and_repair_with_gemini
# Keeping for backward compatibility if needed elsewhere
async def generate_fixed_car_with_nanobanana(image_path: str) -> Optional[str]:
    """
    DEPRECATED: Use analyze_and_repair_with_gemini instead.
    This function is kept for backward compatibility.
    """
    damage_detected, repaired_image = await analyze_and_repair_with_gemini(image_path)
    return repaired_image if damage_detected else None

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
async def analyze_image(file: UploadFile = File(...), real_time_detection: bool = Query(False, description="If true, only use ML model for faster analysis. If false, use full analysis with Gemini and Nano Banana.")):
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
            
            # Conditional analysis based on real_time_detection parameter
            if real_time_detection:
                # Real-time mode: Only use ML model for speed
                logger.info(f"Real-time analysis completed for {file.filename} (ML model only)")
                result['ai_analysis'] = {
                    'damage_detected': len(result.get('heatmap', {}).get('areas', [])) > 0,
                    'ai_repaired_image': None,  # Not generated in real-time mode
                    'mode': 'real_time'
                }
            else:
                # Full analysis mode: Use Gemini and Nano Banana
                logger.info(f"Analyzing and repairing image with Gemini: {file.filename}")
                damage_detected, ai_repaired_image = await analyze_and_repair_with_gemini(temp_file_path)
                
                if damage_detected == 1:
                    if ai_repaired_image:
                        logger.info(f"Damage detected and AI-repaired image generated successfully for {file.filename}")
                    else:
                        logger.warning(f"Damage detected but failed to generate AI-repaired image for {file.filename}")
                else:
                    logger.info(f"No damage detected by Gemini for {file.filename}")
                
                # Add AI results to response
                result['ai_analysis'] = {
                    'damage_detected': damage_detected == 1,
                    'ai_repaired_image': ai_repaired_image,
                    'mode': 'full_analysis'
                }
            
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

@app.post("/generate-image", response_model=GeneratedImageResponse)
async def generate_image_from_text(request: TextToImageRequest):
    """
    Generate an image based on text prompt using Nano Banana (Gemini 2.5 Flash Image Preview).
    """
    try:
        if not GEMINI_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Gemini API key not configured"
            )
        
        # Use Gemini 2.5 Flash Image Preview (Nano Banana) for image generation
        model = genai.GenerativeModel(model_name=NANO_BANANA_MODEL)
        response = model.generate_content(
            contents=[request.prompt]
        )
        
        # Extract the generated image from response
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    image_data = base64.b64encode(part.inline_data.data).decode("utf-8")
                    return GeneratedImageResponse(image_base64=image_data)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate image or get it from response"
        )
        
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating image: {e}"
        )

@app.post("/edit-image", response_model=GeneratedImageResponse)
async def edit_image_with_text(request: ImageToImageRequest):
    """
    Edit an existing image with text prompt using Nano Banana (Gemini 2.5 Flash Image Preview).
    """
    try:
        if not GEMINI_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Gemini API key not configured"
            )
        
        # Decode the image from base64
        image_bytes = base64.b64decode(request.image_base64)
        input_image = Image.open(io.BytesIO(image_bytes))
        
        # Use Gemini 2.5 Flash Image Preview (Nano Banana) for image editing
        model = genai.GenerativeModel(model_name=NANO_BANANA_MODEL)
        response = model.generate_content(
            contents=[input_image, request.prompt]
        )
        
        # Extract the edited image from response
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    edited_image_data = base64.b64encode(part.inline_data.data).decode("utf-8")
                    return GeneratedImageResponse(image_base64=edited_image_data)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to edit image or get it from response"
        )
        
    except Exception as e:
        logger.error(f"Error editing image: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error editing image: {e}"
        )

async def analyze_frame_for_damage(image_data: bytes, real_time_detection: bool = True) -> Dict[str, Any]:
    """
    Analyze a single video frame for car damage detection.
    
    Args:
        image_data: Raw image bytes
        real_time_detection: If True, only use ML model (faster). If False, use Gemini + Nano Banana (slower but more comprehensive)
    
    Optimized for real-time processing when real_time_detection=True.
    """
    try:
        # Save frame to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_file.write(image_data)
            temp_path = temp_file.name
        
        try:
            # Use existing inference engine for damage detection
            if inference_engine is None:
                return {"error": "Inference engine not initialized"}
            
            # Run YOLO detection
            analysis = inference_engine.predict_image(temp_path)
            
            # Convert to frontend format
            img = Image.open(temp_path)
            img_width, img_height = img.size
            formatted_analysis = convert_analysis_to_frontend_format(analysis, img_width, img_height)
            
            # Determine if damage is detected
            damage_detected = len(formatted_analysis.get('heatmap', {}).get('areas', [])) > 0
            
            # Conditional analysis based on real_time_detection flag
            if real_time_detection:
                # Real-time mode: Only use ML model for speed
                result = {
                    "damage_detected": damage_detected,
                    "yolo_detections": len(formatted_analysis.get('heatmap', {}).get('areas', [])),
                    "gemini_confirmed": False,  # Not used in real-time mode
                    "confidence": "high" if damage_detected else "low",
                    "timestamp": int(time.time() * 1000),
                    "mode": "real_time",
                    "heatmap": formatted_analysis.get('heatmap', {'areas': []}),  # Include bbox coordinates
                    "integrity": formatted_analysis.get('integrity', {'damaged': False, 'damageLevel': 'none'})
                }
                return result
            else:
                # Full analysis mode: Use Gemini for additional analysis if YOLO detects something
                gemini_damage = 0
                if damage_detected and GEMINI_API_KEY:
                    gemini_damage, _ = await analyze_and_repair_with_gemini(temp_path)
                
                return {
                    "damage_detected": damage_detected or gemini_damage == 1,
                    "yolo_detections": len(formatted_analysis.get('heatmap', {}).get('areas', [])),
                    "gemini_confirmed": gemini_damage == 1,
                    "confidence": "high" if damage_detected and gemini_damage == 1 else "medium" if damage_detected or gemini_damage == 1 else "low",
                    "timestamp": int(time.time() * 1000),
                    "mode": "full_analysis"
                }
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        logger.error(f"Error analyzing frame: {e}")
        return {"error": str(e)}

@app.websocket("/ws/realtime-detection")
async def websocket_realtime_detection(websocket: WebSocket):
    """
    WebSocket endpoint for real-time car damage detection.
    Receives video frames and returns damage analysis results.
    """
    await websocket.accept()
    logger.info("WebSocket connection established for real-time detection")
    
    try:
        while True:
            # Receive frame data from frontend
            data = await websocket.receive_bytes()
            
            # Analyze the frame in real-time mode (faster, ML-only)
            result = await analyze_frame_for_damage(data, real_time_detection=True)
            
            # Send result back to frontend
            await websocket.send_json(result)
            
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )