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
from ensemble_logic import EnsembleLogic
# Add model2, model3, model4 to path and import
import sys
from pathlib import Path
model2_path = str(Path(__file__).parent / 'model2')
model3_path = str(Path(__file__).parent / 'model3')
model4_path = str(Path(__file__).parent / 'model4')
if model2_path not in sys.path:
    sys.path.append(model2_path)
if model3_path not in sys.path:
    sys.path.append(model3_path)
if model4_path not in sys.path:
    sys.path.append(model4_path)

# Import model2's DefectDetectionInference class
import importlib.util
spec = importlib.util.spec_from_file_location("model2_inference", Path(__file__).parent / 'model2' / 'inference.py')
model2_inference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model2_inference)
DefectDetectionInference = model2_inference.DefectDetectionInference

# Import model3's CarDamageDetectionInference class
spec3 = importlib.util.spec_from_file_location("model3_inference", Path(__file__).parent / 'model3' / 'inference.py')
model3_inference = importlib.util.module_from_spec(spec3)
spec3.loader.exec_module(model3_inference)
CarDamageDetectionInference3 = model3_inference.CarDamageDetectionInference

# Import model4's CarDamageDetectionInference class
spec4 = importlib.util.spec_from_file_location("model4_inference", Path(__file__).parent / 'model4' / 'inference.py')
model4_inference = importlib.util.module_from_spec(spec4)
spec4.loader.exec_module(model4_inference)
CarDamageDetectionInference4 = model4_inference.CarDamageDetectionInference

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

# Global inference engines
inference_engine = None
defect_inference_engine = None
model3_inference_engine = None
model4_inference_engine = None
ensemble_engine = None

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

    # Use the specified yolov8s model path
    yolov8s_model_path = Path("../runs/train/yolov8s/weights/best.pt")
    if yolov8s_model_path.exists():
        logger.info(f"Using yolov8s model: {yolov8s_model_path}")
        return str(yolov8s_model_path)
    
    # Fallback to relative path search
    runs_dir = Path("runs")
    if not runs_dir.exists():
        # Try current directory as fallback
        runs_dir = Path("runs")
        if not runs_dir.exists():
            raise FileNotFoundError("No runs directory found")
    
    # Look for the latest car_defect_detection run
    detection_dirs = list(runs_dir.glob("car_defect_detection*"))
    if not detection_dirs:
        # Also check for train/yolov8s directory
        train_dirs = list(runs_dir.glob("train/yolov8*"))
        if not train_dirs:
            raise FileNotFoundError("No training runs found")
        detection_dirs = train_dirs
    
    # Sort by modification time and get the latest
    latest_dir = max(detection_dirs, key=lambda x: x.stat().st_mtime)
    logger.info(f"Found latest training directory: {latest_dir}")
    
    # Check for best.pt first, then last.pt
    weights_dir = latest_dir / "weights"
    if not weights_dir.exists():
        raise FileNotFoundError(f"No weights directory found in {latest_dir}")
    
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"
    
    if best_pt.exists():
        logger.info(f"Using best checkpoint: {best_pt}")
        return str(best_pt)
    elif last_pt.exists():
        logger.info(f"No best checkpoint found, using last checkpoint: {last_pt}")
        return str(last_pt)
    else:
        raise FileNotFoundError(f"No model weights found in {weights_dir}")

def initialize_inference_engine(model_type: str = 's'):
    """Initialize the inference engine with the specified model type
    
    Args:
        model_type: Model type ('n' for YOLOv8n or 's' for YOLOv8s)
    """
    global inference_engine
    try:
        print("\n" + "="*80)
        print("🚗 INITIALIZING MAIN CAR DEFECT DETECTION MODEL")
        print("="*80)
        print(f"🔧 Model Type: YOLOv8{model_type.upper()}")
        
        # Try to find the best checkpoint first
        try:
            best_checkpoint = find_latest_checkpoint()
            print(f"📁 Model Path: {best_checkpoint}")
            print(f"✅ Using trained checkpoint")
            inference_engine = CarDefectInference(model_path=best_checkpoint, model_type=model_type)
        except FileNotFoundError as e:
            print(f"⚠️  Could not find checkpoint: {e}. Using default model.")
            inference_engine = CarDefectInference(model_type=model_type)
        
        print("🎯 Classes: Dent, Dislocation, Scratch, Shatter, Damaged, Severe Damage")
        print("✅ STATUS: MAIN MODEL LOADED SUCCESSFULLY!")
        print("="*80 + "\n")
        logger.info("🚀 Main inference engine initialized successfully")
    except Exception as e:
        print("❌ STATUS: MAIN MODEL FAILED TO LOAD!")
        print("="*80 + "\n")
        logger.error(f"💥 Failed to initialize inference engine: {e}")
        raise

def initialize_defect_inference_engine():
    """Initialize the crack detection inference engine"""
    global defect_inference_engine
    
    try:
        print("="*80)
        print("🔍 INITIALIZING MODEL2 - ADVANCED DEFECT DETECTION")
        print("="*80)
        
        # Use the correct absolute path for model2
        model_path = Path("../runs/train/model2/weights/best.pt")
        print(f"📁 Model Path: {model_path}")
        
        if not model_path.exists():
            print("❌ STATUS: MODEL2 NOT FOUND!")
            print("⚠️  Advanced defect detection will be unavailable")
            print("="*80 + "\n")
            logger.warning(f"🚫 Model2 not found at {model_path}, defect detection (model2) will be unavailable")
            defect_inference_engine = None
            return
        
        # Initialize the crack detection inference engine with 55% confidence threshold
        defect_inference_engine = DefectDetectionInference(model_path=str(model_path), confidence_threshold=0.55)
        print("🎯 Specialized for: Advanced crack and defect detection")
        print("💻 Device: CPU")
        print("🔧 Confidence Threshold: 0.55 (55%)")
        print("✅ STATUS: MODEL2 LOADED SUCCESSFULLY!")
        print("="*80 + "\n")
        logger.info(f"🚀 Defect detection engine (model2) initialized with model: {model_path}")
        
    except Exception as e:
        print("❌ STATUS: MODEL2 FAILED TO LOAD!")
        print(f"💥 Error: {e}")
        print("="*80 + "\n")
        logger.error(f"💥 Failed to initialize defect inference engine: {e}")
        defect_inference_engine = None

def initialize_model3_inference_engine():
    """Initialize the model3 inference engine"""
    global model3_inference_engine
    
    try:
        print("="*80)
        print("🔍 INITIALIZING MODEL3 - SPECIALIZED DEFECT DETECTION")
        print("="*80)
        
        # Use the correct absolute path for model3
        model_path = Path("../runs/train/model3/weights/best.pt")
        print(f"📁 Model Path: {model_path}")
        
        if not model_path.exists():
            print("❌ STATUS: MODEL3 NOT FOUND!")
            print("⚠️  Model3 defect detection will be unavailable")
            print("="*80 + "\n")
            logger.warning(f"🚫 Model3 not found at {model_path}, model3 detection will be unavailable")
            model3_inference_engine = None
            return
        
        # Initialize the model3 inference engine with 25% confidence threshold
        model3_inference_engine = CarDamageDetectionInference3(model_path=str(model_path), confidence_threshold=0.25)
        print("🎯 Specialized for: Model3 defect detection")
        print("💻 Device: CPU")
        print("🔧 Confidence Threshold: 0.25 (25%)")
        print("✅ STATUS: MODEL3 LOADED SUCCESSFULLY!")
        print("="*80 + "\n")
        logger.info(f"🚀 Model3 inference engine initialized with model: {model_path}")
        
    except Exception as e:
        print("❌ STATUS: MODEL3 FAILED TO LOAD!")
        print(f"💥 Error: {e}")
        print("="*80 + "\n")
        logger.error(f"💥 Failed to initialize model3 inference engine: {e}")
        model3_inference_engine = None

def initialize_model4_inference_engine():
    """Initialize the model4 inference engine"""
    global model4_inference_engine
    
    try:
        print("="*80)
        print("🔍 INITIALIZING MODEL4 - ADVANCED DEFECT DETECTION")
        print("="*80)
        
        # Use the correct absolute path for model4
        model_path = Path("../runs/train/model4/weights/best.pt")
        print(f"📁 Model Path: {model_path}")
        
        if not model_path.exists():
            print("❌ STATUS: MODEL4 NOT FOUND!")
            print("⚠️  Model4 defect detection will be unavailable")
            print("="*80 + "\n")
            logger.warning(f"🚫 Model4 not found at {model_path}, model4 detection will be unavailable")
            model4_inference_engine = None
            return
        
        # Initialize the model4 inference engine with 55% confidence threshold
        model4_inference_engine = CarDamageDetectionInference4(model_path=str(model_path), confidence_threshold=0.55)
        print("🎯 Specialized for: Model4 defect detection")
        print("💻 Device: CPU")
        print("🔧 Confidence Threshold: 0.55 (55%)")
        print("✅ STATUS: MODEL4 LOADED SUCCESSFULLY!")
        print("="*80 + "\n")
        logger.info(f"🚀 Model4 inference engine initialized with model: {model_path}")
        
    except Exception as e:
        print("❌ STATUS: MODEL4 FAILED TO LOAD!")
        print(f"💥 Error: {e}")
        print("="*80 + "\n")
        logger.error(f"💥 Failed to initialize model4 inference engine: {e}")
        model4_inference_engine = None

def convert_analysis_to_frontend_format(analysis, img_width: int, img_height: int) -> Dict[str, Any]:
    """Convert our analysis format to match frontend expectations"""
    
    # Filter out scratch classifications for yolov8s model
    if hasattr(analysis, 'defect_groups') and analysis.defect_groups:
        filtered_groups = []
        for group in analysis.defect_groups:
            # Skip scratch classifications
            if group.defect_type.lower() != 'scratch':
                filtered_groups.append(group)
        analysis.defect_groups = filtered_groups
    
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
        severity_values = []
        for group in analysis.defect_groups:
            severity_val = severity_map.get(group.severity, 1)
            if isinstance(severity_val, (int, float)):
                severity_values.append(severity_val)
            else:
                severity_values.append(1)  # Default fallback
        total_severity = sum(severity_values)
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
    """Initialize all inference engines on startup"""
    global ensemble_engine
    # Use trained YOLOv8s model at startup
    initialize_inference_engine(model_type='s')
    # Initialize all defect detection models
    initialize_defect_inference_engine()
    initialize_model3_inference_engine()
    initialize_model4_inference_engine()
    # Initialize ensemble engine
    ensemble_engine = EnsembleLogic()
    
    # Beautiful startup banner
    print("\n" + "="*80)
    print("🚀 RICHARDSDRIVE CAR DEFECT DETECTION API")
    print("="*80)
    print("🌟 System Status: READY")
    print("🔗 Server URL: http://0.0.0.0:8000")
    print("📚 API Docs: http://0.0.0.0:8000/docs")
    print("\n🤖 AI Models Status:")
    if inference_engine is not None:
        print("   ✅ Main Model (YOLOv8s): ACTIVE")
        print("   📊 Classes: 6 defect types")
    else:
        print("   ❌ Main Model: FAILED")
    
    if defect_inference_engine is not None:
        print("   ✅ Model2 (Advanced): ACTIVE")
        print("   🔍 Specialized: Crack & defect detection")
    else:
        print("   ⚠️  Model2: UNAVAILABLE")
    
    if model3_inference_engine is not None:
        print("   ✅ Model3 (Specialized): ACTIVE")
        print("   🔍 Specialized: Model3 defect detection")
    else:
        print("   ⚠️  Model3: UNAVAILABLE")
    
    if model4_inference_engine is not None:
        print("   ✅ Model4 (Advanced): ACTIVE")
        print("   🔍 Specialized: Model4 defect detection")
    else:
        print("   ⚠️  Model4: UNAVAILABLE")
    
    print("\n🛠️  Available Endpoints:")
    print("   📤 POST /analyze - Single model analysis")
    print("   🔄 POST /analyze-dual - Dual model analysis")
    print("   🎯 POST /analyze-all - All 4 models analysis")
    print("   ❤️  GET /health - Health check")
    print("   ℹ️  GET /model-info - Model information")
    print("\n🎯 Ready to detect car defects with AI precision!")
    print("="*80 + "\n")
    logger.info("🚀 RichardsDrive API startup completed successfully")

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
async def analyze_image(file: UploadFile = File(...), real_time_detection: bool = Query(False, description="If true, only use ML model for faster analysis. If false, use full analysis with Gemini and Nano Banana."), model_type: str = 's'):
    """Analyze uploaded car image for defects"""

    global inference_engine
    
    # Validate model type - only 's' (YOLOv8s) is supported as we only have the trained model
    if model_type.lower() != 's':
        raise HTTPException(status_code=400, detail="Invalid model type. Only 's' for YOLOv8s is supported")
    
    # Initialize inference engine with the specified model type if needed
    if inference_engine is None or inference_engine.model_type != model_type.lower():
        logger.info(f"Switching to model type: {model_type.lower()}")
        initialize_inference_engine(model_type=model_type.lower())
    
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
            logger.info(f"Analyzing image: {file.filename} with model type: {model_type}")
            analysis = inference_engine.predict_image(
                temp_file_path,
                conf_threshold=0.55,  # High confidence threshold to show only reliable detections
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

            # Add model type information to the result (use actual loaded model type)
            actual_model_type = inference_engine.model_type
            result['model_info'] = {
                'type': actual_model_type,
                'name': f"YOLOv8{actual_model_type}",
                'requested_type': model_type.lower(),
                'fallback_used': actual_model_type != model_type.lower()
            }
            
            logger.info(f"Analysis completed for {file.filename}")
            return JSONResponse(content=result)
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
            
    except Exception as e:
        logger.error(f"Error analyzing image {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")

@app.post("/analyze-all")
async def analyze_image_all_models(file: UploadFile = File(...)):
    """Analyze uploaded car image using Gemini + 4 models for comprehensive detection
    
    Process Flow:
    1. Gemini analyzes image for damage detection (returns 0 or 1)
    2. If damage detected (1), Nano Banana generates repaired image
    3. All 4 models (main, model2, model3, model4) analyze the image
    4. Combined results returned including Gemini analysis and repaired image
    
    Args:
        file: The uploaded car image
    
    Returns:
        Combined analysis from Gemini + all 4 models with optional repaired image
    """
    global inference_engine, defect_inference_engine, model3_inference_engine, model4_inference_engine
    
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
            
            # Step 1: Use Gemini for initial damage detection
            logger.info(f"Running Gemini damage detection on: {file.filename}")
            gemini_damage_detected, repaired_image = await analyze_and_repair_with_gemini(temp_file_path)
            
            # Initialize engines if needed
            if inference_engine is None:
                initialize_inference_engine(model_type='s')
            if defect_inference_engine is None:
                initialize_defect_inference_engine()
            if model3_inference_engine is None:
                initialize_model3_inference_engine()
            if model4_inference_engine is None:
                initialize_model4_inference_engine()
            
            # Run analysis with main model
            logger.info(f"Running main defect analysis on: {file.filename}")
            main_analysis = None
            if inference_engine is not None:
                main_analysis = inference_engine.predict_image(
                    temp_file_path,
                    conf_threshold=0.55,
                    iou_threshold=0.5
                )
            
            # Run analysis with model2
            logger.info(f"Running model2 analysis on: {file.filename}")
            model2_analysis = None
            if defect_inference_engine is not None:
                model2_result = defect_inference_engine.predict_single(temp_file_path)
                model2_analysis = {
                    'detections': model2_result.get('detections', []),
                    'num_detections': model2_result.get('num_detections', 0),
                    'inference_time': model2_result.get('inference_time_ms', 0)
                }
            
            # Run analysis with model3
            logger.info(f"Running model3 analysis on: {file.filename}")
            model3_analysis = None
            if model3_inference_engine is not None:
                model3_result = model3_inference_engine.predict_single(temp_file_path)
                model3_analysis = {
                    'detections': model3_result.get('detections', []),
                    'num_detections': model3_result.get('num_detections', 0),
                    'inference_time': model3_result.get('inference_time_ms', 0)
                }
                
                # Apply conflict resolution to Model 3 before sending to frontend
                if ensemble_engine and model3_analysis:
                    resolved_model3_analysis = ensemble_engine.resolve_model3_conflicts(model3_analysis)
                    if resolved_model3_analysis and resolved_model3_analysis.get('conflict_resolved'):
                        logger.info(f"Model 3 conflicts resolved: {resolved_model3_analysis['resolved_detection_count']} detections kept out of {resolved_model3_analysis['original_detection_count']} original")
                        model3_analysis = resolved_model3_analysis
            
            # Run analysis with model4
            logger.info(f"Running model4 analysis on: {file.filename}")
            model4_analysis = None
            if model4_inference_engine is not None:
                model4_result = model4_inference_engine.predict_single(temp_file_path)
                model4_analysis = {
                    'detections': model4_result.get('detections', []),
                    'num_detections': model4_result.get('num_detections', 0),
                    'inference_time': model4_result.get('inference_time_ms', 0)
                }
            
            # Combine results from all models including Gemini
            combined_result = {
                'gemini_analysis': {
                    'damage_detected': gemini_damage_detected == 1,
                    'repaired_image': repaired_image if gemini_damage_detected == 1 else None,
                    'model_info': {
                        'type': 'gemini',
                        'name': 'Gemini - AI Damage Detection & Nano Banana Repair',
                        'description': 'Primary damage detection with AI-powered repair'
                    }
                },
                'main_model': {
                    'available': main_analysis is not None,
                    'analysis': convert_analysis_to_frontend_format(main_analysis, img_width, img_height) if main_analysis else None,
                    'model_info': {
                        'type': 's',
                        'name': 'YOLOv8s - Multi-class Defect Detection',
                        'classes': inference_engine.class_names if inference_engine else []
                    }
                },
                'model2': {
                    'available': model2_analysis is not None,
                    'analysis': convert_defect_analysis_to_frontend_format(model2_analysis, img_width, img_height) if model2_analysis else None,
                    'model_info': {
                        'type': 'defect_detection_model2',
                        'name': 'Model2 - Specialized Defect Detection',
                        'classes': ['crack', 'dent', 'glass shatter', 'lamp broken', 'scratch', 'tire flat']
                    }
                },
                'model3': {
                    'available': model3_analysis is not None,
                    'analysis': convert_defect_analysis_to_frontend_format(model3_analysis, img_width, img_height) if model3_analysis else None,
                    'model_info': {
                        'type': 'defect_detection_model3',
                        'name': 'Model3 - Specialized Defect Detection',
                        'classes': ['crack', 'dent', 'glass shatter', 'lamp broken', 'scratch', 'tire flat']
                    }
                },
                'model4': {
                    'available': model4_analysis is not None,
                    'analysis': convert_defect_analysis_to_frontend_format(model4_analysis, img_width, img_height) if model4_analysis else None,
                    'model_info': {
                        'type': 'defect_detection_model4',
                        'name': 'Model4 - Advanced Defect Detection',
                        'classes': ['crack', 'dent', 'glass shatter', 'lamp broken', 'scratch', 'tire flat']
                    }
                },
                'combined_summary': None  # Temporarily disable to test
            }
            
            # Debug: Check what we're passing to generate_all_models_summary
            logger.info(f"Debug: main_analysis type = {type(main_analysis)}")
            logger.info(f"Debug: model2_analysis type = {type(model2_analysis)}")
            logger.info(f"Debug: model3_analysis type = {type(model3_analysis)}")
            logger.info(f"Debug: model4_analysis type = {type(model4_analysis)}")
            
            combined_summary = generate_all_models_summary(main_analysis, model2_analysis, model3_analysis, model4_analysis)
            combined_result['combined_summary'] = combined_summary
            
            # Log the complete analysis results
            if gemini_damage_detected == 1:
                logger.info(f"Gemini detected damage for {file.filename} - repaired image {'generated' if repaired_image else 'failed to generate'}")
            else:
                logger.info(f"Gemini detected no damage for {file.filename} - proceeding with 4-model analysis only")
            
            logger.info(f"All models analysis (including Gemini) completed for {file.filename}")
            return JSONResponse(content=combined_result)
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
            
    except Exception as e:
        logger.error(f"Error in all models analysis for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")

@app.post("/analyze-dual")
async def analyze_image_dual(file: UploadFile = File(...)):
    """Analyze uploaded car image using both models for comprehensive detection
    
    Args:
        file: The uploaded car image
    
    Returns:
        Combined analysis from both the main defect detection model and secondary defect detection model (model2)
    """
    global inference_engine, defect_inference_engine
    
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
            
            # Initialize engines if needed
            if inference_engine is None:
                initialize_inference_engine(model_type='s')
            if defect_inference_engine is None:
                initialize_defect_inference_engine()
            
            # Run analysis with main model
            logger.info(f"Running main defect analysis on: {file.filename}")
            main_analysis = None
            if inference_engine is not None:
                main_analysis = inference_engine.predict_image(
                    temp_file_path,
                    conf_threshold=0.55,
                    iou_threshold=0.5
                )
            
            # Run analysis with defect detection model (model2)
            logger.info(f"Running defect detection analysis (model2) on: {file.filename}")
            defect_analysis = None
            if defect_inference_engine is not None:
                defect_result = defect_inference_engine.predict_single(temp_file_path)
                # Convert defect detection format to match main analysis
                defect_analysis = {
                    'detections': defect_result.get('detections', []),
                    'num_detections': defect_result.get('num_detections', 0),
                    'inference_time': defect_result.get('inference_time_ms', 0)
                }
            
            # Combine results
            combined_result = {
                'main_model': {
                    'available': main_analysis is not None,
                    'analysis': convert_analysis_to_frontend_format(main_analysis, img_width, img_height) if main_analysis else None,
                    'model_info': {
                        'type': 's',
                        'name': 'YOLOv8s - Multi-class Defect Detection',
                        'classes': inference_engine.class_names if inference_engine else []
                    }
                },
                'defect_model': {
                    'available': defect_analysis is not None,
                    'analysis': convert_defect_analysis_to_frontend_format(defect_analysis, img_width, img_height) if defect_analysis else None,
                    'model_info': {
                        'type': 'defect_detection',
                        'name': 'YOLOv8s - 6-Class Defect Detection Specialist',
                        'classes': ['crack', 'dent', 'glass shatter', 'lamp broken', 'scratch', 'tire flat']
                    }
                },
                'combined_summary': generate_combined_summary(main_analysis, defect_analysis)
            }
            
            logger.info(f"Dual analysis completed for {file.filename}")
            return JSONResponse(content=combined_result)
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
            
    except Exception as e:
        logger.error(f"Error in dual analysis for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")

def convert_defect_analysis_to_frontend_format(defect_analysis, img_width: int, img_height: int) -> Dict[str, Any]:
    """Convert defect detection results to frontend format"""
    if not defect_analysis or not defect_analysis.get('detections'):
        return {
            'cleanliness': {
                'score': 95,
                'status': 'excellent',
                'description': 'No cracks detected'
            },
            'integrity': {
                'damaged': False,
                'damageLevel': 'none',
                'description': 'No structural damage detected'
            },
            'heatmap': {'areas': []}
        }
    
    # Filter out unwanted classifications
    all_detections = defect_analysis['detections']
    detections = []
    for detection in all_detections:
        defect_class = detection.get('class', '').lower()
        # Filter out flat tire classifications for all models
        # Filter out scratch classifications for all models (as requested)
        if defect_class not in ['tire flat', 'flat tire', 'scratch']:
            detections.append(detection)
        else:
            logger.info(f"Filtered out {defect_class} classification (confidence: {detection.get('confidence', 0):.2f})")
    
    # If no detections remain after filtering, return clean result
    if not detections:
        return {
            'cleanliness': {
                'score': 95,
                'status': 'excellent',
                'description': 'No significant defects detected after filtering'
            },
            'integrity': {
                'damaged': False,
                'damageLevel': 'none',
                'description': 'No structural damage detected'
            },
            'heatmap': {'areas': []}
        }
    num_defects = len(detections)
    
    # Calculate score based on number and size of defects
    base_score = 100
    defect_penalty = min(num_defects * 15, 70)  # Max 70 points penalty
    score = max(base_score - defect_penalty, 10)
    
    # Determine status
    if score >= 80:
        status = 'good'
        damage_level = 'minor'
    elif score >= 60:
        status = 'fair'
        damage_level = 'moderate'
    else:
        status = 'poor'
        damage_level = 'severe'
    
    # Convert detections to heatmap areas
    areas = []
    for detection in detections:
        bbox = detection['bbox']  # [x1, y1, x2, y2]
        center_x = ((bbox[0] + bbox[2]) / 2) / img_width * 100
        center_y = ((bbox[1] + bbox[3]) / 2) / img_height * 100
        
        # Convert absolute coordinates to percentages
        bbox_percent = {
            'x1': (bbox[0] / img_width) * 100,
            'y1': (bbox[1] / img_height) * 100,
            'x2': (bbox[2] / img_width) * 100,
            'y2': (bbox[3] / img_height) * 100
        }
        
        # Determine severity based on defect type and size
        defect_class = detection.get('class', 'unknown')
        defect_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        img_area = img_width * img_height
        relative_size = defect_area / img_area
        
        # Base severity on defect type (removed scratch and tire flat as they're filtered out)
        severity_map = {
            'crack': 'high',
            'dent': 'medium', 
            'glass shatter': 'high',
            'lamp broken': 'high'
        }
        severity = severity_map.get(defect_class, 'medium')
        
        # Adjust severity based on size for certain defect types
        if defect_class in ['crack', 'dent'] and relative_size > 0.01:
            if severity == 'low':
                severity = 'medium'
            elif severity == 'medium':
                severity = 'high'
        
        areas.append({
            'x': center_x,
            'y': center_y,
            'severity': severity,
            'description': f'{defect_class.title()} detected (confidence: {detection["confidence"]:.2f})',
            'bbox': bbox_percent,
            'defect_type': defect_class,
            'confidence': detection['confidence']
        })
    
    return {
        'cleanliness': {
            'score': score,
            'status': status,
            'description': f'{num_defects} defect(s) detected'
        },
        'integrity': {
            'damaged': True,
            'damageLevel': damage_level,
            'description': f'Structural integrity compromised by {num_defects} defect(s)'
        },
        'heatmap': {'areas': areas}
    }

def generate_combined_summary(main_analysis, defect_analysis) -> Dict[str, Any]:
    """Generate a combined summary from both analyses"""
    main_available = main_analysis is not None
    defect_available = defect_analysis is not None and defect_analysis.get('detections')
    
    if not main_available and not defect_available:
        return {
            'overall_score': 0,
            'overall_status': 'poor',
            'description': 'Analysis failed for both models',
            'recommendations': ['Please try uploading the image again']
        }
    
    # Calculate combined score
    main_score = 95  # Default if no main analysis
    defect_score = 95  # Default if no defects
    
    if main_available and main_analysis.defect_groups:
        # Calculate main model score based on defects
        severity_map = {'slight': 1, 'moderate': 3, 'severe': 5}
        severity_values = []
        for group in main_analysis.defect_groups:
            # Debug: Check what group.severity actually is
            logger.info(f"Debug: group.severity = {group.severity}, type = {type(group.severity)}")
            
            # Handle case where severity might be a list or other type
            if isinstance(group.severity, list):
                # If it's a list, take the first element or use default
                severity_str = group.severity[0] if group.severity else 'slight'
            elif isinstance(group.severity, str):
                severity_str = group.severity
            else:
                severity_str = str(group.severity) if group.severity else 'slight'
            
            severity_val = severity_map.get(severity_str.lower(), 1)
            severity_values.append(severity_val)
        
        total_severity = sum(severity_values)
        num_defects = len(main_analysis.defect_groups)
        main_score = max(100 - (total_severity * 10) - (num_defects * 5), 10)
        logger.info(f"Debug: main_score calculated = {main_score}, type = {type(main_score)}")
    
    if defect_available:
        num_defects = len(defect_analysis['detections'])
        defect_score = max(100 - (num_defects * 15), 10)
    
    # Combined score (weighted average)
    combined_score = int((main_score * 0.7) + (defect_score * 0.3))
    
    # Determine overall status
    if combined_score >= 85:
        status = 'excellent'
    elif combined_score >= 70:
        status = 'good'
    elif combined_score >= 50:
        status = 'fair'
    else:
        status = 'poor'
    
    # Generate recommendations
    recommendations = []
    if main_available and main_analysis.defect_groups:
        recommendations.append('Address visible defects before vehicle use')
    if defect_available:
        recommendations.append('Inspect and repair structural defects immediately')
    if not recommendations:
        recommendations.append('Vehicle appears to be in good condition')
    
    return {
        'overall_score': combined_score,
        'overall_status': status,
        'description': f'Combined analysis from both detection models',
        'recommendations': recommendations,
        'models_used': {
            'main_defect_detection': main_available,
            'defect_detection': defect_available
        }
    }

def generate_recommendations_from_ensemble(ensemble_result: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on ensemble prediction results"""
    recommendations = []
    
    if ensemble_result.get('override_applied') and ensemble_result.get('override_model') == 'model3':
        recommendations.append("Vehicle appears to be in good condition based on comprehensive analysis")
        recommendations.append("Regular maintenance schedule recommended to maintain current condition")
        return recommendations
    
    if not ensemble_result.get('damage_detected'):
        recommendations.append("No significant damage detected across all models")
        recommendations.append("Vehicle appears to be in good condition")
        recommendations.append("Continue regular maintenance and inspections")
        return recommendations
    
    damage_types = ensemble_result.get('damage_types', [])
    severity_score = ensemble_result.get('severity_score', 0)
    
    if severity_score >= 15:
        recommendations.append("Severe damage detected - immediate professional inspection recommended")
        recommendations.append("Do not operate vehicle until repairs are completed")
    elif severity_score >= 8:
        recommendations.append("Moderate damage detected - schedule repair appointment soon")
        recommendations.append("Monitor damage areas for any worsening")
    else:
        recommendations.append("Minor damage detected - address when convenient")
        recommendations.append("Regular monitoring recommended")
    
    # Specific recommendations based on damage types
    for damage_type in damage_types:
        damage_class = damage_type.get('type', '')
        if damage_class == 'glass_damage':
            recommendations.append("Glass damage detected - replace immediately for safety")
        elif damage_class == 'tire_damage':
            recommendations.append("Tire damage detected - inspect and replace if necessary")
        elif damage_class == 'severe_damage':
            recommendations.append("Severe structural damage - professional assessment required")
        elif damage_class == 'crack':
            recommendations.append("Crack damage detected - monitor for expansion")
    
    return recommendations

def generate_all_models_summary(main_analysis, model2_analysis, model3_analysis, model4_analysis) -> Dict[str, Any]:
    """Generate a combined summary from all 4 models using ensemble logic"""
    global ensemble_engine
    
    logger.info("Starting ensemble-based model summary generation")
    
    # Use ensemble logic if available
    if ensemble_engine is not None:
        try:
            ensemble_result = ensemble_engine.generate_ensemble_prediction(
                main_analysis, model2_analysis, model3_analysis, model4_analysis
            )
            
            # Convert ensemble result to frontend format
            return {
                'overall_score': ensemble_result['ensemble_score'],
                'overall_status': 'excellent' if ensemble_result['ensemble_score'] >= 85 else 
                                'good' if ensemble_result['ensemble_score'] >= 70 else 
                                'fair' if ensemble_result['ensemble_score'] >= 50 else 'poor',
                'description': ensemble_result['reasoning'],
                'recommendations': generate_recommendations_from_ensemble(ensemble_result),
                'models_used': {
                    'main_model': main_analysis is not None,
                    'model2': model2_analysis is not None and bool(model2_analysis.get('detections')),
                    'model3': model3_analysis is not None and bool(model3_analysis.get('detections')),
                    'model4': model4_analysis is not None and bool(model4_analysis.get('detections'))
                },
                'ensemble_details': {
                    'prediction': ensemble_result['prediction'],
                    'confidence': ensemble_result['confidence'],
                    'reasoning': ensemble_result['reasoning'],
                    'damage_detected': ensemble_result['damage_detected'],
                    'damage_types': ensemble_result.get('damage_types', []),
                    'override_applied': ensemble_result.get('override_applied', False),
                    'override_model': ensemble_result.get('override_model'),
                    'severity_score': ensemble_result.get('severity_score', 0),
                    'models_agreement': ensemble_result.get('models_agreement', {}),
                    'unified_detections': ensemble_result.get('unified_detections', [])
                }
            }
        except Exception as e:
            logger.error(f"Error in ensemble logic: {e}")
            # Fallback to original logic
    
    # Fallback to original logic if ensemble fails
    logger.info("DEBUG: Starting generate_all_models_summary function")
    logger.info(f"DEBUG: main_analysis = {main_analysis}")
    logger.info(f"DEBUG: model2_analysis = {model2_analysis}")
    logger.info(f"DEBUG: model3_analysis = {model3_analysis}")
    logger.info(f"DEBUG: model4_analysis = {model4_analysis}")
    
    main_available = main_analysis is not None
    model2_available = model2_analysis is not None and bool(model2_analysis.get('detections'))
    model3_available = model3_analysis is not None and bool(model3_analysis.get('detections'))
    model4_available = model4_analysis is not None and bool(model4_analysis.get('detections'))
    
    logger.info(f"DEBUG: Availability - main:{main_available}, model2:{model2_available}, model3:{model3_available}, model4:{model4_available}")
    
    available_models = sum([int(main_available), int(model2_available), int(model3_available), int(model4_available)])
    
    if available_models == 0:
        return {
            'overall_score': 0,
            'overall_status': 'poor',
            'description': 'Analysis failed for all models',
            'recommendations': ['Please try uploading the image again'],
            'models_used': {
                'main_model': False,
                'model2': False,
                'model3': False,
                'model4': False
            }
        }
    
    # Calculate individual model scores
    main_score = 95  # Default if no main analysis
    model2_score = 95  # Default if no defects
    model3_score = 95  # Default if no defects
    model4_score = 95  # Default if no defects
    
    logger.info(f"Debug: Initial scores - main:{main_score}, model2:{model2_score}, model3:{model3_score}, model4:{model4_score}")
    logger.info(f"Debug: main_available = {main_available}, type = {type(main_available)}")
    logger.info(f"Debug: main_analysis = {main_analysis}, type = {type(main_analysis)}")
    
    try:
        if main_available and main_analysis.defect_groups:
            logger.info(f"Debug: main_analysis.defect_groups length = {len(main_analysis.defect_groups)}")
            # Calculate main model score based on defects
            severity_map = {'slight': 1, 'moderate': 3, 'severe': 5}
            severity_values = []
            for i, group in enumerate(main_analysis.defect_groups):
                logger.info(f"Debug: Processing group {i}: {group}")
                logger.info(f"Debug: group.severity = {group.severity}, type = {type(group.severity)}")
                
                # Handle case where severity might be a list or other type
                if isinstance(group.severity, list):
                    # If it's a list, take the first element or use default
                    severity_str = group.severity[0] if group.severity else 'slight'
                elif isinstance(group.severity, str):
                    severity_str = group.severity
                else:
                    severity_str = str(group.severity) if group.severity else 'slight'
                
                severity_val = severity_map.get(severity_str.lower(), 1)
                severity_values.append(severity_val)
            
            total_severity = sum(severity_values)
            num_defects = len(main_analysis.defect_groups)
            main_score = max(100 - (total_severity * 10) - (num_defects * 5), 10)
    except Exception as e:
        logger.error(f"Error processing main_analysis: {e}")
        main_score = 95
    
    try:
        if model2_available:
            detections = model2_analysis.get('detections', [])
            num_defects = len(detections) if isinstance(detections, list) else 0
            score_calc = 100 - (num_defects * 15)
            logger.info(f"Debug: model2 score_calc = {score_calc}, type = {type(score_calc)}")
            model2_score = max(score_calc, 10)
            logger.info(f"Debug: model2_score calculated = {model2_score}, type = {type(model2_score)}")
    except Exception as e:
        logger.error(f"Error processing model2_score: {e}")
        model2_score = 95
    
    try:
        if model3_available:
            detections = model3_analysis.get('detections', [])
            num_defects = len(detections) if isinstance(detections, list) else 0
            score_calc = 100 - (num_defects * 15)
            logger.info(f"Debug: model3 score_calc = {score_calc}, type = {type(score_calc)}")
            model3_score = max(score_calc, 10)
            logger.info(f"Debug: model3_score calculated = {model3_score}, type = {type(model3_score)}")
    except Exception as e:
        logger.error(f"Error processing model3_score: {e}")
        model3_score = 95
    
    try:
        if model4_available:
            detections = model4_analysis.get('detections', [])
            num_defects = len(detections) if isinstance(detections, list) else 0
            score_calc = 100 - (num_defects * 15)
            logger.info(f"Debug: model4 score_calc = {score_calc}, type = {type(score_calc)}")
            model4_score = max(score_calc, 10)
            logger.info(f"Debug: model4_score calculated = {model4_score}, type = {type(model4_score)}")
    except Exception as e:
        logger.error(f"Error processing model4_score: {e}")
        model4_score = 95
    
    # Combined score (weighted average based on available models)
    total_weight = 0
    weighted_sum = 0
    
    if main_available:
        weighted_sum += main_score * 0.4
        total_weight += 0.4
        logger.info(f"Debug: Added main_score {main_score} with weight 0.4")
    if model2_available:
        weighted_sum += model2_score * 0.2
        total_weight += 0.2
        logger.info(f"Debug: Added model2_score {model2_score} with weight 0.2")
    if model3_available:
        weighted_sum += model3_score * 0.2
        total_weight += 0.2
        logger.info(f"Debug: Added model3_score {model3_score} with weight 0.2")
    if model4_available:
        weighted_sum += model4_score * 0.2
        total_weight += 0.2
        logger.info(f"Debug: Added model4_score {model4_score} with weight 0.2")
    
    # Debug logging to identify the issue
    logger.info(f"Debug: weighted_sum={weighted_sum}, type={type(weighted_sum)}")
    logger.info(f"Debug: total_weight={total_weight}, type={type(total_weight)}")
    
    try:
        if total_weight > 0:
            division_result = weighted_sum / total_weight
            logger.info(f"Debug: division_result={division_result}, type={type(division_result)}")
            
            # Ensure division_result is a number before converting to int
            if isinstance(division_result, (int, float)):
                combined_score = int(division_result)
            else:
                logger.error(f"Error: division_result is not a number: {division_result}, type = {type(division_result)}")
                combined_score = 85
        else:
            combined_score = 0
    except Exception as e:
        logger.error(f"Error calculating combined score: {e}")
        combined_score = 85
    
    # Determine overall status
    if combined_score >= 85:
        status = 'excellent'
    elif combined_score >= 70:
        status = 'good'
    elif combined_score >= 50:
        status = 'fair'
    else:
        status = 'poor'
    
    # Generate comprehensive recommendations
    recommendations = []
    defect_found = False
    
    if main_available and main_analysis.defect_groups:
        recommendations.append('Address visible defects detected by main model')
        defect_found = True
    if model2_available:
        recommendations.append('Inspect areas flagged by Model2 specialist detection')
        defect_found = True
    if model3_available:
        recommendations.append('Review defects identified by Model3 analysis')
        defect_found = True
    if model4_available:
        recommendations.append('Address issues found by Model4 advanced detection')
        defect_found = True
    
    if not defect_found:
        recommendations.append('Vehicle appears to be in good condition across all models')
    else:
        recommendations.append('Consider professional inspection for comprehensive assessment')
    
    return {
        'overall_score': combined_score,
        'overall_status': status,
        'description': f'Comprehensive analysis from {available_models} AI models',
        'recommendations': recommendations,
        'models_used': {
            'main_model': main_available,
            'model2': model2_available,
            'model3': model3_available,
            'model4': model4_available
        },
        'analysis_coverage': f'{available_models}/4 models successfully analyzed the image'
    }

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    global inference_engine
    
    if inference_engine is None:
        raise HTTPException(status_code=500, detail="Inference engine not initialized")
    
    try:
        model_type = inference_engine.model_type
        return {
            "model_type": model_type,
            "model_name": f"YOLOv8{model_type}",
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

async def analyze_frame_for_damage_single_model(image_data: bytes, real_time_detection: bool = True) -> Dict[str, Any]:
    """
    Analyze a single video frame for car damage detection using single model (legacy function).
    
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

async def analyze_frame_with_all_models(image_data: bytes) -> Dict[str, Any]:
    """
    Analyze a single video frame for car damage detection using all 4 models simultaneously.
    
    Args:
        image_data: Raw image bytes
    
    Returns:
        Aggregated results from all 4 models with ensemble decision
    """
    import asyncio
    
    try:
        # Save frame to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_file.write(image_data)
            temp_path = temp_file.name
        
        try:
            # Check if all models are initialized
            if not all([inference_engine, defect_inference_engine, model3_inference_engine, model4_inference_engine]):
                return {"error": "Not all inference engines are initialized"}
            
            # Get image dimensions
            img = Image.open(temp_path)
            img_width, img_height = img.size
            
            # Define async functions for each model
            async def run_model1():
                try:
                    analysis = inference_engine.predict_image(temp_path)
                    return convert_analysis_to_frontend_format(analysis, img_width, img_height)
                except Exception as e:
                    logger.error(f"Model 1 error: {e}")
                    return {"error": str(e)}
            
            async def run_model2():
                try:
                    analysis = defect_inference_engine.predict_image(temp_path)
                    return convert_analysis_to_frontend_format(analysis, img_width, img_height)
                except Exception as e:
                    logger.error(f"Model 2 error: {e}")
                    return {"error": str(e)}
            
            async def run_model3():
                try:
                    analysis = model3_inference_engine.predict_image(temp_path)
                    return convert_analysis_to_frontend_format(analysis, img_width, img_height)
                except Exception as e:
                    logger.error(f"Model 3 error: {e}")
                    return {"error": str(e)}
            
            async def run_model4():
                try:
                    analysis = model4_inference_engine.predict_image(temp_path)
                    return convert_analysis_to_frontend_format(analysis, img_width, img_height)
                except Exception as e:
                    logger.error(f"Model 4 error: {e}")
                    return {"error": str(e)}
            
            # Run all models in parallel
            model1_result, model2_result, model3_result, model4_result = await asyncio.gather(
                run_model1(), run_model2(), run_model3(), run_model4(),
                return_exceptions=True
            )
            
            # Aggregate results
            models_results = {
                "model1": model1_result,
                "model2": model2_result,
                "model3": model3_result,
                "model4": model4_result
            }
            
            # Count damage detections from each model
            damage_votes = 0
            total_detections = 0
            confidence_scores = []
            all_areas = []
            
            for model_name, result in models_results.items():
                if isinstance(result, dict) and "error" not in result:
                    areas = result.get('heatmap', {}).get('areas', [])
                    if len(areas) > 0:
                        damage_votes += 1
                        total_detections += len(areas)
                        # Add areas with model source
                        for area in areas:
                            area['source_model'] = model_name
                            all_areas.append(area)
                        
                        # Calculate average confidence for this model
                        model_confidences = [area.get('confidence', 0.5) for area in areas]
                        if model_confidences:
                            confidence_scores.append(sum(model_confidences) / len(model_confidences))
            
            # Ensemble decision logic
            # Damage is detected if at least 2 out of 4 models detect it
            damage_detected = damage_votes >= 2
            
            # Calculate overall confidence
            if confidence_scores:
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                if damage_votes >= 3:
                    confidence_level = "high"
                elif damage_votes >= 2:
                    confidence_level = "medium"
                else:
                    confidence_level = "low"
            else:
                avg_confidence = 0.0
                confidence_level = "low"
            
            # Determine damage level based on total detections and votes
            if damage_detected:
                if damage_votes >= 3 and total_detections >= 5:
                    damage_level = "severe"
                elif damage_votes >= 3 or total_detections >= 3:
                    damage_level = "moderate"
                else:
                    damage_level = "minor"
            else:
                damage_level = "none"
            
            # Create aggregated result
            result = {
                "damage_detected": damage_detected,
                "models_agreement": damage_votes,
                "total_models": 4,
                "total_detections": total_detections,
                "confidence": confidence_level,
                "confidence_score": avg_confidence,
                "timestamp": int(time.time() * 1000),
                "mode": "multi_model_realtime",
                "heatmap": {"areas": all_areas},
                "integrity": {
                    "damaged": damage_detected,
                    "damageLevel": damage_level
                },
                "models_results": {
                    "model1_detections": len(model1_result.get('heatmap', {}).get('areas', [])) if isinstance(model1_result, dict) and "error" not in model1_result else 0,
                    "model2_detections": len(model2_result.get('heatmap', {}).get('areas', [])) if isinstance(model2_result, dict) and "error" not in model2_result else 0,
                    "model3_detections": len(model3_result.get('heatmap', {}).get('areas', [])) if isinstance(model3_result, dict) and "error" not in model3_result else 0,
                    "model4_detections": len(model4_result.get('heatmap', {}).get('areas', [])) if isinstance(model4_result, dict) and "error" not in model4_result else 0
                }
            }
            
            return result
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        logger.error(f"Error analyzing frame with all models: {e}")
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
            
            # Analyze the frame using all 4 models simultaneously
            result = await analyze_frame_with_all_models(data)
            
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