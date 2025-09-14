# Richards Drive - Car Damage Detection System

A full-stack application for detecting and analyzing car damage using YOLOv8 computer vision model with a React frontend and FastAPI backend. The system uses a fine-tuned YOLOv8s model to detect various car defects including dents, rust, dirt, broken lights, and paint fading.

## Features

- **Image Upload**: Drag and drop or click to upload car images
- **AI Analysis**: YOLOv8s-based damage detection with severity assessment
- **Real-time Results**: Instant analysis with visual damage heatmaps
- **Modern UI**: Clean, responsive interface built with React and Tailwind CSS
- **Multi-class Detection**: Identifies Dent, Dislocation, Scratch, Shatter, damaged, and severe damage defects
- **Severity Estimation**: Analyzes defect size and distribution to estimate severity
- **Checkpointing**: Supports training resumption from checkpoints

## Project Structure

```
├── backend/                 # FastAPI backend
│   ├── app.py              # Main FastAPI application
│   ├── train_small.py      # YOLOv8s model training script
│   ├── inference.py        # Inference engine
│   ├── postprocess.py      # Post-processing utilities
│   ├── prepare_data.py     # Data preparation script
│   └── requirements_api.txt # Backend dependencies
├── frontend/RichardsDrive/  # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   └── lib/           # Utility functions
│   └── package.json       # Frontend dependencies
├── zips/                   # Compressed datasets (data1.zip, data2.zip, etc.)
└── runs/                   # Training outputs and model weights
    └── train/
        └── yolov8s/       # YOLOv8s model outputs
            └── weights/   # Trained model weights
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- Node.js 16+
- CUDA-compatible GPU (recommended for training)

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements_api.txt
   ```

3. Prepare the unified dataset from multiple sources:
   ```bash
   python prepare_data.py
   ```
   This script will:
   - Extract datasets from the zips/ directory
   - Unify class labels using the configurable CLASS_MAPPING
   - Merge all datasets into a single unified dataset
   - Split data into train/val sets
   - Generate data.yaml with the unified class list

4. Train the YOLOv8s model:
   ```bash
   python train_small.py --batch 16 --epochs 50 --workers 0
   ```
   
   To resume training from a checkpoint:
   ```bash
   python train_small.py --resume
   ```

5. Start the FastAPI server:
   ```bash
   python app.py
   ```

   The API will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend/RichardsDrive
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

   The application will be available at `http://localhost:5173`

## Usage

### Web Interface

1. Open your browser and go to `http://localhost:5173`
2. Upload a car image using the drag-and-drop interface
3. Wait for the AI analysis to complete
4. View the results including:
   - Overall condition assessment
   - Defect detection with severity levels
   - Visual damage heatmap overlay

### Command Line Inference

You can also run inference directly from the command line:

```bash
# Basic inference on a single image
python inference.py --image path/to/image.jpg

# Specify model type (default is now 's' for YOLOv8s)
python inference.py --image path/to/image.jpg --model-type s

# Skip severity estimation for faster results
python inference.py --image path/to/image.jpg --no-severity

# Process a directory of images
python inference.py --directory path/to/images/

# Save results to JSON
python inference.py --image path/to/image.jpg --output results.json
```

## API Endpoints

- `GET /` - Health check
- `POST /analyze` - Upload and analyze car image
- `GET /model-info` - Get model information

## Model Training Pipeline

The car defect detection system uses a multi-step pipeline:

1. **Data Preparation** (`prepare_data.py`):
   - Extracts and unifies multiple datasets with different label sets
   - Uses a configurable class mapping to standardize labels
   - Creates a merged dataset with consistent annotations
   - Splits data into train/val sets

2. **Model Training** (`train_small.py`):
   - Fine-tunes YOLOv8s on the unified dataset
   - Uses transfer learning from COCO-pretrained weights
   - Applies data augmentation during training
   - Enables mixed precision (AMP) for faster training
   - Saves checkpoints for resume capability

3. **Inference** (`inference.py`):
   - Loads the trained YOLOv8s model
   - Processes images to detect car defects
   - Runs post-processing to estimate defect severity
   - Generates structured analysis results

## Defect Classes

The model is trained to detect the following defect types:

- **Dent**: Body damage including dents, scratches, and structural damage
- **Rust**: Corrosion and rust spots
- **Dirt**: Dirt, mud, and dust accumulation
- **Broken Light**: Damaged headlights, taillights, and other lighting
- **Paint Fade**: Paint damage, fading, and discoloration

## Model Details

- **Architecture**: YOLOv8s (You Only Look Once v8 Small)
- **Task**: Object detection for car damage
- **Classes**: 5 defect types (dent, rust, dirt, broken_light, paint_fade)
- **Input**: RGB images (any size, automatically resized)
- **Output**: Bounding boxes with confidence scores, defect classifications, and severity estimation
- **Performance**: Optimized for real-time inference with mixed precision support

## Development

### Training Custom Models

To train on your own dataset:

1. Place your dataset zip files in the `zips/` directory
2. Update the `CLASS_MAPPING` in `prepare_data.py` to match your defect classes
3. Run `prepare_data.py` to unify and prepare your dataset
4. Run `train_small.py` with desired parameters
5. Use `--resume` flag to continue training from the latest checkpoint

### API Integration

The frontend communicates with the backend via REST API:
- Images are sent as multipart form data
- Results include damage coordinates, confidence scores, and severity assessments
- Error handling for network issues and processing failures

## Technologies Used

### Backend
- FastAPI - Modern Python web framework
- Ultralytics YOLOv8 - Computer vision model
- OpenCV - Image processing
- PyTorch - Deep learning framework

### Frontend
- React 18 - UI framework
- TypeScript - Type safety
- Tailwind CSS - Styling
- Vite - Build tool
- Radix UI - Component primitives

## License

This project is developed for the Decentrathon hackathon.

A YOLOv8-based computer vision system for detecting and analyzing car defects including dents, scratches, and dirt. The system provides automated severity assessment and detailed analysis reports.

## Features

- **Multi-class Detection**: Detects dents, scratches, and dirt on car surfaces
- **Severity Assessment**: Automatically estimates defect severity (Low, Medium, High, Critical)
- **Intelligent Grouping**: Groups nearby defects using DBSCAN clustering and IoU analysis
- **Comprehensive Analysis**: Provides detailed reports with bounding boxes, confidence scores, and summaries
- **Flexible Inference**: Supports single images, batch processing, and directory scanning
- **GPU Acceleration**: Optimized for NVIDIA GPUs with CUDA support
- **Export Options**: JSON output for integration with other systems

## Project Structure

```
backend/
├── data1/                    # First dataset (extracted from data1.zip)
├── data2/                    # Second dataset (extracted from data2.zip)
├── merged_dataset/           # Combined dataset for training
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
├── runs/                     # Training outputs
│   └── detect/
│       └── train/
│           └── weights/
│               ├── best.pt   # Best model weights
│               └── last.pt   # Latest checkpoint
├── prepare_data.py           # Dataset preparation script
├── train.py                  # Training script
├── postprocess.py            # Post-processing and analysis
├── inference.py              # Inference engine
└── requirements.txt          # Python dependencies
```

## Installation

### Prerequisites

- Python 3.8 or higher
- NVIDIA GPU with CUDA support (recommended)
- At least 8GB RAM
- 10GB free disk space

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd decentrathon-indrive-richards-case1
   ```

2. **Install dependencies**:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Prepare the dataset**:
   ```bash
   # Extract data1.zip and data2.zip to data1/ and data2/ directories
   # Then merge the datasets
   python prepare_data.py
   ```

4. **Train the model** (optional - if you want to retrain):
   ```bash
   python train.py --batch 16 --epochs 50
   ```

## Usage

### Quick Start

```bash
# Run inference on a single image
python inference.py --model runs/detect/train/weights/best.pt --image path/to/car_image.jpg

# Process all images in a directory
python inference.py --model runs/detect/train/weights/best.pt --directory path/to/images/

# Save results to JSON file
python inference.py --model runs/detect/train/weights/best.pt --image car.jpg --output results.json
```

### Training

```bash
# Basic training
python train.py

# Custom parameters
python train.py --batch 32 --epochs 100 --lr 0.001

# Resume from checkpoint
python train.py --resume runs/detect/train/weights/last.pt

# CPU-only training
python train.py --device cpu
```

### Inference Options

```bash
# Single image with custom confidence threshold
python inference.py -m best.pt -i image.jpg --conf 0.3

# Batch processing with IoU threshold
python inference.py -m best.pt -d images/ --iou 0.5

# Multiple specific images
python inference.py -m best.pt --images img1.jpg img2.jpg img3.jpg

# Quiet mode (JSON output only)
python inference.py -m best.pt -i image.jpg --quiet
```

## API Reference

### CarDefectInference Class

```python
from inference import CarDefectInference

# Initialize
inference = CarDefectInference(
    model_path="runs/detect/train/weights/best.pt",
    class_names=["dent", "dirt", "scratch"]
)

# Single image prediction
analysis = inference.predict_image("car.jpg")

# Batch prediction
analyses = inference.predict_batch(["car1.jpg", "car2.jpg"])

# Directory prediction
analyses = inference.predict_directory("images/")
```

### CarAnalysis Object

```python
class CarAnalysis:
    image_path: str              # Path to analyzed image
    car_detected: bool           # Whether a car was detected
    car_bbox: Tuple[float, ...]  # Car bounding box (x1, y1, x2, y2)
    defect_groups: List[DefectGroup]  # Grouped defects
    overall_condition: str       # "Clean", "Minor Issues", "Moderate Damage", "Severe Damage"
    summary: str                 # Human-readable summary
```

### DefectGroup Object

```python
class DefectGroup:
    defect_type: str            # "dent", "dirt", "scratch"
    severity: str               # "Low", "Medium", "High", "Critical"
    confidence: float           # Average confidence score
    total_area: float           # Total defect area in pixels
    bbox: Tuple[float, ...]     # Group bounding box
    detections: List[Detection] # Individual detections
```

## Configuration

### Training Parameters

- `--batch`: Batch size (default: 16)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 0.01)
- `--device`: Device to use ('auto', 'cpu', 'cuda', or specific GPU ID)
- `--resume`: Path to checkpoint for resuming training

### Inference Parameters

- `--conf`: Confidence threshold (default: 0.25)
- `--iou`: IoU threshold for NMS (default: 0.45)
- `--classes`: Class names (default: ['dent', 'dirt', 'scratch'])

### Post-processing Parameters

Configurable in `postprocess.py`:

```python
class DefectPostProcessor:
    def __init__(self):
        self.severity_thresholds = {
            'area_ratio': {'low': 0.01, 'medium': 0.05, 'high': 0.15},
            'detection_count': {'low': 2, 'medium': 5, 'high': 10}
        }
        self.clustering_eps = 50  # DBSCAN epsilon
        self.clustering_min_samples = 2
        self.iou_threshold = 0.3
```

## Dataset Format

The system expects YOLO format annotations:

```
class_id center_x center_y width height
```

Where coordinates are normalized (0-1) relative to image dimensions.

### Class Mapping

- 0: dent
- 1: dirt  
- 2: scratch

## Performance

### Hardware Requirements

- **Minimum**: Intel i5 / AMD Ryzen 5, 8GB RAM, GTX 1060 / RTX 2060
- **Recommended**: Intel i7 / AMD Ryzen 7, 16GB RAM, RTX 3070 / RTX 4060
- **Optimal**: Intel i9 / AMD Ryzen 9, 32GB RAM, RTX 4080 / RTX 4090

### Inference Speed

- **RTX 4060**: ~15-20 FPS (1080p images)
- **RTX 3070**: ~12-18 FPS (1080p images)
- **CPU Only**: ~1-3 FPS (1080p images)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```bash
   # Reduce batch size
   python train.py --batch 8
   ```

2. **Model Not Found**:
   ```bash
   # Check if training completed successfully
   ls runs/detect/train/weights/
   ```

3. **Poor Detection Results**:
   - Lower confidence threshold: `--conf 0.1`
   - Check image quality and lighting
   - Ensure car is clearly visible

4. **Slow Inference**:
   - Verify GPU is being used: Check "Device: cuda" in output
   - Update NVIDIA drivers
   - Install CUDA-compatible PyTorch

### Debug Mode

```bash
# Enable verbose output
python inference.py --model best.pt --image car.jpg --conf 0.1

# Check model info
python -c "from ultralytics import YOLO; model = YOLO('best.pt'); print(model.info())"
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the base detection model
- [OpenCV](https://opencv.org/) for image processing
- [scikit-learn](https://scikit-learn.org/) for clustering algorithms

## Support

For issues and questions:

1. Check the troubleshooting section
2. Search existing issues
3. Create a new issue with:
   - System specifications
   - Error messages
   - Steps to reproduce
   - Sample images (if applicable)

---

**Note**: This system is designed for automotive damage assessment and should be used as a supplementary tool alongside human inspection for critical evaluations.