# Car Defect Detection Backend

A modular ML backend system for detecting and classifying car defects using YOLOv8. This system can identify dents, dirt, and scratches on car images and provide severity assessments.

## Features

- **YOLOv8 Fine-tuning**: Transfer learning from pretrained COCO weights
- **GPU/CPU Support**: Automatic device detection with fallback to CPU
- **Checkpointing**: Resume training from saved checkpoints
- **Data Merging**: Automatic dataset merging and configuration
- **Severity Assessment**: Post-processing with defect grouping and severity estimation
- **Batch Processing**: Support for single images, batches, and directories
- **Structured Output**: JSON and console output formats

## Project Structure

```
backend/
├── prepare_data.py     # Dataset merging and preparation
├── train.py           # YOLOv8 model training with checkpointing
├── inference.py       # Model inference with severity analysis
├── postprocess.py     # Detection grouping and severity estimation
├── README.md          # This file
├── data.yaml          # Generated dataset configuration
├── merged_dataset/    # Generated merged dataset (after running prepare_data.py)
└── runs/              # Training outputs and checkpoints
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install ultralytics opencv-python scikit-learn torch torchvision torchaudio
```

**Note**: For GPU support, ensure you have CUDA installed and compatible PyTorch version. Visit [PyTorch installation guide](https://pytorch.org/get-started/locally/) for specific instructions.

### Alternative Installation

You can also install dependencies from a requirements file:

```bash
# Create requirements.txt with the following content:
echo "ultralytics>=8.0.0" > requirements.txt
echo "opencv-python>=4.5.0" >> requirements.txt
echo "scikit-learn>=1.0.0" >> requirements.txt
echo "torch>=1.12.0" >> requirements.txt
echo "torchvision>=0.13.0" >> requirements.txt
echo "torchaudio>=0.12.0" >> requirements.txt

# Install
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Dataset

Merge the provided datasets and generate configuration:

```bash
cd backend
python prepare_data.py
```

This will:
- Merge `data1/` and `data2/` datasets
- Resolve class mapping conflicts
- Generate `data.yaml` configuration
- Create `merged_dataset/` directory structure

### 2. Train Model

Fine-tune YOLOv8 on the merged dataset:

```bash
# Basic training (auto-detects GPU/CPU)
python train.py

# Custom training parameters
python train.py --epochs 100 --batch-size 16 --img-size 640 --patience 20

# Resume from checkpoint
python train.py --resume runs/detect/train/weights/last.pt
```

**Training Features:**
- Automatic GPU/CPU detection
- Checkpointing every 5 epochs
- Best model tracking (lowest validation loss)
- Data augmentation (resize, brightness, contrast, flips, noise)
- Early stopping with patience

### 3. Run Inference

Use the trained model for defect detection:

```bash
# Single image
python inference.py --model runs/detect/train/weights/best.pt --image test_image.jpg

# Directory of images
python inference.py --model runs/detect/train/weights/best.pt --directory test_images/

# Save results to JSON
python inference.py --model runs/detect/train/weights/best.pt --image test.jpg --output results.json

# Adjust confidence threshold
python inference.py --model runs/detect/train/weights/best.pt --image test.jpg --conf 0.3
```

## Detailed Usage

### Dataset Preparation

```bash
python prepare_data.py [options]
```

**Options:**
- `--data1-path`: Path to first dataset (default: ../data1)
- `--data2-path`: Path to second dataset (default: ../data2)
- `--output-path`: Output directory (default: merged_dataset)
- `--train-split`: Training split ratio (default: 0.8)
- `--val-split`: Validation split ratio (default: 0.15)

### Model Training

```bash
python train.py [options]
```

**Key Options:**
- `--data`: Path to data.yaml (default: data.yaml)
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 16)
- `--img-size`: Image size (default: 640)
- `--device`: Device to use (auto, cpu, 0, 1, etc.)
- `--resume`: Resume from checkpoint path
- `--patience`: Early stopping patience (default: 10)
- `--save-period`: Checkpoint save period (default: 5)

**Example Training Commands:**

```bash
# Quick training for testing
python train.py --epochs 10 --batch-size 8

# Production training
python train.py --epochs 200 --batch-size 32 --patience 30

# Resume interrupted training
python train.py --resume runs/detect/train2/weights/last.pt

# Force CPU training
python train.py --device cpu
```

### Inference

```bash
python inference.py --model MODEL_PATH [input_options] [options]
```

**Input Options (choose one):**
- `--image PATH`: Single image file
- `--directory PATH`: Directory containing images
- `--images PATH1 PATH2 ...`: List of image files

**Other Options:**
- `--output PATH`: Save results to JSON file
- `--conf FLOAT`: Confidence threshold (default: 0.25)
- `--iou FLOAT`: IoU threshold for NMS (default: 0.45)
- `--classes NAME1 NAME2 ...`: Class names (default: dent dirt scratch)
- `--quiet`: Suppress detailed output, show only JSON

**Example Inference Commands:**

```bash
# Basic inference
python inference.py -m runs/detect/train/weights/best.pt -i car_image.jpg

# Batch processing
python inference.py -m best.pt -d test_images/ -o batch_results.json

# High confidence detections only
python inference.py -m best.pt -i car.jpg --conf 0.5

# Programmatic output (JSON only)
python inference.py -m best.pt -i car.jpg --quiet
```

## Output Format

### Console Output

```
============================================================
Image: car_image.jpg
============================================================
Overall Condition: Fair
Summary: Car detected → Moderate dents, Slight dirt

Detailed Analysis:
  1. DENT:
     - Severity: moderate
     - Confidence: 0.847
     - Detection count: 3
     - Total area: 1250.5 pixels
  2. DIRT:
     - Severity: slight
     - Confidence: 0.623
     - Detection count: 1
     - Total area: 450.2 pixels
```

### JSON Output

```json
{
  "model_path": "runs/detect/train/weights/best.pt",
  "confidence_threshold": 0.25,
  "iou_threshold": 0.45,
  "class_names": ["dent", "dirt", "scratch"],
  "total_images": 1,
  "results": [
    {
      "image_path": "car_image.jpg",
      "car_detected": true,
      "car_bbox": [45.2, 67.8, 892.1, 534.6],
      "overall_condition": "Fair",
      "summary": "Car detected → Moderate dents, Slight dirt",
      "total_defects": 2,
      "defect_groups": [
        {
          "defect_type": "dent",
          "severity": "moderate",
          "confidence": 0.847,
          "total_area": 1250.5,
          "detection_count": 3,
          "bbox": [123.4, 234.5, 345.6, 456.7],
          "detections": [
            {
              "class_id": 0,
              "class_name": "dent",
              "confidence": 0.892,
              "bbox": [123.4, 234.5, 200.1, 300.2],
              "area": 567.8,
              "center": [161.8, 267.4]
            }
          ]
        }
      ]
    }
  ]
}
```

## Severity Assessment

The system estimates defect severity based on:

1. **Detection Count**: Number of defect instances
2. **Area Ratio**: Total defect area relative to car size

**Severity Levels:**
- **Slight**: < 2% car area OR < 2 detections
- **Moderate**: 2-5% car area OR 2-5 detections
- **Severe**: > 5% car area OR > 5 detections

**Overall Condition:**
- **Clean**: No defects detected
- **Good**: Only slight defects
- **Fair**: Moderate defects present
- **Poor**: Severe defects present

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
python train.py --batch-size 8

# Use smaller image size
python train.py --img-size 416

# Force CPU training
python train.py --device cpu
```

**2. No GPU Detected**
- Verify CUDA installation: `nvidia-smi`
- Check PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`
- Install CUDA-compatible PyTorch version

**3. Dataset Issues**
```bash
# Check dataset structure
python prepare_data.py --validate-only

# Regenerate dataset
rm -rf merged_dataset data.yaml
python prepare_data.py
```

**4. Training Convergence Issues**
- Increase learning rate: `--lr0 0.01`
- Adjust patience: `--patience 20`
- Use different optimizer: `--optimizer AdamW`
- Check data quality and annotations

### Performance Tips

1. **Training Speed:**
   - Use GPU if available
   - Increase batch size (within memory limits)
   - Use mixed precision: `--amp`
   - Enable multi-GPU: `--device 0,1`

2. **Inference Speed:**
   - Use TensorRT optimization
   - Reduce image size for faster processing
   - Batch multiple images together

3. **Memory Usage:**
   - Reduce batch size
   - Use smaller image sizes
   - Enable gradient checkpointing

## Model Performance

Expected performance metrics after training:

- **mAP@0.5**: 0.7-0.9 (depending on dataset quality)
- **Precision**: 0.8-0.95
- **Recall**: 0.7-0.9
- **Inference Speed**: 10-50 FPS (depending on hardware)

## Advanced Usage

### Custom Class Names

```bash
# Train with custom classes
python train.py --names "rust,dent,scratch,crack"

# Inference with custom classes
python inference.py -m model.pt -i image.jpg --classes rust dent scratch crack
```

### Hyperparameter Tuning

```bash
# Learning rate scheduling
python train.py --lr0 0.01 --lrf 0.001

# Data augmentation
python train.py --hsv_h 0.015 --hsv_s 0.7 --hsv_v 0.4

# Model architecture
python train.py --model yolov8n.pt  # nano (fastest)
python train.py --model yolov8s.pt  # small
python train.py --model yolov8m.pt  # medium
python train.py --model yolov8l.pt  # large (best accuracy)
```

### Integration Examples

```python
# Python API usage
from inference import CarDefectInference

# Initialize
engine = CarDefectInference('best.pt')

# Single prediction
result = engine.predict_image('car.jpg')
print(result.summary)

# Batch prediction
results = engine.predict_directory('images/')
for result in results:
    print(f"{result.image_path}: {result.overall_condition}")
```

## License

This project is for educational and research purposes. Please ensure compliance with YOLOv8 and dataset licenses.

## Support

For issues and questions:
1. Check this README for common solutions
2. Verify dataset format and model paths
3. Check system requirements and dependencies
4. Review error messages for specific guidance