# Complete GPU Training Pipeline - Car Defect Detection

This document provides step-by-step commands for the complete preprocessing and training pipeline optimized for GPU usage.

## Prerequisites

### 1. System Requirements
- NVIDIA GPU with CUDA support
- Python 3.8+
- CUDA 11.8+ installed
- At least 8GB GPU memory (recommended)

### 2. Verify GPU Setup

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version

# Verify PyTorch CUDA support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else "None"}')"
```

### 3. Install Dependencies

```bash
# Install PyTorch with CUDA support (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install ultralytics opencv-python scikit-learn

# Verify installation
python -c "from ultralytics import YOLO; print('YOLOv8 installed successfully')"
```

## Complete Training Pipeline

### Step 1: Navigate to Backend Directory

```bash
cd c:\Users\aidyn\Desktop\PR\Decentraton\decentrathon-indrive-richards-case1\backend
```

### Step 2: Data Preparation

```bash
# Merge datasets and create configuration
python prepare_data.py

# Verify dataset structure
dir merged_dataset
type data.yaml
```

**Expected Output:**
- `merged_dataset/` directory with train/val/test splits
- `data.yaml` configuration file
- Console output showing dataset statistics

### Step 3: GPU Training Commands

#### Basic GPU Training (Recommended Start)

```bash
# Default GPU training - auto-detects best GPU
python train.py --device 0
```

#### Production GPU Training (High Performance)

```bash
# High-performance training with optimal GPU settings
python train.py \
  --device 0 \
  --epochs 100 \
  --batch-size 32 \
  --img-size 640 \
  --patience 20 \
  --save-period 5 \
  --workers 8
```

#### Multi-GPU Training (If Available)

```bash
# Use multiple GPUs
python train.py --device 0,1 --batch-size 64 --epochs 100

# Check GPU utilization during training
watch -n 1 nvidia-smi
```

#### Memory-Optimized Training (For Limited GPU Memory)

```bash
# Reduced batch size for 6-8GB GPU memory
python train.py \
  --device 0 \
  --batch-size 16 \
  --img-size 512 \
  --epochs 100

# For 4GB GPU memory
python train.py \
  --device 0 \
  --batch-size 8 \
  --img-size 416 \
  --epochs 100
```

#### Fast Training (Quick Testing)

```bash
# Quick training for testing pipeline
python train.py \
  --device 0 \
  --epochs 10 \
  --batch-size 16 \
  --img-size 416
```

### Step 4: Monitor Training Progress

#### Real-time GPU Monitoring

```bash
# Monitor GPU usage in separate terminal
watch -n 1 nvidia-smi

# Or use continuous monitoring
nvidia-smi -l 1
```

#### Training Logs

```bash
# View training progress (run in separate terminal)
tail -f runs/detect/train/train.log

# Check latest training directory
dir runs\detect
```

### Step 5: Resume Training (If Interrupted)

```bash
# Resume from last checkpoint
python train.py --resume runs/detect/train/weights/last.pt

# Resume with different settings
python train.py \
  --resume runs/detect/train/weights/last.pt \
  --epochs 150 \
  --batch-size 32
```

### Step 6: Validate Trained Model

```bash
# Test inference with trained model
python inference.py \
  --model runs/detect/train/weights/best.pt \
  --image ../data1/test/images/sample_image.jpg \
  --conf 0.25

# Batch testing on validation set
python inference.py \
  --model runs/detect/train/weights/best.pt \
  --directory ../data1/valid/images \
  --output validation_results.json
```

## Advanced GPU Training Options

### Hyperparameter Optimization

```bash
# Learning rate optimization
python train.py \
  --device 0 \
  --lr0 0.01 \
  --lrf 0.001 \
  --momentum 0.937 \
  --weight_decay 0.0005

# Data augmentation tuning
python train.py \
  --device 0 \
  --hsv_h 0.015 \
  --hsv_s 0.7 \
  --hsv_v 0.4 \
  --degrees 10 \
  --translate 0.1
```

### Model Architecture Selection

```bash
# YOLOv8 Nano (fastest, least memory)
python train.py --model yolov8n.pt --device 0 --batch-size 64

# YOLOv8 Small (balanced)
python train.py --model yolov8s.pt --device 0 --batch-size 32

# YOLOv8 Medium (better accuracy)
python train.py --model yolov8m.pt --device 0 --batch-size 16

# YOLOv8 Large (best accuracy, requires more memory)
python train.py --model yolov8l.pt --device 0 --batch-size 8
```

### Mixed Precision Training (Faster Training)

```bash
# Enable automatic mixed precision for faster training
python train.py \
  --device 0 \
  --amp \
  --batch-size 32 \
  --epochs 100
```

## Performance Optimization Tips

### 1. Optimal Batch Size Selection

```bash
# Test different batch sizes to find optimal for your GPU
for batch in 8 16 32 64; do
  echo "Testing batch size: $batch"
  python train.py --device 0 --batch-size $batch --epochs 1 --img-size 416
done
```

### 2. GPU Memory Management

```bash
# Clear GPU cache before training
python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"

# Monitor memory usage during training
python -c "import torch; print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
```

### 3. Benchmark Training Speed

```bash
# Benchmark different configurations
echo "Benchmarking training configurations..."

# Configuration 1: High batch size
time python train.py --device 0 --batch-size 32 --epochs 5 --img-size 640

# Configuration 2: Medium batch size
time python train.py --device 0 --batch-size 16 --epochs 5 --img-size 640

# Configuration 3: Small image size
time python train.py --device 0 --batch-size 32 --epochs 5 --img-size 416
```

## Troubleshooting GPU Issues

### CUDA Out of Memory

```bash
# Reduce batch size
python train.py --device 0 --batch-size 8 --img-size 416

# Use gradient accumulation (simulates larger batch size)
python train.py --device 0 --batch-size 8 --accumulate 4  # Effective batch size: 32

# Clear cache and retry
python -c "import torch; torch.cuda.empty_cache()" && python train.py --device 0
```

### GPU Not Detected

```bash
# Force CPU training as fallback
python train.py --device cpu

# Check CUDA installation
python -c "import torch; print(torch.version.cuda)"

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Slow Training

```bash
# Enable optimizations
python train.py \
  --device 0 \
  --amp \
  --workers 8 \
  --batch-size 32 \
  --cache ram  # Cache dataset in RAM

# Use smaller image size for faster training
python train.py --device 0 --img-size 416 --batch-size 32
```

## Complete Pipeline Example

```bash
#!/bin/bash
# Complete training pipeline script

echo "Starting Car Defect Detection Training Pipeline..."

# Step 1: Prepare data
echo "Step 1: Preparing dataset..."
python prepare_data.py

# Step 2: Verify GPU
echo "Step 2: Verifying GPU setup..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')"

# Step 3: Start training
echo "Step 3: Starting GPU training..."
python train.py \
  --device 0 \
  --epochs 100 \
  --batch-size 32 \
  --img-size 640 \
  --patience 20 \
  --save-period 5

# Step 4: Test inference
echo "Step 4: Testing trained model..."
python inference.py \
  --model runs/detect/train/weights/best.pt \
  --directory ../data1/test/images \
  --output final_results.json

echo "Training pipeline completed!"
echo "Best model saved at: runs/detect/train/weights/best.pt"
echo "Results saved at: final_results.json"
```

## Expected Training Times (GPU)

| GPU Model | Batch Size | Image Size | Time per Epoch | 100 Epochs |
|-----------|------------|------------|----------------|-------------|
| RTX 4090  | 32         | 640        | ~2 minutes     | ~3.5 hours  |
| RTX 3080  | 32         | 640        | ~3 minutes     | ~5 hours    |
| RTX 3070  | 16         | 640        | ~4 minutes     | ~7 hours    |
| RTX 3060  | 16         | 512        | ~5 minutes     | ~8.5 hours  |
| GTX 1080  | 8          | 416        | ~8 minutes     | ~13 hours   |

## Final Validation Commands

```bash
# Comprehensive model evaluation
echo "Final model validation..."

# Test on individual images
python inference.py --model runs/detect/train/weights/best.pt --image test_car1.jpg
python inference.py --model runs/detect/train/weights/best.pt --image test_car2.jpg

# Batch evaluation
python inference.py \
  --model runs/detect/train/weights/best.pt \
  --directory ../data1/test/images \
  --output comprehensive_results.json \
  --conf 0.25

# Performance summary
echo "Training completed successfully!"
echo "Model location: runs/detect/train/weights/best.pt"
echo "Validation results: comprehensive_results.json"
echo "Training logs: runs/detect/train/"
```

This pipeline ensures optimal GPU utilization and provides comprehensive monitoring and troubleshooting options for the complete car defect detection training process.