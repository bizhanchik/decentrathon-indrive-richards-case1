#!/usr/bin/env python3
"""
YOLOv8 training script for car defect detection.
Supports GPU/CPU training, checkpointing, and resume capability.
"""

import os
import argparse
import torch
from pathlib import Path
from ultralytics import YOLO
import yaml
from datetime import datetime
import json


class YOLOTrainer:
    def __init__(self, data_yaml_path: str, model_size: str = 'n', 
                 epochs: int = 100, batch_size: int = 16, 
                 checkpoint_interval: int = 5, project_name: str = 'car_defect_detection'):
        """
        Initialize YOLO trainer.
        
        Args:
            data_yaml_path: Path to data.yaml configuration file
            model_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x')
            epochs: Number of training epochs
            batch_size: Training batch size
            checkpoint_interval: Save checkpoint every N epochs
            project_name: Name for the training project
        """
        self.data_yaml_path = Path(data_yaml_path)
        self.model_size = model_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.checkpoint_interval = checkpoint_interval
        self.project_name = project_name
        
        # Setup paths
        self.project_root = Path(__file__).parent.parent
        self.runs_dir = self.project_root / 'runs'
        self.checkpoints_dir = self.runs_dir / 'checkpoints'
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Device setup
        self.device = self._setup_device()
        
        # Verify data configuration
        self._verify_data_config()
        
        print(f"=== YOLOv8 Car Defect Detection Training ===")
        print(f"Model: YOLOv8{model_size}")
        print(f"Device: {self.device}")
        print(f"Data config: {self.data_yaml_path}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Checkpoint interval: {checkpoint_interval} epochs")
    
    def _setup_device(self) -> str:
        """Setup and return the best available device."""
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = 'cpu'
            print("No GPU detected, using CPU")
        return device
    
    def _verify_data_config(self) -> None:
        """Verify the data configuration file exists and is valid."""
        if not self.data_yaml_path.exists():
            raise FileNotFoundError(f"Data config file not found: {self.data_yaml_path}")
        
        with open(self.data_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        required_keys = ['train', 'val', 'nc', 'names']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key '{key}' in data config")
        
        print(f"Data config verified: {config['nc']} classes - {config['names']}")
    
    def _get_latest_checkpoint(self) -> str:
        """Find the latest checkpoint file."""
        checkpoint_files = list(self.checkpoints_dir.glob('*.pt'))
        if not checkpoint_files:
            return None
        
        # Sort by modification time and return the latest
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        return str(latest_checkpoint)
    
    def _save_training_config(self, config: dict) -> None:
        """Save training configuration for reproducibility."""
        config_path = self.checkpoints_dir / 'training_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Training config saved to {config_path}")
    
    def train(self, resume: bool = False, pretrained_weights: str = None, workers: int = 8) -> str:
        """
        Train the YOLOv8 model.
        
        Args:
            resume: Whether to resume from the latest checkpoint
            pretrained_weights: Path to custom pretrained weights
            workers: Number of dataloader workers
        
        Returns:
            Path to the best trained model
        """
        # Determine model initialization
        if resume:
            latest_checkpoint = self._get_latest_checkpoint()
            if latest_checkpoint:
                print(f"Resuming training from: {latest_checkpoint}")
                model = YOLO(latest_checkpoint)
            else:
                print("No checkpoint found, starting fresh training")
                model = YOLO(f'yolov8{self.model_size}.pt')
        elif pretrained_weights:
            print(f"Loading custom pretrained weights: {pretrained_weights}")
            model = YOLO(pretrained_weights)
        else:
            print(f"Starting training with COCO pretrained YOLOv8{self.model_size}")
            model = YOLO(f'yolov8{self.model_size}.pt')
        
        # Training configuration
        training_config = {
            'data': str(self.data_yaml_path),
            'epochs': self.epochs,
            'batch': self.batch_size,
            'device': self.device,
            'project': str(self.runs_dir),
            'name': self.project_name,
            'save_period': self.checkpoint_interval,
            'patience': 50,  # Early stopping patience
            'save': True,
            'plots': True,
            'val': True,
            # Data augmentation settings
            'hsv_h': 0.015,  # HSV-Hue augmentation
            'hsv_s': 0.7,    # HSV-Saturation augmentation
            'hsv_v': 0.4,    # HSV-Value augmentation
            'degrees': 0.0,  # Rotation degrees
            'translate': 0.1, # Translation
            'scale': 0.5,    # Scale
            'shear': 0.0,    # Shear
            'perspective': 0.0, # Perspective
            'flipud': 0.0,   # Vertical flip probability
            'fliplr': 0.5,   # Horizontal flip probability
            'mosaic': 1.0,   # Mosaic probability
            'mixup': 0.0,    # Mixup probability
            'copy_paste': 0.0, # Copy-paste probability
            'workers': workers,  # Number of dataloader workers
        }
        
        # Save training configuration
        self._save_training_config({
            'model_size': self.model_size,
            'training_config': training_config,
            'timestamp': datetime.now().isoformat(),
            'device': self.device,
            'resume': resume,
            'pretrained_weights': pretrained_weights
        })
        
        print(f"\n=== Starting Training ===")
        print(f"Training will save checkpoints every {self.checkpoint_interval} epochs")
        print(f"Results will be saved to: {self.runs_dir / self.project_name}")
        
        try:
            # Start training
            results = model.train(**training_config)
            
            # Get paths to best and last models
            run_dir = self.runs_dir / self.project_name
            best_model_path = run_dir / 'weights' / 'best.pt'
            last_model_path = run_dir / 'weights' / 'last.pt'
            
            # Copy best model to checkpoints directory
            if best_model_path.exists():
                best_checkpoint = self.checkpoints_dir / f'best_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
                import shutil
                shutil.copy2(best_model_path, best_checkpoint)
                print(f"\nBest model saved to: {best_checkpoint}")
            
            print(f"\n=== Training Complete ===")
            print(f"Best model: {best_model_path}")
            print(f"Last model: {last_model_path}")
            print(f"Training results: {run_dir}")
            
            return str(best_model_path)
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            latest_checkpoint = self._get_latest_checkpoint()
            if latest_checkpoint:
                print(f"You can resume training with: python train.py --resume")
            raise
        except Exception as e:
            print(f"\nTraining failed with error: {e}")
            raise
    
    def validate(self, model_path: str) -> dict:
        """Validate the trained model."""
        print(f"\n=== Model Validation ===")
        model = YOLO(model_path)
        
        # Run validation
        results = model.val(data=str(self.data_yaml_path))
        
        # Print validation metrics
        print(f"Validation Results:")
        print(f"mAP50: {results.box.map50:.4f}")
        print(f"mAP50-95: {results.box.map:.4f}")
        
        return {
            'map50': float(results.box.map50),
            'map50_95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr)
        }


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 for car defect detection')
    parser.add_argument('--data', type=str, default='../merged_dataset/data.yaml',
                       help='Path to data.yaml file')
    parser.add_argument('--model', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLOv8 model size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--checkpoint-interval', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from latest checkpoint')
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Path to custom pretrained weights')
    parser.add_argument('--project', type=str, default='car_defect_detection',
                       help='Project name for organizing runs')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only run validation on existing model')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to model for validation-only mode')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of dataloader workers (set to 0 for Windows compatibility)')
    
    args = parser.parse_args()
    
    # Resolve data path relative to script location
    script_dir = Path(__file__).parent
    data_path = script_dir / args.data
    
    if not data_path.exists():
        print(f"Error: Data config file not found: {data_path}")
        print("Please run prepare_data.py first to create the merged dataset")
        return
    
    # Initialize trainer
    trainer = YOLOTrainer(
        data_yaml_path=str(data_path),
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        checkpoint_interval=args.checkpoint_interval,
        project_name=args.project
    )
    
    if args.validate_only:
        if not args.model_path:
            print("Error: --model-path required for validation-only mode")
            return
        trainer.validate(args.model_path)
    else:
        # Train the model
        best_model_path = trainer.train(
            resume=args.resume,
            pretrained_weights=args.pretrained,
            workers=args.workers
        )
        
        # Run validation on the best model
        if Path(best_model_path).exists():
            trainer.validate(best_model_path)


if __name__ == '__main__':
    main()