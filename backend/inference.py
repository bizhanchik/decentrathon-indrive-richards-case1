#!/usr/bin/env python3
"""
Inference script for car defect detection.
Loads trained YOLOv8 model and runs predictions with severity analysis.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import cv2
import torch
from ultralytics import YOLO

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from postprocess import DefectPostProcessor, CarAnalysis


class CarDefectInference:
    """Car defect detection inference engine."""
    
    def __init__(self, model_path: str, class_names: List[str] = None):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the trained YOLO model
            class_names: List of class names
        """
        self.model_path = model_path
        self.class_names = class_names or ['dent', 'dirt', 'scratch']
        self.model = None
        self.postprocessor = DefectPostProcessor(class_names=self.class_names)
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """
        Load the trained YOLO model.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        print(f"Loading model from: {self.model_path}")
        
        try:
            self.model = YOLO(self.model_path)
            print(f"Model loaded successfully. Device: {self.model.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def predict_image(self, image_path: str, conf_threshold: float = 0.25, 
                     iou_threshold: float = 0.45) -> CarAnalysis:
        """
        Run inference on a single image.
        
        Args:
            image_path: Path to the input image
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        
        Returns:
            CarAnalysis object with complete analysis
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load image to get dimensions
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        img_height, img_width = image.shape[:2]
        img_shape = (img_height, img_width)
        
        print(f"Processing image: {image_path} ({img_width}x{img_height})")
        
        # Run YOLO inference
        try:
            results = self.model(
                image_path,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )
        except Exception as e:
            raise RuntimeError(f"Inference failed: {e}")
        
        # Post-process results
        analysis = self.postprocessor.process_detections(
            results, img_shape, image_path
        )
        
        return analysis
    
    def predict_batch(self, image_paths: List[str], conf_threshold: float = 0.25,
                     iou_threshold: float = 0.45) -> List[CarAnalysis]:
        """
        Run inference on multiple images.
        
        Args:
            image_paths: List of paths to input images
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        
        Returns:
            List of CarAnalysis objects
        """
        results = []
        
        for image_path in image_paths:
            try:
                analysis = self.predict_image(
                    image_path, conf_threshold, iou_threshold
                )
                results.append(analysis)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                # Create empty analysis for failed images
                analysis = CarAnalysis(
                    image_path=image_path,
                    car_detected=False,
                    car_bbox=(0, 0, 0, 0),
                    defect_groups=[],
                    overall_condition="Unknown",
                    summary=f"Error processing image: {e}"
                )
                results.append(analysis)
        
        return results
    
    def predict_directory(self, directory_path: str, conf_threshold: float = 0.25,
                         iou_threshold: float = 0.45, 
                         extensions: List[str] = None) -> List[CarAnalysis]:
        """
        Run inference on all images in a directory.
        
        Args:
            directory_path: Path to directory containing images
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            extensions: List of image file extensions to process
        
        Returns:
            List of CarAnalysis objects
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Find all image files
        image_paths = []
        for ext in extensions:
            pattern = f"*{ext}"
            image_paths.extend(Path(directory_path).glob(pattern))
            image_paths.extend(Path(directory_path).glob(pattern.upper()))
        
        image_paths = [str(p) for p in image_paths]
        
        if not image_paths:
            print(f"No image files found in {directory_path}")
            return []
        
        print(f"Found {len(image_paths)} images in {directory_path}")
        
        return self.predict_batch(image_paths, conf_threshold, iou_threshold)


def analysis_to_dict(analysis: CarAnalysis) -> Dict[str, Any]:
    """
    Convert CarAnalysis object to dictionary for JSON serialization.
    
    Args:
        analysis: CarAnalysis object
    
    Returns:
        Dictionary representation
    """
    defect_groups_dict = []
    
    for group in analysis.defect_groups:
        detections_dict = []
        for det in group.detections:
            detections_dict.append({
                'class_id': int(det.class_id),
                'class_name': det.class_name,
                'confidence': round(float(det.confidence), 3),
                'bbox': [round(float(x), 1) for x in det.bbox],
                'area': round(float(det.area), 1),
                'center': [round(float(x), 1) for x in det.center]
            })
        
        defect_groups_dict.append({
            'defect_type': group.defect_type,
            'severity': group.severity,
            'confidence': round(float(group.confidence), 3),
            'total_area': round(float(group.total_area), 1),
            'bbox': [round(float(x), 1) for x in group.bbox],
            'detection_count': len(group.detections),
            'detections': detections_dict
        })
    
    return {
        'image_path': analysis.image_path,
        'car_detected': bool(analysis.car_detected),
        'car_bbox': [round(float(x), 1) for x in analysis.car_bbox],
        'overall_condition': analysis.overall_condition,
        'summary': analysis.summary,
        'defect_groups': defect_groups_dict,
        'total_defects': len(analysis.defect_groups)
    }


def print_analysis_summary(analysis: CarAnalysis):
    """
    Print a human-readable summary of the analysis.
    
    Args:
        analysis: CarAnalysis object
    """
    print(f"\n{'='*60}")
    print(f"Image: {os.path.basename(analysis.image_path)}")
    print(f"{'='*60}")
    print(f"Overall Condition: {analysis.overall_condition}")
    print(f"Summary: {analysis.summary}")
    
    if analysis.defect_groups:
        print(f"\nDetailed Analysis:")
        for i, group in enumerate(analysis.defect_groups, 1):
            print(f"  {i}. {group.defect_type.upper()}:")
            print(f"     - Severity: {group.severity}")
            print(f"     - Confidence: {group.confidence:.3f}")
            print(f"     - Detection count: {len(group.detections)}")
            print(f"     - Total area: {group.total_area:.1f} pixels")
    else:
        print("\nNo defects detected.")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Car Defect Detection Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image inference
  python inference.py --model runs/detect/train/weights/best.pt --image test_image.jpg
  
  # Batch inference on directory
  python inference.py --model runs/detect/train/weights/best.pt --directory test_images/
  
  # Save results to JSON file
  python inference.py --model runs/detect/train/weights/best.pt --image test.jpg --output results.json
  
  # Adjust confidence threshold
  python inference.py --model runs/detect/train/weights/best.pt --image test.jpg --conf 0.3
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to trained YOLO model (.pt file)'
    )
    
    # Input arguments (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--image', '-i',
        type=str,
        help='Path to single image file'
    )
    input_group.add_argument(
        '--directory', '-d',
        type=str,
        help='Path to directory containing images'
    )
    input_group.add_argument(
        '--images',
        type=str,
        nargs='+',
        help='List of image file paths'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output JSON file path (optional)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='IoU threshold for NMS (default: 0.45)'
    )
    parser.add_argument(
        '--classes',
        type=str,
        nargs='+',
        default=['dent', 'dirt', 'scratch'],
        help='Class names (default: dent dirt scratch)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress detailed output'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize inference engine
        if not args.quiet:
            print("Initializing Car Defect Detection Inference Engine...")
        
        inference_engine = CarDefectInference(
            model_path=args.model,
            class_names=args.classes
        )
        
        # Run inference based on input type
        if args.image:
            # Single image
            analysis = inference_engine.predict_image(
                args.image, args.conf, args.iou
            )
            analyses = [analysis]
            
        elif args.directory:
            # Directory of images
            analyses = inference_engine.predict_directory(
                args.directory, args.conf, args.iou
            )
            
        elif args.images:
            # List of images
            analyses = inference_engine.predict_batch(
                args.images, args.conf, args.iou
            )
        
        # Process results
        if not args.quiet:
            for analysis in analyses:
                print_analysis_summary(analysis)
        
        # Save to JSON if requested
        if args.output:
            results_dict = {
                'model_path': args.model,
                'confidence_threshold': args.conf,
                'iou_threshold': args.iou,
                'class_names': args.classes,
                'total_images': len(analyses),
                'results': [analysis_to_dict(analysis) for analysis in analyses]
            }
            
            with open(args.output, 'w') as f:
                json.dump(results_dict, f, indent=2)
            
            print(f"\nResults saved to: {args.output}")
        
        # Print summary statistics
        if not args.quiet and len(analyses) > 1:
            total_defects = sum(len(a.defect_groups) for a in analyses)
            clean_cars = sum(1 for a in analyses if a.overall_condition == "Clean")
            
            print(f"\n{'='*60}")
            print(f"BATCH SUMMARY")
            print(f"{'='*60}")
            print(f"Total images processed: {len(analyses)}")
            print(f"Clean cars: {clean_cars}")
            print(f"Cars with defects: {len(analyses) - clean_cars}")
            print(f"Total defect groups detected: {total_defects}")
        
        # Print JSON output for programmatic use
        if args.quiet:
            results_dict = {
                'results': [analysis_to_dict(analysis) for analysis in analyses]
            }
            print(json.dumps(results_dict, indent=2))
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()