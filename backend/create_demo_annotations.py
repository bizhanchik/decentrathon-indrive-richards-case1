#!/usr/bin/env python3
"""
Demo Annotation Generator
Creates annotated car damage detection results for presentation
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import CarDefectInference
from postprocess import CarAnalysis, DefectGroup, Detection

class DemoAnnotationGenerator:
    def __init__(self):
        self.detector = CarDefectInference(model_type='s')  # Use YOLOv8s model
        self.output_dir = Path("demo_annotations")
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "annotated_images").mkdir(exist_ok=True)
        (self.output_dir / "results_json").mkdir(exist_ok=True)
        
    def find_demo_images(self):
        """Find available training batch images for demo"""
        demo_images = []
        runs_dir = Path("../runs/train")
        
        if runs_dir.exists():
            for model_dir in runs_dir.iterdir():
                if model_dir.is_dir():
                    for batch_file in model_dir.glob("train_batch*.jpg"):
                        demo_images.append(batch_file)
        
        return demo_images[:6]  # Limit to 6 images for demo
    
    def draw_annotations(self, image, analysis: CarAnalysis):
        """Draw bounding boxes and labels on image"""
        annotated = image.copy()
        height, width = image.shape[:2]
        
        # Color mapping for damage types
        colors = {
            'dent': (0, 255, 255),       # Yellow
            'dislocation': (255, 0, 0),  # Blue  
            'scratch': (0, 0, 255),      # Red
            'shatter': (0, 165, 255),    # Orange
            'damaged': (128, 0, 128),    # Purple
            'severe damage': (0, 0, 128) # Dark Red
        }
        
        # Draw defect groups
        for defect_group in analysis.defect_groups:
            defect_type = defect_group.defect_type.lower()
            color = colors.get(defect_type, (255, 255, 255))
            
            # Draw each detection in the group
            for detection in defect_group.detections:
                x1, y1, x2, y2 = detection.bbox
                
                # Convert to pixel coordinates (assuming bbox is already in pixel coordinates)
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with confidence
                label = f"{detection.class_name}: {detection.confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Background for text
                cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), color, -1)
                
                # Text
                cv2.putText(annotated, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Add overall assessment
        assessment_text = f"Condition: {analysis.overall_condition}"
        cv2.putText(annotated, assessment_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Add defect count
        defect_count = len(analysis.defect_groups)
        count_text = f"Defects: {defect_count}"
        cv2.putText(annotated, count_text, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return annotated
    
    def process_image(self, image_path):
        """Process a single image and create annotations"""
        print(f"Processing: {image_path.name}")
        
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Could not load image: {image_path}")
                return None
            
            # Run detection using CarDefectInference
            analysis = self.detector.predict_image(str(image_path), conf_threshold=0.25)
            
            # Create annotated image
            annotated_image = self.draw_annotations(image, analysis)
            
            # Save annotated image
            output_name = f"annotated_{image_path.stem}.jpg"
            output_path = self.output_dir / "annotated_images" / output_name
            cv2.imwrite(str(output_path), annotated_image)
            
            # Convert analysis to serializable format
            def convert_to_serializable(obj):
                """Convert numpy types to Python native types for JSON serialization"""
                if hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_serializable(item) for item in obj]
                elif isinstance(obj, dict):
                    return {key: convert_to_serializable(value) for key, value in obj.items()}
                else:
                    return obj
            
            result_data = {
                "image_name": image_path.name,
                "timestamp": datetime.now().isoformat(),
                "car_detected": bool(analysis.car_detected),
                "overall_condition": str(analysis.overall_condition),
                "summary": str(analysis.summary),
                "defect_groups": [
                    {
                        "defect_type": str(group.defect_type),
                        "severity": str(group.severity),
                        "confidence": float(group.confidence),
                        "total_area": float(group.total_area),
                        "detection_count": len(group.detections),
                        "detections": [
                            {
                                "class_name": str(det.class_name),
                                "confidence": float(det.confidence),
                                "bbox": [float(x) for x in det.bbox],
                                "area": float(det.area)
                            } for det in group.detections
                        ]
                    } for group in analysis.defect_groups
                ],
                "annotated_image_path": str(output_path)
            }
            
            # Ensure all values are JSON serializable
            result_data = convert_to_serializable(result_data)
            
            json_path = self.output_dir / "results_json" / f"result_{image_path.stem}.json"
            with open(json_path, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            print(f"✓ Saved: {output_path}")
            print(f"✓ Results: {json_path}")
            print(f"  Condition: {analysis.overall_condition}, Defect Groups: {len(analysis.defect_groups)}")
            
            return result_data
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def generate_demo_annotations(self):
        """Generate all demo annotations"""
        print("=== Car Damage Detection Demo Annotation Generator ===")
        print(f"Output directory: {self.output_dir.absolute()}")
        
        # Find demo images
        demo_images = self.find_demo_images()
        
        if not demo_images:
            print("No demo images found in runs/train directories")
            return
        
        print(f"Found {len(demo_images)} demo images")
        
        results = []
        for image_path in demo_images:
            result = self.process_image(image_path)
            if result:
                results.append(result)
        
        # Create summary report
        summary = {
            "generated_at": datetime.now().isoformat(),
            "total_images": len(demo_images),
            "successful_annotations": len(results),
            "results": results
        }
        
        summary_path = self.output_dir / "demo_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n=== Demo Generation Complete ===")
        print(f"Annotated images: {self.output_dir / 'annotated_images'}")
        print(f"Results JSON: {self.output_dir / 'results_json'}")
        print(f"Summary: {summary_path}")
        print(f"Successfully processed: {len(results)}/{len(demo_images)} images")

if __name__ == "__main__":
    generator = DemoAnnotationGenerator()
    generator.generate_demo_annotations()