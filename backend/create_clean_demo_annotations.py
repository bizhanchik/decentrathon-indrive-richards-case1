#!/usr/bin/env python3
"""
Clean Demo Annotation Generator
Creates clean annotated car damage detection results for presentation
Filters out tire detections and focuses on dents, scratches, and damages only
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

class CleanDemoAnnotationGenerator:
    def __init__(self):
        self.detector = CarDefectInference(model_type='s')  # Use YOLOv8s model
        self.output_dir = Path("clean_demo_annotations")
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "annotated_images").mkdir(exist_ok=True)
        (self.output_dir / "results_json").mkdir(exist_ok=True)
        
        # Define clean damage types to keep (filter out tire and other unwanted classes)
        self.allowed_damage_types = {
            'dent': 'Dent',
            'scratch': 'Scratch', 
            'damaged': 'Damage',
            'severe damage': 'Severe Damage'
        }
        
    def filter_analysis(self, analysis: CarAnalysis) -> CarAnalysis:
        """Filter analysis to keep only relevant damage types"""
        filtered_groups = []
        
        for group in analysis.defect_groups:
            # Check if this defect type should be kept
            defect_type_lower = group.defect_type.lower()
            if defect_type_lower in self.allowed_damage_types:
                # Filter detections within the group
                filtered_detections = []
                for detection in group.detections:
                    class_name_lower = detection.class_name.lower()
                    if class_name_lower in self.allowed_damage_types:
                        filtered_detections.append(detection)
                
                # Only keep group if it has valid detections
                if filtered_detections:
                    # Create new group with filtered detections
                    filtered_group = DefectGroup(
                        defect_type=self.allowed_damage_types[defect_type_lower],
                        detections=filtered_detections,
                        total_area=sum(det.area for det in filtered_detections),
                        bbox=group.bbox,
                        severity=group.severity,
                        confidence=sum(det.confidence for det in filtered_detections) / len(filtered_detections)
                    )
                    filtered_groups.append(filtered_group)
        
        # Create filtered analysis
        filtered_analysis = CarAnalysis(
            image_path=analysis.image_path,
            car_detected=analysis.car_detected,
            car_bbox=analysis.car_bbox,
            defect_groups=filtered_groups,
            overall_condition=analysis.overall_condition,
            summary=self._create_clean_summary(filtered_groups)
        )
        
        return filtered_analysis
    
    def _create_clean_summary(self, defect_groups):
        """Create a clean summary for filtered defects"""
        if not defect_groups:
            return "Clean car - no significant damage detected"
        
        damage_counts = {}
        for group in defect_groups:
            damage_type = group.defect_type
            if damage_type not in damage_counts:
                damage_counts[damage_type] = 0
            damage_counts[damage_type] += len(group.detections)
        
        summary_parts = []
        for damage_type, count in damage_counts.items():
            if count == 1:
                summary_parts.append(f"{damage_type}")
            else:
                summary_parts.append(f"{damage_type} ({count} areas)")
        
        return f"Car detected → {', '.join(summary_parts)}"
    
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
    
    def draw_clean_annotations(self, image, analysis: CarAnalysis):
        """Draw clean bounding boxes and labels on image"""
        annotated = image.copy()
        height, width = image.shape[:2]
        
        # Clean color mapping for damage types
        colors = {
            'dent': (0, 255, 255),       # Bright Yellow
            'scratch': (0, 0, 255),      # Red
            'damage': (128, 0, 128),     # Purple
            'severe damage': (0, 0, 139) # Dark Red
        }
        
        # Draw defect groups
        for defect_group in analysis.defect_groups:
            defect_type = defect_group.defect_type.lower()
            color = colors.get(defect_type, (255, 255, 255))
            
            # Draw each detection in the group
            for detection in defect_group.detections:
                x1, y1, x2, y2 = detection.bbox
                
                # Convert to pixel coordinates
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw bounding box with thicker lines for better visibility
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                
                # Draw label with confidence
                label = f"{detection.class_name}: {detection.confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                
                # Background for text with padding
                padding = 5
                cv2.rectangle(annotated, 
                             (x1, y1 - label_size[1] - padding * 2), 
                             (x1 + label_size[0] + padding * 2, y1), 
                             color, -1)
                
                # Text with better contrast
                cv2.putText(annotated, label, (x1 + padding, y1 - padding), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Add clean overall assessment
        assessment_text = f"Condition: {analysis.overall_condition}"
        cv2.putText(annotated, assessment_text, (15, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(annotated, assessment_text, (15, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        
        # Add damage count
        damage_count = len(analysis.defect_groups)
        count_text = f"Damage Areas: {damage_count}"
        cv2.putText(annotated, count_text, (15, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        cv2.putText(annotated, count_text, (15, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        
        return annotated
    
    def process_image(self, image_path):
        """Process a single image and create clean annotations"""
        print(f"Processing: {image_path.name}")
        
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Could not load image: {image_path}")
                return None
            
            # Run detection using CarDefectInference
            analysis = self.detector.predict_image(str(image_path), conf_threshold=0.3)
            
            # Filter analysis to keep only clean damage types
            clean_analysis = self.filter_analysis(analysis)
            
            # Skip if no relevant damage detected
            if not clean_analysis.defect_groups:
                print(f"  No relevant damage detected, skipping...")
                return None
            
            # Create annotated image
            annotated_image = self.draw_clean_annotations(image, clean_analysis)
            
            # Save annotated image
            output_name = f"clean_annotated_{image_path.stem}.jpg"
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
                "car_detected": bool(clean_analysis.car_detected),
                "overall_condition": str(clean_analysis.overall_condition),
                "summary": str(clean_analysis.summary),
                "damage_types_included": list(self.allowed_damage_types.values()),
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
                    } for group in clean_analysis.defect_groups
                ],
                "annotated_image_path": str(output_path)
            }
            
            # Ensure all values are JSON serializable
            result_data = convert_to_serializable(result_data)
            
            json_path = self.output_dir / "results_json" / f"clean_result_{image_path.stem}.json"
            with open(json_path, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            print(f"✓ Saved: {output_path}")
            print(f"✓ Results: {json_path}")
            print(f"  Condition: {clean_analysis.overall_condition}, Clean Damage Groups: {len(clean_analysis.defect_groups)}")
            
            return result_data
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def generate_clean_demo_annotations(self):
        """Generate all clean demo annotations"""
        print("=== Clean Car Damage Detection Demo Annotation Generator ===")
        print(f"Output directory: {self.output_dir.absolute()}")
        print(f"Filtering for damage types: {', '.join(self.allowed_damage_types.values())}")
        
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
            "filter_criteria": {
                "included_damage_types": list(self.allowed_damage_types.values()),
                "excluded_types": ["Tire", "Dislocation", "Shatter"]
            },
            "total_images_processed": len(demo_images),
            "images_with_relevant_damage": len(results),
            "results": results
        }
        
        summary_path = self.output_dir / "clean_demo_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n=== Clean Demo Generation Complete ===")
        print(f"Clean annotated images: {self.output_dir / 'annotated_images'}")
        print(f"Clean results JSON: {self.output_dir / 'results_json'}")
        print(f"Summary: {summary_path}")
        print(f"Successfully processed: {len(results)}/{len(demo_images)} images with relevant damage")
        print(f"Damage types included: {', '.join(self.allowed_damage_types.values())}")

if __name__ == "__main__":
    generator = CleanDemoAnnotationGenerator()
    generator.generate_clean_demo_annotations()