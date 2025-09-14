#!/usr/bin/env python3
"""
Inference pipeline for car damage detection model (model3)
Supports single image, batch processing, and real-time inference
Detects: damaged-dent, damaged-scratch, good_condition, severe damage
"""

import os
import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import time


class CarDamageDetectionInference:
    def __init__(self, model_path: str, confidence_threshold: float = 0.25, 
                 device: str = 'auto'):
        """
        Initialize car damage detection inference pipeline.
        
        Args:
            model_path: Path to trained YOLO model
            confidence_threshold: Minimum confidence for detections
            device: Device to run inference on ('auto', 'cpu', 'cuda')
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # Data3 classes
        self.class_names = ['damaged-dent', 'damaged-scratch', 'good_condition', 'severe damage']
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load model
        print(f"Loading car damage detection model from: {model_path}")
        self.model = YOLO(str(self.model_path))
        
        # Set device
        if device != 'auto':
            self.model.to(device)
        
        print(f"Model loaded successfully")
        print(f"Classes: {self.class_names}")
        print(f"Confidence threshold: {confidence_threshold}")
        print(f"Device: {self.model.device}")
    
    def predict_single(self, image_path: str, save_result: bool = True, 
                      output_dir: str = 'inference_results') -> Dict:
        """
        Run inference on a single image.
        
        Args:
            image_path: Path to input image
            save_result: Whether to save annotated image
            output_dir: Directory to save results
        
        Returns:
            Dictionary containing detection results
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Run inference
        start_time = time.time()
        results = self.model(str(image_path), conf=self.confidence_threshold)
        inference_time = time.time() - start_time
        
        # Process results
        result = results[0]
        detections = []
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                class_name = self.class_names[int(cls)] if int(cls) < len(self.class_names) else f'class_{int(cls)}'
                
                detection = {
                    'bbox': [float(x) for x in box],  # [x1, y1, x2, y2]
                    'confidence': float(conf),
                    'class': class_name,
                    'class_id': int(cls)
                }
                detections.append(detection)
        
        # Prepare result dictionary
        result_dict = {
            'image_path': str(image_path),
            'image_name': image_path.name,
            'detections': detections,
            'num_detections': len(detections),
            'inference_time_ms': round(inference_time * 1000, 2),
            'timestamp': datetime.now().isoformat(),
            'model_path': str(self.model_path),
            'confidence_threshold': self.confidence_threshold,
            'classes': self.class_names
        }
        
        # Save annotated image if requested
        if save_result and detections:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Create annotated image
            annotated_img = result.plot()
            output_file = output_path / f"annotated_{image_path.name}"
            cv2.imwrite(str(output_file), annotated_img)
            result_dict['annotated_image_path'] = str(output_file)
        
        return result_dict
    
    def predict_batch(self, image_dir: str, output_dir: str = 'batch_results',
                     save_images: bool = True) -> List[Dict]:
        """
        Run inference on a batch of images.
        
        Args:
            image_dir: Directory containing input images
            output_dir: Directory to save results
            save_images: Whether to save annotated images
        
        Returns:
            List of detection results for each image
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_dir.glob(f'*{ext}'))
            image_files.extend(image_dir.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"No images found in {image_dir}")
            return []
        
        print(f"Processing {len(image_files)} images...")
        
        all_results = []
        total_detections = 0
        total_time = 0
        class_counts = {cls: 0 for cls in self.class_names}
        
        for i, image_file in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {image_file.name}")
            
            try:
                result = self.predict_single(
                    str(image_file), 
                    save_result=save_images,
                    output_dir=str(output_dir)
                )
                all_results.append(result)
                total_detections += result['num_detections']
                total_time += result['inference_time_ms']
                
                # Count detections by class
                for detection in result['detections']:
                    class_counts[detection['class']] += 1
                
            except Exception as e:
                print(f"Error processing {image_file.name}: {e}")
                continue
        
        # Save batch results summary
        summary = {
            'total_images': len(image_files),
            'processed_images': len(all_results),
            'total_detections': total_detections,
            'class_distribution': class_counts,
            'average_inference_time_ms': round(total_time / len(all_results), 2) if all_results else 0,
            'timestamp': datetime.now().isoformat(),
            'model_path': str(self.model_path),
            'confidence_threshold': self.confidence_threshold,
            'classes': self.class_names
        }
        
        # Save detailed results
        results_file = output_dir / 'batch_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'summary': summary,
                'results': all_results
            }, f, indent=2)
        
        print(f"\nBatch processing complete!")
        print(f"Total detections: {total_detections}")
        print(f"Class distribution: {class_counts}")
        print(f"Average inference time: {summary['average_inference_time_ms']} ms")
        print(f"Results saved to: {results_file}")
        
        return all_results
    
    def predict_video(self, video_path: str, output_path: str = None,
                     display: bool = False) -> Dict:
        """
        Run inference on a video file.
        
        Args:
            video_path: Path to input video
            output_path: Path to save annotated video
            display: Whether to display video during processing
        
        Returns:
            Dictionary containing video processing results
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        if output_path is None:
            output_path = video_path.parent / f"annotated_{video_path.name}"
        
        cap = cv2.VideoCapture(str(video_path))
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_results = []
        frame_count = 0
        total_detections = 0
        class_counts = {cls: 0 for cls in self.class_names}
        
        print(f"Processing video: {video_path.name}")
        print(f"Total frames: {total_frames}, FPS: {fps}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run inference on frame
            results = self.model(frame, conf=self.confidence_threshold)
            result = results[0]
            
            # Count detections
            num_detections = len(result.boxes) if result.boxes is not None else 0
            total_detections += num_detections
            
            # Count by class
            if result.boxes is not None:
                classes = result.boxes.cls.cpu().numpy()
                for cls in classes:
                    class_name = self.class_names[int(cls)] if int(cls) < len(self.class_names) else f'class_{int(cls)}'
                    class_counts[class_name] += 1
            
            # Annotate frame
            annotated_frame = result.plot()
            
            # Write frame
            out.write(annotated_frame)
            
            # Display if requested
            if display:
                cv2.imshow('Car Damage Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            if frame_count % 30 == 0:  # Progress update every 30 frames
                print(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        out.release()
        if display:
            cv2.destroyAllWindows()
        
        result_dict = {
            'video_path': str(video_path),
            'output_path': str(output_path),
            'total_frames': frame_count,
            'total_detections': total_detections,
            'class_distribution': class_counts,
            'average_detections_per_frame': round(total_detections / frame_count, 2),
            'fps': fps,
            'timestamp': datetime.now().isoformat(),
            'classes': self.class_names
        }
        
        print(f"Video processing complete!")
        print(f"Output saved to: {output_path}")
        print(f"Total detections: {total_detections}")
        print(f"Class distribution: {class_counts}")
        
        return result_dict


def main():
    parser = argparse.ArgumentParser(description='Car Damage Detection Inference (Model3)')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained YOLO model')
    parser.add_argument('--source', type=str, required=True,
                       help='Path to image, directory, or video')
    parser.add_argument('--output', type=str, default='inference_results',
                       help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to run inference on')
    parser.add_argument('--save-images', action='store_true',
                       help='Save annotated images')
    parser.add_argument('--display', action='store_true',
                       help='Display results (for video)')
    
    args = parser.parse_args()
    
    # Initialize inference pipeline
    inference = CarDamageDetectionInference(
        model_path=args.model,
        confidence_threshold=args.conf,
        device=args.device
    )
    
    source_path = Path(args.source)
    
    if source_path.is_file():
        # Check if it's a video file
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        if source_path.suffix.lower() in video_extensions:
            # Process video
            output_path = Path(args.output) / f"annotated_{source_path.name}"
            result = inference.predict_video(
                str(source_path), 
                str(output_path),
                display=args.display
            )
            print(json.dumps(result, indent=2))
        else:
            # Process single image
            result = inference.predict_single(
                str(source_path),
                save_result=args.save_images,
                output_dir=args.output
            )
            print(json.dumps(result, indent=2))
    
    elif source_path.is_dir():
        # Process batch of images
        results = inference.predict_batch(
            str(source_path),
            output_dir=args.output,
            save_images=args.save_images
        )
        print(f"Processed {len(results)} images")
    
    else:
        print(f"Error: Source path does not exist: {source_path}")


if __name__ == '__main__':
    main()