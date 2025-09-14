#!/usr/bin/env python3
"""
Test script for car damage detection model (model4)
Validates model performance and runs sample inference
Classes: damaged, good_condition
"""

import os
import json
from pathlib import Path
from ultralytics import YOLO
import argparse
from inference import CarDamageDetectionInference


def find_best_model(runs_dir: str = 'runs') -> str:
    """
    Find the best trained model in the runs directory.
    
    Returns:
        Path to the best model file
    """
    runs_path = Path(runs_dir)
    
    # Look for car_damage_2class_model4 runs
    model_dirs = list(runs_path.glob('car_damage_2class_model4*'))
    
    if not model_dirs:
        raise FileNotFoundError("No trained models found. Please train a model first.")
    
    # Get the most recent run
    latest_run = max(model_dirs, key=os.path.getmtime)
    best_model_path = latest_run / 'weights' / 'best.pt'
    
    if not best_model_path.exists():
        raise FileNotFoundError(f"Best model not found in {latest_run}")
    
    return str(best_model_path)


def validate_model(model_path: str, data_yaml: str) -> dict:
    """
    Run validation on the trained model.
    
    Args:
        model_path: Path to the trained model
        data_yaml: Path to data.yaml configuration
    
    Returns:
        Validation metrics
    """
    print(f"\n=== Model Validation ===")
    print(f"Model: {model_path}")
    print(f"Data config: {data_yaml}")
    
    model = YOLO(model_path)
    results = model.val(data=data_yaml)
    
    metrics = {
        'map50': float(results.box.map50),
        'map50_95': float(results.box.map),
        'precision': float(results.box.mp),
        'recall': float(results.box.mr),
        'fitness': float(results.fitness)
    }
    
    print(f"\nValidation Results:")
    print(f"mAP@0.5: {metrics['map50']:.4f}")
    print(f"mAP@0.5:0.95: {metrics['map50_95']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Fitness: {metrics['fitness']:.4f}")
    
    return metrics


def test_inference(model_path: str, test_images_dir: str, 
                  confidence_threshold: float = 0.25) -> dict:
    """
    Test inference on sample images.
    
    Args:
        model_path: Path to the trained model
        test_images_dir: Directory containing test images
        confidence_threshold: Confidence threshold for detections
    
    Returns:
        Inference test results
    """
    print(f"\n=== Inference Testing ===")
    
    # Initialize inference pipeline
    inference = CarDamageDetectionInference(
        model_path=model_path,
        confidence_threshold=confidence_threshold
    )
    
    # Run batch inference on test images
    results = inference.predict_batch(
        image_dir=test_images_dir,
        output_dir='test_inference_results',
        save_images=True
    )
    
    # Calculate statistics
    total_images = len(results)
    images_with_detections = sum(1 for r in results if r['num_detections'] > 0)
    total_detections = sum(r['num_detections'] for r in results)
    avg_inference_time = sum(r['inference_time_ms'] for r in results) / total_images if total_images > 0 else 0
    
    # Calculate class distribution
    class_counts = {'damaged': 0, 'good_condition': 0}
    for result in results:
        for detection in result['detections']:
            class_name = detection['class']
            if class_name in class_counts:
                class_counts[class_name] += 1
    
    test_stats = {
        'total_images': total_images,
        'images_with_detections': images_with_detections,
        'detection_rate': images_with_detections / total_images if total_images > 0 else 0,
        'total_detections': total_detections,
        'avg_detections_per_image': total_detections / total_images if total_images > 0 else 0,
        'avg_inference_time_ms': round(avg_inference_time, 2),
        'class_distribution': class_counts
    }
    
    print(f"\nInference Test Results:")
    print(f"Total test images: {test_stats['total_images']}")
    print(f"Images with detections: {test_stats['images_with_detections']}")
    print(f"Detection rate: {test_stats['detection_rate']:.2%}")
    print(f"Total detections: {test_stats['total_detections']}")
    print(f"Average detections per image: {test_stats['avg_detections_per_image']:.2f}")
    print(f"Average inference time: {test_stats['avg_inference_time_ms']} ms")
    print(f"\nClass distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")
    
    return test_stats


def analyze_class_performance(model_path: str, data_yaml: str) -> dict:
    """
    Analyze per-class performance metrics.
    
    Args:
        model_path: Path to the trained model
        data_yaml: Path to data.yaml configuration
    
    Returns:
        Per-class performance metrics
    """
    print(f"\n=== Per-Class Performance Analysis ===")
    
    model = YOLO(model_path)
    results = model.val(data=data_yaml)
    
    # Extract per-class metrics if available
    class_names = ['damaged', 'good_condition']
    class_metrics = {}
    
    if hasattr(results.box, 'ap_class_index') and hasattr(results.box, 'ap'):
        for i, class_name in enumerate(class_names):
            if i < len(results.box.ap):
                class_metrics[class_name] = {
                    'ap50': float(results.box.ap50[i]) if i < len(results.box.ap50) else 0.0,
                    'ap50_95': float(results.box.ap[i]) if i < len(results.box.ap) else 0.0
                }
    
    print(f"Per-class AP@0.5:")
    for class_name, metrics in class_metrics.items():
        print(f"  {class_name}: {metrics['ap50']:.4f}")
    
    return class_metrics


def analyze_binary_classification(inference_stats: dict) -> dict:
    """
    Analyze binary classification performance (damaged vs good_condition).
    
    Args:
        inference_stats: Inference statistics from test_inference
    
    Returns:
        Binary classification analysis
    """
    print(f"\n=== Binary Classification Analysis ===")
    
    class_dist = inference_stats.get('class_distribution', {})
    damaged_count = class_dist.get('damaged', 0)
    good_count = class_dist.get('good_condition', 0)
    total_detections = damaged_count + good_count
    
    if total_detections > 0:
        damaged_ratio = damaged_count / total_detections
        good_ratio = good_count / total_detections
        
        binary_analysis = {
            'total_detections': total_detections,
            'damaged_detections': damaged_count,
            'good_condition_detections': good_count,
            'damaged_ratio': damaged_ratio,
            'good_condition_ratio': good_ratio,
            'classification_balance': abs(damaged_ratio - 0.5)  # How far from 50/50 split
        }
        
        print(f"Total detections: {total_detections}")
        print(f"Damaged detections: {damaged_count} ({damaged_ratio:.2%})")
        print(f"Good condition detections: {good_count} ({good_ratio:.2%})")
        print(f"Classification balance (deviation from 50/50): {binary_analysis['classification_balance']:.2%}")
        
        return binary_analysis
    else:
        print("No detections found for binary classification analysis")
        return {}


def main():
    parser = argparse.ArgumentParser(description='Test car damage detection model (Model4)')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model (auto-detect if not provided)')
    parser.add_argument('--data', type=str, default='data4_processed/data.yaml',
                       help='Path to data.yaml configuration')
    parser.add_argument('--test-images', type=str, default='data4_processed/test/images',
                       help='Directory containing test images')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold for inference testing')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip model validation')
    parser.add_argument('--skip-inference', action='store_true',
                       help='Skip inference testing')
    parser.add_argument('--skip-class-analysis', action='store_true',
                       help='Skip per-class performance analysis')
    
    args = parser.parse_args()
    
    # Find model if not provided
    if args.model is None:
        try:
            model_path = find_best_model()
            print(f"Auto-detected model: {model_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
    else:
        model_path = args.model
        if not Path(model_path).exists():
            print(f"Error: Model not found: {model_path}")
            return
    
    # Verify data configuration
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data configuration not found: {data_path}")
        return
    
    # Verify test images directory
    test_images_path = Path(args.test_images)
    if not test_images_path.exists():
        print(f"Error: Test images directory not found: {test_images_path}")
        return
    
    results = {
        'model_path': model_path,
        'data_config': str(data_path),
        'test_images_dir': str(test_images_path),
        'confidence_threshold': args.conf,
        'classes': ['damaged', 'good_condition']
    }
    
    # Run validation
    if not args.skip_validation:
        try:
            validation_metrics = validate_model(model_path, str(data_path))
            results['validation_metrics'] = validation_metrics
        except Exception as e:
            print(f"Validation failed: {e}")
    
    # Run per-class analysis
    if not args.skip_class_analysis:
        try:
            class_metrics = analyze_class_performance(model_path, str(data_path))
            results['class_metrics'] = class_metrics
        except Exception as e:
            print(f"Class analysis failed: {e}")
    
    # Run inference testing
    if not args.skip_inference:
        try:
            inference_stats = test_inference(model_path, str(test_images_path), args.conf)
            results['inference_stats'] = inference_stats
            
            # Run binary classification analysis
            binary_analysis = analyze_binary_classification(inference_stats)
            if binary_analysis:
                results['binary_analysis'] = binary_analysis
                
        except Exception as e:
            print(f"Inference testing failed: {e}")
    
    # Save results
    results_file = 'model4_test_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Testing Complete ===")
    print(f"Results saved to: {results_file}")
    
    # Print summary
    if 'validation_metrics' in results:
        print(f"\nModel Performance Summary:")
        print(f"mAP@0.5: {results['validation_metrics']['map50']:.4f}")
        print(f"Precision: {results['validation_metrics']['precision']:.4f}")
        print(f"Recall: {results['validation_metrics']['recall']:.4f}")
    
    if 'inference_stats' in results:
        print(f"\nInference Performance:")
        print(f"Detection rate: {results['inference_stats']['detection_rate']:.2%}")
        print(f"Average inference time: {results['inference_stats']['avg_inference_time_ms']} ms")
        print(f"Class distribution: {results['inference_stats']['class_distribution']}")
    
    if 'binary_analysis' in results:
        print(f"\nBinary Classification Summary:")
        print(f"Damaged ratio: {results['binary_analysis']['damaged_ratio']:.2%}")
        print(f"Good condition ratio: {results['binary_analysis']['good_condition_ratio']:.2%}")


if __name__ == '__main__':
    main()