#!/usr/bin/env python3
"""
Simple test script for car defect detection inference.
Makes it easy to test different checkpoints and configurations.
"""

import os
import sys
import argparse
from pathlib import Path


def find_latest_checkpoint(runs_dir="../runs"):
    """
    Find the latest checkpoint from training runs.
    
    Returns:
        Path to the best.pt file from the latest run
    """
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        return None
    
    # Find all training run directories
    run_dirs = [d for d in runs_path.iterdir() if d.is_dir() and d.name.startswith('car_defect_detection')]
    
    if not run_dirs:
        return None
    
    # Sort by modification time (latest first)
    run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Check for best.pt in the latest run
    for run_dir in run_dirs:
        best_pt = run_dir / "weights" / "best.pt"
        if best_pt.exists():
            return str(best_pt)
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Test car defect detection inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with latest checkpoint on test dataset
  python test_inference.py
  
  # Test with specific checkpoint
  python test_inference.py --model ../runs/car_defect_detection3/weights/best.pt
  
  # Test single image
  python test_inference.py --single-image ../merged_dataset/test/images/data1_car-scratch-repair_jpg.rf.0c857f5eb88823b67362aad038bd5d12.jpg
  
  # Test with different confidence threshold
  python test_inference.py --conf 0.5
  
  # Save results to JSON
  python test_inference.py --save-results
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='Path to model checkpoint (default: auto-detect latest)'
    )
    parser.add_argument(
        '--single-image', '-i',
        type=str,
        help='Test on single image instead of test dataset'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.3,
        help='Confidence threshold (default: 0.3)'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='IoU threshold (default: 0.45)'
    )
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Save results to JSON file'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress detailed output'
    )
    
    args = parser.parse_args()
    
    # Find model checkpoint
    if args.model:
        model_path = args.model
    else:
        model_path = find_latest_checkpoint()
        if not model_path:
            print("Error: No trained model found. Please specify --model or train a model first.")
            sys.exit(1)
        print(f"Using latest checkpoint: {model_path}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    # Build inference command
    cmd_parts = [
        "python", "inference.py",
        "--model", f'"{model_path}"',
        "--conf", str(args.conf),
        "--iou", str(args.iou)
    ]
    
    # Add input source
    if args.single_image:
        cmd_parts.extend(["--image", f'"{args.single_image}"'])
    else:
        cmd_parts.extend(["--directory", '"../merged_dataset/test/images"'])
    
    # Add output options
    if args.save_results:
        output_file = "inference_results.json"
        cmd_parts.extend(["--output", f'"{output_file}"'])
    
    if args.quiet:
        cmd_parts.append("--quiet")
    
    # Execute command
    cmd = " ".join(cmd_parts)
    print(f"\nRunning: {cmd}\n")
    
    exit_code = os.system(cmd)
    
    if exit_code == 0:
        print("\n✅ Inference completed successfully!")
        if args.save_results:
            print(f"📄 Results saved to: inference_results.json")
    else:
        print("\n❌ Inference failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()