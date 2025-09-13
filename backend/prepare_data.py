#!/usr/bin/env python3
"""
Data preparation script for car defect detection.
Merges data1/ and data2/ datasets into a single YOLO dataset.

Data1 classes: [0: dent, 1: dirt, 2: scratch]
Data2 classes: [0: scratch] -> maps to class 2 in merged dataset
"""

import os
import shutil
from pathlib import Path
import yaml
import time
from typing import Dict, List


def create_directory_structure(output_dir: Path) -> None:
    """Create the output directory structure for YOLO dataset."""
    for split in ['train', 'valid', 'test']:
        for subdir in ['images', 'labels']:
            (output_dir / split / subdir).mkdir(parents=True, exist_ok=True)
    print(f"Created directory structure in {output_dir}")


def copy_files_and_update_labels(source_data_dir: Path, output_dir: Path, 
                                class_mapping: Dict[int, int], 
                                dataset_name: str) -> Dict[str, int]:
    """Copy images and labels, updating class indices according to mapping."""
    stats = {'train': 0, 'valid': 0, 'test': 0}
    
    for split in ['train', 'valid', 'test']:
        source_images = source_data_dir / split / 'images'
        source_labels = source_data_dir / split / 'labels'
        
        if not source_images.exists():
            print(f"Warning: {source_images} does not exist, skipping {split} split for {dataset_name}")
            continue
            
        target_images = output_dir / split / 'images'
        target_labels = output_dir / split / 'labels'
        
        # Copy and process each image and its corresponding label
        for img_file in source_images.glob('*.jpg'):
            label_file = source_labels / f"{img_file.stem}.txt"
            
            # Generate unique filename to avoid conflicts
            new_name = f"{dataset_name}_{img_file.name}"
            
            # Copy image
            shutil.copy2(img_file, target_images / new_name)
            
            # Process and copy label if it exists
            if label_file.exists():
                new_label_content = []
                with open(label_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            old_class = int(parts[0])
                            new_class = class_mapping.get(old_class, old_class)
                            new_line = f"{new_class} {' '.join(parts[1:])}"
                            new_label_content.append(new_line)
                
                # Use the same new name for label file (without extension)
                new_label_name = f"{dataset_name}_{img_file.stem}.txt"
                with open(target_labels / new_label_name, 'w') as f:
                    f.write('\n'.join(new_label_content))
            
            stats[split] += 1
    
    return stats


def generate_data_yaml(output_dir: Path, class_names: List[str]) -> None:
    """Generate the data.yaml configuration file."""
    data_config = {
        'train': str(output_dir / 'train' / 'images'),
        'val': str(output_dir / 'valid' / 'images'),
        'test': str(output_dir / 'test' / 'images'),
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"Generated data.yaml at {yaml_path}")
    print(f"Classes: {class_names}")


def main():
    """Main function to merge datasets."""
    # Define paths
    project_root = Path(__file__).parent.parent
    data1_dir = project_root / 'data1'
    data2_dir = project_root / 'data2'
    output_dir = project_root / 'merged_dataset'
    
    print("=== Car Defect Dataset Preparation ===")
    print(f"Data1 directory: {data1_dir}")
    print(f"Data2 directory: {data2_dir}")
    print(f"Output directory: {output_dir}")
    
    # Verify input directories exist
    if not data1_dir.exists():
        raise FileNotFoundError(f"Data1 directory not found: {data1_dir}")
    if not data2_dir.exists():
        raise FileNotFoundError(f"Data2 directory not found: {data2_dir}")
    
    # Define class mapping
    # Data1: [0: dent, 1: dirt, 2: scratch]
    # Data2: [0: scratch] -> maps to 2
    merged_classes = ['dent', 'dirt', 'scratch']
    data1_mapping = {0: 0, 1: 1, 2: 2}  # No change needed
    data2_mapping = {0: 2}  # Map scratch (0) to scratch (2)
    
    print(f"\nMerged classes: {merged_classes}")
    print(f"Data1 class mapping: {data1_mapping}")
    print(f"Data2 class mapping: {data2_mapping}")
    
    # Create output directory structure
    if output_dir.exists():
        print(f"\nRemoving existing output directory: {output_dir}")
        # Retry mechanism for Windows file permission issues
        for attempt in range(3):
            try:
                shutil.rmtree(output_dir)
                break
            except PermissionError as e:
                if attempt < 2:
                    print(f"Permission error (attempt {attempt + 1}/3): {e}")
                    print("Waiting 2 seconds and retrying...")
                    time.sleep(2)
                else:
                    print(f"Failed to remove directory after 3 attempts: {e}")
                    print("Please close any file explorers or applications using the merged_dataset folder and try again.")
                    raise
    
    create_directory_structure(output_dir)
    
    # Process data1
    print("\n=== Processing Data1 ===")
    data1_stats = copy_files_and_update_labels(data1_dir, output_dir, data1_mapping, 'data1')
    print(f"Data1 stats: {data1_stats}")
    
    # Process data2
    print("\n=== Processing Data2 ===")
    data2_stats = copy_files_and_update_labels(data2_dir, output_dir, data2_mapping, 'data2')
    print(f"Data2 stats: {data2_stats}")
    
    # Calculate total stats
    total_stats = {}
    for split in ['train', 'valid', 'test']:
        total_stats[split] = data1_stats[split] + data2_stats[split]
    
    print(f"\n=== Merge Complete ===")
    print(f"Total dataset stats: {total_stats}")
    print(f"Total images: {sum(total_stats.values())}")
    
    # Generate data.yaml
    generate_data_yaml(output_dir, merged_classes)
    
    print(f"\n=== Dataset Ready ===")
    print(f"Merged dataset location: {output_dir}")
    print(f"Configuration file: {output_dir / 'data.yaml'}")
    print("\nYou can now run train.py to start training!")


if __name__ == '__main__':
    main()