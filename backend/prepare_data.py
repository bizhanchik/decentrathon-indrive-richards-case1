#!/usr/bin/env python3
"""
Data preparation script for car defect detection.
Merges data1/ and data2/ datasets into a single YOLO dataset.

Data1 classes: [0: dent, 1: dirt, 2: scratch]
Data2 classes: [0: scratch] -> maps to class 2 in merged dataset
"""

import os
import shutil
import zipfile
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


def extract_zip_files(zip_files: List[Path], extract_dir: Path) -> bool:
    """Extract zip files to the extraction directory."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    for zip_file in zip_files:
        print(f"Extracting {zip_file.name}...")
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"Successfully extracted {zip_file.name}")
        except Exception as e:
            print(f"Error extracting {zip_file.name}: {e}")
            return False
    
    return True


# Define class mapping for unified dataset
CLASS_MAPPING = {
    'Dent': 0,
    'Dislocation': 1,
    'Scratch': 2,
    'Shatter': 3,
    'damaged': 4,
    'severe damage': 5
}

def main():
    """Main function to prepare dataset."""
    # Define paths
    project_root = Path(__file__).parent.parent
    zips_dir = project_root / 'zips'
    extract_dir = project_root / 'data'
    output_dir = project_root / 'merged_dataset'
    
    print("=== Car Defect Dataset Preparation ===")
    print(f"Zips directory: {zips_dir}")
    print(f"Extract directory: {extract_dir}")
    print(f"Output directory: {output_dir}")
    
    # Verify input directories exist
    if not zips_dir.exists():
        print(f"Creating zips directory: {zips_dir}")
        zips_dir.mkdir(parents=True, exist_ok=True)
        print("Please place dataset zip files in the 'zips' directory and run this script again.")
        return
    
    # Check if there are zip files in the zips directory
    zip_files = list(zips_dir.glob('*.zip'))
    if not zip_files:
        print(f"No zip files found in {zips_dir}")
        print("Please place dataset zip files in the 'zips' directory and run this script again.")
        return
    
    # Extract zip files if needed
    if not extract_dir.exists() or not (extract_dir / 'train').exists():
        print(f"\n=== Extracting Dataset Files ===")
        if extract_dir.exists():
            print(f"Removing existing extract directory: {extract_dir}")
            shutil.rmtree(extract_dir, ignore_errors=True)
        
        if not extract_zip_files(zip_files, extract_dir):
            print("Error extracting zip files. Please check the zip files and try again.")
            return
    else:
        print(f"\n=== Using existing extracted dataset ===")
    
    # Verify extracted directories exist
    for split in ['train', 'valid', 'test']:
        if not (extract_dir / split).exists():
            print(f"Error: Expected directory '{split}' not found in extracted data.")
            print(f"Please ensure your zip files contain the proper YOLO dataset structure.")
            return
        
        for subdir in ['images', 'labels']:
            if not (extract_dir / split / subdir).exists():
                print(f"Error: Expected subdirectory '{subdir}' not found in '{split}' directory.")
                print(f"Please ensure your zip files contain the proper YOLO dataset structure.")
                return
    
    # Get class names from data.yaml if it exists
    data_yaml_path = extract_dir / 'data.yaml'
    if data_yaml_path.exists():
        try:
            with open(data_yaml_path, 'r') as f:
                data_yaml = yaml.safe_load(f)
                if 'names' in data_yaml:
                    class_names = data_yaml['names']
                    print(f"\nFound class names in data.yaml: {class_names}")
                    # Update CLASS_MAPPING if needed
                    for i, name in enumerate(class_names):
                        if name in CLASS_MAPPING and CLASS_MAPPING[name] != i:
                            print(f"Warning: Class '{name}' has different index in data.yaml ({i}) than in CLASS_MAPPING ({CLASS_MAPPING[name]})")
                        CLASS_MAPPING[name] = i
        except Exception as e:
            print(f"Error reading data.yaml: {e}")
    
    # Define standardized classes from the mapping
    merged_classes = list(CLASS_MAPPING.keys())
    
    print(f"\nStandardized classes: {merged_classes}")
    print(f"Class mapping configuration: {CLASS_MAPPING}")
    
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
    
    # Copy files directly from extract_dir to output_dir
    print("\n=== Processing Dataset ===")
    stats = {'train': 0, 'valid': 0, 'test': 0}
    
    for split in ['train', 'valid', 'test']:
        source_images = extract_dir / split / 'images'
        source_labels = extract_dir / split / 'labels'
        target_images = output_dir / split / 'images'
        target_labels = output_dir / split / 'labels'
        
        print(f"Processing {split} split...")
        
        # Copy images and labels
        for img_file in source_images.glob('*.jpg'):
            label_file = source_labels / f"{img_file.stem}.txt"
            
            # Copy image
            shutil.copy2(img_file, target_images / img_file.name)
            
            # Copy label if it exists
            if label_file.exists():
                shutil.copy2(label_file, target_labels / label_file.name)
            
            stats[split] += 1
    
    print(f"\n=== Processing Complete ===")
    print(f"Dataset stats: {stats}")
    print(f"Total images: {sum(stats.values())}")
    
    # Generate data.yaml
    generate_data_yaml(output_dir, merged_classes)
    
    print(f"\n=== Dataset Ready ===")
    print(f"Merged dataset location: {output_dir}")
    print(f"Configuration file: {output_dir / 'data.yaml'}")
    print("\nYou can now run train.py to start training!")


if __name__ == '__main__':
    main()