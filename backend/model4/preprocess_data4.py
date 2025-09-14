#!/usr/bin/env python3
"""
Preprocessing pipeline for data4.zip - Car Damage Detection
Processes dataset with classes: damaged, good_condition
"""

import os
import shutil
import yaml
from pathlib import Path
import argparse

def process_data4_dataset(source_dir, output_dir):
    """
    Process data4 dataset and prepare it for training
    
    Args:
        source_dir: Path to extracted data4 directory
        output_dir: Path where processed dataset will be created
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create output directory structure
    for split in ['train', 'valid', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Load original data.yaml to get class mapping
    with open(source_path / 'data.yaml', 'r') as f:
        original_config = yaml.safe_load(f)
    
    original_classes = original_config['names']
    print(f"Original classes: {original_classes}")
    print(f"Number of classes: {len(original_classes)}")
    
    stats = {'train': 0, 'valid': 0, 'test': 0}
    class_stats = {split: {cls: 0 for cls in original_classes} for split in ['train', 'valid', 'test']}
    
    # Process each split
    for split in ['train', 'valid', 'test']:
        images_dir = source_path / split / 'images'
        labels_dir = source_path / split / 'labels'
        
        if not images_dir.exists():
            print(f"Warning: {images_dir} does not exist, skipping {split}")
            continue
            
        output_images_dir = output_path / split / 'images'
        output_labels_dir = output_path / split / 'labels'
        
        processed_count = 0
        
        # Process each image
        for image_file in images_dir.glob('*.jpg'):
            label_file = labels_dir / f"{image_file.stem}.txt"
            
            if not label_file.exists():
                continue
                
            # Read label file and process annotations
            annotations = []
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        if 0 <= class_id < len(original_classes):
                            # Keep annotation as is (class IDs are already correct)
                            annotations.append(line.strip())
                            class_stats[split][original_classes[class_id]] += 1
            
            # Copy image and label if there are valid annotations
            if annotations:
                # Copy image
                shutil.copy2(image_file, output_images_dir / image_file.name)
                
                # Create label file
                with open(output_labels_dir / f"{image_file.stem}.txt", 'w') as f:
                    f.write('\n'.join(annotations) + '\n')
                
                processed_count += 1
        
        stats[split] = processed_count
        print(f"Processed {split}: {processed_count} images")
    
    # Create new data.yaml for the processed dataset
    new_config = {
        'train': '../train/images',
        'val': '../valid/images', 
        'test': '../test/images',
        'nc': len(original_classes),
        'names': original_classes
    }
    
    with open(output_path / 'data.yaml', 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False)
    
    print(f"\nDataset processing complete!")
    print(f"Total processed images:")
    for split, count in stats.items():
        print(f"  {split}: {count}")
    
    print(f"\nClass distribution:")
    for split in ['train', 'valid', 'test']:
        print(f"  {split}:")
        for cls, count in class_stats[split].items():
            print(f"    {cls}: {count}")
    
    print(f"\nProcessed dataset created at: {output_path}")
    print(f"Configuration saved to: {output_path / 'data.yaml'}")
    
    return stats, class_stats

def verify_dataset(dataset_dir):
    """
    Verify the processed dataset structure and contents
    """
    dataset_path = Path(dataset_dir)
    
    print(f"\n=== Dataset Verification ===")
    print(f"Dataset path: {dataset_path}")
    
    # Check data.yaml
    data_yaml = dataset_path / 'data.yaml'
    if data_yaml.exists():
        with open(data_yaml, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Classes ({config['nc']}): {config['names']}")
    else:
        print("Warning: data.yaml not found")
        return False
    
    # Check each split
    for split in ['train', 'valid', 'test']:
        images_dir = dataset_path / split / 'images'
        labels_dir = dataset_path / split / 'labels'
        
        if images_dir.exists() and labels_dir.exists():
            image_count = len(list(images_dir.glob('*.jpg')))
            label_count = len(list(labels_dir.glob('*.txt')))
            print(f"{split}: {image_count} images, {label_count} labels")
            
            if image_count != label_count:
                print(f"  Warning: Mismatch between images and labels in {split}")
        else:
            print(f"Warning: {split} directories not found")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Preprocess data4 for car damage detection')
    parser.add_argument('--source', default='data4_extracted', 
                       help='Source directory containing extracted data4')
    parser.add_argument('--output', default='data4_processed',
                       help='Output directory for processed dataset')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify existing processed dataset')
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_dataset(args.output)
        return
    
    source_dir = Path(args.source)
    output_dir = Path(args.output)
    
    if not source_dir.exists():
        print(f"Error: Source directory {source_dir} does not exist")
        return
    
    print(f"Processing data4 dataset from {source_dir} to {output_dir}")
    stats, class_stats = process_data4_dataset(source_dir, output_dir)
    
    # Verify the processed dataset
    verify_dataset(output_dir)
    
    print("\nPreprocessing completed successfully!")

if __name__ == '__main__':
    main()