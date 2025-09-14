#!/usr/bin/env python3
"""
Preprocessing pipeline for data2.zip - Car Exterior Damage Detection
Filters dataset to only detect 'crack' class as requested
"""

import os
import shutil
import yaml
from pathlib import Path
import argparse

def create_crack_only_dataset(source_dir, output_dir):
    """
    Create a new dataset containing only 'crack' class annotations
    
    Args:
        source_dir: Path to extracted data2 directory
        output_dir: Path where filtered dataset will be created
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
    crack_class_id = original_classes.index('crack')  # Should be 0
    
    print(f"Original classes: {original_classes}")
    print(f"Crack class ID: {crack_class_id}")
    
    stats = {'train': 0, 'valid': 0, 'test': 0}
    
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
                
            # Read label file and filter for crack class only
            crack_annotations = []
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        if class_id == crack_class_id:
                            # Keep this annotation but change class_id to 0 (since crack is now the only class)
                            crack_annotations.append(f"0 {' '.join(parts[1:])}")
            
            # Only copy image and create label if there are crack annotations
            if crack_annotations:
                # Copy image
                shutil.copy2(image_file, output_images_dir / image_file.name)
                
                # Create new label file with crack annotations only
                with open(output_labels_dir / f"{image_file.stem}.txt", 'w') as f:
                    f.write('\n'.join(crack_annotations) + '\n')
                
                processed_count += 1
        
        stats[split] = processed_count
        print(f"Processed {split}: {processed_count} images with crack annotations")
    
    # Create new data.yaml for crack-only dataset
    new_config = {
        'train': '../train/images',
        'val': '../valid/images', 
        'test': '../test/images',
        'nc': 1,
        'names': ['crack']
    }
    
    with open(output_path / 'data.yaml', 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False)
    
    print(f"\nDataset filtering complete!")
    print(f"Total images with crack annotations:")
    for split, count in stats.items():
        print(f"  {split}: {count}")
    print(f"\nNew dataset created at: {output_path}")
    print(f"Configuration saved to: {output_path / 'data.yaml'}")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='Preprocess data2 for crack detection only')
    parser.add_argument('--source', default='data2_extracted', 
                       help='Source directory containing extracted data2')
    parser.add_argument('--output', default='data2_crack_only',
                       help='Output directory for filtered dataset')
    
    args = parser.parse_args()
    
    source_dir = Path(args.source)
    output_dir = Path(args.output)
    
    if not source_dir.exists():
        print(f"Error: Source directory {source_dir} does not exist")
        return
    
    print(f"Creating crack-only dataset from {source_dir} to {output_dir}")
    stats = create_crack_only_dataset(source_dir, output_dir)
    
    print("\nPreprocessing completed successfully!")

if __name__ == '__main__':
    main()