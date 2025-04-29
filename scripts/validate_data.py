#!/usr/bin/env python3
"""
Validate dataset structure for VITON-HD
"""

import os
import argparse
from pathlib import Path

def validate_data(dataroot, phases=['train', 'test']):
    dataroot = Path(dataroot)
    errors = []
    
    for phase in phases:
        dirs = {
            'img': dataroot / f'{phase}_img',
            'parse': dataroot / f'{phase}_parse',
            'cloth': dataroot / f'{phase}_cloth',
            'cloth_mask': dataroot / f'{phase}_cloth_mask'
        }
        
        # Check if directories exist
        for name, d in dirs.items():
            if not d.exists():
                errors.append(f"Directory {d} does not exist")
        
        if errors:
            continue
        
        # Get image files
        img_files = sorted([f for f in os.listdir(dirs['img']) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        valid_files = []
        
        for img_file in img_files:
            base_name = os.path.splitext(img_file)[0]
            
            parse_path = dirs['parse'] / f"{base_name}.png"
            cloth_path = dirs['cloth'] / f"{base_name}.jpg"
            cloth_mask_path = dirs['cloth_mask'] / f"{base_name}.png"
            
            if not cloth_path.exists():
                cloth_path = dirs['cloth'] / f"{base_name}.png"
                if not cloth_path.exists():
                    cloth_path = dirs['cloth'] / f"{base_name}.jpeg"
            
            if not parse_path.exists():
                errors.append(f"Missing parse file: {parse_path}")
            if not cloth_path.exists():
                errors.append(f"Missing cloth file for {base_name}")
            if not cloth_mask_path.exists():
                errors.append(f"Missing cloth mask: {cloth_mask_path}")
            
            if parse_path.exists() and cloth_path.exists() and cloth_mask_path.exists():
                valid_files.append(base_name)
        
        print(f"{phase.capitalize()} dataset: Found {len(valid_files)} valid samples")
    
    if errors:
        print("\nErrors found:")
        for error in errors:
            print(f"- {error}")
        return False
    else:
        print("\nDataset validation passed!")
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate VITON-HD dataset")
    parser.add_argument('--dataroot', type=str, default='./data', help='path to dataset')
    args = parser.parse_args()
    
    success = validate_data(args.dataroot)
    exit(0 if success else 1)