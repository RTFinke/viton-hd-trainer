#!/usr/bin/env python3
"""
Dataset implementation for VITON-HD
Based on the paper: Diffusion VTON: High-Fidelity Virtual Try-On Network via Mask-Aware Diffusion Model
"""

import os
import json
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from pathlib import Path

class VITONDataset(Dataset):
    def __init__(self, root, phase='train', img_size=512, mode='paired', 
                 semantic_nc=20, unpaired_prob=0.0, augment=True):
        self.root = Path(root)
        self.phase = phase
        self.img_size = img_size
        self.mode = mode
        self.semantic_nc = semantic_nc
        self.unpaired_prob = unpaired_prob if phase == 'train' else 0.0
        self.augment = augment and phase == 'train'
        
        self.image_dir = self.root / f'{phase}_img'
        self.parse_dir = self.root / f'{phase}_parse'
        self.cloth_dir = self.root / f'{phase}_cloth'
        self.cloth_mask_dir = self.root / f'{phase}_cloth_mask'
        
        for d in [self.image_dir, self.parse_dir, self.cloth_dir, self.cloth_mask_dir]:
            if not d.exists():
                raise ValueError(f"Directory {d} does not exist")
        
        self.data_list = self._get_data_list()
        self._setup_transforms()
    
    def _get_data_list(self):
        image_files = sorted([f for f in os.listdir(self.image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        valid_files = []
        
        for img_file in image_files:
            base_name = os.path.splitext(img_file)[0]
            
            parse_path = self.parse_dir / f"{base_name}.png"
            cloth_path = self.cloth_dir / f"{base_name}.jpg"
            cloth_mask_path = self.cloth_mask_dir / f"{base_name}.png"
            
            if not cloth_path.exists():
                cloth_path = self.cloth_dir / f"{base_name}.png"
                if not cloth_path.exists():
                    cloth_path = self.cloth_dir / f"{base_name}.jpeg"
            
            if (
                parse_path.exists() and
                cloth_path.exists() and
                cloth_mask_path.exists()
            ):
                valid_files.append(base_name)
        
        if not valid_files:
            raise ValueError(f"No valid data samples found in {self.image_dir}")
        
        print(f"Found {len(valid_files)} valid {'paired' if self.mode == 'paired' else 'unpaired'} data samples")
        return valid_files
    
    def _setup_transforms(self):
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.transform_mask = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
        
        if self.augment:
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
            ])
    
    def _get_person_parse(self, parse_path):
        try:
            parse = Image.open(parse_path)
            parse = self.transform_mask(parse)
            
            parse_tensor = torch.zeros(self.semantic_nc, self.img_size, self.img_size)
            for i in range(self.semantic_nc):
                parse_tensor[i] = (parse[0] == i).float()
            
            return parse_tensor
        except Exception as e:
            print(f"Error loading parse {parse_path}: {e}")
            raise
    
    def __getitem__(self, index):
        data_name = self.data_list[index]
        
        try:
            # Load person image
            person_img_path = self.image_dir / f"{data_name}.jpg"
            if not person_img_path.exists():
                person_img_path = self.image_dir / f"{data_name}.png"
                if not person_img_path.exists():
                    person_img_path = self.image_dir / f"{data_name}.jpeg"
            person_img = Image.open(person_img_path).convert('RGB')
            
            # Load person parsing
            parse_path = self.parse_dir / f"{data_name}.png"
            person_parse = self._get_person_parse(parse_path)
            
            if self.mode == 'paired' and (self.phase == 'test' or random.random() > self.unpaired_prob):
                cloth_name = data_name
                target_name = data_name
            else:
                cloth_name = random.choice(self.data_list)
                target_name = data_name
            
            # Load cloth image
            cloth_img_path = self.cloth_dir / f"{cloth_name}.jpg"
            if not cloth_img_path.exists():
                cloth_img_path = self.cloth_dir / f"{cloth_name}.png"
                if not cloth_img_path.exists():
                    cloth_img_path = self.cloth_dir / f"{cloth_name}.jpeg"
            cloth_img = Image.open(cloth_img_path).convert('RGB')
            
            # Load cloth mask
            cloth_mask_path = self.cloth_mask_dir / f"{cloth_name}.png"
            cloth_mask = Image.open(cloth_mask_path).convert('L')
            
            # Load target image
            target_img_path = self.image_dir / f"{target_name}.jpg"
            if not target_img_path.exists():
                target_img_path = self.image_dir / f"{target_name}.png"
                if not target_img_path.exists():
                    target_img_path = self.image_dir / f"{target_name}.jpeg"
            target_img = Image.open(target_img_path).convert('RGB')
            
            if self.augment:
                seed = random.randint(0, 2**32)
                
                random.seed(seed)
                torch.manual_seed(seed)
                person_img = self.aug_transform(person_img)
                
                random.seed(seed)
                torch.manual_seed(seed)
                target_img = self.aug_transform(target_img)
                
                cloth_img = self.aug_transform(cloth_img)
            
            person_img = self.transform(person_img)
            cloth_img = self.transform(cloth_img)
            cloth_mask = self.transform_mask(cloth_mask)
            target_img = self.transform(target_img)
            
            return {
                'person_image': person_img,
                'person_parse': person_parse,
                'cloth_image': cloth_img,
                'cloth_mask': cloth_mask,
                'target_image': target_img,
                'person_name': data_name,
                'cloth_name': cloth_name
            }
        
        except Exception as e:
            print(f"Error loading data {data_name}: {e}")
            raise
    
    def __len__(self):
        return len(self.data_list)