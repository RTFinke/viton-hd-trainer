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
    """Dataset for Virtual Try-On Network (VITON-HD)"""
    
    def __init__(self, root, phase='train', img_size=512, mode='paired', 
                 semantic_nc=20, unpaired_prob=0.0, augment=True):
        """
        Initialize the VITON dataset
        
        Args:
            root (str): Dataset root directory
            phase (str): 'train' or 'test'
            img_size (int): Image size
            mode (str): 'paired' or 'unpaired'
            semantic_nc (int): Number of semantic segmentation classes
            unpaired_prob (float): Probability of creating unpaired data during training
            augment (bool): Whether to use data augmentation
        """
        self.root = Path(root)
        self.phase = phase
        self.img_size = img_size
        self.mode = mode
        self.semantic_nc = semantic_nc
        self.unpaired_prob = unpaired_prob if phase == 'train' else 0.0
        self.augment = augment and phase == 'train'
        
        # Directory structure
        self.image_dir = self.root / f'{phase}_img'
        self.parse_dir = self.root / f'{phase}_parse'
        self.cloth_dir = self.root / f'{phase}_cloth'
        self.cloth_mask_dir = self.root / f'{phase}_cloth_mask'
        
        # Get list of data samples
        self.data_list = self._get_data_list()
        
        # Set up transformations
        self._setup_transforms()
        
    def _get_data_list(self):
        """Get list of data samples"""
        # Get list of images
        image_files = sorted(os.listdir(self.image_dir))
        # Filter for images
        image_files = [f for f in image_files if f.endswith(('.jpg', '.png'))]
        
        # For paired data, verify all components exist
        valid_files = []
        for img_file in image_files:
            base_name = os.path.splitext(img_file)[0]
            
            # Check if all required files exist
            if (
                os.path.exists(self.parse_dir / f"{base_name}.png") and
                os.path.exists(self.cloth_dir / f"{base_name}.jpg") and
                os.path.exists(self.cloth_mask_dir / f"{base_name}.png")
            ):
                valid_files.append(base_name)
        
        print(f"Found {len(valid_files)} valid {'paired' if self.mode == 'paired' else 'unpaired'} data samples")
        return valid_files
    
    def _setup_transforms(self):
        """Set up data transformations"""
        # Basic transformations
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Transform for segmentation masks (no normalization)
        self.transform_mask = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
        
        # Augmentation transforms
        if self.augment:
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
            ])
        
    def _get_person_parse(self, parse_path):
        """
        Load and process person parsing
        
        Args:
            parse_path: Path to parsing segmentation image
            
        Returns:
            One-hot encoded segmentation (C, H, W)
        """
        parse = Image.open(parse_path)
        parse = self.transform_mask(parse)
        
        # Convert to one-hot encoding
        parse_tensor = torch.zeros(self.semantic_nc, self.img_size, self.img_size)
        for i in range(self.semantic_nc):
            parse_tensor[i] = (parse[0] == i).float()
            
        return parse_tensor
    
    def __getitem__(self, index):
        """Get dataset item"""
        data_name = self.data_list[index]
        
        # Load person image
        person_img_path = self.image_dir / f"{data_name}.jpg"
        if not os.path.exists(person_img_path):
            person_img_path = self.image_dir / f"{data_name}.png"
        person_img = Image.open(person_img_path).convert('RGB')
        
        # Load person parsing
        parse_path = self.parse_dir / f"{data_name}.png"
        person_parse = self._get_person_parse(parse_path)
        
        # For paired data
        if self.mode == 'paired' and (self.phase == 'test' or random.random() > self.unpaired_prob):
            # Use the corresponding cloth
            cloth_name = data_name
            target_name = data_name
        else:
            # For unpaired data or with unpaired_prob in training, sample a random cloth
            cloth_name = random.choice(self.data_list)
            # In this case, the target is still the original person with their original clothes
            target_name = data_name
        
        # Load cloth image
        cloth_img_path = self.cloth_dir / f"{cloth_name}.jpg"
        if not os.path.exists(cloth_img_path):
            cloth_img_path = self.cloth_dir / f"{cloth_name}.png"
        cloth_img = Image.open(cloth_img_path).convert('RGB')
        
        # Load cloth mask
        cloth_mask_path = self.cloth_mask_dir / f"{cloth_name}.png"
        cloth_mask = Image.open(cloth_mask_path).convert('L')
        
        # Load target image (person wearing the original clothes)
        target_img_path = self.image_dir / f"{target_name}.jpg"
        if not os.path.exists(target_img_path):
            target_img_path = self.image_dir / f"{target_name}.png"
        target_img = Image.open(target_img_path).convert('RGB')
        
        # Apply the same augmentation to person and target if needed
        if self.augment:
            # Apply the same random transforms to person and target
            seed = random.randint(0, 2**32)
            
            # Set seed for person image
            random.seed(seed)
            torch.manual_seed(seed)
            person_img = self.aug_transform(person_img)
            
            # Set the same seed for target image
            random.seed(seed)
            torch.manual_seed(seed)
            target_img = self.aug_transform(target_img)
            
            # Cloth can have different augmentation
            cloth_img = self.aug_transform(cloth_img)
        
        # Apply transformations
        person_img = self.transform(person_img)
        cloth_img = self.transform(cloth_img)
        cloth_mask = self.transform_mask(cloth_mask)
        target_img = self.transform(target_img)
        
        # Create data dictionary
        data = {
            'person_image': person_img,
            'person_parse': person_parse,
            'cloth_image': cloth_img,
            'cloth_mask': cloth_mask,
            'target_image': target_img,
            'person_name': data_name,
            'cloth_name': cloth_name
        }
        
        return data
    
    def __len__(self):
        """Return the size of the dataset"""
        return len(self.data_list)
