#!/usr/bin/env python3
"""
Inference script for VITON-HD
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import argparse
from pathlib import Path
from tqdm import tqdm

from models.networks import Generator, Discriminator, VGGLoss, FeatureExtractor
from models.mask_generator import MaskGenerator
from models.appearance_flow import AppearanceFlowNet
from data.dataset import VITONDataset

def parse_args():
    parser = argparse.ArgumentParser(description='VITON-HD Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to checkpoint')
    parser.add_argument('--dataroot', type=str, default='./data', help='path to dataset')
    parser.add_argument('--output_dir', type=str, default='./results', help='output directory')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--img_size', type=int, default=512, choices=[256, 512], help='image size')
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        raise ValueError(f"Checkpoint {args.checkpoint} does not exist")
    if not os.path.exists(args.dataroot):
        raise ValueError(f"Dataset path {args.dataroot} does not exist")
    
    return args

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dataset
    dataset = VITONDataset(
        root=args.dataroot,
        phase='test',
        img_size=args.img_size,
        mode='paired'
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Build models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = Generator(input_channels=24, output_channels=3).to(device)
    appearance_flow = AppearanceFlowNet().to(device)
    mask_generator = MaskGenerator().to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    generator.load_state_dict(checkpoint['generator'])
    appearance_flow.load_state_dict(checkpoint['appearance_flow'])
    mask_generator.load_state_dict(checkpoint['mask_generator'])
    
    generator.eval()
    appearance_flow.eval()
    mask_generator.eval()
    
    # Inference loop
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Inference"):
            person_image = data['person_image'].to(device)
            cloth_image = data['cloth_image'].to(device)
            person_parse = data['person_parse'].to(device)
            person_name = data['person_name']
            cloth_name = data['cloth_name']
            
            flow_output = appearance_flow(person_image, cloth_image, person_parse)
            warped_cloth = flow_output['warped_cloth']
            
            mask_output = mask_generator(warped_cloth, person_parse)
            generated_mask = mask_output['mask']
            
            person_representation = torch.cat([person_image, person_parse, warped_cloth, generated_mask], dim=1)
            generated_image = generator(person_representation)
            
            # Save images
            for i, (gen_img, p_name, c_name) in enumerate(zip(generated_image, person_name, cloth_name)):
                gen_img = gen_img.cpu().permute(1, 2, 0).numpy()
                gen_img = (gen_img * 0.5 + 0.5).clip(0, 1) * 255
                gen_img = gen_img.astype(np.uint8)
                
                output_path = os.path.join(args.output_dir, f"{p_name}_{c_name}.png")
                Image.fromarray(gen_img).save(output_path)
    
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()