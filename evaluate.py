#!/usr/bin/env python3
"""
Evaluation script for VITON-HD
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from models.networks import Generator, Discriminator, VGGLoss, FeatureExtractor
from models.mask_generator import MaskGenerator
from models.appearance_flow import AppearanceFlowNet
from data.dataset import VITONDataset

def parse_args():
    parser = argparse.ArgumentParser(description='VITON-HD Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to checkpoint')
    parser.add_argument('--dataroot', type=str, default='./data', help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--img_size', type=int, default=512, choices=[256, 512], help='image size')
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        raise ValueError(f"Checkpoint {args.checkpoint} does not exist")
    if not os.path.exists(args.dataroot):
        raise ValueError(f"Dataset path {args.dataroot} does not exist")
    
    return args

def compute_metrics(generated, target):
    generated = generated.permute(0, 2, 3, 1).cpu().numpy()
    target = target.permute(0, 2, 3, 1).cpu().numpy()
    
    ssim_scores = []
    psnr_scores = []
    
    for g, t in zip(generated, target):
        g = (g * 0.5 + 0.5).clip(0, 1)
        t = (t * 0.5 + 0.5).clip(0, 1)
        
        ssim_score = ssim(g, t, channel_axis=2, data_range=1.0)
        psnr_score = psnr(g, t, data_range=1.0)
        
        ssim_scores.append(ssim_score)
        psnr_scores.append(psnr_score)
    
    return np.mean(ssim_scores), np.mean(psnr_scores)

def main():
    args = parse_args()
    
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
    
    # Evaluation loop
    running_metrics = {'ssim': 0.0, 'psnr': 0.0}
    num_batches = 0
    
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Evaluation"):
            person_image = data['person_image'].to(device)
            cloth_image = data['cloth_image'].to(device)
            person_parse = data['person_parse'].to(device)
            target_image = data['target_image'].to(device)
            
            flow_output = appearance_flow(person_image, cloth_image, person_parse)
            warped_cloth = flow_output['warped_cloth']
            
            mask_output = mask_generator(warped_cloth, person_parse)
            generated_mask = mask_output['mask']
            
            person_representation = torch.cat([person_image, person_parse, warped_cloth, generated_mask], dim=1)
            generated_image = generator(person_representation)
            
            ssim_score, psnr_score = compute_metrics(generated_image, target_image)
            running_metrics['ssim'] += ssim_score
            running_metrics['psnr'] += psnr_score
            num_batches += 1
    
    # Average metrics
    for k in running_metrics:
        running_metrics[k] /= num_batches
    
    print(f"Evaluation results:")
    print(f"SSIM: {running_metrics['ssim']:.4f}")
    print(f"PSNR: {running_metrics['psnr']:.4f}")

if __name__ == "__main__":
    main()