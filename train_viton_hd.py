#!/usr/bin/env python3
"""
VITON-HD Training Script
Based on the paper: Diffusion VTON: High-Fidelity Virtual Try-On Network via Mask-Aware Diffusion Model
Adapted for custom datasets and the CatVTON framework
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
import wandb
import random
import json
from pathlib import Path

# Custom modules - these will need to be implemented based on the paper
from models.networks import Generator, Discriminator, VGGLoss, FeatureExtractor
from models.mask_generator import MaskGenerator
from models.appearance_flow import AppearanceFlowNet
from data.dataset import VITONDataset

def parse_args():
    parser = argparse.ArgumentParser(description='VITON-HD Training')
    
    # Dataset parameters
    parser.add_argument('--dataroot', type=str, default='./data', help='path to dataset')
    parser.add_argument('--dataset_mode', type=str, default='paired', help='paired or unpaired')
    parser.add_argument('--phase', type=str, default='train', help='train or test')
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    
    # Model parameters
    parser.add_argument('--name', type=str, default='viton_hd', help='name of the experiment')
    parser.add_argument('--img_size', type=int, default=512, help='input image size')
    parser.add_argument('--segmentation', action='store_true', help='use segmentation module')
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--save_freq', type=int, default=10, help='frequency of saving checkpoints')
    parser.add_argument('--log_freq', type=int, default=100, help='frequency of logging')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='checkpoint to resume training')
    parser.add_argument('--log_wandb', action='store_true', help='use wandb for logging')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    
    # Loss weights
    parser.add_argument('--lambda_l1', type=float, default=10.0, help='weight for L1 loss')
    parser.add_argument('--lambda_vgg', type=float, default=5.0, help='weight for VGG loss')
    parser.add_argument('--lambda_mask', type=float, default=2.0, help='weight for mask loss')
    parser.add_argument('--lambda_gan', type=float, default=1.0, help='weight for GAN loss')
    
    args = parser.parse_args()
    return args

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def create_dataset(args):
    """Create dataset and dataloader"""
    dataset = VITONDataset(
        root=args.dataroot,
        phase=args.phase,
        img_size=args.img_size,
        mode=args.dataset_mode
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    return dataset, dataloader

def build_models(args):
    """Build generator and discriminator models"""
    generator = Generator(
        input_channels=24,  # Person representation + clothing features
        output_channels=3   # RGB image
    )
    
    discriminator = Discriminator(
        input_channels=6,   # Generated image + ground truth or input
        ndf=64              # Number of discriminator filters
    )
    
    appearance_flow = AppearanceFlowNet()
    mask_generator = MaskGenerator()
    
    # Initialize feature extractor for perceptual loss
    feature_extractor = FeatureExtractor()
    vgg_loss = VGGLoss()
    
    # Move models to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    appearance_flow = appearance_flow.to(device)
    mask_generator = mask_generator.to(device)
    feature_extractor = feature_extractor.to(device)
    vgg_loss = vgg_loss.to(device)
    
    return {
        'generator': generator,
        'discriminator': discriminator,
        'appearance_flow': appearance_flow,
        'mask_generator': mask_generator,
        'feature_extractor': feature_extractor,
        'vgg_loss': vgg_loss,
        'device': device
    }

def build_optimizers(models, args):
    """Initialize optimizers"""
    optimizer_G = Adam(
        list(models['generator'].parameters()) + 
        list(models['appearance_flow'].parameters()) + 
        list(models['mask_generator'].parameters()),
        lr=args.lr,
        betas=(args.beta1, 0.999)
    )
    
    optimizer_D = Adam(
        models['discriminator'].parameters(),
        lr=args.lr * 0.5,  # Discriminator typically uses a lower learning rate
        betas=(args.beta1, 0.999)
    )
    
    scheduler_G = StepLR(optimizer_G, step_size=20, gamma=0.5)
    scheduler_D = StepLR(optimizer_D, step_size=20, gamma=0.5)
    
    return {
        'optimizer_G': optimizer_G,
        'optimizer_D': optimizer_D,
        'scheduler_G': scheduler_G,
        'scheduler_D': scheduler_D
    }

def save_checkpoint(models, optimizers, epoch, args):
    """Save model checkpoints"""
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    save_dict = {
        'epoch': epoch,
        'generator': models['generator'].state_dict(),
        'discriminator': models['discriminator'].state_dict(),
        'appearance_flow': models['appearance_flow'].state_dict(),
        'mask_generator': models['mask_generator'].state_dict(),
        'optimizer_G': optimizers['optimizer_G'].state_dict(),
        'optimizer_D': optimizers['optimizer_D'].state_dict(),
        'scheduler_G': optimizers['scheduler_G'].state_dict(),
        'scheduler_D': optimizers['scheduler_D'].state_dict(),
    }
    
    save_path = os.path.join(args.checkpoint_dir, f"{args.name}_epoch_{epoch}.pth")
    torch.save(save_dict, save_path)
    print(f"Checkpoint saved to {save_path}")

def load_checkpoint(path, models, optimizers):
    """Load model checkpoints"""
    if not os.path.exists(path):
        print(f"Checkpoint {path} does not exist, starting from scratch")
        return 0
    
    print(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location='cpu')
    
    models['generator'].load_state_dict(checkpoint['generator'])
    models['discriminator'].load_state_dict(checkpoint['discriminator'])
    models['appearance_flow'].load_state_dict(checkpoint['appearance_flow'])
    models['mask_generator'].load_state_dict(checkpoint['mask_generator'])
    
    optimizers['optimizer_G'].load_state_dict(checkpoint['optimizer_G'])
    optimizers['optimizer_D'].load_state_dict(checkpoint['optimizer_D'])
    optimizers['scheduler_G'].load_state_dict(checkpoint['scheduler_G'])
    optimizers['scheduler_D'].load_state_dict(checkpoint['scheduler_D'])
    
    return checkpoint['epoch'] + 1

def train_epoch(models, optimizers, dataloader, epoch, args):
    """Train for one epoch"""
    models['generator'].train()
    models['discriminator'].train()
    models['appearance_flow'].train()
    models['mask_generator'].train()
    
    device = models['device']
    running_losses = {
        'g_total': 0.0,
        'd_total': 0.0,
        'l1': 0.0,
        'vgg': 0.0,
        'gan': 0.0,
        'mask': 0.0
    }
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for i, data in enumerate(pbar):
        # Get data and move to device
        person_image = data['person_image'].to(device)
        cloth_image = data['cloth_image'].to(device)
        person_parse = data['person_parse'].to(device)
        cloth_mask = data['cloth_mask'].to(device)
        target_image = data['target_image'].to(device)
        
        batch_size = person_image.size(0)
        
        # First stage: Generate warped clothing using appearance flow
        flow_output = models['appearance_flow'](person_image, cloth_image, person_parse)
        warped_cloth = flow_output['warped_cloth']
        
        # Second stage: Generate mask
        mask_output = models['mask_generator'](warped_cloth, person_parse)
        generated_mask = mask_output['mask']
        
        # Final stage: Generate try-on image
        person_representation = torch.cat([person_image, person_parse, warped_cloth, generated_mask], dim=1)
        generated_image = models['generator'](person_representation)
        
        # Update Discriminator
        optimizers['optimizer_D'].zero_grad()
        
        # Real image pair
        real_pair = torch.cat([target_image, person_image], dim=1)
        pred_real = models['discriminator'](real_pair)
        
        # Fake image pair
        fake_pair = torch.cat([generated_image.detach(), person_image], dim=1)
        pred_fake = models['discriminator'](fake_pair)
        
        # GAN loss
        d_loss_real = -torch.mean(pred_real)
        d_loss_fake = torch.mean(pred_fake)
        d_loss = d_loss_real + d_loss_fake
        
        d_loss.backward()
        optimizers['optimizer_D'].step()
        
        # Update Generator
        optimizers['optimizer_G'].zero_grad()
        
        # GAN loss
        fake_pair = torch.cat([generated_image, person_image], dim=1)
        pred_fake = models['discriminator'](fake_pair)
        g_gan_loss = -torch.mean(pred_fake) * args.lambda_gan
        
        # L1 loss
        g_l1_loss = F.l1_loss(generated_image, target_image) * args.lambda_l1
        
        # VGG perceptual loss
        g_vgg_loss = models['vgg_loss'](generated_image, target_image) * args.lambda_vgg
        
        # Mask loss
        g_mask_loss = F.binary_cross_entropy_with_logits(generated_mask, cloth_mask) * args.lambda_mask
        
        # Total generator loss
        g_total_loss = g_gan_loss + g_l1_loss + g_vgg_loss + g_mask_loss
        
        g_total_loss.backward()
        optimizers['optimizer_G'].step()
        
        # Update losses
        running_losses['g_total'] += g_total_loss.item()
        running_losses['d_total'] += d_loss.item()
        running_losses['l1'] += g_l1_loss.item()
        running_losses['vgg'] += g_vgg_loss.item()
        running_losses['gan'] += g_gan_loss.item()
        running_losses['mask'] += g_mask_loss.item()
        
        if i % args.log_freq == 0:
            # Log current batch losses
            log_info = {
                'epoch': epoch,
                'iter': i,
                'g_total': g_total_loss.item(),
                'd_total': d_loss.item(),
                'l1': g_l1_loss.item(),
                'vgg': g_vgg_loss.item(),
                'gan': g_gan_loss.item(),
                'mask': g_mask_loss.item()
            }
            
            # Update progress bar
            pbar.set_postfix(**{k: f"{v:.4f}" for k, v in log_info.items() if k not in ['epoch', 'iter']})
            
            # Log to wandb if enabled
            if args.log_wandb:
                wandb.log(log_info)
                
                # Log images
                if i % (args.log_freq * 5) == 0:
                    # Sample some images for visualization
                    idx = min(3, batch_size-1)
                    wandb.log({
                        "person_image": wandb.Image(person_image[idx].cpu()),
                        "cloth_image": wandb.Image(cloth_image[idx].cpu()),
                        "warped_cloth": wandb.Image(warped_cloth[idx].cpu()),
                        "generated_mask": wandb.Image(generated_mask[idx].cpu()),
                        "generated_image": wandb.Image(generated_image[idx].cpu()),
                        "target_image": wandb.Image(target_image[idx].cpu())
                    })
    
    # Update learning rate schedulers
    optimizers['scheduler_G'].step()
    optimizers['scheduler_D'].step()
    
    # Calculate average losses
    num_batches = len(dataloader)
    for k in running_losses:
        running_losses[k] /= num_batches
    
    return running_losses

def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create checkpoints directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Initialize wandb if enabled
    if args.log_wandb:
        wandb.init(project=f"viton-hd-{args.name}", config=vars(args))
    
    # Create dataset and dataloader
    dataset, dataloader = create_dataset(args)
    print(f"Dataset size: {len(dataset)}")
    
    # Build models
    models = build_models(args)
    
    # Build optimizers
    optimizers = build_optimizers(models, args)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume is not None:
        start_epoch = load_checkpoint(args.resume, models, optimizers)
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # Train for one epoch
        losses = train_epoch(models, optimizers, dataloader, epoch, args)
        
        # Log epoch losses
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch} completed in {epoch_time:.2f}s")
        print(f"Generator loss: {losses['g_total']:.4f}, Discriminator loss: {losses['d_total']:.4f}")
        print(f"L1 loss: {losses['l1']:.4f}, VGG loss: {losses['vgg']:.4f}")
        print(f"GAN loss: {losses['gan']:.4f}, Mask loss: {losses['mask']:.4f}")
        
        if args.log_wandb:
            wandb.log({
                'epoch': epoch,
                'epoch_g_total': losses['g_total'],
                'epoch_d_total': losses['d_total'],
                'epoch_l1': losses['l1'],
                'epoch_vgg': losses['vgg'],
                'epoch_gan': losses['gan'],
                'epoch_mask': losses['mask'],
                'learning_rate_G': optimizers['scheduler_G'].get_last_lr()[0],
                'learning_rate_D': optimizers['scheduler_D'].get_last_lr()[0]
            })
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            save_checkpoint(models, optimizers, epoch, args)
    
    print("Training completed!")

if __name__ == "__main__":
    main()
