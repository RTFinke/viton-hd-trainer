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
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
import wandb
import random
import json
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Custom modules
from models.networks import Generator, Discriminator, VGGLoss, FeatureExtractor
from models.mask_generator import MaskGenerator
from models.appearance_flow import AppearanceFlowNet
from data.dataset import VITONDataset

def parse_args():
    parser = argparse.ArgumentParser(description='VITON-HD Training')
    
    # Dataset parameters
    parser.add_argument('--dataroot', type=str, default='./data', help='path to dataset')
    parser.add_argument('--dataset_mode', type=str, default='paired', choices=['paired', 'unpaired'], help='paired or unpaired')
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test'], help='train or test')
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    
    # Model parameters
    parser.add_argument('--name', type=str, default='viton_hd', help='name of the experiment')
    parser.add_argument('--img_size', type=int, default=512, choices=[256, 512], help='input image size')
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
    
    # Validate arguments
    if not os.path.exists(args.dataroot):
        raise ValueError(f"Dataset path {args.dataroot} does not exist")
    
    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_dataset(args, phase='train'):
    try:
        dataset = VITONDataset(
            root=args.dataroot,
            phase=phase,
            img_size=args.img_size,
            mode=args.dataset_mode
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(phase == 'train'),
            num_workers=args.workers,
            pin_memory=True,
            drop_last=(phase == 'train')
        )
        return dataset, dataloader
    except Exception as e:
        print(f"Error creating dataset: {e}")
        raise

def build_models(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        generator = Generator(input_channels=24, output_channels=3)
        discriminator = Discriminator(input_channels=6, ndf=64)
        appearance_flow = AppearanceFlowNet()
        mask_generator = MaskGenerator()
        feature_extractor = FeatureExtractor()
        vgg_loss = VGGLoss()
        
        # Move models to GPU and enable multi-GPU if available
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            generator = nn.DataParallel(generator)
            discriminator = nn.DataParallel(discriminator)
            appearance_flow = nn.DataParallel(appearance_flow)
            mask_generator = nn.DataParallel(mask_generator)
        
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
    except Exception as e:
        print(f"Error building models: {e}")
        raise

def build_optimizers(models, args):
    try:
        optimizer_G = Adam(
            list(models['generator'].parameters()) + 
            list(models['appearance_flow'].parameters()) + 
            list(models['mask_generator'].parameters()),
            lr=args.lr,
            betas=(args.beta1, 0.999)
        )
        optimizer_D = Adam(
            models['discriminator'].parameters(),
            lr=args.lr * 0.5,
            betas=(args.beta1, 0.999)
        )
        
        scheduler_G = CosineAnnealingLR(optimizer_G, T_max=args.epochs)
        scheduler_D = CosineAnnealingLR(optimizer_D, T_max=args.epochs)
        
        return {
            'optimizer_G': optimizer_G,
            'optimizer_D': optimizer_D,
            'scheduler_G': scheduler_G,
            'scheduler_D': scheduler_D
        }
    except Exception as e:
        print(f"Error building optimizers: {e}")
        raise

def save_checkpoint(models, optimizers, epoch, args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    save_dict = {
        'epoch': epoch,
        'generator': models['generator'].module.state_dict() if isinstance(models['generator'], nn.DataParallel) else models['generator'].state_dict(),
        'discriminator': models['discriminator'].module.state_dict() if isinstance(models['discriminator'], nn.DataParallel) else models['discriminator'].state_dict(),
        'appearance_flow': models['appearance_flow'].module.state_dict() if isinstance(models['appearance_flow'], nn.DataParallel) else models['appearance_flow'].state_dict(),
        'mask_generator': models['mask_generator'].module.state_dict() if isinstance(models['mask_generator'], nn.DataParallel) else models['mask_generator'].state_dict(),
        'optimizer_G': optimizers['optimizer_G'].state_dict(),
        'optimizer_D': optimizers['optimizer_D'].state_dict(),
        'scheduler_G': optimizers['scheduler_G'].state_dict(),
        'scheduler_D': optimizers['scheduler_D'].state_dict(),
    }
    
    save_path = os.path.join(args.checkpoint_dir, f"{args.name}_epoch_{epoch}.pth")
    torch.save(save_dict, save_path)
    print(f"Checkpoint saved to {save_path}")

def load_checkpoint(path, models, optimizers):
    if not os.path.exists(path):
        print(f"Checkpoint {path} does not exist, starting from scratch")
        return 0
    
    print(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location='cpu')
    
    try:
        models['generator'].load_state_dict(checkpoint['generator'])
        models['discriminator'].load_state_dict(checkpoint['discriminator'])
        models['appearance_flow'].load_state_dict(checkpoint['appearance_flow'])
        models['mask_generator'].load_state_dict(checkpoint['mask_generator'])
        
        optimizers['optimizer_G'].load_state_dict(checkpoint['optimizer_G'])
        optimizers['optimizer_D'].load_state_dict(checkpoint['optimizer_D'])
        optimizers['scheduler_G'].load_state_dict(checkpoint['scheduler_G'])
        optimizers['scheduler_D'].load_state_dict(checkpoint['scheduler_D'])
        
        return checkpoint['epoch'] + 1
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise

def compute_metrics(generated, target):
    generated = generated.permute(0, 2, 3, 1).cpu().numpy()
    target = target.permute(0, 2, 3, 1).cpu().numpy()
    
    ssim_scores = []
    psnr_scores = []
    
    for g, t in zip(generated, target):
        g = (g * 0.5 + 0.5).clip(0, 1)  # Denormalize
        t = (t * 0.5 + 0.5).clip(0, 1)
        
        ssim_score = ssim(g, t, channel_axis=2, data_range=1.0)
        psnr_score = psnr(g, t, data_range=1.0)
        
        ssim_scores.append(ssim_score)
        psnr_scores.append(psnr_score)
    
    return np.mean(ssim_scores), np.mean(psnr_scores)

def train_epoch(models, optimizers, dataloader, epoch, args, phase='train'):
    is_train = phase == 'train'
    
    if is_train:
        models['generator'].train()
        models['discriminator'].train()
        models['appearance_flow'].train()
        models['mask_generator'].train()
    else:
        models['generator'].eval()
        models['discriminator'].eval()
        models['appearance_flow'].eval()
        models['mask_generator'].eval()
    
    device = models['device']
    running_losses = {
        'g_total': 0.0,
        'd_total': 0.0,
        'l1': 0.0,
        'vgg': 0.0,
        'gan': 0.0,
        'mask': 0.0
    }
    running_metrics = {'ssim': 0.0, 'psnr': 0.0}
    
    pbar = tqdm(dataloader, desc=f"{phase.capitalize()} Epoch {epoch}")
    for i, data in enumerate(pbar):
        try:
            person_image = data['person_image'].to(device)
            cloth_image = data['cloth_image'].to(device)
            person_parse = data['person_parse'].to(device)
            cloth_mask = data['cloth_mask'].to(device)
            target_image = data['target_image'].to(device)
            
            batch_size = person_image.size(0)
            
            if is_train:
                # Update Discriminator
                optimizers['optimizer_D'].zero_grad()
                
                flow_output = models['appearance_flow'](person_image, cloth_image, person_parse)
                warped_cloth = flow_output['warped_cloth']
                
                mask_output = models['mask_generator'](warped_cloth, person_parse)
                generated_mask = mask_output['mask']
                
                person_representation = torch.cat([person_image, person_parse, warped_cloth, generated_mask], dim=1)
                generated_image = models['generator'](person_representation)
                
                real_pair = torch.cat([target_image, person_image], dim=1)
                pred_real = models['discriminator'](real_pair)
                
                fake_pair = torch.cat([generated_image.detach(), person_image], dim=1)
                pred_fake = models['discriminator'](fake_pair)
                
                d_loss_real = -torch.mean(pred_real)
                d_loss_fake = torch.mean(pred_fake)
                d_loss = d_loss_real + d_loss_fake
                
                d_loss.backward()
                optimizers['optimizer_D'].step()
                
                # Update Generator
                optimizers['optimizer_G'].zero_grad()
                
                fake_pair = torch.cat([generated_image, person_image], dim=1)
                pred_fake = models['discriminator'](fake_pair)
                g_gan_loss = -torch.mean(pred_fake) * args.lambda_gan
                
                g_l1_loss = F.l1_loss(generated_image, target_image) * args.lambda_l1
                g_vgg_loss = models['vgg_loss'](generated_image, target_image) * args.lambda_vgg
                g_mask_loss = F.binary_cross_entropy_with_logits(generated_mask, cloth_mask) * args.lambda_mask
                
                g_total_loss = g_gan_loss + g_l1_loss + g_vgg_loss + g_mask_loss
                
                g_total_loss.backward()
                optimizers['optimizer_G'].step()
                
                running_losses['g_total'] += g_total_loss.item()
                running_losses['d_total'] += d_loss.item()
                running_losses['l1'] += g_l1_loss.item()
                running_losses['vgg'] += g_vgg_loss.item()
                running_losses['gan'] += g_gan_loss.item()
                running_losses['mask'] += g_mask_loss.item()
            else:
                with torch.no_grad():
                    flow_output = models['appearance_flow'](person_image, cloth_image, person_parse)
                    warped_cloth = flow_output['warped_cloth']
                    
                    mask_output = models['mask_generator'](warped_cloth, person_parse)
                    generated_mask = mask_output['mask']
                    
                    person_representation = torch.cat([person_image, person_parse, warped_cloth, generated_mask], dim=1)
                    generated_image = models['generator'](person_representation)
                
                ssim_score, psnr_score = compute_metrics(generated_image, target_image)
                running_metrics['ssim'] += ssim_score
                running_metrics['psnr'] += psnr_score
            
            if i % args.log_freq == 0:
                log_info = {
                    'epoch': epoch,
                    'iter': i,
                    **{k: v / (i + 1) for k, v in running_losses.items()},
                    **{k: v / (i + 1) for k, v in running_metrics.items()}
                }
                
                pbar.set_postfix(**{k: f"{v:.4f}" for k, v in log_info.items() if k not in ['epoch', 'iter']})
                
                if args.log_wandb and is_train:
                    wandb.log(log_info)
                    if i % (args.log_freq * 5) == 0:
                        idx = min(3, batch_size-1)
                        wandb.log({
                            "person_image": wandb.Image(person_image[idx].cpu()),
                            "cloth_image": wandb.Image(cloth_image[idx].cpu()),
                            "warped_cloth": wandb.Image(warped_cloth[idx].cpu()),
                            "generated_mask": wandb.Image(generated_mask[idx].cpu()),
                            "generated_image": wandb.Image(generated_image[idx].cpu()),
                            "target_image": wandb.Image(target_image[idx].cpu())
                        })
        
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            continue
    
    if is_train:
        optimizers['scheduler_G'].step()
        optimizers['scheduler_D'].step()
    
    num_batches = len(dataloader)
    for k in running_losses:
        running_losses[k] /= num_batches
    for k in running_metrics:
        running_metrics[k] /= num_batches
    
    return running_losses, running_metrics

def main():
    args = parse_args()
    
    set_seed(args.seed)
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    if args.log_wandb:
        try:
            wandb.init(project=f"viton-hd-{args.name}", config=vars(args))
        except Exception as e:
            print(f"Failed to initialize wandb: {e}. Continuing without wandb.")
            args.log_wandb = False
    
    # Create datasets
    train_dataset, train_dataloader = create_dataset(args, phase='train')
    test_dataset, test_dataloader = create_dataset(args, phase='test')
    print(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")
    
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
        
        # Train
        train_losses, _ = train_epoch(models, optimizers, train_dataloader, epoch, args, phase='train')
        
        # Validate
        test_losses, test_metrics = train_epoch(models, optimizers, test_dataloader, epoch, args, phase='test')
        
        # Log epoch results
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch} completed in {epoch_time:.2f}s")
        print(f"Train - Generator loss: {train_losses['g_total']:.4f}, Discriminator loss: {train_losses['d_total']:.4f}")
        print(f"Train - L1: {train_losses['l1']:.4f}, VGG: {train_losses['vgg']:.4f}, GAN: {train_losses['gan']:.4f}, Mask: {train_losses['mask']:.4f}")
        print(f"Test - SSIM: {test_metrics['ssim']:.4f}, PSNR: {test_metrics['psnr']:.4f}")
        
        if args.log_wandb:
            wandb.log({
                'epoch': epoch,
                'train_g_total': train_losses['g_total'],
                'train_d_total': train_losses['d_total'],
                'train_l1': train_losses['l1'],
                'train_vgg': train_losses['vgg'],
                'train_gan': train_losses['gan'],
                'train_mask': train_losses['mask'],
                'test_ssim': test_metrics['ssim'],
                'test_psnr': test_metrics['psnr'],
                'learning_rate_G': optimizers['scheduler_G'].get_last_lr()[0],
                'learning_rate_D': optimizers['scheduler_D'].get_last_lr()[0]
            })
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            save_checkpoint(models, optimizers, epoch, args)
    
    print("Training completed!")

if __name__ == "__main__":
    main()