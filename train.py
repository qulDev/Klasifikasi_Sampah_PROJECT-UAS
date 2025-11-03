#!/usr/bin/env python3
"""
YOLOv8 Training Script for Klasifikasi Sampah

Train object detection model using Ultralytics YOLOv8 framework.

Usage:
    python train.py --model yolov8n --epochs 100 --imgsz 640 --batch 16 --device 0
    python train.py --model yolov8s --epochs 50 --dry-run
    python train.py --model yolov8m --epochs 200 --patience 50 --resume
"""

import argparse
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO

from utils.logger import setup_logger

logger = setup_logger(__name__)


def get_device(device_arg: str = 'auto') -> str:
    """
    Detect and return the best available device.

    Args:
        device_arg: Device specification ('auto', 'cpu', 'cuda', '0', '1', etc.)

    Returns:
        Device string for YOLO training
    """
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = '0'  # Use first GPU
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"✓ Using GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            device = 'cpu'
            logger.warning("⚠ GPU not available, using CPU (training will be slower)")
    else:
        device = device_arg
        if device != 'cpu' and not torch.cuda.is_available():
            logger.warning(f"CUDA not available, falling back to CPU despite --device {device}")
            device = 'cpu'

    return device


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 object detection model for waste classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick smoke test (1 epoch)
  python train.py --model yolov8n --epochs 1 --dry-run

  # Train small model on GPU
  python train.py --model yolov8s --epochs 100 --imgsz 640 --batch 16 --device 0

  # Train medium model with early stopping
  python train.py --model yolov8m --epochs 300 --patience 50

  # Resume from checkpoint
  python train.py --model yolov8s --resume

  # Train on CPU
  python train.py --model yolov8n --epochs 50 --device cpu
        """
    )

    parser.add_argument('--model', type=str, default='yolov8s',
                        choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
                        help='YOLOv8 model variant (default: yolov8s)')
    parser.add_argument('--data', type=str, default='./data.yaml',
                        help='Path to data.yaml configuration file (default: ./data.yaml)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Input image size (must be multiple of 32, default: 640)')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size (default: 16, auto-adjusted for GPU memory)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use: auto, cpu, cuda, 0, 1, etc. (default: auto)')
    parser.add_argument('--project', type=str, default='./runs/detect',
                        help='Project directory to save results (default: ./runs/detect)')
    parser.add_argument('--name', type=str, default='train',
                        help='Experiment name (default: train)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained COCO weights (default: True)')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false',
                        help='Train from scratch without pretrained weights')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience (epochs without improvement, default: 50)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run 1 epoch smoke test and exit')

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Klasifikasi Sampah - YOLOv8 Training")
    logger.info("=" * 70)

    # Validate image size
    if args.imgsz % 32 != 0:
        logger.error(f"Image size must be multiple of 32, got {args.imgsz}")
        return 1

    # Check data.yaml exists
    data_yaml = Path(args.data)
    if not data_yaml.exists():
        logger.error(f"Data YAML not found: {data_yaml}")
        logger.error("Run split_and_prep.py first to generate data.yaml")
        return 1

    # Detect device
    device = get_device(args.device)

    # Dry run mode
    if args.dry_run:
        logger.info("=" * 70)
        logger.info("DRY RUN MODE - Running 1 epoch smoke test")
        logger.info("=" * 70)
        args.epochs = 1
        args.batch = 8  # Smaller batch for quick test

    # Initialize model
    logger.info(f"Initializing model: {args.model}")
    
    if args.resume:
        # Try to find last checkpoint
        last_checkpoint = Path(args.project) / args.name / 'weights' / 'last.pt'
        if last_checkpoint.exists():
            logger.info(f"Resuming from checkpoint: {last_checkpoint}")
            model = YOLO(str(last_checkpoint))
        else:
            logger.warning(f"No checkpoint found at {last_checkpoint}, starting from pretrained")
            model = YOLO(f'{args.model}.pt' if args.pretrained else f'{args.model}.yaml')
    else:
        if args.pretrained:
            model = YOLO(f'{args.model}.pt')  # Load pretrained COCO weights
            logger.info(f"Loaded pretrained {args.model} weights")
        else:
            model = YOLO(f'{args.model}.yaml')  # Load architecture only
            logger.info(f"Training from scratch with {args.model} architecture")

    # Training configuration
    logger.info("")
    logger.info("Training Configuration:")
    logger.info(f"  Model:       {args.model}")
    logger.info(f"  Data:        {args.data}")
    logger.info(f"  Epochs:      {args.epochs}")
    logger.info(f"  Image size:  {args.imgsz}")
    logger.info(f"  Batch size:  {args.batch}")
    logger.info(f"  Device:      {device}")
    logger.info(f"  Project:     {args.project}")
    logger.info(f"  Name:        {args.name}")
    logger.info(f"  Pretrained:  {args.pretrained}")
    logger.info(f"  Patience:    {args.patience}")
    logger.info("")

    # Train model
    try:
        logger.info("Starting training...")
        logger.info("=" * 70)

        results = model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            project=args.project,
            name=args.name,
            patience=args.patience,
            save=True,
            save_period=10,  # Save checkpoint every 10 epochs
            exist_ok=True,  # Allow overwriting existing project
            pretrained=args.pretrained,
            optimizer='AdamW',  # Use AdamW optimizer
            verbose=True,
            seed=42,  # Reproducibility
            deterministic=True,  # Reproducible results
            plots=True,  # Generate training plots
        )

        logger.info("=" * 70)
        logger.info("Training complete!")
        logger.info("=" * 70)

        # Save best model to ./models/
        models_dir = Path('./models')
        models_dir.mkdir(exist_ok=True)

        # Copy best model with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        best_source = Path(args.project) / args.name / 'weights' / 'best.pt'
        
        if best_source.exists():
            best_dest = models_dir / f'{timestamp}_best.pt'
            import shutil
            shutil.copy2(best_source, best_dest)
            logger.info(f"Best model saved to: {best_dest}")

            # Create symlink to latest best model
            best_symlink = models_dir / 'best.pt'
            if best_symlink.exists() or best_symlink.is_symlink():
                best_symlink.unlink()
            best_symlink.symlink_to(best_dest.name)
            logger.info(f"Symlink created: {best_symlink} -> {best_dest.name}")

        # Log metrics
        logger.info("")
        logger.info("Training Metrics:")
        logger.info(f"  Results saved to: {Path(args.project) / args.name}")
        
        if args.dry_run:
            logger.info("")
            logger.info("=" * 70)
            logger.info("DRY RUN COMPLETE - 1 epoch smoke test passed")
            logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        
        if "CUDA out of memory" in str(e):
            logger.error("")
            logger.error("SUGGESTION: Reduce batch size with --batch <smaller_value>")
            logger.error(f"Current batch size: {args.batch}")
            logger.error("Try: --batch 8 or --batch 4")
        
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
