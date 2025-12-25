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
        description="Train YOLOv8 for waste classification - Simple & Smart Defaults",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Simple Usage:
  python train.py                    # Train with all defaults (recommended!)
  
Advanced Usage:
  python train.py --epochs 50        # Quick training
  python train.py --model yolov8m    # Bigger model (more accurate, slower)
  python train.py --dry-run          # Test run (1 epoch)

Model Sizes (accuracy vs speed):
  yolov8n: Fastest, least accurate (3.2M params)
  yolov8s: Balanced - RECOMMENDED (11.2M params)  
  yolov8m: More accurate (25.9M params)
  yolov8l: High accuracy (43.7M params)
  yolov8x: Best accuracy, slowest (68.2M params)
  
Training Parameters Explained:
  --epochs:  How many times to see all images (default: 100)
             More = better learning (but slower)
             
  --batch:   How many images to process at once (default: 16)
             Larger = faster training (needs more GPU memory)
             If "CUDA out of memory", use --batch 8 or --batch 4
             
  --imgsz:   Image size for training (default: 640)
             Larger = more detail (but slower)
             Must be multiple of 32 (e.g., 320, 416, 640, 800)
        """
    )

    # Simple arguments with smart defaults
    parser.add_argument('--model', type=str, default='yolov8s',
                        choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
                        help='Model size (default: yolov8m - balanced)')
    parser.add_argument('--data', type=str, default='./data.yaml',
                        help='Data config file (default: ./data.yaml)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs (default: 100)')
    parser.add_argument('--batch', type=int, default=20,
                        help='Batch size (default: 20)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size (default: 640)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto/cpu/cuda (default: auto)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience - stop if no improvement (default: 10)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    parser.add_argument('--dry-run', action='store_true',
                        help='Test run (1 epoch only)')

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
    
    # Default values for removed args
    project = './runs/detect'
    name = 'train'
    pretrained = True
    
    
    if args.resume:
        # Try to find last checkpoint
        last_checkpoint = Path(project) / name / 'weights' / 'last.pt'
        if last_checkpoint.exists():
            logger.info(f"Resuming from checkpoint: {last_checkpoint}")
            model = YOLO(str(last_checkpoint))
        else:
            logger.warning(f"No checkpoint found at {last_checkpoint}, starting from pretrained")
            model = YOLO(f'{args.model}.pt' if pretrained else f'{args.model}.yaml')
    else:
        if pretrained:
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
    logger.info(f"  Project:     {project}")
    logger.info(f"  Name:        {name}")
    logger.info(f"  Pretrained:  {pretrained}")
    logger.info(f"  Patience:    {args.patience}")
    logger.info("")

    # Train model
    try:
        logger.info("Starting training...")
        logger.info("=" * 70)

        model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            project=project,
            name=name,
            patience=args.patience,
            save=True,
            save_period=10,  # Save checkpoint every 10 epochs
            exist_ok=True,  # Allow overwriting existing project
            pretrained=pretrained,
            resume=args.resume,  # PENTING: Resume training dari checkpoint
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

        # Copy best model with simple name
        best_source = Path(project) / name / 'weights' / 'best.pt'
        
        if best_source.exists():
            best_dest = models_dir / 'best_model.pt'
            import shutil
            from datetime import datetime
            
            # Backup model lama jika ada (sebelum ditimpa)
            if best_dest.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_dest = models_dir / f'best_model_backup_{timestamp}.pt'
                shutil.copy2(best_dest, backup_dest)
                logger.info(f"Previous model backed up to: {backup_dest}")
            
            # Copy model baru
            shutil.copy2(best_source, best_dest)
            logger.info(f"Best model saved to: {best_dest}")

        # Log metrics
        logger.info("")
        logger.info("Training Metrics:")
        logger.info(f"  Results saved to: {Path(project) / name}")
        
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
 