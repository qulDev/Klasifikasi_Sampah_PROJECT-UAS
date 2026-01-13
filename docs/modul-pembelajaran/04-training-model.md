# 📖 Modul 04: Training Model YOLOv8

## Daftar Isi

1. [Pengantar Training Deep Learning](#1-pengantar-training-deep-learning)
2. [Konsep Transfer Learning](#2-konsep-transfer-learning)
3. [Analisis Script train.py](#3-analisis-script-trainpy)
4. [Parameter Training](#4-parameter-training)
5. [Monitoring Training](#5-monitoring-training)
6. [Early Stopping dan Checkpointing](#6-early-stopping-dan-checkpointing)
7. [Evaluasi Model](#7-evaluasi-model)
8. [Tips Optimasi](#8-tips-optimasi)
9. [Troubleshooting](#9-troubleshooting)
10. [Latihan](#10-latihan)

---

## 1. Pengantar Training Deep Learning

### 1.1 Apa Itu Training?

Training adalah proses di mana model neural network belajar dari data untuk melakukan task tertentu (dalam kasus ini: mendeteksi dan mengklasifikasi sampah).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PROSES TRAINING NEURAL NETWORK                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐                                                           │
│  │   Input     │                                                           │
│  │   Image     │                                                           │
│  └──────┬──────┘                                                           │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────┐               │
│  │                   NEURAL NETWORK                         │               │
│  │                                                          │               │
│  │  Weights (w₁, w₂, w₃, ... wₙ)                           │               │
│  │  Parameter yang akan di-learn                            │               │
│  │                                                          │               │
│  └──────────────────────────┬───────────────────────────────┘               │
│                             │                                               │
│                             ▼                                               │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────┐               │
│  │  Prediction │────▶│  Loss Func  │────▶│  Backpropagation│               │
│  │  (Output)   │     │  Compare    │     │  Update Weights │               │
│  └─────────────┘     │  with GT    │     └────────┬────────┘               │
│                      └─────────────┘              │                        │
│                                                   │                        │
│                      ┌────────────────────────────┘                        │
│                      │                                                      │
│                      ▼                                                      │
│              ┌───────────────┐                                             │
│              │ Update:       │                                             │
│              │ w_new = w_old │                                             │
│              │   - lr * ∇L   │                                             │
│              └───────────────┘                                             │
│                                                                             │
│  Repeat for many iterations (epochs) until:                                │
│  - Loss converges (tidak turun lagi)                                       │
│  - Validation performance optimal                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Komponen Utama Training

| Komponen          | Fungsi                       | Contoh dalam Project |
| ----------------- | ---------------------------- | -------------------- |
| **Model**         | Arsitektur neural network    | YOLOv8s              |
| **Dataset**       | Data untuk learning          | 10 kelas sampah      |
| **Loss Function** | Mengukur error prediksi      | Box loss, cls loss   |
| **Optimizer**     | Algoritma update weights     | AdamW                |
| **Learning Rate** | Kecepatan learning           | 0.01 (default)       |
| **Epochs**        | Berapa kali lihat semua data | 120                  |
| **Batch Size**    | Gambar per iterasi           | 20                   |

### 1.3 Training Loop

```python
# Pseudocode training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        predictions = model(batch.images)

        # Compute loss
        loss = loss_function(predictions, batch.labels)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()
        optimizer.zero_grad()

    # Validation setiap epoch
    val_metrics = validate(model, val_dataloader)

    # Save checkpoint jika improvement
    if val_metrics.mAP > best_mAP:
        save_model(model, 'best.pt')
        best_mAP = val_metrics.mAP
```

---

## 2. Konsep Transfer Learning

### 2.1 Apa Itu Transfer Learning?

Transfer Learning adalah teknik menggunakan model yang sudah dilatih pada dataset besar (seperti COCO dengan 80 kelas, jutaan gambar) sebagai starting point untuk task baru.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TRANSFER LEARNING                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TRAINING FROM SCRATCH (Tidak Recommended)                                  │
│  ─────────────────────────────────────────                                  │
│                                                                             │
│  ┌──────────────┐                   ┌──────────────┐                       │
│  │   Random     │     Training      │   Trained    │                       │
│  │   Weights    │ ═══════════════▶  │   Model      │                       │
│  │   (noise)    │   Lama! Banyak    │              │                       │
│  └──────────────┘   data needed     └──────────────┘                       │
│                                                                             │
│  Masalah:                                                                   │
│  - Butuh jutaan gambar                                                     │
│  - Training sangat lama (minggu-bulan)                                     │
│  - Risk overfitting tinggi                                                 │
│                                                                             │
│  ════════════════════════════════════════════════════════════════════════  │
│                                                                             │
│  TRANSFER LEARNING (Recommended) ★                                         │
│  ────────────────────────────────                                          │
│                                                                             │
│  ┌──────────────┐                   ┌──────────────┐                       │
│  │  Pretrained  │    Fine-tuning    │    Our       │                       │
│  │    Model     │ ═══════════════▶  │   Model      │                       │
│  │ (COCO 80cls) │   Cepat! Less     │ (Sampah 10)  │                       │
│  └──────────────┘   data needed     └──────────────┘                       │
│                                                                             │
│  Keuntungan:                                                               │
│  - Sudah belajar fitur visual dasar (edges, textures, shapes)             │
│  - Training lebih cepat (jam-hari)                                        │
│  - Performa lebih baik dengan data terbatas                               │
│  - Risk overfitting lebih rendah                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Pretrained Weights dalam Project

```python
# Dalam train.py
from ultralytics import YOLO

# Load pretrained model (COCO weights)
model = YOLO('yolov8s.pt')  # Download pretrained weights

# Weights sudah terlatih pada COCO dataset:
# - 80 kelas object detection
# - 330K+ training images
# - State-of-the-art features
```

### 2.3 Fine-tuning Process

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FINE-TUNING PROCESS                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                    PRETRAINED MODEL (COCO)                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  BACKBONE (Feature Extractor)                                       │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ Conv layers yang belajar mengenali:                         │   │   │
│  │  │ - Low-level: edges, colors, textures                        │   │   │
│  │  │ - Mid-level: shapes, patterns                               │   │   │
│  │  │ - High-level: object parts                                  │   │   │
│  │  │                                                             │   │   │
│  │  │ ★ Fitur ini UNIVERSAL - berguna untuk semua task!          │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                     │   │
│  │  HEAD (Task-specific)                                               │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ Detection head: bbox prediction                             │   │   │
│  │  │ Classification head: 80 COCO classes                        │   │   │
│  │  │                                                             │   │   │
│  │  │ ★ Ini perlu di-adapt ke 10 kelas sampah                    │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│                              │ Fine-tune                                   │
│                              ▼                                              │
│                                                                             │
│                      FINE-TUNED MODEL (Sampah)                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  BACKBONE: Sedikit update (learning rate kecil)                    │   │
│  │  HEAD: Major update (learning rate normal)                         │   │
│  │        → 10 classes: battery, biological, cardboard, ...           │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Analisis Script train.py

### 3.1 Import dan Setup

```python
#!/usr/bin/env python3
"""
YOLOv8 Training Script for Klasifikasi Sampah

Train object detection model using Ultralytics YOLOv8 framework.

Usage:
    python train.py --model yolov8n --epochs 100 --imgsz 640 --batch 16 --device 0
    python train.py --model yolov8s --epochs 50 --dry-run
    python train.py --model yolov8m --epochs 200 --patience 50 --resume
"""

import argparse                    # Parse command line arguments
from datetime import datetime      # Timestamps
from pathlib import Path          # Path handling

import torch                      # PyTorch for device detection
from ultralytics import YOLO      # YOLOv8 API

from utils.logger import setup_logger  # Custom logging

logger = setup_logger(__name__)
```

### 3.2 Device Detection

```python
def get_device(device_arg: str = 'auto') -> str:
    """
    Detect and return the best available device.

    Priority:
    1. GPU (CUDA) - jika tersedia
    2. CPU - fallback

    Args:
        device_arg: 'auto', 'cpu', 'cuda', '0', '1', etc.

    Returns:
        Device string untuk YOLO training
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
            logger.warning(f"CUDA not available, falling back to CPU")
            device = 'cpu'

    return device
```

### 3.3 Argument Parser

```python
def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 for waste classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Simple Usage:
  python train.py                    # Train with all defaults

Advanced Usage:
  python train.py --epochs 50        # Quick training
  python train.py --model yolov8m    # Bigger model

Model Sizes (accuracy vs speed):
  yolov8n: Fastest, least accurate (3.2M params)
  yolov8s: Balanced - RECOMMENDED (11.2M params)
  yolov8m: More accurate (25.9M params)
  yolov8l: High accuracy (43.7M params)
  yolov8x: Best accuracy, slowest (68.2M params)
        """
    )

    # Model selection
    parser.add_argument('--model', type=str, default='yolov8s',
                        choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
                        help='Model size (default: yolov8s - balanced)')

    # Dataset config
    parser.add_argument('--data', type=str, default='./data.yaml',
                        help='Data config file (default: ./data.yaml)')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=120,
                        help='Training epochs (default: 120)')
    parser.add_argument('--batch', type=int, default=20,
                        help='Batch size (default: 20)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size (default: 640)')

    # Device
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto/cpu/cuda (default: auto)')

    # Early stopping
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (default: 10)')

    # Resume training
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')

    # Test mode
    parser.add_argument('--dry-run', action='store_true',
                        help='Test run (1 epoch only)')

    args = parser.parse_args()
```

### 3.4 Model Initialization

```python
    # Validate image size (harus multiple of 32)
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
        logger.info("DRY RUN MODE - Running 1 epoch smoke test")
        args.epochs = 1
        args.batch = 8  # Smaller batch

    # Initialize model
    logger.info(f"Initializing model: {args.model}")

    # Default settings
    project = './runs/detect'
    name = 'train'
    pretrained = True

    if args.resume:
        # Resume dari checkpoint
        last_checkpoint = Path(project) / name / 'weights' / 'last.pt'
        if last_checkpoint.exists():
            logger.info(f"Resuming from checkpoint: {last_checkpoint}")
            model = YOLO(str(last_checkpoint))
        else:
            logger.warning("No checkpoint found, starting from pretrained")
            model = YOLO(f'{args.model}.pt')
    else:
        # Load pretrained model (COCO weights)
        model = YOLO(f'{args.model}.pt')
        logger.info(f"Loaded pretrained {args.model} weights")
```

### 3.5 Training Execution

```python
    # Log configuration
    logger.info("")
    logger.info("Training Configuration:")
    logger.info(f"  Model:       {args.model}")
    logger.info(f"  Data:        {args.data}")
    logger.info(f"  Epochs:      {args.epochs}")
    logger.info(f"  Image size:  {args.imgsz}")
    logger.info(f"  Batch size:  {args.batch}")
    logger.info(f"  Device:      {device}")
    logger.info(f"  Patience:    {args.patience}")

    # Start training
    try:
        logger.info("Starting training...")

        model.train(
            data=str(data_yaml),       # Path ke data.yaml
            epochs=args.epochs,         # Jumlah epoch
            imgsz=args.imgsz,          # Input image size
            batch=args.batch,          # Batch size
            device=device,             # GPU/CPU
            project=project,           # Output directory
            name=name,                 # Run name
            patience=args.patience,    # Early stopping
            save=True,                 # Save checkpoints
            save_period=10,            # Save setiap 10 epochs
            exist_ok=True,             # Allow overwrite
            pretrained=pretrained,     # Use pretrained weights
            resume=args.resume,        # Resume training
            optimizer='AdamW',         # Optimizer
            verbose=True,              # Print progress
            seed=42,                   # Reproducibility
            deterministic=True,        # Reproducible results
            plots=True,                # Generate plots
        )

        logger.info("Training complete!")
```

### 3.6 Post-Training: Save Best Model

```python
        # Save best model to ./models/
        models_dir = Path('./models')
        models_dir.mkdir(exist_ok=True)

        # Copy best model
        best_source = Path(project) / name / 'weights' / 'best.pt'

        if best_source.exists():
            best_dest = models_dir / 'best_model.pt'

            # Backup model lama jika ada
            if best_dest.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_dest = models_dir / f'best_model_backup_{timestamp}.pt'
                shutil.copy2(best_dest, backup_dest)
                logger.info(f"Previous model backed up to: {backup_dest}")

            # Copy model baru
            shutil.copy2(best_source, best_dest)
            logger.info(f"Best model saved to: {best_dest}")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)

        # Handle CUDA out of memory
        if "CUDA out of memory" in str(e):
            logger.error("SUGGESTION: Reduce batch size with --batch <smaller>")
            logger.error(f"Current batch size: {args.batch}")
            logger.error("Try: --batch 8 or --batch 4")

        return 1

    return 0


if __name__ == '__main__':
    exit(main())
```

---

## 4. Parameter Training

### 4.1 Parameter Penting

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PARAMETER TRAINING                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        EPOCHS                                        │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                      │   │
│  │  Definisi: Berapa kali model melihat SELURUH training data          │   │
│  │                                                                      │   │
│  │  1 epoch = 1x pass through all training images                      │   │
│  │                                                                      │   │
│  │  Dalam project: 120 epochs                                          │   │
│  │  Jika punya 12,000 images, batch=20:                               │   │
│  │  → 600 iterations per epoch                                         │   │
│  │  → 72,000 total iterations                                         │   │
│  │                                                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ Epochs  │ Kemungkinan Hasil                                 │   │   │
│  │  ├─────────┼───────────────────────────────────────────────────┤   │   │
│  │  │ 10-20   │ Underfitting, model belum belajar optimal        │   │   │
│  │  │ 50-100  │ Reasonable untuk dataset sedang                  │   │   │
│  │  │ 100-200 │ Optimal untuk most cases (dengan early stopping) │   │   │
│  │  │ 200+    │ Risk overfitting jika tanpa early stopping       │   │   │
│  │  └─────────┴───────────────────────────────────────────────────┘   │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       BATCH SIZE                                     │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                      │   │
│  │  Definisi: Berapa gambar diproses bersamaan sebelum update weights │   │
│  │                                                                      │   │
│  │  Trade-offs:                                                        │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ Batch Size │ Memory  │ Training Speed │ Gradient Quality   │   │   │
│  │  ├────────────┼─────────┼────────────────┼────────────────────┤   │   │
│  │  │ Kecil (4)  │ Low     │ Slow           │ Noisy (variance)   │   │   │
│  │  │ Sedang (16)│ Medium  │ Balanced       │ Good               │   │   │
│  │  │ Besar (32) │ High    │ Fast           │ Stable             │   │   │
│  │  │ Besar (64+)│ V.High  │ V.Fast         │ V.Stable           │   │   │
│  │  └────────────┴─────────┴────────────────┴────────────────────┘   │   │
│  │                                                                      │   │
│  │  Dalam project: batch=20 (balance memory & speed)                   │   │
│  │                                                                      │   │
│  │  ⚠️ Jika CUDA out of memory: kurangi batch size                    │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       IMAGE SIZE                                     │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                      │   │
│  │  Definisi: Ukuran input gambar setelah resize                       │   │
│  │                                                                      │   │
│  │  Harus multiple of 32 (architectural requirement)                   │   │
│  │                                                                      │   │
│  │  Common sizes: 320, 416, 640, 800, 1280                            │   │
│  │                                                                      │   │
│  │  Trade-offs:                                                        │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ Size  │ Detail   │ Memory │ Speed  │ Best For               │   │   │
│  │  ├───────┼──────────┼────────┼────────┼────────────────────────┤   │   │
│  │  │ 320   │ Low      │ Low    │ Fast   │ Real-time, edge device │   │   │
│  │  │ 640   │ Good     │ Medium │ Medium │ General purpose ★      │   │   │
│  │  │ 1280  │ High     │ High   │ Slow   │ Small object detection │   │   │
│  │  └───────┴──────────┴────────┴────────┴────────────────────────┘   │   │
│  │                                                                      │   │
│  │  Dalam project: imgsz=640 (standard, balanced)                      │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Learning Rate dan Optimizer

```python
# YOLOv8 default training parameters (dalam Ultralytics)

# Optimizer
optimizer = 'AdamW'      # Adam dengan weight decay
lr0 = 0.01              # Initial learning rate
lrf = 0.01              # Final learning rate (lr0 * lrf)
momentum = 0.937        # SGD momentum / Adam beta1
weight_decay = 0.0005   # Weight decay (L2 regularization)

# Learning Rate Schedule
# YOLOv8 menggunakan cosine annealing:
#
#  lr
#  │ ████
#  │     ████
#  │         ████
#  │             ████
#  │                 ████████████████
#  └─────────────────────────────────▶ epoch
```

### 4.3 Data Augmentation (Built-in)

YOLOv8 secara otomatis menerapkan augmentation:

```python
# Default augmentations dalam YOLOv8
augmentation = {
    'hsv_h': 0.015,      # Hue augmentation
    'hsv_s': 0.7,        # Saturation augmentation
    'hsv_v': 0.4,        # Value augmentation
    'degrees': 0.0,      # Rotation
    'translate': 0.1,    # Translation
    'scale': 0.5,        # Scale
    'shear': 0.0,        # Shear
    'perspective': 0.0,  # Perspective
    'flipud': 0.0,       # Flip up-down
    'fliplr': 0.5,       # Flip left-right
    'mosaic': 1.0,       # Mosaic augmentation
    'mixup': 0.0,        # MixUp augmentation
    'copy_paste': 0.0,   # Copy-paste augmentation
}
```

---

## 5. Monitoring Training

### 5.1 Metrics yang Dimonitor

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TRAINING METRICS                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LOSS METRICS (Lower is Better)                                            │
│  ──────────────────────────────                                            │
│                                                                             │
│  box_loss:   Bounding box regression loss                                  │
│              → Mengukur akurasi prediksi koordinat bbox                    │
│              → Ideal: < 0.05                                               │
│                                                                             │
│  cls_loss:   Classification loss                                           │
│              → Mengukur akurasi prediksi kelas                             │
│              → Ideal: < 0.5                                                │
│                                                                             │
│  dfl_loss:   Distribution Focal Loss                                       │
│              → Loss untuk prediksi distribusi bbox                         │
│              → Ideal: < 1.0                                                │
│                                                                             │
│  PERFORMANCE METRICS (Higher is Better)                                    │
│  ─────────────────────────────────────                                     │
│                                                                             │
│  precision:  TP / (TP + FP)                                                │
│              → Dari yang terdeteksi, berapa yang benar?                    │
│              → Ideal: > 0.8                                                │
│                                                                             │
│  recall:     TP / (TP + FN)                                                │
│              → Dari yang seharusnya, berapa yang terdeteksi?               │
│              → Ideal: > 0.8                                                │
│                                                                             │
│  mAP50:      Mean Average Precision @ IoU=0.5                              │
│              → Primary metric untuk object detection                       │
│              → Ideal: > 0.8                                                │
│                                                                             │
│  mAP50-95:   Mean AP dari IoU 0.5 sampai 0.95                             │
│              → Metric yang lebih strict                                    │
│              → Ideal: > 0.5                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Output Training

Selama training, YOLOv8 akan menampilkan:

```
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/120      4.82G      1.845      3.475      1.453         45        640
       2/120      4.82G      1.634      2.856      1.358         67        640
       3/120      4.82G      1.521      2.345      1.298         52        640
       ...
     118/120      4.82G     0.0456     0.4532     0.8934         48        640
     119/120      4.82G     0.0445     0.4498     0.8912         55        640
     120/120      4.82G     0.0441     0.4476     0.8901         61        640

      Class     Images  Instances          P          R      mAP50   mAP50-95
        all       1473      1473      0.912      0.895      0.923      0.678
    battery        147        147      0.945      0.932      0.956      0.712
 biological        183        183      0.889      0.876      0.902      0.654
  cardboard        152        152      0.923      0.908      0.935      0.689
    clothes        142        142      0.867      0.852      0.884      0.623
      glass        138        138      0.934      0.921      0.948      0.701
      metal        143        143      0.918      0.904      0.931      0.687
      paper        152        152      0.901      0.889      0.912      0.668
    plastic        168        168      0.925      0.912      0.938      0.695
      shoes        119        119      0.878      0.863      0.893      0.641
      trash        129        129      0.856      0.841      0.871      0.608
```

### 5.3 Output Files

```
runs/detect/train/
├── weights/
│   ├── best.pt          # Model dengan mAP terbaik
│   └── last.pt          # Model dari epoch terakhir
│
├── args.yaml            # Training arguments
├── results.csv          # Metrics per epoch
│
├── results.png          # Plot semua metrics
├── confusion_matrix.png # Confusion matrix
├── confusion_matrix_normalized.png
│
├── F1_curve.png         # F1 vs confidence threshold
├── P_curve.png          # Precision vs confidence
├── R_curve.png          # Recall vs confidence
├── PR_curve.png         # Precision-Recall curve
│
├── labels.jpg           # Distribution of labels
├── labels_correlogram.jpg
│
└── train_batch0.jpg     # Sample training batches
    train_batch1.jpg
    train_batch2.jpg
```

---

## 6. Early Stopping dan Checkpointing

### 6.1 Early Stopping

Early stopping menghentikan training jika tidak ada improvement:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      EARLY STOPPING                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  patience = 10 (epochs tanpa improvement)                                  │
│                                                                             │
│  mAP                                                                       │
│   │                                                                        │
│   │                    ┌─── Best: mAP = 0.92                              │
│   │                    ▼                                                   │
│  0.9 ─┤               ████                                                 │
│       │           ████    █████████████ ← Plateau                         │
│  0.8 ─┤       ████                      │                                  │
│       │   ████                          │                                  │
│  0.7 ─┤████                             │                                  │
│       │                                 │                                  │
│  0.6 ─┤                                 │                                  │
│       │                                 │                                  │
│       └────────────────────────────────────────────▶ Epoch                 │
│       0    20   40   60   80   100  120                                   │
│                             │                                              │
│                             └─ Stop! 10 epochs tanpa improvement           │
│                                                                             │
│  Tanpa Early Stopping:                                                     │
│  - Training tetap lanjut sampai epoch 120                                  │
│  - Membuang waktu dan resources                                            │
│  - Risk overfitting                                                        │
│                                                                             │
│  Dengan Early Stopping (patience=10):                                      │
│  - Stop di epoch ~100                                                      │
│  - Hemat waktu 20+ epochs                                                  │
│  - Mencegah overfitting                                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Checkpointing

```python
# Dalam train.py
model.train(
    ...
    save=True,           # Enable checkpoint saving
    save_period=10,      # Save setiap 10 epochs
    exist_ok=True,       # Allow overwriting
)

# Files yang disimpan:
# - best.pt: Model dengan validation mAP tertinggi
# - last.pt: Model dari epoch terakhir
# - epoch10.pt, epoch20.pt, ... (jika save_period > 0)
```

### 6.3 Resume Training

```bash
# Jika training interrupted, resume dari checkpoint:
python train.py --resume

# Ini akan:
# 1. Load last.pt
# 2. Continue dari epoch terakhir
# 3. Maintain optimizer state
# 4. Maintain learning rate schedule
```

---

## 7. Evaluasi Model

### 7.1 Metrics Evaluation

```python
# Setelah training, evaluate pada test set:
from ultralytics import YOLO

model = YOLO('./models/best_model.pt')

# Validation metrics
metrics = model.val(data='./data.yaml', split='test')

print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.p.mean():.4f}")
print(f"Recall: {metrics.box.r.mean():.4f}")
```

### 7.2 Confusion Matrix

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CONFUSION MATRIX                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                          PREDICTED CLASS                                    │
│            ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐   │
│            │ bat │ bio │ car │ clo │ gla │ met │ pap │ pla │ sho │ tra │   │
│  ┌─────────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤   │
│ A│ battery │ 142 │  0  │  0  │  0  │  1  │  3  │  0  │  0  │  0  │  1  │   │
│ C│ biologi │  0  │ 176 │  1  │  0  │  0  │  0  │  2  │  0  │  0  │  4  │   │
│ T│ cardboa │  0  │  0  │ 148 │  0  │  0  │  0  │  3  │  0  │  0  │  1  │   │
│ U│ clothes │  0  │  0  │  0  │ 135 │  0  │  0  │  0  │  2  │  4  │  1  │   │
│ A│ glass   │  1  │  0  │  0  │  0  │ 133 │  2  │  0  │  2  │  0  │  0  │   │
│ L│ metal   │  2  │  0  │  0  │  0  │  3  │ 136 │  0  │  1  │  0  │  1  │   │
│  │ paper   │  0  │  1  │  2  │  0  │  0  │  0  │ 145 │  1  │  0  │  3  │   │
│  │ plastic │  0  │  0  │  0  │  1  │  2  │  2  │  1  │ 160 │  0  │  2  │   │
│  │ shoes   │  0  │  0  │  0  │  3  │  0  │  0  │  0  │  0  │ 114 │  2  │   │
│  │ trash   │  1  │  3  │  1  │  2  │  0  │  1  │  2  │  2  │  1  │ 116 │   │
│  └─────────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘   │
│                                                                             │
│  Interpretasi:                                                             │
│  - Diagonal: True Positives (prediksi benar)                               │
│  - Off-diagonal: Confusion/Errors                                          │
│  - Contoh: 3 battery diprediksi sebagai metal                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Tips Optimasi

### 8.1 Memory Optimization

```bash
# Jika CUDA out of memory:

# Option 1: Kurangi batch size
python train.py --batch 8

# Option 2: Kurangi image size
python train.py --imgsz 416

# Option 3: Gunakan model lebih kecil
python train.py --model yolov8n

# Option 4: Mixed precision (otomatis di YOLOv8)
```

### 8.2 Speed Optimization

```bash
# Percepat training:

# Option 1: Gunakan GPU
python train.py --device 0

# Option 2: Gunakan batch size lebih besar (jika memory cukup)
python train.py --batch 32

# Option 3: Gunakan model lebih kecil
python train.py --model yolov8n

# Option 4: Kurangi epochs dengan early stopping
python train.py --epochs 200 --patience 20
```

### 8.3 Accuracy Optimization

```bash
# Tingkatkan akurasi:

# Option 1: Model lebih besar
python train.py --model yolov8l

# Option 2: Image size lebih besar
python train.py --imgsz 800

# Option 3: Training lebih lama
python train.py --epochs 200 --patience 50

# Option 4: Tambah data (data augmentation eksternal)
```

---

## 9. Troubleshooting

### 9.1 Masalah Umum

| Masalah                | Penyebab                      | Solusi                                   |
| ---------------------- | ----------------------------- | ---------------------------------------- |
| CUDA out of memory     | Batch size terlalu besar      | `--batch 8` atau lebih kecil             |
| Training sangat lambat | Menggunakan CPU               | Gunakan GPU dengan `--device 0`          |
| Loss tidak turun       | Learning rate terlalu tinggi  | Coba default, atau kurangi               |
| Overfitting            | Dataset kecil / training lama | Gunakan early stopping, augmentation     |
| mAP rendah             | Model/data tidak cocok        | Cek data quality, coba model lebih besar |

### 9.2 Debug Mode

```bash
# Dry run untuk test setup
python train.py --dry-run

# Ini akan:
# - Run hanya 1 epoch
# - Batch size kecil
# - Memastikan semua berjalan tanpa error
```

---

## 10. Latihan

### Latihan 1: Parameter Tuning

Jelaskan dampak dari perubahan parameter berikut:

1. `--epochs 50` vs `--epochs 200`
2. `--batch 8` vs `--batch 32`
3. `--model yolov8n` vs `--model yolov8x`

### Latihan 2: Training Execution

Jalankan training dengan konfigurasi berikut dan bandingkan hasilnya:

```bash
# Config A
python train.py --model yolov8s --epochs 50 --batch 16

# Config B
python train.py --model yolov8m --epochs 100 --batch 8
```

### Latihan 3: Analisis Hasil

Setelah training selesai:

1. Buka `runs/detect/train/results.csv`
2. Plot loss vs epoch
3. Identifikasi kapan model mulai converge
4. Analisis confusion matrix untuk kelas dengan performance rendah

---

**Selamat! Anda telah menyelesaikan Modul 04: Training Model YOLOv8**

_Lanjut ke: [Modul 05 - Web Application](./05-web-application.md)_
