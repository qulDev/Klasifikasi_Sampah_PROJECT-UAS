# 📚 Panduan Lengkap Training YOLOv8

## Daftar Isi
1. [Konsep Dasar](#konsep-dasar)
2. [Parameter Training](#parameter-training)
3. [Strategi Training](#strategi-training)
4. [Troubleshooting](#troubleshooting)
5. [Best Practices](#best-practices)

---

## Konsep Dasar

### Apa itu Training?

**Training** adalah proses dimana model neural network "belajar" mengenali pola dari data.

Analogi sederhana:
- **Dataset** = buku pelajaran dengan soal dan jawaban
- **Epochs** = berapa kali baca buku dari awal sampai akhir
- **Batch** = berapa soal dikerjakan sekaligus
- **Learning Rate** = seberapa cepat belajar (otomatis di YOLOv8)

### Proses Training Step-by-Step

```
1. Load Dataset
   ├── Baca gambar dan label
   ├── Split jadi batch-batch kecil
   └── Shuffle data

2. Forward Pass (untuk setiap batch)
   ├── Gambar masuk ke model
   ├── Model prediksi bounding box dan class
   └── Bandingkan prediksi vs ground truth

3. Compute Loss
   ├── Hitung seberapa salah prediksi
   ├── Box loss: seberapa akurat posisi box
   ├── Class loss: seberapa akurat klasifikasi
   └── Objectness loss: confidence score

4. Backward Pass
   ├── Hitung gradient (turunan loss)
   ├── Update weights model
   └── Model jadi sedikit lebih baik

5. Validation
   ├── Test model di validation set
   ├── Hitung metrics (mAP, precision, recall)
   └── Save best model

6. Repeat untuk N epochs
```

---

## Parameter Training

### 1. --epochs

**Definisi:** Berapa kali model melihat SELURUH dataset.

**Contoh:**
- Dataset punya 1000 gambar
- 1 epoch = model lihat semua 1000 gambar sekali
- 100 epochs = model lihat semua 1000 gambar, 100 kali

**Impact:**
```
Too Few Epochs (< 50)
├── ❌ Model belum belajar dengan baik
├── ❌ Underfitting (tidak mampu mengenali pola)
└── ❌ Akurasi rendah

Good Epochs (50-200)
├── ✅ Model belajar dengan baik
├── ✅ Balance antara underfitting dan overfitting
└── ✅ Akurasi optimal

Too Many Epochs (> 500)
├── ⚠️ Overfitting (terlalu hapal training data)
├── ⚠️ Tidak generalisasi ke data baru
└── ⚠️ Validation accuracy turun
```

**Recommendations:**
```python
--epochs 10      # Quick test / smoke test
--epochs 50      # Fast training untuk prototyping
--epochs 100     # Standard training (RECOMMENDED)
--epochs 200     # Fine-tuning atau dataset besar
--epochs 500+    # Research / optimal performance
```

**Real Example:**
```bash
# Test apakah setup benar (5 menit)
python train.py --epochs 10

# Training cepat untuk iterasi (1-2 jam)
python train.py --epochs 50

# Training normal (2-4 jam)
python train.py --epochs 100

# Training optimal (4-8 jam)
python train.py --epochs 200
```

**How to Know?**
- Lihat training curves (`results.png`)
- Jika validation loss masih turun → tambah epochs
- Jika validation loss naik, training loss turun → overfitting, kurangi epochs

---

### 2. --batch

**Definisi:** Berapa banyak gambar diproses SEKALIGUS dalam satu iterasi.

**Contoh:**
- Dataset: 1000 gambar
- Batch size: 16
- 1 epoch = 1000 / 16 = 62.5 ≈ 63 iterations

**Impact:**

```
Small Batch (4-8)
├── ➕ GPU memory kecil cukup (2-4GB)
├── ➕ Bisa jalan di laptop biasa
├── ➖ Training SANGAT lambat
├── ➖ Gradient noisy (unstable)
└── ➖ Mungkin stuck di local minima

Medium Batch (16-32) - RECOMMENDED
├── ✅ Balance speed vs memory
├── ✅ Stable training
├── ✅ Good convergence
└── ✅ GPU 6-12GB

Large Batch (64-128)
├── ➕ Training CEPAT
├── ➕ Gradient smooth (stable)
├── ➖ Perlu GPU besar (16-24GB)
└── ➖ Mungkin stuck di local minima (generalisasi kurang)
```

**GPU Memory Requirements:**
```
Batch Size | GPU Memory | Example GPU
-----------|------------|-------------
4          | 2-4 GB     | GTX 1050 Ti, MX150
8          | 4-6 GB     | GTX 1060, GTX 1660
16         | 6-8 GB     | RTX 2060, GTX 1660 Ti
32         | 10-12 GB   | RTX 3060, RTX 2080
64         | 16-20 GB   | RTX 3080, RTX 3090
128        | 24+ GB     | RTX 3090, A100
```

**Recommendations:**
```python
--batch 4        # Laptop dengan GPU kecil / CPU only
--batch 8        # GPU 4-6 GB (GTX 1060, 1660)
--batch 16       # GPU 6-8 GB (RTX 2060) - RECOMMENDED
--batch 32       # GPU 10-12 GB (RTX 3060)
--batch 64       # GPU 16+ GB (RTX 3080, 3090)
```

**Real Example:**
```bash
# Laptop dengan GTX 1660 (6GB)
python train.py --batch 8

# Desktop dengan RTX 3060 (12GB)
python train.py --batch 32

# Workstation dengan RTX 3090 (24GB)
python train.py --batch 64

# CPU only (no GPU)
python train.py --batch 4
```

**Error Handling:**
```
CUDA out of memory
├── Script akan auto-suggest batch size lebih kecil
├── Coba: --batch 8
└── Jika masih error, coba: --batch 4
```

---

### 3. --imgsz

**Definisi:** Ukuran gambar (width dan height) untuk training.

**Impact:**

```
Small Image (320)
├── ➕ Training CEPAT
├── ➕ GPU memory kecil cukup
├── ➖ Detail kurang (objek kecil susah terdeteksi)
└── ➖ Akurasi lebih rendah

Medium Image (640) - RECOMMENDED
├── ✅ Balance speed vs accuracy
├── ✅ Good detail
├── ✅ Moderate GPU usage
└── ✅ Standard untuk YOLOv8

Large Image (1280)
├── ➕ Detail tinggi (objek kecil terdeteksi)
├── ➕ Akurasi lebih tinggi
├── ➖ Training LAMBAT (4x lebih lambat dari 640)
└── ➖ Perlu GPU besar (memory 2-3x lebih banyak)
```

**Use Cases:**
```
320px:
├── Mobile deployment
├── Edge devices (Raspberry Pi, Jetson Nano)
├── Real-time application yang butuh FPS tinggi
└── Objek besar (mudah terdeteksi)

640px: (RECOMMENDED)
├── Standard computer vision
├── Balance speed vs accuracy
├── Most common use case
└── Good for 95% applications

1280px:
├── High-resolution images
├── Small object detection (koin, screws, dll)
├── Medical imaging
└── Research / competition
```

**Recommendations:**
```python
--imgsz 320      # Mobile / edge devices
--imgsz 640      # Standard (RECOMMENDED)
--imgsz 1280     # High accuracy / small objects
```

**Real Example:**
```bash
# Mobile app (fast inference)
python train.py --imgsz 320 --model yolov8n

# Standard web app
python train.py --imgsz 640 --model yolov8s

# Research / high accuracy
python train.py --imgsz 1280 --model yolov8m
```

**Note:** Image size harus multiple of 32 (constraint YOLO architecture).

---

### 4. --patience

**Definisi:** Berapa epochs tunggu sebelum early stopping jika tidak ada improvement.

**Contoh:**
```
Epoch 50: mAP = 0.75 (best)
Epoch 51: mAP = 0.74 (tidak improve)
Epoch 52: mAP = 0.73 (tidak improve)
...
Epoch 100: mAP = 0.72 (tidak improve)

Dengan --patience 50:
├── Training akan stop di epoch 100
└── Karena sudah 50 epochs tidak ada improvement dari epoch 50
```

**Impact:**

```
Low Patience (10-20)
├── ➕ Training stop cepat jika stuck
├── ➕ Save waktu dan resources
├── ➖ Mungkin stop terlalu cepat (belum optimal)
└── ➖ Tidak kasih kesempatan recovery

Medium Patience (30-50) - RECOMMENDED
├── ✅ Balance time vs performance
├── ✅ Kasih kesempatan model improve
└── ✅ Avoid wasted training time

High Patience (100+)
├── ➕ Model punya banyak kesempatan improve
├── ➖ Training bisa sangat lama tanpa hasil
└── ➖ Wasted computational resources
```

**Recommendations:**
```python
--patience 20    # Quick experiments
--patience 50    # Standard training (RECOMMENDED)
--patience 100   # Research / optimal performance
--patience 0     # Disable early stopping (train sampai epochs habis)
```

**Real Example:**
```bash
# Quick test (stop jika stuck 20 epochs)
python train.py --epochs 200 --patience 20

# Standard (stop jika stuck 50 epochs)
python train.py --epochs 200 --patience 50

# No early stopping
python train.py --epochs 100 --patience 0
```

---

### 5. --model

**Definisi:** Ukuran model YOLOv8 yang akan digunakan.

**Model Sizes:**

```
yolov8n (Nano)
├── Parameters: 3.2M
├── Size: 6 MB
├── Speed: Fastest (45+ FPS)
├── Accuracy: Good (mAP 37.3)
└── Use: Mobile, edge devices, real-time

yolov8s (Small) - RECOMMENDED
├── Parameters: 11.2M
├── Size: 22 MB
├── Speed: Fast (35+ FPS)
├── Accuracy: Better (mAP 44.9)
└── Use: General purpose, production

yolov8m (Medium)
├── Parameters: 25.9M
├── Size: 50 MB
├── Speed: Moderate (25+ FPS)
├── Accuracy: Great (mAP 50.2)
└── Use: Higher accuracy needed

yolov8l (Large)
├── Parameters: 43.7M
├── Size: 84 MB
├── Speed: Slow (20+ FPS)
├── Accuracy: Excellent (mAP 52.9)
└── Use: High accuracy, server-side

yolov8x (Extra Large)
├── Parameters: 68.2M
├── Size: 130 MB
├── Speed: Very Slow (15+ FPS)
├── Accuracy: Best (mAP 53.9)
└── Use: Research, competitions
```

**Trade-offs:**

```
Accuracy vs Speed
    yolov8x ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━● Best Accuracy
    yolov8l ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━●
    yolov8m ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━●
    yolov8s ●━━━━━━━━━━━━━━━━━●                                RECOMMENDED
    yolov8n ●━━━━━━━●                                          Fastest
```

**Recommendations:**
```python
--model yolov8n  # Mobile app, Raspberry Pi, Jetson Nano
--model yolov8s  # Web app, production (RECOMMENDED)
--model yolov8m  # Higher accuracy needed
--model yolov8l  # Server-side, high accuracy
--model yolov8x  # Research, competitions, best accuracy
```

**Real Example:**
```bash
# Mobile app dengan real-time detection
python train.py --model yolov8n --imgsz 320 --batch 32

# Production web app
python train.py --model yolov8s --imgsz 640 --batch 16

# High accuracy untuk research
python train.py --model yolov8l --imgsz 1280 --batch 8
```

---

### 6. --data

**Definisi:** Path ke file data.yaml yang berisi konfigurasi dataset.

**Format data.yaml:**
```yaml
path: /path/to/dataset  # Dataset root
train: images/train     # Training images (relative to 'path')
val: images/val         # Validation images (relative to 'path')
test: images/test       # Test images (optional)

names:
  0: cardboard
  1: glass
  2: metal
  3: paper
  4: plastic
  5: trash
```

**Default:** `./datasets/processed/split/data.yaml`

**Example:**
```bash
# Use default dataset
python train.py

# Custom dataset
python train.py --data custom_dataset/data.yaml

# Absolute path
python train.py --data /home/user/my_dataset/data.yaml
```

---

### 7. --resume

**Definisi:** Lanjutkan training dari checkpoint terakhir.

**Use Cases:**
- Training terputus (crash, mati listrik, dll)
- Ingin train lebih lama (tambah epochs)
- Fine-tuning dari checkpoint

**How it Works:**
```
1. Script cari file: ./runs/detect/train/weights/last.pt
2. Jika ada, load checkpoint (weights, optimizer state, epoch)
3. Continue training dari epoch terakhir
```

**Example:**
```bash
# Start fresh training
python train.py --epochs 100

# ... training berjalan sampai epoch 50, terus crash ...

# Resume training
python train.py --resume
# Will continue from epoch 50 → 100
```

**Note:** Checkpoint otomatis disave setiap 10 epochs.

---

### 8. --dry-run

**Definisi:** Test mode, hanya train 1 epoch untuk test setup.

**Use Cases:**
- Validate dataset format
- Test GPU/memory compatibility
- Check if all dependencies installed
- Quick sanity check

**Example:**
```bash
# Test before actual training
python train.py --dry-run

# Expected output:
# "DRY RUN COMPLETE - 1 epoch smoke test passed"
```

**Time:** ~1-5 menit (tergantung dataset size dan batch size).

---

## Strategi Training

### 1. Progressive Training

**Idea:** Train dari model kecil ke besar untuk save waktu.

```bash
# Step 1: Quick test dengan nano (30 min)
python train.py --model yolov8n --epochs 50 --dry-run

# Step 2: Medium training dengan small (2 hours)
python train.py --model yolov8s --epochs 100

# Step 3: Final training dengan medium (4-6 hours)
python train.py --model yolov8m --epochs 200
```

**Benefits:**
- Detect issues early (di model kecil)
- Progressive improvement
- Save computational resources

---

### 2. Batch Size Experimentation

**Find optimal batch size:**

```bash
# Start large, reduce jika error
python train.py --batch 64  # Try first
python train.py --batch 32  # If OOM error
python train.py --batch 16  # If still OOM
python train.py --batch 8   # Last resort
```

**Rule of Thumb:**
- Max batch size yang fit di GPU memory
- Tapi tidak lebih dari 64 (diminishing returns)

---

### 3. Two-Phase Training

**Phase 1: Quick Convergence**
```bash
# Fast training untuk convergence cepat
python train.py --epochs 100 --batch 32 --patience 30
```

**Phase 2: Fine-Tuning**
```bash
# Resume dengan patience lebih tinggi untuk optimal
python train.py --resume --epochs 200 --patience 100
```

---

### 4. Multi-Resolution Training

**Idea:** Train dengan resolution berbeda untuk robustness.

```bash
# Phase 1: Fast training di 320
python train.py --imgsz 320 --epochs 50

# Phase 2: Standard resolution
python train.py --imgsz 640 --epochs 100

# Phase 3: High resolution fine-tuning
python train.py --imgsz 1280 --epochs 50
```

---

## Troubleshooting

### Problem 1: CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions:**
```bash
# Solution 1: Reduce batch size
python train.py --batch 8  # Or --batch 4

# Solution 2: Reduce image size
python train.py --imgsz 320

# Solution 3: Use smaller model
python train.py --model yolov8n

# Solution 4: Kombinasi
python train.py --model yolov8n --batch 4 --imgsz 320
```

---

### Problem 2: Overfitting

**Symptoms:**
- Training loss turun terus
- Validation loss naik atau plateau
- Gap besar antara train dan val metrics

**Solutions:**
```bash
# Solution 1: Early stopping
python train.py --patience 30  # Stop cepat jika overfit

# Solution 2: More data augmentation
# (otomatis di YOLOv8, tapi bisa customize di config)

# Solution 3: Smaller model
python train.py --model yolov8n  # Less capacity, less overfit

# Solution 4: Less epochs
python train.py --epochs 50  # Don't train too long
```

---

### Problem 3: Underfitting

**Symptoms:**
- Training loss tinggi
- Validation loss tinggi
- Model tidak belajar dengan baik

**Solutions:**
```bash
# Solution 1: More epochs
python train.py --epochs 200

# Solution 2: Larger model
python train.py --model yolov8m

# Solution 3: Higher resolution
python train.py --imgsz 1280

# Solution 4: Check data quality
# - Apakah labels benar?
# - Apakah images jelas?
# - Apakah classes balanced?
```

---

### Problem 4: Training Too Slow

**Symptoms:**
- 1 epoch takes hours
- Can't iterate quickly

**Solutions:**
```bash
# Solution 1: Increase batch size (if GPU allows)
python train.py --batch 32

# Solution 2: Reduce image size
python train.py --imgsz 320

# Solution 3: Use smaller model for testing
python train.py --model yolov8n --epochs 10

# Solution 4: Use GPU instead of CPU
# Install CUDA + cuDNN
```

---

### Problem 5: Low mAP

**Symptoms:**
- Validation mAP < 0.5
- Model not accurate enough

**Solutions:**
```bash
# Solution 1: Train longer
python train.py --epochs 200 --patience 100

# Solution 2: Larger model
python train.py --model yolov8m

# Solution 3: Higher resolution
python train.py --imgsz 1280

# Solution 4: More data
# - Collect more images
# - Better data quality
# - Better labels

# Solution 5: Check dataset split
# - Apakah train/val/test balanced?
# - Apakah stratified split?
```

---

## Best Practices

### 1. Start Small, Scale Up
```bash
# ✅ Do this:
python train.py --model yolov8n --epochs 10 --dry-run  # Test
python train.py --model yolov8s --epochs 50             # Quick train
python train.py --model yolov8m --epochs 200            # Full train

# ❌ Don't do this:
python train.py --model yolov8x --epochs 500  # Too much, too soon
```

### 2. Monitor Training
```bash
# Always check:
├── logs/train_*.log                    # Training log
├── runs/detect/train/results.png       # Training curves
├── runs/detect/train/results.csv       # Metrics
└── runs/detect/train/weights/best.pt   # Best model
```

### 3. Use Dry Run First
```bash
# ✅ Always test first:
python train.py --dry-run

# Then actual training:
python train.py
```

### 4. Save Intermediate Checkpoints
```bash
# Checkpoints auto-saved every 10 epochs
# Can resume anytime:
python train.py --resume
```

### 5. Experiment Tracking
```bash
# Keep track of experiments:
├── Change one parameter at a time
├── Document results
├── Compare metrics
└── Use what works best
```

---

## Quick Reference

### Recommended Configurations

**Quick Test:**
```bash
python train.py --model yolov8n --epochs 10 --dry-run
```

**Fast Training (Laptop):**
```bash
python train.py --model yolov8s --epochs 50 --batch 8 --imgsz 640
```

**Standard Training (Desktop with GPU):**
```bash
python train.py --model yolov8s --epochs 100 --batch 16 --imgsz 640
```

**High Accuracy (Workstation):**
```bash
python train.py --model yolov8m --epochs 200 --batch 32 --imgsz 1280
```

**Best Accuracy (Research):**
```bash
python train.py --model yolov8x --epochs 300 --batch 64 --imgsz 1280 --patience 100
```

---

## Metrics Explained

### mAP (mean Average Precision)
- **Range:** 0.0 - 1.0 (higher is better)
- **Meaning:** Overall accuracy of detection
- **Good value:** > 0.5
- **Excellent value:** > 0.7

### mAP50
- **Meaning:** mAP at IoU threshold 0.5
- **Use:** Standard metric untuk object detection

### mAP50-95
- **Meaning:** mAP averaged over IoU 0.5 - 0.95
- **Use:** More strict metric (harder to achieve)

### Precision
- **Range:** 0.0 - 1.0
- **Meaning:** Dari semua yang diprediksi positive, berapa yang benar?
- **Formula:** TP / (TP + FP)

### Recall
- **Range:** 0.0 - 1.0
- **Meaning:** Dari semua yang sebenarnya positive, berapa yang terdeteksi?
- **Formula:** TP / (TP + FN)

---

## Common Commands Cheat Sheet

```bash
# Basic
python train.py

# Quick test
python train.py --dry-run

# Custom model
python train.py --model yolov8m

# Custom duration
python train.py --epochs 200

# Low memory
python train.py --batch 4 --imgsz 320

# High accuracy
python train.py --model yolov8l --imgsz 1280 --epochs 200

# Resume training
python train.py --resume

# Full custom
python train.py --model yolov8m --epochs 200 --batch 32 --imgsz 1280 --patience 100 --data custom.yaml

# Help
python train.py --help
```

---

**Happy Training! 🚀**

*Untuk pertanyaan lebih lanjut, lihat dokumentasi Ultralytics: https://docs.ultralytics.com*
