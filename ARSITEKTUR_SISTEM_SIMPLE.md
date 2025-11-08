# Arsitektur Sistem Klasifikasi Sampah
## Versi Simple untuk PPT

---

## 🔄 Alur Sistem Lengkap

```
┌─────────────────┐
│  Raw Datasets   │  ← 5 dataset berbeda (Kaggle, TACO, dll)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  convert_datasets.py        │  ← Normalisasi ke YOLO format
│  • Parse annotations        │
│  • Verify images            │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  split_and_prep.py          │  ← Split 80/10/10
│  • Deduplicate images       │
│  • Stratified sampling      │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  train.py                   │  ← Train YOLOv8
│  • 100 epochs               │
│  • Transfer learning        │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Trained Model (best.pt)    │  ← ~22 MB, mAP > 0.7
└────────┬────────────────────┘
         │
         ├──────────────────┐
         ▼                  ▼
    ┌─────────┐      ┌─────────────┐
    │detect.py│      │scan_image   │
    │Webcam   │      │.ipynb       │
    │30+ FPS  │      │Interactive  │
    └─────────┘      └─────────────┘
```

---

## 🧠 Arsitektur YOLOv8

```
┌──────────────────────────────────────────┐
│            INPUT (640x640)                │
│         [Gambar Sampah]                   │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  BACKBONE (CSPDarknet53)                 │
│  "Ekstraksi Fitur"                       │
│  • Detect edges                          │
│  • Detect shapes                         │
│  • Detect patterns                       │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  NECK (PANet)                            │
│  "Gabungkan Fitur Multi-Scale"           │
│  • Bottom-up path                        │
│  • Top-down path                         │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  HEAD (Detection Head)                   │
│  "Prediksi Akhir"                        │
│  • Bounding boxes (x,y,w,h)              │
│  • Class probabilities                   │
│  • Confidence scores                     │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  POST-PROCESSING                         │
│  • Confidence filtering (>0.5)           │
│  • NMS (Non-Maximum Suppression)         │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  OUTPUT                                  │
│  [Gambar + Bounding Boxes + Labels]      │
│  "plastic 85%", "metal 72%"              │
└──────────────────────────────────────────┘
```

---

## 📊 3 Komponen Utama

### 1. Dataset Processing

```
Raw Data (berbagai format)
    ↓
Convert ke YOLO format
    ↓
Split train/val/test (80/10/10)
    ↓
Ready for training
```

**Output:** `data.yaml` + organized folders

---

### 2. Model Training

```
Load pretrained YOLOv8
    ↓
Training loop (100 epochs)
    ├─ Forward pass
    ├─ Calculate loss
    ├─ Backpropagation
    └─ Update weights
    ↓
Save best model
```

**Output:** `best.pt` (trained model)

---

### 3. Inference/Detection

```
Load trained model (best.pt)
    ↓
Capture image (webcam/upload)
    ↓
Pre-process (resize, normalize)
    ↓
Model inference
    ↓
Post-process (NMS)
    ↓
Draw results & display
```

**Output:** Detected objects with bounding boxes

---

## 🔑 Konsep Penting

### Transfer Learning
```
❌ Train dari nol:
   • Perlu 100K+ images
   • Training 1-2 minggu
   • Hasil kurang akurat

✅ Transfer learning:
   • Pakai YOLOv8 pretrained
   • Perlu 5K-10K images
   • Training 2-4 jam
   • Hasil lebih akurat
```

### Multi-scale Detection
```
3 skala deteksi:
├─ 80x80 grid → objek kecil
├─ 40x40 grid → objek medium
└─ 20x20 grid → objek besar
```

### Non-Maximum Suppression (NMS)
```
Problem: 1 objek = banyak boxes
Solution: Pilih box terbaik, buang yang overlap
Result: 1 objek = 1 box
```

---

## 📈 Performance Metrics

### Confusion Matrix
```
                Prediksi
             Positive  Negative
Actual  Pos     TP        FN
        Neg     FP        TN
```

### Formulas
```
Precision = TP / (TP + FP)
  → Dari prediksi positive, berapa yang benar?

Recall = TP / (TP + FN)
  → Dari actual positive, berapa yang terdeteksi?

F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
  → Balance keduanya

mAP = mean Average Precision
  → Industry standard, rata-rata AP semua kelas
```

### Target Performance
```
✅ mAP > 0.7     (Excellent)
✅ FPS > 30      (Real-time)
✅ Size < 25 MB  (Deployable)
```

---

## 💻 Scripts Utama

| Script | Fungsi | Input | Output |
|--------|--------|-------|--------|
| `convert_datasets.py` | Konversi format | Raw datasets | YOLO format |
| `split_and_prep.py` | Split dataset | All data | train/val/test |
| `train.py` | Training model | data.yaml | best.pt |
| `detect.py` | Webcam detection | Video stream | Real-time boxes |
| `scan_image.ipynb` | Upload & detect | Image files | Annotated images |

---

## 🎯 Training Process

```
┌──────────────────────────────────────┐
│  TRAINING LOOP (100 epochs)         │
├──────────────────────────────────────┤
│                                      │
│  For each epoch:                    │
│  1. Forward Pass                     │
│     → Model prediksi                 │
│                                      │
│  2. Calculate Loss                   │
│     → Bandingkan dengan label asli   │
│                                      │
│  3. Backward Pass                    │
│     → Update weights                 │
│                                      │
│  4. Validation                       │
│     → Test di validation set         │
│     → Save jika mAP meningkat        │
│                                      │
│  5. Logging                          │
│     → Save metrics & plots           │
│                                      │
└──────────────────────────────────────┘
```

**Early Stopping:** Stop jika tidak improve 50 epochs

---

## 🚀 Real-time Detection

```
┌──────────────────────────────────────┐
│  DETECTION LOOP (30+ FPS)            │
├──────────────────────────────────────┤
│                                      │
│  1. Capture frame                    │
│     → Read from webcam               │
│                                      │
│  2. Pre-process                      │
│     → Resize to 640x640              │
│                                      │
│  3. Inference                        │
│     → Model predict                  │
│                                      │
│  4. Post-process                     │
│     → Filter & NMS                   │
│                                      │
│  5. Draw & Display                   │
│     → Bounding boxes + labels        │
│     → Show FPS & info                │
│                                      │
│  6. Handle input                     │
│     → Q=quit, S=screenshot           │
│                                      │
└──────────────────────────────────────┘
```

---

## 📦 Dataset Format

### YOLO Format
```
File structure:
datasets/
  ├── train/
  │   ├── images/
  │   │   ├── img1.jpg
  │   │   └── img2.jpg
  │   └── labels/
  │       ├── img1.txt
  │       └── img2.txt
  ├── val/
  └── test/

Label format (img1.txt):
0 0.5 0.5 0.3 0.4
│  │   │   │   └── height (normalized)
│  │   │   └────── width (normalized)
│  │   └────────── y_center (normalized)
│  └────────────── x_center (normalized)
└───────────────── class_id (0-5)
```

---

## 🎓 Kesimpulan

### Keunggulan Sistem
✅ **Otomatis** - Pipeline lengkap dari dataset → detection  
✅ **Cepat** - Real-time 30+ FPS (GPU)  
✅ **Akurat** - State-of-the-art YOLOv8  
✅ **Mudah** - Smart defaults, tinggal run  

### Use Cases
- 🏭 Smart waste sorting facilities
- 📱 Mobile apps untuk recycling
- 🎓 Educational tools
- 🤖 Automatic waste management

---

**Teknologi:** YOLOv8 + PyTorch + OpenCV  
**Target:** mAP > 0.7, FPS > 30  
**Model Size:** ~22 MB (deployable)
