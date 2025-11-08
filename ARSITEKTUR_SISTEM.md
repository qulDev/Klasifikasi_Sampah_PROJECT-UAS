# Arsitektur Sistem Klasifikasi Sampah

## 📚 Daftar Isi

1. [Overview Sistem](#overview-sistem)
2. [BAGIAN 1: Dataset Processing Pipeline](#bagian-1-dataset-processing-pipeline)
3. [BAGIAN 2: Model Training Pipeline](#bagian-2-model-training-pipeline)
4. [BAGIAN 3: Inference/Detection Pipeline](#bagian-3-inferencedetection-pipeline)
5. [Complete System Flow](#complete-system-flow)
6. [Konsep Penting](#konsep-penting)
7. [Performance Metrics](#performance-metrics)

---

## Overview Sistem

Sistem klasifikasi sampah ini terdiri dari **3 komponen utama**:

```
┌──────────────────────────────────────────────────────────┐
│                     SISTEM LENGKAP                       │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  1. DATASET PROCESSING                                   │
│     └─ Persiapan data untuk training                     │
│                                                           │
│  2. MODEL TRAINING                                       │
│     └─ Melatih AI untuk mengenali sampah                 │
│                                                           │
│  3. INFERENCE/DETECTION                                  │
│     └─ Menggunakan model untuk deteksi real-time         │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

---

## BAGIAN 1: Dataset Processing Pipeline

### Apa itu Dataset?

**Dataset** adalah kumpulan gambar sampah beserta **labelnya** (misalnya: "ini plastic", "ini metal"). Seperti buku pelajaran dengan soal dan jawaban untuk melatih AI.

### Pipeline Processing

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Mengumpulkan Dataset dari Berbagai Sumber          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: Dataset dalam berbagai format                       │
│  ├── COCO Format (JSON)     → annotations.json             │
│  ├── VOC Format (XML)       → .xml files                    │
│  ├── Folder Structure       → plastic/, metal/, dll         │
│  └── YOLO Format (TXT)      → .txt files                    │
│                                                              │
│  Contoh struktur:                                           │
│  datasets/raw/                                              │
│    ├── Garbage-Kaggle/      (6 classes, folder format)     │
│    ├── TACO/                (COCO format, JSON)             │
│    └── TrashNet/            (folder format)                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Konversi Format → convert_datasets.py              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Proses:                                                    │
│  1. Auto-detect format dataset                              │
│  2. Parse annotations (baca file label)                     │
│  3. Normalize ke format YOLO (ubah semua jadi sama)        │
│  4. Verify images (cek gambar valid/rusak)                  │
│  5. Copy ke folder output                                   │
│                                                              │
│  YOLO Format adalah:                                        │
│  - 1 file .txt per gambar                                   │
│  - Isi: <class_id> <x_center> <y_center> <width> <height> │
│  - Semua nilai normalized (0-1)                             │
│                                                              │
│  Output:                                                    │
│  datasets/processed/all/                                    │
│    ├── images/          → semua gambar (.jpg/.png)         │
│    └── labels/          → semua label (.txt)               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Split Dataset → split_and_prep.py                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Mengapa perlu split?                                       │
│  - Training set: untuk belajar (80%)                        │
│  - Validation set: untuk evaluasi saat training (10%)      │
│  - Test set: untuk evaluasi akhir (10%)                    │
│                                                              │
│  Proses:                                                    │
│  1. Deduplikasi (hapus gambar duplikat)                    │
│  2. Stratified split (distribusi kelas merata)             │
│  3. Copy ke folder train/val/test                          │
│  4. Generate data.yaml (config untuk training)             │
│                                                              │
│  Output:                                                    │
│  datasets/processed/                                        │
│    ├── train/                                               │
│    │   ├── images/ (80% data)                              │
│    │   └── labels/                                          │
│    ├── val/                                                 │
│    │   ├── images/ (10% data)                              │
│    │   └── labels/                                          │
│    ├── test/                                                │
│    │   ├── images/ (10% data)                              │
│    │   └── labels/                                          │
│    └── data.yaml (config file)                             │
└─────────────────────────────────────────────────────────────┘
```

### Penjelasan data.yaml

File konfigurasi yang berisi informasi penting untuk training:

```yaml
# File ini berisi informasi untuk training
path: /path/to/dataset      # Lokasi dataset
train: train/images          # Folder training images
val: val/images              # Folder validation images
test: test/images            # Folder test images

nc: 6                        # Number of classes (jumlah kelas)
names:                       # Nama setiap kelas
  0: cardboard
  1: glass
  2: metal
  3: paper
  4: plastic
  5: trash
```

---

## BAGIAN 2: Model Training Pipeline

### Apa itu Training?

**Training** adalah proses "mengajar" AI untuk mengenali sampah. Seperti mengajari anak kecil membedakan warna - kita tunjukkan banyak contoh sampai dia bisa membedakan sendiri.

### Arsitektur YOLOv8

```
┌─────────────────────────────────────────────────────────────┐
│            YOLO (You Only Look Once) v8                     │
│         "Lihat sekali langsung tahu semua objek"            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  INPUT: Gambar 640x640 pixels                               │
│  ┌────────────────────────────────────────┐                │
│  │                                        │                │
│  │         [Gambar Sampah]                │                │
│  │                                        │                │
│  └────────────────────────────────────────┘                │
│                                                              │
│  Pre-processing:                                            │
│  • Resize gambar ke 640x640                                 │
│  • Normalisasi nilai pixel (0-255 → 0-1)                   │
│  • Data augmentation (flip, rotate, zoom) - saat training  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1: BACKBONE (CSPDarknet53)                          │
│  "Ekstraksi Fitur - Mengenali pola dasar"                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Fungsi: Mengubah gambar menjadi "fitur"                   │
│  Fitur = ciri khas yang AI pahami (warna, tekstur, bentuk) │
│                                                              │
│  Proses bertahap:                                           │
│  640x640x3 → 320x320x64  (detect edges/tepi)               │
│      ↓                                                       │
│  320x320x64 → 160x160x128 (detect simple shapes)           │
│      ↓                                                       │
│  160x160x128 → 80x80x256  (detect patterns)                │
│      ↓                                                       │
│  80x80x256 → 40x40x512    (detect complex patterns)        │
│      ↓                                                       │
│  40x40x512 → 20x20x1024   (detect high-level features)     │
│                                                              │
│  Analogi: Seperti mata manusia                              │
│  - Layer awal: lihat garis dan warna                        │
│  - Layer tengah: lihat bentuk (bulat, kotak)               │
│  - Layer akhir: lihat objek lengkap (botol, kaleng)        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 2: NECK (PANet - Path Aggregation Network)          │
│  "Menggabungkan Informasi dari Berbagai Skala"             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Fungsi: Gabungkan fitur detail + fitur global             │
│                                                              │
│  Mengapa perlu?                                             │
│  - Objek kecil perlu detail tinggi (fitur dari layer awal) │
│  - Objek besar perlu konteks (fitur dari layer akhir)      │
│                                                              │
│  Proses:                                                    │
│  ┌─────────────────────────────────────┐                   │
│  │  Bottom-up path (dari kecil → besar)│                   │
│  │  20x20 → 40x40 → 80x80              │                   │
│  └─────────────────────────────────────┘                   │
│              +                                               │
│  ┌─────────────────────────────────────┐                   │
│  │  Top-down path (dari besar → kecil) │                   │
│  │  80x80 → 40x40 → 20x20              │                   │
│  └─────────────────────────────────────┘                   │
│              =                                               │
│  Feature maps dengan informasi lengkap!                     │
│                                                              │
│  Analogi: Melihat hutan dari dekat DAN jauh               │
│  - Dekat: lihat detail tiap pohon                          │
│  - Jauh: lihat pola keseluruhan hutan                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3: HEAD (Detection Head)                            │
│  "Prediksi Akhir - Dimana objek & kelas apa"              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Fungsi: Menghasilkan 3 output penting                     │
│                                                              │
│  1. BOUNDING BOX (Kotak pembatas)                          │
│     • x, y: koordinat center (pusat kotak)                 │
│     • w, h: lebar dan tinggi kotak                         │
│     Output: [x, y, width, height]                          │
│                                                              │
│  2. OBJECTNESS SCORE (Confidence)                          │
│     • Seberapa yakin ada objek di kotak ini?               │
│     • Nilai 0-1 (0.8 = 80% yakin)                          │
│     Output: [confidence_score]                             │
│                                                              │
│  3. CLASS PROBABILITIES (Probabilitas Kelas)               │
│     • Objek ini termasuk kelas apa?                        │
│     • 6 nilai (untuk 6 kelas sampah)                       │
│     Output: [P(cardboard), P(glass), P(metal), ...]        │
│                                                              │
│  Multi-scale detection:                                     │
│  ├── Small objects: detected at 80x80 grid                 │
│  ├── Medium objects: detected at 40x40 grid                │
│  └── Large objects: detected at 20x20 grid                 │
│                                                              │
│  Total predictions: ~8400 boxes per image!                 │
│  (akan difilter di post-processing)                         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  POST-PROCESSING: Filter & Refine Predictions              │
│  "Pilih prediksi terbaik, buang yang salah"                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  STEP 1: Confidence Filtering                              │
│  • Buang boxes dengan confidence < threshold (0.5)         │
│  • 8400 boxes → ~100 boxes                                 │
│                                                              │
│  STEP 2: Non-Maximum Suppression (NMS)                     │
│  • Hapus boxes yang overlap (deteksi objek yang sama)      │
│  • Pilih box dengan confidence tertinggi                   │
│  • 100 boxes → ~5-10 boxes (objek unik)                    │
│                                                              │
│  Contoh NMS:                                                │
│  Box A: plastic, conf=0.9, area=(100,100,200,200)         │
│  Box B: plastic, conf=0.7, area=(105,95,205,195)          │
│  → Overlap 90% → Pilih Box A (confidence lebih tinggi)    │
│                                                              │
│  STEP 3: Class Mapping                                     │
│  • Convert class ID ke nama kelas                          │
│  • 0 → "cardboard", 1 → "glass", dst                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  OUTPUT: Detection Results                                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Format hasil:                                              │
│  [                                                          │
│    {                                                        │
│      "class": "plastic",                                    │
│      "confidence": 0.85,                                    │
│      "bbox": [100, 150, 200, 300],  # x,y,w,h              │
│      "color": (0, 255, 0)  # Warna kotak (hijau)           │
│    },                                                       │
│    {                                                        │
│      "class": "metal",                                      │
│      "confidence": 0.72,                                    │
│      "bbox": [350, 200, 150, 180]                          │
│      "color": (255, 0, 0)  # Warna kotak (merah)           │
│    }                                                        │
│  ]                                                          │
│                                                              │
│  Visualisasi:                                               │
│  ┌────────────────────────────────────┐                    │
│  │  [Gambar dengan kotak warna]       │                    │
│  │   ┏━━━━━━━┓                         │                    │
│  │   ┃plastic┃ 85%                     │                    │
│  │   ┗━━━━━━━┛                         │                    │
│  │              ┏━━━━┓                 │                    │
│  │              ┃metal┃ 72%            │                    │
│  │              ┗━━━━┛                 │                    │
│  └────────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

### Training Process (train.py)

```
┌─────────────────────────────────────────────────────────────┐
│  TRAINING LOOP (Epoch 1 sampai Epoch 100)                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Epoch = 1 kali lihat semua gambar training                │
│                                                              │
│  Untuk setiap epoch:                                        │
│  ├─ STEP 1: Forward Pass                                    │
│  │   • Gambar masuk ke model                               │
│  │   • Model prediksi: boxes + classes + confidence        │
│  │                                                          │
│  ├─ STEP 2: Calculate Loss                                  │
│  │   • Bandingkan prediksi vs ground truth (label asli)   │
│  │   • 3 jenis loss:                                       │
│  │     - Box loss: seberapa akurat posisi kotak?          │
│  │     - Class loss: seberapa benar klasifikasi?          │
│  │     - Objectness loss: seberapa yakin ada objek?       │
│  │   • Total loss = box_loss + class_loss + obj_loss      │
│  │                                                          │
│  ├─ STEP 3: Backward Pass (Backpropagation)                │
│  │   • Hitung gradient (turunan loss)                      │
│  │   • Update weights model agar loss berkurang            │
│  │   • Menggunakan optimizer: AdamW                        │
│  │                                                          │
│  ├─ STEP 4: Validation                                      │
│  │   • Test model di validation set                        │
│  │   • Hitung metrics:                                     │
│  │     - mAP (mean Average Precision)                      │
│  │     - Precision & Recall                                │
│  │   • Jika mAP naik → save model sebagai "best"          │
│  │                                                          │
│  └─ STEP 5: Logging & Checkpointing                        │
│      • Log metrics ke file                                  │
│      • Save checkpoint setiap 10 epoch                     │
│      • Generate training plots (loss curves)               │
│                                                              │
│  Early Stopping:                                            │
│  • Jika validation mAP tidak improve selama 50 epoch      │
│  • Stop training (avoid overfitting)                       │
│                                                              │
│  Output:                                                    │
│  runs/detect/train/                                         │
│    ├── weights/                                             │
│    │   ├── best.pt    (model terbaik)                      │
│    │   └── last.pt    (checkpoint terakhir)                │
│    ├── results.csv    (training metrics)                   │
│    └── results.png    (training curves)                    │
└─────────────────────────────────────────────────────────────┘
```

---

## BAGIAN 3: Inference/Detection Pipeline

### Apa itu Inference?

**Inference** adalah proses "menggunakan" model yang sudah dilatih untuk mendeteksi objek di gambar atau video baru. Ini adalah "ujian" untuk model kita!

### Real-time Detection (detect.py)

```
┌─────────────────────────────────────────────────────────────┐
│  REAL-TIME DETECTION PIPELINE                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  STEP 1: Capture Frame dari Webcam                         │
│  ├─ OpenCV baca frame dari kamera                          │
│  ├─ Frame rate: 30 FPS (30 gambar per detik)              │
│  └─ Resolution: 640x480 atau sesuai kamera                 │
│                                                              │
│  STEP 2: Pre-processing                                     │
│  ├─ Resize frame ke 640x640 (input size YOLO)             │
│  ├─ Normalisasi pixel values                               │
│  └─ Convert BGR → RGB (OpenCV uses BGR)                    │
│                                                              │
│  STEP 3: Model Inference                                    │
│  ├─ Load model (models/best.pt)                            │
│  ├─ Feed frame ke model                                     │
│  ├─ Model process (Forward Pass)                           │
│  │   • Backbone extract features                           │
│  │   • Neck aggregate features                             │
│  │   • Head predict boxes + classes                        │
│  └─ Post-processing (NMS)                                   │
│                                                              │
│  STEP 4: Draw Results                                       │
│  ├─ Gambar bounding boxes                                   │
│  │   • Warna berbeda per kelas                             │
│  │   • Thickness: 2 pixels                                 │
│  ├─ Tulis label + confidence                                │
│  │   • Format: "plastic 85%"                               │
│  │   • Font: Arial, size 1                                 │
│  └─ Tambah info panel                                       │
│      • FPS counter (speed)                                  │
│      • Timestamp                                            │
│      • Jumlah objek terdeteksi                             │
│                                                              │
│  STEP 5: Display & Controls                                 │
│  ├─ Show frame di window                                    │
│  ├─ Handle keyboard input:                                  │
│  │   • Q: Quit (keluar)                                    │
│  │   • S: Screenshot (save gambar)                         │
│  │   • SPACE: Pause/Resume                                 │
│  └─ Loop ke STEP 1                                          │
│                                                              │
│  Performance:                                               │
│  ├─ GPU: 30+ FPS (real-time smooth)                        │
│  ├─ CPU: 5-10 FPS (agak lambat tapi masih ok)             │
│  └─ Latency: ~30ms per frame (GPU)                         │
└─────────────────────────────────────────────────────────────┘
```

### Interactive Detection (scan_image.ipynb)

```
┌─────────────────────────────────────────────────────────────┐
│  NOTEBOOK DETECTION PIPELINE                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Cell 1: Setup & Imports                                    │
│  ├─ Import libraries (YOLO, PIL, matplotlib)               │
│  ├─ Setup display settings                                  │
│  └─ Configure plot parameters                               │
│                                                              │
│  Cell 2: Load Model                                         │
│  ├─ Path: models/best.pt                                    │
│  ├─ Load YOLOv8 model                                       │
│  ├─ Set confidence threshold (0.5)                          │
│  └─ Display model info                                      │
│                                                              │
│  Cell 3: Upload & Auto-Detect                              │
│  ├─ Upload widget (drag & drop)                            │
│  ├─ Observer pattern:                                       │
│  │   • Saat file uploaded → trigger detection             │
│  │   • Auto-process tanpa klik button                     │
│  ├─ Detection process:                                      │
│  │   1. Load image dari upload                             │
│  │   2. Run inference                                       │
│  │   3. Draw boxes & labels                                │
│  │   4. Display result                                      │
│  └─ Support multiple images                                 │
│                                                              │
│  Cell 4: Help & Documentation                               │
│  ├─ Cara pakai                                              │
│  ├─ Troubleshooting tips                                    │
│  └─ Examples                                                │
│                                                              │
│  User Experience:                                           │
│  1. Run all cells (Shift+Enter)                            │
│  2. Drag gambar ke upload box                               │
│  3. Hasil muncul otomatis (tanpa klik apapun!)             │
│  4. Upload lagi untuk test gambar lain                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Complete System Flow

### End-to-End Flow dari Dataset sampai Deteksi

```
    ┌──────────────┐
    │ Raw Datasets │  ← Multiple sources (Kaggle, TACO, etc)
    └──────┬───────┘
           │
           ▼
    ┌──────────────────────────┐
    │ convert_datasets.py      │  ← Auto-detect & normalize
    │ • Parse annotations      │
    │ • Verify images          │
    │ • Convert to YOLO format │
    └──────┬───────────────────┘
           │
           ▼
    ┌──────────────────────────┐
    │ Processed Dataset        │  ← All in YOLO format
    │ images/ + labels/        │
    └──────┬───────────────────┘
           │
           ▼
    ┌──────────────────────────┐
    │ split_and_prep.py        │  ← Stratified split
    │ • Deduplicate            │
    │ • Split 80/10/10         │
    │ • Generate data.yaml     │
    └──────┬───────────────────┘
           │
           ▼
    ┌──────────────────────────┐
    │ Split Dataset            │  ← Ready for training
    │ train/ val/ test/        │
    └──────┬───────────────────┘
           │
           ▼
    ┌──────────────────────────┐
    │ train.py                 │  ← Train YOLOv8 model
    │ • 100 epochs             │
    │ • Batch size 16          │
    │ • Image size 640         │
    │ • AdamW optimizer        │
    └──────┬───────────────────┘
           │
           ▼
    ┌──────────────────────────┐
    │ Trained Model            │  ← models/best.pt
    │ • Weights: ~22 MB        │
    │ • mAP: [nilai]           │
    └──────┬───────────────────┘
           │
           ├──────────────────────────┐
           │                          │
           ▼                          ▼
    ┌──────────────┐        ┌──────────────────┐
    │  detect.py   │        │ scan_image.ipynb │
    │  Webcam      │        │ Upload images    │
    │  Real-time   │        │ Interactive      │
    │  30+ FPS     │        │ Testing          │
    └──────┬───────┘        └────────┬─────────┘
           │                         │
           ▼                         ▼
    ┌──────────────────────────────────────┐
    │      Detection Results               │
    │  Bounding boxes + Labels + Conf      │
    │  "plastic 85%", "metal 72%"          │
    └──────────────────────────────────────┘
```

---

## Konsep Penting

### 1. Transfer Learning

```
┌─────────────────────────────────────────────────────────┐
│  Transfer Learning: Belajar dari Model Pretrained      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Tanpa Transfer Learning:                               │
│  ├─ Train dari nol (random weights)                     │
│  ├─ Perlu banyak data (100K+ images)                    │
│  ├─ Training lama (1-2 minggu)                          │
│  └─ Hasil kurang akurat                                 │
│                                                          │
│  Dengan Transfer Learning:                              │
│  ├─ Pakai YOLOv8 pretrained di COCO (118K images)      │
│  ├─ Model sudah tahu: edges, shapes, objects           │
│  ├─ Kita hanya "fine-tune" untuk sampah kita           │
│  ├─ Perlu data lebih sedikit (5K-10K images)           │
│  ├─ Training cepat (2-4 jam)                            │
│  └─ Hasil lebih akurat!                                 │
│                                                          │
│  Analogi: Seperti mengajar anak yang sudah bisa        │
│  membaca untuk membaca bahasa baru. Lebih cepat        │
│  daripada mengajar dari nol!                            │
└─────────────────────────────────────────────────────────┘
```

### 2. Batch Processing

```
┌─────────────────────────────────────────────────────────┐
│  Batch Processing: Process Multiple Images at Once     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Sequential Processing (1 by 1):                        │
│  Image 1 → Process → Image 2 → Process → ...           │
│  Lambat! GPU tidak terpakai maksimal                    │
│                                                          │
│  Batch Processing (16 at once):                         │
│  [Img1, Img2, ..., Img16] → Process Parallel → ...     │
│  Cepat! GPU bekerja optimal                             │
│                                                          │
│  Trade-offs:                                             │
│  ├─ Batch besar (32, 64):                              │
│  │   • Lebih cepat                                      │
│  │   • Perlu GPU memory besar (8GB+)                   │
│  │   • Risk: out of memory                             │
│  │                                                      │
│  └─ Batch kecil (8, 16):                               │
│      • Agak lambat                                      │
│      • GPU memory kecil ok (4GB)                        │
│      • Aman                                             │
│                                                          │
│  Rekomendasi: batch=16 (balance antara speed & memory) │
└─────────────────────────────────────────────────────────┘
```

### 3. Data Augmentation

```
┌─────────────────────────────────────────────────────────┐
│  Data Augmentation: Variasi Data untuk Training        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Tanpa Augmentation:                                    │
│  • Model hanya lihat gambar asli                        │
│  • Risk: "hafal" data training (overfitting)           │
│  • Tidak bisa generalisasi ke data baru                │
│                                                          │
│  Dengan Augmentation:                                   │
│  Original Image → Apply random transforms:              │
│  ├─ Flip horizontal (cermin)                           │
│  ├─ Rotate ±10 derajat                                 │
│  ├─ Zoom in/out 10-20%                                 │
│  ├─ Adjust brightness ±20%                             │
│  ├─ Adjust contrast ±20%                               │
│  └─ Add noise                                           │
│                                                          │
│  Benefit:                                               │
│  • Model lebih robust                                   │
│  • Bisa handle variasi lighting, angle, dsb            │
│  • Mengurangi overfitting                               │
│  • "Perbanyak" data tanpa koleksi gambar baru          │
│                                                          │
│  Contoh:                                                │
│  1 gambar → 10 variasi → 10x lebih banyak data!        │
└─────────────────────────────────────────────────────────┘
```

### 4. Multi-scale Detection

```
┌─────────────────────────────────────────────────────────┐
│  Multi-scale Detection: Deteksi Objek Berbagai Ukuran  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Problem:                                               │
│  • Objek kecil (tutup botol): 20x20 pixels             │
│  • Objek medium (botol): 100x100 pixels                │
│  • Objek besar (kardus): 300x300 pixels                │
│  • Bagaimana detect semuanya dengan akurat?            │
│                                                          │
│  Solution: 3-Scale Detection                            │
│                                                          │
│  ┌──────────────────────────────────────────┐          │
│  │  Large Grid (20x20)                      │          │
│  │  ├─ Cell size: 32x32 pixels              │          │
│  │  ├─ Detect: Large objects (kardus)       │          │
│  │  └─ Receptive field: wide                │          │
│  └──────────────────────────────────────────┘          │
│                                                          │
│  ┌──────────────────────────────────────────┐          │
│  │  Medium Grid (40x40)                     │          │
│  │  ├─ Cell size: 16x16 pixels              │          │
│  │  ├─ Detect: Medium objects (botol)       │          │
│  │  └─ Receptive field: medium              │          │
│  └──────────────────────────────────────────┘          │
│                                                          │
│  ┌──────────────────────────────────────────┐          │
│  │  Small Grid (80x80)                      │          │
│  │  ├─ Cell size: 8x8 pixels                │          │
│  │  ├─ Detect: Small objects (tutup botol)  │          │
│  │  └─ Receptive field: narrow              │          │
│  └──────────────────────────────────────────┘          │
│                                                          │
│  Total predictions per image:                           │
│  (20×20 + 40×40 + 80×80) × 3 anchors = ~25,200 boxes  │
│  (akan difilter dengan NMS)                             │
└─────────────────────────────────────────────────────────┘
```

### 5. Non-Maximum Suppression (NMS)

```
┌─────────────────────────────────────────────────────────┐
│  NMS: Pilih Bounding Box Terbaik                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Problem: 1 objek terdeteksi berkali-kali              │
│                                                          │
│  ┌─────────────────────┐                               │
│  │   ┏━━━━━━┓          │                               │
│  │   ┃ Box A┃ 0.9      │  ← Confidence 90%             │
│  │   ┃┏━━━━━┛━┓        │                               │
│  │   ┗┃Box B  ┃ 0.7    │  ← Confidence 70%             │
│  │    ┃┏━━━━━━┛━┓      │                               │
│  │    ┗┃ Box C ┃ 0.6   │  ← Confidence 60%             │
│  │     ┗━━━━━━━┛       │                               │
│  └─────────────────────┘                               │
│  Semua box overlap > 50% → deteksi objek yang sama!    │
│                                                          │
│  Solution: NMS Algorithm                                │
│  1. Sort boxes by confidence (high → low)              │
│     Box A (0.9) > Box B (0.7) > Box C (0.6)           │
│                                                          │
│  2. Pilih box dengan confidence tertinggi (Box A)      │
│                                                          │
│  3. Hitung IoU (Intersection over Union) dengan Box A  │
│     IoU(A,B) = 0.85 > threshold(0.45) → Buang Box B   │
│     IoU(A,C) = 0.70 > threshold(0.45) → Buang Box C   │
│                                                          │
│  4. Keep only Box A                                     │
│                                                          │
│  Result: 1 objek = 1 box (clean detection!)            │
│                                                          │
│  ┌─────────────────────┐                               │
│  │   ┏━━━━━━┓          │                               │
│  │   ┃ Box A┃ 0.9      │  ← Keep this!                 │
│  │   ┗━━━━━━┛          │                               │
│  └─────────────────────┘                               │
└─────────────────────────────────────────────────────────┘
```

---

## Performance Metrics

### Confusion Matrix Explained

```
┌─────────────────────────────────────────────────────────┐
│  Confusion Matrix (untuk 1 kelas, contoh: "plastic")   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│                      Prediksi                           │
│                 Positive    Negative                    │
│  Actual  ┌─────────────────────────────┐               │
│  Positive│   TP         │      FN      │               │
│          │ (Benar)      │  (Miss)      │               │
│  Negative│   FP         │      TN      │               │
│          │(False Alarm) │   (Benar)    │               │
│          └─────────────────────────────┘               │
│                                                          │
│  TP (True Positive):                                    │
│  • Model benar deteksi plastic                          │
│  • Ada plastic, model bilang "plastic" ✓               │
│                                                          │
│  FP (False Positive):                                   │
│  • Model salah deteksi (false alarm)                   │
│  • Bukan plastic, model bilang "plastic" ✗             │
│                                                          │
│  FN (False Negative):                                   │
│  • Model miss detection (tidak terdeteksi)             │
│  • Ada plastic, model tidak deteksi ✗                  │
│                                                          │
│  TN (True Negative):                                    │
│  • Model benar tidak ada objek                          │
│  • Tidak ada plastic, model tidak bilang plastic ✓     │
└─────────────────────────────────────────────────────────┘
```

### Metrics Formulas

```
┌─────────────────────────────────────────────────────────┐
│  Precision: Akurasi Prediksi Positive                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Formula: Precision = TP / (TP + FP)                    │
│                                                          │
│  Pertanyaan: "Dari semua yang diprediksi positive,     │
│               berapa yang beneran positive?"            │
│                                                          │
│  Contoh:                                                │
│  • Model prediksi 100 plastic                           │
│  • Yang benar plastic: 85                               │
│  • Yang salah (bukan plastic): 15                       │
│  • Precision = 85/100 = 0.85 = 85%                     │
│                                                          │
│  Interpretasi:                                          │
│  • High precision (>90%) = sedikit false alarm         │
│  • Low precision (<50%) = banyak salah deteksi         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Recall: Deteksi Semua Positive                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Formula: Recall = TP / (TP + FN)                       │
│                                                          │
│  Pertanyaan: "Dari semua positive yang ada,            │
│               berapa yang terdeteksi?"                  │
│                                                          │
│  Contoh:                                                │
│  • Ada 100 plastic dalam gambar                         │
│  • Yang terdeteksi: 90                                  │
│  • Yang kelewatan (miss): 10                            │
│  • Recall = 90/100 = 0.90 = 90%                        │
│                                                          │
│  Interpretasi:                                          │
│  • High recall (>90%) = sedikit yang kelewatan         │
│  • Low recall (<50%) = banyak yang tidak terdeteksi    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  F1-Score: Balance Precision & Recall                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Formula: F1 = 2 × (Precision × Recall)                │
│                    ───────────────────                  │
│                    (Precision + Recall)                 │
│                                                          │
│  Mengapa perlu?                                         │
│  • Precision tinggi, Recall rendah → miss banyak objek │
│  • Precision rendah, Recall tinggi → banyak false alarm│
│  • F1-Score = harmonic mean (balance keduanya)         │
│                                                          │
│  Interpretasi:                                          │
│  • F1 > 0.8 = Excellent                                │
│  • F1 = 0.6-0.8 = Good                                 │
│  • F1 < 0.6 = Need improvement                         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  mAP: Mean Average Precision                            │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  mAP = Industry standard untuk object detection         │
│                                                          │
│  Cara hitung:                                           │
│  1. Hitung Average Precision (AP) untuk tiap kelas     │
│  2. Rata-rata AP dari semua kelas = mAP                │
│                                                          │
│  mAP50:                                                 │
│  • IoU threshold = 0.5 (50% overlap)                   │
│  • Lebih "generous" (mudah dapat score tinggi)         │
│                                                          │
│  mAP50-95:                                              │
│  • Average IoU threshold dari 0.5 sampai 0.95          │
│  • Lebih "strict" (susah dapat score tinggi)           │
│  • Lebih representative untuk akurasi real             │
│                                                          │
│  Interpretasi:                                          │
│  • mAP > 0.7 = Excellent (production ready)            │
│  • mAP = 0.5-0.7 = Good (usable)                       │
│  • mAP < 0.5 = Need improvement                        │
│                                                          │
│  Contoh:                                                │
│  • cardboard: AP = 0.85                                │
│  • glass: AP = 0.80                                    │
│  • metal: AP = 0.75                                    │
│  • paper: AP = 0.82                                    │
│  • plastic: AP = 0.88                                  │
│  • trash: AP = 0.70                                    │
│  • mAP = (0.85+0.80+0.75+0.82+0.88+0.70)/6 = 0.80     │
└─────────────────────────────────────────────────────────┘
```

---

## Rangkuman Singkat

### Pipeline Lengkap

1. **Dataset Processing** → Siapkan data dalam format YOLO
2. **Model Training** → Latih YOLOv8 dengan transfer learning
3. **Inference** → Deteksi real-time dengan webcam atau upload gambar

### Scripts Utama

- `convert_datasets.py` - Konversi dataset ke YOLO format
- `split_and_prep.py` - Split dataset train/val/test
- `train.py` - Training YOLOv8 model
- `detect.py` - Real-time webcam detection
- `scan_image.ipynb` - Interactive image detection

### Performance Target

- **mAP > 0.7** - Excellent
- **FPS > 30** - Real-time (GPU)
- **Model size < 25 MB** - Deployable

---

**Dibuat untuk:** Proyek UAS Klasifikasi Sampah  
**Teknologi:** YOLOv8 + PyTorch + OpenCV  
**Tanggal:** November 2025
