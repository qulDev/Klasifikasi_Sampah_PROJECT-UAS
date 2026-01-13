# ♻️ Klasifikasi Sampah Anorganik Menggunakan Algoritma YOLO

> Sistem deteksi dan klasifikasi sampah anorganik menggunakan algoritma YOLO (You Only Look Once)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-teal.svg)

Project ini mengimplementasikan sistem deteksi dan klasifikasi **10 jenis sampah** secara otomatis menggunakan deep learning dengan algoritma **YOLO (You Only Look Once)**. Dilengkapi dengan Web App (Streamlit), REST API (FastAPI), dan Real-time Detection.

---

## 🎯 Fitur Utama

| Fitur                      | Deskripsi                                 |
| -------------------------- | ----------------------------------------- |
| 🌐 **Web App**             | Interface web interaktif dengan Streamlit |
| 🔌 **REST API**            | API endpoint untuk integrasi sistem lain  |
| 📹 **Real-time Detection** | Deteksi langsung dari webcam              |
| 📓 **Jupyter Notebook**    | Interface interaktif untuk eksperimen     |
| 🔄 **Auto Pipeline**       | Konversi dataset otomatis ke format YOLO  |
| 🎯 **10 Kelas**            | Klasifikasi lengkap berbagai jenis sampah |

---

## 🗑️ 10 Kelas Sampah

| ID  | Kelas          | Emoji | Kategori       | Tempat Sampah |
| --- | -------------- | ----- | -------------- | ------------- |
| 0   | **battery**    | 🔋    | B3 (Berbahaya) | 🔴 Merah      |
| 1   | **biological** | 🥬    | Organik        | 🟢 Hijau      |
| 2   | **cardboard**  | 📦    | Anorganik      | 🔵 Biru       |
| 3   | **clothes**    | 👕    | Tekstil        | 🔵 Biru       |
| 4   | **glass**      | 🍾    | Anorganik      | 🔵 Biru       |
| 5   | **metal**      | 🥫    | Anorganik      | 🔵 Biru       |
| 6   | **paper**      | 📄    | Anorganik      | 🔵 Biru       |
| 7   | **plastic**    | 🥤    | Anorganik      | 🟡 Kuning     |
| 8   | **shoes**      | 👟    | Tekstil        | 🔵 Biru       |
| 9   | **trash**      | 🗑️    | Residu         | ⚫ Hitam      |

---

## 📦 Instalasi

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (opsional, untuk GPU)
- 8GB+ RAM

### Setup

```bash
# 1. Clone repository
git clone https://github.com/qulDev/Klasifikasi_Sampah_PROJECT-UAS.git
cd Klasifikasi_Sampah_PROJECT-UAS

# 2. Buat virtual environment
python -m venv .venv

# 3. Aktivasi environment
# Windows:
.\.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verifikasi instalasi
python -c "import torch; import ultralytics; print('✓ Ready!')"
```

---

## 🚀 Quick Start

### Opsi 1: Web App (Recommended) 🌐

```bash
streamlit run web_app.py
```

Buka http://localhost:8501 di browser.

**Fitur:**

- Upload gambar atau gunakan kamera
- Visualisasi hasil deteksi
- Info lengkap per objek (kategori, cara buang, tips)
- Summary recyclable vs non-recyclable

---

### Opsi 2: REST API 🔌

```bash
uvicorn api:app --reload --port 8000
```

Buka http://localhost:8000/docs untuk interactive API docs.

**Endpoints:**

| Method | Endpoint        | Deskripsi               |
| ------ | --------------- | ----------------------- |
| `GET`  | `/`             | Info API                |
| `GET`  | `/health`       | Health check            |
| `GET`  | `/classes`      | Daftar 10 kelas         |
| `POST` | `/detect`       | Deteksi sampah (JSON)   |
| `POST` | `/detect/image` | Deteksi (return gambar) |

**Contoh Request:**

```bash
curl -X POST "http://localhost:8000/detect" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@gambar_sampah.jpg" \
  -F "confidence=0.25"
```

---

### Opsi 3: Real-time Webcam 📹

```bash
python detect.py
```

**Kontrol:**

- `Q` = Quit
- `S` = Save screenshot
- `C` = Toggle confidence

---

### Opsi 4: Jupyter Notebook 📓

```bash
jupyter notebook notebooks/scan_image.ipynb
```

---

## 🔧 Training Model

### 1. Persiapan Dataset

```bash
# Convert dataset ke format YOLO
python convert_datasets.py

# Split train/val/test (80/10/10)
python split_and_prep.py
```

### 2. Training

```bash
# Training standar (100 epoch)
python train.py --epochs 100

# Training cepat (test)
python train.py --epochs 10 --dry-run

# Model lebih besar (lebih akurat)
python train.py --model yolov8m --epochs 100

# Lanjutkan training (resume)
python train.py --resume --epochs 50
```

### 3. Model Variants

| Model     | Size  | Speed  | Accuracy   |
| --------- | ----- | ------ | ---------- |
| `yolov8n` | 3.2M  | ⚡⚡⚡ | ⭐         |
| `yolov8s` | 11.2M | ⚡⚡   | ⭐⭐       |
| `yolov8m` | 25.9M | ⚡     | ⭐⭐⭐     |
| `yolov8l` | 43.7M | 🐢     | ⭐⭐⭐⭐   |
| `yolov8x` | 68.2M | 🐢🐢   | ⭐⭐⭐⭐⭐ |

---

## 📁 Struktur Project

```
Klasifikasi_Sampah/
├── 🌐 web_app.py           # Streamlit Web App
├── 🔌 api.py               # FastAPI REST API
├── 📹 detect.py            # Real-time webcam detection
├── 🔄 convert_datasets.py  # Dataset converter
├── ✂️ split_and_prep.py    # Dataset splitter
├── 🎯 train.py             # Model training
├── 📋 data.yaml            # YOLO config
├── 📦 requirements.txt     # Dependencies
│
├── models/
│   └── best_model.pt       # Trained model
│
├── datasets/
│   ├── raw/                # Dataset mentah
│   │   └── garbage-classification-v2/
│   │       ├── battery/
│   │       ├── biological/
│   │       ├── cardboard/
│   │       ├── clothes/
│   │       ├── glass/
│   │       ├── metal/
│   │       ├── paper/
│   │       ├── plastic/
│   │       ├── shoes/
│   │       └── trash/
│   └── processed/          # Dataset siap training
│       ├── train/
│       ├── val/
│       └── test/
│
├── notebooks/
│   └── scan_image.ipynb    # Jupyter notebook
│
├── utils/
│   ├── annotation_parsers.py
│   ├── dataset_stats.py
│   ├── image_utils.py
│   ├── label_mapper.py
│   └── logger.py
│
└── runs/
    └── detect/
        └── train/          # Training results
```

---

## 🛠️ Konfigurasi

### Web App (`web_app.py`)

```python
MODEL_PATH = './models/best_model.pt'
DEFAULT_CONF = 0.25
```

### API (`api.py`)

```python
MODEL_PATH = './models/best_model.pt'
DEFAULT_CONF = 0.25
```

### Webcam Detection (`detect.py`)

```python
MODEL = './models/best_model.pt'
CONF = 0.25
CAM = 0  # Camera index
```

---

## 🐛 Troubleshooting

### Model tidak ditemukan

```bash
# Pastikan sudah training
python train.py --epochs 100
```

### CUDA Out of Memory

```bash
# Kurangi batch size
python train.py --batch 8
```

### Webcam tidak terbuka

```python
# Edit detect.py, ganti CAM
CAM = 1  # Coba index lain
```

### Akurasi rendah

```bash
# Training lebih lama
python train.py --epochs 200 --patience 30

# Atau gunakan model lebih besar
python train.py --model yolov8m
```

---

## 📊 Output Files

Setelah training, file tersimpan di:

| File         | Lokasi                          | Deskripsi               |
| ------------ | ------------------------------- | ----------------------- |
| Best Model   | `models/best_model.pt`          | Model terbaik           |
| Backup Model | `models/best_model_backup_*.pt` | Backup model sebelumnya |
| Checkpoint   | `runs/detect/train/weights/`    | Training checkpoints    |
| Metrics      | `runs/detect/train/results.csv` | Training metrics        |
| Plots        | `runs/detect/train/*.png`       | Visualisasi training    |

---

## 📚 Teknologi yang Digunakan

- **YOLOv8** - Algoritma deteksi objek real-time
- **PyTorch** - Deep learning framework
- **Ultralytics** - YOLOv8 implementation
- **Streamlit** - Web application framework
- **FastAPI** - REST API framework
- **OpenCV** - Computer vision library

---

## 🤝 Contributing

1. Fork repository
2. Buat branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📄 License

MIT License - Lihat [LICENSE](LICENSE) untuk detail.

---

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Streamlit](https://streamlit.io/)
- [FastAPI](https://fastapi.tiangolo.com/)
- Dataset: [garbage-classification-v2](https://www.kaggle.com/datasets)

---

<div align="center">

### ♻️ Klasifikasi Sampah Anorganik Menggunakan Algoritma YOLO

**Bantu jaga lingkungan dengan membuang sampah pada tempatnya! 🌍**

Made with ❤️ for a cleaner world

</div>
