# 🎯 Presentasi Project (30 Menit)

# Klasifikasi Sampah Menggunakan YOLO

---

## ⏱️ Alokasi Waktu (30 Menit)

| Bagian                       | Durasi  | Slide |
| ---------------------------- | ------- | ----- |
| Pendahuluan & Latar Belakang | 3 menit | 1-2   |
| Metodologi & Teknologi       | 4 menit | 3-4   |
| Dataset & Training           | 5 menit | 5-6   |
| Hasil & Evaluasi             | 4 menit | 7     |
| Demo Aplikasi                | 8 menit | 8     |
| Kesimpulan & Saran           | 3 menit | 9-10  |
| Q&A                          | 3 menit | 11    |

---

# Slide 1: Pendahuluan

## ♻️ Klasifikasi Sampah Anorganik dengan YOLO

```
   🗑️  →  📷  →  🤖  →  ♻️
 Sampah   Kamera  AI/YOLO  Terpilah
```

**Disusun oleh:** [Nama] | [NIM] | [Prodi]

### Latar Belakang:

- **68.5 Juta Ton** sampah/tahun di Indonesia
- **7.2%** tingkat daur ulang (sangat rendah)
- Pemilahan manual tidak efisien
- **Solusi:** Sistem AI untuk klasifikasi otomatis

---

# Slide 2: Tujuan & Rumusan Masalah

## 🎯 Tujuan Project

1. Membangun model **YOLO** untuk deteksi **10 jenis sampah**
2. Mengembangkan **3 platform**: Web App, REST API, Realtime Detection
3. Memberikan **edukasi** cara pembuangan yang benar

## ❓ Rumusan Masalah

- Bagaimana membangun sistem klasifikasi sampah otomatis?
- Berapa tingkat akurasi model yang dapat dicapai?
- Bagaimana implementasi ke aplikasi yang user-friendly?

---

# Slide 3: Metodologi

## 📐 Alur Pengembangan

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ 1.Data   │──▶│ 2.Prepro │──▶│ 3.Train  │──▶│ 4.Deploy │
│ Collection│   │ cessing  │   │  Model   │   │   Apps   │
└──────────┘   └──────────┘   └──────────┘   └──────────┘
```

| Tahap | Aktivitas                  | Output          |
| ----- | -------------------------- | --------------- |
| 1     | Kumpul dataset dari Kaggle | Raw images      |
| 2     | Konversi ke YOLO format    | Labeled dataset |
| 3     | Transfer learning YOLOv8   | Model (.pt)     |
| 4     | Web, API, Realtime         | 3 Aplikasi      |

---

# Slide 4: Teknologi

## 🛠️ Technology Stack

| Kategori             | Teknologi                     |
| -------------------- | ----------------------------- |
| **Deep Learning**    | YOLOv8 (Ultralytics), PyTorch |
| **Image Processing** | OpenCV, Pillow                |
| **Web Framework**    | Streamlit, FastAPI            |
| **Language**         | Python 3.8+                   |

### Kenapa YOLOv8?

- ⚡ **Cepat**: Real-time (30+ FPS)
- 🎯 **Akurat**: State-of-the-art mAP
- 📚 **Mudah**: API sederhana, dokumentasi lengkap

---

# Slide 5: Dataset

## 📊 10 Kelas Sampah

| Kelas         | Kategori  | Kelas      | Kategori  |
| ------------- | --------- | ---------- | --------- |
| 🔋 Battery    | B3        | 🥤 Plastic | Anorganik |
| 🥬 Biological | Organik   | 👟 Shoes   | Anorganik |
| 📦 Cardboard  | Anorganik | 🗑️ Trash   | Residu    |
| 👕 Clothes    | Anorganik | 🫙 Glass    | Anorganik |
| 🥫 Metal      | Anorganik | 📄 Paper   | Anorganik |

### Distribusi Data:

| Split      | Jumlah  | Persentase |
| ---------- | ------- | ---------- |
| Training   | ~13,000 | 80%        |
| Validation | ~1,600  | 10%        |
| Testing    | ~1,600  | 10%        |

---

# Slide 6: Training Model

## 🎓 Konfigurasi Training

```python
model.train(
    data='data.yaml',
    epochs=120,        # Iterasi
    batch=20,          # Batch size
    imgsz=640,         # Input size
    patience=10,       # Early stopping
    optimizer='AdamW', # Optimizer
    pretrained=True,   # Transfer learning
)
```

### Transfer Learning Flow:

```
YOLOv8 (COCO - 80 classes)
         │
         ▼ Fine-tune
Custom Model (10 waste classes)
```

---

# Slide 7: Hasil Training

## 📈 Evaluasi Model

| Metrik        | Nilai | Keterangan             |
| ------------- | ----- | ---------------------- |
| **mAP@50**    | ~85%  | Deteksi dengan IoU 0.5 |
| **mAP@50-95** | ~65%  | Deteksi strict         |
| **Precision** | ~82%  | Ketepatan prediksi     |
| **Recall**    | ~78%  | Kelengkapan deteksi    |
| **FPS**       | 25-30 | Kecepatan realtime     |

### Per-Class Top Performers:

| Kelas     | mAP@50 |
| --------- | ------ |
| Cardboard | 90%    |
| Paper     | 89%    |
| Battery   | 88%    |

---

# Slide 8: Demo Aplikasi

## 🎬 3 Platform Implementasi

### 1️⃣ Web Application (Streamlit)

```bash
streamlit run web_app.py
# → http://localhost:8501
```

- Upload gambar → Lihat hasil deteksi
- Info pembuangan sampah

### 2️⃣ REST API (FastAPI)

```bash
python api.py
# → http://localhost:8000/docs
```

- Endpoint: `POST /detect`
- Response: JSON dengan class, confidence, bbox

### 3️⃣ Realtime Detection (OpenCV)

```bash
python detect.py
# → Webcam detection
```

- Live webcam detection
- Tekan 'Q' untuk keluar

---

# Slide 9: Kesimpulan

## ✅ Hasil yang Dicapai

1. **Model YOLOv8** berhasil dilatih dengan **mAP@50 ~85%**
2. **3 Platform** berhasil dikembangkan:
   - Web App (Streamlit) ✓
   - REST API (FastAPI) ✓
   - Realtime Detection (OpenCV) ✓
3. Sistem memberikan **edukasi pembuangan** per kategori

## ⚠️ Keterbatasan

| Keterbatasan     | Dampak                   |
| ---------------- | ------------------------ |
| Dataset terbatas | Generalisasi kurang      |
| 10 kelas saja    | Belum semua jenis sampah |
| GPU recommended  | Lambat di CPU            |

---

# Slide 10: Saran Pengembangan

## 🚀 Pengembangan Selanjutnya

### Jangka Pendek:

- 📊 Tambah data training (5000+ per class)
- 🏷️ Tambah kelas baru (e-waste, medical)
- 📱 Export ke mobile (TFLite/ONNX)

### Jangka Panjang:

- 🗑️ **Smart Bin** - Tempat sampah pintar
- 🏭 **Conveyor Sorting** - Sorting otomatis di TPA
- 📊 **Analytics Dashboard** - Monitoring kota

---

# Slide 11: Q&A

```
┌─────────────────────────────────────────┐
│                                         │
│           ❓ QUESTIONS?                │
│                                         │
│   • Metodologi                         │
│   • Implementasi teknis                │
│   • Hasil & evaluasi                   │
│   • Pengembangan lanjut                │
│                                         │
│           🙏 Terima Kasih              │
│                                         │
└─────────────────────────────────────────┘
```

---

# 📎 Quick Reference

## Cara Menjalankan:

```bash
# Web App
streamlit run web_app.py

# REST API
python api.py

# Realtime
python detect.py
```

## Struktur Utama:

```
├── train.py          # Training
├── detect.py         # Realtime
├── web_app.py        # Web app
├── api.py            # REST API
├── models/best_model.pt
└── datasets/processed/
```

---

**© 2024 - Klasifikasi Sampah dengan YOLO** ♻️
