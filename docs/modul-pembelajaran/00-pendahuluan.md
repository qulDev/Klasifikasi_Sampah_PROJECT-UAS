# 📚 Modul Pembelajaran: Klasifikasi Sampah Menggunakan YOLO

## Pendahuluan

Selamat datang di modul pembelajaran project **Klasifikasi Sampah Anorganik Menggunakan Algoritma YOLO**! Modul ini dirancang untuk membantu Anda memahami secara mendalam bagaimana sistem deteksi objek berbasis deep learning bekerja, mulai dari persiapan data hingga deployment aplikasi.

---

## 🎯 Tujuan Pembelajaran

Setelah mempelajari modul ini, Anda diharapkan mampu:

1. **Memahami konsep dasar Object Detection** dan bagaimana YOLO bekerja
2. **Menyiapkan dan mengkonversi dataset** dari berbagai format ke format YOLO
3. **Melatih model YOLOv8** untuk deteksi dan klasifikasi objek
4. **Membangun aplikasi web** menggunakan Streamlit
5. **Membuat REST API** menggunakan FastAPI
6. **Melakukan real-time detection** menggunakan webcam
7. **Memahami arsitektur project** dan best practices dalam pengembangan ML

---

## 📋 Daftar Modul

| No  | Modul                                                    | Deskripsi                              | Tingkat     |
| --- | -------------------------------------------------------- | -------------------------------------- | ----------- |
| 01  | [Pengenalan YOLO](./01-pengenalan-yolo.md)               | Konsep dasar object detection dan YOLO | Pemula      |
| 02  | [Arsitektur Project](./02-arsitektur-project.md)         | Struktur folder dan file dalam project | Pemula      |
| 03  | [Persiapan Dataset](./03-persiapan-dataset.md)           | Konversi dan persiapan data training   | Menengah    |
| 04  | [Training Model](./04-training-model.md)                 | Melatih model YOLOv8                   | Menengah    |
| 05  | [Web Application](./05-web-application.md)               | Membangun UI dengan Streamlit          | Menengah    |
| 06  | [REST API](./06-rest-api.md)                             | Membuat API dengan FastAPI             | Menengah    |
| 07  | [Real-time Detection](./07-realtime-detection.md)        | Deteksi objek secara real-time         | Lanjutan    |
| 08  | [Utility Functions](./08-utility-functions.md)           | Penjelasan modul utilitas              | Lanjutan    |
| 09  | [Tips dan Troubleshooting](./09-tips-troubleshooting.md) | Solusi masalah umum                    | Semua Level |
| 10  | [Referensi](./10-referensi.md)                           | Referensi dan resource tambahan        | Semua Level |

---

## 🛠️ Prasyarat Teknis

### Hardware Minimum

- **CPU**: Intel i5 atau AMD Ryzen 5 (atau lebih tinggi)
- **RAM**: 8 GB (16 GB direkomendasikan)
- **GPU**: NVIDIA dengan CUDA support (opsional, tapi sangat direkomendasikan)
- **Storage**: 10 GB ruang kosong

### Software

- **Python**: Versi 3.8 atau lebih tinggi
- **CUDA Toolkit**: Versi 11.0+ (jika menggunakan GPU)
- **cuDNN**: Versi yang kompatibel dengan CUDA
- **Git**: Untuk version control

### Pengetahuan Dasar

- Pemrograman Python dasar
- Konsep dasar machine learning (opsional)
- Command line / terminal

---

## 🗑️ Tentang Dataset

Project ini menggunakan dataset **garbage-classification-v2** yang terdiri dari **10 kelas sampah**:

| ID  | Kelas      | Emoji | Kategori       | Tempat Sampah |
| --- | ---------- | ----- | -------------- | ------------- |
| 0   | battery    | 🔋    | B3 (Berbahaya) | 🔴 Merah      |
| 1   | biological | 🥬    | Organik        | 🟢 Hijau      |
| 2   | cardboard  | 📦    | Anorganik      | 🔵 Biru       |
| 3   | clothes    | 👕    | Tekstil        | 🔵 Biru       |
| 4   | glass      | 🍾    | Anorganik      | 🔵 Biru       |
| 5   | metal      | 🥫    | Anorganik      | 🔵 Biru       |
| 6   | paper      | 📄    | Anorganik      | 🔵 Biru       |
| 7   | plastic    | 🥤    | Anorganik      | 🟡 Kuning     |
| 8   | shoes      | 👟    | Tekstil        | 🔵 Biru       |
| 9   | trash      | 🗑️    | Residu         | ⚫ Hitam      |

---

## 📊 Gambaran Umum Alur Kerja

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE KLASIFIKASI SAMPAH                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │   Dataset    │───▶│  Konversi    │───▶│    Split     │               │
│  │     Raw      │    │  ke YOLO     │    │ Train/Val/Test│              │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
│         │                                       │                        │
│         │            convert_datasets.py        │                        │
│         │            split_and_prep.py          │                        │
│         │                                       ▼                        │
│         │                              ┌──────────────┐                  │
│         │                              │   Training   │                  │
│         │                              │   YOLOv8     │                  │
│         │                              └──────────────┘                  │
│         │                                       │                        │
│         │              train.py                 │                        │
│         │                                       ▼                        │
│         │                              ┌──────────────┐                  │
│         │                              │    Model     │                  │
│         │                              │  best_model.pt│                 │
│         │                              └──────────────┘                  │
│         │                                       │                        │
│         │                                       ▼                        │
│  ┌──────┴──────────────────────────────────────┴──────────────────┐     │
│  │                     DEPLOYMENT OPTIONS                          │     │
│  ├─────────────────┬─────────────────┬─────────────────────────────┤     │
│  │   Web App       │   REST API      │   Real-time Detection       │     │
│  │  (Streamlit)    │   (FastAPI)     │      (OpenCV)               │     │
│  │   web_app.py    │    api.py       │      detect.py              │     │
│  └─────────────────┴─────────────────┴─────────────────────────────┘     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

Jika Anda ingin langsung mencoba project ini, ikuti langkah-langkah berikut:

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/Klasifikasi_Sampah.git
cd Klasifikasi_Sampah
```

### 2. Buat Virtual Environment

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
# Jika menggunakan GPU (CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install requirements
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi

**Web App (Recommended):**

```bash
streamlit run web_app.py
```

**REST API:**

```bash
uvicorn api:app --reload --port 8000
```

**Real-time Detection:**

```bash
python detect.py
```

---

## 📁 Struktur Modul

Setiap modul dalam dokumentasi ini akan membahas:

1. **Teori dan Konsep** - Penjelasan teoritis tentang topik
2. **Implementasi Kode** - Penjelasan line-by-line kode
3. **Diagram dan Visualisasi** - Ilustrasi untuk mempermudah pemahaman
4. **Latihan** - Soal latihan untuk menguji pemahaman
5. **Referensi** - Link ke dokumentasi resmi dan resource tambahan

---

## 📝 Catatan Penting

> ⚠️ **Perhatian**: Modul ini ditulis berdasarkan versi project per Januari 2026. Beberapa API atau library mungkin berubah di versi mendatang.

> 💡 **Tips**: Direkomendasikan untuk mempelajari modul secara berurutan untuk pemahaman yang optimal.

> 🔧 **Hands-on**: Setiap modul dilengkapi dengan contoh kode yang bisa langsung dipraktikkan.

---

## 🤝 Kontribusi

Jika Anda menemukan kesalahan atau ingin menambahkan materi, silakan:

1. Buat issue di repository
2. Submit pull request dengan perbaikan
3. Diskusikan di section discussions

---

## 📞 Bantuan

Jika Anda mengalami kesulitan:

1. Baca modul [Tips dan Troubleshooting](./09-tips-troubleshooting.md)
2. Cek FAQ di bagian akhir setiap modul
3. Buat issue di repository GitHub

---

**Selamat Belajar! 🎉**

_Lanjut ke: [Modul 01 - Pengenalan YOLO](./01-pengenalan-yolo.md)_
