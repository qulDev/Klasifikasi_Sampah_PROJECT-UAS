# 🎓 Penjelasan Lengkap Program Klasifikasi Sampah

## Untuk Pemula - Dijelaskan Seperti Mengajar Anak 12 Tahun

---

## 📚 Daftar Isi

1. [Apa Itu Program Ini?](#apa-itu-program-ini)
2. [Cara Kerja Keseluruhan](#cara-kerja-keseluruhan)
3. [Penjelasan Setiap Langkah](#penjelasan-setiap-langkah)
4. [Penjelasan Algoritma](#penjelasan-algoritma)
5. [Analogi Sederhana](#analogi-sederhana)

---

## 🤔 Apa Itu Program Ini?

Bayangkan kamu punya robot pintar yang bisa melihat sampah dan langsung tahu itu sampah apa. 

**Program ini adalah robot pintar tersebut!**

Program ini bisa:
- 👁️ **Melihat** foto atau video sampah
- 🧠 **Berpikir** dan mengenali jenis sampahnya
- 🏷️ **Memberi label**: "Ini plastik!", "Ini metal!", "Ini kaca!"
- 📦 **Menggambar kotak** di sekitar sampah yang ditemukan
- ⚡ **Bekerja cepat** menggunakan kartu grafis komputer (GPU)

**Jenis sampah yang bisa dikenali:**
1. 🟡 **Plastik** (botol, kantong, wrapper)
2. 🔵 **Metal** (kaleng, aluminium)
3. 🟢 **Kaca** (botol kaca, jar)
4. 🔵 **Kertas** (koran, majalah)
5. 🟠 **Kardus** (kotak kardus)
6. ⚫ **Lainnya** (sampah yang tidak termasuk kategori di atas)

---

## 🔄 Cara Kerja Keseluruhan

Seperti memasak makanan, program ini punya **7 langkah besar**:

```
📂 Dataset Mentah (Foto-foto sampah acak)
    ↓
1️⃣ KONVERSI: Rapikan semua foto jadi format yang sama
    ↓
2️⃣ DEDUPLIKASI: Buang foto yang sama/duplikat
    ↓
3️⃣ PEMBAGIAN: Bagi foto jadi 3 kelompok (latihan, validasi, tes)
    ↓
4️⃣ PELATIHAN: Ajari komputer mengenali sampah
    ↓
5️⃣ MODEL: Simpan "otak" yang sudah pintar
    ↓
6️⃣ DETEKSI: Gunakan "otak" untuk mengenali sampah baru
    ↓
7️⃣ REAL-TIME: Gunakan kamera untuk deteksi langsung
```

Mari kita jelaskan **SATU PER SATU** dengan sangat detail!

---

## 📖 Penjelasan Setiap Langkah

### 🔧 Langkah 0: Persiapan Awal (setup.sh)

**Analogi:** Seperti menyiapkan meja belajar sebelum belajar.

**Yang dilakukan:**
1. **Membuat ruang kerja virtual** (virtual environment)
   - Seperti membuat kamar khusus untuk belajar
   - Semua alat (library) disimpan di sini
   - Tidak mengganggu program lain di komputer

2. **Menginstall alat-alat** (pip install)
   - PyTorch: Untuk berpikir seperti otak manusia
   - OpenCV: Untuk melihat dan memproses gambar
   - Ultralytics YOLO: Robot pendeteksi objek
   - Dan lain-lain

3. **Membuat folder-folder**
   ```
   datasets/         ← Tempat foto sampah
   models/          ← Tempat "otak" yang sudah pintar
   logs/            ← Catatan apa yang terjadi
   ```

**Kenapa penting?**
- Tanpa ini, program tidak bisa jalan
- Seperti pergi sekolah tanpa buku dan pensil

---

### 📂 Langkah 1: Konversi Dataset (convert_datasets.py)

**Analogi:** Seperti merapikan buku-buku dari berbagai penerbit agar semua formatnya sama.

#### 🤔 Masalahnya Apa?

Foto sampah datang dari berbagai sumber:
- Ada yang dari Kaggle (format COCO)
- Ada yang dari GitHub (format VOC)
- Ada yang sudah YOLO
- Ada yang cuma folder biasa

**Seperti punya buku:**
- Ada buku cerita
- Ada komik
- Ada ensiklopedia
- Semua beda formatnya!

#### ✨ Solusinya?

Program ini **mengubah semua jadi format yang sama** yaitu **YOLO**.

#### 📝 Format YOLO Itu Apa?

Contoh file label untuk 1 foto:

**Foto:** `botol_plastik.jpg`  
**Label:** `botol_plastik.txt`

Isi file txt:
```
0 0.5 0.5 0.3 0.4
```

Artinya:
- `0` = Kelas 0 (plastik)
- `0.5` = Posisi tengah X (50% dari kiri)
- `0.5` = Posisi tengah Y (50% dari atas)
- `0.3` = Lebar kotak (30% dari lebar foto)
- `0.4` = Tinggi kotak (40% dari tinggi foto)

#### 🔍 Algoritma Konversi

**Langkah-langkah:**

1. **Deteksi Format**
   ```
   Baca folder → Lihat isi file → Tebak formatnya
   
   Kalau ada file .json → COCO
   Kalau ada file .xml → VOC
   Kalau ada file .txt → YOLO
   Kalau cuma folder → Class Folders
   ```

2. **Mapping Label (Pencocokan Nama)**
   
   Ini **SANGAT PENTING**!
   
   **Masalah:**
   - Dataset A bilang "plastic_bottle"
   - Dataset B bilang "PET_container"
   - Dataset C bilang "plastik"
   
   Padahal **SEMUA ITU PLASTIK!**
   
   **Solusi - 6 Cara Pencocokan:**
   
   a. **Exact Match (Persis)**
   ```python
   "plastic" == "plastic" ✓ (100% yakin)
   ```
   
   b. **Fuzzy Match (Mirip)**
   ```python
   "platic" mirip "plastic" ✓ (90% yakin)
   "plastik" mirip "plastic" ✓ (85% yakin)
   ```
   Menggunakan **algoritma Levenshtein**:
   - Hitung berapa huruf yang beda
   - Kalau cuma beda sedikit = mirip
   
   c. **Substring Match (Ada di dalam)**
   ```python
   "plastic" ada di "plastic_bottle_clear" ✓ (95% yakin)
   ```
   
   d. **Keyword Match (Kata Kunci)**
   ```python
   "aluminum_can" mengandung "aluminum"
   "aluminum" termasuk keyword metal ✓ (90% yakin)
   ```
   
   e. **Manual Override**
   ```python
   Kamu bilang: "biodegradable" = "other"
   Program: OK! (100% yakin)
   ```
   
   f. **Fallback (Tidak tahu)**
   ```python
   Tidak cocok sama sekali → Kategori "other" (50% yakin)
   ```

3. **Konversi Koordinat**
   
   **Dari COCO (piksel absolut):**
   ```
   Kotak: x=100, y=150, lebar=200, tinggi=300
   Ukuran foto: 640x480
   
   Hitung posisi tengah:
   center_x = (100 + 200/2) / 640 = 0.3125
   center_y = (150 + 300/2) / 480 = 0.625
   
   Hitung ukuran relatif:
   width = 200 / 640 = 0.3125
   height = 300 / 480 = 0.625
   
   Hasil YOLO: 0 0.3125 0.625 0.3125 0.625
   ```
   
   **Kenapa pakai angka 0-1?**
   - Biar foto besar atau kecil tetap sama
   - Seperti pakai persen, bukan sentimeter

4. **Verifikasi Gambar**
   
   Cek setiap foto:
   ```python
   Apakah file rusak? ✗
   Apakah bisa dibuka? ✓
   Apakah format valid? ✓
   ```

5. **Salin File**
   ```
   Dataset A/foto1.jpg → Processed/all/images/foto1.jpg
   Dataset A/foto1.xml → Processed/all/labels/foto1.txt
   Dataset B/foto2.jpg → Processed/all/images/foto2.jpg
   ...
   ```

**Output:**
```
datasets/processed/all/
    images/
        foto1.jpg
        foto2.jpg
        ...
    labels/
        foto1.txt
        foto2.txt
        ...
```

---

### 🔀 Langkah 2: Split Dataset (split_and_prep.py)

**Analogi:** Seperti membagi kartu flashcard jadi 3 tumpukan: buat belajar, buat latihan, buat ujian.

#### 🎯 Tujuan

Bagi foto jadi 3 kelompok:
- **70%** untuk **latihan** (train) - Belajar mengenali
- **15%** untuk **validasi** (val) - Cek apakah sudah paham
- **15%** untuk **tes** (test) - Ujian akhir

#### 🧹 Algoritma Deduplikasi (Buang Duplikat)

**Masalah:** Ada foto yang sama tapi nama file beda!

**Solusi: Hashing**

1. **Baca foto**
2. **Hitung "sidik jari" foto** (hash SHA256)
   ```python
   Foto A: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   Foto B: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   
   Hash sama = Foto yang sama! Buang salah satu.
   ```

**Analogi:**
- Seperti sidik jari manusia
- Foto yang sama = sidik jari sama
- Foto berbeda = sidik jari berbeda

#### 📊 Algoritma Stratified Split

**Masalah:** Kalau bagi acak, bisa jadi:
- Train: 90% plastik, 10% lainnya ❌
- Validation: 20% plastik, 80% lainnya ❌
- Tidak seimbang!

**Solusi: Stratified (Seimbang)**

```python
Punya 1000 foto:
- 300 plastik (30%)
- 250 metal (25%)
- 200 kaca (20%)
- 150 kertas (15%)
- 100 lainnya (10%)

Split 70-15-15:

Train (700 foto):
- 210 plastik (30%)
- 175 metal (25%)
- 140 kaca (20%)
- 105 kertas (15%)
- 70 lainnya (10%)

Validation (150 foto):
- 45 plastik (30%)
- 37 metal (25%)
- 30 kaca (20%)
- 23 kertas (15%)
- 15 lainnya (10%)

Test (150 foto):
- 45 plastik (30%)
- 38 metal (25%)
- 30 kaca (20%)
- 22 kertas (15%)
- 15 lainnya (10%)

Semua proporsi sama! ✓
```

**Algoritma:**
```python
Untuk setiap kelas:
    Ambil semua foto kelas itu
    Acak urutannya
    Bagi 70-15-15
    Masukkan ke train/val/test
```

#### 📄 Membuat data.yaml

File konfigurasi untuk YOLO:

```yaml
train: /path/to/train/images
val: /path/to/val/images
test: /path/to/test/images

nc: 6  # Jumlah kelas
names: ['plastic', 'metal', 'glass', 'paper', 'cardboard', 'other']
```

**Seperti daftar isi buku** - kasih tahu YOLO di mana foto-fotonya.

---

### 🏋️ Langkah 3: Pelatihan Model (train.py)

**Analogi:** Mengajari anak kecil mengenali binatang dengan flashcard berulang kali.

#### 🧠 Apa Itu YOLOv8?

**YOLO** = **Y**ou **O**nly **L**ook **O**nce

Artinya: Cukup lihat foto sekali, langsung tahu ada apa aja!

**Berbeda dengan metode lama:**
```
Metode lama:
1. Lihat kiri atas → Ada apa?
2. Lihat kanan atas → Ada apa?
3. Lihat tengah → Ada apa?
4. ... (lama banget!)

YOLO:
1. Lihat seluruh foto → Langsung tahu semua objek! ⚡
```

#### 🎓 Cara Belajarnya

**Seperti anak belajar:**

1. **Lihat foto sampah**
   ```
   Guru (Label): "Ini plastik"
   Murid (Model): "Menurutku ini... metal?"
   Guru: "Salah! Ini plastik!"
   ```

2. **Hitung kesalahan**
   ```
   Berapa salahnya?
   Tebakan: metal (kelas 1)
   Jawaban: plastik (kelas 0)
   Error: Besar!
   ```

3. **Perbaiki otak**
   ```
   Ubah koneksi di otak (weights)
   Supaya lain kali lebih benar
   ```

4. **Ulangi ribuan kali**
   ```
   Foto 1 → Belajar
   Foto 2 → Belajar
   ...
   Foto 1000 → Belajar
   
   Ulangi lagi (epoch 2)
   Foto 1 → Belajar lagi
   ...
   
   Sampai 100 epoch!
   ```

#### 🔢 Algoritma Lengkap

**1. Inisialisasi (Persiapan)**

```python
# Buat model baru atau load yang sudah ada
model = YOLO('yolov8s.pt')  # s = small, ada juga n,m,l,x

# Cek GPU
if GPU ada:
    Pakai GPU (lebih cepat 10x)
else:
    Pakai CPU (lebih lambat)
```

**2. Forward Pass (Tebak)**

Untuk setiap foto:

```python
# Foto masuk ke jaringan saraf
Input: Foto 640x640 pixel

Layer 1: Konvolusi (lihat pola kecil)
    → Deteksi tepi, sudut, garis
    
Layer 2-10: Konvolusi lebih dalam
    → Deteksi bentuk (bulat, kotak)
    
Layer 11-20: Konvolusi makin dalam
    → Deteksi fitur kompleks (tutup botol, logo)
    
Layer 21-25: Detection Head
    → Prediksi: "Ada plastik di (0.3, 0.5) ukuran (0.2, 0.3)"

Output: 
    - Kelas: plastik (confidence 0.85)
    - Kotak: x=0.3, y=0.5, w=0.2, h=0.3
```

**3. Loss Calculation (Hitung Kesalahan)**

```python
Prediksi:
    Kelas: metal (0.6)
    Kotak: (0.32, 0.48, 0.19, 0.28)

Ground Truth (Jawaban benar):
    Kelas: plastic
    Kotak: (0.30, 0.50, 0.20, 0.30)

Hitung error:

1. Classification Loss (Salah kelas):
   Prediksi metal tapi harusnya plastic
   Loss = 0.8 (besar!)

2. Localization Loss (Salah posisi):
   Kotak sedikit meleset
   Loss = 0.05 (kecil)

Total Loss = 0.8 + 0.05 = 0.85
```

**4. Backpropagation (Belajar dari Kesalahan)**

```python
Loss = 0.85 (besar = salah banyak)

Hitung gradien (arah perbaikan):
    Layer 25 → Ubah -0.01
    Layer 24 → Ubah -0.009
    ...
    Layer 1 → Ubah -0.001

Update weights (perbaiki otak):
    Weight baru = Weight lama - (learning_rate × gradient)
    
Contoh:
    Weight lama = 0.534
    Learning rate = 0.001
    Gradient = 0.8
    Weight baru = 0.534 - (0.001 × 0.8) = 0.5332
```

**5. Validation (Cek Pemahaman)**

Setiap epoch, cek pakai foto validasi:

```python
Hitung akurasi:
    Benar: 850 foto
    Salah: 150 foto
    Akurasi: 850/1000 = 85%

Hitung mAP (mean Average Precision):
    Seberapa tepat kotak yang digambar?
    mAP = 0.78 (78% tepat)
```

**6. Early Stopping**

```python
Epoch 1: mAP = 0.45
Epoch 2: mAP = 0.53 ↑ (membaik!)
Epoch 3: mAP = 0.61 ↑
...
Epoch 50: mAP = 0.89 ↑
Epoch 51: mAP = 0.89 - (sama)
Epoch 52: mAP = 0.88 ↓ (turun!)
Epoch 53: mAP = 0.87 ↓

Sudah 3 epoch turun terus → STOP!
Pakai model epoch 50 (yang terbaik)
```

#### ⚙️ Optimizer: AdamW

**Analogi:** Cara jalan menuruni gunung mencari lembah terdalam.

```python
Learning Rate (Kecepatan langkah):
    Awal: Besar (0.01) → Langkah cepat
    Tengah: Sedang (0.001) → Mulai hati-hati
    Akhir: Kecil (0.0001) → Langkah pelan

Momentum:
    Seperti bola menggelinding
    Kalau turun terus → Makin cepat
    Kalau naik → Melambat

Weight Decay:
    Jangan terlalu kompleks
    Cegah overfitting (hafalan)
```

#### 💾 Menyimpan Model

```python
Setiap 10 epoch:
    Simpan checkpoint: epoch_10.pt, epoch_20.pt, ...

Epoch terbaik:
    Simpan sebagai: best.pt

Epoch terakhir:
    Simpan sebagai: last.pt

Best model → Copy ke models/20251106_123045_best.pt
           → Buat symlink: models/best.pt
```

---

### 🔍 Langkah 4: Deteksi Gambar (test_detection.py)

**Analogi:** Ujian! Tunjukkan foto baru, model harus bisa jawab.

#### 🎯 Algoritma Inference

**1. Load Model**

```python
model = YOLO('models/best.pt')

# Load weights yang sudah dilatih
# Seperti load "otak" yang sudah pintar
```

**2. Preprocessing (Persiapan Foto)**

```python
Input: foto asli (1920x1080)

Resize ke 640x640:
    1. Pertahankan aspect ratio
    2. Tambah padding (abu-abu) kalau perlu
    
Normalize pixel values:
    Dari: 0-255
    Jadi: 0-1
    
    Contoh: Pixel value 127 → 127/255 = 0.498

Convert ke tensor:
    Format yang dipahami PyTorch
```

**3. Forward Pass (Deteksi)**

```python
results = model(foto)

# Jalankan foto di jaringan saraf
# Tanpa update weights (cuma lihat, tidak belajar)

Output: List of detections
    [
        {
            'class': 0,  # plastic
            'confidence': 0.95,
            'bbox': [100, 150, 300, 400]
        },
        {
            'class': 1,  # metal
            'confidence': 0.87,
            'bbox': [500, 200, 600, 350]
        }
    ]
```

**4. Post-processing (Rapihkan Hasil)**

**a. Confidence Filtering**

```python
Threshold = 0.25

Semua deteksi:
    Det 1: confidence 0.95 ✓ (> 0.25)
    Det 2: confidence 0.87 ✓
    Det 3: confidence 0.12 ✗ (< 0.25, buang!)
    Det 4: confidence 0.54 ✓

Hasil: 3 deteksi valid
```

**b. Non-Maximum Suppression (NMS)**

```python
Masalah: Satu objek terdeteksi berkali-kali!

Kotak A: confidence 0.95
Kotak B: confidence 0.87
Kedua kotak overlap 80%!

Solusi NMS:
1. Urutkan berdasarkan confidence
   [A(0.95), B(0.87)]

2. Ambil yang tertinggi (A)
   Simpan A

3. Hapus kotak yang overlap > 50% dengan A
   B overlap 80% dengan A → Buang B!

4. Ulangi untuk kotak berikutnya

Hasil: 1 kotak per objek
```

**5. Draw Results (Gambar Kotak)**

```python
Untuk setiap deteksi:
    1. Gambar kotak berwarna
       cv2.rectangle(foto, (x1,y1), (x2,y2), warna, thickness)
    
    2. Tulis label
       "PLASTIC: 0.95"
    
    3. Beri nomor
       Gambar lingkaran: "#1"

Simpan hasil: output.jpg
```

---

### 🎥 Langkah 5: Deteksi Real-Time (realtime_detect.py)

**Analogi:** Seperti mata manusia yang terus menerus melihat dan mengenali objek.

#### 📹 Algoritma Real-Time

**1. Setup Kamera**

```python
# Buka kamera (webcam)
cap = cv2.VideoCapture(0)  # 0 = kamera default

# Set resolusi
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)  # 30 frame per detik
```

**2. Loop Deteksi**

```python
while True:  # Loop forever sampai user quit
    # Baca 1 frame dari kamera
    ret, frame = cap.read()
    
    if not ret:
        print("Kamera error!")
        break
    
    # Deteksi objek di frame ini
    results = model(frame, conf=0.25)
    
    # Gambar kotak dan label
    frame = draw_detections(frame, results)
    
    # Hitung FPS
    fps = hitung_fps()
    
    # Tampilkan info panel
    frame = draw_info_panel(frame, fps)
    
    # Tampilkan ke layar
    cv2.imshow('Deteksi Real-Time', frame)
    
    # Cek keyboard
    key = cv2.waitKey(1)
    if key == ord('q'):  # Tekan Q untuk quit
        break
    elif key == ord('s'):  # Tekan S untuk save
        cv2.imwrite('capture.jpg', frame)
```

**3. Optimisasi Kecepatan**

**Menggunakan GPU (CUDA):**

```python
CPU: 5-8 FPS (lambat)
    Proses: CPU → Foto → Deteksi → Hasil
    Waktu: ~150ms per frame

GPU: 30-60 FPS (cepat!)
    Proses: GPU → Foto → Deteksi → Hasil
    Waktu: ~20ms per frame
    
Kenapa lebih cepat?
    GPU punya ribuan core kecil
    Bisa proses banyak pixel sekaligus
    CPU cuma punya 4-8 core besar
```

**Frame Skipping (Opsional):**

```python
# Kalau masih lambat, skip beberapa frame

frame_count = 0
while True:
    ret, frame = cap.read()
    
    frame_count += 1
    if frame_count % 2 == 0:  # Deteksi setiap 2 frame
        results = model(frame)
    
    # Tampilkan setiap frame (meskipun tidak deteksi)
    cv2.imshow('window', frame)
```

**4. Tracking Objek (Sistem Penomoran)**

```python
Deteksi 1: #1 Plastic, #2 Metal
Deteksi 2: #1 Plastic, #2 Metal (sama!)
Deteksi 3: #1 Plastic, #2 Metal, #3 Glass (ada baru!)

Algoritma sederhana:
    Untuk setiap frame:
        Beri nomor 1, 2, 3, ... berdasarkan urutan deteksi

Algoritma canggih (tidak dipakai di sini):
    Tracking by IoU (Intersection over Union)
    Objek yang sama di frame berikutnya = ID sama
```

**5. UI Elements**

```python
Panel atas (Info):
    - Judul: "WASTE CLASSIFICATION - REAL-TIME"
    - FPS: "45.2"
    - Device: "CUDA:0"
    - Object count: "Objects: 3"
    - Controls: "Q: Quit | S: Save | C: Toggle"

Panel kanan (Summary):
    - "#1: PLASTIC"
    - "#2: METAL"
    - "#3: GLASS"

Kotak deteksi:
    - Kotak berwarna (thickness=3)
    - Lingkaran nomor "#1"
    - Label "PLASTIC: 0.95"
```

---

## 🧮 Penjelasan Algoritma dari Dasar

### 1. 🌳 Convolutional Neural Network (CNN)

**Analogi:** Seperti mata manusia melihat gambar.

**Cara Kerja:**

```
Input: Foto 640x640 pixel (RGB)
       = 640 × 640 × 3 = 1,228,800 angka!

Layer 1: Convolution 3x3
    Ambil filter 3x3 (seperti kaca pembesar kecil)
    Geser di seluruh foto
    
    Filter 1: Deteksi garis vertikal
    Filter 2: Deteksi garis horizontal
    Filter 3: Deteksi sudut
    ...
    Filter 32: Deteksi pola lain
    
    Output: 640x640x32

Layer 2: Pooling (Max Pooling)
    Kecilkan ukuran
    Ambil nilai maksimum setiap 2x2
    
    Input: 640x640x32
    Output: 320x320x32
    
    Kenapa? Kurangi komputasi, fokus ke fitur penting

Layer 3-10: Convolution + Pooling berulang
    Makin dalam, makin abstrak
    
    Layer awal: Deteksi garis, tepi
    Layer tengah: Deteksi bentuk, pola
    Layer akhir: Deteksi objek kompleks

Layer 11: Detection Head
    Prediksi kotak dan kelas
```

**Rumus Konvolusi:**

```
Filter 3x3:
    [1  0 -1]
    [1  0 -1]
    [1  0 -1]

Terapkan ke gambar:
    Pixel (x,y) = Σ(filter × pixel tetangga)

Contoh:
    Area 3x3 di gambar:
    [50  60  70]
    [40  50  60]
    [30  40  50]
    
    Konvolusi:
    = 1×50 + 0×60 + (-1)×70
      + 1×40 + 0×50 + (-1)×60
      + 1×30 + 0×40 + (-1)×50
    
    = 50 - 70 + 40 - 60 + 30 - 50
    = -60
    
    Hasil negatif → Ada perubahan intensitas → Deteksi tepi!
```

### 2. 🎯 YOLO Detection Algorithm

**Cara Kerja YOLO:**

```
1. Bagi gambar jadi grid 13x13
   
   +---+---+---+...+---+
   | 1 | 2 | 3 |...| 13|
   +---+---+---+...+---+
   | 14| 15| 16|...|  26|
   +---+---+---+...+---+
   ...
   
2. Setiap sel prediksi:
   - Apakah ada objek? (objectness score)
   - Kelas apa? (class probabilities)
   - Kotak di mana? (bounding box)

3. Setiap sel prediksi 3 kotak (anchors)
   - Kotak kecil (untuk objek kecil)
   - Kotak sedang
   - Kotak besar (untuk objek besar)

4. Total prediksi:
   13 × 13 × 3 = 507 kotak!

5. Filter dengan confidence > 0.25
   507 → 20 kotak

6. NMS (buang duplikat)
   20 → 5 kotak final
```

**Anchor Boxes:**

```
Masalah: Objek punya berbagai ukuran dan bentuk

Solusi: Predefine anchor boxes

Anchor 1: [10, 13]   (kecil, vertikal)
Anchor 2: [16, 30]   (kecil, horizontal)
Anchor 3: [33, 23]   (sedang, horizontal)
Anchor 4: [30, 61]   (besar, vertikal)
...

Model prediksi offset dari anchor:
    Prediksi: dx=+5, dy=+3, dw=+2, dh=+4
    Anchor: [30, 61]
    Kotak final: [35, 64]
```

### 3. 📊 Loss Function (Fungsi Kerugian)

**Komponen Loss YOLO:**

```python
Total Loss = λ₁×L_box + λ₂×L_obj + λ₃×L_cls

1. Box Loss (L_box):
   Seberapa akurat posisi kotak?
   
   L_box = Σ[(x_pred - x_true)² + (y_pred - y_true)² 
            + (w_pred - w_true)² + (h_pred - h_true)²]
   
   Contoh:
   Prediksi: (0.5, 0.5, 0.3, 0.4)
   Truth: (0.52, 0.48, 0.28, 0.42)
   
   L_box = (0.5-0.52)² + (0.5-0.48)² + (0.3-0.28)² + (0.4-0.42)²
         = 0.0004 + 0.0004 + 0.0004 + 0.0004
         = 0.0016

2. Objectness Loss (L_obj):
   Seberapa yakin ada objek?
   
   L_obj = -[y_true × log(y_pred) + (1-y_true) × log(1-y_pred)]
   
   Contoh:
   Prediksi: 0.8 (80% yakin ada objek)
   Truth: 1 (memang ada objek)
   
   L_obj = -[1 × log(0.8) + 0 × log(0.2)]
         = -log(0.8)
         = 0.097

3. Classification Loss (L_cls):
   Seberapa benar kelasnya?
   
   Cross-Entropy Loss per kelas
   
   Prediksi: [0.1, 0.7, 0.1, 0.05, 0.03, 0.02]
             (plastic, metal, glass, paper, cardboard, other)
   Truth: [1, 0, 0, 0, 0, 0]  (plastic)
   
   L_cls = -log(0.1) = 2.3
   
   Kalau prediksi benar:
   Prediksi: [0.9, 0.05, 0.02, 0.01, 0.01, 0.01]
   L_cls = -log(0.9) = 0.046 (kecil!)

Total Loss:
L_total = 1.0×0.0016 + 1.0×0.097 + 1.0×2.3
        = 2.3986
```

### 4. ⬇️ Backpropagation

**Analogi:** Seperti guru mengoreksi kesalahan murid dari belakang ke depan.

**Algoritma:**

```python
1. Forward Pass (Tebak):
   Input → Layer1 → Layer2 → ... → Output
   
2. Hitung Loss:
   L = Loss(Output, Truth)
   
3. Backward Pass (Koreksi):
   
   Untuk Layer terakhir:
   ∂L/∂W_last = Gradient loss terhadap weight
   
   Untuk Layer sebelumnya:
   Gunakan Chain Rule
   
   ∂L/∂W_n = ∂L/∂output × ∂output/∂W_n
   
4. Update Weights:
   W_new = W_old - learning_rate × ∂L/∂W

Contoh Sederhana:

Layer terakhir:
   W_old = 0.5
   ∂L/∂W = 0.8 (gradient besar = salah besar)
   LR = 0.01
   
   W_new = 0.5 - 0.01×0.8
         = 0.5 - 0.008
         = 0.492

Layer tengah:
   W_old = 0.3
   ∂L/∂W = 0.3 (gradient kecil)
   
   W_new = 0.3 - 0.01×0.3
         = 0.297

Gradient makin kecil untuk layer lebih awal
(Vanishing gradient problem)
```

### 5. 🔄 Optimizer: AdamW

**Komponen:**

```python
1. Momentum (m):
   Rata-rata gradien sebelumnya
   Seperti bola menggelinding
   
   m_t = β₁×m_(t-1) + (1-β₁)×gradient_t
   
2. Velocity (v):
   Rata-rata kuadrat gradien
   Untuk adaptive learning rate
   
   v_t = β₂×v_(t-1) + (1-β₂)×gradient_t²

3. Bias Correction:
   m̂ = m_t / (1 - β₁^t)
   v̂ = v_t / (1 - β₂^t)

4. Weight Update:
   W = W - α × m̂ / (√v̂ + ε)
   
   α = learning rate
   ε = small constant (1e-8)

Contoh:
   Epoch 1:
   gradient = 0.8
   m = 0.9×0 + 0.1×0.8 = 0.08
   v = 0.999×0 + 0.001×0.64 = 0.00064
   
   m̂ = 0.08 / (1-0.9¹) = 0.8
   v̂ = 0.00064 / (1-0.999¹) = 0.64
   
   W_new = W - 0.01 × 0.8 / (√0.64 + 1e-8)
         = W - 0.01 × 0.8 / 0.8
         = W - 0.01

Weight Decay (W di AdamW):
   W = W × (1 - λ)
   
   λ = weight decay rate (0.0001)
   
   Setiap update, weight dikurangi sedikit
   Cegah overfitting
```

### 6. 🎯 Non-Maximum Suppression (NMS)

**Algoritma Detail:**

```python
Input: List of detections
   [
       {box: [10,10,50,50], conf: 0.9, class: 0},
       {box: [12,12,48,48], conf: 0.85, class: 0},
       {box: [15,15,55,55], conf: 0.7, class: 0},
       {box: [100,100,150,150], conf: 0.95, class: 1},
   ]

Step 1: Sort by confidence
   [
       {box: [100,100,150,150], conf: 0.95, class: 1},  # A
       {box: [10,10,50,50], conf: 0.9, class: 0},       # B
       {box: [12,12,48,48], conf: 0.85, class: 0},      # C
       {box: [15,15,55,55], conf: 0.7, class: 0},       # D
   ]

Step 2: Pick highest (A), add to output
   Output = [A]
   
Step 3: Calculate IoU with A
   IoU(B, A) = 0.0 (berbeda posisi) → Keep B
   IoU(C, A) = 0.0 → Keep C
   IoU(D, A) = 0.0 → Keep D

Step 4: Pick next highest (B), add to output
   Output = [A, B]
   
Step 5: Calculate IoU with B
   IoU(C, B) = 0.85 (overlap besar!) → Remove C
   IoU(D, B) = 0.65 (overlap sedang) → Remove D

Final Output: [A, B]
```

**Rumus IoU (Intersection over Union):**

```python
Box A: [x1_a, y1_a, x2_a, y2_a]
Box B: [x1_b, y1_b, x2_b, y2_b]

Intersection (Irisan):
   x1 = max(x1_a, x1_b)
   y1 = max(y1_a, y1_b)
   x2 = min(x2_a, x2_b)
   y2 = min(y2_a, y2_b)
   
   width = max(0, x2 - x1)
   height = max(0, y2 - y1)
   
   Intersection_Area = width × height

Union (Gabungan):
   Area_A = (x2_a - x1_a) × (y2_a - y1_a)
   Area_B = (x2_b - x1_b) × (y2_b - y1_b)
   
   Union_Area = Area_A + Area_B - Intersection_Area

IoU:
   IoU = Intersection_Area / Union_Area

Contoh:
   Box A: [10, 10, 50, 50]
   Box B: [30, 30, 70, 70]
   
   Intersection: [30, 30, 50, 50]
      Area = 20 × 20 = 400
   
   Area_A = 40 × 40 = 1600
   Area_B = 40 × 40 = 1600
   Union = 1600 + 1600 - 400 = 2800
   
   IoU = 400 / 2800 = 0.143 (14.3% overlap)
```

---

## 🎭 Analogi Sederhana Keseluruhan

### 🏫 Program Ini Seperti Sekolah

**1. Persiapan (Setup)**
```
Seperti: Menyiapkan ruang kelas, buku, alat tulis
Program: Install library, buat folder
```

**2. Kumpulkan Materi (Convert)**
```
Seperti: Kumpulkan buku dari berbagai toko, rapihkan
Program: Kumpulkan foto dari berbagai dataset, ubah format
```

**3. Bagi Tugas (Split)**
```
Seperti: Bagi soal jadi latihan, PR, ujian
Program: Bagi foto jadi train, validation, test
```

**4. Belajar (Train)**
```
Seperti: Murid belajar dari flashcard berulang kali
Program: Model belajar dari foto berulang kali

Guru tunjukkan foto: "Ini plastik"
Murid tebak: "Metal?"
Guru: "Salah! Ini plastik!"
Murid: "Oh, iya benar. Aku perbaiki ingatan."

Ulangi sampai 100 kali (100 epoch)
Akhirnya murid hapal!
```

**5. Ujian (Detect)**
```
Seperti: Tunjukkan foto baru, murid harus jawab
Program: Kasih foto baru, model kasih jawaban

Murid: "Ini plastik! Aku 95% yakin!"
Guru: "Benar! ✓"
```

**6. Praktik Real (Real-time)**
```
Seperti: Murid di dunia nyata, lihat sampah dan langsung tahu
Program: Kamera terus jalan, langsung deteksi

Lihat botol → "Plastik!"
Lihat kaleng → "Metal!"
Lihat kaca → "Kaca!"
```

### 🧩 Puzzle Besar

```
Program ini seperti menyusun puzzle 10,000 keping:

1. Siapkan meja (Setup)
2. Kelompokkan warna (Convert)
3. Pisahkan tepi dan tengah (Split)
4. Mulai susun perlahan (Train)
5. Cek apakah benar (Validate)
6. Tunjukkan hasil (Detect)
7. Buat orang lain bisa main (Real-time)
```

### 🎮 Game Level

```
Level 1: Setup ✓
    Unlock: Virtual Environment
    
Level 2: Convert ✓
    Unlock: Unified Dataset
    
Level 3: Split ✓
    Unlock: Train/Val/Test Sets
    
Level 4: Train (Boss Fight!)
    Enemy: High Loss
    Weapon: Backpropagation
    Victory: Low Loss, High Accuracy
    
Level 5: Detect ✓
    Test your skills!
    
Level 6: Real-time ✓
    Master level - Real world application!
```

---

## 🎯 Kesimpulan Sederhana

**Program ini melakukan 3 hal utama:**

1. **BELAJAR** (Train)
   - Lihat ribuan foto sampah
   - Ingat pola dan ciri-ciri
   - Seperti anak belajar mengenal warna

2. **MENGENALI** (Detect)
   - Lihat foto baru
   - Cocokkan dengan ingatan
   - Kasih label dan kotak

3. **BERAKSI** (Real-time)
   - Gunakan kamera
   - Deteksi terus menerus
   - Seperti mata manusia

**Teknologi yang dipakai:**
- 🧠 **Deep Learning**: Otak buatan
- 👁️ **Computer Vision**: Mata buatan  
- ⚡ **GPU**: Otak yang super cepat
- 🎯 **YOLO**: Algoritma deteksi pintar

**Hasil akhir:**
```
Input: Foto/Video sampah
Output: "Ini PLASTIK! 95% yakin!"
        [Kotak kuning di sekitar sampah]
```

---

## 🙋 Pertanyaan yang Sering Ditanya

**Q: Kenapa harus pakai GPU?**
```
A: CPU seperti 1 orang kerja keras
   GPU seperti 1000 orang kerja bareng
   Lebih cepat 10-50x!
```

**Q: Kenapa harus 100 epoch?**
```
A: Seperti belajar matematika
   Sekali baca buku: Belum paham
   100 kali latihan soal: Jadi mahir!
```

**Q: Kenapa ada train/val/test?**
```
A: Train: Belajar
   Val: PR (cek sudah paham?)
   Test: Ujian (cek benar-benar paham?)
   
   Seperti sekolah!
```

**Q: Apa bedanya dengan cara lama?**
```
A: Cara lama:
   if warna_kuning and bentuk_silinder:
       return "Plastik"
   
   Ditulis manual, banyak aturan!
   
   Cara baru (Deep Learning):
   Komputer belajar sendiri!
   Lebih fleksibel, lebih akurat.
```

---

## 📚 Belajar Lebih Lanjut

Kalau mau paham lebih dalam:

1. **Python Dasar**
   - Variabel, loop, function
   - List, dictionary
   
2. **NumPy**
   - Array, matrix
   - Operasi matematika
   
3. **Computer Vision**
   - Pixel, gambar digital
   - Filter, konvolusi
   
4. **Machine Learning**
   - Supervised learning
   - Neural network
   
5. **Deep Learning**
   - CNN
   - Backpropagation
   - Optimizer

**Urutan belajar:**
```
Python → NumPy → ML Dasar → Computer Vision → Deep Learning → YOLO
```

---

## 🎉 Selamat!

Kamu sudah paham cara kerja program klasifikasi sampah dari A sampai Z!

Sekarang kamu tahu:
- ✅ Kenapa perlu convert dataset
- ✅ Kenapa perlu split data
- ✅ Bagaimana komputer belajar
- ✅ Bagaimana deteksi objek bekerja
- ✅ Bagaimana real-time detection jalan

**Ingat:** Komputer belajar seperti manusia - butuh latihan berulang kali! 🚀

---

*Dibuat dengan ❤️ untuk pemula yang ingin belajar AI*
