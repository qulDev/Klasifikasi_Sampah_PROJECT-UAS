# 📖 Modul 05: Web Application dengan Streamlit

## Daftar Isi

1. [Pengantar Streamlit](#1-pengantar-streamlit)
2. [Arsitektur Web App](#2-arsitektur-web-app)
3. [Analisis web_app.py Line-by-Line](#3-analisis-web_apppy-line-by-line)
4. [Komponen UI Streamlit](#4-komponen-ui-streamlit)
5. [Integrasi dengan YOLO](#5-integrasi-dengan-yolo)
6. [State Management](#6-state-management)
7. [Deployment](#7-deployment)
8. [Customization](#8-customization)
9. [Troubleshooting](#9-troubleshooting)
10. [Latihan](#10-latihan)

---

## 1. Pengantar Streamlit

### 1.1 Apa Itu Streamlit?

Streamlit adalah framework Python untuk membuat web application dengan cepat dan mudah, terutama untuk Machine Learning dan Data Science projects.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STREAMLIT OVERVIEW                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TRADITIONAL WEB DEVELOPMENT                                                │
│  ───────────────────────────                                               │
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │  Frontend   │  │  Backend    │  │  Database   │  │    API      │       │
│  │  HTML/CSS   │  │  Python/    │  │  SQL/       │  │  REST/      │       │
│  │  JavaScript │  │  Flask      │  │  NoSQL      │  │  GraphQL    │       │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘       │
│        │               │               │               │                   │
│        └───────────────┴───────────────┴───────────────┘                   │
│                          Kompleks! Banyak yang harus dipelajari            │
│                                                                             │
│  ════════════════════════════════════════════════════════════════════════  │
│                                                                             │
│  STREAMLIT WAY ★                                                           │
│  ───────────────                                                           │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │   PYTHON SAJA!                                                      │   │
│  │                                                                     │   │
│  │   import streamlit as st                                           │   │
│  │                                                                     │   │
│  │   st.title("My App")                                               │   │
│  │   uploaded = st.file_uploader("Upload image")                      │   │
│  │   if uploaded:                                                     │   │
│  │       st.image(uploaded)                                           │   │
│  │                                                                     │   │
│  │   → Auto-generates HTML, CSS, JS                                   │   │
│  │   → Built-in state management                                      │   │
│  │   → Hot reload development                                         │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Keunggulan Streamlit:                                                     │
│  ✓ Python-only (no HTML/CSS/JS)                                           │
│  ✓ Reaktif (UI update otomatis)                                           │
│  ✓ Widget bawaan lengkap                                                  │
│  ✓ Caching untuk performance                                              │
│  ✓ Deploy mudah                                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Mengapa Streamlit untuk ML?

| Fitur             | Deskripsi                  | Contoh dalam Project          |
| ----------------- | -------------------------- | ----------------------------- |
| **File Uploader** | Upload gambar dengan mudah | User upload foto sampah       |
| **Image Display** | Tampilkan gambar hasil     | Show deteksi dengan bbox      |
| **Slider**        | Adjust parameter           | Confidence threshold          |
| **Metrics**       | Tampilkan statistik        | Total objek, recyclable count |
| **Expander**      | Collapsible content        | Detail per deteksi            |
| **Caching**       | Optimasi performance       | Cache model loading           |

---

## 2. Arsitektur Web App

### 2.1 Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      WEB APP ARCHITECTURE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  USER                                                                       │
│    │                                                                        │
│    │ 1. Upload Image                                                        │
│    ▼                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    STREAMLIT APP (web_app.py)                        │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                      │   │
│  │  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐        │   │
│  │  │   Sidebar    │     │    Main      │     │   Footer     │        │   │
│  │  │              │     │   Content    │     │              │        │   │
│  │  │ - Settings   │     │              │     │ - Credits    │        │   │
│  │  │ - Confidence │     │ - Upload     │     │              │        │   │
│  │  │ - Guide      │     │ - Display    │     │              │        │   │
│  │  │              │     │ - Results    │     │              │        │   │
│  │  └──────────────┘     └──────┬───────┘     └──────────────┘        │   │
│  │                              │                                      │   │
│  └──────────────────────────────│──────────────────────────────────────┘   │
│                                 │                                           │
│                                 │ 2. Process Image                          │
│                                 ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        YOLO MODEL                                    │   │
│  │                   (models/best_model.pt)                             │   │
│  │                                                                      │   │
│  │  Input: PIL Image                                                    │   │
│  │  Output: Detections (boxes, classes, confidence)                    │   │
│  │                                                                      │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                           │
│                                 │ 3. Return Results                         │
│                                 ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       DISPLAY RESULTS                                │   │
│  │                                                                      │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐        │   │
│  │  │  Annotated     │  │   Metrics      │  │    Details     │        │   │
│  │  │  Image         │  │   Summary      │  │    Per Object  │        │   │
│  │  │                │  │                │  │                │        │   │
│  │  │  [Bbox drawn]  │  │  Total: 3      │  │  🔋 Battery    │        │   │
│  │  │                │  │  Recyclable: 2 │  │  Conf: 85%     │        │   │
│  │  │                │  │  Avg Conf: 78% │  │  Tips: ...     │        │   │
│  │  └────────────────┘  └────────────────┘  └────────────────┘        │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 File Structure

```
web_app.py
│
├── Configuration (Line 1-30)
│   ├── Page config
│   ├── Model path
│   └── Class info dictionary
│
├── Model Loading (Line 31-45)
│   └── @st.cache_resource decorator
│
├── Detection Function (Line 46-55)
│   └── Run YOLO inference
│
└── Main Function (Line 56-end)
    ├── Sidebar
    │   ├── Confidence slider
    │   └── Class guide
    │
    └── Main Content
        ├── Title & description
        ├── File uploader
        ├── Camera input
        ├── Detection display
        └── Results details
```

---

## 3. Analisis web_app.py Line-by-Line

### 3.1 Import dan Page Config

```python
#!/usr/bin/env python3
"""
Klasifikasi Sampah Anorganik Menggunakan Algoritma YOLO
========================================================
Web application untuk klasifikasi sampah anorganik menggunakan YOLOv8.

Usage:
    streamlit run web_app.py

Requirements:
    pip install streamlit
"""

import streamlit as st        # Framework utama
from pathlib import Path      # Path handling
from PIL import Image        # Image processing
import numpy as np           # Numerical operations

# Page configuration - HARUS di awal sebelum widget lain
st.set_page_config(
    page_title="Klasifikasi Sampah Anorganik - YOLO",  # Browser tab title
    page_icon="♻️",          # Favicon (emoji atau path ke .ico)
    layout="wide",           # "wide" atau "centered"
    initial_sidebar_state="expanded"  # Sidebar terbuka/tertutup
)
```

**Penjelasan `st.set_page_config()`:**

| Parameter               | Nilai                  | Fungsi               |
| ----------------------- | ---------------------- | -------------------- |
| `page_title`            | String                 | Title di browser tab |
| `page_icon`             | Emoji/Path             | Favicon              |
| `layout`                | "wide"/"centered"      | Lebar konten         |
| `initial_sidebar_state` | "expanded"/"collapsed" | State awal sidebar   |

### 3.2 Configuration Constants

```python
# Config
MODEL_PATH = './models/best_model.pt'  # Path ke trained model
DEFAULT_CONF = 0.25                     # Default confidence threshold

# Class info dengan metadata lengkap
CLASS_INFO = {
    'battery': {
        'emoji': '🔋',
        'category': 'B3 (Berbahaya)',
        'bin_color': '🔴 Merah',
        'disposal': 'Tempat khusus limbah B3, jangan buang sembarangan!',
        'recyclable': False,
        'tips': 'Bawa ke drop point khusus baterai di mall atau kantor kelurahan'
    },
    'biological': {
        'emoji': '🥬',
        'category': 'Organik',
        'bin_color': '🟢 Hijau',
        'disposal': 'Tong sampah organik/hijau, bisa dijadikan kompos',
        'recyclable': True,
        'tips': 'Pisahkan dari plastik, bisa diolah jadi pupuk kompos'
    },
    # ... (10 kelas total)
}
```

**Struktur CLASS_INFO per kelas:**

```
CLASS_INFO['plastic'] = {
    'emoji': '🥤',         # Emoji untuk visualisasi
    'category': 'Anorganik - Dapat Didaur Ulang',
    'bin_color': '🟡 Kuning',   # Warna tempat sampah
    'disposal': 'Bank sampah, pilah berdasarkan jenis plastik',
    'recyclable': True,    # Boolean untuk statistik
    'tips': 'Cuci bersih dan keringkan, lepas label jika ada'
}
```

### 3.3 Model Loading dengan Caching

```python
@st.cache_resource  # ← Decorator penting!
def load_model():
    """
    Load YOLO model dengan caching.

    @st.cache_resource:
    - Model di-load sekali saja
    - Di-cache di memory
    - Tidak perlu reload setiap rerun
    - Hemat waktu & resources
    """
    from ultralytics import YOLO

    if not Path(MODEL_PATH).exists():
        return None  # Model tidak ditemukan

    return YOLO(MODEL_PATH)
```

**Pentingnya `@st.cache_resource`:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CACHING DALAM STREAMLIT                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TANPA CACHING:                                                            │
│  ──────────────                                                            │
│                                                                             │
│  User Action      →   Script Rerun      →   Model Load (3-5 detik)         │
│  User Action      →   Script Rerun      →   Model Load (3-5 detik)         │
│  User Action      →   Script Rerun      →   Model Load (3-5 detik)         │
│                                                                             │
│  Setiap interaksi = reload model = lambat!                                 │
│                                                                             │
│  ════════════════════════════════════════════════════════════════════════  │
│                                                                             │
│  DENGAN @st.cache_resource:                                                │
│  ──────────────────────────                                                │
│                                                                             │
│  First Run       →   Model Load (3-5 detik)  →  Cache model               │
│  User Action     →   Script Rerun            →  Use cached (instant!)     │
│  User Action     →   Script Rerun            →  Use cached (instant!)     │
│  User Action     →   Script Rerun            →  Use cached (instant!)     │
│                                                                             │
│  Model di-load sekali, digunakan berkali-kali!                             │
│                                                                             │
│  Kapan gunakan mana?                                                       │
│  - @st.cache_data: Untuk data (DataFrame, dict, list)                     │
│  - @st.cache_resource: Untuk resources (model, database connection)       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.4 Detection Function

```python
def detect_waste(image, model, confidence):
    """
    Run detection on image.

    Args:
        image: PIL Image atau numpy array
        model: YOLO model instance
        confidence: Float confidence threshold (0-1)

    Returns:
        YOLO Results object
    """
    # YOLO accept berbagai format input:
    # - PIL Image
    # - numpy array
    # - file path
    # - URL
    results = model(image, conf=confidence, verbose=False)
    return results
```

### 3.5 Main Function - Sidebar

```python
def main():
    # ==================== SIDEBAR ====================
    with st.sidebar:
        # Logo/icon
        st.image("https://img.icons8.com/color/96/000000/recycle-sign.png", width=80)
        st.title("⚙️ Pengaturan")

        # Confidence slider
        confidence = st.slider(
            "🎚️ Confidence Threshold",
            min_value=0.1,      # Minimum value
            max_value=0.9,      # Maximum value
            value=DEFAULT_CONF, # Default value (0.25)
            step=0.05,          # Step increment
            help="Semakin rendah = lebih banyak deteksi, tapi mungkin kurang akurat"
        )

        st.divider()  # Horizontal line

        # Panduan warna tempat sampah
        st.markdown("### 🗑️ Panduan Tempat Sampah")
        st.markdown("""
        | Warna | Jenis |
        |-------|-------|
        | 🔴 Merah | B3/Berbahaya |
        | 🟢 Hijau | Organik |
        | 🔵 Biru | Anorganik Daur Ulang |
        | 🟡 Kuning | Plastik |
        | ⚫ Hitam | Residu/Umum |
        """)

        st.divider()

        # Daftar 10 kelas
        st.markdown("### 📊 10 Kelas Deteksi")
        for cls, info in CLASS_INFO.items():
            st.markdown(f"{info['emoji']} {cls.capitalize()}")
```

### 3.6 Main Function - Content Area

```python
    # ==================== MAIN CONTENT ====================

    # Title
    st.title("♻️ Klasifikasi Sampah Anorganik")
    st.markdown("**Menggunakan Algoritma YOLO (You Only Look Once)**")
    st.caption("Deteksi dan klasifikasi 10 jenis sampah secara otomatis")

    # Load model
    model = load_model()

    if model is None:
        # Handle model not found
        st.error("❌ Model tidak ditemukan!")
        st.info("Jalankan `python train.py` terlebih dahulu.")
        st.code("python train.py", language="bash")
        return

    st.success(f"✅ Model loaded: {MODEL_PATH}")

    st.divider()

    # Two-column layout
    col1, col2 = st.columns([1, 1])  # Equal width columns

    with col1:
        st.markdown("### 📷 Upload Gambar")

        # File uploader
        uploaded_file = st.file_uploader(
            "Pilih gambar sampah...",
            type=['jpg', 'jpeg', 'png'],  # Accepted file types
            help="Upload gambar sampah yang ingin dideteksi"
        )

        # Camera input option
        use_camera = st.checkbox("📹 Atau gunakan kamera")
        if use_camera:
            camera_image = st.camera_input("Ambil foto")
            if camera_image:
                uploaded_file = camera_image  # Override dengan foto kamera
```

### 3.7 Main Function - Processing & Display

```python
    # Process image jika ada
    if uploaded_file is not None:
        # Load image sebagai PIL Image
        image = Image.open(uploaded_file)

        with col1:
            # Display original image
            st.image(image, caption="Gambar Input", use_container_width=True)

        # Run detection dengan spinner (loading indicator)
        with st.spinner("🔍 Mendeteksi sampah..."):
            results = detect_waste(image, model, confidence)

        with col2:
            st.markdown("### 🎯 Hasil Deteksi")

            # Get annotated image (dengan bounding boxes)
            annotated = results[0].plot()  # Returns numpy array
            st.image(annotated, caption="Hasil Deteksi", use_container_width=True)

        # ==================== RESULTS SECTION ====================
        st.divider()
        st.markdown("### 📊 Detail Hasil")

        # Extract detections
        detections = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0].cpu().numpy())     # Class ID
                conf_score = float(box.conf[0].cpu().numpy())  # Confidence
                class_name = result.names[cls]          # Class name
                info = CLASS_INFO.get(class_name, {})   # Get metadata

                detections.append({
                    'class': class_name,
                    'confidence': conf_score,
                    'info': info
                })
```

### 3.8 Main Function - Metrics Display

```python
        if detections:
            # Calculate summary metrics
            recyclable_count = sum(
                1 for d in detections if d['info'].get('recyclable', False)
            )
            non_recyclable_count = len(detections) - recyclable_count
            avg_conf = np.mean([d['confidence'] for d in detections])

            # Display metrics in columns
            metric_cols = st.columns(4)

            with metric_cols[0]:
                st.metric("🎯 Total Objek", len(detections))

            with metric_cols[1]:
                st.metric("♻️ Dapat Didaur Ulang", recyclable_count)

            with metric_cols[2]:
                st.metric("🚫 Tidak Dapat Didaur Ulang", non_recyclable_count)

            with metric_cols[3]:
                st.metric("📈 Rata-rata Confidence", f"{avg_conf:.1%}")

            st.divider()

            # Detailed results per detection
            for i, det in enumerate(detections, 1):
                info = det['info']
                emoji = info.get('emoji', '❓')

                # Expandable section per detection
                with st.expander(
                    f"{emoji} **{i}. {det['class'].upper()}** - "
                    f"Confidence: {det['confidence']:.1%}",
                    expanded=True
                ):
                    cols = st.columns([1, 2])

                    with cols[0]:
                        st.markdown(f"**Kategori:** {info.get('category', 'Unknown')}")
                        st.markdown(f"**Tempat Sampah:** {info.get('bin_color', '-')}")

                        if info.get('recyclable'):
                            st.success("♻️ Dapat didaur ulang")
                        else:
                            st.warning("🚫 Tidak dapat didaur ulang")

                    with cols[1]:
                        st.info(f"📍 **Cara Buang:** {info.get('disposal', '-')}")
                        st.markdown(f"💡 **Tips:** {info.get('tips', '-')}")
```

### 3.9 Main Function - No Detection Handling & Footer

```python
        else:
            # No objects detected
            st.warning("⚠️ Tidak ada objek terdeteksi")
            st.markdown("""
            **Kemungkinan penyebab:**
            - Gambar kurang jelas atau gelap
            - Objek terlalu kecil dalam gambar
            - Confidence threshold terlalu tinggi

            **Solusi:**
            - Coba turunkan confidence threshold di sidebar
            - Upload gambar dengan pencahayaan lebih baik
            - Pastikan objek terlihat jelas dalam gambar
            """)

    else:
        # Placeholder saat belum ada image
        with col2:
            st.markdown("### 🎯 Hasil Deteksi")
            st.info("👆 Upload gambar untuk memulai deteksi")

    # ==================== FOOTER ====================
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
        <p><strong>Klasifikasi Sampah Anorganik Menggunakan Algoritma YOLO</strong></p>
        <p>Powered by YOLOv8 & Streamlit</p>
        <p>Bantu jaga lingkungan dengan membuang sampah pada tempatnya! 🌍</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
```

---

## 4. Komponen UI Streamlit

### 4.1 Widget Reference

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STREAMLIT WIDGETS                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TEXT ELEMENTS                                                              │
│  ──────────────                                                            │
│  st.title("Title")              # Big title                                │
│  st.header("Header")            # Section header                           │
│  st.subheader("Subheader")      # Subsection                              │
│  st.markdown("**bold**")        # Markdown support                         │
│  st.caption("Caption")          # Small gray text                         │
│  st.code("print('hi')")         # Code block                              │
│                                                                             │
│  INPUT WIDGETS                                                              │
│  ─────────────                                                             │
│  st.text_input("Label")         # Single line text                        │
│  st.text_area("Label")          # Multi-line text                         │
│  st.number_input("Label")       # Numeric input                           │
│  st.slider("Label", min, max)   # Slider                                  │
│  st.selectbox("Label", list)    # Dropdown                                │
│  st.multiselect("Label", list)  # Multi-select                            │
│  st.checkbox("Label")           # Checkbox                                │
│  st.radio("Label", list)        # Radio buttons                           │
│  st.button("Label")             # Button                                  │
│                                                                             │
│  FILE HANDLING                                                              │
│  ─────────────                                                             │
│  st.file_uploader("Label")      # File upload                             │
│  st.camera_input("Label")       # Camera capture                          │
│  st.download_button("Label")    # File download                           │
│                                                                             │
│  DISPLAY ELEMENTS                                                           │
│  ────────────────                                                          │
│  st.image(img)                  # Display image                           │
│  st.video(video)                # Display video                           │
│  st.audio(audio)                # Audio player                            │
│  st.dataframe(df)               # Interactive table                       │
│  st.table(df)                   # Static table                            │
│  st.metric("Label", value)      # Metric card                             │
│                                                                             │
│  STATUS ELEMENTS                                                            │
│  ───────────────                                                           │
│  st.success("Message")          # Green box                               │
│  st.info("Message")             # Blue box                                │
│  st.warning("Message")          # Yellow box                              │
│  st.error("Message")            # Red box                                 │
│  st.spinner("Loading...")       # Loading spinner                         │
│  st.progress(0.5)               # Progress bar                            │
│                                                                             │
│  LAYOUT                                                                     │
│  ──────                                                                    │
│  st.columns([1,2,1])            # Column layout                           │
│  st.expander("Click")           # Collapsible section                     │
│  st.sidebar.xyz()               # Sidebar widgets                         │
│  st.tabs(["Tab1", "Tab2"])      # Tabs                                    │
│  st.divider()                   # Horizontal line                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Layout Examples

```python
# Column layout
col1, col2, col3 = st.columns([1, 2, 1])  # Ratio 1:2:1
with col1:
    st.write("Left")
with col2:
    st.write("Center (wider)")
with col3:
    st.write("Right")

# Tabs
tab1, tab2, tab3 = st.tabs(["Upload", "Camera", "Settings"])
with tab1:
    st.file_uploader("Upload image")
with tab2:
    st.camera_input("Take photo")
with tab3:
    st.slider("Confidence", 0.1, 0.9)

# Expander
with st.expander("Click to expand"):
    st.write("Hidden content here")

# Sidebar
with st.sidebar:
    st.title("Settings")
    confidence = st.slider("Confidence", 0.1, 0.9)
```

---

## 5. Integrasi dengan YOLO

### 5.1 Flow Integrasi

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      YOLO + STREAMLIT INTEGRATION                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. User uploads image via st.file_uploader()                              │
│     │                                                                       │
│     ▼                                                                       │
│  2. Image loaded as PIL Image                                              │
│     │                                                                       │
│     │   image = Image.open(uploaded_file)                                  │
│     ▼                                                                       │
│  3. Pass to YOLO model                                                     │
│     │                                                                       │
│     │   results = model(image, conf=confidence)                            │
│     ▼                                                                       │
│  4. YOLO returns Results object                                            │
│     │                                                                       │
│     │   results[0].boxes  → Bounding boxes                                 │
│     │   results[0].names  → Class name mapping                             │
│     │   results[0].plot() → Annotated image                                │
│     ▼                                                                       │
│  5. Display annotated image                                                │
│     │                                                                       │
│     │   st.image(results[0].plot())                                        │
│     ▼                                                                       │
│  6. Extract and display details                                            │
│     │                                                                       │
│     │   for box in results[0].boxes:                                       │
│     │       class_id = box.cls[0]                                          │
│     │       confidence = box.conf[0]                                       │
│     │       coordinates = box.xyxy[0]                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Accessing Detection Results

```python
# YOLO Results structure
results = model(image)  # Returns list of Results

# Untuk setiap result (biasanya 1 untuk single image)
for result in results:
    # Boxes object berisi semua deteksi
    boxes = result.boxes

    # Iterate setiap detection
    for box in boxes:
        # Coordinates (multiple formats available)
        xyxy = box.xyxy[0]        # [x1, y1, x2, y2] top-left, bottom-right
        xywh = box.xywh[0]        # [x_center, y_center, width, height]
        xywhn = box.xywhn[0]      # Normalized xywh

        # Class information
        cls_id = int(box.cls[0])          # Class ID (0-9)
        cls_name = result.names[cls_id]   # Class name ('plastic', etc)

        # Confidence
        conf = float(box.conf[0])  # Confidence score (0-1)

    # Get annotated image (numpy array)
    annotated_frame = result.plot()

    # Other attributes
    orig_shape = result.orig_shape  # Original image shape
    speed = result.speed            # Inference time
```

---

## 6. State Management

### 6.1 Cara Kerja Streamlit

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STREAMLIT EXECUTION MODEL                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  IMPORTANT: Streamlit re-runs ENTIRE script pada SETIAP interaksi!        │
│                                                                             │
│  User clicks button                                                         │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────┐                                           │
│  │    Script runs from top     │                                           │
│  │         ↓                   │                                           │
│  │    import streamlit as st   │                                           │
│  │    st.title("App")          │                                           │
│  │    ...                      │                                           │
│  │    if st.button("Click"):   │ ← Button state checked                   │
│  │        do_something()       │                                           │
│  │    ...                      │                                           │
│  │    (end of script)          │                                           │
│  └─────────────────────────────┘                                           │
│                                                                             │
│  MASALAH: Variabel di-reset setiap rerun!                                  │
│                                                                             │
│  SOLUSI: Gunakan Session State atau Caching                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Session State

```python
# Session state untuk persist data across reruns

# Initialize
if 'counter' not in st.session_state:
    st.session_state.counter = 0

# Update
if st.button('Increment'):
    st.session_state.counter += 1

# Access
st.write(f"Counter: {st.session_state.counter}")

# Dalam context project, bisa digunakan untuk:
# - Store detection history
# - Remember user preferences
# - Cache intermediate results
```

### 6.3 Caching Decorators

```python
# @st.cache_data - untuk data (DataFrame, dict, list)
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

# @st.cache_resource - untuk resources (model, connections)
@st.cache_resource
def load_model():
    return YOLO('model.pt')

# Perbedaan:
# - cache_data: Deep copy setiap akses (safe untuk mutable data)
# - cache_resource: Single instance (efficient untuk large objects)
```

---

## 7. Deployment

### 7.1 Local Deployment

```bash
# Run locally
streamlit run web_app.py

# Custom port
streamlit run web_app.py --server.port 8080

# Allow external access
streamlit run web_app.py --server.address 0.0.0.0

# Run with specific browser
streamlit run web_app.py --browser.serverAddress localhost
```

### 7.2 Streamlit Cloud

```yaml
# requirements.txt harus ada
# Struktur repo:
project/
├── web_app.py
├── requirements.txt
├── data.yaml
└── models/
└── best_model.pt
# Deploy:
# 1. Push ke GitHub
# 2. Buka share.streamlit.io
# 3. Connect repo
# 4. Deploy!
```

### 7.3 Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "web_app.py", "--server.port=8501"]
```

```bash
# Build dan run
docker build -t waste-classifier .
docker run -p 8501:8501 waste-classifier
```

---

## 8. Customization

### 8.1 Custom CSS

```python
# Inject custom CSS
st.markdown("""
<style>
    /* Custom styling */
    .stApp {
        background-color: #f0f2f6;
    }

    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
    }

    .stMetric {
        background-color: white;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)
```

### 8.2 Adding Features

```python
# Download annotated image
if detections:
    # Convert annotated image to bytes
    from PIL import Image
    import io

    img_pil = Image.fromarray(annotated)
    buf = io.BytesIO()
    img_pil.save(buf, format='JPEG')

    st.download_button(
        label="📥 Download Hasil",
        data=buf.getvalue(),
        file_name="hasil_deteksi.jpg",
        mime="image/jpeg"
    )

# Detection history
if 'history' not in st.session_state:
    st.session_state.history = []

# Add to history
st.session_state.history.append({
    'timestamp': datetime.now(),
    'detections': detections
})

# Show history
st.markdown("### 📜 Riwayat Deteksi")
for item in st.session_state.history[-5:]:  # Last 5
    st.write(f"{item['timestamp']}: {len(item['detections'])} objek")
```

---

## 9. Troubleshooting

### 9.1 Masalah Umum

| Masalah                 | Penyebab             | Solusi                     |
| ----------------------- | -------------------- | -------------------------- |
| Model tidak load        | Path salah           | Check MODEL_PATH           |
| Slow pada setiap upload | Tidak pakai cache    | Gunakan @st.cache_resource |
| Image tidak display     | Format tidak support | Convert ke RGB             |
| Error "No module"       | Dependency missing   | pip install ulang          |
| Memory error            | Model terlalu besar  | Gunakan model lebih kecil  |

### 9.2 Debug Tips

```python
# Print debug info
st.write("Debug:", uploaded_file)
st.write("Image shape:", np.array(image).shape)
st.write("Results:", results)

# Show raw results
with st.expander("Raw Detection Data"):
    for result in results:
        st.json({
            'boxes': result.boxes.xyxy.tolist() if result.boxes else [],
            'confidence': result.boxes.conf.tolist() if result.boxes else [],
            'classes': result.boxes.cls.tolist() if result.boxes else []
        })
```

---

## 10. Latihan

### Latihan 1: Basic Modification

Modifikasi web app untuk menambahkan:

1. Dropdown untuk memilih model (yolov8n, yolov8s, yolov8m)
2. Checkbox untuk toggle bounding box display

### Latihan 2: Feature Addition

Tambahkan fitur:

1. Button untuk download hasil deteksi sebagai JSON
2. History panel yang menampilkan 5 deteksi terakhir

### Latihan 3: UI Enhancement

Improve UI dengan:

1. Dark mode toggle
2. Custom color scheme
3. Progress bar saat loading

---

**Selamat! Anda telah menyelesaikan Modul 05: Web Application dengan Streamlit**

_Lanjut ke: [Modul 06 - REST API](./06-rest-api.md)_
