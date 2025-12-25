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

import streamlit as st
from pathlib import Path
from PIL import Image
import numpy as np

# Page config
st.set_page_config(
    page_title="Klasifikasi Sampah Anorganik - YOLO",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Config
MODEL_PATH = './models/best_model.pt'
DEFAULT_CONF = 0.25

# Class info dengan emoji, kategori, dan saran pembuangan
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
    'cardboard': {
        'emoji': '📦',
        'category': 'Anorganik - Dapat Didaur Ulang',
        'bin_color': '🔵 Biru',
        'disposal': 'Bank sampah atau pengepul kardus',
        'recyclable': True,
        'tips': 'Lipat rata agar tidak makan tempat, jaga tetap kering'
    },
    'clothes': {
        'emoji': '👕',
        'category': 'Tekstil',
        'bin_color': '🔵 Biru',
        'disposal': 'Donasikan jika masih layak, atau bawa ke bank sampah',
        'recyclable': True,
        'tips': 'Pakaian layak pakai bisa didonasikan ke panti atau yang membutuhkan'
    },
    'glass': {
        'emoji': '🍾',
        'category': 'Anorganik - Dapat Didaur Ulang',
        'bin_color': '🔵 Biru',
        'disposal': 'Bank sampah atau pengepul botol/kaca',
        'recyclable': True,
        'tips': 'Hati-hati pecahan kaca, bungkus dengan koran jika pecah'
    },
    'metal': {
        'emoji': '🥫',
        'category': 'Anorganik - Dapat Didaur Ulang',
        'bin_color': '🔵 Biru',
        'disposal': 'Bank sampah atau pengepul logam/besi tua',
        'recyclable': True,
        'tips': 'Cuci bersih kaleng bekas makanan sebelum dibuang'
    },
    'paper': {
        'emoji': '📄',
        'category': 'Anorganik - Dapat Didaur Ulang',
        'bin_color': '🔵 Biru',
        'disposal': 'Bank sampah atau pengepul kertas/koran bekas',
        'recyclable': True,
        'tips': 'Jaga tetap kering, kertas basah sulit didaur ulang'
    },
    'plastic': {
        'emoji': '🥤',
        'category': 'Anorganik - Dapat Didaur Ulang',
        'bin_color': '🟡 Kuning',
        'disposal': 'Bank sampah, pilah berdasarkan jenis plastik',
        'recyclable': True,
        'tips': 'Cuci bersih dan keringkan, lepas label jika ada'
    },
    'shoes': {
        'emoji': '👟',
        'category': 'Tekstil',
        'bin_color': '🔵 Biru',
        'disposal': 'Donasikan jika masih layak, atau bawa ke bank sampah',
        'recyclable': True,
        'tips': 'Sepatu layak pakai bisa didonasikan'
    },
    'trash': {
        'emoji': '🗑️',
        'category': 'Residu',
        'bin_color': '⚫ Hitam',
        'disposal': 'Tong sampah umum/hitam, akan dibawa ke TPA',
        'recyclable': False,
        'tips': 'Kurangi sampah residu dengan memilah lebih baik'
    },
}


@st.cache_resource
def load_model():
    """Load YOLO model with caching"""
    from ultralytics import YOLO
    
    if not Path(MODEL_PATH).exists():
        return None
    return YOLO(MODEL_PATH)


def detect_waste(image, model, confidence):
    """Run detection on image"""
    results = model(image, conf=confidence, verbose=False)
    return results


def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/recycle-sign.png", width=80)
        st.title("⚙️ Pengaturan")
        
        confidence = st.slider(
            "🎚️ Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=DEFAULT_CONF,
            step=0.05,
            help="Semakin rendah = lebih banyak deteksi, tapi mungkin kurang akurat"
        )
        
        st.divider()
        
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
        st.markdown("### 📊 10 Kelas Deteksi")
        for cls, info in CLASS_INFO.items():
            st.markdown(f"{info['emoji']} {cls.capitalize()}")

    # Main content
    st.title("♻️ Klasifikasi Sampah Anorganik")
    st.markdown("**Menggunakan Algoritma YOLO (You Only Look Once)**")
    st.caption("Deteksi dan klasifikasi 10 jenis sampah secara otomatis dengan Deep Learning")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("❌ Model tidak ditemukan!")
        st.info("Jalankan `python train.py` terlebih dahulu untuk melatih model.")
        st.code("python train.py", language="bash")
        return
    
    st.success(f"✅ Model loaded: {MODEL_PATH}")
    
    # Upload section
    st.divider()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📷 Upload Gambar")
        uploaded_file = st.file_uploader(
            "Pilih gambar sampah...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload gambar sampah yang ingin dideteksi"
        )
        
        # Camera input option
        use_camera = st.checkbox("📹 Atau gunakan kamera")
        if use_camera:
            camera_image = st.camera_input("Ambil foto")
            if camera_image:
                uploaded_file = camera_image
    
    # Process image
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        
        with col1:
            st.image(image, caption="Gambar Input", use_container_width=True)
        
        # Run detection
        with st.spinner("🔍 Mendeteksi sampah..."):
            results = detect_waste(image, model, confidence)
        
        with col2:
            st.markdown("### 🎯 Hasil Deteksi")
            
            # Get annotated image
            annotated = results[0].plot()
            st.image(annotated, caption="Hasil Deteksi", use_container_width=True)
        
        # Results section
        st.divider()
        st.markdown("### 📊 Detail Hasil")
        
        detections = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0].cpu().numpy())
                conf_score = float(box.conf[0].cpu().numpy())
                class_name = result.names[cls]
                info = CLASS_INFO.get(class_name, {})
                
                detections.append({
                    'class': class_name,
                    'confidence': conf_score,
                    'info': info
                })
        
        if detections:
            # Summary metrics
            recyclable_count = sum(1 for d in detections if d['info'].get('recyclable', False))
            non_recyclable_count = len(detections) - recyclable_count
            
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("🎯 Total Objek", len(detections))
            with metric_cols[1]:
                st.metric("♻️ Dapat Didaur Ulang", recyclable_count)
            with metric_cols[2]:
                st.metric("🚫 Tidak Dapat Didaur Ulang", non_recyclable_count)
            with metric_cols[3]:
                avg_conf = np.mean([d['confidence'] for d in detections])
                st.metric("📈 Rata-rata Confidence", f"{avg_conf:.1%}")
            
            st.divider()
            
            # Detailed results
            for i, det in enumerate(detections, 1):
                info = det['info']
                emoji = info.get('emoji', '❓')
                
                with st.expander(f"{emoji} **{i}. {det['class'].upper()}** - Confidence: {det['confidence']:.1%}", expanded=True):
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
        
        else:
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
        # Placeholder when no image
        with col2:
            st.markdown("### 🎯 Hasil Deteksi")
            st.info("👆 Upload gambar untuk memulai deteksi")
    
    # Footer
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
