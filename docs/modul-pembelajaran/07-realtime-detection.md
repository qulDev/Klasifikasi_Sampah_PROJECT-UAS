# 📖 Modul 07: Realtime Detection dengan OpenCV

## Daftar Isi

1. [Pengantar OpenCV](#1-pengantar-opencv)
2. [Konsep Video Processing](#2-konsep-video-processing)
3. [Analisis detect.py Line-by-Line](#3-analisis-detectpy-line-by-line)
4. [Video Capture](#4-video-capture)
5. [Frame Processing](#5-frame-processing)
6. [Drawing & Annotation](#6-drawing--annotation)
7. [Performance Optimization](#7-performance-optimization)
8. [Multi-Camera Support](#8-multi-camera-support)
9. [Troubleshooting](#9-troubleshooting)
10. [Latihan](#10-latihan)

---

## 1. Pengantar OpenCV

### 1.1 Apa Itu OpenCV?

OpenCV (Open Source Computer Vision Library) adalah library untuk computer vision dan image processing yang paling populer di dunia.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       OPENCV OVERVIEW                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  OpenCV = Open Source Computer Vision                                      │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │   CAPABILITIES                                                      │   │
│  │   ────────────                                                     │   │
│  │                                                                     │   │
│  │   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐       │   │
│  │   │ Image I/O     │   │ Video         │   │ Image         │       │   │
│  │   │               │   │ Processing    │   │ Processing    │       │   │
│  │   │ - Read/Write  │   │               │   │               │       │   │
│  │   │ - Webcam      │   │ - Capture     │   │ - Filters     │       │   │
│  │   │ - Video files │   │ - Streaming   │   │ - Transform   │       │   │
│  │   │ - Screenshots │   │ - Recording   │   │ - Color conv  │       │   │
│  │   └───────────────┘   └───────────────┘   └───────────────┘       │   │
│  │                                                                     │   │
│  │   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐       │   │
│  │   │ Drawing       │   │ Object        │   │ ML/DL         │       │   │
│  │   │               │   │ Detection     │   │ Integration   │       │   │
│  │   │ - Shapes      │   │               │   │               │       │   │
│  │   │ - Text        │   │ - Face        │   │ - DNN module  │       │   │
│  │   │ - Lines       │   │ - Objects     │   │ - YOLO        │       │   │
│  │   │ - Annotations │   │ - Features    │   │ - TensorFlow  │       │   │
│  │   └───────────────┘   └───────────────┘   └───────────────┘       │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  DALAM PROJECT INI:                                                        │
│  ─────────────────                                                         │
│                                                                             │
│  OpenCV digunakan untuk:                                                   │
│  ✓ Capture video dari webcam (cv2.VideoCapture)                           │
│  ✓ Read frame by frame (cap.read())                                       │
│  ✓ Display hasil deteksi (cv2.imshow)                                     │
│  ✓ Draw bounding boxes dan text                                           │
│  ✓ Handle keyboard input (cv2.waitKey)                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Kenapa Realtime Detection?

| Use Case              | Aplikasi                                          |
| --------------------- | ------------------------------------------------- |
| **Smart Bin**         | Kamera di tempat sampah untuk classify on-the-fly |
| **Sorting Line**      | Conveyor belt dengan kamera untuk auto-sorting    |
| **Education**         | Demo interaktif untuk awareness                   |
| **Quality Control**   | Cek kontaminasi dalam recycling                   |
| **Mobile Inspection** | Handheld device untuk field inspection            |

---

## 2. Konsep Video Processing

### 2.1 Video = Sequence of Frames

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       VIDEO PROCESSING CONCEPT                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  VIDEO = Kumpulan gambar (frames) yang ditampilkan cepat                   │
│                                                                             │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐        │
│  │ Frame  │ │ Frame  │ │ Frame  │ │ Frame  │ │ Frame  │ │ Frame  │ ...    │
│  │   1    │ │   2    │ │   3    │ │   4    │ │   5    │ │   6    │        │
│  │        │ │        │ │        │ │        │ │        │ │        │        │
│  │ t=0ms  │ │ t=33ms │ │ t=66ms │ │ t=100ms│ │ t=133ms│ │ t=166ms│        │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └────────┘        │
│                                                                             │
│  FPS (Frames Per Second):                                                  │
│  - 30 FPS = 30 gambar per detik = 33.3ms per frame                        │
│  - 60 FPS = 60 gambar per detik = 16.6ms per frame                        │
│                                                                             │
│  REALTIME PROCESSING CHALLENGE:                                            │
│  ────────────────────────────────                                          │
│                                                                             │
│  Frame Arrives → Process (Detection) → Display → Next Frame               │
│       │                  │                │            │                   │
│       │                  │                │            │                   │
│       └──────────────────┼────────────────┼────────────┘                   │
│                          │                │                                 │
│                    Must complete within frame interval!                    │
│                                                                             │
│  Contoh: 30 FPS → harus selesai dalam 33ms                                │
│  Jika detection butuh 50ms → FPS turun ke ~20 FPS                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       REALTIME DETECTION PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐                                                          │
│  │   WEBCAM     │                                                          │
│  │              │                                                          │
│  │  [Capture]   │                                                          │
│  └──────┬───────┘                                                          │
│         │                                                                   │
│         │ Raw Frame (BGR)                                                  │
│         ▼                                                                   │
│  ┌──────────────┐                                                          │
│  │ PREPROCESSING│                                                          │
│  │              │                                                          │
│  │ - Resize     │  (Optional: reduce resolution for speed)                │
│  │ - Color conv │  (BGR → RGB for YOLO)                                   │
│  └──────┬───────┘                                                          │
│         │                                                                   │
│         │ Processed Frame                                                  │
│         ▼                                                                   │
│  ┌──────────────┐                                                          │
│  │ YOLO MODEL   │                                                          │
│  │              │                                                          │
│  │ Inference    │  → Bottleneck! (20-100ms depending on model)            │
│  │              │                                                          │
│  └──────┬───────┘                                                          │
│         │                                                                   │
│         │ Detections (boxes, classes, confidence)                          │
│         ▼                                                                   │
│  ┌──────────────┐                                                          │
│  │ ANNOTATION   │                                                          │
│  │              │                                                          │
│  │ - Draw boxes │                                                          │
│  │ - Add labels │                                                          │
│  │ - Add info   │                                                          │
│  └──────┬───────┘                                                          │
│         │                                                                   │
│         │ Annotated Frame                                                  │
│         ▼                                                                   │
│  ┌──────────────┐                                                          │
│  │   DISPLAY    │                                                          │
│  │              │                                                          │
│  │ cv2.imshow() │ → Show window with result                               │
│  │              │                                                          │
│  └──────────────┘                                                          │
│                                                                             │
│  Loop terus sampai user tekan 'q' untuk quit                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Analisis detect.py Line-by-Line

### 3.1 Import dan Setup

```python
#!/usr/bin/env python3
"""
Klasifikasi Sampah Anorganik Menggunakan Algoritma YOLO
========================================================
Realtime detection menggunakan webcam.

Usage:
    python detect.py

Controls:
    q - Quit
    s - Screenshot current frame
    + - Increase confidence threshold
    - - Decrease confidence threshold
"""

import cv2                    # OpenCV untuk video processing
import numpy as np           # Numerical operations
from pathlib import Path     # Path handling
from datetime import datetime  # Untuk timestamp screenshot
import time                  # FPS calculation

# YOLO
from ultralytics import YOLO

# Configuration
MODEL_PATH = './models/best_model.pt'
DEFAULT_CONF = 0.25          # Default confidence threshold
WINDOW_NAME = 'Waste Classification - YOLO'
```

### 3.2 Color Mapping untuk Classes

```python
# Warna untuk setiap class (BGR format karena OpenCV)
# Note: OpenCV menggunakan BGR, bukan RGB!
CLASS_COLORS = {
    'battery': (0, 0, 255),      # Red (berbahaya)
    'biological': (0, 255, 0),    # Green (organik)
    'cardboard': (139, 90, 43),   # Brown
    'clothes': (255, 0, 255),     # Magenta
    'glass': (255, 255, 0),       # Cyan
    'metal': (128, 128, 128),     # Gray
    'paper': (255, 255, 255),     # White
    'plastic': (0, 255, 255),     # Yellow
    'shoes': (0, 165, 255),       # Orange
    'trash': (50, 50, 50),        # Dark gray
}

# Default color jika class tidak dikenal
DEFAULT_COLOR = (0, 255, 0)  # Green
```

**Perbedaan BGR vs RGB:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       BGR vs RGB                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  RGB (Most libraries: PIL, matplotlib, etc.)                               │
│  ───────────────────────────────────────────                               │
│                                                                             │
│  (255, 0, 0)   = RED                                                       │
│  (0, 255, 0)   = GREEN                                                     │
│  (0, 0, 255)   = BLUE                                                      │
│                                                                             │
│  BGR (OpenCV)                                                              │
│  ────────────                                                              │
│                                                                             │
│  (0, 0, 255)   = RED    ← Reversed!                                       │
│  (0, 255, 0)   = GREEN  ← Same                                            │
│  (255, 0, 0)   = BLUE   ← Reversed!                                       │
│                                                                             │
│  KENAPA BGR?                                                               │
│  ───────────                                                               │
│  Historical reason - early camera manufacturers stored colors as BGR       │
│  OpenCV kept this for backward compatibility                               │
│                                                                             │
│  CONVERSION:                                                               │
│  ──────────                                                                │
│  rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)                   │
│  bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Load Model

```python
def load_model():
    """
    Load YOLO model dari file.

    Returns:
        YOLO model instance atau None jika gagal
    """
    model_path = Path(MODEL_PATH)

    if not model_path.exists():
        print(f"❌ Error: Model tidak ditemukan di {MODEL_PATH}")
        print("   Jalankan 'python train.py' terlebih dahulu")
        return None

    print(f"📥 Loading model: {MODEL_PATH}")
    model = YOLO(str(model_path))
    print("✅ Model loaded successfully")

    return model
```

### 3.4 Main Detection Loop

```python
def run_detection():
    """
    Main function untuk realtime detection.

    Flow:
    1. Load model
    2. Open webcam
    3. Loop: capture → detect → display
    4. Handle keyboard input
    5. Cleanup
    """

    # Load model
    model = load_model()
    if model is None:
        return

    # Open webcam
    # 0 = default camera, 1 = external camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Tidak bisa membuka webcam")
        return

    # Set camera resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Get actual camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_camera = cap.get(cv2.CAP_PROP_FPS)

    print(f"📹 Camera: {width}x{height} @ {fps_camera} FPS")

    # Variables for FPS calculation
    confidence = DEFAULT_CONF
    frame_count = 0
    start_time = time.time()
    fps_display = 0.0

    print("\n🎮 Controls:")
    print("   q - Quit")
    print("   s - Screenshot")
    print("   +/= - Increase confidence")
    print("   -/_ - Decrease confidence")
    print("\n" + "="*50)
```

### 3.5 Frame Processing Loop

```python
    # Main loop
    while True:
        # Capture frame
        ret, frame = cap.read()

        if not ret:
            print("❌ Error: Gagal membaca frame")
            break

        # Run YOLO detection
        # verbose=False untuk suppress console output
        results = model(frame, conf=confidence, verbose=False)

        # Get annotated frame dari YOLO (sudah dengan bbox)
        annotated_frame = results[0].plot()

        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:  # Update setiap 1 detik
            fps_display = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()

        # Draw info overlay
        info_text = [
            f"FPS: {fps_display:.1f}",
            f"Confidence: {confidence:.2f}",
            f"Detections: {len(results[0].boxes)}",
        ]

        # Draw semi-transparent background untuk text
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (10, 10), (250, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, annotated_frame, 0.5, 0, annotated_frame)

        # Draw text
        y_offset = 30
        for text in info_text:
            cv2.putText(
                annotated_frame,
                text,
                (20, y_offset),           # Position
                cv2.FONT_HERSHEY_SIMPLEX, # Font
                0.7,                       # Font scale
                (255, 255, 255),          # Color (white)
                2,                         # Thickness
                cv2.LINE_AA               # Anti-aliased
            )
            y_offset += 25

        # Show frame
        cv2.imshow(WINDOW_NAME, annotated_frame)
```

### 3.6 Keyboard Handling

```python
        # Handle keyboard input
        # waitKey returns ASCII code of pressed key
        # & 0xFF masks to get last 8 bits (for 64-bit systems)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            # Quit
            print("\n👋 Exiting...")
            break

        elif key == ord('s'):
            # Screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, annotated_frame)
            print(f"📸 Screenshot saved: {filename}")

        elif key in [ord('+'), ord('=')]:
            # Increase confidence
            confidence = min(0.95, confidence + 0.05)
            print(f"🎚️ Confidence: {confidence:.2f}")

        elif key in [ord('-'), ord('_')]:
            # Decrease confidence
            confidence = max(0.05, confidence - 0.05)
            print(f"🎚️ Confidence: {confidence:.2f}")

    # Cleanup
    cap.release()  # Release webcam
    cv2.destroyAllWindows()  # Close all windows
    print("✅ Cleanup complete")
```

### 3.7 Main Entry Point

```python
if __name__ == "__main__":
    print("="*50)
    print("♻️  Waste Classification - Realtime Detection")
    print("="*50)
    print()

    run_detection()
```

---

## 4. Video Capture

### 4.1 cv2.VideoCapture

```python
# Berbagai sumber video
cap = cv2.VideoCapture(0)           # Webcam default
cap = cv2.VideoCapture(1)           # Webcam kedua (external)
cap = cv2.VideoCapture("video.mp4") # Video file
cap = cv2.VideoCapture("rtsp://...")  # IP camera stream

# Check if opened successfully
if not cap.isOpened():
    print("Error opening video source")
```

### 4.2 Camera Properties

```python
# GET properties
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cap.get(cv2.CAP_PROP_FOURCC)
brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
exposure = cap.get(cv2.CAP_PROP_EXPOSURE)

# SET properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)  # Disable auto exposure
```

### 4.3 Frame Reading

```python
# Read single frame
ret, frame = cap.read()

# ret: Boolean - True jika berhasil
# frame: numpy array (height, width, channels)

if ret:
    # Process frame
    print(f"Frame shape: {frame.shape}")  # e.g., (720, 1280, 3)
else:
    print("Failed to grab frame")
```

---

## 5. Frame Processing

### 5.1 Image Operations

```python
# Resize
resized = cv2.resize(frame, (640, 480))
resized = cv2.resize(frame, None, fx=0.5, fy=0.5)  # Scale by factor

# Color conversion
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Flip
flipped = cv2.flip(frame, 1)  # Horizontal flip (mirror)
flipped = cv2.flip(frame, 0)  # Vertical flip
flipped = cv2.flip(frame, -1) # Both

# Rotate
rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
rotated = cv2.rotate(frame, cv2.ROTATE_180)
```

### 5.2 Image Filtering

```python
# Blur (reduce noise)
blurred = cv2.GaussianBlur(frame, (5, 5), 0)
blurred = cv2.medianBlur(frame, 5)

# Sharpen
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpened = cv2.filter2D(frame, -1, kernel)

# Edge detection
edges = cv2.Canny(gray, 100, 200)
```

---

## 6. Drawing & Annotation

### 6.1 Basic Shapes

```python
# Rectangle (bounding box)
cv2.rectangle(
    frame,           # Image
    (x1, y1),        # Top-left corner
    (x2, y2),        # Bottom-right corner
    (0, 255, 0),     # Color (BGR)
    2                # Thickness (negatif = filled)
)

# Circle
cv2.circle(
    frame,
    (cx, cy),        # Center
    radius,          # Radius
    (255, 0, 0),     # Color
    2                # Thickness
)

# Line
cv2.line(
    frame,
    (x1, y1),        # Start point
    (x2, y2),        # End point
    (0, 0, 255),     # Color
    2                # Thickness
)

# Polygon
points = np.array([[100, 50], [200, 100], [150, 200], [50, 150]])
cv2.polylines(frame, [points], True, (0, 255, 0), 2)
```

### 6.2 Text

```python
cv2.putText(
    frame,                    # Image
    "Hello World",            # Text
    (50, 50),                 # Position (bottom-left of text)
    cv2.FONT_HERSHEY_SIMPLEX, # Font
    1.0,                      # Font scale
    (255, 255, 255),          # Color
    2,                        # Thickness
    cv2.LINE_AA               # Anti-aliased (smooth edges)
)

# Available fonts:
# FONT_HERSHEY_SIMPLEX
# FONT_HERSHEY_PLAIN
# FONT_HERSHEY_DUPLEX
# FONT_HERSHEY_COMPLEX
# FONT_HERSHEY_TRIPLEX
# FONT_HERSHEY_SCRIPT_SIMPLEX
# FONT_HERSHEY_SCRIPT_COMPLEX
# FONT_ITALIC (can be combined with others)

# Get text size (untuk positioning)
(text_width, text_height), baseline = cv2.getTextSize(
    "Hello World",
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    2
)
```

### 6.3 Semi-transparent Overlays

```python
# Create overlay dengan transparency
overlay = frame.copy()

# Draw on overlay
cv2.rectangle(overlay, (10, 10), (200, 100), (0, 0, 0), -1)  # Filled black

# Blend overlay dengan original
alpha = 0.5  # Transparency (0 = fully transparent, 1 = fully opaque)
cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
```

### 6.4 Custom Detection Drawing

```python
def draw_detection(frame, box, class_name, confidence, color):
    """
    Draw bounding box dengan label.

    Args:
        frame: Image array
        box: [x1, y1, x2, y2]
        class_name: String class name
        confidence: Float 0-1
        color: Tuple (B, G, R)
    """
    x1, y1, x2, y2 = [int(v) for v in box]

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Prepare label
    label = f"{class_name}: {confidence:.2f}"

    # Get text size
    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )

    # Draw label background
    cv2.rectangle(
        frame,
        (x1, y1 - th - 10),
        (x1 + tw + 10, y1),
        color,
        -1  # Filled
    )

    # Draw label text
    cv2.putText(
        frame,
        label,
        (x1 + 5, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),  # White text
        1,
        cv2.LINE_AA
    )

    return frame
```

---

## 7. Performance Optimization

### 7.1 FPS vs Accuracy Trade-offs

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PERFORMANCE OPTIMIZATION                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  FACTORS AFFECTING FPS:                                                    │
│  ──────────────────────                                                    │
│                                                                             │
│  1. Model Size                                                             │
│     ┌──────────────────────────────────────────────────────────────────┐  │
│     │  Model      Parameters    Speed (GPU)    Speed (CPU)    mAP     │  │
│     ├──────────────────────────────────────────────────────────────────┤  │
│     │  YOLOv8n    3.2M         0.6ms          45ms           37.3    │  │
│     │  YOLOv8s    11.2M        1.2ms          98ms           44.9    │  │
│     │  YOLOv8m    25.9M        2.5ms          187ms          50.2    │  │
│     │  YOLOv8l    43.7M        4.4ms          321ms          52.9    │  │
│     │  YOLOv8x    68.2M        7.1ms          498ms          53.9    │  │
│     └──────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  2. Input Resolution                                                       │
│     - 640x640 (default) → Balanced                                        │
│     - 320x320 → Faster, less accurate                                     │
│     - 1280x1280 → Slower, more accurate                                   │
│                                                                             │
│  3. Hardware                                                               │
│     - GPU (CUDA) → 10-50x faster than CPU                                 │
│     - CPU only → Slower but works everywhere                              │
│     - Quantization → Reduce precision for speed                           │
│                                                                             │
│  4. Frame Skip                                                             │
│     - Process every N frames                                              │
│     - Trade latency for throughput                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Optimization Techniques

```python
# 1. Use smaller model
model = YOLO('yolov8n.pt')  # Nano version

# 2. Reduce inference resolution
results = model(frame, imgsz=320)  # Instead of 640

# 3. Skip frames
frame_skip = 2
frame_count = 0

while True:
    ret, frame = cap.read()
    frame_count += 1

    if frame_count % frame_skip != 0:
        continue  # Skip this frame

    results = model(frame)

# 4. Use half precision (GPU only)
model.half()  # FP16 instead of FP32

# 5. Reduce camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

### 7.3 Threading for Better FPS

```python
import threading
from queue import Queue

class ThreadedCamera:
    """
    Separate thread untuk camera capture.
    Main thread bisa focus pada processing.
    """

    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.frame = None
        self.running = True

        # Start capture thread
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        """Continuously read frames in background."""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame

    def read(self):
        """Return latest frame."""
        return self.frame

    def stop(self):
        """Stop capture thread."""
        self.running = False
        self.thread.join()
        self.cap.release()

# Usage
camera = ThreadedCamera(0)

while True:
    frame = camera.read()
    if frame is not None:
        results = model(frame)
        cv2.imshow('Detection', results[0].plot())

    if cv2.waitKey(1) == ord('q'):
        break

camera.stop()
```

---

## 8. Multi-Camera Support

### 8.1 Multiple Cameras

```python
def multi_camera_detection():
    """
    Run detection on multiple cameras.
    """
    # Open multiple cameras
    cameras = []
    for i in range(2):  # 2 cameras
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append(cap)
            print(f"✅ Camera {i} opened")

    if not cameras:
        print("No cameras found")
        return

    model = YOLO(MODEL_PATH)

    while True:
        frames = []

        # Capture from all cameras
        for i, cap in enumerate(cameras):
            ret, frame = cap.read()
            if ret:
                # Run detection
                results = model(frame, verbose=False)
                annotated = results[0].plot()

                # Add camera label
                cv2.putText(
                    annotated, f"Camera {i}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
                )
                frames.append(annotated)

        # Stack frames horizontally
        if len(frames) >= 2:
            combined = np.hstack(frames)
        elif len(frames) == 1:
            combined = frames[0]
        else:
            continue

        cv2.imshow('Multi-Camera Detection', combined)

        if cv2.waitKey(1) == ord('q'):
            break

    # Cleanup
    for cap in cameras:
        cap.release()
    cv2.destroyAllWindows()
```

### 8.2 Picture-in-Picture

```python
def pip_display(main_frame, pip_frame, position='bottom-right', scale=0.3):
    """
    Overlay smaller frame on main frame (Picture-in-Picture).

    Args:
        main_frame: Large main frame
        pip_frame: Smaller PiP frame
        position: 'bottom-right', 'bottom-left', 'top-right', 'top-left'
        scale: Size ratio of PiP to main (0-1)

    Returns:
        Combined frame
    """
    h_main, w_main = main_frame.shape[:2]

    # Resize PiP frame
    pip_h = int(h_main * scale)
    pip_w = int(w_main * scale)
    pip_resized = cv2.resize(pip_frame, (pip_w, pip_h))

    # Calculate position
    margin = 10
    if position == 'bottom-right':
        x = w_main - pip_w - margin
        y = h_main - pip_h - margin
    elif position == 'bottom-left':
        x = margin
        y = h_main - pip_h - margin
    elif position == 'top-right':
        x = w_main - pip_w - margin
        y = margin
    else:  # top-left
        x = margin
        y = margin

    # Overlay
    result = main_frame.copy()
    result[y:y+pip_h, x:x+pip_w] = pip_resized

    # Add border
    cv2.rectangle(result, (x, y), (x+pip_w, y+pip_h), (255, 255, 255), 2)

    return result
```

---

## 9. Troubleshooting

### 9.1 Common Issues

| Masalah              | Penyebab                 | Solusi                                        |
| -------------------- | ------------------------ | --------------------------------------------- |
| Webcam tidak terbuka | Driver atau permission   | Check device manager                          |
| FPS sangat rendah    | CPU bottleneck           | Gunakan GPU atau model lebih kecil            |
| Frame delay (lag)    | Processing lambat        | Frame skipping atau threading                 |
| Window tidak muncul  | WSL atau headless server | Gunakan X server atau headless mode           |
| Color salah          | BGR/RGB mismatch         | cv2.cvtColor()                                |
| Memory leak          | Tidak release resources  | Always call release() dan destroyAllWindows() |

### 9.2 Debugging Tips

```python
# Check camera availability
def list_cameras():
    """Find available cameras."""
    available = []
    for i in range(10):  # Check first 10 indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            ret, frame = cap.read()
            if ret:
                print(f"Camera {i}: {frame.shape}")
            cap.release()
    return available

# Check OpenCV build info
print(cv2.getBuildInformation())

# Check if CUDA is available
print(f"CUDA available: {cv2.cuda.getCudaEnabledDeviceCount() > 0}")
```

---

## 10. Latihan

### Latihan 1: Custom Overlay

Tambahkan overlay yang menampilkan:

- Waktu sekarang
- Total deteksi dalam 10 detik terakhir
- Kelas yang paling sering terdeteksi

### Latihan 2: Recording

Implementasi fitur untuk merekam video dengan deteksi:

```python
# Hint: Gunakan cv2.VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
out.write(frame)
out.release()
```

### Latihan 3: ROI (Region of Interest)

Modifikasi agar deteksi hanya dilakukan pada area tertentu:

```python
# Hint: Crop frame sebelum detection
roi = frame[y1:y2, x1:x2]
results = model(roi)
```

---

**Selamat! Anda telah menyelesaikan Modul 07: Realtime Detection dengan OpenCV**

_Lanjut ke: [Modul 08 - Utility Functions](./08-utility-functions.md)_
