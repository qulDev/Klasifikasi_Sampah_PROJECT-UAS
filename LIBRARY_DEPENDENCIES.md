# Library & Dependencies

## 📚 Daftar Lengkap Library yang Digunakan

Proyek Klasifikasi Sampah menggunakan berbagai library Python untuk deep learning, computer vision, dan data processing.

---

## 🎯 Core Libraries (Wajib)

### 1. **Ultralytics** (YOLOv8)
```bash
pip install ultralytics
```

**Versi:** 8.0.0+  
**Fungsi:** Framework utama untuk YOLOv8 object detection  
**Digunakan di:** `train.py`, `detect.py`, `scan_image.ipynb`

**Fitur yang dipakai:**
- `YOLO()` - Load dan inference model
- `model.train()` - Training model
- `model.predict()` - Deteksi objek
- `model.val()` - Validasi model

**Contoh:**
```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8s.pt')

# Training
model.train(data='data.yaml', epochs=100)

# Inference
results = model.predict('image.jpg')
```

**Website:** https://github.com/ultralytics/ultralytics

---

### 2. **PyTorch** (torch)
```bash
pip install torch torchvision
```

**Versi:** 2.0.0+  
**Fungsi:** Deep learning framework backend untuk YOLOv8  
**Digunakan di:** Semua script (dependency dari Ultralytics)

**Fitur yang dipakai:**
- Tensor operations
- GPU acceleration (CUDA)
- Automatic differentiation
- Neural network modules

**Contoh:**
```python
import torch

# Check CUDA
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Tensor operations
tensor = torch.tensor([1, 2, 3])
```

**Website:** https://pytorch.org/

---

### 3. **OpenCV** (cv2)
```bash
pip install opencv-python
```

**Versi:** 4.8.0+  
**Fungsi:** Computer vision untuk image/video processing  
**Digunakan di:** `detect.py`, `convert_datasets.py`

**Fitur yang dipakai:**
- `cv2.VideoCapture()` - Baca video dari webcam
- `cv2.imread()` - Baca gambar
- `cv2.imwrite()` - Save gambar
- `cv2.rectangle()` - Gambar bounding box
- `cv2.putText()` - Tulis teks di gambar
- `cv2.resize()` - Resize gambar
- `cv2.cvtColor()` - Convert color space

**Contoh:**
```python
import cv2

# Read image
img = cv2.imread('image.jpg')

# Draw rectangle
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Show image
cv2.imshow('Image', img)
cv2.waitKey(0)
```

**Website:** https://opencv.org/

---

### 4. **NumPy**
```bash
pip install numpy
```

**Versi:** 1.24.0+  
**Fungsi:** Array operations dan mathematical computations  
**Digunakan di:** Semua script

**Fitur yang dipakai:**
- Array manipulation
- Mathematical operations
- Random number generation
- Linear algebra

**Contoh:**
```python
import numpy as np

# Create array
arr = np.array([1, 2, 3, 4, 5])

# Operations
mean = np.mean(arr)
std = np.std(arr)

# Random
random_array = np.random.rand(10)
```

**Website:** https://numpy.org/

---

### 5. **Pillow (PIL)**
```bash
pip install pillow
```

**Versi:** 10.0.0+  
**Fungsi:** Image processing dan manipulation  
**Digunakan di:** `scan_image.ipynb`, `convert_datasets.py`

**Fitur yang dipakai:**
- `Image.open()` - Baca gambar
- `Image.save()` - Save gambar
- `Image.resize()` - Resize gambar
- Format conversion (JPEG, PNG, dll)

**Contoh:**
```python
from PIL import Image

# Open image
img = Image.open('image.jpg')

# Resize
img_resized = img.resize((640, 640))

# Save
img_resized.save('resized.jpg')
```

**Website:** https://pillow.readthedocs.io/

---

## 📊 Data Processing Libraries

### 6. **Pandas**
```bash
pip install pandas
```

**Versi:** 2.0.0+  
**Fungsi:** Data manipulation dan analysis  
**Digunakan di:** `split_and_prep.py`, `convert_datasets.py`

**Fitur yang dipakai:**
- DataFrame operations
- CSV reading/writing
- Data filtering dan sorting
- Statistical analysis

**Contoh:**
```python
import pandas as pd

# Read CSV
df = pd.read_csv('data.csv')

# Filter data
filtered = df[df['class'] == 'plastic']

# Group by
grouped = df.groupby('class').size()
```

**Website:** https://pandas.pydata.org/

---

### 7. **tqdm**
```bash
pip install tqdm
```

**Versi:** 4.65.0+  
**Fungsi:** Progress bar untuk loops  
**Digunakan di:** `convert_datasets.py`, `split_and_prep.py`

**Fitur yang dipakai:**
- Progress bar untuk iterasi
- Estimated time remaining
- Custom descriptions

**Contoh:**
```python
from tqdm import tqdm

# Progress bar for loop
for i in tqdm(range(100), desc="Processing"):
    # Do something
    pass

# Progress bar for list
for item in tqdm(items, desc="Converting"):
    process(item)
```

**Website:** https://github.com/tqdm/tqdm

---

## 🎨 Visualization Libraries

### 8. **Matplotlib**
```bash
pip install matplotlib
```

**Versi:** 3.7.0+  
**Fungsi:** Plotting dan visualization  
**Digunakan di:** `scan_image.ipynb`, training results visualization

**Fitur yang dipakai:**
- `plt.imshow()` - Display images
- `plt.plot()` - Line plots
- `plt.bar()` - Bar charts
- `plt.figure()` - Create figures

**Contoh:**
```python
import matplotlib.pyplot as plt

# Display image
plt.imshow(img)
plt.title('Detected Objects')
plt.axis('off')
plt.show()

# Plot training curves
plt.plot(epochs, loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
```

**Website:** https://matplotlib.org/

---

## 📓 Jupyter Notebook Libraries

### 9. **IPython**
```bash
pip install ipython
```

**Versi:** 8.0.0+  
**Fungsi:** Interactive Python shell dan widgets  
**Digunakan di:** `scan_image.ipynb`

**Fitur yang dipakai:**
- IPython display functions
- Jupyter widgets
- Interactive output

**Contoh:**
```python
from IPython.display import display, Image

# Display image
display(Image(filename='result.jpg'))
```

**Website:** https://ipython.org/

---

### 10. **ipywidgets**
```bash
pip install ipywidgets
```

**Versi:** 8.0.0+  
**Fungsi:** Interactive widgets untuk Jupyter  
**Digunakan di:** `scan_image.ipynb`

**Fitur yang dipakai:**
- `FileUpload` - Upload file widget
- Observer pattern untuk auto-detect
- Interactive UI components

**Contoh:**
```python
from ipywidgets import FileUpload

# Create upload widget
upload = FileUpload(accept='image/*', multiple=False)

# Observer
def on_upload(change):
    # Process uploaded file
    pass

upload.observe(on_upload, names='value')
```

**Website:** https://ipywidgets.readthedocs.io/

---

## 🔧 Utility Libraries

### 11. **pathlib** (Built-in)
```python
from pathlib import Path
```

**Fungsi:** Object-oriented filesystem paths  
**Digunakan di:** Semua script

**Fitur yang dipakai:**
- Path manipulation
- File operations
- Directory operations
- Cross-platform compatibility

**Contoh:**
```python
from pathlib import Path

# Create path
data_dir = Path('datasets/processed')

# Check existence
if data_dir.exists():
    print("Directory exists")

# List files
images = list(data_dir.glob('*.jpg'))

# Create directory
output_dir = Path('output')
output_dir.mkdir(exist_ok=True)
```

---

### 12. **json** (Built-in)
```python
import json
```

**Fungsi:** JSON parsing dan serialization  
**Digunakan di:** `convert_datasets.py` (COCO format)

**Fitur yang dipakai:**
- Parse JSON annotations
- Load/dump JSON files

**Contoh:**
```python
import json

# Load JSON
with open('annotations.json', 'r') as f:
    data = json.load(f)

# Save JSON
with open('output.json', 'w') as f:
    json.dump(data, f, indent=2)
```

---

### 13. **shutil** (Built-in)
```python
import shutil
```

**Fungsi:** High-level file operations  
**Digunakan di:** `convert_datasets.py`, `split_and_prep.py`

**Fitur yang dipakai:**
- `shutil.copy2()` - Copy files with metadata
- `shutil.move()` - Move files
- `shutil.rmtree()` - Remove directory tree

**Contoh:**
```python
import shutil

# Copy file
shutil.copy2('source.jpg', 'dest.jpg')

# Remove directory
shutil.rmtree('temp_dir')
```

---

### 14. **xml.etree.ElementTree** (Built-in)
```python
import xml.etree.ElementTree as ET
```

**Fungsi:** XML parsing (VOC format)  
**Digunakan di:** `convert_datasets.py`

**Fitur yang dipakai:**
- Parse VOC XML annotations
- Extract bounding box coordinates

**Contoh:**
```python
import xml.etree.ElementTree as ET

# Parse XML
tree = ET.parse('annotation.xml')
root = tree.getroot()

# Find elements
for obj in root.findall('object'):
    name = obj.find('name').text
    bbox = obj.find('bndbox')
```

---

### 15. **hashlib** (Built-in)
```python
import hashlib
```

**Fungsi:** Hashing untuk deduplication  
**Digunakan di:** `split_and_prep.py`

**Fitur yang dipakai:**
- MD5 hashing untuk detect duplicate images

**Contoh:**
```python
import hashlib

def hash_file(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()
```

---

### 16. **logging** (Built-in)
```python
import logging
```

**Fungsi:** Logging untuk debugging dan monitoring  
**Digunakan di:** `train.py`, `convert_datasets.py`

**Fitur yang dipakai:**
- Log training progress
- Error logging
- Info messages

**Contoh:**
```python
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train.log'),
        logging.StreamHandler()
    ]
)

# Log messages
logging.info("Training started")
logging.warning("GPU not available")
logging.error("File not found")
```

---

### 17. **argparse** (Built-in)
```python
import argparse
```

**Fungsi:** Command-line argument parsing  
**Digunakan di:** Semua script

**Fitur yang dipakai:**
- Parse command-line arguments
- Default values
- Help messages

**Contoh:**
```python
import argparse

parser = argparse.ArgumentParser(description='Train YOLOv8')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch', type=int, default=16)
args = parser.parse_args()
```

---

### 18. **datetime** (Built-in)
```python
from datetime import datetime
```

**Fungsi:** Date and time operations  
**Digunakan di:** `detect.py`, `train.py`

**Fitur yang dipakai:**
- Timestamps
- Time formatting
- Duration calculation

**Contoh:**
```python
from datetime import datetime

# Current time
now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")

# Save with timestamp
filename = f"screenshot_{timestamp}.jpg"
```

---

## 📦 Requirements File

### requirements.txt
```txt
# Core Deep Learning
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0

# Computer Vision
opencv-python>=4.8.0

# Image Processing
pillow>=10.0.0
numpy>=1.24.0

# Data Processing
pandas>=2.0.0

# Progress Bars
tqdm>=4.65.0

# Visualization
matplotlib>=3.7.0

# Jupyter
ipython>=8.0.0
ipywidgets>=8.0.0
jupyter>=1.0.0

# Optional: GPU Support
# Install CUDA-enabled PyTorch from:
# https://pytorch.org/get-started/locally/
```

### Installation Commands

**Install semua sekaligus:**
```bash
pip install -r requirements.txt
```

**Install individual:**
```bash
# Core libraries
pip install ultralytics torch torchvision opencv-python

# Data processing
pip install numpy pandas pillow tqdm

# Visualization & Jupyter
pip install matplotlib ipython ipywidgets jupyter
```

**Untuk GPU (CUDA 11.8):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 🔍 Library Dependencies Tree

```
Klasifikasi_Sampah
│
├── Ultralytics (YOLOv8)
│   ├── PyTorch (backend)
│   │   ├── NumPy
│   │   └── Pillow
│   ├── OpenCV (preprocessing)
│   ├── Matplotlib (visualization)
│   └── tqdm (progress)
│
├── Data Processing
│   ├── Pandas
│   ├── NumPy
│   └── Pillow
│
├── Computer Vision
│   ├── OpenCV
│   └── Pillow
│
├── Jupyter Notebook
│   ├── IPython
│   ├── ipywidgets
│   └── Matplotlib
│
└── Built-in Libraries
    ├── pathlib
    ├── json
    ├── xml.etree.ElementTree
    ├── shutil
    ├── hashlib
    ├── logging
    ├── argparse
    └── datetime
```

---

## 📊 Library Usage by Script

### convert_datasets.py
- ✅ ultralytics
- ✅ cv2 (OpenCV)
- ✅ numpy
- ✅ PIL (Pillow)
- ✅ pandas
- ✅ tqdm
- ✅ json (COCO)
- ✅ xml.etree.ElementTree (VOC)
- ✅ pathlib
- ✅ shutil

### split_and_prep.py
- ✅ numpy
- ✅ PIL (Pillow)
- ✅ pandas
- ✅ tqdm
- ✅ pathlib
- ✅ shutil
- ✅ hashlib

### train.py
- ✅ ultralytics (YOLO)
- ✅ torch
- ✅ logging
- ✅ argparse
- ✅ pathlib
- ✅ datetime

### detect.py
- ✅ ultralytics (YOLO)
- ✅ cv2 (OpenCV)
- ✅ numpy
- ✅ pathlib
- ✅ datetime

### scan_image.ipynb
- ✅ ultralytics (YOLO)
- ✅ PIL (Pillow)
- ✅ matplotlib
- ✅ ipywidgets
- ✅ IPython.display
- ✅ pathlib

---

## 🎓 Learning Resources

### Dokumentasi Resmi:
1. **YOLOv8:** https://docs.ultralytics.com/
2. **PyTorch:** https://pytorch.org/docs/
3. **OpenCV:** https://docs.opencv.org/
4. **NumPy:** https://numpy.org/doc/
5. **Pandas:** https://pandas.pydata.org/docs/
6. **Pillow:** https://pillow.readthedocs.io/
7. **Matplotlib:** https://matplotlib.org/stable/contents.html

### Tutorials:
1. **YOLOv8 Quick Start:** https://docs.ultralytics.com/quickstart/
2. **PyTorch Tutorials:** https://pytorch.org/tutorials/
3. **OpenCV Python Tutorial:** https://docs.opencv.org/master/d6/d00/tutorial_py_root.html

---

## ⚠️ Version Compatibility

### Python Version
- **Required:** Python 3.8+
- **Recommended:** Python 3.10 atau 3.11
- **Not supported:** Python 3.7 atau lebih lama

### CUDA Version (untuk GPU)
- **CUDA 11.8** → PyTorch 2.0.0+
- **CUDA 12.1** → PyTorch 2.1.0+

Check CUDA version:
```bash
nvidia-smi
```

### Operating System
- ✅ **Linux** (Ubuntu, Debian, etc.) - Recommended
- ✅ **Windows** 10/11
- ✅ **macOS** (CPU only, no CUDA)

---

## 🔧 Troubleshooting

### Common Issues:

**1. PyTorch CUDA not available**
```bash
# Install CUDA-enabled PyTorch
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**2. OpenCV error (libGL)**
```bash
# Ubuntu/Debian
sudo apt-get install libgl1-mesa-glx

# If still error
pip uninstall opencv-python
pip install opencv-python-headless
```

**3. Jupyter widgets not showing**
```bash
# Enable widgets
jupyter nbextension enable --py widgetsnbextension
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

**4. Out of memory (CUDA)**
- Reduce batch size: `--batch 8` atau `--batch 4`
- Use smaller model: `yolov8n.pt` instead of `yolov8s.pt`
- Close other GPU applications

---

## 📈 Performance Optimization

### GPU Acceleration
```python
import torch

# Check GPU
if torch.cuda.is_available():
    device = 'cuda'
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = 'cpu'
    print("Using CPU")
```

### Memory Management
```python
# Clear GPU cache
torch.cuda.empty_cache()

# Use half precision (faster, less memory)
model = YOLO('yolov8s.pt')
results = model.predict('image.jpg', half=True)
```

---

## 📝 Summary

### Total Libraries: 18
- **External:** 10 libraries (perlu install)
- **Built-in:** 8 libraries (sudah ada di Python)

### Install Time: ~5-10 menit
### Total Size: ~5-8 GB (with CUDA)

### Essential Libraries (Top 5):
1. 🥇 **Ultralytics** - Core YOLOv8
2. 🥈 **PyTorch** - Deep learning backend
3. 🥉 **OpenCV** - Computer vision
4. **NumPy** - Numerical computing
5. **Pillow** - Image processing

---

**Dibuat untuk:** Proyek Klasifikasi Sampah  
**Teknologi:** Python 3.8+ dengan YOLOv8  
**Last Updated:** November 2025
