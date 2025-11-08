# Library Essentials - Quick Reference

## 🚀 Quick Installation

```bash
# Install semua library sekaligus
pip install -r requirements.txt
```

---

## 📦 Core Libraries (Wajib)

| Library | Versi | Fungsi | Install |
|---------|-------|--------|---------|
| **Ultralytics** | 8.0.0+ | YOLOv8 framework | `pip install ultralytics` |
| **PyTorch** | 2.0.0+ | Deep learning backend | `pip install torch` |
| **OpenCV** | 4.8.0+ | Computer vision | `pip install opencv-python` |
| **NumPy** | 1.24.0+ | Array operations | `pip install numpy` |
| **Pillow** | 10.0.0+ | Image processing | `pip install pillow` |

---

## 📊 Data & Visualization

| Library | Fungsi | Digunakan di |
|---------|--------|--------------|
| **Pandas** | Data manipulation | `convert_datasets.py`, `split_and_prep.py` |
| **Matplotlib** | Plotting | `scan_image.ipynb`, visualization |
| **tqdm** | Progress bar | Semua script processing |

---

## 🎨 Jupyter Notebook

| Library | Fungsi |
|---------|--------|
| **IPython** | Interactive shell |
| **ipywidgets** | Upload widget |
| **jupyter** | Notebook environment |

---

## 🔧 Built-in Python (Tidak Perlu Install)

| Module | Fungsi | Digunakan di |
|--------|--------|--------------|
| `pathlib` | File paths | Semua script |
| `json` | JSON parsing | COCO format |
| `xml.etree` | XML parsing | VOC format |
| `shutil` | File operations | Copy, move files |
| `hashlib` | MD5 hashing | Deduplication |
| `logging` | Logging | Training, debugging |
| `argparse` | CLI arguments | Semua script |
| `datetime` | Timestamps | Screenshots, logs |

---

## 📋 requirements.txt

```txt
# Core
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
pillow>=10.0.0
numpy>=1.24.0

# Data & Progress
pandas>=2.0.0
tqdm>=4.65.0

# Visualization
matplotlib>=3.7.0

# Jupyter
ipython>=8.0.0
ipywidgets>=8.0.0
jupyter>=1.0.0
```

---

## 🎯 Library Usage per Script

### `convert_datasets.py`
```python
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
```

### `split_and_prep.py`
```python
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import shutil
import hashlib
```

### `train.py`
```python
from ultralytics import YOLO
import torch
import logging
import argparse
from pathlib import Path
from datetime import datetime
```

### `detect.py`
```python
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
```

### `scan_image.ipynb`
```python
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
from ipywidgets import FileUpload
from IPython.display import display
from pathlib import Path
```

---

## 🖥️ Platform-Specific

### Windows
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python
```

### Linux (Ubuntu/Debian)
```bash
sudo apt-get install libgl1-mesa-glx  # Untuk OpenCV
pip install -r requirements.txt
```

### macOS (CPU only)
```bash
pip install torch torchvision  # No CUDA
pip install -r requirements.txt
```

---

## ⚡ GPU Support (Optional)

### CUDA 11.8
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### CUDA 12.1
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Check GPU
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

---

## 🔍 Dependency Tree

```
YOLOv8 Project
│
├─ ultralytics (YOLOv8)
│  └─ torch (PyTorch)
│     ├─ numpy
│     └─ pillow
│
├─ opencv-python
│  └─ numpy
│
├─ pandas
│  └─ numpy
│
└─ matplotlib
   └─ numpy
```

---

## 🎓 Key Functions per Library

### Ultralytics (YOLOv8)
```python
model = YOLO('yolov8s.pt')              # Load model
model.train(data='data.yaml')           # Training
results = model.predict('image.jpg')    # Inference
metrics = model.val()                   # Validation
```

### PyTorch
```python
torch.cuda.is_available()               # Check GPU
torch.cuda.empty_cache()                # Clear memory
device = torch.device('cuda')           # Set device
```

### OpenCV
```python
cap = cv2.VideoCapture(0)               # Open webcam
ret, frame = cap.read()                 # Read frame
cv2.rectangle(img, (x1,y1), (x2,y2))   # Draw box
cv2.imshow('Window', img)               # Display
```

### NumPy
```python
arr = np.array([1,2,3])                 # Create array
mean = np.mean(arr)                     # Calculate mean
normalized = arr / 255.0                # Normalize
```

### Pillow
```python
img = Image.open('image.jpg')           # Open image
img_resized = img.resize((640,640))     # Resize
img.save('output.jpg')                  # Save
```

### Pandas
```python
df = pd.read_csv('data.csv')            # Read CSV
grouped = df.groupby('class').size()    # Group by
df.to_csv('output.csv')                 # Write CSV
```

---

## 📊 Size & Performance

| Library | Size | Load Time | Memory |
|---------|------|-----------|--------|
| Ultralytics | ~50 MB | ~2s | ~200 MB |
| PyTorch | ~800 MB | ~5s | ~500 MB |
| OpenCV | ~90 MB | ~1s | ~100 MB |
| NumPy | ~30 MB | <1s | ~50 MB |
| Others | ~100 MB | <1s | ~100 MB |
| **Total** | **~1 GB** | **~10s** | **~1 GB** |

*dengan CUDA: +4-5 GB*

---

## ⚠️ Common Issues & Solutions

### Issue 1: CUDA not available
```bash
# Solution
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Issue 2: OpenCV libGL error
```bash
# Solution
sudo apt-get install libgl1-mesa-glx
# or
pip install opencv-python-headless
```

### Issue 3: Jupyter widgets not showing
```bash
# Solution
jupyter nbextension enable --py widgetsnbextension
```

### Issue 4: Out of memory
```python
# Solution: reduce batch size
model.train(data='data.yaml', batch=8)  # Instead of 16
```

---

## 🎯 Minimum vs Recommended

### Minimum (CPU only)
```txt
ultralytics>=8.0.0
torch>=2.0.0
opencv-python>=4.8.0
numpy>=1.24.0
pillow>=10.0.0
```

### Recommended (GPU)
```txt
# All minimum +
pandas>=2.0.0
tqdm>=4.65.0
matplotlib>=3.7.0
jupyter>=1.0.0
```

---

## 📚 Learning Resources

### Dokumentasi:
- **YOLOv8:** https://docs.ultralytics.com/
- **PyTorch:** https://pytorch.org/docs/
- **OpenCV:** https://docs.opencv.org/

### Quick Start:
- **YOLOv8 Quickstart:** https://docs.ultralytics.com/quickstart/
- **PyTorch Tutorials:** https://pytorch.org/tutorials/
- **OpenCV Python:** https://docs.opencv.org/master/d6/d00/tutorial_py_root.html

---

## ✅ Verification

### Check all installations:
```python
import ultralytics
import torch
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

print("✅ All libraries imported successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

---

**Quick Reference untuk:** Proyek Klasifikasi Sampah  
**Total Libraries:** 10 external + 8 built-in  
**Install Time:** ~5-10 menit
