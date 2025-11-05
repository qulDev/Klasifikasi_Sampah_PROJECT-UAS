# Quick Start Guide 🚀

## ✅ Step 7 is Working!

Your model has been trained successfully. Here's how to use it:

### 🎯 Test Detection (Simple & Fast)

```bash
# Activate virtual environment
source .venv/bin/activate

# Test on a single image
python test_detection.py \
    --weights ./models/best.pt \
    --image ./datasets/processed/val/images/cardboard5.jpg \
    --save output.jpg

# View the results
# The script will print detections in the terminal
# The annotated image will be saved as output.jpg
```

### 📸 Available Test Images

```bash
# Test on different waste categories
python test_detection.py --weights ./models/best.pt --image ./datasets/processed/val/images/cardboard5.jpg --save cardboard_detected.jpg
python test_detection.py --weights ./models/best.pt --image ./datasets/processed/val/images/cardboard30.jpg --save cardboard30_detected.jpg
python test_detection.py --weights ./models/best.pt --image ./datasets/processed/val/images/cardboard57.jpg --save cardboard57_detected.jpg
```

### 🎥 Real-time Webcam Detection (NEW!)

**Super Simple - Test with Your Real Waste Samples!**

```bash
# Activate venv first
source .venv/bin/activate

# Run real-time detection - ONE COMMAND!
python realtime_detect.py

# That's it! Camera opens automatically
# Hold your waste in front of camera
# Press 'Q' to quit, 'S' to save frame
```

**Features:**
- ⚡ Auto-detects CUDA/GPU for faster performance
- 🎨 Color-coded detection (Yellow=Plastic, Blue=Metal, Green=Glass, etc.)
- 📊 Live FPS counter
- 💾 Save frames with 'S' key
- 🎮 Toggle confidence with 'C' key

**See REALTIME_GUIDE.md for full instructions!**

**Advanced Options (more control):**
```bash
python detect_webcam.py --weights ./models/best.pt --source 0 --conf 0.25
```

### 📹 Video File Detection

```bash
source .venv/bin/activate

python detect_webcam.py \
    --weights ./models/best.pt \
    --source /path/to/your/video.mp4 \
    --conf 0.25 \
    --save-txt \
    --save-img
```

---

## 🔧 Common Issues & Solutions

### Issue: `ModuleNotFoundError: No module named 'cv2'`

**Solution:** Activate the virtual environment first!

```bash
# Always run this before executing scripts:
source .venv/bin/activate

# Or use the venv Python directly:
.venv/bin/python test_detection.py --weights ./models/best.pt --image ./datasets/processed/val/images/cardboard5.jpg
```

### Issue: No display / headless environment

**Solution:** Use `test_detection.py` instead of `detect_webcam.py`

```bash
# This works without X11/display:
python test_detection.py --weights ./models/best.pt --image ./datasets/processed/val/images/cardboard5.jpg --save output.jpg

# This requires display:
python detect_webcam.py --weights ./models/best.pt --source 0
```

### Issue: Low confidence detections

**Solution:** Adjust the confidence threshold

```bash
# Lower threshold (more detections, some false positives)
python test_detection.py --weights ./models/best.pt --image ./path/to/image.jpg --conf 0.1

# Higher threshold (fewer detections, more accurate)
python test_detection.py --weights ./models/best.pt --image ./path/to/image.jpg --conf 0.5
```

---

## 📊 Your Model Info

- **Model:** YOLOv8 (`./models/best.pt`)
- **Classes:** plastic, metal, glass, paper, cardboard, other
- **Latest Training:** `./models/20251105_024145_best.pt` (23 MB)
- **Previous Training:** `./models/20251104_212405_best.pt` (6.2 MB)

---

## 🎨 Example Detection Result

**Tested on:** `cardboard5.jpg`
**Result:**
```
Found 1 object(s):

  1. CARDBOARD
     Confidence: 0.981
     BBox: (0, 0) to (512, 384)
```

**Command used:**
```bash
python test_detection.py --weights ./models/best.pt --image ./datasets/processed/val/images/cardboard5.jpg --save test_output.jpg
```

---

## 📝 Next Steps

1. **Test on your own images:**
   ```bash
   python test_detection.py --weights ./models/best.pt --image /path/to/your/image.jpg --save result.jpg
   ```

2. **Use in Jupyter Notebook:**
   ```bash
   source .venv/bin/activate
   jupyter notebook
   # Navigate to: notebooks/scan_image.ipynb
   # Run all cells and upload your images!
   ```
   
   ⚠️ **Note:** The notebook uses `../models/best.pt` (relative to notebooks/ directory)
   
   See **NOTEBOOK_GUIDE.md** for detailed instructions!

3. **Integrate into your application:**
   ```python
   from ultralytics import YOLO
   
   model = YOLO('./models/best.pt')
   results = model('/path/to/image.jpg')
   
   for result in results:
       boxes = result.boxes
       for box in boxes:
           cls = int(box.cls[0])
           conf = float(box.conf[0])
           class_name = result.names[cls]
           print(f"{class_name}: {conf:.3f}")
   ```

---

## 💡 Tips

- **Always activate venv:** `source .venv/bin/activate`
- **Use test_detection.py for quick tests** (no display needed)
- **Use detect_webcam.py for real-time** (requires display)
- **Lower confidence for more detections:** `--conf 0.1`
- **Higher confidence for accuracy:** `--conf 0.5`
- **Save results:** `--save output.jpg`

---

**✅ Everything is working! Your model detected cardboard with 98.1% confidence!**
