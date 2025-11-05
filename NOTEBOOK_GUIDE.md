# 📓 Jupyter Notebook Guide - Fixed Model Path Issue

## ✅ Issue Resolved!

**Problem:** Notebook couldn't find the model at `./models/best.pt`

**Root Cause:** When running notebooks from the `notebooks/` directory, the relative path needs to go up one level.

**Solution:** Changed model path from `./models/best.pt` → `../models/best.pt`

---

## 🚀 How to Use the Notebook

### Option 1: Using Jupyter Notebook (Recommended)

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Start Jupyter Notebook
jupyter notebook

# 3. Navigate to: notebooks/scan_image.ipynb

# 4. Run all cells (Cell → Run All)
```

### Option 2: Using Jupyter Lab

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Install JupyterLab if not installed
pip install jupyterlab

# 3. Start Jupyter Lab
jupyter lab

# 4. Open: notebooks/scan_image.ipynb
```

### Option 3: Using VS Code

```bash
# 1. Open the notebook in VS Code
# File → Open → notebooks/scan_image.ipynb

# 2. Select kernel: .venv (Python 3.x)
# Click "Select Kernel" in the top right

# 3. Run cells one by one or "Run All"
```

---

## 🔧 Verify Setup Before Running

Run this diagnostic script to check if everything is ready:

```bash
cd notebooks
../.venv/bin/python test_notebook_model.py
```

**Expected output:**
```
✅ SUCCESS! Model loaded
   Classes: {0: 'plastic', 1: 'metal', 2: 'glass', 3: 'paper', 4: 'cardboard', 5: 'other'}
   Number of classes: 6
```

---

## 📝 Notebook Cells Overview

1. **Cell 1** (Markdown): Introduction
2. **Cell 2** (Python): Import libraries
3. **Cell 3** (Python): **NEW** - Verify paths and model location
4. **Cell 4** (Python): Load YOLO model (now uses `../models/best.pt`)
5. **Cell 5** (Python): Create file upload widget
6. **Cell 6** (Python): Display uploaded image
7. **Cell 7** (Python): Run detection
8. **Cell 8** (Python): Visualize results with bounding boxes
9. **Cell 9** (Python): Display results table
10. **Cell 10** (Markdown): Usage instructions

---

## 🎯 Step-by-Step Usage

### 1. Start Jupyter and Open Notebook
```bash
source .venv/bin/activate
jupyter notebook
# Browser will open → Navigate to notebooks/scan_image.ipynb
```

### 2. Run Setup Cells (Cells 1-4)
- **Cell 2**: Import libraries (should print "✓ All libraries loaded successfully")
- **Cell 3**: Verify paths (should show "✓ Use '../models/best.pt'")
- **Cell 4**: Load model (should print "✓ Model loaded successfully")

### 3. Upload and Test Image (Cells 5-9)
- **Cell 5**: Click "Upload Image" button, select a waste image
- **Cell 6**: View your uploaded image
- **Cell 7**: Run detection (prints "✓ Detection complete")
- **Cell 8**: See image with bounding boxes
- **Cell 9**: View detection results table

---

## 🖼️ Test with Sample Images

You can test with the validation images:

1. When prompted to upload, navigate to:
   ```
   /home/remus/Development/Python/Klasifikasi_Sampah/datasets/processed/val/images/
   ```

2. Select one of:
   - `cardboard5.jpg` (verified: 98.1% confidence)
   - `cardboard30.jpg` (verified: 76.4% confidence)
   - `cardboard57.jpg`
   - `cardboard61.jpg`

---

## 🐛 Troubleshooting

### Issue: "Model not found"

**Check 1:** Verify model exists
```bash
ls -lh ../models/best.pt
# Should show ~21 MB file
```

**Check 2:** Run diagnostic
```bash
cd notebooks
../.venv/bin/python test_notebook_model.py
```

**Check 3:** Verify you're in notebooks/ directory
```python
# Run this in a notebook cell
import os
print(os.getcwd())
# Should end with /Klasifikasi_Sampah/notebooks
```

### Issue: "No module named 'ipywidgets'"

**Solution:** Install dependencies
```bash
source .venv/bin/activate
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

### Issue: Upload widget not showing

**Solution:** Enable ipywidgets extension
```bash
source .venv/bin/activate
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension --sys-prefix
```

### Issue: Kernel keeps restarting

**Solution:** Check memory usage, try smaller images or restart Jupyter
```bash
# Restart Jupyter
# Press Ctrl+C in terminal, then restart with:
jupyter notebook
```

---

## 💡 Tips for Best Results

1. **Image Quality:**
   - Use well-lit images
   - Clear focus on waste items
   - Avoid blurry or dark images

2. **Confidence Threshold:**
   - Default: 0.25 (balanced)
   - Lower (0.15): More detections, may include false positives
   - Higher (0.4): Fewer detections, more accurate

3. **Multiple Objects:**
   - The model can detect multiple waste items in one image
   - Each will get its own bounding box and label

4. **Performance:**
   - GPU: Fast inference (~30 FPS)
   - CPU: Slower but works (~3-5 FPS)

---

## 📊 Example Detection Output

**Input:** `cardboard5.jpg` (512x384 pixels)

**Output:**
```
Found 1 object(s):

  1. CARDBOARD
     Confidence: 0.981
     BBox: (0, 0) to (512, 384)
```

**Visual:** Green bounding box with label "cardboard: 0.98"

---

## 🔗 Related Files

- **Notebook:** `notebooks/scan_image.ipynb`
- **Model:** `models/best.pt` (21 MB)
- **Test Script:** `notebooks/test_notebook_model.py`
- **Validation Images:** `datasets/processed/val/images/`

---

## ✅ Quick Test Checklist

Before using the notebook, verify:

- [ ] Virtual environment activated (`source .venv/bin/activate`)
- [ ] Jupyter installed (`jupyter --version`)
- [ ] Model exists (`ls -lh models/best.pt`)
- [ ] Can run diagnostic (`cd notebooks && ../.venv/bin/python test_notebook_model.py`)
- [ ] Notebook opens without errors
- [ ] Cell 2 runs successfully (imports work)
- [ ] Cell 3 shows "✓ Use '../models/best.pt'"
- [ ] Cell 4 loads model successfully

---

**🎉 Your notebook is ready to use! The model path has been fixed from `./models/best.pt` to `../models/best.pt`**
