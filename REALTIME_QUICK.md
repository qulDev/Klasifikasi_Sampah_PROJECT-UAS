# 🚀 REAL-TIME DETECTION - ONE COMMAND

## Super Simple Usage

```bash
source .venv/bin/activate
python realtime_detect.py
```

**That's it!** Camera opens automatically.

---

## Controls

| Key | Action |
|-----|--------|
| **Q** | Quit |
| **S** | Save frame |
| **C** | Toggle confidence |

---

## What You'll See

```
============================================================
🚀 CUDA ENABLED - Using GPU for faster detection!
============================================================
GPU Device: NVIDIA GeForce RTX 3060
============================================================

📦 Loading model...
✓ Model loaded successfully

📹 Opening camera 0...
✓ Camera opened successfully

🎬 STARTING REAL-TIME DETECTION
```

**Then:** Live video with colored boxes around waste objects!

---

## Color Codes

- 🟡 Yellow = PLASTIC
- 🔵 Blue = METAL
- 🟢 Green = GLASS
- 🔵 Cyan = PAPER
- 🟠 Orange = CARDBOARD
- ⚫ Gray = OTHER

---

## Testing Your Waste

1. Hold plastic bottle → Yellow "PLASTIC: 0.95"
2. Hold metal can → Blue "METAL: 0.88"
3. Hold glass jar → Green "GLASS: 0.92"
4. Press 'S' to save good detections
5. Press 'Q' when done

---

## Performance

- **GPU (RTX 3060)**: 30-60 FPS ⚡ (Recommended)
- **CPU**: 3-8 FPS 🐌 (Still works)

---

## Troubleshooting

**Camera won't open?**
```python
# Edit realtime_detect.py line 16:
CAMERA_ID = 1  # Try 1 instead of 0
```

**Low FPS?**
```python
# Edit realtime_detect.py line 15:
CONFIDENCE_THRESHOLD = 0.35  # Increase from 0.25
```

**No CUDA?**
```bash
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

---

## Full Documentation

- **REALTIME_GUIDE.md** - Complete guide with all options
- **QUICKSTART.md** - All detection methods
- **NOTEBOOK_GUIDE.md** - Jupyter notebook usage

---

**🎉 Just run it and test your waste samples!**

```bash
python realtime_detect.py
```
