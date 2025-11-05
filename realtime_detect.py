#!/usr/bin/env python3
"""
Real-time Waste Classification with Webcam
Simple script to detect waste objects in real-time using your webcam

Usage:
    python realtime_detect.py

Controls:
    - Press 'q' to quit
    - Press 's' to save current frame
    - Press 'c' to toggle confidence display
"""

import cv2
import time
from pathlib import Path
from ultralytics import YOLO
import torch

# Configuration
MODEL_PATH = './models/best.pt'
CONFIDENCE_THRESHOLD = 0.25
CAMERA_ID = 0  # 0 for default webcam, 1 for external camera

# Colors for each class (BGR format)
CLASS_COLORS = {
    'plastic': (0, 255, 255),    # Yellow
    'metal': (255, 0, 0),         # Blue
    'glass': (0, 255, 0),         # Green
    'paper': (255, 255, 0),       # Cyan
    'cardboard': (0, 165, 255),   # Orange
    'other': (128, 128, 128)      # Gray
}

def check_cuda():
    """Check if CUDA is available and display GPU info."""
    if torch.cuda.is_available():
        print("=" * 60)
        print("🚀 CUDA ENABLED - Using GPU for faster detection!")
        print("=" * 60)
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        print("=" * 60)
        return True
    else:
        print("=" * 60)
        print("⚠️  CUDA NOT AVAILABLE - Using CPU")
        print("=" * 60)
        print("Detection will be slower on CPU.")
        print("For GPU support, install CUDA and PyTorch with CUDA.")
        print("=" * 60)
        return False

def load_model(model_path):
    """Load YOLO model and set device."""
    if not Path(model_path).exists():
        print(f"❌ Error: Model not found at {model_path}")
        print("Please train the model first: python train.py")
        return None
    
    print(f"\n📦 Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Set device (CUDA if available, else CPU)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    print(f"✓ Model loaded successfully")
    print(f"  Device: {device}")
    print(f"  Classes: {list(model.names.values())}")
    print()
    
    return model

def draw_detections(frame, results, show_conf=True):
    """Draw bounding boxes and labels on frame."""
    detection_info = []  # Store detection info for summary
    
    for result in results:
        boxes = result.boxes
        
        for idx, box in enumerate(boxes, 1):
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            
            # Get confidence and class
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            class_name = result.names[cls]
            
            # Store detection info
            detection_info.append({
                'index': idx,
                'class': class_name,
                'confidence': conf
            })
            
            # Get color for this class
            color = CLASS_COLORS.get(class_name, (255, 255, 255))
            
            # Draw thicker bounding box with object number
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw object number in top-left corner of box
            obj_num = f"#{idx}"
            cv2.circle(frame, (x1 + 15, y1 + 15), 15, color, -1)
            cv2.putText(
                frame,
                obj_num,
                (x1 + 7, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2
            )
            
            # Prepare label
            if show_conf:
                label = f"{class_name.upper()}: {conf:.2f}"
            else:
                label = class_name.upper()
            
            # Calculate label size for background
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            
            # Draw label background
            cv2.rectangle(
                frame,
                (x1, y1 - label_height - 15),
                (x1 + label_width + 15, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label,
                (x1 + 7, y1 - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2
            )
    
    return frame, detection_info

def draw_info_panel(frame, fps, device, show_conf, detection_info):
    """Draw information panel on frame."""
    height, width = frame.shape[:2]
    
    # Semi-transparent panel at top
    panel_height = 100
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Title
    cv2.putText(
        frame,
        "WASTE CLASSIFICATION - REAL-TIME",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )
    
    # FPS counter
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        fps_text,
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )
    
    # Device info
    device_color = (0, 255, 0) if 'cuda' in device else (0, 165, 255)
    device_text = f"Device: {device.upper()}"
    cv2.putText(
        frame,
        device_text,
        (150, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        device_color,
        1
    )
    
    # Object count
    obj_count = len(detection_info)
    count_text = f"Objects: {obj_count}"
    count_color = (0, 255, 255) if obj_count > 0 else (128, 128, 128)
    cv2.putText(
        frame,
        count_text,
        (300, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        count_color,
        1
    )
    
    # Controls
    controls = "Q: Quit | S: Save | C: Toggle Confidence"
    cv2.putText(
        frame,
        controls,
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (200, 200, 200),
        1
    )
    
    # Detection summary (right side)
    if detection_info:
        summary_y = 30
        for det in detection_info:
            summary_text = f"#{det['index']}: {det['class'].upper()}"
            color = CLASS_COLORS.get(det['class'], (255, 255, 255))
            cv2.putText(
                frame,
                summary_text,
                (width - 180, summary_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
            summary_y += 25
    
    return frame

def main():
    """Main function to run real-time detection."""
    print("=" * 60)
    print("🎥 REAL-TIME WASTE CLASSIFICATION")
    print("=" * 60)
    print()
    
    # Check CUDA availability
    cuda_available = check_cuda()
    
    # Load model
    model = load_model(MODEL_PATH)
    if model is None:
        return
    
    # Get device info
    device = 'cuda:0' if cuda_available else 'cpu'
    
    # Open webcam
    print(f"📹 Opening camera {CAMERA_ID}...")
    cap = cv2.VideoCapture(CAMERA_ID)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open camera {CAMERA_ID}")
        print("Try changing CAMERA_ID in the script (0 for built-in, 1 for external)")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✓ Camera opened successfully")
    print()
    print("=" * 60)
    print("🎬 STARTING REAL-TIME DETECTION")
    print("=" * 60)
    print("Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save current frame")
    print("  - Press 'c' to toggle confidence display")
    print("=" * 60)
    print()
    
    # Variables for FPS calculation
    fps = 0
    frame_count = 0
    start_time = time.time()
    show_conf = True
    save_count = 0
    
    try:
        while True:
            # Read frame from camera
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Error: Failed to read frame from camera")
                break
            
            # Run detection
            results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
            
            # Draw detections and get detection info
            frame, detection_info = draw_detections(frame, results, show_conf)
            
            # Calculate FPS
            frame_count += 1
            if frame_count >= 10:
                end_time = time.time()
                fps = frame_count / (end_time - start_time)
                frame_count = 0
                start_time = time.time()
            
            # Draw info panel with detection summary
            frame = draw_info_panel(frame, fps, device, show_conf, detection_info)
            
            # Display frame
            cv2.imshow('Waste Classification - Real-Time', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n👋 Quitting...")
                break
            elif key == ord('s'):
                # Save current frame
                save_count += 1
                filename = f"detection_capture_{save_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"💾 Frame saved as {filename}")
            elif key == ord('c'):
                # Toggle confidence display
                show_conf = not show_conf
                status = "ON" if show_conf else "OFF"
                print(f"🔄 Confidence display: {status}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Camera released")
        print("✓ Windows closed")
        print()
        print("=" * 60)
        print("👋 Thank you for using Waste Classification!")
        print("=" * 60)

if __name__ == '__main__':
    main()
