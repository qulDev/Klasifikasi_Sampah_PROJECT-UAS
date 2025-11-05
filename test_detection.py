#!/usr/bin/env python3
"""
Simple detection test script - works without display

Usage:
    python test_detection.py --weights ./models/best.pt --image ./datasets/processed/val/images/cardboard5.jpg
"""

import argparse
from pathlib import Path

from ultralytics import YOLO
from PIL import Image as PILImage

def main():
    parser = argparse.ArgumentParser(description="Test YOLO detection on a single image")
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--image', type=str, required=True, help='Path to test image')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--save', type=str, default=None, help='Path to save annotated image')
    
    args = parser.parse_args()
    
    # Check files exist
    weights_path = Path(args.weights)
    image_path = Path(args.image)
    
    if not weights_path.exists():
        print(f"❌ Model weights not found: {weights_path}")
        return 1
    
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return 1
    
    print("=" * 60)
    print("YOLO Detection Test")
    print("=" * 60)
    print(f"Model: {weights_path}")
    print(f"Image: {image_path}")
    print(f"Confidence threshold: {args.conf}")
    print("")
    
    # Load model
    print("Loading model...")
    model = YOLO(str(weights_path))
    print("✓ Model loaded")
    print(f"  Classes: {model.names}")
    print("")
    
    # Load image
    print("Loading image...")
    img = PILImage.open(image_path)
    print(f"✓ Image loaded: {img.size[0]}x{img.size[1]}")
    print("")
    
    # Run detection
    print("Running detection...")
    results = model(img, conf=args.conf, verbose=False)
    print("✓ Detection complete")
    print("")
    
    # Display results
    print("=" * 60)
    print("DETECTION RESULTS")
    print("=" * 60)
    
    detection_count = 0
    for result in results:
        boxes = result.boxes
        
        if len(boxes) == 0:
            print("⚠ No objects detected above confidence threshold")
            continue
        
        print(f"Found {len(boxes)} object(s):")
        print("")
        
        for idx, box in enumerate(boxes, 1):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            class_name = result.names[cls]
            
            print(f"  {idx}. {class_name.upper()}")
            print(f"     Confidence: {conf:.3f}")
            print(f"     BBox: ({int(x1)}, {int(y1)}) to ({int(x2)}, {int(y2)})")
            print("")
            
            detection_count += 1
    
    print("=" * 60)
    print(f"Total detections: {detection_count}")
    print("=" * 60)
    
    # Save annotated image if requested
    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get annotated image from results
        annotated = results[0].plot()
        
        # Convert BGR to RGB (OpenCV to PIL)
        annotated_rgb = annotated[:, :, ::-1]  # BGR to RGB
        img_annotated = PILImage.fromarray(annotated_rgb)
        img_annotated.save(save_path)
        
        print(f"✓ Annotated image saved to: {save_path}")
    
    return 0

if __name__ == '__main__':
    exit(main())
