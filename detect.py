#!/usr/bin/env python3
"""
Real-time Waste Detection - Simplified

Usage: python detect.py
"""

import cv2
import time
from pathlib import Path
from ultralytics import YOLO
import torch

# Config
MODEL = './models/best.pt'
CONF = 0.25
CAM = 0

COLORS = {'plastic': (0,255,255), 'metal': (255,0,0), 'glass': (0,255,0),
          'paper': (255,255,0), 'cardboard': (0,165,255), 'other': (128,128,128)}

def load():
    if not Path(MODEL).exists():
        print(f"❌ Model not found: {MODEL}\n   Run: python train.py")
        return None
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = YOLO(MODEL).to(device)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"✓ Model loaded on {gpu}")
    return model, device

def draw(frame, box, idx, model, show_conf):
    x1,y1,x2,y2 = map(int, box.xyxy[0].cpu().numpy())
    conf, cls = float(box.conf[0]), int(box.cls[0])
    name = model.names[cls]
    color = COLORS.get(name, (255,255,255))
    
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 3)
    cv2.circle(frame, (x1+15,y1+15), 15, color, -1)
    cv2.putText(frame, f"#{idx}", (x1+7,y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    
    label = f"{name.upper()}: {conf:.2f}" if show_conf else name.upper()
    (w,h),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x1,y1-h-15), (x1+w+15,y1), color, -1)
    cv2.putText(frame, label, (x1+7,y1-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    return name

def info(frame, fps, device, dets):
    h,w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (w,100), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    cv2.putText(frame, "WASTE CLASSIFICATION", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(frame, f"FPS:{fps:.1f}|{device.upper()}|Objects:{len(dets)}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(frame, "Q:Quit|S:Save|C:Confidence", (10,75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
    
    if dets:
        y=30
        for i,n in enumerate(dets,1):
            cv2.putText(frame, f"#{i}:{n.upper()}", (w-180,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS.get(n,(255,255,255)), 2)
            y+=25

def main():
    print("="*50)
    print("🎥 REAL-TIME WASTE CLASSIFICATION")
    print("="*50)
    
    r = load()
    if not r: return
    model, device = r
    
    print(f"📹 Opening camera {CAM}...")
    cap = cv2.VideoCapture(CAM)
    if not cap.isOpened():
        print(f"❌ Could not open camera {CAM}")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("✓ Ready\nControls: Q=Quit|S=Save|C=Confidence\n")
    
    fps,fc,st = 0,0,time.time()
    show,save = True,0
    
    try:
        while True:
            ret,frame = cap.read()
            if not ret: break
            
            results = model(frame, conf=CONF, verbose=False)
            dets = []
            for result in results:
                for idx,box in enumerate(result.boxes,1):
                    dets.append(draw(frame,box,idx,model,show))
            
            fc += 1
            if fc >= 10:
                fps = fc/(time.time()-st)
                fc,st = 0,time.time()
            
            info(frame,fps,device,dets)
            cv2.imshow('Waste Classification',frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('s'):
                save += 1
                f = f"capture_{save}.jpg"
                cv2.imwrite(f,frame)
                print(f"💾 {f}")
            elif key == ord('c'):
                show = not show
                print(f"🔄 Confidence:{'ON' if show else 'OFF'}")
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Done!")

if __name__ == '__main__':
    main()
