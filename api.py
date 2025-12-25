#!/usr/bin/env python3
"""
Klasifikasi Sampah Anorganik Menggunakan Algoritma YOLO - REST API
===================================================================
FastAPI REST API untuk klasifikasi sampah anorganik.

Usage:
    uvicorn api:app --reload --port 8000

Endpoints:
    POST /detect     - Detect waste in uploaded image
    GET  /health     - Health check
    GET  /classes    - List available classes

Requirements:
    pip install fastapi uvicorn python-multipart
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pathlib import Path
from PIL import Image
from typing import List, Optional
import io
import base64
from datetime import datetime

# Config
MODEL_PATH = './models/best_model.pt'
DEFAULT_CONF = 0.25

# Class info
CLASS_INFO = {
    'battery': {
        'emoji': '🔋',
        'category': 'B3 (Berbahaya)',
        'bin_color': 'Merah',
        'disposal': 'Tempat khusus limbah B3',
        'recyclable': False
    },
    'biological': {
        'emoji': '🥬',
        'category': 'Organik',
        'bin_color': 'Hijau',
        'disposal': 'Tong sampah organik, bisa dijadikan kompos',
        'recyclable': True
    },
    'cardboard': {
        'emoji': '📦',
        'category': 'Anorganik',
        'bin_color': 'Biru',
        'disposal': 'Bank sampah atau pengepul kardus',
        'recyclable': True
    },
    'clothes': {
        'emoji': '👕',
        'category': 'Tekstil',
        'bin_color': 'Biru',
        'disposal': 'Donasi atau bank sampah',
        'recyclable': True
    },
    'glass': {
        'emoji': '🍾',
        'category': 'Anorganik',
        'bin_color': 'Biru',
        'disposal': 'Bank sampah atau pengepul kaca',
        'recyclable': True
    },
    'metal': {
        'emoji': '🥫',
        'category': 'Anorganik',
        'bin_color': 'Biru',
        'disposal': 'Bank sampah atau pengepul logam',
        'recyclable': True
    },
    'paper': {
        'emoji': '📄',
        'category': 'Anorganik',
        'bin_color': 'Biru',
        'disposal': 'Bank sampah atau pengepul kertas',
        'recyclable': True
    },
    'plastic': {
        'emoji': '🥤',
        'category': 'Anorganik',
        'bin_color': 'Kuning',
        'disposal': 'Bank sampah, pilah berdasarkan jenis',
        'recyclable': True
    },
    'shoes': {
        'emoji': '👟',
        'category': 'Tekstil',
        'bin_color': 'Biru',
        'disposal': 'Donasi atau bank sampah',
        'recyclable': True
    },
    'trash': {
        'emoji': '🗑️',
        'category': 'Residu',
        'bin_color': 'Hitam',
        'disposal': 'Tong sampah umum',
        'recyclable': False
    },
}

# Initialize FastAPI
app = FastAPI(
    title="Klasifikasi Sampah Anorganik - YOLO API",
    description="REST API untuk Klasifikasi Sampah Anorganik Menggunakan Algoritma YOLO (YOLOv8)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None


def load_model():
    """Load YOLO model"""
    global model
    if model is None:
        from ultralytics import YOLO
        if Path(MODEL_PATH).exists():
            model = YOLO(MODEL_PATH)
            print(f"✓ Model loaded: {MODEL_PATH}")
        else:
            print(f"❌ Model not found: {MODEL_PATH}")
    return model


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "♻️ Klasifikasi Sampah API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "detect": "POST /detect - Detect waste in image",
            "health": "GET /health - Health check",
            "classes": "GET /classes - List available classes"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = model is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_path": MODEL_PATH,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/classes")
async def get_classes():
    """Get available classes with info"""
    return {
        "total_classes": len(CLASS_INFO),
        "classes": CLASS_INFO
    }


@app.post("/detect")
async def detect_waste(
    file: UploadFile = File(..., description="Image file to analyze"),
    confidence: float = Query(DEFAULT_CONF, ge=0.1, le=0.9, description="Confidence threshold"),
    return_image: bool = Query(False, description="Return annotated image as base64")
):
    """
    Detect and classify waste in uploaded image.
    
    - **file**: Image file (JPG, PNG)
    - **confidence**: Detection confidence threshold (0.1-0.9)
    - **return_image**: Include annotated image in response (base64)
    
    Returns detected objects with classification and disposal info.
    """
    
    # Check model
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run 'python train.py' first."
        )
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Expected image."
        )
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Run detection
        results = model(image, conf=confidence, verbose=False)
        
        # Process detections
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                cls = int(box.cls[0].cpu().numpy())
                conf_score = float(box.conf[0].cpu().numpy())
                class_name = result.names[cls]
                info = CLASS_INFO.get(class_name, {})
                
                detections.append({
                    "class": class_name,
                    "confidence": round(conf_score, 4),
                    "bbox": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2)
                    },
                    "info": {
                        "emoji": info.get('emoji', '❓'),
                        "category": info.get('category', 'Unknown'),
                        "bin_color": info.get('bin_color', '-'),
                        "disposal": info.get('disposal', '-'),
                        "recyclable": info.get('recyclable', False)
                    }
                })
        
        # Build response
        response = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "image_info": {
                "filename": file.filename,
                "size": f"{image.width}x{image.height}",
                "format": image.format or "Unknown"
            },
            "parameters": {
                "confidence_threshold": confidence
            },
            "summary": {
                "total_detections": len(detections),
                "recyclable": sum(1 for d in detections if d['info']['recyclable']),
                "non_recyclable": sum(1 for d in detections if not d['info']['recyclable'])
            },
            "detections": detections
        }
        
        # Add annotated image if requested
        if return_image and detections:
            annotated = results[0].plot()
            # Convert to PIL Image and then to base64
            annotated_pil = Image.fromarray(annotated)
            buffer = io.BytesIO()
            annotated_pil.save(buffer, format='JPEG', quality=85)
            buffer.seek(0)
            response["annotated_image"] = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return JSONResponse(content=response)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )


@app.post("/detect/image")
async def detect_waste_return_image(
    file: UploadFile = File(...),
    confidence: float = Query(DEFAULT_CONF, ge=0.1, le=0.9)
):
    """
    Detect waste and return annotated image directly.
    
    Returns JPEG image with bounding boxes drawn.
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        results = model(image, conf=confidence, verbose=False)
        annotated = results[0].plot()
        
        # Convert to JPEG
        annotated_pil = Image.fromarray(annotated)
        buffer = io.BytesIO()
        annotated_pil.save(buffer, format='JPEG', quality=90)
        buffer.seek(0)
        
        return StreamingResponse(
            buffer,
            media_type="image/jpeg",
            headers={"Content-Disposition": f"inline; filename=detected_{file.filename}"}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
