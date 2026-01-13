# 📖 Modul 06: REST API dengan FastAPI

## Daftar Isi

1. [Pengantar FastAPI](#1-pengantar-fastapi)
2. [Arsitektur REST API](#2-arsitektur-rest-api)
3. [Analisis api.py Line-by-Line](#3-analisis-apipy-line-by-line)
4. [Endpoint Documentation](#4-endpoint-documentation)
5. [Request & Response Handling](#5-request--response-handling)
6. [Testing API](#6-testing-api)
7. [Authentication & Security](#7-authentication--security)
8. [Performance Optimization](#8-performance-optimization)
9. [Deployment](#9-deployment)
10. [Latihan](#10-latihan)

---

## 1. Pengantar FastAPI

### 1.1 Apa Itu FastAPI?

FastAPI adalah modern web framework untuk membangun APIs dengan Python 3.7+ berdasarkan standard Python type hints.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       FASTAPI OVERVIEW                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TRADITIONAL API FRAMEWORKS                                                 │
│  ──────────────────────────                                                │
│                                                                             │
│  Flask:                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐          │
│  │  @app.route('/detect', methods=['POST'])                    │          │
│  │  def detect():                                              │          │
│  │      data = request.get_json()  # Manual parsing           │          │
│  │      # No auto validation                                  │          │
│  │      # No auto documentation                               │          │
│  │      return jsonify(result)                                │          │
│  └─────────────────────────────────────────────────────────────┘          │
│                                                                             │
│  Django REST Framework:                                                    │
│  ┌─────────────────────────────────────────────────────────────┐          │
│  │  class DetectView(APIView):                                 │          │
│  │      def post(self, request):                              │          │
│  │          serializer = DetectSerializer(data=request.data)  │          │
│  │          # Banyak boilerplate                              │          │
│  │          return Response(result)                           │          │
│  └─────────────────────────────────────────────────────────────┘          │
│                                                                             │
│  ════════════════════════════════════════════════════════════════════════  │
│                                                                             │
│  FASTAPI WAY ★                                                             │
│  ─────────────                                                             │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────┐          │
│  │  from fastapi import FastAPI                                │          │
│  │  from pydantic import BaseModel                             │          │
│  │                                                             │          │
│  │  class Detection(BaseModel):                                │          │
│  │      class_name: str                                        │          │
│  │      confidence: float                                      │          │
│  │                                                             │          │
│  │  @app.post("/detect")                                       │          │
│  │  async def detect(file: UploadFile) -> list[Detection]:     │          │
│  │      # Auto validation ✓                                   │          │
│  │      # Auto documentation ✓                                │          │
│  │      # Async support ✓                                     │          │
│  │      return detections                                      │          │
│  └─────────────────────────────────────────────────────────────┘          │
│                                                                             │
│  FastAPI Features:                                                         │
│  ✓ Auto validation dengan Pydantic                                        │
│  ✓ Auto documentation (Swagger UI & ReDoc)                               │
│  ✓ Async/await support                                                    │
│  ✓ Type hints = less bugs                                                 │
│  ✓ High performance (on par with Node.js/Go)                             │
│  ✓ Easy to learn                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 REST API Concepts

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       REST API FUNDAMENTALS                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  REST = REpresentational State Transfer                                    │
│                                                                             │
│  HTTP METHODS:                                                             │
│  ─────────────                                                             │
│                                                                             │
│  ┌────────────┬─────────────────────────────────────────────────────────┐ │
│  │   Method   │   Usage                                                 │ │
│  ├────────────┼─────────────────────────────────────────────────────────┤ │
│  │   GET      │   Retrieve data (read-only)                            │ │
│  │   POST     │   Create new resource / send data                      │ │
│  │   PUT      │   Update entire resource                               │ │
│  │   PATCH    │   Update partial resource                              │ │
│  │   DELETE   │   Delete resource                                      │ │
│  └────────────┴─────────────────────────────────────────────────────────┘ │
│                                                                             │
│  DALAM PROJECT INI:                                                        │
│  ─────────────────                                                         │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  GET  /health     → Check API status                                │ │
│  │  GET  /classes    → Get list of 10 waste classes                    │ │
│  │  POST /detect     → Detect waste in uploaded image                  │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  REQUEST FLOW:                                                             │
│  ─────────────                                                             │
│                                                                             │
│  Client                          Server (api.py)                          │
│    │                                  │                                    │
│    │  POST /detect                    │                                    │
│    │  Content-Type: multipart/form    │                                    │
│    │  [image file]                    │                                    │
│    │ ────────────────────────────────>│                                    │
│    │                                  │                                    │
│    │                                  │  1. Validate request               │
│    │                                  │  2. Load image                     │
│    │                                  │  3. Run YOLO detection             │
│    │                                  │  4. Format response                │
│    │                                  │                                    │
│    │  200 OK                          │                                    │
│    │  Content-Type: application/json  │                                    │
│    │  {"detections": [...]}           │                                    │
│    │ <────────────────────────────────│                                    │
│    │                                  │                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Kenapa REST API untuk ML?

| Use Case             | Deskripsi                                                  |
| -------------------- | ---------------------------------------------------------- |
| **Mobile App**       | Android/iOS app bisa request deteksi ke server             |
| **Web Frontend**     | React/Vue app bisa integrate dengan ML model               |
| **IoT Devices**      | Smart bin dengan kamera kirim gambar untuk klasifikasi     |
| **Batch Processing** | Script automation untuk proses banyak gambar               |
| **Microservices**    | Integrate dengan sistem lain (database, notification, dll) |

---

## 2. Arsitektur REST API

### 2.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       API SYSTEM ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CLIENTS                                                                   │
│  ───────                                                                   │
│                                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │  Mobile  │  │   Web    │  │  IoT     │  │  Script  │  │  Other   │    │
│  │   App    │  │   App    │  │ Devices  │  │   CLI    │  │  APIs    │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
│       │             │             │             │             │            │
│       └─────────────┴─────────────┴──────┬──────┴─────────────┘            │
│                                          │                                  │
│                                    HTTP Request                             │
│                                          │                                  │
│                                          ▼                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         FASTAPI SERVER (api.py)                       │  │
│  │                         Port: 8000                                    │  │
│  ├──────────────────────────────────────────────────────────────────────┤  │
│  │                                                                       │  │
│  │  ENDPOINTS                                                            │  │
│  │  ─────────                                                           │  │
│  │                                                                       │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐       │  │
│  │  │  GET /health    │  │  GET /classes   │  │  POST /detect   │       │  │
│  │  │                 │  │                 │  │                 │       │  │
│  │  │  Health check   │  │  List classes   │  │  Run detection  │       │  │
│  │  │  Status API     │  │  Return 10      │  │  Return results │       │  │
│  │  │                 │  │  waste types    │  │                 │       │  │
│  │  └─────────────────┘  └─────────────────┘  └────────┬────────┘       │  │
│  │                                                      │                │  │
│  └──────────────────────────────────────────────────────│────────────────┘  │
│                                                         │                   │
│                                                         │                   │
│  ┌──────────────────────────────────────────────────────│────────────────┐  │
│  │                          YOLO MODEL                  │                │  │
│  │                                                      ▼                │  │
│  │  ┌────────────────────────────────────────────────────────────────┐  │  │
│  │  │              models/best_model.pt                              │  │  │
│  │  │                                                                │  │  │
│  │  │  Input:  PIL Image / numpy array                              │  │  │
│  │  │  Output: Detections (class, confidence, bbox)                 │  │  │
│  │  │                                                                │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 File Structure

```
api.py
│
├── Imports (Line 1-15)
│   ├── FastAPI core
│   ├── Pydantic models
│   └── Ultralytics YOLO
│
├── Pydantic Models (Line 16-50)
│   ├── BoundingBox
│   ├── Detection
│   ├── DetectionResponse
│   └── ClassInfo
│
├── Global Variables (Line 51-70)
│   ├── Model path
│   ├── Model instance
│   └── Class info dictionary
│
├── Startup Event (Line 71-85)
│   └── Load model on startup
│
└── Endpoints (Line 86-end)
    ├── GET /health
    ├── GET /classes
    └── POST /detect
```

---

## 3. Analisis api.py Line-by-Line

### 3.1 Imports dan FastAPI Setup

```python
#!/usr/bin/env python3
"""
Klasifikasi Sampah Anorganik Menggunakan Algoritma YOLO
========================================================
REST API untuk klasifikasi sampah menggunakan YOLOv8.

Usage:
    uvicorn api:app --reload

    atau

    python api.py

Endpoints:
    GET  /health    - Health check
    GET  /classes   - List semua class
    POST /detect    - Deteksi sampah dari image
"""

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Pydantic untuk data validation
from pydantic import BaseModel, Field
from typing import List, Optional

# Image processing
from PIL import Image
import io

# Pathlib untuk path handling
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="Klasifikasi Sampah API",       # Judul di dokumentasi
    description="REST API untuk klasifikasi 10 jenis sampah menggunakan YOLOv8",
    version="1.0.0",
    docs_url="/docs",      # Swagger UI URL
    redoc_url="/redoc"     # ReDoc URL
)
```

**Penjelasan Import:**

| Import             | Fungsi                          |
| ------------------ | ------------------------------- |
| `FastAPI`          | Main framework class            |
| `File, UploadFile` | Handle file upload              |
| `HTTPException`    | Raise HTTP errors               |
| `JSONResponse`     | JSON response builder           |
| `CORSMiddleware`   | Cross-Origin Resource Sharing   |
| `BaseModel`        | Pydantic model untuk validation |
| `Field`            | Field metadata dan validation   |

### 3.2 CORS Configuration

```python
# CORS middleware - penting untuk web clients!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # Allow semua origins (⚠️ kurang aman untuk prod)
    allow_credentials=True,      # Allow cookies
    allow_methods=["*"],         # Allow semua HTTP methods
    allow_headers=["*"],         # Allow semua headers
)
```

**Apa Itu CORS?**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       CORS (Cross-Origin Resource Sharing)                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  MASALAH:                                                                  │
│  ────────                                                                  │
│  Browser memblokir request dari domain berbeda untuk keamanan              │
│                                                                             │
│  ┌────────────────────┐         ┌────────────────────┐                    │
│  │   Frontend         │         │   API Server       │                    │
│  │   localhost:3000   │ ──X──> │   localhost:8000   │                    │
│  └────────────────────┘         └────────────────────┘                    │
│                                                                             │
│  Browser: "Request dari origin berbeda diblokir!"                          │
│                                                                             │
│  SOLUSI: CORS Middleware                                                   │
│  ───────────────────────                                                   │
│                                                                             │
│  Server mengirim header yang memberitahu browser:                          │
│  "Saya mengizinkan request dari origin ini"                                │
│                                                                             │
│  Response Headers:                                                         │
│  Access-Control-Allow-Origin: *                                            │
│  Access-Control-Allow-Methods: GET, POST, PUT, DELETE                     │
│  Access-Control-Allow-Headers: *                                           │
│                                                                             │
│  ⚠️ PRODUCTION NOTE:                                                       │
│  allow_origins=["*"] kurang aman!                                         │
│  Untuk production, spesifikasi origins yang diizinkan:                     │
│  allow_origins=["https://yourdomain.com"]                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Pydantic Models

```python
# ==================== PYDANTIC MODELS ====================

class BoundingBox(BaseModel):
    """
    Bounding box coordinates untuk satu detection.

    Format: [x1, y1, x2, y2] (top-left dan bottom-right corners)
    """
    x1: float = Field(..., description="Left x coordinate")
    y1: float = Field(..., description="Top y coordinate")
    x2: float = Field(..., description="Right x coordinate")
    y2: float = Field(..., description="Bottom y coordinate")


class Detection(BaseModel):
    """
    Single detection result.
    """
    class_name: str = Field(..., description="Detected class name", example="plastic")
    confidence: float = Field(
        ...,
        ge=0.0,        # Greater than or equal to 0
        le=1.0,        # Less than or equal to 1
        description="Detection confidence score",
        example=0.85
    )
    bounding_box: BoundingBox = Field(..., description="Object location")
    class_info: Optional[dict] = Field(None, description="Additional class metadata")


class DetectionResponse(BaseModel):
    """
    Response model untuk detection endpoint.
    """
    success: bool = Field(..., description="Whether detection was successful")
    message: str = Field(..., description="Status message")
    total_detections: int = Field(..., description="Number of objects detected")
    detections: List[Detection] = Field(default=[], description="List of detections")

    class Config:
        # Example untuk dokumentasi
        schema_extra = {
            "example": {
                "success": True,
                "message": "Detection completed successfully",
                "total_detections": 2,
                "detections": [
                    {
                        "class_name": "plastic",
                        "confidence": 0.92,
                        "bounding_box": {"x1": 100, "y1": 150, "x2": 300, "y2": 400},
                        "class_info": {"emoji": "🥤", "recyclable": True}
                    }
                ]
            }
        }


class HealthResponse(BaseModel):
    """Response model untuk health check."""
    status: str = Field(..., example="healthy")
    model_loaded: bool = Field(..., example=True)
    model_path: str = Field(..., example="models/best_model.pt")
```

**Pydantic Validation Flow:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PYDANTIC VALIDATION                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  REQUEST MASUK:                                                            │
│  ─────────────                                                             │
│                                                                             │
│  POST /detect                                                              │
│  {                                                                         │
│    "confidence": "invalid"  ← String instead of float                     │
│  }                                                                         │
│                                                                             │
│                        │                                                    │
│                        ▼                                                    │
│                                                                             │
│  PYDANTIC VALIDATION:                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  class Detection(BaseModel):                                        │  │
│  │      confidence: float = Field(..., ge=0.0, le=1.0)                │  │
│  │                                                                     │  │
│  │  1. Type check: "invalid" bukan float → ERROR                      │  │
│  │  2. Constraint check: ge=0.0, le=1.0                               │  │
│  │  3. Auto-generate error message                                    │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│                        │                                                    │
│                        ▼                                                    │
│                                                                             │
│  AUTO RESPONSE (422 Unprocessable Entity):                                │
│  {                                                                         │
│    "detail": [                                                            │
│      {                                                                    │
│        "loc": ["body", "confidence"],                                    │
│        "msg": "value is not a valid float",                              │
│        "type": "type_error.float"                                        │
│      }                                                                    │
│    ]                                                                      │
│  }                                                                         │
│                                                                             │
│  ✓ Tidak perlu validasi manual!                                           │
│  ✓ Error message otomatis dan konsisten                                   │
│  ✓ Type hints = dokumentasi                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.4 Global Variables dan Model Loading

```python
# ==================== GLOBAL VARIABLES ====================

MODEL_PATH = Path('./models/best_model.pt')
model = None  # Will be loaded on startup

# Class info (sama dengan web_app.py)
CLASS_INFO = {
    'battery': {
        'emoji': '🔋',
        'category': 'B3 (Berbahaya)',
        'bin_color': '🔴 Merah',
        'disposal': 'Tempat khusus limbah B3',
        'recyclable': False
    },
    # ... (10 kelas)
}


# ==================== STARTUP EVENT ====================

@app.on_event("startup")
async def load_model():
    """
    Load YOLO model saat server startup.

    Menggunakan event handler untuk:
    - Load model sekali saja
    - Validasi model exists
    - Global variable untuk reuse
    """
    global model

    if MODEL_PATH.exists():
        from ultralytics import YOLO
        model = YOLO(str(MODEL_PATH))
        print(f"✅ Model loaded: {MODEL_PATH}")
    else:
        print(f"⚠️ Warning: Model not found at {MODEL_PATH}")
        print("   Run 'python train.py' first to train the model")
```

**Startup Event Pattern:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       FASTAPI LIFECYCLE                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Server Start                                                              │
│       │                                                                     │
│       │  uvicorn api:app --reload                                          │
│       │                                                                     │
│       ▼                                                                     │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │  @app.on_event("startup")                                         │     │
│  │  async def load_model():                                          │     │
│  │      # Load heavy resources                                       │     │
│  │      # Initialize connections                                     │     │
│  │      # Warm up model                                             │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│       │                                                                     │
│       ▼                                                                     │
│  Server Ready                                                              │
│       │                                                                     │
│       │  Handling requests...                                              │
│       │                                                                     │
│       ▼                                                                     │
│  Server Shutdown (Ctrl+C)                                                  │
│       │                                                                     │
│       ▼                                                                     │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │  @app.on_event("shutdown")                                        │     │
│  │  async def cleanup():                                             │     │
│  │      # Close connections                                          │     │
│  │      # Save state                                                 │     │
│  │      # Cleanup resources                                          │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.5 GET /health Endpoint

```python
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Status"],
    summary="Health check endpoint",
    description="Check if API is running and model is loaded"
)
async def health_check():
    """
    Health check endpoint.

    Returns:
        HealthResponse dengan status dan info model

    Penggunaan:
        - Load balancer health probe
        - Monitoring systems
        - Pre-flight check sebelum kirim request
    """
    return HealthResponse(
        status="healthy" if model else "degraded",
        model_loaded=model is not None,
        model_path=str(MODEL_PATH)
    )
```

### 3.6 GET /classes Endpoint

```python
@app.get(
    "/classes",
    tags=["Info"],
    summary="Get list of detection classes",
    description="Returns all 10 waste classes with their metadata"
)
async def get_classes():
    """
    Get semua class yang bisa dideteksi.

    Returns:
        List of class names dengan info (emoji, category, etc.)
    """
    return {
        "total_classes": len(CLASS_INFO),
        "classes": CLASS_INFO
    }
```

### 3.7 POST /detect Endpoint

```python
@app.post(
    "/detect",
    response_model=DetectionResponse,
    tags=["Detection"],
    summary="Detect waste in image",
    description="Upload an image and get waste classification results"
)
async def detect(
    file: UploadFile = File(..., description="Image file (jpg, jpeg, png)"),
    confidence: float = 0.25
):
    """
    Deteksi sampah dalam gambar.

    Args:
        file: UploadFile - gambar yang akan dideteksi
        confidence: float - minimum confidence threshold (default 0.25)

    Returns:
        DetectionResponse dengan list deteksi

    Raises:
        HTTPException 400: Invalid file type
        HTTPException 500: Model not loaded
        HTTPException 500: Detection error
    """

    # Validate model is loaded
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Run 'python train.py' first."
        )

    # Validate file type
    allowed_types = ['image/jpeg', 'image/png', 'image/jpg']
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Allowed: {allowed_types}"
        )

    try:
        # Read file content
        contents = await file.read()

        # Convert to PIL Image
        image = Image.open(io.BytesIO(contents))

        # Convert to RGB if necessary (handle PNG with alpha channel)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Run detection
        results = model(image, conf=confidence, verbose=False)

        # Parse results
        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = result.names[cls_id]
                conf_score = float(box.conf[0].cpu().numpy())
                xyxy = box.xyxy[0].cpu().numpy().tolist()

                detections.append(Detection(
                    class_name=cls_name,
                    confidence=conf_score,
                    bounding_box=BoundingBox(
                        x1=xyxy[0], y1=xyxy[1],
                        x2=xyxy[2], y2=xyxy[3]
                    ),
                    class_info=CLASS_INFO.get(cls_name)
                ))

        return DetectionResponse(
            success=True,
            message=f"Detected {len(detections)} object(s)",
            total_detections=len(detections),
            detections=detections
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Detection error: {str(e)}"
        )
```

### 3.8 Main Entry Point

```python
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",           # Module:app format
        host="0.0.0.0",      # Listen on all interfaces
        port=8000,           # Port number
        reload=True          # Auto-reload on code changes
    )
```

---

## 4. Endpoint Documentation

### 4.1 Swagger UI (Auto-generated!)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       SWAGGER UI (http://localhost:8000/docs)               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │   Klasifikasi Sampah API                                    v1.0.0 │   │
│  │   REST API untuk klasifikasi 10 jenis sampah menggunakan YOLOv8   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ▼ Status                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  GET   /health    Health check endpoint                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ▼ Info                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  GET   /classes   Get list of detection classes                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ▼ Detection                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  POST  /detect    Detect waste in image                            │   │
│  │                                                                     │   │
│  │  [Try it out]                                                      │   │
│  │                                                                     │   │
│  │  Parameters:                                                        │   │
│  │  ┌───────────────────────────────────────────────────────────────┐ │   │
│  │  │ file*   [Select File]  Image file (jpg, jpeg, png)           │ │   │
│  │  │ confidence  0.25       Minimum confidence threshold           │ │   │
│  │  └───────────────────────────────────────────────────────────────┘ │   │
│  │                                                                     │   │
│  │  [Execute]                                                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ▼ Schemas                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  BoundingBox  │  Detection  │  DetectionResponse  │  HealthResponse│   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Endpoint Summary

| Method | Endpoint   | Description   | Parameters           | Response            |
| ------ | ---------- | ------------- | -------------------- | ------------------- |
| GET    | `/health`  | Health check  | None                 | `HealthResponse`    |
| GET    | `/classes` | List classes  | None                 | Class dictionary    |
| POST   | `/detect`  | Run detection | `file`, `confidence` | `DetectionResponse` |

---

## 5. Request & Response Handling

### 5.1 Request Examples

```bash
# Health check
curl http://localhost:8000/health

# Response:
# {
#   "status": "healthy",
#   "model_loaded": true,
#   "model_path": "models/best_model.pt"
# }

# Get classes
curl http://localhost:8000/classes

# Response:
# {
#   "total_classes": 10,
#   "classes": {
#     "battery": {...},
#     "biological": {...},
#     ...
#   }
# }

# Detect (with file upload)
curl -X POST http://localhost:8000/detect \
  -F "file=@test_image.jpg" \
  -F "confidence=0.3"

# Response:
# {
#   "success": true,
#   "message": "Detected 2 object(s)",
#   "total_detections": 2,
#   "detections": [
#     {
#       "class_name": "plastic",
#       "confidence": 0.89,
#       "bounding_box": {"x1": 100, "y1": 150, "x2": 300, "y2": 400},
#       "class_info": {...}
#     }
#   ]
# }
```

### 5.2 Python Client Example

```python
import requests

# Base URL
BASE_URL = "http://localhost:8000"

# Health check
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# Detect waste
with open("test_image.jpg", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/detect",
        files={"file": f},
        data={"confidence": 0.25}
    )

result = response.json()
print(f"Detected: {result['total_detections']} objects")

for det in result['detections']:
    print(f"- {det['class_name']}: {det['confidence']:.2%}")
```

### 5.3 JavaScript Client Example

```javascript
// Health check
fetch("http://localhost:8000/health")
  .then((res) => res.json())
  .then((data) => console.log(data));

// Detect waste
const formData = new FormData();
formData.append("file", imageFile); // File from input element
formData.append("confidence", 0.25);

fetch("http://localhost:8000/detect", {
  method: "POST",
  body: formData,
})
  .then((res) => res.json())
  .then((result) => {
    console.log(`Detected: ${result.total_detections} objects`);
    result.detections.forEach((det) => {
      console.log(`- ${det.class_name}: ${(det.confidence * 100).toFixed(1)}%`);
    });
  });
```

---

## 6. Testing API

### 6.1 Automated Testing dengan pytest

```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from api import app
import io
from PIL import Image

client = TestClient(app)

def test_health_check():
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data

def test_get_classes():
    """Test classes endpoint."""
    response = client.get("/classes")
    assert response.status_code == 200
    data = response.json()
    assert data["total_classes"] == 10
    assert "battery" in data["classes"]

def test_detect_invalid_file():
    """Test detect with invalid file type."""
    response = client.post(
        "/detect",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400

def test_detect_valid_image():
    """Test detect with valid image."""
    # Create dummy image
    img = Image.new('RGB', (640, 640), color='white')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)

    response = client.post(
        "/detect",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert "detections" in data
```

### 6.2 Running Tests

```bash
# Install pytest
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/test_api.py -v

# Run with coverage
pytest tests/test_api.py --cov=api --cov-report=html
```

---

## 7. Authentication & Security

### 7.1 Basic API Key Authentication

```python
from fastapi import Header, Depends

API_KEY = "your-secret-api-key"

async def verify_api_key(x_api_key: str = Header(...)):
    """Validate API key from header."""
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return x_api_key

@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)  # Require API key
):
    # ... detection logic
    pass
```

### 7.2 Request dengan API Key

```bash
# With API key header
curl -X POST http://localhost:8000/detect \
  -H "X-API-Key: your-secret-api-key" \
  -F "file=@image.jpg"
```

### 7.3 Rate Limiting

```python
from fastapi import Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/detect")
@limiter.limit("10/minute")  # Max 10 requests per minute
async def detect(request: Request, file: UploadFile = File(...)):
    # ... detection logic
    pass
```

---

## 8. Performance Optimization

### 8.1 Async Processing

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

async def run_detection_async(image, confidence):
    """Run detection in thread pool to not block event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        lambda: model(image, conf=confidence, verbose=False)
    )

@app.post("/detect")
async def detect(file: UploadFile = File(...), confidence: float = 0.25):
    # ... validation ...

    # Run detection asynchronously
    results = await run_detection_async(image, confidence)

    # ... parse results ...
```

### 8.2 Response Caching

```python
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache

@app.on_event("startup")
async def startup():
    FastAPICache.init(InMemoryBackend())

@app.get("/classes")
@cache(expire=3600)  # Cache for 1 hour
async def get_classes():
    return {"classes": CLASS_INFO}
```

### 8.3 Performance Tips

| Tip                  | Deskripsi                       | Implementation                   |
| -------------------- | ------------------------------- | -------------------------------- |
| **Model Warming**    | Run dummy inference on startup  | Prevents first-request latency   |
| **Batch Processing** | Process multiple images at once | Use `model.predict(images_list)` |
| **Image Resizing**   | Resize before inference         | Reduce memory usage              |
| **GPU Usage**        | Use CUDA if available           | Model auto-detects GPU           |

---

## 9. Deployment

### 9.1 Production Server dengan Gunicorn

```bash
# Install gunicorn
pip install gunicorn uvicorn[standard]

# Run with gunicorn (multiple workers)
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

### 9.2 Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY api.py .
COPY models/ models/

# Expose port
EXPOSE 8000

# Run with uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: "3.8"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_PATH=/app/models/best_model.pt
```

### 9.3 Cloud Deployment Options

| Platform                      | Pros                        | Cons                    |
| ----------------------------- | --------------------------- | ----------------------- |
| **AWS Lambda**                | Serverless, auto-scale      | Cold start, size limits |
| **Google Cloud Run**          | Container-based, auto-scale | Cold start              |
| **Azure Container Instances** | Easy setup                  | Cost                    |
| **DigitalOcean App Platform** | Simple deployment           | Limited regions         |
| **Railway**                   | Git-push deploy             | Newer platform          |

---

## 10. Latihan

### Latihan 1: New Endpoint

Tambahkan endpoint baru:

```python
@app.post("/detect/batch")
async def detect_batch(files: List[UploadFile]):
    """Process multiple images at once."""
    # Implementation here
    pass
```

### Latihan 2: Image URL Support

Modifikasi `/detect` untuk accept image URL:

```python
@app.post("/detect/url")
async def detect_url(image_url: str):
    """Detect from image URL instead of upload."""
    # Fetch image from URL
    # Run detection
    pass
```

### Latihan 3: WebSocket Real-time

Implementasi WebSocket endpoint untuk real-time detection:

```python
from fastapi import WebSocket

@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    await websocket.accept()
    # Receive and process frames
    pass
```

---

**Selamat! Anda telah menyelesaikan Modul 06: REST API dengan FastAPI**

_Lanjut ke: [Modul 07 - Realtime Detection](./07-realtime-detection.md)_
