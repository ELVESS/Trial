#!/usr/bin/env python3
"""
FastAPI server for SAM browser app.
Serves static files, handles model loading, and computes image embeddings.
"""
import io
import json
import tempfile
import webbrowser
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from utils.sam_loader import SAMLoader
from utils.embed import compute_embedding
from utils.io_helpers import generate_image_id, save_embedding, load_embedding, resize_image_if_needed
from utils.geometry import extract_contours, dilate_mask


# Initialize FastAPI app
app = FastAPI(title="SAM Browser ONNX App")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
ONNX_DIR = MODELS_DIR / "onnx"
STATIC_DIR = BASE_DIR / "static"
TEMP_DIR = Path(tempfile.gettempdir()) / "sam_browser_onnx"
TEMP_DIR.mkdir(exist_ok=True)

# Global state
sam_loader = SAMLoader(CHECKPOINTS_DIR)
image_cache: Dict[str, Dict[str, Any]] = {}


# Request/Response models
class LoadModelRequest(BaseModel):
    model_type: str


class LoadModelResponse(BaseModel):
    model_type: str
    status: str


class UploadImageResponse(BaseModel):
    image_id: str
    width: int
    height: int
    embedding_url: str
    model_type: str


class ExportRequest(BaseModel):
    clicks: list
    dilation: int
    contours: list


# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root():
    """Serve the main HTML page."""
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/load_model", response_model=LoadModelResponse)
async def load_model(request: LoadModelRequest):
    """
    Load a SAM model.

    Args:
        request: LoadModelRequest with model_type

    Returns:
        LoadModelResponse with status
    """
    model_type = request.model_type

    if model_type not in ["vit_b", "vit_l", "vit_h"]:
        raise HTTPException(status_code=400, detail=f"Invalid model_type: {model_type}")

    # Check if ONNX model exists
    onnx_path = ONNX_DIR / f"sam_decoder_{model_type}.onnx"
    if not onnx_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"ONNX model not found: {onnx_path}. Please run: python models/export_onnx.py"
        )

    try:
        # Load model (cached)
        sam_loader.load_model(model_type)
        return LoadModelResponse(model_type=model_type, status="loaded")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.post("/api/upload_image", response_model=UploadImageResponse)
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image and compute its embedding.

    Args:
        file: Image file (multipart upload)

    Returns:
        UploadImageResponse with image metadata and embedding URL
    """
    # Check if model is loaded
    predictor = sam_loader.get_current_model()
    if predictor is None:
        raise HTTPException(
            status_code=400,
            detail="No model loaded. Please load a model first."
        )

    model_type = sam_loader.get_current_model_type()

    try:
        # Read and decode image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Optionally resize if too large
        image, was_resized = resize_image_if_needed(image, max_dimension=1024)

        # Get dimensions
        width, height = image.size

        # Compute embedding
        embedding = compute_embedding(predictor, image)

        # Generate image ID
        image_id = generate_image_id()

        # Save embedding to temp file
        embedding_path = TEMP_DIR / f"{image_id}.npy"
        save_embedding(embedding, embedding_path)

        # Save image for potential export
        image_path = TEMP_DIR / f"{image_id}.png"
        image.save(image_path)

        # Cache image metadata
        image_cache[image_id] = {
            "width": width,
            "height": height,
            "model_type": model_type,
            "embedding_path": embedding_path,
            "image_path": image_path,
        }

        return UploadImageResponse(
            image_id=image_id,
            width=width,
            height=height,
            embedding_url=f"/api/embedding/{image_id}",
            model_type=model_type,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")


@app.get("/api/embedding/{image_id}")
async def get_embedding(image_id: str):
    """
    Get image embedding as .npy bytes.

    Args:
        image_id: Image ID

    Returns:
        Raw .npy file bytes
    """
    if image_id not in image_cache:
        raise HTTPException(status_code=404, detail="Image not found")

    embedding_path = image_cache[image_id]["embedding_path"]

    if not embedding_path.exists():
        raise HTTPException(status_code=404, detail="Embedding file not found")

    # Read .npy file
    with open(embedding_path, "rb") as f:
        npy_bytes = f.read()

    return Response(
        content=npy_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={image_id}.npy"}
    )


@app.post("/api/export/{image_id}")
async def export_rois(image_id: str, request: ExportRequest):
    """
    Export ROI data (clicks, contours, metadata) as JSON.

    Args:
        image_id: Image ID
        request: ExportRequest with clicks, dilation, contours

    Returns:
        JSON export data
    """
    if image_id not in image_cache:
        raise HTTPException(status_code=404, detail="Image not found")

    metadata = image_cache[image_id]

    export_data = {
        "image_id": image_id,
        "image_size": {
            "width": metadata["width"],
            "height": metadata["height"]
        },
        "model_type": metadata["model_type"],
        "clicks": request.clicks,
        "dilation": request.dilation,
        "contours": request.contours,
    }

    return JSONResponse(content=export_data)


@app.get("/api/models")
async def list_models():
    """
    List available ONNX models.

    Returns:
        List of available model types
    """
    available_models = []

    for model_type in ["vit_b", "vit_l", "vit_h"]:
        onnx_path = ONNX_DIR / f"sam_decoder_{model_type}.onnx"
        if onnx_path.exists():
            available_models.append(model_type)

    return {"models": available_models}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "current_model": sam_loader.get_current_model_type(),
        "cached_images": len(image_cache)
    }


def open_browser():
    """Open browser after server starts."""
    import time
    time.sleep(1)  # Wait for server to start
    webbrowser.open("http://127.0.0.1:8000")


if __name__ == "__main__":
    # Check if ONNX models exist
    onnx_models = list(ONNX_DIR.glob("sam_decoder_*.onnx"))
    if not onnx_models:
        print("=" * 60)
        print("⚠️  No ONNX models found!")
        print("=" * 60)
        print("\nPlease run the following commands first:")
        print("  1. python models/download_models.py")
        print("  2. python models/export_onnx.py")
        print("\nThen restart the server.")
        print("=" * 60)
    else:
        print("=" * 60)
        print("SAM Browser ONNX App")
        print("=" * 60)
        print(f"Available models: {len(onnx_models)}")
        print("\nStarting server at http://127.0.0.1:8000")
        print("=" * 60)

        # Start server with auto-reload
        import threading
        threading.Thread(target=open_browser, daemon=True).start()

        uvicorn.run(
            "app:app",
            host="127.0.0.1",
            port=8000,
            reload=False,
            log_level="info"
        )
