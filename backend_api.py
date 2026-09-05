"""
Backend API Service for OCR + Inpainting Pipeline (v4 - Production Hardened)
Secure, validated, and optimized for production deployment.
"""

import asyncio
import base64
import io
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, Dict, Any

import cv2
import httpx
import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field, validator

load_dotenv()

# ==============================================================================
# SECURE CONFIGURATION
# ==============================================================================
CKEY_BASE_URL = os.getenv("CKEY_BASE_URL")
CKEY_API_KEY = os.getenv("CKEY_API_KEY")
CKEY_VISION_MODEL = os.getenv("CKEY_VISION_MODEL", "qwen2.5-vl-72b-instruct")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EDIT_URL = os.getenv("OPENAI_EDIT_URL")
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")

MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB limit
COORDINATE_SCALE = 1000  # Expected normalization scale from Vision API

if not CKEY_API_KEY or not OPENAI_API_KEY:
    raise RuntimeError("CRITICAL: CKEY_API_KEY and OPENAI_API_KEY must be set in environment variables")

THREAD_POOL = ThreadPoolExecutor(max_workers=4)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("InpaintingService")

app = FastAPI(title="ID Card Inpainting API v4")

ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["Authorization", "Content-Type"],
)


# ==============================================================================
# DATA MODELS WITH VALIDATION
# ==============================================================================
class OCRRequest(BaseModel):
    image_base64: str = Field(..., min_length=100)

    @validator("image_base64")
    def validate_image(cls, v):
        if len(v) > MAX_IMAGE_SIZE_BYTES * 1.5:
            raise ValueError(f"Image exceeds maximum size of {MAX_IMAGE_SIZE_BYTES} bytes")
        return v

class InpaintRequest(BaseModel):
    image_base64: str = Field(..., min_length=100)
    coordinates: Dict[str, Any]
    prompt: str = Field(..., min_length=1, max_length=500)

    @validator("image_base64")
    def validate_image(cls, v):
        if len(v) > MAX_IMAGE_SIZE_BYTES * 1.5:
            raise ValueError(f"Image exceeds maximum size of {MAX_IMAGE_SIZE_BYTES} bytes")
        return v

    @validator("coordinates")
    def validate_coordinates(cls, v):
        valid_regions = {k: val for k, val in v.items() if isinstance(val, list) and len(val) == 4}
        if not valid_regions:
            raise ValueError("No valid coordinate regions found. Expected format: {'field': [ymin, xmin, ymax, xmax]}")
        for name, bbox in valid_regions.items():
            if not all(isinstance(c, (int, float)) and 0 <= c <= COORDINATE_SCALE * 2 for c in bbox):
                raise ValueError(f"Invalid coordinate values for '{name}'. Must be numeric and within reasonable bounds.")
        return valid_regions


# ==============================================================================
# CORE LOGIC WITH SAFETY & SCALING
# ==============================================================================

def decode_base64_image_safe(b64_string: str) -> Tuple[bytes, Tuple[int, int]]:
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    
    try:
        image_bytes = base64.b64decode(b64_string, validate=True)
    except Exception as e:
        raise ValueError(f"Invalid base64 encoding: {e}")
    
    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise ValueError(f"Decoded image exceeds {MAX_IMAGE_SIZE_BYTES} byte limit")
    
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise ValueError(f"Invalid or corrupted image file: {e}")
        
    return image_bytes, image.size

def normalize_coordinates(bbox: list, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    ymin, xmin, ymax, xmax = bbox
    max_val = max(ymin, xmin, ymax, xmax)
    
    if max_val <= 1.0:
        scale = 1.0
    elif max_val <= COORDINATE_SCALE:
        scale = COORDINATE_SCALE
    else:
        scale = max(img_w, img_h)
    
    x0 = int((xmin / scale) * img_w)
    y0 = int((ymin / scale) * img_h)
    x1 = int((xmax / scale) * img_w)
    y1 = int((ymax / scale) * img_h)
    
    x0, y0 = max(0, min(x0, img_w)), max(0, min(y0, img_h))
    x1, y1 = max(0, min(x1, img_w)), max(0, min(y1, img_h))
    
    return x0, y0, x1, y1

async def call_vision_api_secure(image_base64: str) -> dict:
    headers = {
        "Authorization": f"Bearer {CKEY_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": CKEY_VISION_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    f"You are a Vietnamese CCCD (Citizen Identity Card) OCR specialist. "
                    f"Return ONLY raw JSON with normalized coordinates (scale {COORDINATE_SCALE}). "
                    f"Use visual anchors on the card to locate fields precisely under adverse conditions "
                    f"(glare, blur, low light, perspective tilt):\n"
                    f"- \"name_text\": Find the label \"Họ và tên\" or \"Full name\", then bound the actual name text below/beside it.\n"
                    f"- \"dob_text\": Find the label \"Ngày sinh\" or \"Date of birth\", then bound the date string (DD/MM/YYYY) beside it.\n"
                    f"Output format: {{\"name_text\": [ymin, xmin, ymax, xmax], \"dob_text\": [ymin, xmin, ymax, xmax]}}"
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": (
                        "Analyze this Vietnamese CCCD image. Use contextual visual anchors (label text, card layout, "
                        "security patterns) to locate both fields even if the photo has glare, motion blur, uneven lighting, "
                        "or camera tilt. Return precise bounding boxes for the name value and date of birth value."
                    )}
                ],
            }
        ],
        "max_tokens": 300,
        "temperature": 0.1,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        for attempt in range(3):
            try:
                response = await client.post(f"{CKEY_BASE_URL}/chat/completions", headers=headers, json=payload)
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                
                json_str = content.strip()
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0].strip()
                
                result = json.loads(json_str)
                if not isinstance(result, dict):
                    raise ValueError("Vision API returned non-dict JSON")
                
                expected_keys = {"name_text", "dob_text"}
                found_keys = set(result.keys()) & expected_keys
                if not found_keys:
                    alt_map = {"name": "name_text", "dob": "dob_text", "full_name": "name_text", "date_of_birth": "dob_text"}
                    for old_key, new_key in alt_map.items():
                        if old_key in result and new_key not in result:
                            result[new_key] = result[old_key]
                    found_keys = set(result.keys()) & expected_keys
                    if not found_keys:
                        raise ValueError(f"Vision API missing required keys. Got: {list(result.keys())}")
                
                return {k: result[k] for k in found_keys}
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error on attempt {attempt + 1}: {e}")
                if attempt == 2:
                    raise HTTPException(status_code=502, detail="Vision API returned invalid JSON")
            except ValueError:
                raise
            except Exception as e:
                if attempt == 2:
                    raise
                logger.warning(f"Vision API attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(1 * (attempt + 1))

def clean_text_opencv_safe(image_bytes: bytes, coordinates: dict, img_dims: Tuple[int, int]) -> bytes:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Failed to decode image with OpenCV")
        
    h, w = img.shape[:2]
    full_mask = np.zeros((h, w), dtype=np.uint8)
    
    for region_name, bbox in coordinates.items():
        x0, y0, x1, y1 = normalize_coordinates(bbox, w, h)
        if x1 > x0 and y1 > y0:
            pad_x = max(2, int((x1 - x0) * 0.05))
            pad_y = max(2, int((y1 - y0) * 0.05))
            x0p, y0p = max(0, x0 - pad_x), max(0, y0 - pad_y)
            x1p, y1p = min(w, x1 + pad_x), min(h, y1 + pad_y)
            cv2.rectangle(full_mask, (x0p, y0p), (x1p, y1p), 255, -1)
    
    cleaned_img = cv2.inpaint(img, full_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    
    gray = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2GRAY)
    text_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    residual = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, text_kernel)
    _, residual_mask = cv2.threshold(residual, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    combined_mask = cv2.bitwise_or(full_mask, cv2.bitwise_not(residual_mask))
    cleaned_img = cv2.inpaint(cleaned_img, combined_mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
    
    _, buffer = cv2.imencode('.jpg', cleaned_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buffer.tobytes()

def estimate_perspective_transform(bbox: list, img_w: int, img_h: int) -> Optional[np.ndarray]:
    x0, y0, x1, y1 = normalize_coordinates(bbox, img_w, img_h)
    w_region = x1 - x0
    h_region = y1 - y0
    if w_region < 10 or h_region < 5:
        return None
    
    skew_threshold = h_region * 0.15
    pts_src = np.array([
        [x0, y0],
        [x1, y0 + int(skew_threshold * 0.3)],
        [x1, y1 - int(skew_threshold * 0.2)],
        [x0, y1]
    ], dtype=np.float32)
    
    pts_dst = np.array([
        [x0, y0],
        [x1, y0],
        [x1, y1],
        [x0, y1]
    ], dtype=np.float32)
    
    vertical_diff = abs(pts_src[1][1] - pts_src[0][1]) + abs(pts_src[2][1] - pts_src[3][1])
    if vertical_diff < 3:
        return None
        
    return cv2.getPerspectiveTransform(pts_src, pts_dst)

def create_mask_with_perspective(image_bytes: bytes, coordinates: dict, img_dims: Tuple[int, int]) -> io.BytesIO:
    image = Image.open(io.BytesIO(image_bytes))
    width, height = image.size
    np_img = np.array(image)

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for region_name, bbox in coordinates.items():
        x0, y0, x1, y1 = normalize_coordinates(bbox, width, height)
        if x1 <= x0 or y1 <= y0:
            continue
            
        M = estimate_perspective_transform(bbox, width, height)
        if M is not None:
            region_pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32).reshape(-1, 1, 2)
            warped_pts = cv2.perspectiveTransform(region_pts, M)
            pts_int = warped_pts.astype(int).reshape(-1, 2)
            pts_clamped = np.clip(pts_int, [0, 0], [width - 1, height - 1])
            cv2.fillConvexPoly(np.array(mask), pts_clamped, 255)
        else:
            draw.rectangle([x0, y0, x1, y1], fill=255)

    mask_np = np.array(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_np = cv2.dilate(mask_np, kernel, iterations=1)
    mask_np = cv2.GaussianBlur(mask_np, (3, 3), sigmaX=1.0)
    _, mask_np = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

    mask_final = Image.fromarray(mask_np, mode="L")
    mask_buffer = io.BytesIO()
    mask_final.save(mask_buffer, format="PNG", optimize=True)
    mask_buffer.seek(0)
    return mask_buffer

def enhance_prompt_realism(original_prompt: str) -> str:
    realism_suffix = (
        " Render with photorealistic fidelity: match the ambient low-light shading of the surrounding card surface, "
        "apply subtle sub-pixel anti-aliasing on character edges, replicate the exact ink density and color temperature "
        "of adjacent printed text, include natural lens grain/noise consistent with smartphone camera capture at ISO 400-800, "
        "and preserve micro-shadows along the embossed card texture. No digital artifacts or oversharpening."
    )
    return f"{original_prompt}.{realism_suffix}"

async def call_openai_edit_api_secure(image_bytes: bytes, mask_buffer: io.BytesIO, prompt: str) -> io.BytesIO:
    enhanced_prompt = enhance_prompt_realism(prompt)
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    files = {
        "image": ("id_card.jpg", image_bytes, "image/jpeg"),
        "mask": ("mask.png", mask_buffer, "image/png"),
    }
    
    data = {
        "prompt": enhanced_prompt,
        "model": OPENAI_IMAGE_MODEL,
        "n": "1",
        "size": "1024x1024"
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(OPENAI_EDIT_URL, headers=headers, data=data, files=files)
            
            if response.status_code != 200:
                logger.error(f"OpenAI API Error: {response.status_code} - {response.text[:500]}")
                raise HTTPException(status_code=502, detail=f"Image editing service error: {response.status_code}")

            result = response.json()
            
            if "data" in result and len(result["data"]) > 0:
                entry = result["data"][0]
                
                if "b64_json" in entry and entry["b64_json"]:
                    img_bytes = base64.b64decode(entry["b64_json"])
                    buf = io.BytesIO(img_bytes)
                    buf.seek(0)
                    return buf
                
                url = entry.get("url")
                if url:
                    img_response = await client.get(url)
                    img_response.raise_for_status()
                    buf = io.BytesIO(img_response.content)
                    buf.seek(0)
                    return buf
                
                raise HTTPException(status_code=502, detail="No image data in API response")
            
            raise HTTPException(status_code=502, detail="Malformed response from image editing service")
            
        except httpx.TimeoutException:
            logger.error("OpenAI API timeout after 120s")
            raise HTTPException(status_code=504, detail="Image editing service timeout")
        except httpx.RequestError as e:
            logger.error(f"Network error calling OpenAI: {e}")
            raise HTTPException(status_code=502, detail=f"Network connectivity error: {str(e)}")


# ==============================================================================
# PRODUCTION ENDPOINTS
# ==============================================================================

@app.post("/inpaint")
async def inpaint_image(req: InpaintRequest):
    start_time = time.time()
    
    try:
        image_bytes, (w, h) = decode_base64_image_safe(req.image_base64)
        logger.info(f"Image validated: {w}x{h}")

        loop = asyncio.get_running_loop()
        cleaned_image_bytes = await loop.run_in_executor(
            THREAD_POOL, 
            clean_text_opencv_safe, 
            image_bytes, 
            req.coordinates,
            (w, h)
        )

        mask_buffer = create_mask_with_perspective(cleaned_image_bytes, req.coordinates, (w, h))

        result_image_buffer = await call_openai_edit_api_secure(
            image_bytes=cleaned_image_bytes,
            mask_buffer=mask_buffer,
            prompt=req.prompt
        )

        total_time = time.time() - start_time
        logger.info(f"Pipeline completed in {total_time:.2f}s")
        
        result_image_buffer.seek(0)
        return StreamingResponse(
            result_image_buffer,
            media_type="image/jpeg",
            headers={
                "X-Processing-Time-Ms": str(round(total_time * 1000)),
                "X-Original-Dimensions": f"{w}x{h}",
            }
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Inpainting Pipeline Error")
        raise HTTPException(status_code=500, detail="Internal processing error")

@app.post("/ocr")
async def ocr_endpoint(req: OCRRequest):
    try:
        _, dims = decode_base64_image_safe(req.image_base64)
        coords = await call_vision_api_secure(req.image_base64)
        return {"status": "success", "coordinates": coords, "image_dimensions": dims}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("OCR Error")
        raise HTTPException(status_code=500, detail="OCR processing failed")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": time.time(),
        "config_loaded": bool(CKEY_API_KEY and OPENAI_API_KEY)
    }

@app.on_event("shutdown")
async def shutdown_event():
    THREAD_POOL.shutdown(wait=True)
    logger.info("Thread pool shut down gracefully")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
