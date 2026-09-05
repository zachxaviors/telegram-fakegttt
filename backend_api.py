"""
Backend API Service for OCR + Inpainting Pipeline (v4 - Production Hardened)
Secure, validated, and optimized for production deployment.
"""

import asyncio
import base64
import glob
import io
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, Dict, Any, List

import cv2
import httpx
import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont, ImageFilter
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

FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
FONT_CACHE: List[str] = []

def load_font_cache():
    global FONT_CACHE
    if os.path.isdir(FONTS_DIR):
        FONT_CACHE = sorted(glob.glob(os.path.join(FONTS_DIR, "*.ttf")))
        logger.info(f"Loaded {len(FONT_CACHE)} fonts from {FONTS_DIR}")
    else:
        logger.warning(f"Fonts directory not found: {FONTS_DIR}")

load_font_cache()

CCCD_FONT_PRIORITY = [
    "Arial Narrow Bold",
    "HelveticaNeue-CondensedBold",
    "Arial Bold",
    "Helvetica-Bold",
    "HelveticaNeue-Bold",
]

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

def find_best_font(target_height: int) -> Optional[str]:
    for priority_name in CCCD_FONT_PRIORITY:
        for font_path in FONT_CACHE:
            if priority_name.lower().replace(" ", "").replace("-", "") in os.path.basename(font_path).lower().replace(" ", "").replace("-", ""):
                try:
                    test_font = ImageFont.truetype(font_path, target_height)
                    bbox = test_font.getbbox("A")
                    if bbox[3] - bbox[1] > 0:
                        logger.info(f"Matched font: {os.path.basename(font_path)}")
                        return font_path
                except Exception:
                    continue
    if FONT_CACHE:
        fallback = FONT_CACHE[0]
        logger.warning(f"No priority font matched, falling back to: {os.path.basename(fallback)}")
        return fallback
    return None

def analyze_text_style(img_np: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
    x0, y0, x1, y1 = bbox
    roi = img_np[y0:y1, x0:x1]
    if roi.size == 0:
        return {"height": max(1, y1 - y0), "color": (0, 0, 0), "weight": "bold"}
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    avg_stroke_width = 0
    if contours:
        widths = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                widths.append(2 * area / perimeter)
        if widths:
            avg_stroke_width = np.median(widths)
    
    text_pixels = roi[binary > 0]
    if len(text_pixels) > 0:
        avg_color = tuple(int(c) for c in np.median(text_pixels.reshape(-1, 3), axis=0))
    else:
        avg_color = (0, 0, 0)
    
    height = max(1, y1 - y0)
    weight = "bold" if avg_stroke_width > height * 0.08 else "regular"
    
    return {
        "height": height,
        "color": avg_color,
        "weight": weight,
        "stroke_width": avg_stroke_width,
    }

def render_text_on_canvas(
    cleaned_img_np: np.ndarray,
    coordinates: dict,
    new_text: str,
    img_dims: Tuple[int, int],
) -> np.ndarray:
    result = cleaned_img_np.copy()
    h, w = result.shape[:2]
    
    pil_img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    name_bbox = None
    for key in ["name_text", "name"]:
        if key in coordinates:
            name_bbox = normalize_coordinates(coordinates[key], w, h)
            break
    
    if name_bbox is None:
        logger.warning("No name_text coordinate found for text rendering")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    style = analyze_text_style(cleaned_img_np, name_bbox)
    target_height = int(style["height"] * 0.85)
    font_path = find_best_font(target_height)
    
    if font_path is None:
        logger.warning("No font available, skipping local text rendering")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    try:
        font_size = target_height
        font = ImageFont.truetype(font_path, font_size)
        test_bbox = font.getbbox(new_text)
        text_w = test_bbox[2] - test_bbox[0]
        region_w = name_bbox[2] - name_bbox[0]
        
        while text_w > region_w and font_size > 6:
            font_size -= 1
            font = ImageFont.truetype(font_path, font_size)
            test_bbox = font.getbbox(new_text)
            text_w = test_bbox[2] - test_bbox[0]
        
        text_x = name_bbox[0]
        text_y = name_bbox[1] + int((name_bbox[3] - name_bbox[1] - (test_bbox[3] - test_bbox[1])) / 2)
        
        text_color = style["color"]
        
        shadow_offset = max(1, int(style.get("stroke_width", 1) * 0.3))
        shadow_color = tuple(min(255, c + 40) for c in text_color)
        draw.text((text_x + shadow_offset, text_y + shadow_offset), new_text, font=font, fill=shadow_color)
        
        draw.text((text_x, text_y), new_text, font=font, fill=text_color)
        
        result_pil = pil_img.filter(ImageFilter.GaussianBlur(radius=0.3))
        result_np = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
        
        roi_y0, roi_y1 = max(0, text_y - 2), min(h, text_y + font_size + 4)
        roi_x0, roi_x1 = max(0, text_x - 2), min(w, text_x + text_w + 4)
        
        original_roi = cleaned_img_np[roi_y0:roi_y1, roi_x0:roi_x1].astype(np.float64)
        rendered_roi = result_np[roi_y0:roi_y1, roi_x0:roi_x1].astype(np.float64)
        
        if original_roi.size > 0 and rendered_roi.size > 0:
            orig_mean = np.mean(original_roi)
            rend_mean = np.mean(rendered_roi)
            if rend_mean > 0:
                scale_factor = orig_mean / rend_mean
                rendered_roi = np.clip(rendered_roi * scale_factor, 0, 255)
            
            noise_std = np.std(original_roi) * 0.15
            noise = np.random.normal(0, noise_std, rendered_roi.shape)
            rendered_roi = np.clip(rendered_roi + noise, 0, 255)
            
            result_np[roi_y0:roi_y1, roi_x0:roi_x1] = rendered_roi.astype(np.uint8)
        
        logger.info(f"Text rendered locally with {os.path.basename(font_path)} at size {font_size}")
        return result_np
        
    except Exception as e:
        logger.error(f"Local text rendering failed: {e}, falling back to API inpainting")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def extract_glyph_templates(img_np: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict[str, np.ndarray]:
    x0, y0, x1, y1 = bbox
    roi = img_np[y0:y1, x0:x1]
    if roi.size == 0:
        return {}
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    glyphs = {}
    char_positions = []
    for cnt in contours:
        gx, gy, gw, gh = cv2.boundingRect(cnt)
        if gw < 3 or gh < 3 or gw > (x1 - x0) * 0.5:
            continue
        char_positions.append((gx, gy, gw, gh))
    
    char_positions.sort(key=lambda p: p[0])
    
    for idx, (gx, gy, gw, gh) in enumerate(char_positions):
        pad = 2
        cy0 = max(0, gy - pad)
        cy1 = min(roi.shape[0], gy + gh + pad)
        cx0 = max(0, gx - pad)
        cx1 = min(roi.shape[1], gx + gw + pad)
        glyph_roi = roi[cy0:cy1, cx0:cx1].copy()
        glyphs[f"glyph_{idx}"] = glyph_roi
    
    logger.info(f"Extracted {len(glyphs)} glyph templates from bbox")
    return glyphs

def composite_from_glyphs(
    cleaned_img_np: np.ndarray,
    coordinates: dict,
    new_text: str,
    img_dims: Tuple[int, int],
) -> Optional[np.ndarray]:
    name_bbox = None
    for key in ["name_text", "name"]:
        if key in coordinates:
            name_bbox = normalize_coordinates(coordinates[key], img_dims[0], img_dims[1])
            break
    
    if name_bbox is None:
        return None
    
    glyphs = extract_glyph_templates(cleaned_img_np, name_bbox)
    if not glyphs:
        logger.warning("No glyphs extracted, cannot composite")
        return None
    
    sample_glyph = next(iter(glyphs.values()))
    glyph_h, glyph_w = sample_glyph.shape[:2]
    
    result = cleaned_img_np.copy()
    x_cursor = name_bbox[0]
    y_center = name_bbox[1] + (name_bbox[3] - name_bbox[1] - glyph_h) // 2
    
    spacing = int(glyph_w * 0.15)
    
    for char in new_text:
        if char == " ":
            x_cursor += glyph_w // 2 + spacing
            continue
        
        best_glyph = None
        best_score = float("inf")
        for gname, gimg in glyphs.items():
            gh, gw = gimg.shape[:2]
            if abs(gh - glyph_h) > glyph_h * 0.3:
                continue
            score = abs(gw - glyph_w)
            if score < best_score:
                best_score = score
                best_glyph = gimg
        
        if best_glyph is not None:
            gh, gw = best_glyph.shape[:2]
            paste_y = max(0, min(y_center, result.shape[0] - gh))
            paste_x = max(0, min(x_cursor, result.shape[1] - gw))
            
            if paste_y + gh <= result.shape[0] and paste_x + gw <= result.shape[1]:
                gray_glyph = cv2.cvtColor(best_glyph, cv2.COLOR_BGR2GRAY) if len(best_glyph.shape) == 3 else best_glyph
                _, mask = cv2.threshold(gray_glyph, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                roi = result[paste_y:paste_y+gh, paste_x:paste_x+gw]
                if roi.shape == best_glyph.shape:
                    mask_3ch = cv2.merge([mask, mask, mask])
                    blended = cv2.bitwise_and(best_glyph, mask_3ch)
                    inv_mask = cv2.bitwise_not(mask_3ch)
                    bg_part = cv2.bitwise_and(roi, inv_mask)
                    result[paste_y:paste_y+gh, paste_x:paste_x+gw] = cv2.add(blended, bg_part)
            
            x_cursor += gw + spacing
        else:
            x_cursor += glyph_w + spacing
    
    logger.info(f"Glyph compositing completed for {len(new_text)} characters")
    return result

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

        cleaned_np = cv2.imdecode(np.frombuffer(cleaned_image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        local_result = await loop.run_in_executor(
            THREAD_POOL,
            render_text_on_canvas,
            cleaned_np,
            req.coordinates,
            req.prompt,
            (w, h),
        )
        
        if local_result is not None and local_result.size > 0:
            _, buf = cv2.imencode('.jpg', local_result, [cv2.IMWRITE_JPEG_QUALITY, 95])
            result_buffer = io.BytesIO(buf.tobytes())
            result_buffer.seek(0)
            
            total_time = time.time() - start_time
            logger.info(f"Local rendering completed in {total_time:.2f}s")
            
            return StreamingResponse(
                result_buffer,
                media_type="image/jpeg",
                headers={
                    "X-Processing-Time-Ms": str(round(total_time * 1000)),
                    "X-Original-Dimensions": f"{w}x{h}",
                    "X-Render-Method": "local-font",
                }
            )
        
        logger.info("Local rendering unavailable, falling back to API inpainting")
        mask_buffer = create_mask_with_perspective(cleaned_image_bytes, req.coordinates, (w, h))

        result_image_buffer = await call_openai_edit_api_secure(
            image_bytes=cleaned_image_bytes,
            mask_buffer=mask_buffer,
            prompt=req.prompt
        )

        total_time = time.time() - start_time
        logger.info(f"API inpainting completed in {total_time:.2f}s")
        
        result_image_buffer.seek(0)
        return StreamingResponse(
            result_image_buffer,
            media_type="image/jpeg",
            headers={
                "X-Processing-Time-Ms": str(round(total_time * 1000)),
                "X-Original-Dimensions": f"{w}x{h}",
                "X-Render-Method": "api-inpaint",
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

@app.get("/", response_class=HTMLResponse)
async def serve_webapp():
    index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>WebApp not found</h1>", status_code=404)

@app.get("/keys.json")
async def serve_keys():
    keys_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keys.json")
    if os.path.exists(keys_path):
        return FileResponse(keys_path, media_type="application/json")
    return {"keys": []}

@app.on_event("shutdown")
async def shutdown_event():
    THREAD_POOL.shutdown(wait=True)
    logger.info("Thread pool shut down gracefully")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
