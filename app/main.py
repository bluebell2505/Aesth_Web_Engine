# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.responses import JSONResponse, StreamingResponse
from .analyzers import analyze_image_bytes, load_image_bytes, preprocess_image_to_pil, get_preprocessed_debug_image, extract_palette_kmeans
import io, os, json

app = FastAPI(title="BrandDNA Service - Week1/2 (Abhinav)")

@app.post("/api/extract-aesthetic")
async def extract_aesthetic(file: UploadFile = File(...)):
    try:
        content = await file.read()
        branddna = analyze_image_bytes(content)
        return JSONResponse(content={"status":"ok","brandDNA": branddna})
    except Exception as e:
        return JSONResponse(content={"status":"error", "message": str(e)}, status_code=500)

@app.post("/api/ocr-debug")
async def ocr_debug(file: UploadFile = File(...)):
    """
    Returns a PNG (preprocessed image) as binary plus the OCR debug info.
    Use this to visually inspect what OCR sees.
    """
    try:
        content = await file.read()
        pil = load_image_bytes(content)
        # preprocessed image bytes
        png_io = get_preprocessed_debug_image(pil, upscale=1.5)
        # run full analyze to get typography.debug info
        branddna = analyze_image_bytes(content)
        # stream PNG and include debug as headers (or return multipart-like JSON with base64)
        png_io.seek(0)
        headers = {
            "X-OCR-Debug": json.dumps(branddna.get("typography", {}).get("debug", {}))
        }
        return StreamingResponse(png_io, media_type="image/png", headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tokens")
async def get_tokens():
    """
    Returns TokenSet mapping for three vibes: Fun, Premium, Tech.
    These are recommended UI tokens (radius, shadow, type scale, spacing, motion).
    """
    tokens = {
        "fun": {
            "border_radius": "16px",
            "shadows": "soft",
            "typography": {"primary": "Rounded Sans", "scale": {"h1":"36px","body":"16px"}},
            "spacing_scale": "airy",
            "motion": "high",
            "notes": "Playful, rounded, colorful CTAs"
        },
        "premium": {
            "border_radius": "8px",
            "shadows": "elevated-subtle",
            "typography": {"primary":"Serif Pair", "scale":{"h1":"42px","body":"18px"}},
            "spacing_scale": "balanced",
            "motion": "subtle",
            "notes": "Elegant, spacious, refined color accents"
        },
        "tech": {
            "border_radius": "6px",
            "shadows": "minimal",
            "typography": {"primary":"Neutral Sans", "scale":{"h1":"34px","body":"15px"}},
            "spacing_scale": "compact",
            "motion": "moderate",
            "notes": "Sharp, efficient, neutral color palettes"
        }
    }
    return JSONResponse(content={"status":"ok","tokens": tokens})
