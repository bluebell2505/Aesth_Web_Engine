# app/analyzers.py
"""
BrandDNA analyzers module (final improved version)

Features:
- Palette extraction (kmeans in LAB)
- Entropy / edge density calculations
- Tone heuristics
- Multi-scale + multi-PSM OCR
- EAST text-region detection + per-crop multi-PSM OCR
- Smart merge of multi-scale and EAST results (prefers EAST when multi-scale yields tiny fragments)
- Debug helpers: preprocessed image output, per-word confidences
"""

import io
import os
import shutil
import numpy as np
from PIL import Image, ImageOps
from sklearn.cluster import KMeans
import cv2
import pytesseract

# ensure tesseract binary path for Windows
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------- Constants / Defaults ----------
EAST_MODEL_REL_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "models", "frozen_east_text_detection.pb"))

# ---------- Preprocessing ----------
def load_image_bytes(image_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return img

def preprocess_image_to_pil(img: Image.Image, max_size=1200):
    img_copy = img.copy()
    img_copy.thumbnail((max_size, max_size), Image.LANCZOS)
    arr = np.asarray(img_copy).astype(np.float32) / 255.0
    return img_copy, arr

def get_preprocessed_debug_image(img_pil: Image.Image, upscale=1.5):
    """
    Returns PNG bytes of CLAHE + thresholded image useful for visual debugging of OCR preprocessing.
    """
    cv_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    H, W = cv_img.shape[:2]
    if upscale and max(H, W) < 1200:
        cv_img = cv2.resize(cv_img, (int(W*upscale), int(H*upscale)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    is_success, buffer = cv2.imencode(".png", th)
    return io.BytesIO(buffer.tobytes())

# ---------- Palette extraction (KMeans in LAB) ----------
def extract_palette_kmeans(img: Image.Image, n_colors=5, sample_pixels=15000):
    arr = np.asarray(img).astype(np.float32) / 255.0
    h,w,_ = arr.shape
    pixels = arr.reshape(-1,3)
    if pixels.shape[0] > sample_pixels:
        idx = np.random.choice(pixels.shape[0], sample_pixels, replace=False)
        pixels_sample = pixels[idx]
    else:
        pixels_sample = pixels

    # convert to LAB using OpenCV
    pixels_lab = cv2.cvtColor((pixels_sample*255).astype('uint8').reshape(-1,1,3), cv2.COLOR_RGB2LAB)
    pixels_lab = pixels_lab.reshape(-1,3)

    kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(pixels_lab)
    centers_lab = kmeans.cluster_centers_.astype(int)

    centers_rgb = []
    for lab in centers_lab:
        lab_pixel = np.uint8([[lab]])
        rgb_pixel = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2RGB)[0,0]
        centers_rgb.append(tuple(int(x) for x in rgb_pixel))

    hex_colors = ['#%02x%02x%02x' % c for c in centers_rgb]
    return hex_colors

# ---------- Entropy / Density ----------
def image_entropy(img: Image.Image):
    gray = ImageOps.grayscale(img)
    arr = np.asarray(gray)
    hist, _ = np.histogram(arr.flatten(), bins=256, range=(0,255))
    hist = hist / hist.sum()
    hist_nonzero = hist[hist > 0]
    ent = -np.sum(hist_nonzero * np.log2(hist_nonzero))
    return float(ent)

def edge_count(img: Image.Image):
    arr = np.asarray(img.convert('L'))
    edges = cv2.Canny(arr, 100, 200)
    return int(np.sum(edges > 0))

# ---------- Tone classifier ----------
def tone_from_palette(hex_colors):
    import colorsys
    rgbs = [(int(h[1:3],16)/255.0, int(h[3:5],16)/255.0, int(h[5:7],16)/255.0)
            for h in hex_colors]
    hs, ss, vs = [], [], []
    for r,g,b in rgbs:
        h,s,v = colorsys.rgb_to_hsv(r,g,b)
        hs.append(h); ss.append(s); vs.append(v)
    avg_s = np.mean(ss)
    avg_v = np.mean(vs)
    avg_h = np.mean(hs)

    if avg_s > 0.7 and avg_v > 0.7:
        return "neon"
    if avg_v < 0.5 and avg_s < 0.5:
        return "muted"
    if (avg_h < 0.12) or (avg_h > 0.88) or (0.08 < avg_h < 0.17):
        return "warm"
    if 0.17 <= avg_h <= 0.6:
        return "cool"
    if avg_s < 0.35 and avg_v > 0.8:
        return "pastel"
    return "neutral"

# ---------- EAST text boxes ----------
def east_text_boxes(cv_img, east_path=None, min_conf=0.5):
    """
    Uses the EAST model to return a list of text box rectangles (startX,startY,endX,endY).
    If model not found, returns [].
    """
    if east_path is None:
        east_path = EAST_MODEL_REL_PATH
    if not os.path.exists(east_path):
        return []

    H, W = cv_img.shape[:2]
    newW, newH = (W//32)*32, (H//32)*32
    if newW == 0 or newH == 0:
        return []
    rW = W / float(newW)
    rH = H / float(newH)

    blob = cv2.dnn.blobFromImage(cv_img, 1.0, (newW, newH), (123.68, 116.78, 103.94), True, False)
    net = cv2.dnn.readNet(east_path)
    net.setInput(blob)
    (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"])

    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    for y in range(0, numRows):
        scoresData = scores[0,0,y]
        xData0 = geometry[0,0,y]
        xData1 = geometry[0,1,y]
        xData2 = geometry[0,2,y]
        xData3 = geometry[0,3,y]
        anglesData = geometry[0,4,y]
        for x in range(0, numCols):
            if scoresData[x] < min_conf:
                continue
            offsetX, offsetY = x * 4.0, y * 4.0
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            startX = int(startX * rW); startY = int(startY * rH)
            endX = int(endX * rW); endY = int(endY * rH)
            startX = max(0, startX); startY = max(0, startY)
            endX = min(W, endX); endY = min(H, endY)
            if endX - startX < 6 or endY - startY < 6:
                continue
            rects.append((startX, startY, endX, endY))
            confidences.append(float(scoresData[x]))

    if len(rects) == 0:
        return []

    # Non-maximum suppression (cv2.dnn.NMSBoxes expects specific formats)
    try:
        boxes = cv2.dnn.NMSBoxes(rects, confidences, min_conf, 0.4)
        out_boxes = []
        if len(boxes) > 0:
            for i in boxes.flatten():
                out_boxes.append(rects[i])
        return out_boxes
    except Exception:
        return rects

# ---------- Per-crop OCR helpers ----------
def ocr_crop_with_multi_psm(crop_gray, psm_list=(6,11), upscale=3.0):
    """
    OCR a single grayscale crop with multiple PSMs. Returns list of (word, conf) preserving approximate order.
    Uses highest per-word confidence across PSM variations.
    """
    results = {}  # key -> (word, conf, idx)
    for psm in psm_list:
        cfg = f"--oem 1 --psm {psm}"
        try:
            img_for_ocr = crop_gray
            if upscale != 1.0:
                img_for_ocr = cv2.resize(crop_gray, (int(crop_gray.shape[1]*upscale), int(crop_gray.shape[0]*upscale)), interpolation=cv2.INTER_CUBIC)
            img_for_ocr = cv2.fastNlMeansDenoising(img_for_ocr, h=10)
            _, th_crop = cv2.threshold(img_for_ocr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            data = pytesseract.image_to_data(th_crop, output_type=pytesseract.Output.DICT, config=cfg, lang='eng')
        except Exception:
            continue
        for i, txt in enumerate(data.get('text', [])):
            txt_s = (txt or "").strip()
            if not txt_s:
                continue
            try:
                conf = float(data.get('conf', [])[i])
            except Exception:
                conf = -1.0
            key = f"{i}-{txt_s}"
            prev = results.get(key)
            if (prev is None) or (conf > prev[1]):
                results[key] = (txt_s, conf, i)
    pairs = []
    for key, (w, c, idx) in results.items():
        pairs.append((idx, w, c))
    pairs.sort(key=lambda x: x[0])
    return [(w, c) for (_, w, c) in pairs]

def ocr_via_east_then_tesseract_improved(cv_img, east_path=None, min_conf=0.5, crop_upscale=4.0, psm_list=(6,11)):
    """
    Run EAST to get boxes, then OCR each box with multiple PSMs and return:
      { text, words, avg_conf, per_word: [{word,conf,box}], boxes }
    """
    if east_path is None:
        east_path = EAST_MODEL_REL_PATH
    boxes = east_text_boxes(cv_img, east_path=east_path, min_conf=min_conf)
    all_words = []
    all_confs = []
    per_word_conf_list = []
    for (sx, sy, ex, ey) in boxes:
        sx = max(0, sx); sy = max(0, sy); ex = min(cv_img.shape[1], ex); ey = min(cv_img.shape[0], ey)
        if ex - sx < 6 or ey - sy < 6:
            continue
        crop = cv_img[sy:ey, sx:ex]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        pairs = ocr_crop_with_multi_psm(gray, psm_list=psm_list, upscale=crop_upscale)
        for w, c in pairs:
            all_words.append(w)
            all_confs.append(c)
            per_word_conf_list.append({"word": w, "conf": float(c), "box": [int(sx), int(sy), int(ex), int(ey)]})
    avg_conf = (sum(all_confs)/len(all_confs))/100.0 if all_confs else 0.0
    return {"text": " ".join(all_words).strip(), "words": len(all_words), "avg_conf": round(avg_conf, 3), "per_word": per_word_conf_list, "boxes": boxes}

# ---------- Merge strategy ----------
def merge_multiscale_and_east(multires, eastres, conf_threshold=0.60):
    """
    Merge strategy:
      - keep all multires words IF multires.raw_conf >= conf_threshold
      - add east words with per-word conf >= conf_threshold that aren't duplicates (case-insensitive)
      - if empty, relax threshold once
    """
    final_words = []
    seen = set()

    if multires.get("text") and multires.get("raw_conf", 0.0) >= conf_threshold:
        for w in multires.get("text").split():
            final_words.append(w); seen.add(w.lower())

    for entry in eastres.get("per_word", []):
        w = entry.get("word"); c = entry.get("conf", 0)
        if (c / 100.0) >= conf_threshold and w.lower() not in seen:
            final_words.append(w); seen.add(w.lower())

    if len(final_words) == 0 and conf_threshold > 0.45:
        return merge_multiscale_and_east(multires, eastres, conf_threshold=0.45)
    return " ".join(final_words).strip(), len(final_words)

# ---------- Multi-scale OCR + EAST + Merge (main OCR routine) ----------
def typography_preproc_and_ocr(img_pil):
    """
    Runs multi-scale OCR and EAST-crop OCR, merges results heuristically.
    Returns:
      {
        primary, confidence, notes, text_sample, word_count, debug
      }
    """
    # guard: tesseract presence
    if shutil.which("tesseract") is None and not getattr(pytesseract.pytesseract, 'tesseract_cmd', None):
        return {"primary": "unknown", "confidence": 0.0, "notes": "tesseract binary not found", "text_sample": "", "word_count": 0, "debug": {}}

    # ---- MULTI-SCALE OCR (stronger defaults) ----
    def run_multiscale(img_pil_local):
        orig = cv2.cvtColor(np.array(img_pil_local), cv2.COLOR_RGB2BGR)
        H, W = orig.shape[:2]
        scales = [2.0, 3.0, 4.0]            # stronger upscaling for tiny/stylized text
        psm_modes = [11, 6, 3, 7]           # prefer sparse/line first, then block, then auto
        best = {"avg_conf": -1.0, "raw_conf": 0.0, "text": "", "words": 0, "psm": None, "scale": None}
        for scale in scales:
            img = orig.copy()
            if max(H, W) < 1600 or scale > 1.0:
                new_w = int(W * scale); new_h = int(H * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.fastNlMeansDenoising(gray, h=10)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
            th_closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

            for psm in psm_modes:
                config = f'--oem 1 --psm {psm}'
                try:
                    data = pytesseract.image_to_data(th_closed, output_type=pytesseract.Output.DICT, config=config, lang='eng')
                except Exception:
                    continue

                words = []
                confs = []
                for t, c in zip(data.get('text', []), data.get('conf', [])):
                    if t and t.strip():
                        try:
                            cval = float(c)
                        except:
                            cval = -1.0
                        if cval > 0:
                            words.append(t.strip()); confs.append(cval)
                word_count = len(words)
                avg_conf = (sum(confs)/len(confs))/100.0 if confs else 0.0
                text_joined = " ".join(words).strip()
                score = avg_conf * (1.0 if word_count >= 2 else 0.45)   # penalize tiny fragments
                if score > best["avg_conf"]:
                    best = {"avg_conf": score, "raw_conf": avg_conf, "text": text_joined, "words": word_count, "psm": psm, "scale": scale}
        return best

    # ---- EAST-based OCR ----
    def run_east(img_pil_local):
        cv_img = cv2.cvtColor(np.array(img_pil_local), cv2.COLOR_RGB2BGR)
        if not os.path.exists(EAST_MODEL_REL_PATH):
            return {"text":"", "words":0, "avg_conf":0.0, "per_word": [], "boxes": []}
        try:
            r = ocr_via_east_then_tesseract_improved(cv_img, east_path=EAST_MODEL_REL_PATH, min_conf=0.5, crop_upscale=4.0, psm_list=(6,11))
            return r
        except Exception:
            return {"text":"", "words":0, "avg_conf":0.0, "per_word": [], "boxes": []}

    multi = run_multiscale(img_pil)
    east = run_east(img_pil)

    # prefer EAST when multi-scale returns very short fragment but EAST has >=2 words
    prefer_east = (multi.get("words", 0) < 4 and east.get("words", 0) >= 2)

    merged_text, merged_count = merge_multiscale_and_east(multires=multi, eastres=east, conf_threshold=0.60)

    final_text = merged_text
    final_count = merged_count
    final_conf = 0.0
    source = "merged"

    if prefer_east and east.get("per_word"):
        # pick EAST high-confidence words
        words_keep = [e['word'] for e in east.get("per_word", []) if (e.get("conf",0)/100.0) >= 0.6]
        if len(words_keep) >= 1:
            final_text = " ".join(words_keep)
            final_count = len(words_keep)
            final_conf = east.get("avg_conf", 0.0)
            source = "east-prefer"
        else:
            # relax merge threshold
            final_text, final_count = merge_multiscale_and_east(multires=multi, eastres=east, conf_threshold=0.45)
            final_conf = max(multi.get("raw_conf",0.0), east.get("avg_conf",0.0))
            source = "merged-relaxed"
    else:
        if merged_count > 0:
            final_text = merged_text
            final_count = merged_count
            final_conf = max(multi.get("raw_conf",0.0), east.get("avg_conf",0.0))
            source = "merged"
        else:
            # fallback heuristics
            multi_conf = multi.get("raw_conf", 0.0)
            east_conf = east.get("avg_conf", 0.0)
            if east.get("words",0) > multi.get("words",0) and east_conf >= 0.25:
                final_text = east.get("text",""); final_count = east.get("words",0); final_conf = east_conf; source = "east-fallback"
            else:
                final_text = multi.get("text",""); final_count = multi.get("words",0); final_conf = multi_conf; source = "multi-fallback"

    primary = "sans" if final_count >= 1 and final_conf >= 0.40 else "unknown"
    notes = []
    if final_count >= 1 and final_conf >= 0.40:
        notes.append(f"Detected readable text (source={source})")
    elif final_count >= 1:
        notes.append(f"Low-confidence text detected (source={source})")
    else:
        notes.append("No text detected")

    debug = {
        "multi": {"psm": multi.get("psm"), "scale": multi.get("scale"), "words": multi.get("words"), "raw_conf": round(multi.get("raw_conf",0.0),3)},
        "east": {"words": east.get("words"), "avg_conf": east.get("avg_conf"), "boxes": east.get("boxes", [])},
        "merged": {"words": final_count, "prefer_east": prefer_east}
    }
    if east.get("per_word"):
        debug["east"]["per_word"] = east.get("per_word")

    return {
        "primary": primary,
        "confidence": round(final_conf, 3),
        "notes": " | ".join(notes),
        "text_sample": final_text[:400],
        "word_count": int(final_count),
        "debug": debug
    }

# ---------- Full pipeline ----------
def analyze_image_bytes(image_bytes: bytes):
    img = load_image_bytes(image_bytes)
    img_small, _ = preprocess_image_to_pil(img, max_size=1000)
    palette = extract_palette_kmeans(img_small, n_colors=5)
    ent = image_entropy(img_small)
    edges = edge_count(img_small)
    tone = tone_from_palette(palette)
    try:
        typography = typography_preproc_and_ocr(img_small)
    except Exception as e:
        typography = {"primary":"unknown","confidence":0.0,"notes":f"ocr runtime error: {e}","text_sample":"","word_count":0,"debug":{}}

    # spacing heuristic
    if ent < 4.0 or edges < 2000:
        spacing = "airy"
    elif ent > 6.0 or edges > 8000:
        spacing = "compact"
    else:
        spacing = "balanced"

    vibe_tags = []
    if tone in ("warm", "pastel"):
        vibe_tags.append("friendly")
    if tone == "neon":
        vibe_tags.append("bold")
    if typography.get("primary") == "sans":
        vibe_tags.append("modern")

    raw = dict(entropy=ent, edges=edges)

    branddna = {
        "palette": palette,
        "typography": typography,
        "spacing": spacing,
        "density_score": float(ent),
        "tone": tone,
        "vibe_tags": vibe_tags,
        "raw": raw
    }
    return branddna
