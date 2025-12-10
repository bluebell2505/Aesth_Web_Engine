# ocr_test.py
from PIL import Image
import cv2, numpy as np, pytesseract, shutil, sys
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

src = "tests/sample1.jpg"
img = cv2.imread(src)

# upscale if small
h,w = img.shape[:2]
scale = 1.0
if max(h,w) < 1000:
    scale = 2.0
if scale != 1.0:
    img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

# grayscale + denoise
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.fastNlMeansDenoising(gray, h=10)

# increase contrast with CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray = clahe.apply(gray)

# adaptive threshold
th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                           cv2.THRESH_BINARY, 11, 2)

# save preprocessed image for inspection
cv2.imwrite("tests/sample1_preprocessed.png", th)

# run tesseract on the preprocessed image
conf = r'--oem 1 --psm 6'
try:
    data = pytesseract.image_to_data(th, output_type=pytesseract.Output.DICT, config=conf, lang='eng')
    text = " ".join([t for t in data['text'] if t.strip() != ""])
    print("---- OCR TEXT ----")
    print(text[:1000])
    print("---- WORDS & CONFIDENCES ----")
    for w,c in zip(data['text'], data['conf']):
        if w.strip():
            print(w, c)
except Exception as e:
    print("OCR error:", e)
    sys.exit(1)

print("Saved preprocessed image: tests/sample1_preprocessed.png")
