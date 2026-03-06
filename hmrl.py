import argparse
import io
import json
import os
import re
import shutil
import fitz  
import pdfplumber

from collections import Counter
from PIL import Image

try:
    import pytesseract
    HAS_TESS = True
except Exception:
    HAS_TESS = False


def configure_tesseract(path=None):
    if not HAS_TESS:
        raise RuntimeError(
            "pytesseract is not installed.\n"
            "Run: pip install pytesseract\n"
            "You also need the Tesseract OCR application installed."
        )

    candidates = [
        path,
        os.environ.get("TESSERACT_CMD"),
        shutil.which("tesseract"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    ]

    for cmd in filter(None, candidates):
        if os.path.exists(cmd):
            pytesseract.pytesseract.tesseract_cmd = cmd
            return

    raise FileNotFoundError(
        "Tesseract executable not found.\n\n"
        "Install Tesseract OCR, then run with:\n"
        '  --tesseract "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"\n\n'
        "Or set:\n"
        '  setx TESSERACT_CMD "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"'
    )

def extract_text_layers(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return [page.extract_text() or "" for page in pdf.pages]


def is_good_text(text):
    text = (text or "").strip()
    if len(text) < 40 or "(cid:" in text:
        return False
    letters = sum(ch.isalpha() for ch in text)
    return letters / max(len(text), 1) >= 0.12

def ocr_page(page, zoom=2.0):
    if not HAS_TESS:
        raise RuntimeError(
            "OCR was requested but pytesseract is not installed.\n"
            "Run: pip install pytesseract"
        )

    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img, config="--oem 3 --psm 6")


def get_page_text(page, text_layer, zoom=2.0, force_ocr=False):
    clean = (text_layer or "").strip()
    if not force_ocr and is_good_text(clean):
        return clean
    return ocr_page(page, zoom)

def block_after_label(text, labels):
    for label in labels:
        match = re.search(
            rf"(?is)\b{label}\s*[:\-]\s*(.*?)(?:\n\s*\n|\n[A-Z][A-Za-z0-9 /&()'.,-]{{2,}}\s*[:\-]|$)",
            text,
        )
        if match:
            value = match.group(1).strip()
            value = re.sub(r"[ \t]+", " ", value)
            return re.sub(r"\n[ \t]+", "\n", value).strip()
    return None