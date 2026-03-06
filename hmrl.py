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