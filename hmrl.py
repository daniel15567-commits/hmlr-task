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

def line_for_match(text, start, end):
    left = text.rfind("\n", 0, start) + 1
    right = text.find("\n", end)
    if right == -1:
        right = len(text)
    return text[left:right]


def find_application_numbers(text):
    norm = text.replace("\r\n", "\n").replace("\r", "\n")
    norm = re.sub(r"[ \t]+", " ", norm)

    hits = set()

    patterns = [
        (
            r"(?i)\bApplication\s*(?:Number|No\.?|Ref(?:erence)?)\s*[:\-]?\s*([A-Za-z0-9\/\-]{5,})",
            1,
        ),
        (
            r"(?i)\bApplication\s*(?:No\.?|Number)\b.*?(\d{2}\/\d{2}\/\d{3,6})",
            1,
        ),
        (
            r"\b[A-Za-z]{1,2}\/\d{2}\/\d{3,6}\b",
            0,
        ),
    ]

    for pattern, group in patterns:
        for match in re.finditer(pattern, norm):
            value = match.group(group).strip().rstrip(".,;:|")
            hits.add(value)

    cleaned = []
    for value in hits:
        if not re.search(r"\d", value):
            continue
        if re.fullmatch(r"0\d{3,4}-\d{3,4}", value):
            continue
        if re.fullmatch(r"(19|20)\d{2}\/(19|20)\d{2}", value):
            continue

        idx = norm.find(value)
        if idx != -1:
            line = line_for_match(norm, idx, idx + len(value)).lower()
            if any(key in line for key in ["date of application", "d. of application", "d of application", "date:"]):
                continue

        cleaned.append(value)

    return sorted(set(cleaned))


def looks_like_person_name(text):
    text = (text or "").strip().strip(",")
    if len(text) < 3 or len(text) > 80:
        return False

    low = text.lower()
    bad_words = [
        "applicant", "agent", "named", "reverse", "ref", "date", "part",
        "yours", "faithfully", "sincerely", "council", "office", "notice",
        "town and country", "planning act", "building regulations",
    ]
    if any(word in low for word in bad_words):
        return False

    if any(ch in text for ch in "()[]{}|"):
        return False

    if re.search(r"(?i)\b(mr|mrs|ms|miss|dr|prof)\b", text):
        return True

    tokens = re.findall(r"[A-Za-z]+\.?", text)
    if not (2 <= len(tokens) <= 5):
        return False

    if any(len(token.strip(".")) > 15 for token in tokens):
        return False

    return True

def classify_page_heuristic(text):
    text = (text or "").lower()
    words = re.findall(r"[A-Za-z]{2,}", text)
    word_count = len(words)

    if "decision notice" in text or "town and country planning act" in text:
        return "decision_notice", 0.70
    if "appeal" in text and ("inspectorate" in text or "inspector" in text):
        return "appeal_decision", 0.65
    if "condition" in text and ("reason" in text or "conditions" in text):
        return "conditions_or_reasons", 0.60
    if word_count < 40:
        if any(k in text for k in ["scale", "north", "legend", "ordnance survey", "site plan", "location plan"]):
            return "plan_or_map", 0.60
        return "low_text_page", 0.55
    if any(k in text for k in ["dear", "yours sincerely", "yours faithfully"]):
        return "correspondence_letter", 0.55
    return "other", 0.40


def classify_pages_zero_shot(page_texts, candidate_labels, model_name="facebook/bart-large-mnli"):
    from transformers import pipeline

    clf = pipeline("zero-shot-classification", model=model_name)
    results = []

    for text in page_texts:
        snippet = (text or "")[:2500]
        if not snippet.strip():
            results.append(("other", 0.0))
            continue
        res = clf(snippet, candidate_labels)
        results.append((res["labels"][0], float(res["scores"][0])))

    return results

def looks_like_person_name(text):
    text = (text or "").strip().strip(",")
    if len(text) < 3 or len(text) > 80:
        return False

    low = text.lower()
    bad_words = [
        "applicant", "agent", "named", "reverse", "ref", "date", "part",
        "yours", "faithfully", "sincerely", "council", "office", "notice",
        "town and country", "planning act", "building regulations",
    ]
    if any(word in low for word in bad_words):
        return False

    if any(ch in text for ch in "()[]{}|"):
        return False

    if re.search(r"(?i)\b(mr|mrs|ms|miss|dr|prof)\b", text):
        return True

    tokens = re.findall(r"[A-Za-z]+\.?", text)
    if not (2 <= len(tokens) <= 5):
        return False

    if any(len(token.strip(".")) > 15 for token in tokens):
        return False

    return True


def extract_applicant_names(text):
    norm = text.replace("\r\n", "\n").replace("\r", "\n")
    norm = re.sub(r"[ \t]+", " ", norm)

    block = block_after_label(norm, [
        r"Applicant",
        r"Name of Applicant",
        r"Applicant Name",
        r"Applicant\(s\)",
    ])
    if block:
        first_line = block.splitlines()[0].strip().strip(",")
        if looks_like_person_name(first_line):
            return [first_line]

    match = re.search(r"(?is)\bApplicant\b.*?\n([^\n]{2,80})", norm)
    if match:
        line = match.group(1).strip().strip(",")
        if looks_like_person_name(line):
            return [line]

    names = []
    for match in re.finditer(r"(?i)\bgranted\s+to\s+(.+?)(?:—|-|dated|under|\n)", norm):
        name = match.group(1).strip(" ,.-")
        if looks_like_person_name(name):
            names.append(name)

    return sorted(set(names))
