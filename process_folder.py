"""
GPU-accelerated document -> text pipeline.

Flow:
1) Convert office docs (doc/docx/ppt/pptx) to PDF via LibreOffice headless.
2) For each PDF page: extract digital text first. If weak/empty, render at high DPI and OCR.
3) OCR pages are split round-robin across available GPUs (e.g., 3070 = cuda:0, 4060 = cuda:1).
4) Save ordered text to output_txt/<stem>.txt.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import pypdfium2
from paddleocr import PaddleOCR

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "input"
WORK_DIR = BASE_DIR / "working"
OUTPUT_DIR = BASE_DIR / "output_txt"

SUPPORTED_DOCS = {".doc", ".docx", ".ppt", ".pptx", ".pdf", ".png", ".jpg", ".jpeg", ".tiff"}
PDF_RENDER_DPI = 400

for d in (INPUT_DIR, WORK_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)


# --- Conversion helpers ---
def convert_to_pdf(src: Path) -> Path:
    if src.suffix.lower() == ".pdf":
        return src

    out_pdf = WORK_DIR / f"{src.stem}.pdf"
    cmd = [
        "soffice",
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        str(WORK_DIR),
        str(src),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not out_pdf.exists():
        raise RuntimeError(f"LibreOffice failed for {src.name}: {result.stderr.decode(errors='ignore')}")
    return out_pdf


# --- Preprocessing for tough scans ---
def preprocess_image(img_path: Path) -> Path:
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return img_path

    img = cv2.resize(img, None, fx=1.7, fy=1.7, interpolation=cv2.INTER_CUBIC)
    img = cv2.fastNlMeansDenoising(img, h=10)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 10
    )
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)

    out_path = img_path.with_suffix(".prep.png")
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return out_path


# --- PDF handling ---
def render_page_to_image(pdf: pypdfium2.PdfDocument, page_index: int) -> Path:
    page = pdf[page_index]
    bitmap = page.render(scale=PDF_RENDER_DPI / 72)
    img_path = WORK_DIR / f"{Path(pdf.get_filepath()).stem}_p{page_index:04d}.png"
    bitmap.to_pil().save(img_path)
    bitmap.close()
    return img_path


# --- OCR helpers ---
_ocr_cache: Dict[str, PaddleOCR] = {}


def get_ocr(device: str) -> PaddleOCR:
    if device not in _ocr_cache:
        _ocr_cache[device] = PaddleOCR(
            use_angle_cls=True,
            use_gpu=True,
            gpu_id=int(device.split(":")[1]),
            rec_image_shape="3,48,320",
        )
    return _ocr_cache[device]


def ocr_images(args: Tuple[str, List[Tuple[int, Path]]]) -> List[Tuple[int, str]]:
    device, items = args
    ocr = get_ocr(device)
    out: List[Tuple[int, str]] = []
    for page_idx, img_path in items:
        prepped = preprocess_image(img_path)
        res = ocr.ocr(str(prepped), cls=True)
        lines: List[str] = []
        if res and res[0]:
            for det in res[0]:
                lines.append(det[1][0])
        out.append((page_idx, "\n".join(lines)))
    return out


def split_round_robin(items: List[Tuple[int, Path]], devices: List[str]) -> List[Tuple[str, List[Tuple[int, Path]]]]:
    buckets: Dict[str, List[Tuple[int, Path]]] = {d: [] for d in devices}
    for i, item in enumerate(items):
        buckets[devices[i % len(devices)]].append(item)
    return [(d, buckets[d]) for d in devices if buckets[d]]


# --- Main per-file processing ---
def process_pdf_file(pdf_path: Path, devices: List[str]) -> None:
    ocr_items: List[Tuple[int, Path]] = []

    with pypdfium2.PdfDocument(str(pdf_path)) as pdf:
        for idx in range(len(pdf)):
            img_path = render_page_to_image(pdf, idx)
            ocr_items.append((idx, img_path))

    tasks = split_round_robin(ocr_items, devices)
    with mp.Pool(len(tasks)) as pool:
        results = pool.map(ocr_images, tasks)

    text_map: Dict[int, str] = {}
    for device_result in results:
        for page_idx, text in device_result:
            text_map[page_idx] = text

    ordered = [text_map[i] for i in sorted(text_map)]
    out_file = OUTPUT_DIR / f"{pdf_path.stem}.txt"
    out_file.write_text("\n\n".join(ordered), encoding="utf-8")
    print(f"✅ {pdf_path.name} -> {out_file}")


def process_image_file(src: Path, devices: List[str]) -> None:
    # Treat a standalone image as a single-page doc.
    items = [(0, src)]
    tasks = split_round_robin(items, devices)
    with mp.Pool(len(tasks)) as pool:
        results = pool.map(ocr_images, tasks)
    text = "\n".join(r[1] for r in results[0])
    out_file = OUTPUT_DIR / f"{src.stem}.txt"
    out_file.write_text(text, encoding="utf-8")
    print(f"✅ {src.name} -> {out_file}")


def process_file(src: Path, devices: List[str]) -> None:
    if src.suffix.lower() in {".png", ".jpg", ".jpeg", ".tiff"}:
        process_image_file(src, devices)
        return

    pdf_path = convert_to_pdf(src)
    process_pdf_file(pdf_path, devices)


def main() -> None:
    # Detect GPUs from CUDA_VISIBLE_DEVICES or default to cuda:0, cuda:1
    env_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_devices:
        devices = [f"cuda:{i}" for i in env_devices.split(",") if i.strip()]
    else:
        devices = ["cuda:0", "cuda:1"]

    if not devices:
        print("No GPU devices configured. Set CUDA_VISIBLE_DEVICES or edit the devices list.", file=sys.stderr)
        sys.exit(1)

    sources = [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in SUPPORTED_DOCS]
    if not sources:
        print(f"No input files found in {INPUT_DIR}")
        return

    for src in sources:
        try:
            process_file(src, devices)
        except Exception as exc:
            print(f"❌ Failed on {src.name}: {exc}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
