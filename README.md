# Python GPU OCR (Dual GPU)

Turn mixed office docs/PDFs/images into text using both GPUs. The pipeline tries digital text first, then OCRs only the image pages with heavy preprocessing for hard scans.

## Prereqs
- Windows + NVIDIA GPUs (3070 = `cuda:0`, 4060 = `cuda:1`)
- Python 3.10+ (recommended)
- LibreOffice on PATH (`soffice`) for doc/docx/ppt/pptx → PDF
- CUDA drivers + cuDNN (usual PaddleOCR GPU requirements)

## Install
```powershell
cd C:\xampp\htdocs\python-gpu-ocr
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install paddleocr paddlepaddle-gpu pypdfium2 opencv-python-headless numpy
# If you prefer docx2pdf/pdf2image, add: pip install docx2pdf pdf2image
```

## Layout
```
python-gpu-ocr/
  input/       # drop doc/docx/ppt/pptx/pdf/png/jpg here
  working/     # auto: converted PDFs + rendered pages
  output_txt/  # auto: final text per source file
  process_folder.py
```

## Run
```powershell
cd C:\xampp\htdocs\python-gpu-ocr
.\.venv\Scripts\activate
python process_folder.py
```
Devices default to `cuda:0` and `cuda:1`. Override with `CUDA_VISIBLE_DEVICES=0,1` if needed.

## How it works
1) Convert office docs to PDF via headless LibreOffice.
2) Render every PDF page at 400 DPI → preprocess (upscale, denoise, CLAHE, adaptive threshold, morphology) → OCR with PaddleOCR (angle classifier on).
3) Pages are split round-robin across GPUs for parallel OCR.
4) Ordered text is saved to `output_txt/<file>.txt`.

Images dropped in `input/` are OCR’d directly (single page). Every page is OCR’d, even if it has digital text.

## Tuning
- `MIN_TEXT_CHARS` (in script): raise/lower the “digital text is good enough” threshold.
- `PDF_RENDER_DPI`: 350–400 is a good balance; raise for tiny text.
- Preprocessing: adjust CLAHE/threshold/morphology in `preprocess_image` if scans differ.
- PaddleOCR: set `lang="en"` (default) or other langs; bump `rec_image_shape` or `max_text_length` for long lines.

## Notes
- If LibreOffice isn’t on PATH, set the full path to `soffice` in `convert_to_pdf`.
- For large batches, LibreOffice startup dominates doc conversion; consider batching runs.
- If a page is digitally readable but still poor, force OCR by lowering `MIN_TEXT_CHARS` or adding a “force OCR” list. 
