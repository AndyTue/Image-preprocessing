from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import cv2

from app.models import ProcessedPage


def save_image(page: ProcessedPage, output_dir: str | Path) -> Path:
    """Persist the preprocessed page as a PNG. Returns the output path."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_name = Path(page.source).stem
    path = out_dir / f"{src_name}_p{page.page_number}.png"
    cv2.imwrite(str(path), page.image)
    return path


def save_ocr_json(page: ProcessedPage, output_dir: str | Path) -> Path:
    """Persist the OCR result (agent contract) as JSON. Returns the output path."""
    if page.ocr is None:
        raise ValueError("Page has no OCR result to serialize")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_name = Path(page.source).stem
    path = out_dir / f"{src_name}_p{page.page_number}.json"
    path.write_text(
        json.dumps(asdict(page.ocr), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path
