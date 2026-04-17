from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from app.config import Config
from app.io.loader import SUPPORTED_ALL, load
from app.models import BatchResult, Page, ProcessedPage
from app.ocr import agent_output, detect, engine as ocr_engine
from app.preprocessing.enhance import enhance
from app.preprocessing.preprocess import preprocess
from app.quality.assess import assess


def process_file(
    file_path: str,
    cfg: Config,
    ocr_reader=None,
) -> List[ProcessedPage]:
    """Process a single document file end-to-end.

    If the first quality pass fails, retry with a CLAHE + unsharp rescue and
    keep whichever version scores higher. Only pages that pass quality get
    sent through OCR. OCR runs serially — EasyOCR's Reader is not thread-safe.
    """
    pages = load(file_path, cfg.io.pdf_render_dpi)
    results: List[ProcessedPage] = []

    for page in pages:
        processed = preprocess(page, cfg.preprocessing)
        report = assess(processed, cfg.quality)

        if not report.passed:
            rescued_img = enhance(processed.image, cfg.preprocessing.enhance)
            rescued_page = Page(
                image=rescued_img,
                page_number=processed.page_number,
                dpi=processed.dpi,
                source=processed.source,
            )
            rescued_report = assess(rescued_page, cfg.quality)
            if rescued_report.score > report.score:
                processed = rescued_page
                report = rescued_report
                report.rescued = True

        ocr_result = None
        if report.passed and ocr_reader is not None:
            words = detect.detect(processed.image, ocr_reader)
            ocr_result = agent_output.build(words, cfg.ocr)

        results.append(ProcessedPage(
            image=processed.image,
            page_number=processed.page_number,
            dpi=processed.dpi,
            source=processed.source,
            quality=report,
            ocr=ocr_result,
        ))
    return results


def process_batch(
    file_paths: List[str],
    cfg: Config,
    run_ocr: bool = True,
) -> BatchResult:
    """Process multiple files. Preprocess+quality run in parallel, OCR serially."""
    result = BatchResult()
    t0 = time.perf_counter()

    reader = ocr_engine.build(cfg.ocr) if run_ocr else None

    # Stage 1: parallel preprocess + quality (no OCR yet).
    intermediate: List[ProcessedPage] = []
    with ThreadPoolExecutor(max_workers=cfg.pipeline.workers) as pool:
        futures = {
            pool.submit(process_file, fp, cfg, None): fp
            for fp in file_paths
        }
        for future in as_completed(futures):
            fp = futures[future]
            try:
                intermediate.extend(future.result())
            except Exception as exc:
                result.errors.append(f"{fp}: {exc}")

    # Stage 2: serial OCR on pages that passed quality.
    for p in intermediate:
        if p.quality.passed and reader is not None:
            try:
                words = detect.detect(p.image, reader)
                p.ocr = agent_output.build(words, cfg.ocr)
            except Exception as exc:
                result.errors.append(f"{p.source} p{p.page_number}: OCR failed: {exc}")

        if p.quality.passed:
            result.passed.append(p)
        else:
            result.failed.append(p)

    elapsed = time.perf_counter() - t0
    print(f"  Batch: {result.total} pages from {len(file_paths)} files in {elapsed:.1f}s")
    return result


def process_directory(directory: str, cfg: Config, run_ocr: bool = True) -> BatchResult:
    """Discover and process all supported files in a directory."""
    files = sorted(
        str(p) for p in Path(directory).iterdir()
        if p.suffix.lower() in SUPPORTED_ALL
    )
    if not files:
        print(f"  No supported files found in: {directory}")
        return BatchResult()
    print(f"  Found {len(files)} files in: {directory}")
    return process_batch(files, cfg, run_ocr)
