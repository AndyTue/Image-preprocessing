from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from app.config import load as load_config
from app.io.writer import save_image, save_ocr_json
from app.models import BatchResult
from app.pipeline import process_directory, process_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Document preprocessing + OCR pipeline")
    parser.add_argument("input", help="File path or directory to process")
    parser.add_argument("--output", default="output", help="Output directory (default: output/)")
    parser.add_argument("--config", default=None, help="Path to a TOML config (default: config/defaults.toml)")
    parser.add_argument("--no-ocr", action="store_true", help="Skip the OCR stage")
    parser.add_argument("--visualize", action="store_true", help="Render bbox overlays next to the PNG output")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_ocr = not args.no_ocr

    print("=" * 60)
    print("  Document OCR Pipeline")
    print(f"  Input  : {args.input}")
    print(f"  Output : {args.output}")
    print(f"  OCR    : {'on' if run_ocr else 'off'}")
    print("=" * 60)

    t0 = time.perf_counter()

    if os.path.isdir(args.input):
        result = process_directory(args.input, cfg, run_ocr=run_ocr)
    elif os.path.isfile(args.input):
        if run_ocr:
            from app.ocr import engine as ocr_engine
            reader = ocr_engine.build(cfg.ocr)
        else:
            reader = None
        pages = process_file(args.input, cfg, ocr_reader=reader)
        result = BatchResult(
            passed=[p for p in pages if p.quality.passed],
            failed=[p for p in pages if not p.quality.passed],
        )
    else:
        print(f"[ERROR] Not found: {args.input}")
        sys.exit(1)

    _persist(result, args.output, cfg, visualize=args.visualize)
    _print_summary(result, time.perf_counter() - t0, args.output)

    if not result.passed and result.failed:
        sys.exit(1)


def _persist(result: BatchResult, output_dir: str, cfg, *, visualize: bool) -> None:
    for page in result.passed:
        save_image(page, output_dir)
        if page.ocr is not None:
            save_ocr_json(page, output_dir)
            if visualize:
                _render_overlays(page, output_dir, cfg)


def _render_overlays(page, output_dir: str, cfg) -> None:
    from app.visualization import mpl_overlay
    stem = Path(page.source).stem
    overlay_path = Path(output_dir) / f"{stem}_p{page.page_number}_overlay.png"
    mpl_overlay.render(page.image, page.ocr, overlay_path, cfg.visualization.font_path)


def _print_summary(result: BatchResult, elapsed: float, output_dir: str) -> None:
    print(f"\n{'=' * 60}")
    print("  Results")
    print(f"  Passed : {len(result.passed)} pages  ->  saved to {output_dir}/")
    print(f"  Failed : {len(result.failed)} pages")
    if result.errors:
        print(f"  Errors : {len(result.errors)} items")
    print(f"  Time   : {elapsed:.1f}s")
    print(f"{'=' * 60}")

    for p in result.passed:
        src = os.path.basename(p.source)
        tag = "PASS*" if p.quality.rescued else "PASS"
        ocr_info = ""
        if p.ocr is not None:
            ocr_info = (f"  en={len(p.ocr.english.words)} hi={len(p.ocr.hindi.words)} "
                        f"low={len(p.ocr.low_confidence)}")
        print(f"  [{tag}] {src} p{p.page_number}  score={p.quality.score:.3f}  "
              f"sharp={p.quality.text_sharpness:.1f}  x_height={p.quality.x_height_px:.0f}px{ocr_info}")

    for p in result.failed:
        src = os.path.basename(p.source)
        print(f"  [FAIL] {src} p{p.page_number}  score={p.quality.score:.3f}  {p.quality.reason}")

    for err in result.errors:
        print(f"  [ERR]  {err}")


if __name__ == "__main__":
    main()
