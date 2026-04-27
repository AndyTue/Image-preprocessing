"""
Dev scratchpad: edita INPUT y corre `python run.py`.

Para probar varios documentos, o pasa una lista, o agrégalos uno por uno
cambiando INPUT. No toca el pipeline oficial — solo envuelve app.cli para
iterar rápido.
"""
from __future__ import annotations

import sys
from pathlib import Path

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore

from app.config import load as load_config
from app.io.writer import save_image, save_ocr_json
from app.ocr import engine as ocr_engine
from app.pipeline import process_file

# ─────────────────────────────────────────────────────────────────
INPUT = "../examples/visaliz.pdf"  # str or list[str] 
OUTPUT = "output"
RUN_OCR = True
VISUALIZE = True
# ──────────────────────────────────────────────────────────────────


def run_one(path: str, cfg, reader, output_dir: str, visualize: bool) -> None:
    print(f"\n {path}")
    pages = process_file(path, cfg, ocr_reader=reader)
    for p in pages:
        if not p.quality.passed:
            print(f"  [FAIL] p{p.page_number}  score={p.quality.score:.3f}  {p.quality.reason}")
            continue

        img_path = save_image(p, output_dir)
        tag = "PASS*" if p.quality.rescued else "PASS"
        ocr_info = ""
        if p.ocr is not None:
            save_ocr_json(p, output_dir)
            ocr_info = (f"  en={len(p.ocr.english.words)} "
                        f"hi={len(p.ocr.hindi.words)} "
                        f"low={len(p.ocr.low_confidence)}")
        print(f"  [{tag}] p{p.page_number}  score={p.quality.score:.3f}{ocr_info}  → {img_path.name}")

        if visualize and p.ocr is not None:
            from app.visualization import mpl_overlay
            stem = Path(p.source).stem
            out = Path(output_dir)
            mpl_overlay.render(p.image, p.ocr, out / f"{stem}_p{p.page_number}_overlay.png",
                               cfg.visualization.font_path)


def main() -> None:
    cfg = load_config()
    reader = ocr_engine.build(cfg.ocr) if RUN_OCR else None

    inputs = [INPUT] if isinstance(INPUT, str) else list(INPUT)
    for path in inputs:
        run_one(path, cfg, reader, OUTPUT, VISUALIZE)


if __name__ == "__main__":
    main()
