"""Dev scratchpad — edit the INPUT block and run `python run.py`.

Runs the full tampering pipeline (loader -> YuNet -> DocTamper -> MVSS-Net
stub -> decision -> artifacts) and prints one PageReport per page.
"""
from __future__ import annotations

import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

from detector.engine import build_engine
from detector.pipeline import analyze
from detector.report import DocumentType, format_report

# ────────────────────────────────────────────────────────────────
INPUT = "../examples/INE.pdf"
OUTPUT_DIR = "output"
MODEL = "auto"                          
DOCUMENT_TYPE = DocumentType.UNKNOWN   
# ─────────────────────────────────────────────────────────────────


def main() -> None:
    engine = build_engine(model_name=MODEL)
    paths = [INPUT] if isinstance(INPUT, str) else list(INPUT)
    for path in paths:
        reports = analyze(
            path,
            engine,
            output_dir=OUTPUT_DIR,
            document_type=DOCUMENT_TYPE,
        )
        for report in reports:
            print(format_report(report))
            print()


if __name__ == "__main__":
    main()
