"""Image-tampering detection module.

Top-level surface:
  * `analyze(path, engine, ...)` — run the full pipeline and get a
    `PageReport` per page.
  * `build_engine(...)` — build a low-level tampering engine (DocTamper, mock).
  * `format_report(report)` — human-readable formatter for CLI / logs.
  * Contract dataclasses (PageReport, Verdict, Region, ...).
"""
from detector.detectors import DocTamperDetector, MVSSNetDetector
from detector.engine import (
    DocTamperEngine,
    MVSSNetEngine,
    MockEngine,
    TamperingEngine,
    build_engine,
    build_mvssnet_engine,
)
from detector.loader import Page, load
from detector.localizers import FaceLocalizer, build_face_localizer
from detector.pipeline import analyze
from detector.report import (
    Artifacts,
    Confidence,
    DocTamperResult,
    DocumentType,
    ExecutionMetadata,
    FaceDetection,
    MVSSNetResult,
    PageReport,
    Region,
    Verdict,
    Zone,
    format_report,
)
from detector.thresholds import DEFAULT, Thresholds
from detector.visualization import save_face_crop, save_heatmap, save_overlay

__all__ = [
    "Page",
    "load",
    "TamperingEngine",
    "MockEngine",
    "DocTamperEngine",
    "MVSSNetEngine",
    "build_engine",
    "build_mvssnet_engine",
    "DocTamperDetector",
    "MVSSNetDetector",
    "FaceLocalizer",
    "build_face_localizer",
    "Verdict",
    "Confidence",
    "Zone",
    "DocumentType",
    "Region",
    "FaceDetection",
    "DocTamperResult",
    "MVSSNetResult",
    "Artifacts",
    "ExecutionMetadata",
    "PageReport",
    "format_report",
    "Thresholds",
    "DEFAULT",
    "analyze",
    "save_heatmap",
    "save_overlay",
    "save_face_crop",
]
