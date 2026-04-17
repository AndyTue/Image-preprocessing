from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# Normalized geometry: ((x1, y1), (x2, y2)) in [0, 1] coordinates.
Geometry = Tuple[Tuple[float, float], Tuple[float, float]]


@dataclass
class Page:
    """A single document page as a raster image."""

    image: np.ndarray
    page_number: int
    dpi: int
    source: str


@dataclass
class QualityReport:
    """Quality assessment result for a single page."""

    text_sharpness: float
    text_contrast: float
    x_height_px: float
    score: float
    passed: bool
    reason: str
    rescued: bool = False


@dataclass
class Word:
    """A single OCR detection with normalized geometry."""

    text: str
    confidence: float
    geometry: Geometry
    original: Optional[str] = None   # set when the word was split from a mixed token


@dataclass
class ScriptBucket:
    """Per-script OCR output: reconstructed text + constituent words."""

    text: str
    words: List[Word]


@dataclass
class OCRResult:
    """Bilingual OCR output — contract consumed by the downstream Gemma 3 agent.

    The shape of this dataclass IS the agent contract. Any change here must
    be coordinated with the agent.
    """

    english: ScriptBucket
    hindi: ScriptBucket
    low_confidence: List[Word]


@dataclass
class ProcessedPage:
    """A page after preprocessing, quality assessment, and (optionally) OCR."""

    image: np.ndarray
    page_number: int
    dpi: int
    source: str
    quality: QualityReport
    ocr: Optional[OCRResult] = None


@dataclass
class BatchResult:
    """Aggregated result of processing one or more documents."""

    passed: List[ProcessedPage] = field(default_factory=list)
    failed: List[ProcessedPage] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.passed) + len(self.failed)
