from __future__ import annotations

from typing import Protocol

import easyocr
import numpy as np

from app.config import OCRConfig


class OCREngine(Protocol):
    """Minimal interface a pipeline-compatible OCR engine must expose."""

    def readtext(self, image: np.ndarray) -> list:
        ...


def build(cfg: OCRConfig) -> easyocr.Reader:
    """Instantiate the EasyOCR Reader. Expensive — call exactly once per process.

    First invocation downloads model weights (~200 MB) to the user cache.
    """
    return easyocr.Reader(cfg.languages, gpu=cfg.use_gpu, verbose=False)
