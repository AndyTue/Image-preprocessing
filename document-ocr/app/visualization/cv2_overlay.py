from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.models import OCRResult, Word

_GREEN = (34, 197, 94)
_ORANGE = (0, 165, 255)
_RED = (59, 52, 220)


def render(image: np.ndarray, ocr: OCRResult, output_path: str | Path) -> Path:
    """Write a PNG with bboxes colored by bucket: green/orange/red."""
    vis = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = vis.shape[:2]

    for word in ocr.english.words:
        _draw_box(vis, word, w, h, _GREEN, with_text=True)
    for word in ocr.hindi.words:
        _draw_box(vis, word, w, h, _ORANGE, with_text=False)
    for word in ocr.low_confidence:
        _draw_box(vis, word, w, h, _RED, with_text=False)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), vis)
    return path


def _draw_box(vis: np.ndarray, word: Word, w: int, h: int, color: tuple, with_text: bool) -> None:
    (x1, y1), (x2, y2) = word.geometry
    pt1 = (int(x1 * w), int(y1 * h))
    pt2 = (int(x2 * w), int(y2 * h))
    cv2.rectangle(vis, pt1, pt2, color, 2)
    if with_text:
        cv2.putText(vis, word.text, (pt1[0], pt1[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
