# ocr_app/utils.py
from __future__ import annotations

from pathlib import Path

def ensure_dir(path: Path) -> None:
    """
    Create directory if it does not exist.
    """
    path.mkdir(parents=True, exist_ok=True)

def is_pdf(path: Path) -> bool:
    """
    True if path looks like a PDF file.
    """
    return path.suffix.lower() == ".pdf"
