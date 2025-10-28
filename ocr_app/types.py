# ocr_app/types.py
from __future__ import annotations

from typing import TypedDict

class ConvertResult(TypedDict):
    """
    Result of converting a file to PDF.
    """
    md_path: str
    images_dir: str
