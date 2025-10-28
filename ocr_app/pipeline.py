# ocr_app/pipeline.py
from __future__ import annotations

from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    EasyOcrOptions,
    RapidOcrOptions,
    TesseractOcrOptions,
    OcrMacOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling_core.types.doc.base import ImageRefMode


# ---- Public API -------------------------------------------------------------

def convert_pdf_to_markdown(
    input_pdf: Path,
    out_md_path: Path,
    out_images_dir: Path,
    *,
    device: str = "cuda:0",
    threads: int = 8,
    ocr_backend: str = "easyocr",
    force_full_page_ocr: bool = False,
    images_scale: float = 1.5,
    generate_page_images: bool = False,
    generate_picture_images: bool = True,
    pdf_backend: str = "auto",     # NEW
) -> None:
    """
    Convert a PDF document to Markdown with OCR, saving images to a specified directory.
    """
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_md_path.parent.mkdir(parents=True, exist_ok=True)

    accel = _build_accelerator(device=device, threads=threads)
    ocr_opts = _build_ocr_options(
        backend=ocr_backend,
        force_full_page_ocr=force_full_page_ocr,
        prefer_gpu=_device_is_cuda_like(device),
    )

    pdf_opts = PdfPipelineOptions()
    pdf_opts.accelerator_options = accel
    pdf_opts.do_ocr = True
    pdf_opts.ocr_options = ocr_opts
    pdf_opts.images_scale = float(images_scale)
    pdf_opts.generate_page_images = bool(generate_page_images)
    pdf_opts.generate_picture_images = bool(generate_picture_images)

    # choose initial backend
    fmt = _build_pdf_format_option(pdf_backend, pdf_opts)
    converter = DocumentConverter(format_options={InputFormat.PDF: fmt})

    # TODO:
    # Trim the code duplication with the fallback logic in convert_document_to_markdown().
    try:
        conv = converter.convert(input_pdf)
    except RuntimeError as e:
        msg = str(e)
        # Known docling-parse failure class -> retry with pypdfium2
        if "Invalid code point" in msg or "code point" in msg:
            # hard fallback: pypdfium2 + (optional) force full-page OCR
            fb_opts = PdfPipelineOptions(**pdf_opts.__dict__)
            if not force_full_page_ocr:
                fb_opts.do_ocr = True
                # make sure we bypass a bad text layer
                fb_opts.ocr_options = _build_ocr_options(
                    backend=ocr_backend, force_full_page_ocr=True, prefer_gpu=_device_is_cuda_like(device)
                )
            converter = DocumentConverter(format_options={
                InputFormat.PDF: PdfFormatOption(
                    backend=PyPdfiumDocumentBackend, pipeline_options=fb_opts
                )
            })
            conv = converter.convert(input_pdf)
        else:
            raise

    conv.document.save_as_markdown(
        out_md_path,
        image_mode=ImageRefMode.REFERENCED,
        artifacts_dir=out_images_dir,
    )

# ---- Internals --------------------------------------------------------------

def _build_pdf_format_option(kind: str, pdf_opts: PdfPipelineOptions) -> PdfFormatOption:
    """
    Build a PdfFormatOption based on user preference.
    """
    k = (kind or "auto").strip().lower()
    if k == "pypdfium":
        return PdfFormatOption(backend=PyPdfiumDocumentBackend, pipeline_options=pdf_opts)
    # default: docling-parse v4 backend
    return PdfFormatOption(pipeline_options=pdf_opts)

def _build_accelerator(*, device: str, threads: int) -> AcceleratorOptions:
    """
    Map a user string to Docling's AcceleratorOptions.
    Accepts raw 'cuda:1' too (EasyOCR will still default to cuda:0 per upstream note).
    """
    dev: str | AcceleratorDevice
    key = device.strip().lower()

    if key in {"auto", ""}:
        dev = AcceleratorDevice.AUTO
    elif key == "cpu":
        dev = AcceleratorDevice.CPU
    elif key == "cuda":
        dev = AcceleratorDevice.CUDA
    elif key == "mps":
        dev = AcceleratorDevice.MPS
    elif key.startswith("cuda:") or key.startswith("mps:"):
        # Docling accepts raw device strings on some stacks; EasyOCR may still bind to cuda:0.
        dev = device
    else:
        # Fallback to AUTO if someone passes garbage.
        dev = AcceleratorDevice.AUTO

    return AcceleratorOptions(num_threads=int(threads), device=dev)

def _build_ocr_options(
    *, backend: str, force_full_page_ocr: bool, prefer_gpu: bool
):
    """
    Construct OCR options per selected backend and desired flags.
    - EasyOCR: toggle use_gpu
    - RapidOCR: relies on onnxruntime provider; install `onnxruntime-gpu` to get CUDA
    - Tesseract / Tesseract CLI / OcrMac: CPU stacks
    """
    b = backend.strip().lower()

    if b in {"", "auto"}:
        # Let Docling choose (usually EasyOCR). Pass no options so defaults apply.
        # If you need hard control, pick a backend explicitly.
        return EasyOcrOptions(use_gpu=prefer_gpu, force_full_page_ocr=force_full_page_ocr)

    if b == "easyocr":
        return EasyOcrOptions(use_gpu=prefer_gpu, force_full_page_ocr=force_full_page_ocr)

    if b == "rapidocr":
        # GPU path requires `onnxruntime-gpu` wheels present in the env.
        # Docling discovers providers via onnxruntime; no explicit "device" flag here.
        return RapidOcrOptions(force_full_page_ocr=force_full_page_ocr)

    if b == "tesseract":
        return TesseractOcrOptions(force_full_page_ocr=force_full_page_ocr)

    if b == "tesseract_cli":
        from docling.datamodel.pipeline_options import TesseractCliOcrOptions
        return TesseractCliOcrOptions(force_full_page_ocr=force_full_page_ocr)

    if b == "ocrmac":
        return OcrMacOptions(force_full_page_ocr=force_full_page_ocr)

    # Fallback: same as auto
    return EasyOcrOptions(use_gpu=prefer_gpu, force_full_page_ocr=force_full_page_ocr)

def _device_is_cuda_like(device: str) -> bool:
    d = device.strip().lower()
    return d == "cuda" or d.startswith("cuda:")

