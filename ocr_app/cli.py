# ocr_app/cli.py  (only the relevant bits)
import typer
from pathlib import Path
from .pipeline import convert_pdf_to_markdown
from .utils import ensure_dir, is_pdf

app = typer.Typer(help="OCR Application CLI")

@app.command()
def run(
    input_document: str         = typer.Option(..., "--input-document", "-i"),
    out_dir: str                = typer.Option("./output", "--out-dir", "-o"),
    device: str                 = typer.Option("cuda:0", "--device"),
    threads: int                = typer.Option(8, "--threads"),
    ocr_backend: str            = typer.Option("easyocr", "--ocr-backend"),
    force_full_page_ocr: bool   = typer.Option(False, "--force-full-page-ocr"),
    images_scale: float         = typer.Option(1.5, "--images-scale"),
    pdf_backend: str            = typer.Option("auto", "--pdf-backend", help="auto|pypdfium"),
):
    input_path = Path(input_document)
    if not input_path.exists():
        typer.secho(f"Input not found: {input_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    out_root = Path(out_dir)
    md_path = out_root / f"{input_path.stem}.md"
    imgs_dir = out_root / "images"
    ensure_dir(out_root)
    ensure_dir(imgs_dir)

    if not is_pdf(input_path):
        typer.secho("Only PDF is supported in this command.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    convert_pdf_to_markdown(
        input_pdf=input_path,
        out_md_path=md_path,
        out_images_dir=imgs_dir,
        device=device,
        threads=threads,
        ocr_backend=ocr_backend,
        force_full_page_ocr=force_full_page_ocr,
        images_scale=images_scale,
        generate_page_images=False,
        generate_picture_images=True,
        pdf_backend=pdf_backend,   
    )

    typer.secho(f"[✓] Markdown: {md_path}", fg=typer.colors.BRIGHT_GREEN)
    typer.secho(f"[✓] Images  : {imgs_dir}", fg=typer.colors.BRIGHT_GREEN)

