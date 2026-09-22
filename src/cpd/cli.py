"""cpd CLI: fetch -> extract -> crop -> ocr pipeline."""
from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from cpd import __version__
from cpd.extractors.pdf import PyMuPdfExtractor
from cpd.extractors.structure_cropper import StructureCropper, parse_pages
from cpd.scrapers.wipo import WipoFetcher

console = Console()


@click.group()
@click.version_option(__version__, "-V", "--version")
@click.option("--data-dir", type=click.Path(), default="data", show_default=True,
              help="Base data directory.")
@click.pass_context
def cli(ctx: click.Context, data_dir: str) -> None:
    """Build a cyclic-peptide patent dataset."""
    ctx.ensure_object(dict)
    ctx.obj["data_dir"] = Path(data_dir)


@cli.command()
@click.argument("patent_id")
@click.pass_context
def fetch(ctx: click.Context, patent_id: str) -> None:
    """Download a patent HTML by ID (e.g. WO2025162428)."""
    raw = ctx.obj["data_dir"] / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    with WipoFetcher(cache_dir=raw) as fetcher:
        doc = fetcher.fetch(patent_id)
    console.print(f"[bold green]OK[/bold green] {doc.patent_id}: {doc.title[:80]}")
    console.print(f"  applicants: {', '.join(doc.applicants) or '-'}")
    console.print(f"  abstract:   {len(doc.abstract)} chars")
    if doc.pdf_url:
        console.print(f"  pdf:        {doc.pdf_url}")


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True, dir_okay=False))
@click.pass_context
def extract(ctx: click.Context, pdf_path: str) -> None:
    """Split a patent PDF into per-page text + embedded images."""
    images_dir = ctx.obj["data_dir"] / "images"
    extractor = PyMuPdfExtractor(images_dir)
    n_pages = n_imgs = 0
    with console.status("Extracting..."):
        for page in extractor.extract_pages(Path(pdf_path)):
            n_pages += 1
            n_imgs += len(page.images)
    console.print(f"[bold green]OK[/bold green] {n_pages} pages, {n_imgs} images -> {images_dir}")


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--patent-id", default=None,
              help="Patent ID used in the output folder name. "
                   "Defaults to the PDF file stem.")
@click.option("--pages", default=None,
              help="Page selection: '1-5', '1,2,93', or 'all'. "
                   "Use this to iterate quickly on a few pages before "
                   "running over the whole patent.")
@click.option("--dpi", default=200, type=int, show_default=True,
              help="Render resolution for the per-page pixmap.")
@click.option("--out-dir", default="data/structures", type=click.Path(),
              show_default=True,
              help="Where to write the cropped single-structure PNGs.")
@click.option("--padding", default=12, type=int, show_default=True,
              help="Padding (pixels) around each crop, at the rendered DPI.")
@click.pass_context
def crop(
    ctx: click.Context, pdf_path: str, patent_id: str | None,
    pages: str | None, dpi: int, out_dir: str, padding: int,
) -> None:
    """Crop isolated structure figures from each page of a patent PDF.

    Unlike `extract`, which dumps every embedded raster XRef, this renders
    each page and locates structure drawings via vector strokes, dropping
    barcodes / headers / footers / full-page tables.
    """
    import pymupdf  # lazy: gated by the [pdf] extra

    pdf = Path(pdf_path)
    pid = patent_id or pdf.stem
    cropper = StructureCropper(
        Path(out_dir), patent_id=pid, dpi=dpi, padding_px=padding,
    )

    doc = pymupdf.open(pdf)
    try:
        total = len(doc)
    finally:
        doc.close()
    page_list = parse_pages(pages, total=total)

    seen_pages: set[int] = set()
    n_crops = 0
    with console.status(f"[bold]Cropping[/bold] {len(page_list)} pages of {pdf.name}..."):
        for region in cropper.crop_pages(pdf, page_list):
            seen_pages.add(region.page)
            n_crops += 1
    console.print(
        f"[bold green]OK[/bold green] {n_crops} crops across "
        f"{len(seen_pages)}/{len(page_list)} pages -> {cropper.out_dir}"
    )


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show what's been downloaded / processed."""
    data: Path = ctx.obj["data_dir"]
    table = Table(title=f"cyclic-peptide dataset ({data})")
    table.add_column("Stage", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Path", style="dim")

    def count(pattern: str) -> tuple[int, Path]:
        parent = data / Path(pattern).parent
        return sum(1 for _ in parent.glob(Path(pattern).name)), parent

    for label, sub in (
        ("Raw HTML", "raw/*.html"),
        ("Raw PDFs", "raw/*.pdf"),
        ("Cropped images", "images/*.png"),
        ("Cropped structures", "structures/**/*.png"),
        ("Compounds (JSONL)", "processed/*.jsonl"),
        ("Compounds (CSV)", "processed/*.csv"),
    ):
        n, p = count(sub)
        table.add_row(label, str(n), str(p))
    console.print(table)


if __name__ == "__main__":
    cli()
