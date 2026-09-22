"""cpd CLI: fetch -> extract -> ocr pipeline."""
from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from cpd import __version__
from cpd.extractors.pdf import PyMuPdfExtractor
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
        ("Compounds (JSONL)", "processed/*.jsonl"),
        ("Compounds (CSV)", "processed/*.csv"),
    ):
        n, p = count(sub)
        table.add_row(label, str(n), str(p))
    console.print(table)


if __name__ == "__main__":
    cli()
