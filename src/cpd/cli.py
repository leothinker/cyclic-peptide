"""cpd CLI: fetch -> extract -> crop -> parse-xml -> extract-tables pipeline."""
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
@click.argument("xml_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--out", default="data/processed/image_compound_map.json",
              type=click.Path(), show_default=True,
              help="Where to write the JSON map of <img> -> compound/formula.")
@click.option("--patent-id", default=None,
              help="Patent ID stored in the output JSON. "
                   "Defaults to inferring from the XML path (e.g. 'WO2025162428').")
@click.pass_context
def parse_xml(ctx: click.Context, xml_path: str, out: str, patent_id: str | None) -> None:
    """Parse WIPO body XML and map every <img> figure to a compound/formula.

    Reads ``<img file=... id=...>`` tags out of the WIPO FullText body XML,
    looks at 2-3 preceding and 1-2 following <p> siblings for identifier
    keywords (Formula (A), Compound 42, Example 7, Scheme 3, Figure 1),
    and writes a structured JSON map for downstream OCSR.
    """
    from cpd.parsers.wipo_xml import parse_body_xml, write_map_json

    xml = Path(xml_path)
    out_path = Path(out)
    pid = patent_id
    if pid is None:
        # Try: .../WO2025162428/FullText/wo-published-application-body.xml
        #      -> "WO2025162428"
        try:
            pid = xml.parent.parent.name
        except IndexError:
            pid = xml.stem

    with console.status(f"[bold]Parsing[/bold] {xml.name}..."):
        records = parse_body_xml(xml)

    write_map_json(records, out_path, patent_id=pid, source_xml=xml)

    # Summary table.
    counts: dict[str, int] = {}
    for r in records:
        counts[r.category] = counts.get(r.category, 0) + 1

    table = Table(title=f"image_compound_map -> {out_path}", show_lines=False)
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right")
    for cat in ("formula", "scheme", "example", "compound", "figure", "unknown"):
        if cat in counts:
            table.add_row(cat, str(counts[cat]))
    table.add_row("[bold]total[/bold]", f"[bold]{len(records)}[/bold]")
    console.print(table)
    console.print(
        f"[bold green]OK[/bold green] {len(records)} figures mapped -> {out_path}"
    )


@cli.command()
@click.argument("img_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--patent-id", default="WO2025162428", show_default=True,
              help="Patent ID used for naming the output JSON files.")
@click.option("--findings-out", default="data/processed/table_findings.json",
              type=click.Path(), show_default=True,
              help="Where to write the per-image classification JSON.")
@click.option("--blocks-out", default="data/processed/table_blocks.json",
              type=click.Path(), show_default=True,
              help="Where to write the contiguous block summary JSON.")
@click.option("--activity-out", default="data/processed/activity_rows.jsonl",
              type=click.Path(), show_default=True,
              help="Where to write extracted activity rows (JSONL).")
@click.option("--smiles-out", default="data/processed/smiles_rows.jsonl",
              type=click.Path(), show_default=True,
              help="Where to write extracted SMILES rows (JSONL).")
@click.option("--backend", default="stub", show_default=True,
              type=click.Choice(["stub", "regex", "vision-llm"]),
              help="Table content extraction backend. "
                   "Use 'stub' on machines without OCR / VLM access "
                   "(default; the finder still classifies images).")
@click.option("--include-types", default="activity,smiles",
              show_default=True,
              help="Comma-separated table types to extract.")
@click.option("--gap-tolerance", default=2, type=int, show_default=True,
              help="Maximum id-gap between consecutive table pages when "
                   "clustering into a TableBlock.")
@click.pass_context
def extract_tables(
    ctx: click.Context, img_dir: str, patent_id: str,
    findings_out: str, blocks_out: str,
    activity_out: str, smiles_out: str,
    backend: str, include_types: str, gap_tolerance: int,
) -> None:
    """Locate activity / SMILES table images, extract rows, and write JSON.

    Pipeline:

      1. Scan ``img_dir`` and classify every FullText JPG by size, aspect
         ratio, and (optionally) header keywords into activity / smiles /
         structure / other.
      2. Cluster candidates into contiguous ``TableBlock`` ranges so
         downstream consumers can iterate block-by-block.
      3. Run the chosen :class:`TableExtractor` backend over every image
         in the requested blocks and collect rows.
      4. Persist findings JSON, blocks JSON, and per-type row JSONL so
         other tools (and humans) can sanity-check the results.

    No external OCR / VLM is invoked by the ``stub`` backend. To wire a
    real OCR / GPT-4o / Claude backend, use
    :class:`cpd.parsers.table_extractor.RegexTableExtractor` (text-only)
    or :class:`cpd.parsers.table_extractor.VisionLlmTableExtractor`
    (image + LLM, pluggable).
    """
    from cpd.parsers.table_finder import (
        TableType,
        find_tables,
        scan_directory,
        write_findings_json,
    )
    from cpd.parsers.table_extractor import (
        RegexTableExtractor,
        SmilesRow,
        StubTableExtractor,
        VisionLlmTableExtractor,
        enrich_smiles_with_rdkit,
    )
    from cpd.merge import to_jsonl

    img_path = Path(img_dir)
    f_out = Path(findings_out)
    b_out = Path(blocks_out)
    a_out = Path(activity_out)
    s_out = Path(smiles_out)
    types = {t.strip() for t in include_types.split(",") if t.strip()}

    with console.status(f"[bold]Scanning[/bold] {img_path}..."):
        images = scan_directory(img_path)
    blocks = find_tables(images, gap_tolerance=gap_tolerance, table_types=types)

    write_findings_json(images, blocks, f_out, patent_id=patent_id, source_dir=img_path)
    b_out.parent.mkdir(parents=True, exist_ok=True)
    b_out.write_text(
        __import__("json").dumps(
            {
                "patent_id": patent_id,
                "blocks": [b.to_dict() for b in blocks],
                "include_types": sorted(types),
                "gap_tolerance": gap_tolerance,
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    # Build the extractor backend.
    if backend == "stub":
        extractor: object = StubTableExtractor()
    elif backend == "regex":
        extractor = RegexTableExtractor()
    else:  # vision-llm
        extractor = VisionLlmTableExtractor()  # .llm_call must be set externally

    activity_rows: list = []
    smiles_rows: list = []
    if types & {TableType.ACTIVITY}:
        with console.status("[bold]Extracting activity rows[/bold]..."):
            for img in images:
                if img.table_type != TableType.ACTIVITY:
                    continue
                rows = extractor.extract_activity(img.image_path)
                activity_rows.extend(rows)
    if types & {TableType.SMILES}:
        with console.status("[bold]Extracting SMILES rows[/bold]..."):
            for img in images:
                if img.table_type != TableType.SMILES:
                    continue
                rows = extractor.extract_smiles(img.image_path)
                smiles_rows.extend(rows)

    # RDKit canonicalisation for SMILES (silently skipped if RDKit missing).
    smiles_rows = enrich_smiles_with_rdkit(smiles_rows)

    n_a = to_jsonl(activity_rows, a_out)
    n_s = to_jsonl(smiles_rows, s_out)

    # Pretty summary table.
    type_counts: dict[str, int] = {}
    for img in images:
        type_counts[img.table_type] = type_counts.get(img.table_type, 0) + 1
    table = Table(title=f"table extraction ({patent_id})")
    table.add_column("Bucket", style="cyan")
    table.add_column("Count", justify="right")
    for t in TableType.ALL:
        if t in type_counts:
            table.add_row(t, str(type_counts[t]))
    table.add_row("[bold]blocks[/bold]", f"[bold]{len(blocks)}[/bold]")
    table.add_row("activity rows", str(n_a))
    table.add_row("smiles rows", str(n_s))
    console.print(table)
    console.print(f"[bold green]OK[/bold green] findings -> {f_out}")
    console.print(f"[bold green]OK[/bold green] blocks   -> {b_out}")
    console.print(f"[bold green]OK[/bold green] activity -> {a_out}  ({n_a} rows)")
    console.print(f"[bold green]OK[/bold green] smiles   -> {s_out}  ({n_s} rows)")


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
        ("Image-compound map", "processed/image_compound_map.json"),
        ("Table findings", "processed/table_findings.json"),
        ("Table blocks", "processed/table_blocks.json"),
        ("Compounds (JSONL)", "processed/*.jsonl"),
        ("Compounds (CSV)", "processed/*.csv"),
    ):
        n, p = count(sub)
        table.add_row(label, str(n), str(p))
    console.print(table)


if __name__ == "__main__":
    cli()
