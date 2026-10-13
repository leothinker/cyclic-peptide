"""Render all 405 DECIMER SMILES back to 2D structures and build a
side-by-side grid vs the original cropped structure PNG.

Render style: publication-grade via cpd.render (CoordGen + ring-system
templates + coloured heteroatoms + stereo wedges), much closer to
ChemDraw than RDKit's default macrocycle "donut" layout.

Output:
  data/processed/roundtrip_grid.png        (default: --hard-cap 200)
  data/processed/roundtrip_grid.json
  data/processed/roundtrip_grid.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, "src")

from rdkit import Chem
from rdkit import RDLogger
from PIL import Image, ImageDraw, ImageFont

RDLogger.DisableLog("rdApp.*")

from cpd.render import render_mol_pil  # type: ignore[import-not-found]

CELLS_DIR = Path("data/processed/cells")
CACHE = Path("data/processed/decimer_cache.json")
OUT_JSON = Path("data/processed/roundtrip_grid.json")
OUT_TXT = Path("data/processed/roundtrip_grid.txt")
OUT_PNG = Path("data/processed/roundtrip_grid.png")

TILE = (220, 160)
FONT = None


def _font(size: int = 11):
    global FONT
    if FONT is None:
        try:
            FONT = ImageFont.truetype("arial.ttf", size)
        except OSError:
            FONT = ImageFont.load_default()
    return FONT


def _render_orig(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGBA")
    img.thumbnail(TILE)
    canvas = Image.new("RGBA", TILE, (255, 255, 255, 255))
    canvas.paste(img, ((TILE[0] - img.width) // 2, (TILE[1] - img.height) // 2), img)
    return canvas


def _render_decimer(smi: str) -> Image.Image | None:
    return render_mol_pil(smi, size=TILE, linearize_macrocycle=True)


def _blank_tile(reason: str) -> Image.Image:
    canvas = Image.new("RGBA", TILE, (245, 245, 245, 255))
    draw = ImageDraw.Draw(canvas)
    f = _font(10)
    draw.text((10, 70), reason, fill=(150, 50, 50), font=f)
    return canvas


def _label_tile(tile: Image.Image, label: str) -> Image.Image:
    h = tile.height + 16
    canvas = Image.new("RGBA", (tile.width, h), (255, 255, 255, 255))
    canvas.paste(tile, (0, 0), tile)
    draw = ImageDraw.Draw(canvas)
    draw.text((4, tile.height + 1), label, fill=(20, 20, 20), font=_font(10))
    return canvas


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hard-cap", type=int, default=200, help="max cmpds to render (sorted by min_conf ascending)")
    p.add_argument("--out", type=Path, default=OUT_PNG)
    p.add_argument("--only-low-conf", action="store_true", help="only include cmpds where DECIMER min_conf < 0.5")
    args = p.parse_args()

    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    entries = cache.get("entries", cache) if isinstance(cache, dict) else cache
    items = []
    for cmpd_id, entry in entries.items():
        if not isinstance(entry, dict):
            continue
        raw = (entry.get("raw") or entry.get("cleaned") or "").strip()
        if not raw:
            continue
        items.append((cmpd_id, raw, float(entry.get("min_conf") or 0.0)))

    items.sort(key=lambda x: x[2])
    if args.only_low_conf:
        items = [it for it in items if it[2] < 0.5]
    items = items[: args.hard_cap]

    print(f"[grid] rendering top {len(items)} cmpds (sorted by min_conf asc) with CoordGen")
    t0 = time.time()

    rows: list[list[tuple[str, Image.Image]]] = []
    meta: list[dict] = []
    aborted: list[tuple[str, str]] = []

    for idx, (cmpd_id, raw, min_conf) in enumerate(items):
        if time.time() - t0 > 12 * 60:
            print(f"[grid] hard time budget reached after {idx} cmpds")
            break
        struct_png = list(CELLS_DIR.glob(f"*_{cmpd_id}_struct.png"))
        if not struct_png:
            aborted.append((cmpd_id, "no struct png"))
            continue
        orig = _label_tile(_render_orig(struct_png[0]), f"orig {cmpd_id}")
        rd = _render_decimer(raw)
        if rd is None:
            rd_labelled = _label_tile(_blank_tile("DECIMER\ninvalid"), f"DECIMER {cmpd_id}")
            formula = None
        else:
            mol = Chem.MolFromSmiles(raw)
            formula = Chem.rdMolDescriptors.CalcMolFormula(mol) if mol else None
            rd_labelled = _label_tile(rd, f"DECIMER {cmpd_id}\nf={formula}\nconf={min_conf:.2f}")
        rows.append([(f"orig_{cmpd_id}", orig), (f"decimer_{cmpd_id}", rd_labelled)])
        meta.append({"cmpd_id": cmpd_id, "min_conf": min_conf, "formula": formula, "raw_len": len(raw)})
        if (idx + 1) % 20 == 0:
            print(f"[grid] {idx+1}/{len(items)}  ({time.time()-t0:.1f}s)")

    n_rows = len(rows)
    if n_rows == 0:
        print("[grid] nothing to render")
        return 1
    cell_w = TILE[0]
    cell_h = TILE[1] + 18
    gap_w = 8
    gap_h = 12
    width = cell_w * 2 + gap_w
    height = n_rows * (cell_h + gap_h)
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    for r, row in enumerate(rows):
        for c, (_, img) in enumerate(row):
            x = c * (cell_w + gap_w)
            y = r * (cell_h + gap_h)
            canvas.paste(img, (x, y), img)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(args.out, "PNG", optimize=True)
    OUT_JSON.write_text(
        json.dumps({"cap": args.hard_cap, "rendered": n_rows, "items": meta, "aborted": aborted}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_lines = [
        "# Round-trip grid summary (CoordGen render)",
        f"# rendered {n_rows} cmpds (out of {len(items)} candidates)",
        f"# time taken: {time.time()-t0:.1f}s",
        "",
        "## cmpds (sorted by DECIMER min_conf ascending)",
    ]
    for m in meta:
        summary_lines.append(
            f"  - cmpd {m['cmpd_id']:<6s} min_conf={m['min_conf']:.3f} formula={m['formula']!r:<25s} raw_len={m['raw_len']}"
        )
    if aborted:
        summary_lines += ["", "## aborted"]
        summary_lines += [f"  - {c}: {r}" for c, r in aborted]
    summary_lines.append("")
    OUT_TXT.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"[grid] wrote {args.out} ({n_rows} cmpds, {time.time()-t0:.1f}s)")
    print(f"[grid] wrote {OUT_JSON} and {OUT_TXT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
