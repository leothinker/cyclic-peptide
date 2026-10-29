r"""Round-trip visual validation: original structure vs DECIMER SMILES re-rendered.

For every DECIMER-predicted SMILES that is RDKit-valid, this demo draws
the molecule from the SMILES back into a 2D structure PNG and lays the
two side-by-side (original cropped structure on the left, RDKit
re-render on the right). This is the standard way OCSR papers compare
output quality: you look at the two PNGs and see whether they describe
the same connectivity.

Render style: publication-grade via ``cpd.render.render_mol`` (CoordGen
+ ring templates + coloured heteroatoms + stereo wedges), which is
much closer to ChemDraw than RDKit's default macrocycle "donut" layout.

Run::

    .venv\\Scripts\\python.exe -u _demo_roundtrip.py

Writes to ``demo_roundtrip/``.

By default we render cmpd 1036 (the one we just ran) and 1037
(neighbour on the same page) so we have two reference points for the
morning report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rdkit import Chem
from rdkit import RDLogger
from PIL import Image, ImageDraw, ImageFont

RDLogger.DisableLog("rdApp.*")

from cpd.render import render_mol_pil  # type: ignore[import-not-found]
from DECIMER import predict_SMILES  # type: ignore[import-not-found]

CMPD_IDS = ["1036", "1037"]
PAGE = "I100297"
CELLS_DIR = Path("data/processed/cells")
OUT = Path("demo_roundtrip")
IMG_SIZE = (520, 360)


def _render_original(path: Path, size=IMG_SIZE) -> Image.Image:
    img = Image.open(path).convert("RGBA")
    img.thumbnail(size)
    canvas = Image.new("RGBA", size, (255, 255, 255, 255))
    canvas.paste(img, ((size[0] - img.width) // 2, (size[1] - img.height) // 2), img)
    return canvas


def _render_decimer(smi: str, size=IMG_SIZE) -> Image.Image | None:
    return render_mol_pil(smi, size=size, linearize_macrocycle=True)


def _label(img: Image.Image, text: str, size: int = 16) -> Image.Image:
    new_h = img.height + size + 12
    canvas = Image.new("RGBA", (img.width, new_h), (255, 255, 255, 255))
    canvas.paste(img, (0, 0), img)
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, img.height + 4), text, fill=(20, 20, 20), font=font)
    return canvas


def _side_by_side(left: Image.Image, right: Image.Image, gap: int = 24) -> Image.Image:
    h = max(left.height, right.height)
    w = left.width + gap + right.width
    canvas = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    canvas.paste(left, (0, 0), left)
    canvas.paste(right, (left.width + gap, 0), right)
    return canvas


def _predict_one(cmpd_id: str, struct_png: Path) -> tuple[str | None, list]:
    if not struct_png.exists():
        return None, []
    print(f"[decimer] running on cmpd {cmpd_id} ({struct_png.name}) ...")
    raw = predict_SMILES(str(struct_png), confidence=True)
    if raw is None:
        return None, []
    if isinstance(raw, str):
        return raw, []
    smi, tokens_raw = raw
    return smi, [(t, float(c)) for t, c in tokens_raw]


def _one(cmpd_id: str) -> dict:
    OUT.mkdir(exist_ok=True)
    struct_png = CELLS_DIR / f"PCTCN2025075389-ftappb-{PAGE}_{cmpd_id}_struct.png"
    payload: dict = {"cmpd_id": cmpd_id, "source_png": str(struct_png)}

    smi, tokens = _predict_one(cmpd_id, struct_png)
    payload["smiles"] = smi
    payload["n_tokens"] = len(tokens)
    if tokens:
        confs = [c for _, c in tokens]
        payload["min_conf"] = round(min(confs), 4)
        payload["mean_conf"] = round(sum(confs) / len(confs), 4)
        payload["low_conf_count"] = sum(1 for c in confs if c < 0.5)
    mol = Chem.MolFromSmiles(smi) if smi else None
    payload["rdkit_valid"] = mol is not None
    if mol is not None:
        payload["formula"] = Chem.rdMolDescriptors.CalcMolFormula(mol)

    if struct_png.exists():
        orig_img = _label(_render_original(struct_png), f"original ({struct_png.name})")
        payload["original_png"] = f"cmpd_{cmpd_id}_original.png"
        orig_img.convert("RGB").save(OUT / payload["original_png"], "PNG")

    if mol is not None:
        rd_img = _render_decimer(smi)
        if rd_img is not None:
            rd_labelled = _label(
                rd_img,
                f"DECIMER -> RDKit re-render (CoordGen)\nformula={payload.get('formula', '?')}\ntokens={payload['n_tokens']} min_conf={payload.get('min_conf', '?')}",
            )
            payload["decimer_render_png"] = f"cmpd_{cmpd_id}_decimer_render.png"
            rd_labelled.convert("RGB").save(OUT / payload["decimer_render_png"], "PNG")
            sb = _side_by_side(orig_img, rd_labelled)
            payload["side_by_side_png"] = f"cmpd_{cmpd_id}_side_by_side.png"
            sb.convert("RGB").save(OUT / payload["side_by_side_png"], "PNG")

    return payload


def main() -> int:
    OUT.mkdir(exist_ok=True)
    print(f"[roundtrip] CMPD_IDS={CMPD_IDS}")

    payloads = []
    for cmpd_id in CMPD_IDS:
        try:
            payloads.append(_one(cmpd_id))
        except Exception as exc:
            print(f"[!] cmpd {cmpd_id} failed: {exc}")
            payloads.append({"cmpd_id": cmpd_id, "error": str(exc)})

    (OUT / "roundtrip.json").write_text(
        json.dumps(payloads, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary = [
        "# Round-trip visual validation (CoordGen render)",
        "",
        "# DECIMER predicted SMILES, RDKit re-rendered with CoordGen + ring templates + stereo.",
        "",
    ]
    for p in payloads:
        summary.append(f"## cmpd {p['cmpd_id']}")
        summary.append(f"  source : {p.get('source_png', '?')}")
        summary.append(f"  smiles : {p.get('smiles', '(none)')[:140]!r}")
        summary.append(
            f"  tokens : {p.get('n_tokens', '?')}, mean_conf={p.get('mean_conf', '?')}, "
            f"min_conf={p.get('min_conf', '?')}, low_conf={p.get('low_conf_count', '?')}"
        )
        summary.append(
            f"  rdkit  : {p.get('rdkit_valid', '?')}, formula={p.get('formula', '?')}"
        )
        if "side_by_side_png" in p:
            summary.append(f"  side-by-side image : {p['side_by_side_png']}")
        summary.append("")
    (OUT / "roundtrip.txt").write_text("\n".join(summary), encoding="utf-8")

    print("[roundtrip] summary:")
    for p in payloads:
        print(
            f"  cmpd {p['cmpd_id']}: rdkit_valid={p.get('rdkit_valid')}, "
            f"formula={p.get('formula')}, "
            f"side_by_side={'side_by_side_png' in p}"
        )
    print(f"[roundtrip] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
