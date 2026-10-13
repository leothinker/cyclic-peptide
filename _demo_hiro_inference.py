"""Hiro-OCSR inference on cmpd 1036 (+ 1037 for context).

Standalone script that loads ``PatSnap/Hiro-OCSR`` (Qwen3-VL-8B) and
runs a single image->SMILES prediction. We pin ``dtype="bfloat16"``,
``device="auto"`` (CPU by default), ``attn_implementation=None`` (HF
default), and ``local_files_only=True`` (so we cannot accidentally
re-download). The script saves:

  * the raw Hiro SMILES string
  * the Hiro-OCSR ``classify_smiles`` result (image_type + features)
  * the time taken
  * an RDKit round-trip render PNG so the result can be visually
    compared to the original cropped structure and the DECIMER render

We DO NOT save anything to the global cache; this is a standalone
report artifact.

Run::

    .venv\\Scripts\\python.exe -u _demo_hiro_inference.py

Writes to ``demo_hiro_inference/``.

Environment expectations::

    HIRO_MODEL_DIR=.hiro-model       # downloaded checkpoint path
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, "src")

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem, Draw
from PIL import Image, ImageDraw, ImageFont

RDLogger.DisableLog("rdApp.*")

from hiro_ocsr import HiroOCSR  # type: ignore[import-not-found]
from hiro_ocsr.classify import classify_smiles  # type: ignore[import-not-found]

CMPD_IDS = ["1036", "1037"]
PAGE = "I100297"
CELLS_DIR = Path("data/processed/cells")
MODEL_DIR = os.environ.get("HIRO_MODEL_DIR", ".hiro-model")
OUT = Path("demo_hiro_inference")
IMG_SIZE = (520, 360)


def _image_grid(items: list[tuple[str, Image.Image]]) -> Image.Image:
    rows = len(items)
    cols = len(items[0])
    cell_w = IMG_SIZE[0]
    cell_h = IMG_SIZE[1] + 28
    canvas = Image.new("RGBA", (cols * cell_w + (cols - 1) * 18, rows * cell_h + (rows - 1) * 18), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    for r, row in enumerate(items):
        for c, (label, img) in enumerate(row):
            x = c * (cell_w + 18)
            y = r * (cell_h + 18)
            labelled = Image.new("RGBA", (cell_w, cell_h), (255, 255, 255, 255))
            labelled.paste(img, ((cell_w - img.width) // 2, 4))
            draw_lbl = ImageDraw.Draw(labelled)
            draw_lbl.text((8, img.height + 6), label, fill=(20, 20, 20), font=font)
            canvas.paste(labelled, (x, y), labelled)
    return canvas


def _hiro_render(smi: str) -> Image.Image | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        AllChem.Compute2DCoords(mol)
        png = Draw.MolToImage(mol, size=IMG_SIZE)
        return png.convert("RGBA")
    except Exception:
        return None


def _orig_render(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGBA")
    img.thumbnail(IMG_SIZE)
    canvas = Image.new("RGBA", IMG_SIZE, (255, 255, 255, 255))
    canvas.paste(img, ((IMG_SIZE[0] - img.width) // 2, (IMG_SIZE[1] - img.height) // 2), img)
    return canvas


def main() -> int:
    OUT.mkdir(exist_ok=True)
    model_dir = Path(MODEL_DIR)
    if not model_dir.exists():
        print(f"[!] model dir missing: {model_dir}")
        print("    download PatSnap/Hiro-OCSR first (see hiro_download.log)")
        return 1

    print(f"[hiro] loading model from {model_dir} ...")
    t0 = time.time()
    model = HiroOCSR(model_path=str(model_dir), load=True, dtype="bfloat16")
    t_load = time.time() - t0
    print(f"[hiro] loaded in {t_load:.1f}s, device={model.device}, dtype={model.dtype}")

    payloads = []
    for cmpd_id in CMPD_IDS:
        struct_png = CELLS_DIR / f"PCTCN2025075389-ftappb-{PAGE}_{cmpd_id}_struct.png"
        if not struct_png.exists():
            print(f"[!] missing {struct_png}")
            continue
        print(f"[hiro] running inference on cmpd {cmpd_id} ({struct_png.name}) ...")
        t1 = time.time()
        result = model.run({"image": str(struct_png), "depict": False})
        t_inf = time.time() - t1
        if not result:
            print(f"[!] no result for cmpd {cmpd_id}")
            continue
        entry = result[0]
        smi = entry.get("SMILES") if isinstance(entry, dict) else None
        if not smi:
            print(f"[!] no SMILES for cmpd {cmpd_id}: entry={entry}")
            continue
        mol = Chem.MolFromSmiles(smi)
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol) if mol else None
        cls = classify_smiles(smi) if smi else ("invalid", {})
        row = {
            "cmpd_id": cmpd_id,
            "source_png": str(struct_png),
            "smiles": smi,
            "smiles_len": len(smi),
            "rdkit_valid": mol is not None,
            "formula": formula,
            "image_type": cls[0] if isinstance(cls, tuple) else "n/a",
            "features": cls[1] if isinstance(cls, tuple) else {},
            "inference_seconds": round(t_inf, 2),
            "load_seconds": round(t_load, 2),
        }
        try:
            orig = _orig_render(struct_png)
            (OUT / f"cmpd_{cmpd_id}_original.png").write_bytes(
                Image.new("RGB", IMG_SIZE, (255, 255, 255)).tobytes()
            )
            (OUT / f"cmpd_{cmpd_id}_original.png").unlink()
            row["original_png"] = f"cmpd_{cmpd_id}_original.png"
            orig.convert("RGB").save(OUT / row["original_png"], "PNG")
            rd_img = _hiro_render(smi)
            if rd_img is not None:
                row["hiro_render_png"] = f"cmpd_{cmpd_id}_hiro_render.png"
                rd_img.convert("RGB").save(OUT / row["hiro_render_png"], "PNG")
                items = [
                    [("original", orig), ("Hiro -> RDKit re-render", rd_img)]
                ]
                grid = _image_grid(items)
                row["side_by_side_png"] = f"cmpd_{cmpd_id}_side_by_side.png"
                grid.convert("RGB").save(OUT / row["side_by_side_png"], "PNG")
        except Exception as exc:
            row["render_error"] = str(exc)
        payloads.append(row)
        print(f"  -> [{t_inf:.1f}s] smi_len={len(smi)}, formula={formula}, rdkit={mol is not None}, image_type={cls[0] if isinstance(cls, tuple) else 'n/a'}")

    (OUT / "hiro_inference.json").write_text(
        json.dumps(payloads, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = ["# Hiro-OCSR inference on cmpd 1036 + 1037", ""]
    summary.append(f"# model load time: {t_load:.1f}s, device={model.device}, dtype={model.dtype}")
    for p in payloads:
        summary.append(f"## cmpd {p['cmpd_id']}")
        summary.append(f"  source   : {p['source_png']}")
        summary.append(f"  smiles   : {p['smiles'][:120]!r}")
        summary.append(f"  formula  : {p['formula']}")
        summary.append(f"  rdkit    : {p['rdkit_valid']}")
        summary.append(f"  image_type: {p.get('image_type')}")
        summary.append(f"  inference time: {p['inference_seconds']}s")
        if "side_by_side_png" in p:
            summary.append(f"  side_by_side : {p['side_by_side_png']}")
        summary.append("")
    (OUT / "hiro_inference.txt").write_text("\n".join(summary), encoding="utf-8")
    print(f"[hiro] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
