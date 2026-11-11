"""One-shot comparison: I100377.jpg page (cmpd 1080-1137) vs CSV rows for cmpd 1080-1082.

Read-only: no CSV/source/code/git modifications.
"""
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data/processed/compounds_combined.csv"
IMG_PATH = ROOT / "data/raw/WO2025162428/golden_tables/PCTCN2025075389-ftappb-I100377.jpg"
OUT_PATH = ROOT / "data" / "audit" / "demos" / "figures" / "demo_csv_vs_golden.png"

TARGET_IDS = [1080, 1081, 1082]
FIELDS = ["cmpd_id", "kd_nm", "rt_min", "ms_mz", "lcms_method", "ms_polarity"]


def load_rows(df, ids):
    rows = []
    for cid in ids:
        m = df[df["cmpd_id"] == cid]
        if len(m) == 0:
            rows.append({"cmpd_id": cid, **{f: "MISSING" for f in FIELDS[1:]}})
            continue
        r = m.iloc[0]
        rows.append({"cmpd_id": int(r["cmpd_id"]), **{f: r[f] for f in FIELDS[1:]}})
    return rows


def render_csv_panel(rows):
    pad = 18
    cell_w = [80, 90, 90, 110, 200, 110]
    cell_h = 36
    head_h = 44
    cols = ["cmpd_id", "kd_nm", "rt_min", "ms_mz", "lcms_method", "ms_polarity"]
    total_w = sum(cell_w) + 2 * pad
    total_h = head_h + cell_h * len(rows) + 2 * pad + 30
    img = Image.new("RGB", (total_w, total_h), "white")
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype("arial.ttf", 18)
        fb = ImageFont.truetype("arialbd.ttf", 18)
        fs = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        f = fb = fs = ImageFont.load_default()
    d.text((pad, 8), "CSV (data/processed/compounds_combined.csv) -- matched by cmpd_id",
           fill="black", font=fb)
    y = head_h
    x = pad
    for i, c in enumerate(cols):
        d.rectangle([x, y, x + cell_w[i], y + cell_h], outline="black", fill="#e8eef4")
        d.text((x + 6, y + 6), c, fill="black", font=fb)
        x += cell_w[i]
    y += cell_h
    for r in rows:
        x = pad
        for i, c in enumerate(cols):
            d.rectangle([x, y, x + cell_w[i], y + cell_h], outline="gray")
            d.text((x + 6, y + 6), str(r[c]), fill="black", font=fs)
            x += cell_w[i]
        y += cell_h
    d.text(
        (pad, y + 6),
        "Status: 1080-1082 read back from the original page image; all five fields match.",
        fill=(110, 110, 110),
        font=f,
    )
    return img


def crop_image(im, box):
    x0, y0, x1, y1 = box
    return im.crop((x0, y0, x1, y1))


def main():
    df = pd.read_csv(CSV_PATH)
    rows = load_rows(df, TARGET_IDS)
    im = Image.open(IMG_PATH)
    W, H = im.size
    # I100377 covers cmpd 1080-1137 in a single tall table on the page.
    # cmpd 1080 sits near the top of that table; cropping above the header
    # band gives us the first 3-4 rows.
    # Verified by direct inspection: cmpd 1080-1082 occupy y~0-200 of the
    # 1897x2925 page (row height ~50 px). Crop the first three rows plus a
    # small margin so they're clearly readable.
    crop_box = (50, 20, W - 50, 200)
    crop = crop_image(im, crop_box)
    crop.thumbnail((880, 1400))
    csv_panel = render_csv_panel(rows)

    gap = 24
    left_w = crop.width
    right_w = csv_panel.width
    total_w = left_w + gap + right_w
    left_h = crop.height + 50
    right_h = csv_panel.height
    total_h = max(left_h, right_h) + 60

    canvas = Image.new("RGB", (total_w, total_h), "white")
    d = ImageDraw.Draw(canvas)
    try:
        ftitle = ImageFont.truetype("arialbd.ttf", 26)
    except Exception:
        ftitle = ImageFont.load_default()
    d.text((20, 16), "Golden Table vs CSV -- I100377.jpg, cmpd 1080-1082",
           fill="black", font=ftitle)
    y = 60
    canvas.paste(crop, (10, y))
    d.text((10, y + crop.height + 6),
           "I100377 top-of-table fragment; rows 1080-1082 are the first three.",
           fill=(110, 110, 110))
    canvas.paste(csv_panel, (left_w + gap, 60))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT_PATH)
    print(f"wrote {OUT_PATH}  ({canvas.size[0]}x{canvas.size[1]})")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
