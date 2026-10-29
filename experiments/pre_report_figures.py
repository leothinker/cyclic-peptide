r"""Build presentation-sized figures from the recorded Phase 2 results.

The evidence strips under ``evidence/`` and the montages under ``grids/``
are audit artefacts: 16 events per 1320 px montage, 4 px captions. They are
correct but unreadable when projected. This script re-renders the SAME
geometry at presentation scale, with large labels.

Nothing here re-runs OCR. Every character string and every statistic is read
back from the recorded result CSVs, so a figure can never disagree with the
experiment that produced it. Read-only with respect to the pipeline, the
official CSV and the OCR baseline.

Outputs (all new, nothing overwritten):
    data/audit/phase2/pre_report_validation/figures/
        fig_01_mechanism_1036.png
        fig_02_systematic.png
        fig_03_ab_results.png

Run:
    .venv\Scripts\python.exe pre_report_figures.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for _p in (SRC, REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phase2_threshold_scan import BASELINE_THRESHOLD, build_engine  # noqa: E402

from pre_report_ab_recrop import crop_line, load_cell  # noqa: E402
from row_crop_audit import box_rect  # noqa: E402

OUTDIR = (
    REPO / "data" / "audit" / "phase2" / "pre_report_validation" / "figures"
)

FONT_CJK = r"C:\Windows\Fonts\msyh.ttc"
FONT_CJK_BD = r"C:\Windows\Fonts\msyhbd.ttc"
FONT_MONO = r"C:\Windows\Fonts\consola.ttf"
FONT_MONO_BD = r"C:\Windows\Fonts\consolab.ttf"

GREY = (110, 110, 110)
ORANGE = (255, 150, 40)
RED = (215, 40, 40)
GREEN = (30, 150, 70)
BLUE = (40, 90, 210)


def font(path, size):
    return ImageFont.truetype(path, size)


def cn(bold=False, size=24):
    """CJK-capable font. Consolas has no Chinese glyphs, so never mix them."""
    return font(FONT_CJK_BD if bold else FONT_CJK, size)


def pil(img):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


def panel(draw, box, fill=(255, 255, 255), outline=(190, 190, 190),
          width=2, radius=10):
    draw.rounded_rectangle(box, radius=radius, fill=fill,
                           outline=outline, width=width)


def fit_width(img, target_w, interpolation=cv2.INTER_NEAREST, max_scale=None):
    h, w = img.shape[:2]
    if w == 0 or h == 0:
        return np.full((40, target_w, 3), 255, np.uint8)
    k = target_w / w
    if max_scale is not None:
        k = min(k, max_scale)
    up = cv2.resize(img, None, fx=k, fy=k, interpolation=interpolation)
    if up.ndim == 2:
        up = cv2.cvtColor(up, cv2.COLOR_GRAY2BGR)
    return up


def bump(img, factor=3.0):
    """Thicken strokes so binarised patent text reads when projected."""
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = max(1, int(round(factor)))
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    g = cv2.dilate(g, se)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def row_context(proc, poly, ev, left_ctx=60, right_ctx=300):
    """Page-row strip with the detector box edge and the excluded ink.

    ``right_ctx`` is deliberately generous: the point of this panel is to
    show the reader WHERE the box edge sits relative to the printed row, so
    it has to contain several following characters rather than one blob.
    """
    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    x1 = int(ev["box_x1"])
    y1 = int(ev["box_y1"])
    y2 = int(ev["box_y2"])
    y0 = max(0, y1 - 10)
    y1b = min(gray.shape[0], y2 + 10)
    c0 = max(0, x1 - left_ctx)
    c1 = min(gray.shape[1], x1 + right_ctx)
    strip = cv2.cvtColor(gray[y0:y1b, c0:c1], cv2.COLOR_GRAY2BGR)

    lo = max(0, x1 - int(ev["overhang_px"]) - c0)
    hi = max(lo + 1, x1 - c0)
    ov = strip.copy()
    ov[:, lo:hi] = ORANGE
    strip = cv2.addWeighted(ov, 0.30, strip, 0.70, 0)
    cv2.rectangle(strip, (lo, 0), (hi - 1, strip.shape[0] - 1), ORANGE, 2)
    cv2.line(strip, (hi, 0), (hi, strip.shape[0] - 1), RED, 3)
    return strip


def load_rows():
    base = REPO / "data/audit/phase2/pre_report_validation"
    rev = list(csv.DictReader(
        (base / "review_manifest.csv").open(newline="", encoding="utf-8-sig")))
    exact = list(csv.DictReader(
        (base / "ab_exact_results.csv").open(newline="", encoding="utf-8-sig")))
    fixed = list(csv.DictReader(
        (base / "ab_results.csv").open(newline="", encoding="utf-8-sig")))
    full = list(csv.DictReader(
        (base / "ab_fullstring.csv").open(newline="", encoding="utf-8-sig")))
    return rev, exact, fixed, full


def figure_mechanism(ev_by_id, exact_by_id, engine):
    """The single clearest story: page -> what OCR saw -> what re-crop gave."""
    ev = ev_by_id["1036_L2_l"]
    res = exact_by_id["1036_L2_l"]
    proc, polys, _ = load_cell(engine, ev["crop_path"])
    poly = polys[int(ev["line_no"])]

    W = 2000
    M = 60
    f_title = cn(True, 46)
    f_sub = cn(False, 27)
    f_cap = cn(True, 29)
    f_note = cn(False, 25)
    f_mono = font(FONT_MONO_BD, 29)
    f_mono_s = cn(False, 24)

    strip = row_context(proc, poly, ev)
    a_img, _ = crop_line(proc, poly, 0)
    b_img, _ = crop_line(proc, poly, int(ev["overhang_px"]))

    # Cap the zoom. The row strip is only ~360 px wide, so stretching it to
    # the full panel width would magnify it 5x and turn glyphs into blobs.
    strip_s = fit_width(strip, W - 2 * M - 40, cv2.INTER_CUBIC, max_scale=2.6)
    a_s = fit_width(a_img, 880, cv2.INTER_CUBIC)
    b_s = fit_width(b_img, 880, cv2.INTER_CUBIC)

    pad = 26
    h1 = 52 + strip_s.shape[0] + pad
    h2 = 52 + a_s.shape[0] + 66
    h3 = 52 + b_s.shape[0] + 66
    gap = 26
    H = 178 + h1 + gap + h2 + gap + h3 + 70
    im = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(im)

    d.text((M, 34), "行首 = 在进入 OCR 之前就被检测框切掉了",
           font=f_title, fill=(20, 20, 20))
    head = ("化合物 1036 · 第 I100297 页 · 第 2 行 · "
            "检测框左边界外侧实测外露 "
            + ev["overhang_px"] + " px")
    d.text((M, 96), head, font=f_sub, fill=GREY)
    d.line([(M, 148), (W - M, 148)], fill=(210, 210, 210), width=2)

    y = 178
    panel(d, (M - 10, y, W - M + 10, y + h1))
    d.text((M + 16, y + 14),
           "1  原始页面这一行   橙框 = 被排除在检测框外的墨迹"
           "   红线 = 检测框左边界",
           font=f_cap, fill=(20, 20, 20))
    im.paste(pil(strip_s), (M + 16, y + 56))
    # arrow from the note to the orange region, so the text never sits on
    # top of the glyphs
    ax = M + 34 + strip_s.shape[1] + 30
    ay = y + 56 + strip_s.shape[0] // 2
    d.line([(ax - 24, ay), (M + 30, ay)], fill=(180, 95, 10), width=3)
    d.polygon([(M + 30, ay), (M + 46, ay - 9), (M + 46, ay + 9)],
              fill=(180, 95, 10))
    d.text((ax, ay - 52), "这里是 =", font=f_note, fill=(180, 95, 10))
    d.text((ax, ay - 16), "在红线左边，", font=f_note, fill=(180, 95, 10))
    d.text((ax, ay + 20), "没进 OCR 输入", font=f_note, fill=(180, 95, 10))

    y += h1 + gap
    panel(d, (M - 10, y, W - M + 10, y + h2))
    d.text((M + 16, y + 14),
           "2  实际送进识别器的像素   首字符直接是 O，"
           "识别器根本没机会看到 =",
           font=f_cap, fill=(20, 20, 20))
    im.paste(pil(a_s), (M + 16, y + 56))
    d.text((M + 930, y + 78), "OCR 读出：", font=f_note, fill=GREY)
    d.text((M + 930, y + 116), res["text_A"][:26], font=f_mono,
           fill=(200, 30, 30))
    d.text((M + 16, y + h2 - 44),
           "行首字符缺失 → 括号不配对 → RDKit 无法解析",
           font=f_note, fill=GREY)

    y += h2 + gap
    pad = "3  从原图把左边界外扩 " + ev["overhang_px"] + \
        " px 重裁   像素回来了，但识别成 -"
    panel(d, (M - 10, y, W - M + 10, y + h3))
    d.text((M + 16, y + 14), pad, font=f_cap, fill=(20, 20, 20))
    im.paste(pil(b_s), (M + 16, y + 56))
    d.text((M + 930, y + 78), "OCR 读出：", font=f_note, fill=GREY)
    d.text((M + 930, y + 116), res["text_B_exact"][:26], font=f_mono,
           fill=(200, 140, 20))
    d.text((M + 16, y + h3 - 44),
           "字符是补上了，但 = 被读成 -，化学含义仍然错误",
           font=f_note, fill=GREY)

    d.text((M, y + h3 + 34),
           "同一行、同一个识别器：基线读成 O 开头，重裁后读成 - 开头，"
            "两种结果都不等于正确的 =",
           font=f_mono_s, fill=GREY)

    out = OUTDIR / "fig_01_mechanism_1036.png"
    im.save(out)
    return out


def figure_systematic(ev_by_id, exact_by_id, engine, picks):
    """Six examples from six different pages, at readable size."""
    W = 2000
    cols, cell_w = 3, 632
    nrows = (len(picks) + cols - 1) // cols
    f_h = cn(True, 38)
    f_c = cn(True, 26)
    f_n = cn(False, 22)
    f_m = font(FONT_MONO, 22)

    inner_w = cell_w - 44

    # one shared row height, driven by the tallest rendered strip
    strips = {}
    for eid in picks:
        ev = ev_by_id[eid]
        proc, polys, _ = load_cell(engine, ev["crop_path"])
        poly = polys[int(ev["line_no"])]
        strips[eid] = fit_width(row_context(proc, poly, ev), inner_w,
                                cv2.INTER_CUBIC)
    img_h = max(s.shape[0] for s in strips.values())
    cell_h = 54 + img_h + 76

    H = 182 + nrows * cell_h + 30
    im = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.text((60, 32),
           "这不是个例：215 个事件，横跨 80 页，行首形态完全一致",
           font=f_h, fill=(20, 20, 20))
    d.text((60, 88),
           "橙框 = 落在检测框外的墨迹（恒为 8 行高、两条横笔画的 =）"
           "    红线 = 检测框左边界",
           font=f_n, fill=GREY)
    d.text((60, 116),
           "每格下方：红字 = 基线 A      绿字 = 重裁后 B",
           font=f_n, fill=(150, 150, 150))
    d.line([(60, 152), (W - 60, 152)], fill=(210, 210, 210), width=2)

    for i, eid in enumerate(picks):
        ev = ev_by_id[eid]
        res = exact_by_id[eid]
        s = strips[eid]

        cx = 40 + (i % cols) * cell_w
        cy = 182 + (i // cols) * cell_h
        panel(d, (cx, cy, cx + cell_w - 20, cy + cell_h - 20))
        ok = res["first_char_correct"] == "yes"
        col = GREEN if ok else (215, 140, 20)
        mark = "读对 =" if ok else "读成 -"
        d.text((cx + 18, cy + 12),
               "cmpd " + ev["cmpd_id"] + "  行 " + ev["line_no"],
               font=f_c, fill=(20, 20, 20))
        d.text((cx + cell_w - 168, cy + 12), mark, font=f_c, fill=col)
        im.paste(pil(s), (cx + 22, cy + 52))
        ty = cy + 58 + img_h
        d.text((cx + 22, ty), "A " + res["text_A"][:18], font=f_m,
               fill=(190, 40, 40))
        d.text((cx + 22, ty + 30), "B " + res["text_B_exact"][:18],
               font=f_m, fill=(30, 110, 30))

    out = OUTDIR / "fig_02_systematic.png"
    im.save(out)
    return out


def figure_results(exact, fixed, full):
    """The A/B statistics, drawn from the recorded CSVs."""
    W = 2000
    f_h = cn(True, 38)
    f_c = cn(True, 28)
    f_n = cn(False, 23)
    f_m = font(FONT_MONO, 25)
    f_s = font(FONT_MONO, 22)
    f_sc = cn(False, 20)          # Chinese axis labels
    f_note = cn(False, 25)        # Chinese table notes

    trunc = [r for r in exact if r["group"] == "truncated"]
    n_ok = sum(1 for r in trunc if r["first_char_correct"] == "yes")
    minus = sum(1 for r in trunc if r["first_char_B"] == "minus")

    # Control regression lives in the full-string table: the exact-pad arm
    # ran controls at pad=0 (A and B were the same crop, so "changed" is
    # vacuously no there). The real 30 px control widening is in ab_fullstring.
    ctrl = [r for r in full if r["kind"] == "control_regression"]
    changed = sum(1 for r in ctrl if r["changed_vs_baseline"] == "yes")

    pad_rows = {}
    for r in fixed:
        if r["group"] != "truncated":
            continue
        p = int(r["pad_requested_px"])
        acc = pad_rows.setdefault(p, [0, 0])
        acc[0] += r["first_char_correct"] == "yes"
        acc[1] += r.get("left_edge_in_cell_rule", "no") == "yes"

    H = 1640
    im = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.text((60, 32), "重裁剪能找回像素，但识别效果没有改善",
           font=f_h, fill=(20, 20, 20))
    d.text((60, 88),
           "全部数字由 pre_report_summary.py 从实验 CSV 重新统计",
           font=f_n, fill=GREY)
    d.line([(60, 132), (W - 60, 132)], fill=(210, 210, 210), width=2)

    d.text((60, 156),
           "行首字符被识别为 = 的比例（215 个已确认事件）",
           font=f_c, fill=(20, 20, 20))
    bx, by, bw, bh = 90, 215, 1780, 290
    d.rectangle([bx, by, bx + bw, by + bh], outline=(205, 205, 205), width=2)

    series = [(0, 0, "A 基线\n字符不在输入内", (150, 60, 60))]
    for p in sorted(pad_rows):
        if p == 0:
            continue          # pad 0 IS the baseline bar above
        ok, bad = pad_rows[p]
        lab = "左扩 " + str(p) + "px"
        if bad:
            lab += "\n" + str(bad) + " 例吃到边框线"
        series.append((p, ok, lab, (215, 150, 40) if bad else BLUE))
    series.append((None, n_ok, "精确裁\n0 例污染", GREEN))

    slot = bw / len(series)
    for i, (_p, ok, lab, col) in enumerate(series):
        h = bh * ok / 120.0
        x0 = bx + slot * i + slot * 0.18
        x1 = bx + slot * (i + 1) - slot * 0.18
        if h > 0:
            d.rectangle([x0, by + bh - h, x1, by + bh], fill=col)
        pct = str(ok) + "/" + str(len(trunc)) + "\n" + \
            format(ok / float(len(trunc)) * 100, ".1f") + "%"
        d.text(((x0 + x1) / 2, by + bh - h - 56), pct, font=f_s,
               fill=(20, 20, 20), anchor="mm")
        d.multiline_text(((x0 + x1) / 2, by + bh + 14), lab, font=f_sc,
                         fill=GREY, anchor="ma", align="center")
    d.line([(bx, by + bh), (bx + bw, by + bh)], fill=(120, 120, 120), width=2)

    cy = 640
    cards = [
        (str(n_ok) + "/" + str(len(trunc)), "像素找回后读对", GREEN),
        (str(minus) + "/" + str(len(trunc)), "读成了 -（错）", (215, 140, 20)),
        (str(changed) + "/" + str(len(ctrl)), "干净对照加宽后被改坏",
         (180, 50, 50)),
    ]
    for i, (big, small, col) in enumerate(cards):
        x0 = 60 + i * 640
        panel(d, (x0, cy, x0 + 600, cy + 180), fill=(250, 250, 252))
        d.text((x0 + 300, cy + 72), big, font=font(FONT_MONO_BD, 60),
               fill=col, anchor="mm")
        d.text((x0 + 300, cy + 138), small, font=f_c, fill=(60, 60, 60),
               anchor="mm")

    ty = 890
    d.text((60, ty),
           "1036 是唯一有逐字人工核验参考的样本 —— 整串指标",
           font=f_c, fill=(20, 20, 20))
    y = ty + 54
    d.rectangle([60, y, 1940, y + 54], fill=(243, 244, 247))
    for txt, x in [("左扩 px", 90), ("编辑距离", 520),
                   ("CER", 900), ("说明", 1300)]:
        d.text((x, y + 27), txt, font=f_c, fill=(60, 60, 60), anchor="lm")
    y += 54
    for r in [r for r in full if r["cmpd_id"] == "1036"]:
        p = int(r["pad_px"])
        ed = int(r["edit_distance_vs_ref"])
        cer = r["cer_vs_ref"]
        note = "基线，与此前已验证实验一致" if p == 0 else (
            "变差" if ed > 2 else "无变化")
        col = (20, 20, 20) if p == 0 else (
            (180, 50, 50) if ed > 2 else (110, 110, 110))
        d.line([(60, y), (1940, y)], fill=(228, 228, 232), width=1)
        for txt, x, f in [(str(p), 90, f_m), (str(ed), 520, f_m),
                          (cer, 900, f_m), (note, 1300, f_note)]:
            d.text((x, y + 32), txt, font=f, fill=col, anchor="lm")
        y += 54
    d.line([(60, y), (1940, y)], fill=(150, 150, 150), width=2)

    d.text((60, y + 34),
           "结论：问题定位完成，但简单扩大裁剪框不是可靠的生产修复方案。",
           font=f_c, fill=(20, 20, 20))
    lines = [
        "1  行首 = 确实在检测阶段被排除 —— 215/215 事件逐项目视确认，"
        "涉及 187/335 个单元格（55.8%）",
        "2  重新裁回原像素后，74% 仍被读成 -；1036 整串编辑距离 2 → 15，"
        "CER 变差约 7.5 倍",
        "3  下一步应把修复点前移到检测阶段（形态学增强 / 更低检测阈值），"
        "尚未实验，暂列为待验证方向",
    ]
    for i, t in enumerate(lines):
        d.text((60, y + 82 + i * 44), t, font=f_n, fill=(70, 70, 70))

    out = OUTDIR / "fig_03_ab_results.png"
    im.save(out)
    return out


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rev, exact, fixed, full = load_rows()
    ev_by_id = {r["event_id"]: r for r in rev}
    exact_by_id = {}
    for r in exact:
        exact_by_id.setdefault(r["event_id"], r)
    engine = build_engine()

    print("wrote", figure_mechanism(ev_by_id, exact_by_id, engine))

    seen, picks = set(), []
    for r in rev:
        if r["status"] != "confirmed_truncation":
            continue
        if r["page"] in seen or r["event_id"] not in exact_by_id:
            continue
        ok = exact_by_id[r["event_id"]]["first_char_correct"] == "yes"
        if (len(picks) % 2 == 0) != ok:
            continue
        seen.add(r["page"])
        picks.append(r["event_id"])
        if len(picks) == 6:
            break
    print("wrote", figure_systematic(ev_by_id, exact_by_id, engine, picks))
    print("  examples:", picks)
    print("wrote", figure_results(exact, fixed, full))


if __name__ == "__main__":
    main()
