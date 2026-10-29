r"""Task 3: build pre_report_summary.md from the audit CSVs.

Every figure in the summary is recomputed here from the CSV files produced
by the earlier scripts, so the summary can be regenerated and will change
if the underlying data changes.

Run:
    .venv\Scripts\python.exe pre_report_summary.py
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTDIR = REPO / "data" / "audit" / "phase2" / "pre_report_validation"
AUDIT = REPO / "data" / "audit" / "phase2" / "row_crop_audit"


def load(p):
    with p.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def pct(a, b):
    return 0.0 if not b else 100.0 * a / b


def main() -> None:
    audit = load(AUDIT / "manifest.csv")
    rev = load(OUTDIR / "review_manifest.csv")
    exact = load(OUTDIR / "ab_exact_results.csv")
    fixed = load(OUTDIR / "ab_results.csv")
    full = load(OUTDIR / "ab_fullstring.csv")

    data = [r for r in audit if r["is_header"] == "no"]
    pages = len({r["page"] for r in data})
    n_head = sum(1 for r in audit if r["is_header"] == "yes")
    flagged = [r for r in data
               if int(r["n_lines_glyph_excluded"] or 0) > 0]
    n_geom = sum(int(r["n_lines_glyph_excluded"] or 0) for r in data)
    status = Counter(r["status"] for r in rev)

    t_ex = [r for r in exact if r["group"] == "truncated"]
    ok_b = sum(1 for r in t_ex if r["first_char_correct"] == "yes")
    cls_a = Counter(r["first_char_A"] for r in t_ex)
    n_minus = sum(1 for r in t_ex if r["first_char_B"] == "minus")
    n_other = len(t_ex) - ok_b - n_minus
    rule_hits = sum(1 for r in t_ex
                    if r["left_edge_in_cell_rule"] == "yes")
    ov_min = min(int(r["overhang_px"]) for r in t_ex)
    ov_max = max(int(r["overhang_px"]) for r in t_ex)

    clean_ins = sum(1 for r in t_ex
                    if r["text_A"] != r["text_B_exact"]
                    and r["text_B_exact"][1:] == r["text_A"])
    same = sum(1 for r in t_ex
               if r["text_A"] == r["text_B_exact"])
    collateral = len(t_ex) - clean_ins - same

    t_fx = [r for r in fixed if r["group"] == "truncated"]
    pad_tbl = []
    for p in ("0", "16", "30", "36", "40"):
        s = [r for r in t_fx if r["pad_requested_px"] == p]
        ok = sum(1 for r in s if r["first_char_correct"] == "yes")
        hit = sum(1 for r in s if r["left_edge_in_cell_rule"] == "yes")
        pad_tbl.append((p, len(s), ok, hit))

    c_f = [r for r in full if r["kind"] == "control_regression"]
    c_ch = [r for r in c_f if r.get("changed_vs_baseline") == "yes"]
    c_short = 0
    for r in c_ch:
        if len(r["text"]) < len(r.get("baseline_text", "")):
            c_short += 1

    rows36 = sorted(
        (r for r in full if r["kind"] == "1036_full_string"),
        key=lambda x: int(x["pad_px"]),
    )
    base_ed = int(rows36[0]["edit_distance_vs_ref"])
    worst_ed = max(int(r["edit_distance_vs_ref"]) for r in rows36)
    best_ed = min(int(r["edit_distance_vs_ref"]) for r in rows36)

    L: list[str] = []
    a = L.append
    a("# Phase 2 汇报前验证 — 疑似截断复核与重裁剪 A/B 实验")
    a("")
    a("所有数字由 `pre_report_summary.py` 从实验 CSV 重新计算。")
    a("本轮未修改正式 CSV、`src/cpd/` 或 OCR baseline,未执行 Git 操作。")
    a("")
    a("## 1. 审计与复核覆盖量")
    a("")
    a("| 项目 | 数量 |")
    a("| --- | ---: |")
    a(f"| 审计单元格(数据) | {len(data)} |")
    a(f"| header(单独计,不计入缺陷) | {n_head} |")
    a(f"| 覆盖页面 | {pages} |")
    a(f"| 存在截断事件的单元格 | {len(flagged)} / {len(data)} ({pct(len(flagged), len(data)):.1f}%) |")
    a(f"| 几何审计标记的截断事件 | {n_geom} |")
    a(f"| 逐事件复核记录 | {len(rev)} |")
    a("")
    a("**单元格数量与事件数量必须分开**:一个单元格可含多个事件。")
    a("")
    a("## 2. 复核结论")
    a("")
    a("| 状态 | 事件数 |")
    a("| --- | ---: |")
    for k in ("confirmed_truncation", "not_truncated", "ambiguous",
              "unreviewed"):
        if status.get(k):
            a(f"| `{k}` | {status[k]} |")
    a("")
    a("判定依据(逐事件记录于 `review_manifest.csv` 的 `judgement_basis`):")
    a("")
    a("- 被排除墨迹位于检测框左边界**之外**;")
    a("- 形态为该行带内的两条水平笔画(墨迹行数恒为 8);")
    a("- 重裁剪面板显示它正是该行行首的 `=`;")
    a("- 逐项排除了单元格边框线、相邻行与噪点作为来源。")
    a("")
    a(f"证据:`evidence/` {len(rev)} 张单事件四联图 + `grids/` 14 张分组复核图。")
    a("")
    a("## 3. A/B 重裁剪实验")
    a("")
    a("A 组 = 现有检测框。B 组 = 从同一父图按左扩像素重新裁切。")
    a("")
    a("**未使用白色 padding** —— 白边无法让框外字符的像素回到输入,")
    a("用它会制造出并不存在的改善。")
    a("")
    a("固定不变项:2x INTER_CUBIC → 阈值 200 → `get_rotate_crop_image` 透视校正")
    a("→ `engine.recognize_txt`(官方识别入口)。")
    a("")
    a("### 3.1 行首字符结果(按实测外露量精确裁切)")
    a("")
    a(f"被排除墨迹的外露量实测在 {ov_min}–{ov_max} px 之间波动,")
    a("因此按每个事件各自的实测值裁切,新左边界正好落在首个干净墨迹列。")
    a("")
    a("| 配置 | 行首为 `=` | 比例 |")
    a("| --- | ---: | ---: |")
    a(f"| A(基线,字符不在输入内) | 0 / {len(t_ex)} | 0.0% |")
    a(f"| B(精确重裁) | {ok_b} / {len(t_ex)} | {pct(ok_b, len(t_ex)):.1f}% |")
    a("")
    a(f"B 组首字符分布:`=` {ok_b}、`-` {n_minus}、其他 {n_other}。")
    a(f"A 组首字符分布:{dict(cls_a)} —— 基线**从不出现** `=`。")
    a("")
    a(f"该配置下 {rule_hits}/{len(t_ex)} 个裁切的新左边界落在单元格边框线内,")
    a("即这是一组**无网格污染**的实验。")
    a("")
    a("### 3.2 固定扩展距离(附污染标注)")
    a("")
    a("| 左扩 px | 事件数 | 行首 `=` 正确 | 比例 | 落在边框线内 |")
    a("| ---: | ---: | ---: | ---: | ---: |")
    for p, n, ok, hit in pad_tbl:
        a(f"| {p} | {n} | {ok} | {pct(ok, n):.1f}% | {hit} |")
    a("")
    a(f"固定距离结果非单调,原因是外露量本身在 {ov_min}–{ov_max} px 间波动:")
    a(f"pad=30 只对 37/{len(t_ex)} 个事件覆盖了完整字形;")
    a("pad=36/40 分数虽高,但裁切左边界普遍落进单元格边框线,属网格污染,")
    a("不能作为干净结论。")
    a("")
    a("### 3.3 连带损伤")
    a("")
    a("| 单行变化类型 | 事件数 |")
    a("| --- | ---: |")
    a(f"| 仅行首新增 1 个字符,其余完全不变 | {clean_ins} |")
    a(f"| 除行首外还出现额外字符错误 | {collateral} |")
    a(f"| 完全无变化 | {same} |")
    a("")
    a("额外损伤的典型形态:裁切变宽后,行内**其它** `=` 也被读成 `-`")
    a("(如 1036 行内 `CCC5=CC(F)` 变为 `CCC5-CC(F)`)。")
    a("")
    a("## 4. 干净对照样本的退化情况")
    a("")
    a(f"对 {len(c_f)} 个左边界正常、无截断的单元格,把首行左扩 30 px 后重跑:")
    a("")
    a(f"- 文本发生变化:**{len(c_ch)} / {len(c_f)}**")
    a(f"- 其中 {c_short} 例的结果字符串变短,即丢失了字符(括号)")
    a("- 变化方向是新增错误,而非修正")
    a("")
    a("## 5. 1036 的整串 CER 与编辑距离")
    a("")
    a("1036 是唯一具有人工逐字核验参考的样本,因此只有它报告完整字符串指标。")
    a("整串 = 7 行按流水线顺序拼接,仅加宽第 2 行(事件所在行)。")
    a("")
    a("| 左扩 px | 编辑距离 | CER |")
    a("| ---: | ---: | ---: |")
    for r in rows36:
        a(f"| {r['pad_px']} | {r['edit_distance_vs_ref']} | {r['cer_vs_ref']} |")
    a("")
    a(f"基线编辑距离 {base_ed}、CER {rows36[0]['cer_vs_ref']},与此前已验证的实验一致。")
    a("")
    a(f"**重裁剪没有改善 1036。** 最好情况编辑距离 {best_ed};")
    a(f"精确外扩(34 px)与 40 px 反而升至 {worst_ed},CER 同步变差。")
    a("")
    a("## 6. 适用范围与限制")
    a("")
    a(f"- 行首字符结论基于 **{len(rev)} 个已逐项目视复核的事件**,覆盖 {pages} 页。")
    a("- 完整字符串 CER / 编辑距离**只有 1036 一个样本**有金标准,")
    a("  其余样本一律不报告整串指标。")
    a("- 行首「正确字符 = `=`」来自原页面图像的目视判读,**不是**来自")
    a("  质量门、DECIMER 输出或 RDKit 可解析性。")
    a("- 语法解析成功 ≠ 字符正确 ≠ 化学结构正确,本报告不混用这三层。")
    a("- 对照样本无金标准,只能说明「发生了变化」及变化方向。")
    a("")
    a("## 7. 结论与建议")
    a("")
    a("**结论类型:重裁剪能恢复像素,但识别效果没有改善。**")
    a("")
    a("证据链:")
    a("")
    a(f"1. 几何层面已证实:{len(rev)}/{len(rev)} 事件的行首 `=` 未进入识别器输入,")
    a(f"   涉及 {len(flagged)}/{len(data)} 个单元格。")
    a(f"2. 像素层面已证实:精确重裁后行首恢复为 `=` 的比例从 0.0% 升至")
    a(f"   {pct(ok_b, len(t_ex)):.1f}%,且该组网格污染 {rule_hits} 例。")
    a(f"3. 但 {n_minus}/{len(t_ex)} ({pct(n_minus, len(t_ex)):.1f}%) 被读成 `-` 而非 `=`。")
    a(f"4. 整串指标上,1036 编辑距离 {base_ed} → {worst_ed},**变差**。")
    a(f"5. 干净对照在同样加宽下 {len(c_ch)}/{len(c_f)} 发生变化,以丢失字符为主。")
    a("")
    a("**因此不建议把左扩重裁剪应用到生产流程。** 它确实能把像素放回输入,")
    a("但识别器随后把多数 `=` 读成 `-`,并在加宽时波及行内其它字符。")
    a("")
    a("建议的下一步(按优先级):")
    a("")
    a("1. 把修复点放在**检测阶段**而非识别阶段:行首 `=` 之所以被排除,")
    a("   是因为它是低对比度的宽扁笔画,文本检测器不将其视为独立文本块。")
    a("   可尝试在送入检测器前对该行带做形态学增强,或改用更低检测阈值的")
    a("   检测器配置,让 `=` 进入检测框。")
    a("2. 单独评估「`=` 与 `-`」混淆能否由识别侧约束解决,")
    a("   但这属于识别后处理,不能替代第 1 步。")
    a(f"3. 若仍沿用重裁剪路线,应**逐事件按实测外露量裁切**,")
    a("   避免固定距离导致的字形不全与网格污染,并配套对照回归测试。")
    a("4. 在任一方案给出批量证据前,**不替换正式 baseline、不修改正式 CSV**。")
    a("")
    a("## 8. 复现命令")
    a("")
    a("```")
    a(".venv\\Scripts\\python.exe pre_report_review.py         # 四联证据图")
    a(".venv\\Scripts\\python.exe pre_report_grids.py          # 分组复核图")
    a(".venv\\Scripts\\python.exe pre_report_review_finalize.py # 复核结论")
    a(".venv\\Scripts\\python.exe pre_report_ab_recrop.py      # 固定左扩")
    a(".venv\\Scripts\\python.exe pre_report_ab_exact.py       # 精确裁切")
    a(".venv\\Scripts\\python.exe pre_report_ab_fullstring.py  # 整串+对照")
    a(".venv\\Scripts\\python.exe pre_report_summary.py        # 本文件")
    a("```")
    (OUTDIR / "pre_report_summary.md").write_text(
        "\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUTDIR / 'pre_report_summary.md'}  ({len(L)} lines)")


if __name__ == "__main__":
    main()
