# 早上 brief（2026-10-14）

> 你起床后看这个就够，详细在 PROGRESS_REPORT.md。

## TL;DR
1. **夜间 pipeline 跑完了**：~27 min 跑完 103 张表、406 cmpd，CSV 现在 24 列全满
2. **DECIMER 2D 重绘接入**：327/327 valid SMILES 都画成 PNG（`linearize_macrocycle=True` 打开环），CSV 有 `decimer_render_path` 列
3. **OCR 熵门接通了，但救不回数据**——`canonical_smiles` 仍然 0 个，**top-k swap 路线触底**

## 负面结论（最重要的发现）
- 335 个有 OCR 的 cmpd，**全部 `is_valid=False`、`canonical_smiles` 为空**
- OCR raw 输出错误是**累积型**（同时缺 `(`、多 `)`、`C=ON` 压平、`C-O` 替 `=O`、截断）
- 针对性正则（`C=O)N` → `C(=O)N` 等）**改了 265+66+99 个 SMILES，0 个变 valid**
- **结论**：这条路再投入是空转

## 数字（实测 406 cmpd）

| 指标 | 值 |
|---|---|
| cmpds 总数 | 406 |
| 有 OCR 数据 | 335 |
| OCR canonical valid | **0** (0%) |
| DECIMER canonical valid | 327 (81%) |
| ocr_avg_entropy median | 0.244 bits/char |
| ocr_low_conf_count median | 42 / ~270 chars |
| OCR ∩ DECIMER InChI agree | 0 / 335 (0%) |
| L1 phantom | 259 (77%) |
| L1 real OCR error | 76 (23%) |

## 报告素材（已 commit）

- `demo_pipeline_render/grid_12.png` — 12 个 SMILES 表 cmpd 的"原图 vs DECIMER 重绘"，标题含 `ocr_H` 数字
- `demo_roundtrip/cmpd_1036_side_by_side.png` — 单 cmpd 1036 高清对比
- `compounds_combined.csv` — 24 列主交付物（gitignored，可重生成）

## commit 状态
```
ef190cf docs: log overnight OCR re-run + root-cause for swap-fix failure
580e8e2 fix: decimer_render in --skip-decimer branch; demo prefers SMILES-table cmpds
0e74ad0 docs: log OCR entropy + DECIMER render integration in PROGRESS_REPORT
d3a46d1 feat(render): wire DECIMER 2D re-render into pipeline
16d0e90 feat(ocr): per-char entropy trust score + DECIMER-fallback gate
22e72e5 feat: add RDKit render module, demo scripts, progress log
b82adff chore: drop superseded test scripts and AB experiments
```

## 中午汇报要讲的（建议三个 bullet）
1. **OCR 现状**：熵门 + DECIMER 兜底已就位，CSV 24 列全满，327 PNG 已画；但 OCR 自身质量卡住了，**top-k swap 触底**（正则实验 0/335）
2. **DECIMER 2D 重绘**：用 CoordGen + 宏环线性化，渲染出 ChemDraw 风格的展开骨架，**不再是"甜甜圈"**
3. **下一步选项**（要你决定）：
   - A. **chemistry-aware beam search**（每步 swap 验证 RDKit 局部 parse，最有希望突破）
   - B. **多 pass OCR 投票**（不同预处理 × 多次 OCR × 字符投票，稳但慢）
   - C. **Hiro-OCSR**（等 HF 审批通过，做跨模型对比）
   - D. **接受现状**：用 DECIMER canon 作为最终答案（327/405 = 81% 覆盖率），人工 review 剩下的 19%

## 还没动的
- Hiro-OCSR 仍在等 HF 审批（你可以去 https://huggingface.co/settings/gated-repos 看看批了没）
- 任何 references/ 下的源码修改（结论是不需要改——OCR 错误不是模型问题，是预处理/字符混淆问题）
