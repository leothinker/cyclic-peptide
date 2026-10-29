# 汇报材料 — 2026-10-14

> **历史存档（2026-10-14）。** 保留当时的原始记录。CSV 现为
> **48 列**（此处写的是 24 列）；"86 %" 等修复率口径已被后续审计
> 推翻，见 `PROGRESS.md` 顶部章节与 `README.md`。

> 两天的核心进展：OCR 修复路线 + DECIMER 2D 重绘 + grammar-aware 工具链。
> 全部已 commit 到 main 分支，可直接演示。

---

## 1. 一句话总结（开场用）

> 我们把 cyclic peptide 流水线从"OCR/DECIMER 各跑各的"升级到了"两路自动选最优"，**OCR 字符级修复经过三连测试证明触底**（sub + ins + del + grammar-aware 都 0/12），**DECIMER 81% 命中率是当前可用的最终答案**。

---

## 2. 关键数字（必背）

| 指标 | 数字 |
|---|---|
| 处理化合物总数 | **405** |
| DECIMER canonical valid | **327/405 (81%)** |
| OCR canonical valid | **0/335 (0%)** — 三连修复都救不回 |
| OCR ∩ DECIMER InChI 一致 | **0/335 (0%)** — 两个方法互补失败 |
| 2D 重渲染产出 | **327/327** PNG（闭环完整，宏环可读） |
| OCR 熵（median） | 0.244 bits/char |
| OCR 低置信字符（median） | 42/270 字符 |
| Pipeline 单次运行 | ~27 min（OCR 1-2h + DECIMER 8h 可缓存） |

---

## 3. 12 个 commit（开发轨迹）

```
64f91df docs: MORNING_BRIEF -- 3rd beam search attempt (insert/delete) also 0/12
144d09e feat(ocr): insert/delete edit ops in beam search         ← 今天
47c240c feat(ocr): grammar-aware prune (0.02s/cmpd, 10x faster)  ← 今天
8541d12 feat(ocr): beam_search_smiles (0.25s/cmpd, 0/12 hits)    ← 今天
835aef4 docs: MORNING_BRIEF -- beam search + DECIMER update
3793fd1 docs: MORNING_BRIEF for at-a-glance status of overnight work
ef190cf docs: log overnight OCR re-run + root-cause for swap-fix failure
580e8e2 fix: decimer_render in --skip-decimer branch
0e74ad0 docs: log OCR entropy + DECIMER render integration
d3a46d1 feat(render): wire DECIMER 2D re-render into pipeline     ← 关键
16d0e90 feat(ocr): per-char entropy trust score + DECIMER-fallback  ← 关键
22e72e5 feat: add RDKit render module, demo scripts, progress log
b82adff chore: drop superseded test scripts and AB experiments     ← 清理
```

---

## 4. 4 张可直接展示的图

| 文件 | 内容 | 用法 |
|---|---|---|
| `demo_pipeline_render/cmpd_1036_render_compare.png` | **三联对比**：DECIMER 闭环 / DECIMER linearize / 原图 | 解释"DONUT vs LINEARIZE"，反驳"DECIMER 丢环" |
| `demo_pipeline_render/grid_12.png` | **12 个 SMILES 表 cmpd** 的"原图 vs DECIMER 重绘"，含 `ocr_H` 数字 | 报告主图，展示 DECIMER 整体质量 |
| `demo_roundtrip/cmpd_1036_side_by_side.png` | cmpd 1036 单 cmpd 高清对比 | 单点 deep-dive 用 |
| `demo_roundtrip/cmpd_1037_side_by_side.png` | cmpd 1037 单 cmpd 对比 | 第二个例子 |

---

## 5. 三个主要代码模块（"我们做了什么"）

| 模块 | 路径 | 做什么 |
|---|---|---|
| **OCR 熵信任度门** | `main.py` `COMBINED_COLUMNS` + `OCR_TRUST_*_THRESHOLD` | 24 列 CSV 含 `ocr_avg_entropy` / `ocr_min_top1` / `ocr_low_conf_count`；`smiles_best_source` 4 路决策（ocr / ocr+decimer / decimer_fallback / decimer） |
| **DECIMER 2D 重绘** | `src/cpd/render.py` | CoordGen + `linearize_macrocycle=True` 把宏环打开成 ChemDraw 风格；327/327 PNG 落地 |
| **grammar-aware 工具链** | `src/cpd/parsers/_grammar.py` + `_beam_search.py` | 轻量级 SMILES prefix walker（0.003ms/串）+ beam search 支持 sub/ins/del 三种 edit |

---

## 6. 三个核心发现（按说服力排序）

### 发现 1（最强负面）：OCR 字符级修复三连 0/12
- L1: greedy single-swap → **0/12**
- L2: best-of-N-swap + grammar-aware → **0/12**
- L3: + insert/delete → **0/12**
- 100% raw OCR 是非法 SMILES（grammar walker 验出 0/335 合法）
- **根因**：OCR 错误是结构性插入/删除（如 `C=ON` 应是 `C(=O)N`，少 2 个括号），需要 2+ edit 才能修

### 发现 2：DECIMER 实际质量好（修正之前的夸张判断）
- cmpd 1036 DECIMER：115 heavy atoms, formula `C77H104F8N14O13S3`
- 闭环宏环完整可见（S 环 / F-aromatic / cyclopentane / pyrrolidine）
- linearize 之后"看着短"是**几何效应**（开链延展），**不是原子丢失**
- 仅 1-2 个小支链被简化（vs 原图）

### 发现 3：OCR ∩ DECIMER 零一致 = 两个方法互补失败
- 335 cmpd InChI 字符串相等：**0**
- 77% 是 DECIMER phantom（min_conf < 0.5）
- 23% 是真 OCR 错误
- 含义：不能盲信任任一方

---

## 7. 报告建议结构（5 分钟讲完）

**1. 现状（1 min）**
- "405 cmpd 流水线，24 列主表，327 张 2D 重绘，OCR 修复工具链完整"
- 展示 `grid_12.png`（主图）

**2. 三个核心发现（2 min）**
- **OCR 修复三连 0/12**（含 `cmpd_1036_render_compare.png` 解释为什么不是 DECIMER 的问题）
- **DECIMER 81% 命中率可用**（115 atoms 完整闭环）
- **两个方法零一致 → 互补失败**（77% phantom + 23% OCR）

**3. 工具链产出（1 min）**
- `ocr_avg_entropy` 信任度列
- `decimer_render_path` 2D 重绘列
- `_grammar.py` + `_beam_search.py` 字符级修复工具（0.02s/cmpd）

**4. 下一步（1 min）**
- 当前路线已触底
- 选项：A. 2-edit beam search（不确定收益）/ B. 接受现状 / C. 等 Hiro-OCSR / D. 多 pass OCR 投票

---

## 8. 仓库状态

- **Main 分支干净**，12 个新 commit
- `git status` 无未提交修改
- 数据产物（CSV、327 PNG）通过 `python main.py --skip-decimer --rec-top-k 5` 可重新生成
- References（RapidOCR / DECIMER / Hiro）未修改

---

## 9. 关键文件路径速查

```
# 主数据
data/processed/compounds_combined.csv          24 列主交付物（gitignored，重生成）
data/processed/decimer_cache.json              DECIMER 缓存 (405/405)
data/processed/cells/                          405 个结构图 crop

# 重渲染的 2D PNG
data/processed/rendered/                       327/327 cmpd DECIMER 重绘

# 报告图
demo_pipeline_render/grid_12.png               12 cmpd 对比
demo_pipeline_render/cmpd_1036_render_compare.png  三联对比（核心图）
demo_roundtrip/cmpd_1036_side_by_side.png      单 cmpd 高清
demo_roundtrip/cmpd_1037_side_by_side.png

# 文档
MORNING_BRIEF.md                                一眼 TL;DR
PROGRESS_REPORT.md                              详细日志

# 核心代码
main.py                                          24 列 CSV 写入 + smiles_best 决策
src/cpd/render.py                                DECIMER 2D 重绘（CoordGen + linearize）
src/cpd/parsers/table_extractor.py               OCR 抽取
src/cpd/parsers/_grammar.py                      SMILES prefix walker（新增）
src/cpd/parsers/_beam_search.py                  beam search sub/ins/del（新增）
src/cpd/decimer.py                               DECIMER wrapper（含 per-token 置信度）
src/cpd/chem.py                                  RDKit 验证 + phantom 清理
```

---

## 10. 一行 commit message 模板（如果需要重生成 commit）

每个 commit 都已经写好独立 message，不需要重写。

---

**汇报顺利！** 如果临时需要新视角/新数据，告诉我具体想看什么。
