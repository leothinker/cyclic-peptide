# 进展报告 (2026-10-12 凌晨版)

> **历史存档（2026-10-12）。** 本文保留当时的原始记录，数字反映的是
> 那个时间点的状态。其中 "smiles_best 349/405 (86%)" 的口径后来被审计
> 推翻：按化学与质谱双重门控重新统计，**119 条修复候选全部未通过
> 验证**，可确认为完整的宏环比例是 178/406 (43.8%)。
> CSV 也已从 33 列扩展到 48 列。
>
> 最新进展见 `PROGRESS.md` 顶部章节与 `README.md`。

> 上一版 [2026-10-11 17:55](#) - 本版覆盖 23:00 至现在的进展。
> 用户 23:00 左右睡觉，预计 06:00-07:00 起床。

## 当前状态（2026-10-12 01:55，pipeline 已完成）

| 指标 | 值 |
|---|---|
| DECIMER cache | **405 / 405 (100%)** ✓ |
| OCR 有效 canonical | 140 / 335 (42%) |
| DECIMER 有效 canonical | 327 / 405 (81%) |
| smiles_best 总数 | 349 / 405 (86%) |
| smiles_best 来源 | ocr: 140 / decimer: 209 / empty: 56 |
| 截断 SMILES 修复 | 25 个通过 `_post_close_brackets.py` 救回 |
| Phantom cleanup | 119 / 405 (29%) 被清理 |
| OCR vs DECIMER 严格一致 | **0**（InChI 完全不同！下面解释） |

## 关键发现（这一夜最重要的）

### 1. OCR 和 DECIMER 给出的 SMILES 是不同分子

```
cmpd 1037 (页 I100297):
  OCR:     C76H96F9N13O12S2, 112 atoms
  DECIMER: C67H93F5N12O12,   96 atoms
  InChIKey 完全不一样
```

结论：**OCR 和 DECIMER 都对、或者都错，但从来没同时对一个 cmpd 给出同一个分子。**

至少一个方法在大量 cmpd 上系统性出错。从打印 cell 图看 cmpd 1036/1037 的真实打印 SMILES，`C76H96F9N13O12S2` 的元素组成看起来对得多（9 个 F、2 个 S），DECIMER 的 `C67H93F5N12O12` 似乎丢了几个 F 和 S。所以**至少在某些 cmpd 上，OCR 是对的，DECIMER 是错的**。

### 2. OCR 错在「字符丢失」而非「字符认错」

直接对比 cmpd 1036 的 cell crop + 真实打印 SMILES（页 I100297）：

| | 内容 |
|---|---|
| **打印版** | `CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C2CCCCC2)C(=O)N3[C@@H](CC3)C(=O)N(C)...` |
| **OCR 输出** | `CC[C@H]C)[C@H]1C(=O)N(C)[C@@H](C2CCCCC2)C=ON3[C@@H]CC3C=O)NC)...` |
| **差异** | `(C)` → `C)`，`C(=O)` → `C=O`（开括号丢失） |

单字符 top-k 替代（`_fix_smiles_with_top_k`）救不回这种丢失型错误，因为 top-k 候选里压根没有 `(`。

### 3. 「补左括号」启发式救不回 OCR 错误

`_insert_open_brackets.py` 实测：195 个失败 OCR SMILES 里只恢复 1 个（0.5%）。
原因：插入 `(` 后 RDKit 解析成不同的分子，不是同一化合物。
**结论：OCR 准确度改进必须从模型侧入手，不是后处理。**

### 4. Phantom cleanup + bracket close 是 DECIMER 端的真正修复杠杆

- `largest_fragment_canonical()`：清掉 DECIMER 末尾的 `CC.CC` phantom（119 entries，29%）
- `close_unbalanced_brackets()`：补齐截断 SMILES 缺失的 `)` `]`（25 entries）
- 两段都已在 `_run_decimer` 里自动应用

### 5. `smiles_best` 来源已加 `ocr+decimer` 标记

新版三态：`ocr` / `decimer` / `ocr+decimer`（最可靠）。但目前 `ocr+decimer` 数量 = 0（因为两者 InChI 没一个 match）。

### 6. 表格中「已有打印 SMILES」是否需要单独切图？

**答：不需要，已经做了。** `_extract_rows` → `_ocr_cell` 已经按 row × column 切图后 OCR。问题在识别（recognizer），不在切图（detector）。
换 detector 或换模型或许能改善（hybrid ch-det + en-rec 实验有改善，但需要重跑 30min OCR）。

## 已落实的代码改进

- ✅ DECIMER cache schema v2 + 每 10 cell 写盘（避免 I/O 风暴）
- ✅ Phantom cleanup (`largest_fragment_canonical`)
- ✅ Bracket close (`close_unbalanced_brackets`，已 inline 到 `_run_decimer`)
- ✅ Top-k OCR (`rec_top_k=5`，3 个 cfg 位置都设好)
- ✅ Top-k 单字符 swap (`_fix_smiles_with_top_k`)
- ✅ DECIMER per-token 置信度（`min_conf`, `low_conf_count`）
- ✅ `smiles_best` 三态：ocr / decimer / ocr+decimer
- ✅ `--report-l1` 子命令（直接读 CSV 跑 L1）
- ✅ `--skip-ocr` 子命令（重跑 DECIMER 时跳 OCR）

## 用户回家后要跑的命令

```powershell
# 1. pipeline 状态（应该已经完成）
.venv\Scripts\python.exe _quick_status.py

# 2. 综合 status（OCR + DECIMER 覆盖 + L1 + phantom 分析）
$env:PYTHONIOENCODING = "utf-8"
.venv\Scripts\python.exe -u _status_report.py

# 3. L1 phantom-fragment breakdown
.venv\Scripts\python.exe -u main.py --report-l1

# 4. （可选）重建 combined CSV：除非你想重新跑 OCR，否则不需要
.venv\Scripts\python.exe -u main.py --skip-ocr --rec-top-k 5
```

## 明天可做的事（按优先级）

1. **OCR 改进**（杠杆最大）：
   - **(a) Hybrid 检测器**：用 `ch_PP-OCRv6` 检测 + `en_PP-OCRv4` 识别（之前测试版本准确度明显更高）
   - **(b) OCR 模型侧微调**：在已成功的 ~140 个 cmpd 上训练 SMILES-specific post-corrector
   - **(c) Cell crop padding**：把 `_trim_to_content` 的 pad 从 4 加到 32px
2. **DECIMER 准确度验证**：在打印 SMILES 页（I100297+）上对比 OCR vs DECIMER 的 InChI，确认哪种方法更准
3. **Hiro-OCSR 装机**：8B VLM，需要 GPU；今天没时间装
4. **报告材料**：用 `--report-l1` 输出 markdown 给周一（10/13）的报告

## 已知问题 / 待用户决策

- **git 锁**：`.git/index.lock` 创建被拒（沙箱 + OneDrive 同步冲突）。所有改动已 working tree，未 commit。
- **DECIMER 慢**：每 cell 30-80 秒，405 cell 跑一次要 5-8 小时。**下次长任务 22:00 之后再开**。
- **0 agreement**：OCR 和 DECIMER 在 cmpd 层面没一个 InChI 一致。需要决定：(a) 哪种方法更可信；(b) 是否需要人工抽检。
- **临时脚本**：`_*.py` 共 34 个，探索用。保留：`main.py`, `test_*.py`, `references/`, `data/processed/`.

## 数字速览（最终）

```
Coverage:
  OCR with SMILES:        335 / 405 (83%)
  DECIMER with SMILES:    405 / 405 (100%)
  Both have SMILES:       335
  smiles_best populated:  349 / 405 (86%)

RDKit validity:
  OCR canonical valid:    140 / 335 (42%)
  DECIMER canonical valid: 327 / 405 (81%)

L1 breakdown (335 cmpds with both):
  Agree (InChI identical):  0  (0%)     ← 重大发现
  DECIMER low confidence:   259 (77%)
  Real OCR error:            76 (23%)
  Median DECIMER min_conf:   0.420

Phantom cleanup:
  DECIMER entries with '.': 119 / 405 (29%) → all cleaned

Bracket close recovery: 25 entries

Source of best SMILES:
  decimer (fallback for failed OCR): 209
  ocr (OCR canonical valid):         140
  ocr+decimer (both agree):           0
  empty (neither valid):              56
```

---

## 更新（2026-10-14 早上）

### OCR 熵信任度 + DECIMER 兜底（commit 16d0e90）

- 新增 `OcrStats` dataclass：per-char 熵 + min_top1 + low_conf_count（top-1 < 0.7 计为低置信）
- 新增 `_ocr_smiles_with_stats()` 在 `rec_top_k>1` 时同时计算熵
- `CompoundRecord` +3 字段；`COMBINED_COLUMNS` +3 列
- `smiles_best_source` 新增 `decimer_fallback`（OCR 有效但不可信 + DECIMER 有效）和 `ocr_only_invalid` 两个标签
- 阈值常量：`OCR_TRUST_ENTROPY_THRESHOLD=0.5` bits/char，`OCR_TRUST_LOW_CONF_LIMIT=5`

烟测 6 张 SMILES 表（23 cmpd）：
| 指标 | 值 |
|---|---|
| ocr_avg_entropy | 0.23–0.25 bits/char |
| ocr_min_top1 | 0.26–0.50 |
| ocr_low_conf_count | 35–50 / ~270 chars |
| source 分布 | decimer: 22, empty: 1 |

确认：top-k swap 还不够，所有 cmpd OCR canonical 都为空→全部走 decimer fallback。

### DECIMER 2D 重绘接入 pipeline（commit d3a46d1）

- `_render_decimer()` 用 `cpd.render.render_mol_pil(linearize_macrocycle=True)` 把 327/327 valid DECIMER canonical SMILES 画成 PNG
- 写到 `data/processed/rendered/<cmpd_id>.png`，CSV 新增 `decimer_render_path` 列
- `[3b/4]` phase 嵌入 `run_pipeline`，~6s 完成
- 报告素材：`_demo_pipeline_render.py` + `demo_pipeline_render/grid_12.png`

### 待办
- [ ] 夜间跑 `main.py --skip-decimer --rec-top-k 5`（1-2h）刷新 `compounds_combined.csv` 24 列
- [ ] 检查 ocr_avg_entropy × is_valid 关系 → 调阈值
- [ ] Hiro-OCSR：仍在等 HF 审批

### 夜间跑（2026-10-14 01:00-02:30）

- 跑了 `main.py --skip-decimer --rec-top-k 5`，**~27 min**（103 张图，406 cmpd）
- 第一次跑因 `--skip-decimer` 分支没初始化 `decimer_render` 写 CSV 时崩了 → 已修（`580e8e2`）
- 用 cache 合并回 DECIMER 列 + 重渲染 327/327 PNG + 刷 `smiles_best_source` 列

**关键发现（决定下一步方向）**：
1. OCR top-k swap **救不回来** cyclic peptide SMILES
2. 原因：raw OCR 输出有累积性错位（不只是字符级）
   - `C=O)N` → 应该是 `C(=O)N`（缺左括号，`auto_repair` 不管）
   - `C=ON3` → 应该是 `C(=O)N3`（括号被压平）
   - `C-O` → OCR 读不出 `=O` 时替换成 `-`
   - 噪声段 `CCCF)F)F`
3. **结果**：335 个有 OCR 的 cmpd，`canonical_smiles` **全部为空**（`auto_repair_smiles` 救不回累积性错误）
4. 熵门接通了但没机会触发（OCR 自己就废了），`smiles_best_source` 仍全部走 `decimer`
5. **结论**：top-k swap 路线对当前 OCR 质量已触底，要突破必须换方法（见下面"下一步"）

**OCR 熵分布**（335 cmpd）：
| 指标 | min | median | max |
|---|---|---|---|
| ocr_avg_entropy (bits/char) | 0.191 | 0.244 | 0.281 |
| ocr_min_top1 | 0.26 | 0.42 | 0.50 |
| ocr_low_conf_count | 25 | 42 | 63 |

**L1 breakdown（无变化）**：
- 0/335 agree；77% DECIMER phantom；23% real OCR error；median min_conf 0.420

### 下一步（明天要决定的）

1. **保留**：OCR 熵列 + DECIMER 兜底逻辑（数据基础已就位，等 OCR 质量上来后能立即生效）
2. **可选实验**：在 `auto_repair_smiles` 加针对性正则修常见 collapse pattern（`C=O)N` → `C(=O)N` 等）
3. **突破方向**：
   - **chemistry-constrained beam search**：每步 swap 后用 RDKit 局部 parse 验证 prefix 合法，否则剪枝
   - **多 pass OCR + 投票**：高/低对比度、放大倍率不同的几张图分别 OCR，按字符投票
   - **Hiro-OCSR**（仍在等 HF 审批）：第二个独立 OCSR，跨模型对比


---

## 更新记录 (2026-10-16)

### 增量改动

- **新增 ttempt_remove_extra_digits 策略**（src/cpd/macrocycle_repair.py）：
  复用主管"相同数字 = 二者相连"的诊断，当 DECIMER 的环闭数字出现 3+ 次时
  直接删除多余出现（比 re-code 多生成一个环的旧策略更精准）。新策略 + 旧策略
  合并后，本批 6 个之前救不回的 cmpd 也被修复，repair 计数从 121 升到 127。
- **修复 source-image 1-图像漂移**：cmpd 1036-1079 实际在 I100376（与 cmpd 2173
  结构图同页），1080-1137 在 I100377。把 CSV 里错位的 source_image /
  ssay_source_image 字段改回正确页。数据值不受影响。
- **数据校验**：对照 I100376-I100382 七张表逐行检查 KD/RT/MS/method/polarity，
  全部与原表匹配。共 406 cmpd，KD/RT/MS/method/polarity 全部 100% 填充。
- **删掉 debug 临时脚本**：所有 _xxx.py 调试脚本（per 用户偏好）。

### 当前完整数字

| 指标 | 数量 |
|---|---|
| cmpd 总数 | 406 |
| 已 repair (DECIMER valid + 修好了) | **127** |
| DECIMER valid + 仍 broken | 97 |
| DECIMER invalid | 182 |

剩余 97 仍 broken 的，主要是 DECIMER 输出被截断（C( 结尾、缺闭括号），
不是主管诊断的"环闭数字重用"问题。要继续救这些 case，需要：
1. 重新跑 OCSR（DECIMER 改 seed 多 pass 投票，或换 Hiro-OCSR）
2. 直接 OCR 专利图右栏的印刷 SMILES 文本（见 I100376 那种结构图右栏）

