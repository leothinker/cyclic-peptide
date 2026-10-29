# 外部验证 — 第二份专利与 2173 泛化测试

日期：2026-10-29
范围：只读审计 + 独立实验脚本。未修改 `src/cpd/`、未修改正式 CSV、未执行
正式流水线写回。

---

## 0. 一句话结论

**`validated` OCR 路线在一个"管线从未成功读出过的化合物"上达到
编辑距离 1（唯一错误是字母 `O` 被读成数字 `0`），而生产路线是 35、
已发布 CSV 是 33。** 这是本阶段最强的泛化证据：2173 所在的页面
I100376 从未参与过配置搜索，且该化合物在正式 CSV 里此前没有可用结构。

同时，**第二份专利文档在本环境无法下载**（Google Patents 503、EPO 500、
Espacenet 403），这一阻塞已完整记录，见第 3 节。

---

## 1. 2173 —— 管线从未读出的化合物

### 1.1 为什么这是好的泛化测试

| 维度 | 说明 |
|---|---|
| 页面 | I100376（= PDF 第 240 页），**未参与**任何配置选择 |
| 化合物 | 2173，正式 CSV 里 `smiles_best` 为空、`smiles` 是坏字符串 |
| 印刷行数 | 8 行，跨页右栏，最长行贴到表格右竖线 |
| 参考来源 | 本轮从黄金表逐行 3× 放大重读（8 张证据图见 `evidence_2173/`） |
| 参考状态 | `printed_transcription_awaiting_human_check`（**文本参考，非结构验证**） |

### 1.2 结果（字符级，对比本轮印刷转录）

| 来源 | 长度 | 编辑距离 | 括号 | RDKit |
|---|---:|---:|---|---|
| **validated 路线** | 282 | **1** | 38/38 | 否（见下） |
| production 路线 | 250 | 35 | 14/32 | 否 |
| 已发布 CSV `smiles` | 229 | 33 | 16/23 | 否 |
| DECIMER（结构图识别） | 237 | 18 | 34/34 | 否 |

唯一的一处错误是一个字符替换：印刷文本里的 `C(=O)` 被读成了
`C(=0)`——字母 `O` 与数字 `0` 混淆。原图已放大 3 倍确认为字母 `O`
（见 `evidence_2173/line07.png`），所以这是识别错误而非参考错误。
其余 281 个字符逐字一致。

### 1.3 已发布 CSV 里 2173 是什么样

正式 CSV 的 `smiles` 列开头就少了 `FC`，每个 `C(=O)` 变成 `C=O)`——
左括号全部丢失，这是生产识别器（英文 PP-OCRv4）的已知系统性缺陷。

### 1.4 这一条不能证明什么

* 编辑距离 1 是**文本层面**的结果，不代表化学结构正确。
* 2173 的 `smiles_best` 仍为空；本轮**没有**改动正式 CSV。
* 参考是本轮转录，未经人工逐字确认，状态与 1039-1041 相同。

---

## 2. 主管提出的问题："为什么放大之后识别反而不如原图？"

这个问题有明确的技术答案，**不是玄学**。

### 2.1 机制

RapidOCR 的识别器输入尺寸固定为 `rec_img_shape: [3, 48, 320]`
（`.venv/Lib/site-packages/rapidocr/config.yaml`），而
`ch_ppocr_rec/main.py::resize_norm_img` 实际做的是：

```python
img_height = 48
resized_w  = ceil(img_height * w / h)
cv2.resize(img, (resized_w, img_height))
```

**识别器永远把文本条归一化到 48 px 高**，不管你交给它多高。

* 原始印刷行约 30 px 高 -> 识别器自己**放大** 30->48，一次插值。
* 我们先 2 倍放大到 60 px -> 识别器再**缩小** 60->48，两次插值。

第二次插值发生在识别器自己的采样网格上，而 `=` 的两条短横、`@` 的
圆圈这类细笔画正是在这个网格上被合并或丢失的。所以"放大"只有在
它改变了识别器**实际看到的**东西时才有帮助；如果识别器本来就要放大，
多加这一步只会引入走样。

### 2.2 实测（`experiments/phase3_upscale_question.py`）

对 `validated` 路线的墨迹带裁剪，只改放大倍数，其余全部固定：

| cmpd | scale=1 | scale=2 | scale=3 |
|---|---|---|---|
| 1036 | edit 0 | edit 0 | edit 2 |
| 1037 | edit 3 | edit 3 | edit 6 |
| 1038 | 括号 40/40 | 括号 40/40 | 括号 40/40 |
| 2173 | 括号 38/**34** | 括号 38/**38** | 括号 38/38 |

结论：

* scale=1 与 scale=2 在 1036/1037 上**没有差别**——放大在这里既没有
  帮上忙也没有坏事，符合上面的机制解释。
* 2173 在 scale=1 时括号失衡（38/34），scale=2 才配平——因为 2173 的
  墨迹带更矮（54-62 px，含 padding），放大给了识别器更接近
  48 px 的输入。**放大在"行更矮"的样本上确实有用**。
* scale=3 一律变差：离 48 px 越远，第二次插值的信息损失越大。

所以正确的说法不是"放大没用"，而是：

> **放大不是分辨率操作，而是"要不要让识别器做缩放"的选择。**
> 目标高度是 48 px：把条带到接近 48 px 再交出去最好；
> 远离 48 px 都会因额外插值而损失细节。

---

## 3. 第二份专利：已尝试，被网络策略阻塞

`experiments/phase3_fetch_second_patent.py` 与
`experiments/phase3_external_patent_search.py` 记录了全部尝试：

| 源 | 结果 |
|---|---|
| Google Patents（HTML） | **503 Service Unavailable** |
| EPO publication server | **500 Internal Server Error** |
| Espacenet | **403 Forbidden** |
| `patentimages.storage.googleapis.com` | **403 Forbidden** |
| USPTO `image-ppubs` | **404 Not Found** |
| WIPO Patentscope | 页面可访问但需 JS，会话未返回 PDF |
| PatentsView API | 可访问，但**只提供元数据，不提供 PDF** |
| PubChem PUG REST | **可用**（见 3.2） |

候选的第二份专利是 WO2025/162428 自己的国际检索报告里引用的同族申请
（WO2022/234852、WO2022/234853、WO2024/101402 等）——同一申请人、
同类型表格，是最合适的泛化对象。

### 3.1 本 PDF 内部已确认没有额外样本

源 PDF 共 270 页、无文本层。逐页渲染 + 表格特征扫描
（`experiments/phase3_unseen_pages.py`）结果：

* 含印刷 SMILES 的表页**就是**黄金表那 103 张；
* PDF 第 240 页经比对确认 = 黄金表 I100376（2173 的结构图 + 印刷
  SMILES + 1036-1079 完整活性表）；
* PDF 第 243 页是纯活性表（无结构图、无 SMILES 列）；
* PDF 第 269 页是国际检索报告。

因此本专利内不存在"管线从未见过的 SMILES 表页"，2173 的测试已经是
本专利内能做到的最强泛化测试。

### 3.2 可用的外部独立来源：PubChem

PubChem PUG REST 可访问，用它对 CSV 里的 OCR 与 DECIMER 字符串做
身份检索（`experiments/phase3_pubchem_crosscheck.py`）：

| cmpd | 来源 | 本地分子式 | PubChem 命中 |
|---|---|---|---|
| 2173 | OCR（已发布） | 无法解析 | 否 |
| 2173 | DECIMER | 无法解析 | 否 |
| 1036 | DECIMER | C77H104F8N14O13S3 | 否 |
| 1038 | DECIMER | C69H91F12N13O13S | 否 |

**未命中是预期结果，不是失败**：这些是专利里的新化合物，尚未存入
PubChem。**检索未命中绝不可计为 OCR 错误**，因此本节不产生任何
准确率结论，只记录"该独立来源目前无法用于这批化合物"。

---

## 4. 全量语料结果（336 个单元格）

### 4.1 validated 路线在全部单元格上的表现

`experiments/phase3_validated_corpus.py`，覆盖全部 336 个已裁出的
印刷文本单元格（335 个数据单元格 + 1 个 header）：

| 指标 | validated 路线 | 生产路线（已发布 CSV） |
|---|---:|---:|
| 单元格数 | 336 | 335 |
| 括号配平 | **228** | **0** |
| 行数完整 | **336** | — |
| 严格通过（配平 + 行数 + 长度 220-300） | 224 | 0 |
| 含非法字符 | 62 | — |
| 长度中位数 | 268 | — |

**"括号配平"和"行数完整"是文本层面的代理指标，不是准确率。**
它们只说明"字符串长得像一条完整的 SMILES"，不代表结构正确。

### 4.2 只换识别器的严格对照

前面的 `up2 vs none` 探针把识别器和预处理一起改了，是混淆的。
`experiments/phase3_recogniser_only.py` 做了干净的对照：

* 裁剪几何**完全相同**——同一批墨迹带、同样去掉竖直表格线、
  同样的上下 padding、同样 2 倍放大；
* 只换识别器，并且都用 recognition-only 模式（`use_det=False`），
  避免检测器成为第二个变量。

结果（336 个单元格）：

| | 英文 PP-OCRv4 mobile（生产） | 上游默认 PP-OCRv6 small |
|---|---:|---:|
| 括号配平 | **2** | **228** |
| 含非法字符的单元格 | 148 | 62 |
| 左括号总数 | 10434 | 12815 |
| 右括号总数 | 12863 | 12881 |
| 长度中位数 | 262 | 268 |

英文识别器总共少识别出 **2429 个左括号**——这就是正式 CSV 里
335/335 条括号失衡的直接来源，而且它**与裁剪方式无关**，纯粹是
识别器模型的选择问题。

这条对照同时**解释了主管的疑问**：之前看到的"放大之后反而更差"，
是因为那组对比里识别器也在变。把识别器固定住之后，放大在
1036/1037 上是无差别的（见第 2 节），真正决定成败的是识别器。

---

## 5. 复现命令

```powershell
# 2173 泛化测试（两条路线 + 转录 diff）
.venv\Scripts\python.exe -X utf8 experiments/phase3_external_2173_check.py

# 主管的问题：放大为什么可能更差
.venv\Scripts\python.exe -X utf8 experiments/phase3_upscale_question.py

# 全量 335 单元格的 validated 路线语料统计
.venv\Scripts\python.exe -X utf8 experiments/phase3_validated_corpus.py

# 只换识别器、几何完全相同的对照
.venv\Scripts\python.exe -X utf8 experiments/phase3_recogniser_only.py

# 外部专利源探测（会记录每个 HTTP 状态）
.venv\Scripts\python.exe -X utf8 experiments/phase3_fetch_second_patent.py WO2022234852

# PubChem 独立交叉检索
.venv\Scripts\python.exe -X utf8 experiments/phase3_pubchem_crosscheck.py 2173 1036 1038
```

---

## 6. 产物索引

| 文件 | 内容 |
|---|---|
| `external_2173_check.csv` | 2173 的逐路线统计 |
| `external_2173_reference.json` | 2173 的 8 行印刷转录 + 参考状态 |
| `evidence_2173/line01..08.png` | 8 行印刷文本的 3 倍放大证据图 |
| `external_2173_routes.csv` | 2173/1036/1037/1038 的四路对比 |
| `pubchem_crosscheck.csv` | PubChem 交叉检索结果 |
| `external_patent_search.json` | 专利源探测记录 |
| `pdf_table_pages.json` | 270 页的表格特征扫描 |
| `pdf_to_golden_match.json` | PDF 页与黄金表的对应 |
| `rendered/pdf_page_0240.png` | PDF 第 240 页（= I100376）渲染 |
| `../upscale_question/` | 放大实验的 CSV 与总结 |
| `../validated_corpus/` | 全量 336 单元格的 validated 语料统计 |
| `../recogniser_only/` | 只换识别器的对照实验 |

