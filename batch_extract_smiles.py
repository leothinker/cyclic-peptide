import cv2
import numpy as np
import re
import glob
import os
import pandas as pd
from rapidocr import RapidOCR
from rdkit import Chem
from rdkit.Chem import Descriptors

engine = RapidOCR()


def auto_repair_smiles(smiles: str) -> str:
    s = smiles
    s = re.sub(r"[^\x00-\x7F]+", "", s)
    s = re.sub(r"C\(O\)N", "C(=O)N", s)
    s = re.sub(r"C\(=[0O]\)?N", "C(=O)N", s)
    s = re.sub(r"(\[C@{1,2}H)\(", r"\1](", s)
    return s


def find_exact_table_rows(img):
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(W * 0.4), 1))
    lines_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    row_sums = np.sum(lines_img, axis=1)
    line_ys = np.where(row_sums > 0)[0]

    clusters = []
    if len(line_ys) > 0:
        cur = [line_ys[0]]
        for y in line_ys[1:]:
            if y - cur[-1] <= 10:
                cur.append(y)
            else:
                clusters.append(int(np.mean(cur)))
                cur = [y]
        clusters.append(int(np.mean(cur)))

    inner_lines = [y for y in clusters if 40 < y < H - 40]

    if len(inner_lines) == 3:
        dividers = [0] + inner_lines + [H]
    else:
        dividers = [int(i * H / 4) for i in range(5)]

    slices = []
    for i in range(len(dividers) - 1):
        y1 = dividers[i] + 1 if i > 0 else 0
        y2 = dividers[i + 1] - 1 if i < len(dividers) - 2 else H
        slices.append((y1, y2))
    return slices


def process_all_images(
    img_dir="data/golden_tables", output_csv="extracted_cyclic_peptides.csv"
):
    image_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    print(f"🚀 找到 {len(image_paths)} 张表格图片，开始全自动批量提取...")

    success_records = []
    failed_records = []

    for img_idx, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            continue

        H, W = img.shape[:2]
        slices = find_exact_table_rows(img)

        # 识别编号
        left_col = img[:, : int(W * 0.15)]
        id_res = engine(left_col)
        found_ids = []
        if id_res:
            txts = (
                id_res.txts if hasattr(id_res, "txts") else [item[1] for item in id_res]
            )
            boxes = (
                id_res.boxes
                if hasattr(id_res, "boxes")
                else [item[0] for item in id_res]
            )
            for box, txt in zip(boxes, txts):
                if re.match(r"^\d{4}$", txt.strip()):
                    center_y = (box[0][1] + box[2][1]) / 2.0
                    found_ids.append((center_y, txt.strip()))

        found_ids.sort(key=lambda x: x[0])
        cmpd_names = [item[1] for item in found_ids]
        while len(cmpd_names) < len(slices):
            cmpd_names.append(f"Row_{len(cmpd_names) + 1}")

        # 逐行提取 SMILES
        for i, (y_start, y_end) in enumerate(slices):
            cmpd_id = cmpd_names[i]
            smiles_crop = img[y_start:y_end, int(W * 0.55) :]

            padded = cv2.copyMakeBorder(
                smiles_crop, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )
            zoomed = cv2.resize(
                padded, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC
            )

            smiles_res = engine(zoomed)
            line_items = []
            if smiles_res:
                boxes = (
                    smiles_res.boxes
                    if hasattr(smiles_res, "boxes")
                    else [item[0] for item in smiles_res]
                )
                txts = (
                    smiles_res.txts
                    if hasattr(smiles_res, "txts")
                    else [item[1] for item in smiles_res]
                )

                for box, t in zip(boxes, txts):
                    clean_t = t.strip()
                    if len(clean_t) > 2 and not re.match(r"^[().=]+$", clean_t):
                        box_y = (box[0][1] + box[2][1]) / 2.0
                        line_items.append((box_y, clean_t))

            line_items.sort(key=lambda x: x[0])
            text_lines = [item[1] for item in line_items]

            raw_smiles = "".join(text_lines)
            fixed_smiles = auto_repair_smiles(raw_smiles)

            # RDKit 语法验证
            mol = Chem.MolFromSmiles(fixed_smiles)
            if mol:
                mw = round(Descriptors.MolWt(mol), 2)
                heavy_atoms = mol.GetNumHeavyAtoms()
                success_records.append(
                    {
                        "cmpd_id": cmpd_id,
                        "canonical_smiles": Chem.MolToSmiles(mol),
                        "raw_smiles": fixed_smiles,
                        "molecular_weight": mw,
                        "heavy_atom_count": heavy_atoms,
                        "source_image": img_name,
                        "status": "VALID",
                    }
                )
            else:
                failed_records.append(
                    {
                        "cmpd_id": cmpd_id,
                        "raw_smiles": fixed_smiles,
                        "source_image": img_name,
                        "status": "PARSE_ERROR",
                    }
                )

        print(
            f"[{img_idx + 1}/{len(image_paths)}] {img_name} -> 成功累计: {len(success_records)} 个 | 待微调: {len(failed_records)} 个"
        )

    # 导出最终高精度数据集！
    df_valid = pd.DataFrame(success_records)
    df_valid.to_csv(output_csv, index=False, encoding="utf-8-sig")

    # 导出待微调清单备用
    df_failed = pd.DataFrame(failed_records)
    df_failed.to_csv("failed_need_review.csv", index=False, encoding="utf-8-sig")

    print(f"\n==========================================")
    print(f"🎉 批量提取全部完成！")
    print(f"✅ 100% 验证通过的高精度分子数: {len(df_valid)}")
    print(f"⚠️ 待微调分子数: {len(df_failed)}")
    print(f"📁 最终数据集已保存至: {output_csv}")
    print(f"==========================================")


if __name__ == "__main__":
    process_all_images()
