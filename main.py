"""Pipeline entry point: Extract cyclic-peptide SMILES & RAS bioactivity from patent tables."""

from pathlib import Path
import pandas as pd
from cpd.parsers.table_finder import TableType, scan_directory
from cpd.parsers.table_extractor import RapidOcrTableExtractor


def run_pipeline(
    img_dir: str = "data/raw/WO2025162428/golden_tables",
    out_csv: str = "data/processed/cyclic_peptides_benchmark.csv",
) -> None:
    source_dir = Path(img_dir)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"🚀 [1/3] 扫描目录中的表格图片: {source_dir}...")
    images = scan_directory(source_dir)
    smiles_images = [img for img in images if img.table_type == TableType.SMILES]
    activity_images = [img for img in images if img.table_type == TableType.ACTIVITY]

    print(f"   -> 发现 {len(smiles_images)} 张 SMILES 结构表")
    print(f"   -> 发现 {len(activity_images)} 张 G12V KD 活性表")

    extractor = RapidOcrTableExtractor()

    # 1. 提取所有 SMILES
    print(f"\n📦 [2/3] 正在提取 SMILES 结构并进行 RDKit 语法自愈...")
    all_smiles = []
    for idx, img in enumerate(smiles_images):
        rows = extractor.extract_smiles(img.image_path)
        all_smiles.extend([r.to_dict() for r in rows if r.is_valid])
        print(
            f"   [{idx + 1}/{len(smiles_images)}] {img.filename} -> 累计有效环肽: {len(all_smiles)} 个",
            end="\r",
        )
    print()

    # 2. 提取所有活性数值
    print(f"\n🧪 [3/3] 正在提取 RAS G12V GDP KD (nM) 亲和力数值...")
    all_activity = []
    for idx, img in enumerate(activity_images):
        rows = extractor.extract_activity(img.image_path)
        all_activity.extend([r.to_dict() for r in rows if r.kd_nm is not None])
        print(
            f"   [{idx + 1}/{len(activity_images)}] {img.filename} -> 累计活性数据: {len(all_activity)} 条",
            end="\r",
        )
    print()

    # 3. 按 Cmpd # 连接并输出最终数据宽表
    df_smiles = pd.DataFrame(all_smiles)
    df_act = pd.DataFrame(all_activity)

    if df_smiles.empty:
        print("❌ 未能提取到有效的 SMILES 数据，请检查图片路径。")
        return

    df_smiles["cmpd_id"] = df_smiles["cmpd_id"].astype(str).str.strip()

    if not df_act.empty:
        df_act["cmpd_id"] = df_act["cmpd_id"].astype(str).str.strip()
        # 过滤掉非数字的脏 ID（例如可能偶发的 Row_X）
        df_act = df_act[df_act["cmpd_id"].str.match(r"^\d+$")]
        df_act = df_act.drop_duplicates(subset=["cmpd_id"])

        # 执行左连接 (保留所有有效的 SMILES，有活性的填入 kd_nm)
        final_df = pd.merge(
            df_smiles, df_act[["cmpd_id", "kd_nm"]], on="cmpd_id", how="left"
        )
    else:
        final_df = df_smiles

    final_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 55)
    print("🎉 数据集构建圆满完成！")
    print(f"📊 提取到的总环肽分子数: {len(final_df)}")
    if "kd_nm" in final_df.columns:
        valid_kd = final_df["kd_nm"].notna().sum()
        print(f"🎯 成功匹配到 G12V KD 亲和力的分子数: {valid_kd}")
    print(f"📁 最终交付数据集已保存至: {out_path}")
    print("=" * 55)


if __name__ == "__main__":
    # 如果你想先拿 golden_tables 测试，可以把参数改为 data/raw/WO2025162428/golden_tables
    run_pipeline()
