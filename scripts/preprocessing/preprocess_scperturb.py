#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scPerturb数据预处理脚本

对scPerturb原始数据进行标准化预处理，包括：
1. 基因筛选（高变基因）
2. 细胞质控
3. 归一化和对数转换
4. 数据划分（训练/验证/测试）

用法:
    python preprocess_scperturb.py \
        --input data/raw/scperturb/raw_data.h5ad \
        --output data/processed/scperturb/ \
        --n_top_genes 2000 \
        --min_cells 100 \
        --test_split 0.15 \
        --val_split 0.15
"""

import argparse
import scanpy as sc
import numpy as np
from pathlib import Path
import json
import re

from src.utils.perturbation import normalize_perturbation_label


def quality_control(adata, min_genes=200, min_cells=100):
    """
    质量控制

    参数:
        adata: AnnData对象
        min_genes: 细胞最少表达基因数
        min_cells: 基因最少表达细胞数

    返回:
        adata: 过滤后的AnnData对象
    """
    print("\n质量控制...")
    print(f"  原始数据: {adata.n_obs} 细胞, {adata.n_vars} 基因")

    # 过滤细胞
    sc.pp.filter_cells(adata, min_genes=min_genes)
    print(f"  过滤后（细胞最少{min_genes}个基因）: {adata.n_obs} 细胞")

    # 过滤基因
    sc.pp.filter_genes(adata, min_cells=min_cells)
    print(f"  过滤后（基因最少{min_cells}个细胞）: {adata.n_vars} 基因")

    return adata


def normalize_and_log(adata):
    """
    归一化和对数转换

    参数:
        adata: AnnData对象

    返回:
        adata: 归一化后的AnnData对象
    """
    print("\n归一化和对数转换...")

    # 总计数归一化（每个细胞总和归一化到10000）
    sc.pp.normalize_total(adata, target_sum=1e4)

    # log1p转换
    sc.pp.log1p(adata)

    print("  完成归一化和log1p转换")

    return adata


def select_highly_variable_genes(adata, n_top_genes=2000):
    """
    选择高变基因

    参数:
        adata: AnnData对象
        n_top_genes: 高变基因数量

    返回:
        adata: 包含高变基因的AnnData对象
    """
    print(f"\n选择高变基因（top {n_top_genes}）...")

    # 计算高变基因
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat_v3")

    # 筛选高变基因
    adata = adata[:, adata.var["highly_variable"]].copy()

    print(f"  选择了 {adata.n_vars} 个高变基因")

    return adata


def split_data_by_condition(adata, test_split=0.15, val_split=0.15, seed=42):
    """
    按条件划分数据集

    确保同一条件的细胞不会跨越训练/验证/测试集。

    参数:
        adata: AnnData对象
        test_split: 测试集比例
        val_split: 验证集比例
        seed: 随机种子

    返回:
        adata_train, adata_val, adata_test: 划分后的数据集
    """
    print(f"\n划分数据集（测试集={test_split}, 验证集={val_split}）...")

    np.random.seed(seed)

    # 获取所有唯一条件（扰动+组织+细胞类型）
    # 使用"||"作为分隔符避免扰动名称中的"_"干扰
    adata.obs["condition_key"] = (
        adata.obs["perturbation"].astype(str) + "||" +
        adata.obs["tissue"].astype(str) + "||" +
        adata.obs.get("cell_type", "unknown").astype(str)
    )

    unique_conditions = adata.obs["condition_key"].unique()
    n_conditions = len(unique_conditions)

    print(f"  总条件数: {n_conditions}")

    # 随机打乱条件
    shuffled_conditions = np.random.permutation(unique_conditions)

    # 计算划分点
    n_test = int(n_conditions * test_split)
    n_val = int(n_conditions * val_split)

    test_conditions = set(shuffled_conditions[:n_test])
    val_conditions = set(shuffled_conditions[n_test:n_test + n_val])
    train_conditions = set(shuffled_conditions[n_test + n_val:])

    # 划分数据
    test_mask = adata.obs["condition_key"].isin(test_conditions)
    val_mask = adata.obs["condition_key"].isin(val_conditions)
    train_mask = adata.obs["condition_key"].isin(train_conditions)

    adata_test = adata[test_mask].copy()
    adata_val = adata[val_mask].copy()
    adata_train = adata[train_mask].copy()

    print(f"  训练集: {adata_train.n_obs} 细胞 ({len(train_conditions)} 条件)")
    print(f"  验证集: {adata_val.n_obs} 细胞 ({len(val_conditions)} 条件)")
    print(f"  测试集: {adata_test.n_obs} 细胞 ({len(test_conditions)} 条件)")

    return adata_train, adata_val, adata_test


def _parse_perturbation_components(label: str) -> tuple:
    """解析组合扰动的组成部分"""
    normalized = normalize_perturbation_label(label)
    if normalized == "control":
        return ("control",)
    parts = [p for p in re.split(r"[+,&;|]+", normalized) if p.strip()]
    return tuple(sorted(p.strip() for p in parts)) if parts else (normalized,)


def split_data_systema(
    adata,
    test_split=0.15,
    val_split=0.15,
    seed=42,
    combo_unseen_mode: str = "strict"
):
    """
    Systema风格划分策略

    1) 扰动级别unseen split（单扰动）
    2) 组合扰动按组合阶数分层
    3) control样本按细胞级随机划分
    """
    print(
        f"\nSystema划分（测试集={test_split}, 验证集={val_split}, "
        f"组合模式={combo_unseen_mode}）..."
    )

    rng = np.random.default_rng(seed)

    if "perturbation_norm" not in adata.obs.columns:
        adata.obs["perturbation_norm"] = adata.obs["perturbation"].map(
            normalize_perturbation_label
        )

    perturbation_labels = adata.obs["perturbation_norm"].unique().tolist()
    perturbation_components = {
        p: _parse_perturbation_components(p) for p in perturbation_labels
    }
    perturbation_sizes = {
        p: len(components) for p, components in perturbation_components.items()
    }

    control_label = "control"
    control_mask = adata.obs["perturbation_norm"] == control_label

    single_perts = [p for p in perturbation_labels if perturbation_sizes[p] == 1 and p != control_label]
    combo_perts = [p for p in perturbation_labels if perturbation_sizes[p] > 1]

    rng.shuffle(single_perts)
    n_single = len(single_perts)
    n_single_test = int(n_single * test_split)
    n_single_val = int(n_single * val_split)

    single_test = set(single_perts[:n_single_test])
    single_val = set(single_perts[n_single_test:n_single_test + n_single_val])
    single_train = set(single_perts[n_single_test + n_single_val:])

    combo_train, combo_val, combo_test = set(), set(), set()
    combo_sizes = sorted({perturbation_sizes[p] for p in combo_perts})
    for size in combo_sizes:
        perts_of_size = [p for p in combo_perts if perturbation_sizes[p] == size]
        if not perts_of_size:
            continue
        if combo_unseen_mode == "strict":
            eligible = []
            for pert in perts_of_size:
                components = set(perturbation_components[pert])
                if components.issubset(single_train):
                    eligible.append(pert)
                else:
                    combo_train.add(pert)
            perts_of_size = eligible
        rng.shuffle(perts_of_size)
        n_combo = len(perts_of_size)
        n_combo_test = int(n_combo * test_split)
        n_combo_val = int(n_combo * val_split)
        combo_test.update(perts_of_size[:n_combo_test])
        combo_val.update(perts_of_size[n_combo_test:n_combo_test + n_combo_val])
        combo_train.update(perts_of_size[n_combo_test + n_combo_val:])

    train_perts = single_train | combo_train
    val_perts = single_val | combo_val
    test_perts = single_test | combo_test

    train_mask = adata.obs["perturbation_norm"].isin(train_perts)
    val_mask = adata.obs["perturbation_norm"].isin(val_perts)
    test_mask = adata.obs["perturbation_norm"].isin(test_perts)

    # control样本按细胞级随机划分，确保三份数据都包含对照
    control_indices = adata.obs[control_mask].index.to_numpy()
    rng.shuffle(control_indices)
    n_control = len(control_indices)
    n_control_test = int(n_control * test_split)
    n_control_val = int(n_control * val_split)
    control_test_idx = set(control_indices[:n_control_test])
    control_val_idx = set(control_indices[n_control_test:n_control_test + n_control_val])
    control_train_idx = set(control_indices[n_control_test + n_control_val:])

    control_mask_train = adata.obs.index.isin(control_train_idx)
    control_mask_val = adata.obs.index.isin(control_val_idx)
    control_mask_test = adata.obs.index.isin(control_test_idx)

    train_mask = train_mask | control_mask_train
    val_mask = val_mask | control_mask_val
    test_mask = test_mask | control_mask_test

    adata_train = adata[train_mask].copy()
    adata_val = adata[val_mask].copy()
    adata_test = adata[test_mask].copy()

    print(f"  单扰动train/val/test: {len(single_train)}/{len(single_val)}/{len(single_test)}")
    print(f"  组合扰动train/val/test: {len(combo_train)}/{len(combo_val)}/{len(combo_test)}")
    print(f"  训练集: {adata_train.n_obs} 细胞")
    print(f"  验证集: {adata_val.n_obs} 细胞")
    print(f"  测试集: {adata_test.n_obs} 细胞")

    return adata_train, adata_val, adata_test


def save_datasets(adata_train, adata_val, adata_test, output_dir):
    """
    保存数据集

    参数:
        adata_train: 训练集
        adata_val: 验证集
        adata_test: 测试集
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n保存数据集到: {output_dir}")

    # 保存h5ad文件
    adata_train.write_h5ad(output_dir / "scperturb_merged_train.h5ad")
    adata_val.write_h5ad(output_dir / "scperturb_merged_val.h5ad")
    adata_test.write_h5ad(output_dir / "scperturb_merged_test.h5ad")

    print(f"  ✓ 训练集: scperturb_merged_train.h5ad")
    print(f"  ✓ 验证集: scperturb_merged_val.h5ad")
    print(f"  ✓ 测试集: scperturb_merged_test.h5ad")

    # 保存元数据
    metadata = {
        "n_cells_train": int(adata_train.n_obs),
        "n_cells_val": int(adata_val.n_obs),
        "n_cells_test": int(adata_test.n_obs),
        "n_genes": int(adata_train.n_vars),
        "tissues": list(adata_train.obs["tissue"].unique()),
        "perturbations": list(adata_train.obs["perturbation"].unique())
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"  ✓ 元数据: metadata.json")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="scPerturb数据预处理脚本")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入h5ad文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出目录"
    )
    parser.add_argument(
        "--n_top_genes",
        type=int,
        default=2000,
        help="高变基因数量（默认2000）"
    )
    parser.add_argument(
        "--min_genes",
        type=int,
        default=200,
        help="细胞最少表达基因数（默认200）"
    )
    parser.add_argument(
        "--min_cells",
        type=int,
        default=100,
        help="基因最少表达细胞数（默认100）"
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.15,
        help="测试集比例（默认0.15）"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.15,
        help="验证集比例（默认0.15）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认42）"
    )
    parser.add_argument(
        "--split_strategy",
        type=str,
        choices=["condition", "systema"],
        default="systema",
        help="数据划分策略（condition或systema，默认systema）"
    )
    parser.add_argument(
        "--combo_unseen_mode",
        type=str,
        choices=["strict", "relaxed"],
        default="strict",
        help="组合扰动划分模式（strict保证单扰动在训练集中出现）"
    )

    args = parser.parse_args()

    # 检查输入文件
    if not Path(args.input).exists():
        print(f"错误: 输入文件不存在: {args.input}")
        return

    print("=" * 80)
    print("scPerturb数据预处理")
    print("=" * 80)

    # 加载数据
    print(f"\n加载数据: {args.input}")
    adata = sc.read_h5ad(args.input)

    # 检查必需的obs字段
    required_fields = ["perturbation", "tissue", "timepoint"]
    missing_fields = [f for f in required_fields if f not in adata.obs.columns]
    if missing_fields:
        print(f"警告: 缺少必需字段: {missing_fields}")
        print("请确保数据包含以下字段: perturbation, tissue, timepoint")

    # 预处理流程
    adata = quality_control(adata, min_genes=args.min_genes, min_cells=args.min_cells)
    adata = normalize_and_log(adata)
    adata = select_highly_variable_genes(adata, n_top_genes=args.n_top_genes)

    # 划分数据集
    if args.split_strategy == "systema":
        adata_train, adata_val, adata_test = split_data_systema(
            adata,
            test_split=args.test_split,
            val_split=args.val_split,
            seed=args.seed,
            combo_unseen_mode=args.combo_unseen_mode
        )
    else:
        adata_train, adata_val, adata_test = split_data_by_condition(
            adata,
            test_split=args.test_split,
            val_split=args.val_split,
            seed=args.seed
        )

    # 保存数据集
    save_datasets(adata_train, adata_val, adata_test, args.output)

    print("\n" + "=" * 80)
    print("预处理完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
