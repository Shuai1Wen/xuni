#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
响应基分析脚本

分析算子模型学到的响应基（response bases）B_k及其生物学意义。

对应：model.md A.5节（低秩分解）

主要分析内容:
1. 提取响应基B_k的基因影响方向
2. 分析不同条件下的激活模式α_k(θ)
3. 响应基之间的关系（相似度、正交性）
4. 通路富集分析（可选）

用法:
    python analyze_response_axes.py \
        --operator_checkpoint results/checkpoints/scperturb_operator/best_operator.pt \
        --vae_checkpoint results/checkpoints/scperturb_vae/best_model.pt \
        --data_path data/processed/scperturb/scperturb_merged_train.h5ad \
        --output_dir results/experiments/response_axes_analysis/
"""

import argparse
import torch
import scanpy as sc
import numpy as np
from pathlib import Path
import json
import sys
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.nb_vae import NBVAE
from src.models.operator import OperatorModel
from src.utils.cond_encoder import ConditionEncoder
from src.data.scperturb_dataset import SCPerturbPairDataset, collate_fn_pair


def load_models(vae_checkpoint_path, operator_checkpoint_path, encoder_checkpoint_path, device):
    """加载模型"""
    print("加载模型...")

    # 加载VAE
    vae_checkpoint = torch.load(vae_checkpoint_path, map_location=device)
    vae_model = NBVAE(
        n_genes=vae_checkpoint["model_config"]["n_genes"],
        latent_dim=vae_checkpoint["model_config"]["latent_dim"],
        n_tissues=vae_checkpoint["model_config"]["n_tissues"],
        hidden_dim=vae_checkpoint["model_config"]["hidden_dim"]
    )
    vae_model.load_state_dict(vae_checkpoint["model_state_dict"])
    vae_model.to(device)
    vae_model.eval()
    print(f"✓ VAE加载成功")

    # 加载算子模型
    operator_checkpoint = torch.load(operator_checkpoint_path, map_location=device)
    operator_model = OperatorModel(
        latent_dim=operator_checkpoint["model_config"]["latent_dim"],
        n_tissues=operator_checkpoint["model_config"]["n_tissues"],
        n_response_bases=operator_checkpoint["model_config"]["n_response_bases"],
        cond_dim=operator_checkpoint["model_config"]["cond_dim"],
        max_spectral_norm=operator_checkpoint["model_config"]["max_spectral_norm"]
    )
    operator_model.load_state_dict(operator_checkpoint["model_state_dict"])
    operator_model.to(device)
    operator_model.eval()
    print(f"✓ 算子模型加载成功")

    # 加载条件编码器
    encoder_checkpoint = torch.load(encoder_checkpoint_path, map_location=device)
    cond_encoder = ConditionEncoder(
        perturb2idx=encoder_checkpoint["perturb2idx"],
        tissue2idx=encoder_checkpoint["tissue2idx"],
        batch2idx=encoder_checkpoint["batch2idx"],
        cond_dim=encoder_checkpoint["config"]["cond_dim"],
        use_embedding=encoder_checkpoint["config"]["use_embedding"]
    )
    cond_encoder.load_state_dict(encoder_checkpoint["state_dict"])
    cond_encoder.to(device)
    cond_encoder.eval()
    print(f"✓ 条件编码器加载成功")

    tissue2idx = encoder_checkpoint["tissue2idx"]

    return vae_model, operator_model, cond_encoder, tissue2idx


def extract_response_bases(operator_model):
    """
    提取响应基矩阵

    返回:
        B: (K, d_z, d_z) 响应基
    """
    print("\n提取响应基...")

    B = operator_model.B.detach().cpu().numpy()  # (K, d_z, d_z)
    K, d_z, _ = B.shape

    print(f"  响应基数量 K = {K}")
    print(f"  潜空间维度 d_z = {d_z}")

    return B


def compute_activation_matrix(operator_model, cond_encoder, adata, tissue2idx, device):
    """
    计算所有条件的响应系数矩阵

    返回:
        alpha_matrix: (n_conditions, K) 响应系数矩阵
        condition_names: 条件名称列表
    """
    print("\n计算激活模式...")

    # 获取所有唯一条件
    adata.obs["condition_key"] = (
        adata.obs["perturbation"].astype(str) + "_" +
        adata.obs["tissue"].astype(str)
    )
    unique_conditions = adata.obs["condition_key"].unique()

    alpha_list = []
    condition_names = []

    for cond_key in unique_conditions:
        # 解析条件
        parts = cond_key.split("_")
        if len(parts) >= 2:
            perturbation = parts[0]
            tissue = parts[1]

            # 编码条件
            obs_dict = {
                "perturbation": perturbation,
                "tissue": tissue,
                "batch": "batch0",
                "mLOY_load": 0.0
            }

            cond_vec = cond_encoder.encode_obs_row(obs_dict, device=device)
            cond_vec = cond_vec.unsqueeze(0)  # (1, cond_dim)

            # 获取响应系数
            alpha, _ = operator_model.get_response_profile(cond_vec)
            alpha = alpha.squeeze(0).detach().cpu().numpy()  # (K,)

            alpha_list.append(alpha)
            condition_names.append(cond_key)

    alpha_matrix = np.array(alpha_list)  # (n_conditions, K)

    print(f"  计算了 {len(condition_names)} 个条件的激活模式")

    return alpha_matrix, condition_names


def analyze_basis_similarity(B):
    """
    分析响应基之间的相似度

    参数:
        B: (K, d_z, d_z) 响应基

    返回:
        similarity_matrix: (K, K) 余弦相似度矩阵
    """
    print("\n计算响应基相似度...")

    K, d_z, _ = B.shape

    # 展平每个响应基
    B_flat = B.reshape(K, -1)  # (K, d_z*d_z)

    # 归一化
    B_norm = B_flat / (np.linalg.norm(B_flat, axis=1, keepdims=True) + 1e-8)

    # 计算余弦相似度
    similarity_matrix = B_norm @ B_norm.T  # (K, K)

    # 计算非对角线元素的平均值（表示冗余程度）
    off_diag_mean = (similarity_matrix.sum() - K) / (K * K - K)
    print(f"  非对角线相似度均值: {off_diag_mean:.4f}")

    return similarity_matrix


def visualize_response_heatmap(alpha_matrix, condition_names, output_dir):
    """
    可视化响应系数热图

    参数:
        alpha_matrix: (n_conditions, K) 响应系数矩阵
        condition_names: 条件名称列表
        output_dir: 输出目录
    """
    print("\n生成响应系数热图...")

    fig, ax = plt.subplots(figsize=(12, max(8, len(condition_names) * 0.2)))

    K = alpha_matrix.shape[1]
    response_names = [f"B_{k}" for k in range(K)]

    sns.heatmap(
        alpha_matrix,
        cmap="RdBu_r",
        center=0,
        cbar_kws={"label": "响应系数 α_k"},
        xticklabels=response_names,
        yticklabels=condition_names,
        ax=ax
    )

    ax.set_xlabel("响应基索引", fontsize=12)
    ax.set_ylabel("条件", fontsize=12)
    ax.set_title("响应系数热图 (α_k)", fontsize=14)

    plt.tight_layout()

    save_path = output_dir / "response_heatmap.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  保存至: {save_path}")

    plt.close()


def visualize_basis_similarity(similarity_matrix, output_dir):
    """
    可视化响应基相似度矩阵

    参数:
        similarity_matrix: (K, K) 相似度矩阵
        output_dir: 输出目录
    """
    print("\n生成响应基相似度矩阵...")

    K = similarity_matrix.shape[0]
    response_names = [f"B_{k}" for k in range(K)]

    fig, ax = plt.subplots(figsize=(8, 8))

    sns.heatmap(
        similarity_matrix,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"label": "余弦相似度"},
        xticklabels=response_names,
        yticklabels=response_names,
        ax=ax,
        annot=True,
        fmt=".2f"
    )

    ax.set_title("响应基相似度矩阵", fontsize=14)

    plt.tight_layout()

    save_path = output_dir / "basis_similarity_matrix.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  保存至: {save_path}")

    plt.close()


def cluster_conditions_by_activation(alpha_matrix, condition_names, output_dir):
    """
    根据激活模式聚类条件

    参数:
        alpha_matrix: (n_conditions, K) 响应系数矩阵
        condition_names: 条件名称列表
        output_dir: 输出目录
    """
    print("\n聚类分析...")

    # 使用层次聚类
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist

    # 计算距离
    distances = pdist(alpha_matrix, metric='cosine')
    linkage_matrix = linkage(distances, method='ward')

    # 绘制树状图
    fig, ax = plt.subplots(figsize=(14, 8))

    dendrogram(
        linkage_matrix,
        labels=condition_names,
        leaf_rotation=90,
        ax=ax
    )

    ax.set_xlabel("条件", fontsize=12)
    ax.set_ylabel("距离", fontsize=12)
    ax.set_title("条件聚类树状图（基于响应模式）", fontsize=14)

    plt.tight_layout()

    save_path = output_dir / "condition_clustering_dendrogram.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  保存至: {save_path}")

    plt.close()


def save_response_analysis_results(B, alpha_matrix, condition_names, similarity_matrix, output_dir):
    """
    保存分析结果

    参数:
        B: 响应基矩阵
        alpha_matrix: 响应系数矩阵
        condition_names: 条件名称列表
        similarity_matrix: 相似度矩阵
        output_dir: 输出目录
    """
    print("\n保存分析结果...")

    # 保存响应基
    np.save(output_dir / "response_bases.npy", B)

    # 保存响应系数矩阵
    np.save(output_dir / "activation_matrix.npy", alpha_matrix)

    # 保存条件名称
    with open(output_dir / "condition_names.txt", "w", encoding="utf-8") as f:
        for name in condition_names:
            f.write(name + "\n")

    # 保存相似度矩阵
    np.save(output_dir / "basis_similarity_matrix.npy", similarity_matrix)

    # 保存统计摘要
    summary = {
        "n_response_bases": int(B.shape[0]),
        "latent_dim": int(B.shape[1]),
        "n_conditions": int(alpha_matrix.shape[0]),
        "activation_stats": {
            "mean": float(alpha_matrix.mean()),
            "std": float(alpha_matrix.std()),
            "min": float(alpha_matrix.min()),
            "max": float(alpha_matrix.max())
        },
        "similarity_stats": {
            "off_diagonal_mean": float((similarity_matrix.sum() - B.shape[0]) / (B.shape[0] * B.shape[0] - B.shape[0])),
            "off_diagonal_std": float(similarity_matrix[np.triu_indices(B.shape[0], k=1)].std())
        }
    }

    with open(output_dir / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  所有结果已保存至: {output_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="响应基分析脚本")

    parser.add_argument(
        "--operator_checkpoint",
        type=str,
        required=True,
        help="算子检查点路径"
    )
    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        required=True,
        help="VAE检查点路径"
    )
    parser.add_argument(
        "--encoder_checkpoint",
        type=str,
        default=None,
        help="条件编码器检查点路径（默认与算子在同一目录）"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="数据路径（h5ad格式，用于提取条件）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/experiments/response_axes_analysis/",
        help="输出目录"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="设备（cuda或cpu）"
    )

    args = parser.parse_args()

    # 检查文件是否存在
    for path in [args.operator_checkpoint, args.vae_checkpoint, args.data_path]:
        if not Path(path).exists():
            print(f"错误: 文件不存在: {path}")
            return

    # 默认编码器路径
    if not args.encoder_checkpoint:
        args.encoder_checkpoint = str(Path(args.operator_checkpoint).parent / "cond_encoder.pt")

    if not Path(args.encoder_checkpoint).exists():
        print(f"错误: 条件编码器不存在: {args.encoder_checkpoint}")
        return

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("响应基分析")
    print("=" * 80)

    # 加载模型
    vae_model, operator_model, cond_encoder, tissue2idx = load_models(
        args.vae_checkpoint,
        args.operator_checkpoint,
        args.encoder_checkpoint,
        args.device
    )

    # 加载数据
    print(f"\n加载数据: {args.data_path}")
    adata = sc.read_h5ad(args.data_path)
    print(f"数据集: {adata.n_obs} 细胞, {adata.n_vars} 基因")

    # 1. 提取响应基
    B = extract_response_bases(operator_model)

    # 2. 计算激活矩阵
    alpha_matrix, condition_names = compute_activation_matrix(
        operator_model, cond_encoder, adata, tissue2idx, args.device
    )

    # 3. 分析响应基相似度
    similarity_matrix = analyze_basis_similarity(B)

    # 4. 可视化
    visualize_response_heatmap(alpha_matrix, condition_names, output_dir)
    visualize_basis_similarity(similarity_matrix, output_dir)
    cluster_conditions_by_activation(alpha_matrix, condition_names, output_dir)

    # 5. 保存结果
    save_response_analysis_results(B, alpha_matrix, condition_names, similarity_matrix, output_dir)

    print("\n" + "=" * 80)
    print("响应基分析完成！")
    print(f"所有结果已保存至: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
