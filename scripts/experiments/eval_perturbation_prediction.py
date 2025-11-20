#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
扰动预测评估脚本

评估训练好的VAE+算子模型在scPerturb测试集上的性能。

对应：model.md A.9节（评估指标）

用法:
    python eval_perturbation_prediction.py \
        --vae_checkpoint results/checkpoints/scperturb_vae/best_model.pt \
        --operator_checkpoint results/checkpoints/scperturb_operator/best_operator.pt \
        --data_path data/processed/scperturb/scperturb_merged_test.h5ad \
        --output_dir results/experiments/scperturb_evaluation/
"""

import argparse
import torch
import scanpy as sc
from pathlib import Path
import json
import numpy as np
import sys
from torch.utils.data import DataLoader

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.nb_vae import NBVAE
from src.models.operator import OperatorModel
from src.utils.cond_encoder import ConditionEncoder
from src.data.scperturb_dataset import SCPerturbPairDataset, collate_fn_pair
from src.evaluation.metrics import (
    reconstruction_metrics,
    distribution_metrics,
    de_gene_prediction_metrics,
    operator_quality_metrics,
    comprehensive_evaluation
)
from src.visualization.plotting import (
    plot_latent_space_umap,
    plot_de_genes_scatter,
    plot_spectral_norm_histogram,
    plot_comprehensive_evaluation_report
)


def load_models(vae_checkpoint_path: str, operator_checkpoint_path: str, encoder_checkpoint_path: str, device: str):
    """
    加载模型和条件编码器

    参数:
        vae_checkpoint_path: VAE检查点路径
        operator_checkpoint_path: 算子检查点路径
        encoder_checkpoint_path: 条件编码器检查点路径
        device: 设备

    返回:
        vae_model: VAE模型
        operator_model: 算子模型
        cond_encoder: 条件编码器
        tissue2idx: 组织索引映射
    """
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
    print(f"✓ VAE加载成功: {vae_checkpoint_path}")

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
    print(f"✓ 算子模型加载成功: {operator_checkpoint_path}")

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
    print(f"✓ 条件编码器加载成功: {encoder_checkpoint_path}")

    tissue2idx = encoder_checkpoint["tissue2idx"]

    return vae_model, operator_model, cond_encoder, tissue2idx


def evaluate_model(
    vae_model,
    operator_model,
    dataloader,
    device: str,
    compute_de_metrics: bool = True
):
    """
    评估模型性能

    参数:
        vae_model: VAE模型
        operator_model: 算子模型
        dataloader: 数据加载器
        device: 设备
        compute_de_metrics: 是否计算差异基因指标

    返回:
        all_metrics: 所有评估指标
        predictions: 预测结果字典
    """
    print("\n开始评估...")

    all_x0 = []
    all_x1_true = []
    all_x1_pred = []
    all_z0 = []
    all_z1_true = []
    all_z1_pred = []
    all_tissue_idx = []
    all_cond_vec = []
    all_spectral_norms = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if (i + 1) % 10 == 0:
                print(f"  处理批次 {i+1}/{len(dataloader)}...")

            x0 = batch["x0"].to(device)
            x1 = batch["x1"].to(device)
            tissue_idx = batch["tissue_idx"].to(device)
            cond_vec = batch["cond_vec"].to(device)

            # 编码 x0 → z0
            mu0, _ = vae_model.encoder(x0, tissue_idx)
            z0 = mu0  # 使用均值

            # 应用算子 z0 → z1_pred
            z1_pred = operator_model(z0, tissue_idx, cond_vec)

            # 解码 z1_pred → x1_pred
            x1_pred = vae_model.decoder.get_mean(z1_pred, tissue_idx)

            # 真实z1（用于分布指标）
            mu1, _ = vae_model.encoder(x1, tissue_idx)
            z1_true = mu1

            # 计算谱范数
            norms = operator_model.compute_operator_norm(tissue_idx, cond_vec, norm_type="spectral")

            all_x0.append(x0.cpu())
            all_x1_true.append(x1.cpu())
            all_x1_pred.append(x1_pred.cpu())
            all_z0.append(z0.cpu())
            all_z1_true.append(z1_true.cpu())
            all_z1_pred.append(z1_pred.cpu())
            all_tissue_idx.append(tissue_idx.cpu())
            all_cond_vec.append(cond_vec.cpu())
            all_spectral_norms.append(norms.cpu())

    # 拼接所有批次
    x0_all = torch.cat(all_x0, dim=0)
    x1_true_all = torch.cat(all_x1_true, dim=0)
    x1_pred_all = torch.cat(all_x1_pred, dim=0)
    z0_all = torch.cat(all_z0, dim=0)
    z1_true_all = torch.cat(all_z1_true, dim=0)
    z1_pred_all = torch.cat(all_z1_pred, dim=0)
    tissue_idx_all = torch.cat(all_tissue_idx, dim=0)
    cond_vec_all = torch.cat(all_cond_vec, dim=0)
    spectral_norms_all = torch.cat(all_spectral_norms, dim=0)

    print(f"总样本数: {x0_all.shape[0]}")

    # 计算各类指标
    all_metrics = {}

    print("\n计算评估指标...")

    # 1. 重建质量
    print("  - 重建质量指标")
    all_metrics["reconstruction"] = reconstruction_metrics(x1_true_all, x1_pred_all)

    # 2. 分布匹配
    print("  - 分布匹配指标")
    all_metrics["distribution"] = distribution_metrics(z1_true_all, z1_pred_all)

    # 3. 差异基因预测
    if compute_de_metrics:
        print("  - 差异基因预测指标")
        all_metrics["de_genes"] = de_gene_prediction_metrics(
            x0_all, x1_true_all, x1_pred_all, top_k=200
        )

    # 4. 算子质量
    print("  - 算子质量指标")
    all_metrics["operator"] = operator_quality_metrics(
        operator_model, tissue_idx_all.to(device), cond_vec_all.to(device), device=device
    )

    # 收集预测结果
    predictions = {
        "x0": x0_all.numpy(),
        "x1_true": x1_true_all.numpy(),
        "x1_pred": x1_pred_all.numpy(),
        "z0": z0_all.numpy(),
        "z1_true": z1_true_all.numpy(),
        "z1_pred": z1_pred_all.numpy(),
        "spectral_norms": spectral_norms_all.numpy()
    }

    return all_metrics, predictions


def save_results(all_metrics, predictions, output_dir: Path):
    """
    保存评估结果

    参数:
        all_metrics: 评估指标字典
        predictions: 预测结果字典
        output_dir: 输出目录
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存指标
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\n指标已保存至: {metrics_path}")

    # 保存预测结果
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)

    for key, value in predictions.items():
        save_path = predictions_dir / f"{key}.npy"
        np.save(save_path, value)

    print(f"预测结果已保存至: {predictions_dir}")


def generate_visualizations(all_metrics, predictions, output_dir: Path):
    """
    生成可视化图表

    参数:
        all_metrics: 评估指标字典
        predictions: 预测结果字典
        output_dir: 输出目录
    """
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("\n生成可视化图表...")

    # 1. 综合评估报告
    print("  - 综合评估报告")
    plot_comprehensive_evaluation_report(
        all_metrics,
        save_dir=str(figures_dir),
        prefix=""
    )

    # 2. 潜空间UMAP（真实vs预测）
    print("  - 潜空间UMAP可视化")
    z_combined = np.vstack([
        predictions["z1_true"],
        predictions["z1_pred"]
    ])
    labels = np.array([0] * len(predictions["z1_true"]) + [1] * len(predictions["z1_pred"]))
    label_names = {0: "真实", 1: "预测"}

    plot_latent_space_umap(
        z_combined,
        labels,
        label_names,
        title="潜空间UMAP：真实 vs 预测",
        save_path=str(figures_dir / "latent_space_umap.png")
    )

    # 3. 差异基因散点图
    if "de_genes" in all_metrics:
        print("  - 差异基因散点图")
        # 计算log2FC
        eps = 1e-8
        mean_x0 = predictions["x0"].mean(axis=0) + eps
        mean_x1_true = predictions["x1_true"].mean(axis=0) + eps
        mean_x1_pred = predictions["x1_pred"].mean(axis=0) + eps

        log2fc_true = np.log2(mean_x1_true / mean_x0)
        log2fc_pred = np.log2(mean_x1_pred / mean_x0)

        plot_de_genes_scatter(
            log2fc_true,
            log2fc_pred,
            top_k=50,
            save_path=str(figures_dir / "de_genes_scatter.png")
        )

    # 4. 谱范数直方图
    print("  - 谱范数直方图")
    plot_spectral_norm_histogram(
        predictions["spectral_norms"],
        max_norm_threshold=1.05,
        save_path=str(figures_dir / "spectral_norm_histogram.png")
    )

    print(f"\n所有图表已保存至: {figures_dir}")


def print_metrics_summary(all_metrics):
    """打印指标摘要"""
    print("\n" + "=" * 80)
    print("评估结果摘要")
    print("=" * 80)

    if "reconstruction" in all_metrics:
        print("\n【重建质量】")
        recon = all_metrics["reconstruction"]
        print(f"  MSE:             {recon['mse']:.4f}")
        print(f"  Pearson (mean):  {recon['pearson_mean']:.4f}")
        print(f"  R² score:        {recon['r2_score']:.4f}")

    if "distribution" in all_metrics:
        print("\n【分布匹配】")
        dist = all_metrics["distribution"]
        print(f"  E-distance:      {dist['energy_distance']:.4f}")
        print(f"  Mean L2 dist:    {dist['mean_l2_dist']:.4f}")

    if "de_genes" in all_metrics:
        print("\n【差异基因预测】")
        de = all_metrics["de_genes"]
        print(f"  AUROC:           {de['auroc']:.4f}")
        print(f"  AUPRC:           {de['auprc']:.4f}")
        print(f"  Jaccard:         {de['jaccard']:.4f}")
        print(f"  Rank corr:       {de['rank_corr']:.4f}")

    if "operator" in all_metrics:
        print("\n【算子质量】")
        op = all_metrics["operator"]
        print(f"  Spectral norm (mean): {op['spectral_norm_mean']:.4f}")
        print(f"  Spectral norm (max):  {op['spectral_norm_max']:.4f}")
        print(f"  Response sparsity:    {op['response_sparsity']:.4f}")

    print("=" * 80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="扰动预测评估脚本")

    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        required=True,
        help="VAE检查点路径"
    )
    parser.add_argument(
        "--operator_checkpoint",
        type=str,
        required=True,
        help="算子检查点路径"
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
        help="测试数据路径（h5ad格式）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/experiments/scperturb_evaluation/",
        help="输出目录"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="设备（cuda或cpu）"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="批次大小"
    )
    parser.add_argument(
        "--no_de_metrics",
        action="store_true",
        help="不计算差异基因指标（加快评估速度）"
    )

    args = parser.parse_args()

    # 检查文件是否存在
    for path in [args.vae_checkpoint, args.operator_checkpoint, args.data_path]:
        if not Path(path).exists():
            print(f"错误: 文件不存在: {path}")
            return

    # 默认编码器路径
    if not args.encoder_checkpoint:
        args.encoder_checkpoint = str(Path(args.operator_checkpoint).parent / "cond_encoder.pt")

    if not Path(args.encoder_checkpoint).exists():
        print(f"错误: 条件编码器不存在: {args.encoder_checkpoint}")
        return

    # 加载模型
    vae_model, operator_model, cond_encoder, tissue2idx = load_models(
        args.vae_checkpoint,
        args.operator_checkpoint,
        args.encoder_checkpoint,
        args.device
    )

    # 加载测试数据
    print(f"\n加载测试数据: {args.data_path}")
    adata_test = sc.read_h5ad(args.data_path)
    print(f"测试集: {adata_test.n_obs} 细胞, {adata_test.n_vars} 基因")

    # 创建数据集
    test_dataset = SCPerturbPairDataset(adata_test, cond_encoder, tissue2idx)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn_pair
    )
    print(f"测试配对数: {len(test_dataset)}")

    # 评估模型
    all_metrics, predictions = evaluate_model(
        vae_model,
        operator_model,
        test_loader,
        args.device,
        compute_de_metrics=not args.no_de_metrics
    )

    # 保存结果
    output_dir = Path(args.output_dir)
    save_results(all_metrics, predictions, output_dir)

    # 生成可视化
    generate_visualizations(all_metrics, predictions, output_dir)

    # 打印摘要
    print_metrics_summary(all_metrics)

    print(f"\n评估完成！所有结果已保存至: {args.output_dir}")


if __name__ == "__main__":
    main()
