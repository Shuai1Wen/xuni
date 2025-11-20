# -*- coding: utf-8 -*-
"""
评估指标模块

提供完整的评估指标计算函数，用于评估模型性能。

对应：model.md A.9节（评估指标）

指标类别：
1. 重建质量指标：评估VAE的重建能力
2. 分布级别指标：评估算子的分布匹配质量
3. 差异基因预测指标：评估生物学相关性
4. 算子质量指标：评估算子模型的内在质量
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr, pearsonr
from ..utils.edistance import energy_distance


def reconstruction_metrics(
    x_true: torch.Tensor,
    x_pred: torch.Tensor
) -> Dict[str, float]:
    """
    计算重建质量指标

    评估VAE的重建能力，用于验证潜空间是否保留足够信息。

    参数:
        x_true: (B, G) 真实基因表达
        x_pred: (B, G) 预测基因表达

    返回:
        metrics: 字典，包含：
            - mse: 均方误差
            - mae: 平均绝对误差
            - pearson_mean: Pearson相关系数（gene-wise平均）
            - pearson_median: Pearson相关系数中位数
            - spearman_mean: Spearman秩相关（gene-wise平均）
            - r2_score: R²分数

    示例:
        >>> x_true = torch.randn(1000, 2000)
        >>> x_pred = x_true + torch.randn(1000, 2000) * 0.1
        >>> metrics = reconstruction_metrics(x_true, x_pred)
        >>> print(f"Pearson correlation: {metrics['pearson_mean']:.3f}")
        Pearson correlation: 0.912
    """
    x_true_np = x_true.cpu().numpy()
    x_pred_np = x_pred.cpu().numpy()

    B, G = x_true.shape

    # MSE和MAE
    mse = float(((x_true - x_pred) ** 2).mean())
    mae = float(torch.abs(x_true - x_pred).mean())

    # Gene-wise Pearson相关
    pearson_corrs = []
    for g in range(G):
        if x_true_np[:, g].std() > 1e-8 and x_pred_np[:, g].std() > 1e-8:
            corr, _ = pearsonr(x_true_np[:, g], x_pred_np[:, g])
            if not np.isnan(corr):
                pearson_corrs.append(corr)

    # Gene-wise Spearman相关
    spearman_corrs = []
    for g in range(G):
        if len(np.unique(x_true_np[:, g])) > 1 and len(np.unique(x_pred_np[:, g])) > 1:
            corr, _ = spearmanr(x_true_np[:, g], x_pred_np[:, g])
            if not np.isnan(corr):
                spearman_corrs.append(corr)

    # R² score
    ss_res = ((x_true - x_pred) ** 2).sum()
    ss_tot = ((x_true - x_true.mean()) ** 2).sum()
    r2 = float(1 - ss_res / (ss_tot + 1e-8))

    return {
        "mse": mse,
        "mae": mae,
        "pearson_mean": float(np.mean(pearson_corrs)) if pearson_corrs else 0.0,
        "pearson_median": float(np.median(pearson_corrs)) if pearson_corrs else 0.0,
        "spearman_mean": float(np.mean(spearman_corrs)) if spearman_corrs else 0.0,
        "r2_score": r2
    }


def distribution_metrics(
    z_true: torch.Tensor,
    z_pred: torch.Tensor,
    use_energy_distance: bool = True
) -> Dict[str, float]:
    """
    计算分布级别指标

    评估算子模型的分布匹配质量，核心指标是E-distance。

    对应：model.md A.4节（E-distance定义）

    参数:
        z_true: (n, d) 真实潜变量分布
        z_pred: (m, d) 预测潜变量分布
        use_energy_distance: 是否计算E-distance（可能耗时）

    返回:
        metrics: 字典，包含：
            - energy_distance: E-distance（主要指标）
            - mean_l2_dist: 均值的L2距离
            - cov_frobenius_dist: 协方差矩阵的Frobenius距离

    示例:
        >>> z_true = torch.randn(1000, 32)
        >>> z_pred = torch.randn(1000, 32) + 0.5
        >>> metrics = distribution_metrics(z_true, z_pred)
        >>> print(f"E-distance: {metrics['energy_distance']:.4f}")
        E-distance: 0.2341
    """
    metrics = {}

    # E-distance（核心指标）
    if use_energy_distance:
        ed2 = energy_distance(z_pred, z_true)
        metrics["energy_distance"] = float(ed2)

    # 均值距离
    mean_true = z_true.mean(dim=0)
    mean_pred = z_pred.mean(dim=0)
    mean_dist = torch.norm(mean_true - mean_pred).item()
    metrics["mean_l2_dist"] = mean_dist

    # 协方差距离（防止除零：当batch_size=1时，使用1代替0）
    z_true_centered = z_true - mean_true
    z_pred_centered = z_pred - mean_pred
    n_true = max(z_true.shape[0] - 1, 1)
    n_pred = max(z_pred.shape[0] - 1, 1)
    cov_true = (z_true_centered.T @ z_true_centered) / n_true
    cov_pred = (z_pred_centered.T @ z_pred_centered) / n_pred
    cov_dist = torch.norm(cov_true - cov_pred, p='fro').item()
    metrics["cov_frobenius_dist"] = cov_dist

    return metrics


def de_gene_prediction_metrics(
    x0: torch.Tensor,
    x1_true: torch.Tensor,
    x1_pred: torch.Tensor,
    top_k: int = 200,
    eps: float = 1e-8
) -> Dict[str, float]:
    """
    计算差异基因预测指标

    评估模型是否能够捕获真实的差异表达基因（DE genes）。
    这是生物学验证的关键指标。

    对应：model.md A.9节（生物学验证）

    参数:
        x0: (B, G) 对照（t0）基因表达
        x1_true: (B, G) 真实处理（t1）基因表达
        x1_pred: (B, G) 预测处理基因表达
        top_k: 取top k个差异基因
        eps: 数值稳定性epsilon

    返回:
        metrics: 字典，包含：
            - auroc: DE基因二分类AUROC
            - auprc: DE基因二分类AUPRC
            - jaccard: top_k基因集合的Jaccard相似度
            - rank_corr: DE分数排名的Spearman相关
            - mean_log2fc_corr: 平均log2FC的Pearson相关

    流程:
        1. 计算真实DE: log2FC(x1_true / x0)
        2. 计算预测DE: log2FC(x1_pred / x0)
        3. 排序基因：取top_k
        4. 计算AUROC、AUPRC、Jaccard相似度

    示例:
        >>> x0 = torch.randn(1000, 2000).abs()
        >>> x1_true = x0 + torch.randn(1000, 2000) * 0.5
        >>> x1_pred = x0 + torch.randn(1000, 2000) * 0.4
        >>> metrics = de_gene_prediction_metrics(x0, x1_true, x1_pred, top_k=200)
        >>> print(f"AUROC: {metrics['auroc']:.3f}")
        AUROC: 0.834
    """
    # 转为numpy
    x0_np = x0.cpu().numpy()
    x1_true_np = x1_true.cpu().numpy()
    x1_pred_np = x1_pred.cpu().numpy()

    B, G = x0.shape

    # 计算log2 fold change（平均across细胞）
    # 注意：在log内部添加pseudocount避免引入bias
    mean_x0 = x0_np.mean(axis=0)
    mean_x1_true = x1_true_np.mean(axis=0)
    mean_x1_pred = x1_pred_np.mean(axis=0)

    # 在除法和log操作中同时添加eps，保持比例关系
    log2fc_true = np.log2((mean_x1_true + eps) / (mean_x0 + eps))
    log2fc_pred = np.log2((mean_x1_pred + eps) / (mean_x0 + eps))

    # 计算DE分数（这里使用|log2FC|）
    de_score_true = np.abs(log2fc_true)
    de_score_pred = np.abs(log2fc_pred)

    # 构建二分类标签：top_k为正例
    top_k_genes_true = set(np.argsort(de_score_true)[-top_k:])
    top_k_genes_pred = set(np.argsort(de_score_pred)[-top_k:])

    # 计算Jaccard
    if len(top_k_genes_true | top_k_genes_pred) > 0:
        jaccard = len(top_k_genes_true & top_k_genes_pred) / len(top_k_genes_true | top_k_genes_pred)
    else:
        jaccard = 0.0

    # AUROC和AUPRC：用真实top_k作为标签，预测分数作为score
    y_true_binary = np.zeros(G)
    y_true_binary[list(top_k_genes_true)] = 1

    try:
        auroc = roc_auc_score(y_true_binary, de_score_pred)
    except ValueError:
        auroc = 0.5  # 如果只有一个类，默认0.5

    try:
        auprc = average_precision_score(y_true_binary, de_score_pred)
    except ValueError:
        auprc = np.sum(y_true_binary) / len(y_true_binary)

    # 排名相关性
    rank_corr, _ = spearmanr(de_score_true, de_score_pred)
    if np.isnan(rank_corr):
        rank_corr = 0.0

    # log2FC相关性
    log2fc_corr, _ = pearsonr(log2fc_true, log2fc_pred)
    if np.isnan(log2fc_corr):
        log2fc_corr = 0.0

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "jaccard": float(jaccard),
        "rank_corr": float(rank_corr),
        "mean_log2fc_corr": float(log2fc_corr)
    }


def operator_quality_metrics(
    operator_model,
    tissue_idx: torch.Tensor,
    cond_vec: torch.Tensor,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    计算算子质量指标

    评估算子模型的内在质量，如谱范数、稀疏度等。

    对应：model.md A.7节（稳定性约束）

    参数:
        operator_model: OperatorModel实例
        tissue_idx: (B,) 组织索引
        cond_vec: (B, cond_dim) 条件向量
        device: 设备

    返回:
        metrics: 字典，包含：
            - spectral_norm_mean: 谱范数均值
            - spectral_norm_max: 谱范数最大值
            - spectral_norm_std: 谱范数标准差
            - response_sparsity: 响应系数α_k的稀疏度（非零比例）
            - response_magnitude: 响应系数的平均幅值

    示例:
        >>> from src.models.operator import OperatorModel
        >>> operator = OperatorModel(latent_dim=32, n_tissues=3, n_response_bases=5, cond_dim=64)
        >>> tissue_idx = torch.tensor([0, 1, 2])
        >>> cond_vec = torch.randn(3, 64)
        >>> metrics = operator_quality_metrics(operator, tissue_idx, cond_vec)
        >>> print(f"Spectral norm: {metrics['spectral_norm_mean']:.3f}")
        Spectral norm: 1.023
    """
    operator_model.to(device)
    tissue_idx = tissue_idx.to(device)
    cond_vec = cond_vec.to(device)

    # 计算谱范数
    norms = operator_model.compute_operator_norm(
        tissue_idx, cond_vec, norm_type="spectral"
    )  # (B,)

    # 响应系数分析
    alpha, beta = operator_model.get_response_profile(cond_vec)  # (B, K), (B, K)

    # 稀疏度：非零元素比例（阈值1e-3）
    alpha_sparsity = (torch.abs(alpha) > 1e-3).float().mean().item()

    # 幅值统计
    alpha_magnitude = torch.abs(alpha).mean().item()

    return {
        "spectral_norm_mean": float(norms.mean()),
        "spectral_norm_max": float(norms.max()),
        "spectral_norm_min": float(norms.min()),
        "spectral_norm_std": float(norms.std()),
        "response_sparsity": float(alpha_sparsity),
        "response_magnitude": float(alpha_magnitude)
    }


def comprehensive_evaluation(
    vae_model,
    operator_model,
    dataloader,
    device: str = "cuda",
    compute_de_metrics: bool = True,
    top_k: int = 200
) -> Dict[str, Dict[str, float]]:
    """
    全面评估模型性能

    在给定数据集上计算所有评估指标。

    参数:
        vae_model: VAE模型
        operator_model: 算子模型
        dataloader: 数据加载器（返回配对数据）
        device: 设备
        compute_de_metrics: 是否计算DE基因指标（耗时）
        top_k: DE基因的top k数量

    返回:
        all_metrics: 嵌套字典，包含：
            - "reconstruction": 重建质量指标
            - "distribution": 分布匹配指标
            - "de_genes": 差异基因指标（可选）
            - "operator": 算子质量指标

    示例:
        >>> all_metrics = comprehensive_evaluation(
        ...     vae_model, operator_model, test_loader, device="cuda"
        ... )
        >>> print(f"Overall Pearson: {all_metrics['reconstruction']['pearson_mean']:.3f}")
        >>> print(f"Overall E-distance: {all_metrics['distribution']['energy_distance']:.4f}")
    """
    vae_model.eval()
    operator_model.eval()
    vae_model.to(device)
    operator_model.to(device)

    all_x0 = []
    all_x1_true = []
    all_x1_pred = []
    all_z1_true = []
    all_z1_pred = []
    all_tissue_idx = []
    all_cond_vec = []

    with torch.no_grad():
        for batch in dataloader:
            x0 = batch["x0"].to(device)
            x1 = batch["x1"].to(device)
            tissue_idx = batch["tissue_idx"].to(device)
            cond_vec = batch["cond_vec"].to(device)

            # 转换tissue_idx为one-hot编码
            tissue_onehot = F.one_hot(tissue_idx, num_classes=vae_model.n_tissues).float()

            # 编码 x0 → z0
            mu0, _ = vae_model.encoder(x0, tissue_onehot)
            z0 = mu0  # 使用均值

            # 应用算子 z0 → z1_pred（operator返回3个值）
            z1_pred, _, _ = operator_model(z0, tissue_idx, cond_vec)

            # 解码 z1_pred → x1_pred（decoder.forward返回(mu, r)）
            x1_pred, _ = vae_model.decoder(z1_pred, tissue_onehot)

            # 真实z1（用于分布指标）
            mu1, _ = vae_model.encoder(x1, tissue_onehot)
            z1_true = mu1

            all_x0.append(x0.cpu())
            all_x1_true.append(x1.cpu())
            all_x1_pred.append(x1_pred.cpu())
            all_z1_true.append(z1_true.cpu())
            all_z1_pred.append(z1_pred.cpu())
            all_tissue_idx.append(tissue_idx.cpu())
            all_cond_vec.append(cond_vec.cpu())

    # 拼接所有批次
    x0_all = torch.cat(all_x0, dim=0)
    x1_true_all = torch.cat(all_x1_true, dim=0)
    x1_pred_all = torch.cat(all_x1_pred, dim=0)
    z1_true_all = torch.cat(all_z1_true, dim=0)
    z1_pred_all = torch.cat(all_z1_pred, dim=0)
    tissue_idx_all = torch.cat(all_tissue_idx, dim=0)
    cond_vec_all = torch.cat(all_cond_vec, dim=0)

    # 计算各类指标
    all_metrics = {}

    # 1. 重建质量
    all_metrics["reconstruction"] = reconstruction_metrics(x1_true_all, x1_pred_all)

    # 2. 分布匹配
    all_metrics["distribution"] = distribution_metrics(z1_true_all, z1_pred_all)

    # 3. 差异基因预测（可选，耗时）
    if compute_de_metrics:
        all_metrics["de_genes"] = de_gene_prediction_metrics(
            x0_all, x1_true_all, x1_pred_all, top_k=top_k
        )

    # 4. 算子质量
    all_metrics["operator"] = operator_quality_metrics(
        operator_model, tissue_idx_all.to(device), cond_vec_all.to(device), device=device
    )

    return all_metrics
