# -*- coding: utf-8 -*-
"""
可视化工具模块

提供实验结果的可视化功能，包括潜空间、训练曲线、基因表达等。

主要功能:
1. 潜空间可视化（UMAP/t-SNE）
2. 训练曲线绘制
3. 响应轮廓热图
4. 基因表达对比图
5. 差异基因散点图
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import torch
from pathlib import Path

# 设置matplotlib中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_latent_space_umap(
    z: np.ndarray,
    labels: np.ndarray,
    label_names: Optional[Dict[int, str]] = None,
    title: str = "潜空间UMAP可视化",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    **umap_kwargs
):
    """
    绘制潜空间的UMAP可视化

    参数:
        z: (n, d) 潜变量
        labels: (n,) 标签（如扰动类型、组织类型）
        label_names: 标签索引→名称的映射
        title: 图表标题
        save_path: 保存路径
        figsize: 图表大小
        **umap_kwargs: 传递给UMAP的额外参数

    示例:
        >>> z = np.random.randn(1000, 32)
        >>> labels = np.random.randint(0, 5, 1000)
        >>> label_names = {0: "对照", 1: "药物A", 2: "药物B", 3: "基因敲除", 4: "mLOY"}
        >>> plot_latent_space_umap(z, labels, label_names, save_path="results/umap.png")
    """
    try:
        import umap
    except ImportError:
        print("警告: umap-learn未安装，无法绘制UMAP图。请安装: pip install umap-learn")
        return

    # 默认UMAP参数
    default_umap_params = {
        "n_neighbors": 15,
        "min_dist": 0.1,
        "metric": "euclidean",
        "random_state": 42
    }
    default_umap_params.update(umap_kwargs)

    # 运行UMAP
    print(f"运行UMAP降维 (n_samples={z.shape[0]}, n_features={z.shape[1]})...")
    reducer = umap.UMAP(**default_umap_params)
    z_umap = reducer.fit_transform(z)

    # 绘图
    fig, ax = plt.subplots(figsize=figsize)
    unique_labels = np.unique(labels)

    for label in unique_labels:
        mask = labels == label
        label_name = label_names.get(label, f"标签{label}") if label_names else f"标签{label}"
        ax.scatter(
            z_umap[mask, 0],
            z_umap[mask, 1],
            label=label_name,
            alpha=0.6,
            s=20
        )

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"UMAP图已保存至: {save_path}")

    plt.close()


def plot_training_curves(
    history: Dict[str, List[float]],
    title: str = "训练曲线",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4)
):
    """
    绘制训练曲线（损失、指标随epoch变化）

    参数:
        history: 训练历史字典，键如"train_loss", "val_loss", "train_pearson"等
        title: 图表标题
        save_path: 保存路径
        figsize: 图表大小

    示例:
        >>> history = {
        ...     "train_loss": [1.5, 1.2, 1.0, 0.9, 0.8],
        ...     "val_loss": [1.6, 1.3, 1.1, 1.0, 0.95],
        ...     "train_pearson": [0.5, 0.6, 0.7, 0.75, 0.8]
        ... }
        >>> plot_training_curves(history, save_path="results/training_curves.png")
    """
    # 分离损失和指标
    loss_keys = [k for k in history.keys() if "loss" in k.lower()]
    metric_keys = [k for k in history.keys() if "loss" not in k.lower()]

    n_plots = (1 if loss_keys else 0) + (1 if metric_keys else 0)
    if n_plots == 0:
        print("警告: history中没有数据可绘制")
        return

    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    # 绘制损失曲线
    if loss_keys:
        ax = axes[plot_idx]
        for key in loss_keys:
            epochs = range(1, len(history[key]) + 1)
            ax.plot(epochs, history[key], label=key, marker='o', markersize=3)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("损失曲线", fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # 绘制指标曲线
    if metric_keys:
        ax = axes[plot_idx]
        for key in metric_keys:
            epochs = range(1, len(history[key]) + 1)
            ax.plot(epochs, history[key], label=key, marker='o', markersize=3)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Metric", fontsize=12)
        ax.set_title("评估指标曲线", fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存至: {save_path}")

    plt.close()


def plot_response_heatmap(
    alpha_matrix: np.ndarray,
    condition_names: Optional[List[str]] = None,
    response_names: Optional[List[str]] = None,
    title: str = "响应系数热图 (α_k)",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = "RdBu_r"
):
    """
    绘制响应系数α_k的热图

    用于分析不同条件下各响应基的激活模式。

    对应：model.md A.5节（低秩分解）

    参数:
        alpha_matrix: (n_conditions, K) 响应系数矩阵
        condition_names: 条件名称列表
        response_names: 响应基名称列表
        title: 图表标题
        save_path: 保存路径
        figsize: 图表大小
        cmap: 颜色映射

    示例:
        >>> alpha = np.random.randn(50, 5)  # 50个条件，5个响应基
        >>> conditions = [f"条件{i}" for i in range(50)]
        >>> responses = [f"响应基{k}" for k in range(5)]
        >>> plot_response_heatmap(alpha, conditions, responses, save_path="results/response_heatmap.png")
    """
    n_cond, K = alpha_matrix.shape

    if condition_names is None:
        condition_names = [f"Cond_{i}" for i in range(n_cond)]
    if response_names is None:
        response_names = [f"B_{k}" for k in range(K)]

    fig, ax = plt.subplots(figsize=figsize)

    # 绘制热图
    im = ax.imshow(alpha_matrix, aspect='auto', cmap=cmap, interpolation='nearest')

    # 设置刻度
    ax.set_xticks(np.arange(K))
    ax.set_yticks(np.arange(n_cond))
    ax.set_xticklabels(response_names, fontsize=10)
    ax.set_yticklabels(condition_names, fontsize=8)

    # 旋转x轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("系数值", fontsize=12)

    ax.set_xlabel("响应基索引", fontsize=12)
    ax.set_ylabel("条件索引", fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"响应热图已保存至: {save_path}")

    plt.close()


def plot_gene_expression_comparison(
    x_true: np.ndarray,
    x_pred: np.ndarray,
    gene_names: Optional[List[str]] = None,
    top_n_genes: int = 20,
    title: str = "基因表达对比（真实 vs 预测）",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6)
):
    """
    绘制基因表达对比图（真实vs预测）

    选取方差最大的top N基因，绘制小提琴图对比。

    参数:
        x_true: (n_cells, n_genes) 真实表达
        x_pred: (n_cells, n_genes) 预测表达
        gene_names: 基因名称列表
        top_n_genes: 展示的top基因数量
        title: 图表标题
        save_path: 保存路径
        figsize: 图表大小

    示例:
        >>> x_true = np.random.randn(1000, 2000)
        >>> x_pred = x_true + np.random.randn(1000, 2000) * 0.2
        >>> gene_names = [f"Gene{i}" for i in range(2000)]
        >>> plot_gene_expression_comparison(x_true, x_pred, gene_names, top_n_genes=10)
    """
    n_cells, n_genes = x_true.shape

    if gene_names is None:
        gene_names = [f"Gene_{i}" for i in range(n_genes)]

    # 选择方差最大的基因
    gene_vars = x_true.var(axis=0)
    top_gene_idx = np.argsort(gene_vars)[-top_n_genes:]

    # 准备数据
    data_list = []
    for idx in top_gene_idx:
        gene_name = gene_names[idx]
        # 真实表达
        for val in x_true[:, idx]:
            data_list.append({"Gene": gene_name, "Type": "真实", "Expression": val})
        # 预测表达
        for val in x_pred[:, idx]:
            data_list.append({"Gene": gene_name, "Type": "预测", "Expression": val})

    import pandas as pd
    df = pd.DataFrame(data_list)

    # 绘制小提琴图
    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(data=df, x="Gene", y="Expression", hue="Type", split=True, ax=ax)

    ax.set_xlabel("基因", fontsize=12)
    ax.set_ylabel("表达水平", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    ax.legend(title="类型", fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"基因表达对比图已保存至: {save_path}")

    plt.close()


def plot_de_genes_scatter(
    log2fc_true: np.ndarray,
    log2fc_pred: np.ndarray,
    gene_names: Optional[List[str]] = None,
    top_k: int = 50,
    title: str = "差异基因log2FC对比",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 10)
):
    """
    绘制差异基因的log2FC散点图（真实vs预测）

    对应：model.md A.9节（差异表达基因评估）

    参数:
        log2fc_true: (n_genes,) 真实log2 fold change
        log2fc_pred: (n_genes,) 预测log2 fold change
        gene_names: 基因名称列表
        top_k: 标注top k个差异基因
        title: 图表标题
        save_path: 保存路径
        figsize: 图表大小

    示例:
        >>> log2fc_true = np.random.randn(2000)
        >>> log2fc_pred = log2fc_true + np.random.randn(2000) * 0.5
        >>> plot_de_genes_scatter(log2fc_true, log2fc_pred, top_k=20)
    """
    n_genes = len(log2fc_true)

    if gene_names is None:
        gene_names = [f"Gene_{i}" for i in range(n_genes)]

    # 计算相关系数
    from scipy.stats import pearsonr, spearmanr
    pearson_r, _ = pearsonr(log2fc_true, log2fc_pred)
    spearman_r, _ = spearmanr(log2fc_true, log2fc_pred)

    # 绘制散点图
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(log2fc_true, log2fc_pred, alpha=0.3, s=20, c='steelblue')

    # 绘制对角线（理想情况）
    min_val = min(log2fc_true.min(), log2fc_pred.min())
    max_val = max(log2fc_true.max(), log2fc_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想线 (y=x)')

    # 标注top k差异基因
    de_scores = np.abs(log2fc_true)
    top_indices = np.argsort(de_scores)[-top_k:]

    for idx in top_indices:
        ax.annotate(
            gene_names[idx],
            (log2fc_true[idx], log2fc_pred[idx]),
            fontsize=8,
            alpha=0.7,
            xytext=(5, 5),
            textcoords='offset points'
        )

    ax.set_xlabel("真实 log2FC", fontsize=12)
    ax.set_ylabel("预测 log2FC", fontsize=12)
    ax.set_title(f"{title}\nPearson r={pearson_r:.3f}, Spearman r={spearman_r:.3f}", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"DE基因散点图已保存至: {save_path}")

    plt.close()


def plot_spectral_norm_histogram(
    spectral_norms: np.ndarray,
    max_norm_threshold: float = 1.05,
    title: str = "算子谱范数分布",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    绘制算子谱范数的直方图

    用于检查稳定性约束是否得到满足。

    对应：model.md A.7节（稳定性约束）

    参数:
        spectral_norms: (n_operators,) 谱范数数组
        max_norm_threshold: 谱范数上界（垂直线标记）
        title: 图表标题
        save_path: 保存路径
        figsize: 图表大小

    示例:
        >>> norms = np.random.gamma(2, 0.5, 1000)  # 模拟谱范数分布
        >>> plot_spectral_norm_histogram(norms, max_norm_threshold=1.05)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制直方图
    ax.hist(spectral_norms, bins=50, alpha=0.7, color='steelblue', edgecolor='black')

    # 标记阈值线
    ax.axvline(
        max_norm_threshold,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'阈值 = {max_norm_threshold}'
    )

    # 统计信息
    mean_norm = spectral_norms.mean()
    max_norm = spectral_norms.max()
    violate_ratio = (spectral_norms > max_norm_threshold).mean()

    ax.axvline(mean_norm, color='green', linestyle=':', linewidth=2, label=f'均值 = {mean_norm:.3f}')

    # 添加文本注释
    info_text = f"最大值: {max_norm:.3f}\n超出比例: {violate_ratio*100:.1f}%"
    ax.text(
        0.95, 0.95, info_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    ax.set_xlabel("谱范数 ρ(A_θ)", fontsize=12)
    ax.set_ylabel("频数", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"谱范数直方图已保存至: {save_path}")

    plt.close()


def plot_comprehensive_evaluation_report(
    metrics_dict: Dict[str, Dict[str, float]],
    save_dir: str,
    prefix: str = ""
):
    """
    生成全面的评估报告图表

    综合绘制多个子图，展示所有评估指标。

    参数:
        metrics_dict: 评估指标字典（来自comprehensive_evaluation）
        save_dir: 保存目录
        prefix: 文件名前缀

    示例:
        >>> metrics = {
        ...     "reconstruction": {"mse": 0.1, "pearson_mean": 0.8},
        ...     "distribution": {"energy_distance": 0.23},
        ...     "operator": {"spectral_norm_mean": 1.02}
        ... }
        >>> plot_comprehensive_evaluation_report(metrics, "results/figures/", prefix="exp1_")
    """
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    # 创建指标摘要图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("模型评估指标摘要", fontsize=16)

    # 1. 重建质量指标
    if "reconstruction" in metrics_dict:
        ax = axes[0, 0]
        recon_metrics = metrics_dict["reconstruction"]
        metric_names = list(recon_metrics.keys())
        metric_values = list(recon_metrics.values())

        ax.barh(metric_names, metric_values, color='steelblue')
        ax.set_xlabel("值", fontsize=11)
        ax.set_title("重建质量指标", fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')

    # 2. 分布匹配指标
    if "distribution" in metrics_dict:
        ax = axes[0, 1]
        dist_metrics = metrics_dict["distribution"]
        metric_names = list(dist_metrics.keys())
        metric_values = list(dist_metrics.values())

        ax.barh(metric_names, metric_values, color='coral')
        ax.set_xlabel("值", fontsize=11)
        ax.set_title("分布匹配指标", fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')

    # 3. 差异基因指标
    if "de_genes" in metrics_dict:
        ax = axes[1, 0]
        de_metrics = metrics_dict["de_genes"]
        metric_names = list(de_metrics.keys())
        metric_values = list(de_metrics.values())

        ax.barh(metric_names, metric_values, color='seagreen')
        ax.set_xlabel("值", fontsize=11)
        ax.set_title("差异基因预测指标", fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')

    # 4. 算子质量指标
    if "operator" in metrics_dict:
        ax = axes[1, 1]
        op_metrics = metrics_dict["operator"]
        metric_names = list(op_metrics.keys())
        metric_values = list(op_metrics.values())

        ax.barh(metric_names, metric_values, color='mediumpurple')
        ax.set_xlabel("值", fontsize=11)
        ax.set_title("算子质量指标", fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    save_path = save_dir_path / f"{prefix}evaluation_summary.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"评估摘要图已保存至: {save_path}")

    plt.close()
