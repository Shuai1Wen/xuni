# -*- coding: utf-8 -*-
"""
配置系统

本模块定义项目所有配置的数据类结构。

对应文档：
- suanfa.md 第21-53行
- details.md 配置文件部分
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class NumericalConfig:
    """
    数值稳定性配置

    集中管理项目中所有数值计算的epsilon和容差参数。

    参数:
        eps_distance: 距离计算的epsilon (默认1e-8)
            - 用于：pairwise_distances中的sqrt
            - 防止：梯度在0处不稳定

        eps_division: 除法运算的epsilon (默认1e-8)
            - 用于：归一化、Pearson系数等
            - 防止：除以零错误

        eps_log: 对数计算的epsilon (默认1e-8)
            - 用于：NB likelihood、KL散度等
            - 防止：log(0)错误

        eps_model_output: 模型输出的下界 (默认1e-8)
            - 用于：VAE解码器的mu输出
            - 防止：极端情况下的数值问题

        tol_test: 测试验证的容差 (默认1e-6)
            - 用于：数学性质验证（对称性、同一性等）
            - 判断：两个浮点数是否"相等"

    示例:
        >>> num_cfg = NumericalConfig()
        >>> print(num_cfg.eps_distance)
        1e-08

        >>> # 自定义配置
        >>> num_cfg = NumericalConfig(eps_distance=1e-10, tol_test=1e-5)
    """
    eps_distance: float = 1e-8
    eps_division: float = 1e-8
    eps_log: float = 1e-8
    eps_model_output: float = 1e-8
    tol_test: float = 1e-6


@dataclass
class ModelConfig:
    """
    模型配置参数

    定义潜空间模型和算子模型的核心超参数。

    参数:
        n_genes: 基因数量 G
        latent_dim: 潜空间维度 d_z，对应 model.md A.2节中的z ∈ ℝ^{d_z}
        n_response_bases: 响应基数量 K，对应 model.md A.5.1节中的求和上限
        max_spectral_norm: 最大谱范数阈值 ρ₀，对应 model.md A.7.1节
        use_generator_view: 是否使用生成元视角（连续时间），对应 model.md A.7.2节

    示例:
        >>> config = ModelConfig(n_genes=2000, latent_dim=32)
        >>> print(config.latent_dim)
        32
    """
    n_genes: int
    latent_dim: int = 32
    n_response_bases: int = 4  # K
    max_spectral_norm: float = 1.05
    use_generator_view: bool = False


@dataclass
class TrainingConfig:
    """
    训练配置参数

    定义训练过程的超参数，包括学习率、批次大小、损失权重等。

    参数:
        lr_embed: 潜空间模型学习率
        lr_operator: 算子模型学习率
        batch_size: 批次大小
        n_epochs_embed: 潜空间模型训练轮数
        n_epochs_operator: 算子模型训练轮数
        lambda_e: E-distance损失权重 λ₁，对应 model.md A.6节公式
        lambda_stab: 稳定性正则化权重 λ₂，对应 model.md A.6节公式
        gradient_clip: 梯度裁剪阈值，用于防止梯度爆炸
        beta_kl: KL散度权重系数，用于β-VAE变体
        warmup_epochs: 学习率预热轮数

    示例:
        >>> config = TrainingConfig(lr_embed=1e-3, batch_size=512)
        >>> print(config.lambda_e)
        1.0
    """
    lr_embed: float = 1e-3
    lr_operator: float = 1e-3
    batch_size: int = 512
    n_epochs_embed: int = 100
    n_epochs_operator: int = 100
    lambda_e: float = 1.0      # weight for E-distance
    lambda_stab: float = 1e-3  # weight for stability regularization
    gradient_clip: float = 1.0
    beta_kl: float = 1.0
    warmup_epochs: int = 0


@dataclass
class ConditionMeta:
    """
    条件元信息

    描述一个条件向量θ的元信息结构，用于索引和编码。
    对应 model.md A.1节中的条件向量定义。

    参数:
        dataset_id: 数据集标识符
        tissue: 组织类型，如 'blood', 'kidney', 'brain' 等，对应 t ∈ T
        perturbation: 扰动类型，如药物名称、CRISPR靶点、'LOY'、'control'，对应 p ∈ P
        timepoint: 时间点索引，如 't0', 't1'，对应 s ∈ {0,1}
        donor_id: 个体ID（可选）
        mLOY_load: 个体级mLOY负荷（可选），范围[0,1]，对应 model.md A.1节中的m
        batch: 批次标识符（可选）
        age: 年龄（可选）
        disease_state: 疾病状态（可选）

    示例:
        >>> cond = ConditionMeta(
        ...     dataset_id="scperturb_001",
        ...     tissue="kidney",
        ...     perturbation="drug_A",
        ...     timepoint="t1"
        ... )
        >>> print(cond.tissue)
        kidney
    """
    dataset_id: str
    tissue: str           # 'blood', 'kidney', 'brain', ...
    perturbation: str     # drug / KO / 'LOY' / 'control'
    timepoint: str        # 't0', 't1'
    donor_id: Optional[str] = None
    mLOY_load: Optional[float] = None  # donor-level mLOY, range [0, 1]
    batch: Optional[str] = None
    age: Optional[int] = None
    disease_state: Optional[str] = None


@dataclass
class DataConfig:
    """
    数据配置参数

    定义数据加载和预处理的参数。

    参数:
        data_path: 数据文件路径
        min_cells: 最小细胞数过滤阈值
        min_genes: 最小基因数过滤阈值
        highly_variable_genes: 是否只使用高变基因
        n_top_genes: 高变基因数量
        normalize: 是否进行归一化
        log_transform: 是否进行log1p变换
        scale: 是否进行标准化

    示例:
        >>> config = DataConfig(
        ...     data_path="data/processed/scperturb/scperturb_merged.h5ad",
        ...     n_top_genes=2000
        ... )
    """
    data_path: str
    min_cells: int = 100
    min_genes: int = 200
    highly_variable_genes: bool = True
    n_top_genes: int = 2000
    normalize: bool = True
    log_transform: bool = True
    scale: bool = True


@dataclass
class ExperimentConfig:
    """
    实验配置

    组合所有子配置，形成完整的实验配置。

    参数:
        model: 模型配置
        training: 训练配置
        data: 数据配置
        experiment_name: 实验名称
        seed: 随机种子
        device: 计算设备 ('cuda' 或 'cpu')
        num_workers: 数据加载并行工作进程数
        save_checkpoints: 是否保存检查点
        checkpoint_freq: 检查点保存频率（每N个epoch）
        log_freq: 日志记录频率（每N个step）

    示例:
        >>> exp_config = ExperimentConfig(
        ...     model=ModelConfig(n_genes=2000),
        ...     training=TrainingConfig(),
        ...     data=DataConfig(data_path="..."),
        ...     experiment_name="scperturb_baseline"
        ... )
    """
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    experiment_name: str = "default"
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 4
    save_checkpoints: bool = True
    checkpoint_freq: int = 10
    log_freq: int = 100


def set_seed(seed: int) -> None:
    """
    设置全局随机种子，确保结果可重复

    参数:
        seed: 随机种子

    功能:
        - 设置Python随机种子
        - 设置NumPy随机种子
        - 设置PyTorch随机种子（CPU和CUDA）
        - 设置PyTorch后端为确定性模式

    示例:
        >>> set_seed(42)
    """
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 设置为确定性模式（可能影响性能）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# 导出的公共接口
__all__ = [
    "NumericalConfig",
    "ModelConfig",
    "TrainingConfig",
    "ConditionMeta",
    "set_seed",
]
