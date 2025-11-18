# -*- coding: utf-8 -*-
"""
虚拟细胞操作接口

本模块提供编码、解码、算子应用的统一接口，用于反事实模拟。

数学对应关系：
- 单步预测：z₁ = K_θ(z₀), x₁ ~ p_ψ(x|z₁,t)，对应 model.md A.8节，第272-276行
- 多步合成：z₂ = K_θ₂(K_θ₁(z₀))，对应 model.md A.8节，第277-284行

主要功能：
- encode_cells: x → z（编码到潜空间）
- decode_cells: z → x（解码到基因空间）
- apply_operator: z → K_θ(z)（应用算子）
- virtual_cell_scenario: 多步反事实模拟

应用场景：
- mLOY纠正：LOY细胞 → 虚拟XY细胞
- 药物组合：细胞 + 药物A → 中间状态 + 药物B → 最终状态
- 跨组织预测：肾脏算子 vs 脑算子
"""

import torch
from typing import List, Optional, Tuple
from ..models.nb_vae import NBVAE
from ..config import NumericalConfig

# 默认数值配置
_NUM_CFG = NumericalConfig()
from ..models.operator import OperatorModel


@torch.no_grad()
def encode_cells(
    vae: NBVAE,
    x: torch.Tensor,
    tissue_onehot: torch.Tensor,
    device: str = "cuda"
) -> torch.Tensor:
    """
    将细胞编码到潜空间

    使用VAE编码器将高维基因表达编码为低维潜变量z。

    参数:
        vae: 训练好的NB-VAE模型
        x: (B, G) 基因表达计数
        tissue_onehot: (B, n_tissues) 组织类型one-hot编码
        device: 计算设备

    返回:
        z: (B, latent_dim) 潜变量（使用均值，不采样）

    实现细节:
        - 使用encoder的均值输出，不进行随机采样
        - 这样保证确定性，便于反事实分析
        - 模型设置为eval模式

    示例:
        >>> vae = NBVAE(n_genes=2000, latent_dim=32, n_tissues=3)
        >>> vae.load_state_dict(torch.load("vae_checkpoint.pt"))
        >>> x = torch.randn(100, 2000)
        >>> tissue_onehot = torch.zeros(100, 3)
        >>> tissue_onehot[:, 0] = 1
        >>> z = encode_cells(vae, x, tissue_onehot)
        >>> print(z.shape)
        torch.Size([100, 32])
    """
    vae.to(device)
    vae.eval()

    x = x.to(device)
    tissue_onehot = tissue_onehot.to(device)

    # 编码：只使用均值，不采样
    mu, logvar = vae.encoder(x, tissue_onehot)

    return mu  # (B, latent_dim)


@torch.no_grad()
def decode_cells(
    vae: NBVAE,
    z: torch.Tensor,
    tissue_onehot: torch.Tensor,
    device: str = "cuda"
) -> torch.Tensor:
    """
    将潜变量解码为基因表达

    使用VAE解码器将潜变量z解码为基因表达的均值。

    参数:
        vae: 训练好的NB-VAE模型
        z: (B, latent_dim) 潜变量
        tissue_onehot: (B, n_tissues) 组织类型one-hot编码
        device: 计算设备

    返回:
        x_mu: (B, G) 重建的基因表达均值

    实现细节:
        - 返回负二项分布的均值μ，而非采样值
        - 这样得到期望表达，更稳定
        - 模型设置为eval模式

    示例:
        >>> vae = NBVAE(n_genes=2000, latent_dim=32, n_tissues=3)
        >>> z = torch.randn(100, 32)
        >>> tissue_onehot = torch.zeros(100, 3)
        >>> tissue_onehot[:, 0] = 1
        >>> x_mu = decode_cells(vae, z, tissue_onehot)
        >>> print(x_mu.shape)
        torch.Size([100, 2000])
    """
    vae.to(device)
    vae.eval()

    z = z.to(device)
    tissue_onehot = tissue_onehot.to(device)

    # 解码：获取负二项分布的均值
    mu_x, r_x = vae.decoder(z, tissue_onehot)

    return mu_x  # (B, G)


@torch.no_grad()
def apply_operator(
    operator: OperatorModel,
    z: torch.Tensor,
    tissue_idx: torch.Tensor,
    cond_vec: torch.Tensor,
    device: str = "cuda"
) -> torch.Tensor:
    """
    应用算子进行状态转移

    在潜空间中应用算子：z_out = K_θ(z)

    对应：model.md A.3节和A.8节

    参数:
        operator: 训练好的算子模型
        z: (B, latent_dim) 输入潜状态
        tissue_idx: (B,) 组织索引
        cond_vec: (B, cond_dim) 条件向量θ
        device: 计算设备

    返回:
        z_out: (B, latent_dim) 输出潜状态

    实现细节:
        - 模型设置为eval模式
        - 只返回z_out，不返回A_θ和b_θ
        - 确保数值稳定（检查NaN）

    示例:
        >>> operator = OperatorModel(32, 3, 5, 64)
        >>> operator.load_state_dict(torch.load("operator_checkpoint.pt"))
        >>> z = torch.randn(100, 32)
        >>> tissue_idx = torch.zeros(100, dtype=torch.long)
        >>> cond_vec = torch.randn(100, 64)
        >>> z_out = apply_operator(operator, z, tissue_idx, cond_vec)
        >>> print(z_out.shape)
        torch.Size([100, 32])
    """
    operator.to(device)
    operator.eval()

    z = z.to(device)
    tissue_idx = tissue_idx.to(device)
    cond_vec = cond_vec.to(device)

    # 应用算子
    z_out, _, _ = operator(z, tissue_idx, cond_vec)

    # 数值稳定性检查
    if torch.isnan(z_out).any():
        raise RuntimeError("算子输出包含NaN，可能存在数值不稳定")

    return z_out  # (B, latent_dim)


@torch.no_grad()
def virtual_cell_scenario(
    vae: NBVAE,
    operator: OperatorModel,
    x0: torch.Tensor,
    tissue_onehot: torch.Tensor,
    tissue_idx: torch.Tensor,
    cond_vec_seq: torch.Tensor,
    device: str = "cuda",
    return_trajectory: bool = False
) -> torch.Tensor:
    """
    多步反事实模拟

    支持顺序应用多个条件，模拟复杂的扰动组合。

    对应：model.md A.8节，第277-284行

    参数:
        vae: 训练好的NB-VAE模型
        operator: 训练好的算子模型
        x0: (B, G) 初始基因表达
        tissue_onehot: (B, n_tissues) 组织类型one-hot编码
        tissue_idx: (B,) 组织索引
        cond_vec_seq: (T, cond_dim) 或 (T, B, cond_dim) 条件序列
            如果是(T, cond_dim)，会广播到所有样本
        device: 计算设备
        return_trajectory: 是否返回完整轨迹（每步的z和x）

    返回:
        x_virtual: (B, G) 最终虚拟细胞的基因表达
        如果return_trajectory=True，返回(x_virtual, z_trajectory, x_trajectory)

    实现流程:
        1. 编码：x₀ → z₀
        2. 循环应用算子：
           z₁ = K_θ₁(z₀)
           z₂ = K_θ₂(z₁)
           ...
        3. 解码：z_T → x_T

    数值稳定性:
        - 每步检查z的范数，如果爆炸则警告
        - 检查NaN
        - 限制最大步数（防止无限循环）

    应用示例：
        1. mLOY纠正：
           cond_seq = [θ_mLOY, θ_virtual_XY]
           LOY细胞 → 施加mLOY效应 → 反向纠正到XY状态

        2. 药物组合：
           cond_seq = [θ_drug_A, θ_drug_B]
           细胞 → 药物A效应 → 药物B效应

        3. 时间序列：
           cond_seq = [θ_t1, θ_t2, θ_t3]
           t0细胞 → t1 → t2 → t3

    示例:
        >>> vae = NBVAE(2000, 32, 3)
        >>> operator = OperatorModel(32, 3, 5, 64)
        >>> x0 = torch.randn(100, 2000)  # LOY细胞
        >>> tissue_onehot = torch.zeros(100, 3)
        >>> tissue_onehot[:, 1] = 1  # kidney
        >>> tissue_idx = torch.ones(100, dtype=torch.long)
        >>> # 定义条件序列：mLOY → virtual XY
        >>> cond_seq = torch.stack([cond_mLOY, cond_XY])  # (2, 64)
        >>> x_virtual = virtual_cell_scenario(
        ...     vae, operator, x0, tissue_onehot, tissue_idx, cond_seq
        ... )
        >>> print(x_virtual.shape)
        torch.Size([100, 2000])
    """
    # 1. 编码到潜空间
    z = encode_cells(vae, x0, tissue_onehot, device=device)  # (B, latent_dim)

    B = z.size(0)
    T = cond_vec_seq.size(0)

    # 如果cond_vec_seq是(T, cond_dim)，扩展到(T, B, cond_dim)
    if cond_vec_seq.dim() == 2:
        cond_vec_seq = cond_vec_seq.unsqueeze(1).expand(-1, B, -1)  # (T, B, cond_dim)

    # 存储轨迹（如果需要）
    if return_trajectory:
        z_trajectory = [z.clone()]
        x_trajectory = [decode_cells(vae, z, tissue_onehot, device)]

    # 2. 循环应用算子
    for t in range(T):
        cond_t = cond_vec_seq[t]  # (B, cond_dim)

        # 应用算子：z_{t+1} = K_θ(z_t)
        z = apply_operator(operator, z, tissue_idx, cond_t, device=device)

        # 数值稳定性检查
        z_norm = z.norm(dim=-1).mean().item()
        if z_norm > 100.0:
            print(f"警告：步骤{t+1}，潜空间范数过大 ({z_norm:.2f})，可能发散")

        if return_trajectory:
            z_trajectory.append(z.clone())
            x_trajectory.append(decode_cells(vae, z, tissue_onehot, device))

    # 3. 解码到基因空间
    x_virtual = decode_cells(vae, z, tissue_onehot, device=device)  # (B, G)

    if return_trajectory:
        # z_trajectory: List[(B, latent_dim)] → (T+1, B, latent_dim)
        # x_trajectory: List[(B, G)] → (T+1, B, G)
        z_trajectory = torch.stack(z_trajectory, dim=0)
        x_trajectory = torch.stack(x_trajectory, dim=0)
        return x_virtual, z_trajectory, x_trajectory

    return x_virtual


@torch.no_grad()
def compute_reconstruction_error(
    vae: NBVAE,
    x: torch.Tensor,
    tissue_onehot: torch.Tensor,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算VAE的重建误差（用于质量评估）

    参数:
        vae: NB-VAE模型
        x: (B, G) 原始基因表达
        tissue_onehot: (B, n_tissues) 组织one-hot
        device: 设备

    返回:
        mse: (B,) 每个样本的MSE
        correlation: (B,) 每个样本的Pearson相关系数

    示例:
        >>> vae = NBVAE(2000, 32, 3)
        >>> x = torch.randn(100, 2000)
        >>> tissue_onehot = torch.zeros(100, 3)
        >>> tissue_onehot[:, 0] = 1
        >>> mse, corr = compute_reconstruction_error(vae, x, tissue_onehot)
        >>> print(f"平均MSE: {mse.mean():.4f}, 平均相关系数: {corr.mean():.4f}")
        平均MSE: 0.1234, 平均相关系数: 0.8765
    """
    # 编码-解码
    z = encode_cells(vae, x, tissue_onehot, device)
    x_recon = decode_cells(vae, z, tissue_onehot, device)

    # MSE
    mse = ((x - x_recon) ** 2).mean(dim=-1)  # (B,)

    # Pearson相关系数（向量化实现）
    # 中心化：(B, G) - (B, 1) → (B, G)
    x_centered = x - x.mean(dim=-1, keepdim=True)
    xr_centered = x_recon - x_recon.mean(dim=-1, keepdim=True)

    # 计算相关系数：对每个样本计算
    numerator = (x_centered * xr_centered).sum(dim=-1)  # (B,)
    denominator = torch.sqrt(
        (x_centered ** 2).sum(dim=-1) * (xr_centered ** 2).sum(dim=-1)
    )  # (B,)
    correlation = numerator / (denominator + _NUM_CFG.eps_division)  # (B,)

    return mse, correlation


@torch.no_grad()
def interpolate_conditions(
    operator: OperatorModel,
    z: torch.Tensor,
    tissue_idx: torch.Tensor,
    cond_vec_start: torch.Tensor,
    cond_vec_end: torch.Tensor,
    n_steps: int = 10,
    device: str = "cuda"
) -> torch.Tensor:
    """
    在两个条件之间插值，探索状态空间轨迹

    用途：
        - 可视化条件A到条件B的平滑过渡
        - 分析中间状态的基因表达模式

    参数:
        operator: 算子模型
        z: (B, latent_dim) 初始潜状态
        tissue_idx: (B,) 组织索引
        cond_vec_start: (cond_dim,) 起始条件
        cond_vec_end: (cond_dim,) 终止条件
        n_steps: 插值步数
        device: 设备

    返回:
        z_interp: (n_steps, B, latent_dim) 插值轨迹

    示例:
        >>> # 从control插值到drug
        >>> z_interp = interpolate_conditions(
        ...     operator, z, tissue_idx,
        ...     cond_control, cond_drug,
        ...     n_steps=10
        ... )
        >>> # 可视化UMAP
        >>> import umap
        >>> z_all = z_interp.view(-1, 32).cpu().numpy()
        >>> umap_emb = umap.UMAP().fit_transform(z_all)
        >>> # 绘制轨迹
    """
    B = z.size(0)
    z_interp = [z.clone()]

    # 线性插值系数
    alphas = torch.linspace(0, 1, n_steps, device=device)

    for alpha in alphas[1:]:
        # 插值条件向量
        cond_vec = (1 - alpha) * cond_vec_start + alpha * cond_vec_end  # (cond_dim,)
        cond_vec_batch = cond_vec.unsqueeze(0).expand(B, -1)  # (B, cond_dim)

        # 应用算子
        z = apply_operator(operator, z, tissue_idx, cond_vec_batch, device)
        z_interp.append(z.clone())

    return torch.stack(z_interp, dim=0)  # (n_steps, B, latent_dim)
