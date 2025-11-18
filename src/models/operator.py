# -*- coding: utf-8 -*-
"""
扰动响应算子模型

本模块实现低秩算子族，建模单细胞在不同条件下的状态转移。

数学对应关系：
- 算子定义：K_θ(z) = A_θz + b_θ，对应 model.md A.3节，第77-79行
- 低秩分解（线性部分）：A_θ = A_t^(0) + Σ_k α_k(θ) B_k，对应 model.md A.5.1节，第145-151行
- 低秩分解（平移部分）：b_θ = b_t^(0) + Σ_k β_k(θ) u_k，对应 model.md A.5.3节，第195-200行
- 系数网络：α_k(θ) = g_k(θ; ω)，对应 model.md A.5.2节，第173-191行
- 稳定性约束：R_stab = Σ_θ max(0, ρ(A_θ) - ρ₀)²，对应 model.md A.7.1节，第231-246行

关键特性：
- 低秩结构：共享全局响应基B_k和u_k，条件特异系数α_k(θ)和β_k(θ)
- 参数效率：K个响应基 >> |Θ|个独立算子
- 泛化能力：新条件可通过MLP预测系数
- 稳定性保证：谱范数约束防止潜空间发散
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from ..config import NumericalConfig

# 默认数值配置
_NUM_CFG = NumericalConfig()


class OperatorModel(nn.Module):
    """
    扰动响应算子模型（低秩结构）

    实现算子族：K_θ(z) = A_θz + b_θ，其中：
        A_θ = A_t^(0) + Σ_{k=1}^K α_k(θ) B_k
        b_θ = b_t^(0) + Σ_{k=1}^K β_k(θ) u_k

    对应：model.md A.3节 + A.5节

    参数:
        latent_dim: 潜空间维度 d_z
        n_tissues: 组织类型数量
        n_response_bases: 响应基数量 K
        cond_dim: 条件向量维度
        hidden_dim: α_k和β_k网络的隐藏层维度

    属性:
        A0_tissue: (n_tissues, d, d) 每个组织的基线算子 A_t^(0)
        b0_tissue: (n_tissues, d) 每个组织的基线平移 b_t^(0)
        B: (K, d, d) 全局响应基算子 B_k
        u: (K, d) 全局平移基 u_k
        alpha_mlp: 条件→系数α的神经网络
        beta_mlp: 条件→系数β的神经网络

    设计理由：
        1. 基线算子A_t^(0)：捕捉组织特异的自然演化（如无扰动时的轨迹）
        2. 响应基B_k：捕捉跨组织、跨扰动的共享响应模式
        3. 系数网络：参数化α_k(θ)和β_k(θ)，支持泛化到新条件

    示例:
        >>> model = OperatorModel(
        ...     latent_dim=32,
        ...     n_tissues=3,
        ...     n_response_bases=5,
        ...     cond_dim=64
        ... )
        >>> z = torch.randn(batch, 32)
        >>> tissue_idx = torch.zeros(batch, dtype=torch.long)
        >>> cond_vec = torch.randn(batch, 64)
        >>> z_out, A_theta, b_theta = model(z, tissue_idx, cond_vec)
        >>> print(z_out.shape, A_theta.shape)
        torch.Size([batch, 32]) torch.Size([batch, 32, 32])
    """

    def __init__(
        self,
        latent_dim: int,
        n_tissues: int,
        n_response_bases: int,
        cond_dim: int,
        hidden_dim: int = 64
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_tissues = n_tissues
        self.K = n_response_bases
        self.cond_dim = cond_dim

        # 每个组织的基线算子和偏置
        # A_t^(0)：组织特异的基线转移矩阵
        # 初始化接近单位阵（表示无扰动时状态基本不变）
        self.A0_tissue = nn.Parameter(torch.zeros(n_tissues, latent_dim, latent_dim))
        # 初始化为单位阵
        for t in range(n_tissues):
            self.A0_tissue.data[t] = torch.eye(latent_dim)

        # b_t^(0)：组织特异的基线平移
        # 初始化为0（表示无扰动时无偏移）
        self.b0_tissue = nn.Parameter(torch.zeros(n_tissues, latent_dim))

        # 全局响应基 B_k（线性部分）和 u_k（平移部分）
        # B_k：(K, d, d) 响应基算子
        # 初始化为小随机值，避免初始时响应过强
        self.B = nn.Parameter(torch.randn(self.K, latent_dim, latent_dim) * 0.01)

        # u_k：(K, d) 平移基
        self.u = nn.Parameter(torch.randn(self.K, latent_dim) * 0.01)

        # 用小网络从条件向量 θ 预测 α_k(θ) 和 β_k(θ)
        # α_k(θ)：响应基B_k的激活强度
        self.alpha_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.K)
        )

        # β_k(θ)：平移基u_k的激活强度
        self.beta_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.K)
        )

    def forward(
        self,
        z: torch.Tensor,
        tissue_idx: torch.Tensor,
        cond_vec: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播：应用算子 K_θ(z) = A_θz + b_θ

        参数:
            z: (B, d) 输入潜状态
            tissue_idx: (B,) 每个样本的组织索引（long类型）
            cond_vec: (B, cond_dim) 条件向量θ的编码

        返回:
            z_out: (B, d) 输出潜状态 K_θ(z)
            A_theta: (B, d, d) 每个样本的算子矩阵
            b_theta: (B, d) 每个样本的平移向量

        实现流程:
            1. 计算系数：α_k(θ)和β_k(θ)
            2. 获取基线：A_t^(0)和b_t^(0)
            3. 组合响应基：
               A_θ = A_t^(0) + Σ_k α_k(θ) B_k
               b_θ = b_t^(0) + Σ_k β_k(θ) u_k
            4. 应用算子：z_out = A_θz + b_θ

        数学对应：
            - model.md A.3节：K_θ(z)定义
            - model.md A.5.1节：A_θ的低秩分解
            - model.md A.5.3节：b_θ的低秩分解

        向量化实现关键：
            - 使用torch.bmm进行批量矩阵乘法
            - 使用广播机制组合响应基
            - 所有操作支持变长batch

        示例:
            >>> model = OperatorModel(32, 3, 5, 64)
            >>> z = torch.randn(128, 32)
            >>> tissue_idx = torch.randint(0, 3, (128,))
            >>> cond_vec = torch.randn(128, 64)
            >>> z_out, A, b = model(z, tissue_idx, cond_vec)
            >>> print(z_out.shape)
            torch.Size([128, 32])
        """
        B = z.size(0)  # batch size
        d = self.latent_dim

        # 1. 计算响应基的激活系数
        # α_k(θ): (B, K) - 线性响应基的权重
        alpha = self.alpha_mlp(cond_vec)  # (B, K)

        # β_k(θ): (B, K) - 平移响应基的权重
        beta = self.beta_mlp(cond_vec)    # (B, K)

        # 2. 获取对应组织的基线算子
        # A_t^(0): (B, d, d)
        A0 = self.A0_tissue[tissue_idx]   # (B, d, d)

        # b_t^(0): (B, d)
        b0 = self.b0_tissue[tissue_idx]   # (B, d)

        # 3. 组合响应基构造 A_θ
        # A_θ = A_t^(0) + Σ_{k=1}^K α_k(θ) B_k

        # 使用einsum计算加权求和，避免expand带来的内存开销
        # einsum('bk,kij->bij', alpha, self.B) 等价于：
        #   对每个b: Σ_k α[b,k] * B[k,i,j]
        # 这样避免创建 (B, K, d, d) 的中间张量，节省 5倍内存
        A_res = torch.einsum('bk,kij->bij', alpha, self.B)  # (B, d, d)

        # 最终算子：A_θ = A_t^(0) + Σ α_k B_k
        A_theta = A0 + A_res  # (B, d, d)

        # 4. 组合平移基构造 b_θ
        # b_θ = b_t^(0) + Σ_{k=1}^K β_k(θ) u_k

        # 使用einsum计算加权求和，避免expand带来的内存开销
        # einsum('bk,ki->bi', beta, self.u) 等价于：
        #   对每个b: Σ_k β[b,k] * u[k,i]
        b_res = torch.einsum('bk,ki->bi', beta, self.u)  # (B, d)

        # 最终平移：b_θ = b_t^(0) + Σ β_k u_k
        b_theta = b0 + b_res  # (B, d)

        # 5. 应用算子：z_out = A_θz + b_θ
        # 使用bmm进行批量矩阵乘法
        # (B, d, d) @ (B, d, 1) → (B, d, 1) → squeeze → (B, d)
        z_out = torch.bmm(A_theta, z.unsqueeze(-1)).squeeze(-1) + b_theta  # (B, d)

        return z_out, A_theta, b_theta

    def spectral_penalty(
        self,
        max_allowed: float = 1.05,
        n_iterations: int = 5
    ) -> torch.Tensor:
        """
        计算谱范数稳定性正则化项

        数学定义：
            R_stab = Σ_A max(0, ρ(A) - ρ₀)²
            其中 ρ(A) 是矩阵A的谱半径（最大特征值的模）

        对应：model.md A.7.1节，第231-246行

        参数:
            max_allowed: 允许的最大谱范数 ρ₀（默认1.05）
            n_iterations: power iteration的迭代次数（默认5）

        返回:
            penalty: 标量，谱范数惩罚项

        实现策略:
            由于精确计算谱半径需要特征值分解（开销大且不可微），
            我们使用 power iteration 近似最大特征值（谱范数）。

        Power Iteration 原理:
            对于矩阵A，重复 v ← Av / ||Av|| 会收敛到最大特征值对应的特征向量。
            此时 Rayleigh商 v^T Av / v^T v ≈ λ_max

        为什么只对A_t^(0)和B_k计算：
            - 每个样本的A_θ都不同，逐一计算开销太大
            - A_t^(0)和B_k是共享参数，约束它们可间接约束所有A_θ
            - A_θ = A_t^(0) + Σ α_k B_k，如果基和响应基都稳定，组合也稳定

        数值稳定性:
            - 在归一化时添加epsilon避免除零
            - 使用.abs()确保谱值非负

        复杂度:
            时间：O(n_tissues · d² · n_iter + K · d² · n_iter)
            空间：O(d)

        示例:
            >>> model = OperatorModel(32, 3, 5, 64)
            >>> penalty = model.spectral_penalty(max_allowed=1.05)
            >>> print(penalty)
            tensor(0.0234)
        """
        penalty = torch.tensor(0.0, device=self.A0_tissue.device)

        # 对每个组织的基线算子 A_t^(0) 计算谱范数
        for t in range(self.n_tissues):
            A0 = self.A0_tissue[t]  # (d, d)

            # Power iteration 估计谱范数（最大奇异值）
            # 对 A^T A 进行power iteration
            with torch.no_grad():
                v = torch.randn(A0.size(0), device=A0.device)  # (d,)
                for _ in range(n_iterations):
                    # v ← A^T A v
                    v = A0.T @ (A0 @ v)
                    # 归一化：v ← v / ||v||
                    v = v / (v.norm() + _NUM_CFG.eps_division)

            # 计算谱范数：||A||_2 = sqrt(v^T A^T A v)
            v_detached = v.detach()
            ATA_v = A0.T @ (A0 @ v_detached)
            spec = torch.sqrt((v_detached @ ATA_v).abs() + _NUM_CFG.eps_log)  # 标量

            # 使用ReLU避免if分支（保持可微性和TorchScript兼容）
            excess = spec - max_allowed
            penalty = penalty + torch.nn.functional.relu(excess) ** 2

        # 对每个响应基 B_k 计算谱范数
        for k in range(self.K):
            Bk = self.B[k]  # (d, d)

            # Power iteration 估计谱范数（对 B_k^T B_k 迭代）
            with torch.no_grad():
                v = torch.randn(Bk.size(0), device=Bk.device)  # (d,)
                for _ in range(n_iterations):
                    v = Bk.T @ (Bk @ v)
                    v = v / (v.norm() + _NUM_CFG.eps_division)

            # 计算谱范数：||B_k||_2 = sqrt(v^T B_k^T B_k v)
            v_detached = v.detach()
            BTB_v = Bk.T @ (Bk @ v_detached)
            spec = torch.sqrt((v_detached @ BTB_v).abs() + _NUM_CFG.eps_log)

            # 使用ReLU避免if分支
            excess = spec - max_allowed
            penalty = penalty + torch.nn.functional.relu(excess) ** 2

        return penalty

    def get_response_profile(
        self,
        cond_vec: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取条件θ的响应轮廓（用于分析和可视化）

        响应轮廓定义：
            r(θ) = (α_1(θ), ..., α_K(θ), β_1(θ), ..., β_K(θ))

        用途：
            - 比较不同条件的响应模式
            - 识别主导响应轴
            - 分析药物-扰动的相似性

        对应：model.md A.5.2节，系数网络的输出

        参数:
            cond_vec: (B, cond_dim) 或 (cond_dim,) 条件向量

        返回:
            alpha: (B, K) 或 (K,) 线性响应系数
            beta: (B, K) 或 (K,) 平移响应系数

        示例:
            >>> model = OperatorModel(32, 3, 5, 64)
            >>> cond_vec = torch.randn(64)
            >>> alpha, beta = model.get_response_profile(cond_vec)
            >>> print(alpha.shape)
            torch.Size([5])
            >>> # 找到最强的响应轴
            >>> dominant_k = alpha.abs().argmax()
            >>> print(f"主导响应轴: k={dominant_k}, α={alpha[dominant_k]:.3f}")
            主导响应轴: k=2, α=1.234
        """
        # 如果输入是1D，添加batch维度
        if cond_vec.dim() == 1:
            cond_vec = cond_vec.unsqueeze(0)  # (1, cond_dim)
            squeeze_output = True
        else:
            squeeze_output = False

        alpha = self.alpha_mlp(cond_vec)  # (B, K)
        beta = self.beta_mlp(cond_vec)    # (B, K)

        # 如果输入是1D，移除batch维度
        if squeeze_output:
            alpha = alpha.squeeze(0)  # (K,)
            beta = beta.squeeze(0)    # (K,)

        return alpha, beta

    @torch.no_grad()
    def compute_operator_norm(
        self,
        tissue_idx: torch.Tensor,
        cond_vec: torch.Tensor,
        norm_type: str = "spectral"
    ) -> torch.Tensor:
        """
        计算算子A_θ的范数（用于监控稳定性，不需要梯度）

        参数:
            tissue_idx: (B,) 组织索引
            cond_vec: (B, cond_dim) 条件向量
            norm_type: 范数类型，可选 "spectral"（谱范数）或 "frobenius"（F范数）

        返回:
            norms: (B,) 每个算子的范数

        示例:
            >>> model = OperatorModel(32, 3, 5, 64)
            >>> tissue_idx = torch.zeros(10, dtype=torch.long)
            >>> cond_vec = torch.randn(10, 64)
            >>> norms = model.compute_operator_norm(tissue_idx, cond_vec)
            >>> print(norms.mean(), norms.max())
            tensor(1.0234) tensor(1.0567)
        """
        # 构造虚拟输入（不实际使用z）
        B = tissue_idx.size(0)
        z_dummy = torch.zeros(B, self.latent_dim, device=tissue_idx.device)

        # 获取A_θ
        _, A_theta, _ = self.forward(z_dummy, tissue_idx, cond_vec)  # (B, d, d)

        if norm_type == "frobenius":
            # Frobenius范数：||A||_F = sqrt(Σᵢⱼ A²ᵢⱼ)
            norms = torch.norm(A_theta.view(B, -1), dim=-1)  # (B,)
        elif norm_type == "spectral":
            # 谱范数：使用向量化的power iteration
            # 对 A^T A 进行迭代（正确的谱范数计算）
            v = torch.randn(B, self.latent_dim, device=A_theta.device)  # (B, d)
            for _ in range(n_iterations):
                # v ← A^T A v，使用bmm进行批量矩阵乘法
                v = torch.bmm(A_theta.transpose(1, 2), torch.bmm(A_theta, v.unsqueeze(-1))).squeeze(-1)
                v = v / (v.norm(dim=-1, keepdim=True) + _NUM_CFG.eps_division)

            # 计算谱范数：||A||_2 = sqrt(v^T A^T A v)
            ATA_v = torch.bmm(A_theta.transpose(1, 2), torch.bmm(A_theta, v.unsqueeze(-1))).squeeze(-1)
            norms = torch.sqrt((v * ATA_v).sum(dim=-1).abs() + _NUM_CFG.eps_log)  # (B,)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        return norms
