# -*- coding: utf-8 -*-
"""
条件编码器

本模块将条件元信息编码为算子模型可用的向量表示。

数学对应关系：
- 条件向量：θ = θ(p,t,m,c)，对应 model.md A.1节，第27-33行
- θ包含：扰动p、组织t、mLOY状态m、协变量c

编码策略：
- 分类变量：one-hot编码或learned embedding
- 连续变量：标准化后直接使用
- 最终拼接并通过线性层降维到cond_dim

设计目标：
- 支持新扰动/新组织的泛化
- 控制维度避免过高
- 保留关键信息
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List
import pandas as pd


class ConditionEncoder(nn.Module):
    """
    条件向量编码器

    将扰动、组织、mLOY状态等信息编码为固定维度的向量θ。

    对应：model.md A.1节和A.5.2节

    编码方案：
        方案A（简单拼接）：
            θ = [perturb_onehot, tissue_onehot, mLOY_scalar, batch_onehot]
            缺点：维度爆炸（100扰动 + 10组织 = 110维）

        方案B（embedding + 拼接，推荐）：
            θ = [perturb_embed, tissue_embed, mLOY_scalar, batch_onehot]
            优点：维度可控、支持泛化

    参数:
        perturb2idx: 扰动→索引的映射字典
        tissue2idx: 组织→索引的映射字典
        batch2idx: 批次→索引的映射字典
        cond_dim: 输出条件向量的维度
        use_embedding: 是否使用learned embedding（推荐True）
        perturb_embed_dim: 扰动embedding维度
        tissue_embed_dim: 组织embedding维度

    示例:
        >>> perturb2idx = {"drug_A": 0, "drug_B": 1, "control": 2}
        >>> tissue2idx = {"blood": 0, "kidney": 1, "brain": 2}
        >>> batch2idx = {"batch1": 0, "batch2": 1}
        >>> encoder = ConditionEncoder(
        ...     perturb2idx, tissue2idx, batch2idx,
        ...     cond_dim=64, use_embedding=True
        ... )
        >>> obs_row = {
        ...     "perturbation": "drug_A",
        ...     "tissue": "kidney",
        ...     "batch": "batch1",
        ...     "mLOY_load": 0.15
        ... }
        >>> cond_vec = encoder(obs_row)
        >>> print(cond_vec.shape)
        torch.Size([64])
    """

    def __init__(
        self,
        perturb2idx: Dict[str, int],
        tissue2idx: Dict[str, int],
        batch2idx: Dict[str, int],
        cond_dim: int,
        use_embedding: bool = True,
        perturb_embed_dim: int = 16,
        tissue_embed_dim: int = 8
    ):
        super().__init__()
        self.perturb2idx = perturb2idx
        self.tissue2idx = tissue2idx
        self.batch2idx = batch2idx
        self.cond_dim = cond_dim
        self.use_embedding = use_embedding

        n_pert = len(perturb2idx)
        n_tissue = len(tissue2idx)
        n_batch = len(batch2idx)

        if use_embedding:
            # 使用learned embedding
            # +1 for unknown/OOV
            self.perturb_embedding = nn.Embedding(n_pert + 1, perturb_embed_dim)
            self.tissue_embedding = nn.Embedding(n_tissue + 1, tissue_embed_dim)

            # 计算输入维度：embedding + batch_onehot + mLOY_scalar + age + disease
            input_dim = perturb_embed_dim + tissue_embed_dim + n_batch + 1
            # +1 for mLOY_load，未来可以扩展为+3（mLOY_load, age, disease）

        else:
            # 使用one-hot编码
            input_dim = n_pert + n_tissue + n_batch + 1

        # 线性层：降维到cond_dim
        self.linear = nn.Linear(input_dim, cond_dim)

    def _one_hot(
        self,
        idx: Optional[int],
        n: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        创建one-hot向量

        参数:
            idx: 索引（可以是None，表示unknown）
            n: one-hot向量长度
            device: 设备

        返回:
            v: (n,) one-hot向量
        """
        v = torch.zeros(n, device=device)
        if idx is not None and 0 <= idx < n:
            v[idx] = 1.0
        return v

    def encode_obs_row(
        self,
        obs_row: Dict[str, any],
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        编码单个观测的元信息

        参数:
            obs_row: 字典，包含 'perturbation', 'tissue', 'batch', 'mLOY_load' 等字段
            device: 目标设备

        返回:
            cond_vec: (cond_dim,) 条件向量

        字段说明:
            - perturbation: 扰动名称（str）
            - tissue: 组织名称（str）
            - batch: 批次名称（str，可选）
            - mLOY_load: mLOY负荷（float，可选，默认0）
            - age: 年龄（int，可选，未来扩展）
            - disease_state: 疾病状态（str，可选，未来扩展）

        示例:
            >>> obs_row = {
            ...     "perturbation": "drug_A",
            ...     "tissue": "kidney",
            ...     "batch": "batch1",
            ...     "mLOY_load": 0.15
            ... }
            >>> cond_vec = encoder.encode_obs_row(obs_row)
        """
        if device is None:
            device = next(self.parameters()).device

        # 获取各字段值
        perturb = obs_row.get("perturbation", "control")
        tissue = obs_row.get("tissue", "unknown")
        batch = obs_row.get("batch", "batch0")
        mLOY = float(obs_row.get("mLOY_load", 0.0))

        # 获取索引
        p_idx = self.perturb2idx.get(perturb, len(self.perturb2idx))  # OOV → last index
        t_idx = self.tissue2idx.get(tissue, len(self.tissue2idx))
        b_idx = self.batch2idx.get(batch, 0)  # 默认batch0

        if self.use_embedding:
            # Embedding方式
            p_idx_tensor = torch.tensor([p_idx], dtype=torch.long, device=device)
            t_idx_tensor = torch.tensor([t_idx], dtype=torch.long, device=device)

            v_p = self.perturb_embedding(p_idx_tensor).squeeze(0)  # (perturb_embed_dim,)
            v_t = self.tissue_embedding(t_idx_tensor).squeeze(0)  # (tissue_embed_dim,)
            v_b = self._one_hot(b_idx, len(self.batch2idx), device)  # (n_batch,)
            v_m = torch.tensor([mLOY], dtype=torch.float32, device=device)  # (1,)

            v = torch.cat([v_p, v_t, v_b, v_m], dim=0)

        else:
            # One-hot方式
            v_p = self._one_hot(p_idx if p_idx < len(self.perturb2idx) else None,
                                len(self.perturb2idx), device)
            v_t = self._one_hot(t_idx if t_idx < len(self.tissue2idx) else None,
                                len(self.tissue2idx), device)
            v_b = self._one_hot(b_idx, len(self.batch2idx), device)
            v_m = torch.tensor([mLOY], dtype=torch.float32, device=device)

            v = torch.cat([v_p, v_t, v_b, v_m], dim=0)

        # 通过线性层降维
        cond_vec = self.linear(v)  # (cond_dim,)

        return cond_vec

    def forward(
        self,
        obs_rows: List[Dict[str, any]]
    ) -> torch.Tensor:
        """
        批量编码观测元信息

        参数:
            obs_rows: 字典列表，每个字典包含一个样本的元信息

        返回:
            cond_vecs: (B, cond_dim) 条件向量批次

        示例:
            >>> obs_rows = [
            ...     {"perturbation": "drug_A", "tissue": "kidney", ...},
            ...     {"perturbation": "drug_B", "tissue": "brain", ...},
            ... ]
            >>> cond_vecs = encoder(obs_rows)
            >>> print(cond_vecs.shape)
            torch.Size([2, 64])
        """
        device = next(self.parameters()).device
        cond_vecs = [self.encode_obs_row(obs, device) for obs in obs_rows]
        return torch.stack(cond_vecs, dim=0)  # (B, cond_dim)

    @classmethod
    def from_anndata(
        cls,
        adata,
        cond_dim: int = 64,
        use_embedding: bool = True
    ) -> "ConditionEncoder":
        """
        从AnnData对象自动构建ConditionEncoder

        自动提取obs中的唯一值，构建索引映射。

        参数:
            adata: AnnData对象
            cond_dim: 条件向量维度
            use_embedding: 是否使用embedding

        返回:
            encoder: ConditionEncoder实例

        要求adata.obs包含字段：
            - perturbation: 扰动类型
            - tissue: 组织类型
            - batch: 批次（可选）

        示例:
            >>> import scanpy as sc
            >>> adata = sc.read_h5ad("data.h5ad")
            >>> encoder = ConditionEncoder.from_anndata(adata, cond_dim=64)
        """
        # 提取唯一值
        perturbations = adata.obs["perturbation"].unique().tolist()
        tissues = adata.obs["tissue"].unique().tolist()

        if "batch" in adata.obs.columns:
            batches = adata.obs["batch"].unique().tolist()
        else:
            batches = ["batch0"]

        # 构建索引映射
        perturb2idx = {p: i for i, p in enumerate(perturbations)}
        tissue2idx = {t: i for i, t in enumerate(tissues)}
        batch2idx = {b: i for i, b in enumerate(batches)}

        return cls(
            perturb2idx,
            tissue2idx,
            batch2idx,
            cond_dim=cond_dim,
            use_embedding=use_embedding
        )
