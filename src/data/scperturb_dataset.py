# -*- coding: utf-8 -*-
"""
scPerturb数据加载器

本模块为scPerturb数据集提供PyTorch数据加载接口。

数据集类型：
1. SCPerturbEmbedDataset：用于VAE训练的统一数据集
2. SCPerturbPairDataset：用于算子训练的配对数据集

数据结构：
- 输入：AnnData对象，包含obs元信息和X表达矩阵
- 输出：PyTorch张量，适合DataLoader使用

对应：suanfa.md第459-531行
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import numpy as np
from ..utils.cond_encoder import ConditionEncoder


class SCPerturbEmbedDataset(Dataset):
    """
    scPerturb嵌入数据集（用于VAE训练）

    将所有细胞（不区分时间点和扰动）统一加载用于训练VAE。

    对应：suanfa.md第464-485行

    参数:
        adata: AnnData对象，包含预处理后的单细胞数据
        tissue2idx: 组织名称→索引的映射字典

    要求adata包含：
        - adata.X: (n_cells, n_genes) 表达矩阵（通常log1p+标准化）
        - adata.obs["tissue"]: 每个细胞的组织类型

    示例:
        >>> import scanpy as sc
        >>> adata = sc.read_h5ad("data/processed/scperturb/merged.h5ad")
        >>> tissue2idx = {"blood": 0, "kidney": 1, "brain": 2}
        >>> dataset = SCPerturbEmbedDataset(adata, tissue2idx)
        >>> print(len(dataset))
        50000
        >>> batch = dataset[0]
        >>> print(batch["x"].shape, batch["tissue_onehot"].shape)
        torch.Size([2000]) torch.Size([3])
    """

    def __init__(
        self,
        adata,
        tissue2idx: Dict[str, int]
    ):
        self.adata = adata
        self.tissue2idx = tissue2idx
        self.n_tissues = len(tissue2idx)

    def __len__(self) -> int:
        """返回数据集大小"""
        return self.adata.n_obs

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本

        参数:
            idx: 样本索引

        返回:
            batch: 字典，包含：
                - x: (G,) 基因表达向量
                - tissue_onehot: (n_tissues,) 组织one-hot编码
                - tissue_idx: 标量，组织索引（long类型）
        """
        # 获取表达向量
        x = self.adata.X[idx]
        if hasattr(x, "toarray"):  # 如果是稀疏矩阵
            x = x.toarray().squeeze()
        x = torch.from_numpy(x.astype(np.float32))

        # 获取组织信息
        tissue = self.adata.obs.iloc[idx]["tissue"]
        t_idx = self.tissue2idx.get(tissue, 0)  # 默认0

        # 构造one-hot编码
        t_onehot = torch.zeros(self.n_tissues, dtype=torch.float32)
        t_onehot[t_idx] = 1.0

        return {
            "x": x,                                            # (G,)
            "tissue_onehot": t_onehot,                        # (n_tissues,)
            "tissue_idx": torch.tensor(t_idx, dtype=torch.long)  # 标量
        }


class SCPerturbPairDataset(Dataset):
    """
    scPerturb配对数据集（用于算子训练）

    为每个(dataset, cell_type, perturbation, tissue)条件构建t0→t1的细胞对。

    对应：suanfa.md第486-531行

    数据组织：
        按条件θ分组，每组内：
        - t0细胞：对照或时间点0
        - t1细胞：扰动或时间点T
        配对策略：随机采样形成(x0, x1)对

    参数:
        adata: AnnData对象
        cond_encoder: ConditionEncoder实例
        tissue2idx: 组织名称→索引的映射
        max_pairs_per_condition: 每个条件最多采样多少对（控制数据集大小）

    要求adata包含：
        - adata.obs["tissue"]: 组织类型
        - adata.obs["perturbation"]: 扰动类型
        - adata.obs["timepoint"]: 时间点（"t0"或"t1"）
        - adata.obs["cell_type"]: 细胞类型（可选）
        - adata.obs["dataset_id"]: 数据集ID

    示例:
        >>> adata = sc.read_h5ad("data/processed/scperturb/merged.h5ad")
        >>> tissue2idx = {"blood": 0, "kidney": 1}
        >>> cond_encoder = ConditionEncoder.from_anndata(adata, cond_dim=64)
        >>> dataset = SCPerturbPairDataset(adata, cond_encoder, tissue2idx)
        >>> print(len(dataset))
        10000
        >>> batch = dataset[0]
        >>> print(batch["x0"].shape, batch["x1"].shape, batch["cond_vec"].shape)
        torch.Size([2000]) torch.Size([2000]) torch.Size([64])
    """

    def __init__(
        self,
        adata,
        cond_encoder: ConditionEncoder,
        tissue2idx: Dict[str, int],
        max_pairs_per_condition: int = 500,
        seed: Optional[int] = None
    ):
        self.adata = adata
        self.cond_encoder = cond_encoder
        self.tissue2idx = tissue2idx
        self.n_tissues = len(tissue2idx)
        self.max_pairs_per_condition = max_pairs_per_condition
        self.seed = seed  # 保存seed用于可重复性

        # 构建配对
        self.pairs = self._build_pairs()

    def _build_pairs(self) -> List[Tuple[int, int, Dict]]:
        """
        构建细胞配对列表

        返回:
            pairs: List[(idx0, idx1, obs_dict)]
                - idx0: t0细胞的索引
                - idx1: t1细胞的索引
                - obs_dict: 条件元信息字典
        """
        pairs = []

        # 按条件分组
        # 条件键：(dataset_id, tissue, cell_type, perturbation)
        obs_df = self.adata.obs

        # 确保需要的列存在
        required_cols = ["tissue", "perturbation", "timepoint", "dataset_id"]
        for col in required_cols:
            if col not in obs_df.columns:
                raise ValueError(f"adata.obs缺少必需列: {col}")

        # 按条件分组
        if "cell_type" in obs_df.columns:
            group_keys = ["dataset_id", "tissue", "cell_type", "perturbation"]
        else:
            group_keys = ["dataset_id", "tissue", "perturbation"]

        grouped = obs_df.groupby(group_keys)

        for condition, group in grouped:
            # 分离t0和t1
            t0_indices = group[group["timepoint"] == "t0"].index.tolist()
            t1_indices = group[group["timepoint"] == "t1"].index.tolist()

            if len(t0_indices) == 0 or len(t1_indices) == 0:
                # 跳过没有配对的条件
                continue

            # 采样配对
            n_pairs = min(
                len(t0_indices),
                len(t1_indices),
                self.max_pairs_per_condition
            )

            # 随机采样
            rng = np.random.RandomState(self.seed)  # 使用可控的随机种子
            t0_sampled = rng.choice(t0_indices, size=n_pairs, replace=True)
            t1_sampled = rng.choice(t1_indices, size=n_pairs, replace=True)

            # 构造obs_dict（使用t0的元信息，但标记为配对）
            for i0, i1 in zip(t0_sampled, t1_sampled):
                obs_dict = obs_df.iloc[self.adata.obs.index.get_loc(i0)].to_dict()
                pairs.append((
                    self.adata.obs.index.get_loc(i0),  # AnnData内部索引
                    self.adata.obs.index.get_loc(i1),
                    obs_dict
                ))

        return pairs

    def __len__(self) -> int:
        """返回配对数量"""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个配对样本

        参数:
            idx: 配对索引

        返回:
            batch: 字典，包含：
                - x0: (G,) t0时刻的基因表达
                - x1: (G,) t1时刻的基因表达
                - tissue_onehot: (n_tissues,) 组织one-hot编码
                - tissue_idx: 标量，组织索引
                - cond_vec: (cond_dim,) 条件向量θ
        """
        idx0, idx1, obs_dict = self.pairs[idx]

        # 获取表达向量
        x0 = self.adata.X[idx0]
        x1 = self.adata.X[idx1]

        if hasattr(x0, "toarray"):
            x0 = x0.toarray().squeeze()
        if hasattr(x1, "toarray"):
            x1 = x1.toarray().squeeze()

        x0 = torch.from_numpy(x0.astype(np.float32))
        x1 = torch.from_numpy(x1.astype(np.float32))

        # 获取组织信息
        tissue = obs_dict["tissue"]
        t_idx = self.tissue2idx.get(tissue, 0)

        # 构造one-hot编码
        t_onehot = torch.zeros(self.n_tissues, dtype=torch.float32)
        t_onehot[t_idx] = 1.0

        # 编码条件向量
        cond_vec = self.cond_encoder.encode_obs_row(obs_dict)

        return {
            "x0": x0,                                          # (G,)
            "x1": x1,                                          # (G,)
            "tissue_onehot": t_onehot,                        # (n_tissues,)
            "tissue_idx": torch.tensor(t_idx, dtype=torch.long),  # 标量
            "cond_vec": cond_vec                              # (cond_dim,)
        }


def collate_fn_embed(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    自定义collate函数（用于EmbedDataset）

    参数:
        batch: List[Dict]，每个Dict是__getitem__的返回值

    返回:
        batched: Dict[str, Tensor]，各字段stack成batch

    示例:
        >>> from torch.utils.data import DataLoader
        >>> dataloader = DataLoader(
        ...     dataset,
        ...     batch_size=128,
        ...     collate_fn=collate_fn_embed
        ... )
    """
    return {
        "x": torch.stack([item["x"] for item in batch]),  # (B, G)
        "tissue_onehot": torch.stack([item["tissue_onehot"] for item in batch]),  # (B, n_tissues)
        "tissue_idx": torch.stack([item["tissue_idx"] for item in batch])  # (B,)
    }


def collate_fn_pair(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    自定义collate函数（用于PairDataset）

    参数:
        batch: List[Dict]

    返回:
        batched: Dict[str, Tensor]
    """
    return {
        "x0": torch.stack([item["x0"] for item in batch]),  # (B, G)
        "x1": torch.stack([item["x1"] for item in batch]),  # (B, G)
        "tissue_onehot": torch.stack([item["tissue_onehot"] for item in batch]),  # (B, n_tissues)
        "tissue_idx": torch.stack([item["tissue_idx"] for item in batch]),  # (B,)
        "cond_vec": torch.stack([item["cond_vec"] for item in batch])  # (B, cond_dim)
    }
