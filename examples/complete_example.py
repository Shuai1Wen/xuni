#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整示例：训练虚拟细胞算子模型

本脚本演示如何使用项目代码进行端到端训练和预测。

运行要求：
- Python 3.9+
- PyTorch 2.0+
- Scanpy 1.9+
- 完整的依赖请参考 requirements.txt

运行方式：
    python examples/complete_example.py
"""

import torch
import numpy as np
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入项目模块
from src.config import ModelConfig, TrainingConfig, set_seed
from src.models.nb_vae import NBVAE
from src.models.operator import OperatorModel
from src.utils.cond_encoder import ConditionEncoder
from src.train.train_embed_core import train_embedding
from src.train.train_operator_core import train_operator
from src.utils.virtual_cell import virtual_cell_scenario

def create_dummy_data(n_cells=1000, n_genes=2000):
    """创建虚拟测试数据（用于演示）"""
    logger.info("创建虚拟测试数据...")
    
    # 模拟基因表达数据（负二项分布）
    mu = np.random.gamma(2, 2, (n_cells, n_genes))
    X = np.random.negative_binomial(n=5, p=5/(5+mu))
    
    # 创建obs元数据
    obs_data = {
        "tissue": np.random.choice(["kidney", "brain", "blood"], n_cells),
        "perturbation": np.random.choice(["control", "drug_A", "drug_B"], n_cells),
        "timepoint": np.random.choice(["t0", "t1"], n_cells),
        "dataset_id": ["dataset_001"] * n_cells,
        "batch": np.random.choice(["batch1", "batch2"], n_cells),
    }
    
    # 创建AnnData对象
    import anndata
    import pandas as pd
    
    adata = anndata.AnnData(X=X, obs=pd.DataFrame(obs_data))
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    
    logger.info(f"✓ 数据创建完成: {adata.shape}")
    return adata


def example_1_train_vae():
    """示例1：训练VAE模型"""
    logger.info("\n" + "="*60)
    logger.info("示例1：训练VAE潜空间嵌入模型")
    logger.info("="*60)
    
    # 设置随机种子
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")
    
    # 创建数据
    adata = create_dummy_data(n_cells=1000, n_genes=500)
    
    # 准备组织映射
    tissue2idx = {"kidney": 0, "brain": 1, "blood": 2}
    
    # 创建数据集
    from src.data.scperturb_dataset import SCPerturbEmbedDataset
    dataset = SCPerturbEmbedDataset(adata, tissue2idx)
    
    # 创建数据加载器
    from torch.utils.data import DataLoader
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    
    # 创建模型
    model = NBVAE(
        n_genes=adata.n_vars,
        latent_dim=32,
        n_tissues=len(tissue2idx),
        hidden_dims=[256, 128]
    )
    
    # 训练配置
    config = TrainingConfig(
        lr_embed=1e-3,
        batch_size=64,
        n_epochs_embed=5,  # 演示用，实际应更多
        warmup_epochs=2,
        gradient_clip=1.0
    )
    
    # 训练
    logger.info("开始训练VAE...")
    history = train_embedding(
        model=model,
        train_loader=train_loader,
        config=config,
        checkpoint_dir="results/checkpoints/vae_example",
        device=device
    )
    
    logger.info(f"✓ VAE训练完成！最终损失: {history['train_loss'][-1]:.4f}")
    return model, adata, tissue2idx


def example_2_train_operator(vae_model, adata, tissue2idx):
    """示例2：训练算子模型"""
    logger.info("\n" + "="*60)
    logger.info("示例2：训练扰动响应算子模型")
    logger.info("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建条件编码器
    cond_encoder = ConditionEncoder.from_anndata(adata, cond_dim=64)
    
    # 创建配对数据集
    from src.data.scperturb_dataset import SCPerturbPairDataset
    dataset = SCPerturbPairDataset(
        adata, cond_encoder, tissue2idx,
        max_pairs_per_condition=100,
        seed=42  # 固定seed确保可重复
    )
    
    # 数据加载器
    from torch.utils.data import DataLoader
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    # 创建算子模型
    operator = OperatorModel(
        latent_dim=32,
        n_tissues=len(tissue2idx),
        n_response_bases=5,
        cond_dim=64,
        max_spectral_norm=1.05
    )
    
    # 训练配置
    config = TrainingConfig(
        lr_operator=1e-3,
        batch_size=32,
        n_epochs_operator=5,  # 演示用
        lambda_e=1.0,
        lambda_stab=1e-3,
        gradient_clip=1.0
    )
    
    # 训练
    logger.info("开始训练算子...")
    history = train_operator(
        operator_model=operator,
        embed_model=vae_model,
        train_loader=train_loader,
        config=config,
        checkpoint_dir="results/checkpoints/operator_example",
        device=device,
        freeze_embed=True
    )
    
    logger.info(f"✓ 算子训练完成！最终E-distance: {history['train_edist'][-1]:.4f}")
    return operator, cond_encoder


def example_3_virtual_prediction(vae_model, operator, cond_encoder, adata, tissue2idx):
    """示例3：虚拟细胞预测"""
    logger.info("\n" + "="*60)
    logger.info("示例3：虚拟细胞扰动响应预测")
    logger.info("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae_model.to(device)
    operator.to(device)
    
    # 选择一些control细胞作为初始状态
    control_mask = adata.obs["perturbation"] == "control"
    x0 = torch.from_numpy(adata.X[control_mask][:10].astype(np.float32))  # 取10个细胞
    
    # 准备tissue信息
    tissue_idx = torch.zeros(10, dtype=torch.long)  # kidney
    tissue_onehot = torch.zeros(10, 3)
    tissue_onehot[:, 0] = 1.0  # kidney
    
    # 创建药物A的条件向量
    cond_drug_A = cond_encoder.encode_obs_row({
        "perturbation": "drug_A",
        "tissue": "kidney",
        "batch": "batch1",
        "dataset_id": "dataset_001"
    })
    cond_vec_seq = cond_drug_A.unsqueeze(0)  # (1, cond_dim)
    
    # 预测
    logger.info("预测control细胞在drug_A作用下的响应...")
    x_virtual = virtual_cell_scenario(
        vae=vae_model,
        operator=operator,
        x0=x0,
        tissue_onehot=tissue_onehot,
        tissue_idx=tissue_idx,
        cond_vec_seq=cond_vec_seq,
        device=device
    )
    
    logger.info(f"✓ 预测完成！虚拟细胞表达: {x_virtual.shape}")
    
    # 分析差异
    diff = (x_virtual - x0).abs().mean(dim=1)
    logger.info(f"  平均表达变化: {diff.mean().item():.4f} ± {diff.std().item():.4f}")
    
    return x_virtual


def example_4_multi_step_simulation(vae_model, operator, cond_encoder, adata, tissue2idx):
    """示例4：多步序列模拟"""
    logger.info("\n" + "="*60)
    logger.info("示例4：多步药物序列响应模拟")
    logger.info("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 准备初始细胞
    control_mask = adata.obs["perturbation"] == "control"
    x0 = torch.from_numpy(adata.X[control_mask][:5].astype(np.float32))
    
    tissue_idx = torch.zeros(5, dtype=torch.long)
    tissue_onehot = torch.zeros(5, 3)
    tissue_onehot[:, 0] = 1.0
    
    # 创建药物序列：drug_A → drug_B
    cond_A = cond_encoder.encode_obs_row({
        "perturbation": "drug_A",
        "tissue": "kidney",
        "batch": "batch1",
        "dataset_id": "dataset_001"
    })
    
    cond_B = cond_encoder.encode_obs_row({
        "perturbation": "drug_B",
        "tissue": "kidney",
        "batch": "batch1",
        "dataset_id": "dataset_001"
    })
    
    cond_vec_seq = torch.stack([cond_A, cond_B])  # (2, cond_dim)
    
    # 两步模拟
    logger.info("模拟drug_A → drug_B 序列效应...")
    x_final = virtual_cell_scenario(
        vae=vae_model,
        operator=operator,
        x0=x0,
        tissue_onehot=tissue_onehot,
        tissue_idx=tissue_idx,
        cond_vec_seq=cond_vec_seq,
        device=device
    )
    
    logger.info(f"✓ 序列模拟完成！最终状态: {x_final.shape}")
    return x_final


def main():
    """主函数：运行所有示例"""
    logger.info("="*60)
    logger.info("虚拟细胞算子模型 - 完整示例")
    logger.info("="*60)
    
    try:
        # 示例1：训练VAE
        vae_model, adata, tissue2idx = example_1_train_vae()
        
        # 示例2：训练算子
        operator, cond_encoder = example_2_train_operator(vae_model, adata, tissue2idx)
        
        # 示例3：单步预测
        x_virtual = example_3_virtual_prediction(vae_model, operator, cond_encoder, adata, tissue2idx)
        
        # 示例4：多步模拟
        x_final = example_4_multi_step_simulation(vae_model, operator, cond_encoder, adata, tissue2idx)
        
        logger.info("\n" + "="*60)
        logger.info("✓ 所有示例运行成功！")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
