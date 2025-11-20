#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scPerturb基准实验训练脚本

用于训练VAE和算子模型的端到端脚本。

对应：model.md A.2节（VAE）+ A.5节（算子）

用法:
    # 训练VAE
    python train_scperturb_baseline.py --phase vae --config configs/scperturb_vae.yaml

    # 训练算子
    python train_scperturb_baseline.py --phase operator --config configs/scperturb_operator.yaml
"""

import argparse
import yaml
import torch
from pathlib import Path
import scanpy as sc
from torch.utils.data import DataLoader
import sys
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.nb_vae import NBVAE
from src.models.operator import OperatorModel
from src.data.scperturb_dataset import SCPerturbEmbedDataset, SCPerturbPairDataset
from src.data.scperturb_dataset import collate_fn_embed, collate_fn_pair
from src.train.train_embed_core import train_embedding
from src.train.train_operator_core import train_operator
from src.utils.cond_encoder import ConditionEncoder
from src.config import set_seed, ModelConfig, TrainingConfig


def load_config(config_path: str) -> dict:
    """
    加载YAML配置文件

    参数:
        config_path: 配置文件路径

    返回:
        config: 配置字典
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def train_vae_phase(args):
    """
    训练VAE阶段

    流程:
        1. 加载配置
        2. 准备数据集
        3. 创建VAE模型
        4. 训练VAE
        5. 保存最佳模型

    参数:
        args: 命令行参数
    """
    print("=" * 80)
    print("阶段1: 训练VAE（潜空间嵌入）")
    print("=" * 80)

    config = load_config(args.config)
    set_seed(config["experiment"]["seed"])

    # 创建保存目录
    checkpoint_dir = Path(config["experiment"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(config["experiment"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置
    config_save_path = checkpoint_dir / "config.yaml"
    with open(config_save_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)
    print(f"配置已保存至: {config_save_path}")

    # 加载数据
    print("\n加载训练数据...")
    adata_train = sc.read_h5ad(config["data"]["data_path"])
    print(f"训练集: {adata_train.n_obs} 细胞, {adata_train.n_vars} 基因")

    adata_val = None
    if "val_data_path" in config["data"] and Path(config["data"]["val_data_path"]).exists():
        adata_val = sc.read_h5ad(config["data"]["val_data_path"])
        print(f"验证集: {adata_val.n_obs} 细胞, {adata_val.n_vars} 基因")

    # 构建tissue2idx
    tissue2idx = {t: i for i, t in enumerate(sorted(adata_train.obs["tissue"].unique()))}
    print(f"组织类型: {list(tissue2idx.keys())}")

    # 创建数据集
    print("\n创建数据集...")
    train_dataset = SCPerturbEmbedDataset(adata_train, tissue2idx)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["experiment"]["num_workers"],
        collate_fn=collate_fn_embed,
        pin_memory=True if config["experiment"]["device"] == "cuda" else False
    )

    val_loader = None
    if adata_val is not None:
        val_dataset = SCPerturbEmbedDataset(adata_val, tissue2idx)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=config["experiment"]["num_workers"],
            collate_fn=collate_fn_embed,
            pin_memory=True if config["experiment"]["device"] == "cuda" else False
        )

    # 创建模型
    print("\n创建VAE模型...")
    model = NBVAE(
        n_genes=config["model"]["n_genes"],
        latent_dim=config["model"]["latent_dim"],
        n_tissues=len(tissue2idx),
        hidden_dim=config["model"]["hidden_dim"]
    )
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练配置
    train_config = TrainingConfig(
        lr_embed=config["training"]["lr_embed"],
        batch_size=config["training"]["batch_size"],
        n_epochs_embed=config["training"]["n_epochs_embed"],
        gradient_clip=config["training"]["gradient_clip"],
        beta_kl=config["training"]["beta_kl"],
        warmup_epochs=config["training"]["warmup_epochs"]
    )

    # 训练
    print("\n开始训练VAE...")
    history = train_embedding(
        model=model,
        train_loader=train_loader,
        config=train_config,
        val_loader=val_loader,
        checkpoint_dir=str(checkpoint_dir),
        device=config["experiment"]["device"]
    )

    # 保存训练历史
    history_path = log_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\n训练历史已保存至: {history_path}")

    print("\n" + "=" * 80)
    print("VAE训练完成！")
    print(f"最佳模型保存在: {checkpoint_dir}/best_model.pt")
    print("=" * 80)


def train_operator_phase(args):
    """
    训练算子阶段

    流程:
        1. 加载配置
        2. 准备数据集（配对数据）
        3. 加载预训练VAE
        4. 创建算子模型
        5. 训练算子
        6. 保存最佳模型

    参数:
        args: 命令行参数
    """
    print("=" * 80)
    print("阶段2: 训练算子（扰动响应建模）")
    print("=" * 80)

    config = load_config(args.config)
    set_seed(config["experiment"]["seed"])

    # 创建保存目录
    checkpoint_dir = Path(config["experiment"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(config["experiment"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置
    config_save_path = checkpoint_dir / "config.yaml"
    with open(config_save_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)
    print(f"配置已保存至: {config_save_path}")

    # 加载数据
    print("\n加载训练数据...")
    adata_train = sc.read_h5ad(config["data"]["data_path"])
    print(f"训练集: {adata_train.n_obs} 细胞, {adata_train.n_vars} 基因")

    adata_val = None
    if "val_data_path" in config["data"] and Path(config["data"]["val_data_path"]).exists():
        adata_val = sc.read_h5ad(config["data"]["val_data_path"])
        print(f"验证集: {adata_val.n_obs} 细胞, {adata_val.n_vars} 基因")

    # 构建词汇表
    tissue2idx = {t: i for i, t in enumerate(sorted(adata_train.obs["tissue"].unique()))}
    print(f"组织类型: {list(tissue2idx.keys())}")

    # 创建条件编码器
    print("\n创建条件编码器...")
    cond_encoder = ConditionEncoder.from_anndata(
        adata_train,
        cond_dim=config["model"]["cond_dim"],
        use_embedding=config["cond_encoder"]["use_embedding"]
    )
    print(f"扰动类型数量: {len(cond_encoder.perturb2idx)}")
    print(f"条件向量维度: {config['model']['cond_dim']}")

    # 创建数据集
    print("\n创建配对数据集...")
    train_dataset = SCPerturbPairDataset(adata_train, cond_encoder, tissue2idx)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["experiment"]["num_workers"],
        collate_fn=collate_fn_pair,
        pin_memory=True if config["experiment"]["device"] == "cuda" else False
    )

    val_loader = None
    if adata_val is not None:
        val_dataset = SCPerturbPairDataset(adata_val, cond_encoder, tissue2idx)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=config["experiment"]["num_workers"],
            collate_fn=collate_fn_pair,
            pin_memory=True if config["experiment"]["device"] == "cuda" else False
        )

    # 加载VAE
    print("\n加载预训练VAE...")
    vae_checkpoint_path = args.vae_checkpoint if args.vae_checkpoint else config["experiment"]["vae_checkpoint"]
    vae_checkpoint = torch.load(vae_checkpoint_path, map_location="cpu")

    embed_model = NBVAE(
        n_genes=vae_checkpoint["model_config"]["n_genes"],
        latent_dim=vae_checkpoint["model_config"]["latent_dim"],
        n_tissues=vae_checkpoint["model_config"]["n_tissues"],
        hidden_dim=vae_checkpoint["model_config"]["hidden_dim"]
    )
    embed_model.load_state_dict(vae_checkpoint["model_state_dict"])
    print(f"VAE加载成功: {vae_checkpoint_path}")

    # 创建算子模型
    print("\n创建算子模型...")
    operator_model = OperatorModel(
        latent_dim=config["model"]["latent_dim"],
        n_tissues=len(tissue2idx),
        n_response_bases=config["model"]["n_response_bases"],
        cond_dim=config["model"]["cond_dim"],
        max_spectral_norm=config["model"]["max_spectral_norm"]
    )
    print(f"模型参数量: {sum(p.numel() for p in operator_model.parameters()):,}")
    print(f"响应基数量: {config['model']['n_response_bases']}")

    # 训练配置
    train_config = TrainingConfig(
        lr_operator=config["training"]["lr_operator"],
        batch_size=config["training"]["batch_size"],
        n_epochs_operator=config["training"]["n_epochs_operator"],
        lambda_e=config["training"]["lambda_e"],
        lambda_stab=config["training"]["lambda_stab"],
        gradient_clip=config["training"]["gradient_clip"]
    )

    # 训练
    print("\n开始训练算子...")
    history = train_operator(
        operator_model=operator_model,
        embed_model=embed_model,
        train_loader=train_loader,
        config=train_config,
        val_loader=val_loader,
        checkpoint_dir=str(checkpoint_dir),
        device=config["experiment"]["device"],
        freeze_embed=config["training"]["freeze_vae"]
    )

    # 保存训练历史
    history_path = log_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\n训练历史已保存至: {history_path}")

    # 保存条件编码器
    encoder_path = checkpoint_dir / "cond_encoder.pt"
    torch.save({
        "perturb2idx": cond_encoder.perturb2idx,
        "tissue2idx": cond_encoder.tissue2idx,
        "batch2idx": cond_encoder.batch2idx,
        "state_dict": cond_encoder.state_dict(),
        "config": {
            "cond_dim": config["model"]["cond_dim"],
            "use_embedding": config["cond_encoder"]["use_embedding"]
        }
    }, encoder_path)
    print(f"条件编码器已保存至: {encoder_path}")

    print("\n" + "=" * 80)
    print("算子训练完成！")
    print(f"最佳模型保存在: {checkpoint_dir}/best_operator.pt")
    print("=" * 80)


def main():
    """主函数：解析命令行参数并执行训练"""
    parser = argparse.ArgumentParser(
        description="scPerturb基准实验训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 训练VAE
  python train_scperturb_baseline.py --phase vae --config configs/scperturb_vae.yaml

  # 训练算子
  python train_scperturb_baseline.py --phase operator --config configs/scperturb_operator.yaml --vae_checkpoint results/checkpoints/scperturb_vae/best_model.pt
        """
    )

    parser.add_argument(
        "--phase",
        choices=["vae", "operator"],
        required=True,
        help="训练阶段：vae（VAE训练）或operator（算子训练）"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML配置文件路径"
    )
    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        default=None,
        help="VAE检查点路径（仅operator阶段需要）"
    )

    args = parser.parse_args()

    # 检查配置文件是否存在
    if not Path(args.config).exists():
        print(f"错误: 配置文件不存在: {args.config}")
        return

    # 执行对应阶段
    if args.phase == "vae":
        train_vae_phase(args)
    elif args.phase == "operator":
        if not args.vae_checkpoint:
            # 尝试从配置文件读取
            config = load_config(args.config)
            if "vae_checkpoint" in config.get("experiment", {}):
                args.vae_checkpoint = config["experiment"]["vae_checkpoint"]
            else:
                print("错误: operator阶段需要指定--vae_checkpoint参数")
                return

        if not Path(args.vae_checkpoint).exists():
            print(f"错误: VAE检查点不存在: {args.vae_checkpoint}")
            return

        train_operator_phase(args)


if __name__ == "__main__":
    main()
