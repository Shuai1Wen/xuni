# -*- coding: utf-8 -*-
"""
算子模型训练循环

本模块实现扰动响应算子模型的训练流程。
对应: suanfa.md第384-451行, model.md A.3-A.4节
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, Optional
from pathlib import Path
import logging
from tqdm import tqdm
import json

from ..models.nb_vae import NBVAE
from ..models.operator import OperatorModel
from ..utils.edistance import energy_distance
from ..config import TrainingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_operator(
    operator_model: OperatorModel,
    embed_model: NBVAE,
    train_loader: DataLoader,
    config: TrainingConfig,
    val_loader: Optional[DataLoader] = None,
    checkpoint_dir: Optional[str] = None,
    device: str = "cuda",
    freeze_embed: bool = True
) -> Dict[str, list]:
    """训练扰动响应算子模型"""
    operator_model.to(device)
    embed_model.to(device)
    
    if freeze_embed:
        embed_model.eval()
        for param in embed_model.parameters():
            param.requires_grad = False
    
    optimizer = torch.optim.Adam(operator_model.parameters(), lr=config.lr_operator)
    
    if checkpoint_dir is not None:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    history = {"train_loss": [], "train_edist": [], "train_stab": []}
    if val_loader is not None:
        history.update({"val_loss": [], "val_edist": [], "val_stab": []})
    
    best_val_loss = float("inf")
    
    for epoch in range(config.n_epochs_operator):
        operator_model.train()
        epoch_loss = epoch_edist = epoch_stab = n_samples = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.n_epochs_operator}")
        for batch in pbar:
            x0 = batch["x0"].to(device)
            x1 = batch["x1"].to(device)
            tissue_onehot = batch["tissue_onehot"].to(device)
            tissue_idx = batch["tissue_idx"].to(device)
            cond_vec = batch["cond_vec"].to(device)
            
            if freeze_embed:
                with torch.no_grad():
                    mu0, _ = embed_model.encoder(x0, tissue_onehot)
                    mu1, _ = embed_model.encoder(x1, tissue_onehot)
                    z0, z1 = mu0, mu1
            else:
                mu0, _ = embed_model.encoder(x0, tissue_onehot)
                mu1, _ = embed_model.encoder(x1, tissue_onehot)
                z0, z1 = mu0, mu1
            
            z1_pred, A_theta, b_theta = operator_model(z0, tissue_idx, cond_vec)

            # 数值稳定性检查
            if torch.isnan(z1_pred).any() or torch.isinf(z1_pred).any():
                logger.error(f"Epoch {epoch+1}: 检测到NaN/Inf在z1_pred中")
                logger.error(f"A_theta范数: max={A_theta.norm(dim=(1,2)).max():.4f}, min={A_theta.norm(dim=(1,2)).min():.4f}")
                logger.error(f"z0范数: max={z0.norm(dim=1).max():.4f}, min={z0.norm(dim=1).min():.4f}")
                logger.error(f"b_theta范数: max={b_theta.norm(dim=1).max():.4f}, min={b_theta.norm(dim=1).min():.4f}")
                raise RuntimeError("数值不稳定：检测到NaN或Inf，训练终止")

            ed2 = energy_distance(z1_pred, z1)
            stab_penalty = operator_model.spectral_penalty(max_allowed=operator_model.max_spectral_norm)
            loss = config.lambda_e * ed2 + config.lambda_stab * stab_penalty

            # 损失值稳定性检查
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"Epoch {epoch+1}: 检测到NaN/Inf在损失函数中")
                logger.error(f"E-distance: {ed2.item():.4f}")
                logger.error(f"谱惩罚: {stab_penalty.item():.4f}")
                raise RuntimeError("数值不稳定：损失函数为NaN或Inf，训练终止")
            
            optimizer.zero_grad()
            loss.backward()
            if config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(operator_model.parameters(), config.gradient_clip)
            optimizer.step()
            
            batch_size = x0.size(0)
            epoch_loss += loss.item() * batch_size
            epoch_edist += ed2.item() * batch_size
            epoch_stab += stab_penalty.item() * batch_size
            n_samples += batch_size
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "E2": f"{ed2.item():.4f}"})
        
        avg_train_loss = epoch_loss / n_samples
        avg_train_edist = epoch_edist / n_samples
        avg_train_stab = epoch_stab / n_samples
        
        history["train_loss"].append(avg_train_loss)
        history["train_edist"].append(avg_train_edist)
        history["train_stab"].append(avg_train_stab)
        
        if val_loader is not None:
            val_metrics = validate_operator(operator_model, embed_model, val_loader, config, device)
            history["val_loss"].append(val_metrics["loss"])
            history["val_edist"].append(val_metrics["edist"])
            history["val_stab"].append(val_metrics["stab"])
            
            logger.info(
                f"Epoch {epoch+1} | Train: {avg_train_loss:.4f} | Val: {val_metrics['loss']:.4f}"
            )
            
            if checkpoint_dir and val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_operator_checkpoint(operator_model, optimizer, epoch, history, checkpoint_path / "best_operator.pt")
        else:
            logger.info(f"Epoch {epoch+1} | Train: {avg_train_loss:.4f}")
        
        if checkpoint_dir and (epoch + 1) % 10 == 0:
            save_operator_checkpoint(operator_model, optimizer, epoch, history, checkpoint_path / f"operator_epoch_{epoch+1}.pt")
    
    if checkpoint_dir:
        save_operator_checkpoint(operator_model, optimizer, config.n_epochs_operator - 1, history, checkpoint_path / "final_operator.pt")
        with open(checkpoint_path / "operator_history.json", "w") as f:
            json.dump(history, f, indent=2)
    
    return history


@torch.no_grad()
def validate_operator(operator_model, embed_model, val_loader, config, device):
    """在验证集上评估算子模型"""
    operator_model.eval()
    embed_model.eval()
    
    total_loss = total_edist = total_stab = n_samples = 0.0
    
    for batch in val_loader:
        x0 = batch["x0"].to(device)
        x1 = batch["x1"].to(device)
        tissue_onehot = batch["tissue_onehot"].to(device)
        tissue_idx = batch["tissue_idx"].to(device)
        cond_vec = batch["cond_vec"].to(device)
        
        mu0, _ = embed_model.encoder(x0, tissue_onehot)
        mu1, _ = embed_model.encoder(x1, tissue_onehot)
        z0, z1 = mu0, mu1
        
        z1_pred, _, _ = operator_model(z0, tissue_idx, cond_vec)
        
        ed2 = energy_distance(z1_pred, z1)
        stab_penalty = operator_model.spectral_penalty(max_allowed=operator_model.max_spectral_norm)
        loss = config.lambda_e * ed2 + config.lambda_stab * stab_penalty
        
        batch_size = x0.size(0)
        total_loss += loss.item() * batch_size
        total_edist += ed2.item() * batch_size
        total_stab += stab_penalty.item() * batch_size
        n_samples += batch_size
    
    return {
        "loss": total_loss / n_samples,
        "edist": total_edist / n_samples,
        "stab": total_stab / n_samples,
    }


def save_operator_checkpoint(operator_model, optimizer, epoch, history, path):
    """保存算子checkpoint"""
    checkpoint = {
        "model_state_dict": operator_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "history": history,
        "model_config": {
            "latent_dim": operator_model.latent_dim,
            "n_tissues": operator_model.n_tissues,
            "n_response_bases": operator_model.K,
            "cond_dim": operator_model.cond_dim,
        },
    }
    torch.save(checkpoint, path)


def load_operator_checkpoint(operator_model, optimizer, path, device="cuda"):
    """加载算子checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    operator_model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return operator_model, optimizer, checkpoint["epoch"], checkpoint["history"]
