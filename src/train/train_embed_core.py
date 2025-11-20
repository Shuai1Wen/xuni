# -*- coding: utf-8 -*-
"""
潜空间嵌入训练循环

本模块实现NB-VAE模型的训练流程。
对应: suanfa.md第351-379行, model.md A.2节
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
from pathlib import Path
import logging
from tqdm import tqdm
import json

from ..models.nb_vae import NBVAE, elbo_loss
from ..config import TrainingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_embedding(
    model: NBVAE,
    train_loader: DataLoader,
    config: TrainingConfig,
    val_loader: Optional[DataLoader] = None,
    checkpoint_dir: Optional[str] = None,
    device: str = "cuda"
) -> Dict[str, list]:
    """训练NB-VAE潜空间嵌入模型"""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_embed)
    
    warmup_epochs = config.warmup_epochs
    beta_kl = config.beta_kl
    
    if checkpoint_dir is not None:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    history = {"train_loss": [], "train_recon": [], "train_kl": []}
    if val_loader is not None:
        history.update({"val_loss": [], "val_recon": [], "val_kl": []})
    
    best_val_loss = float("inf")
    
    for epoch in range(config.n_epochs_embed):
        beta = beta_kl * min(1.0, (epoch + 1) / max(1, warmup_epochs))
        
        # 训练
        model.train()
        epoch_loss = epoch_recon = epoch_kl = n_samples = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.n_epochs_embed}")
        for batch in pbar:
            x = batch["x"].to(device)
            tissue_onehot = batch["tissue_onehot"].to(device)
            
            loss, loss_dict = elbo_loss(x, tissue_onehot, model, beta=beta)
            
            optimizer.zero_grad()
            loss.backward()
            if config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            
            batch_size = x.size(0)
            epoch_loss += loss.item() * batch_size
            epoch_recon += loss_dict["recon_loss"].item() * batch_size
            epoch_kl += loss_dict["kl_loss"].item() * batch_size
            n_samples += batch_size
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_train_loss = epoch_loss / n_samples
        avg_train_recon = epoch_recon / n_samples
        avg_train_kl = epoch_kl / n_samples
        
        history["train_loss"].append(avg_train_loss)
        history["train_recon"].append(avg_train_recon)
        history["train_kl"].append(avg_train_kl)
        
        # 验证
        if val_loader is not None:
            val_metrics = validate_embedding(model, val_loader, device, beta)
            history["val_loss"].append(val_metrics["loss"])
            history["val_recon"].append(val_metrics["recon_loss"])
            history["val_kl"].append(val_metrics["kl_loss"])
            
            logger.info(
                f"Epoch {epoch+1} | Train: {avg_train_loss:.4f} | Val: {val_metrics['loss']:.4f}"
            )
            
            if checkpoint_dir and val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(model, optimizer, epoch, history, checkpoint_path / "best_model.pt")
        else:
            logger.info(f"Epoch {epoch+1} | Train: {avg_train_loss:.4f}")
        
        if checkpoint_dir and (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, history, checkpoint_path / f"checkpoint_epoch_{epoch+1}.pt")
    
    if checkpoint_dir:
        save_checkpoint(model, optimizer, config.n_epochs_embed - 1, history, checkpoint_path / "final_model.pt")
        with open(checkpoint_path / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)
    
    return history


@torch.no_grad()
def validate_embedding(model: NBVAE, val_loader: DataLoader, device: str, beta: float = 1.0):
    """在验证集上评估模型"""
    model.eval()
    total_loss = total_recon = total_kl = n_samples = 0.0
    
    for batch in val_loader:
        x = batch["x"].to(device)
        tissue_onehot = batch["tissue_onehot"].to(device)
        
        loss, loss_dict = elbo_loss(x, tissue_onehot, model, beta=beta)
        
        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_recon += loss_dict["recon_loss"].item() * batch_size
        total_kl += loss_dict["kl_loss"].item() * batch_size
        n_samples += batch_size
    
    return {
        "loss": total_loss / n_samples,
        "recon_loss": total_recon / n_samples,
        "kl_loss": total_kl / n_samples,
    }


def save_checkpoint(model, optimizer, epoch, history, path):
    """保存checkpoint"""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "history": history,
        "model_config": {
            "n_genes": model.n_genes,
            "latent_dim": model.latent_dim,
            "n_tissues": model.n_tissues,
            "hidden_dim": model.hidden_dim,
        },
    }
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, path, device="cuda"):
    """加载checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer, checkpoint["epoch"], checkpoint["history"]
