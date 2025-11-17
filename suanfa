好，那这一步我们就把上面那个方法真正“拆成代码骨架”。
我会用 **PyTorch 风格** 给出各个模块的核心类 / 函数结构，包含：

1. 数据与配置结构
2. 潜空间模型（NB-VAE / encoder & decoder）
3. E-distance 计算模块
4. 局部算子拟合（per-θ 的 A, b）
5. 全局低秩算子模型（K_\theta = A_t^(0) + Σ α_k B_k, b_\theta 类似）
6. 端到端训练循环骨架（含稳定性正则）
7. scPerturb 实验管线骨架
8. mLOY kidney + brain 跨组织实验骨架
9. 反事实模拟接口

下面代码是**框架级**的：API、模块拆分和关键计算都给你写出来了，但很多细节（例如加载具体数据、超参数设置）你可以根据实际项目填充。你可以把这些直接拆成多个 `.py` 文件。

---

## 1. 公共配置与数据结构

```python
# config.py
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class ModelConfig:
    n_genes: int
    latent_dim: int = 32
    n_response_bases: int = 4  # K
    max_spectral_norm: float = 1.05
    use_generator_view: bool = False

@dataclass
class TrainingConfig:
    lr_embed: float = 1e-3
    lr_operator: float = 1e-3
    batch_size: int = 512
    n_epochs_embed: int = 100
    n_epochs_operator: int = 100
    lambda_e: float = 1.0      # weight for E-distance
    lambda_stab: float = 1e-3  # weight for stability regularization

@dataclass
class ConditionMeta:
    """描述一个 θ 条件的元信息，用于索引和编码."""
    dataset_id: str
    tissue: str           # 'blood', 'kidney', 'brain', ...
    perturbation: str     # drug / KO / 'LOY' / 'control'
    timepoint: str        # 't0', 't1'
    donor_id: Optional[str] = None
    mLOY_load: Optional[float] = None  # donor-level mLOY
    batch: Optional[str] = None
```

---

## 2. 潜空间模型：NB-VAE（encoder + decoder）

这里给一个简化的 NB-VAE，重点是结构；你可以根据实际需求增加层数或引入 library（如 scvi-tools）。

```python
# models/nb_vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, n_genes: int, latent_dim: int, n_tissues: int):
        super().__init__()
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        self.n_tissues = n_tissues
        
        hidden_dim = 512
        
        self.input_layer = nn.Linear(n_genes, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim + n_tissues, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim + n_tissues, latent_dim)
    
    def forward(self, x: torch.Tensor, tissue_onehot: torch.Tensor):
        """
        x: (B, G) 计数（可以先 log1p + 标准化）
        tissue_onehot: (B, n_tissues)
        """
        h = F.relu(self.input_layer(x))
        h_cat = torch.cat([h, tissue_onehot], dim=-1)
        mu = self.fc_mean(h_cat)
        logvar = self.fc_logvar(h_cat)
        return mu, logvar

class DecoderNB(nn.Module):
    def __init__(self, n_genes: int, latent_dim: int, n_tissues: int):
        super().__init__()
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        self.n_tissues = n_tissues
        
        hidden_dim = 512
        self.fc = nn.Linear(latent_dim + n_tissues, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, n_genes)
        
        # gene-wise dispersion log(r_g)
        self.log_dispersion = nn.Parameter(torch.zeros(n_genes))
    
    def forward(self, z: torch.Tensor, tissue_onehot: torch.Tensor):
        h = F.relu(self.fc(torch.cat([z, tissue_onehot], dim=-1)))
        mu = F.softplus(self.fc_mu(h)) + 1e-8  # (B, G), mean of NB
        r = torch.exp(self.log_dispersion).unsqueeze(0)  # (1, G)
        return mu, r

def sample_z(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def nb_log_likelihood(x, mu, r, eps=1e-8):
    """
    NB pmf: p(x) = Gamma(x+r) / (Gamma(r) x!) * (r/(r+mu))^r * (mu/(r+mu))^x
    返回 log p(x|mu,r)
    """
    x = x.float()
    log_coef = (
        torch.lgamma(x + r) - torch.lgamma(r) - torch.lgamma(x + 1.0)
    )
    log_p = (
        log_coef
        + r * torch.log(r / (r + mu) + eps)
        + x * torch.log(mu / (r + mu) + eps)
    )
    return log_p.sum(-1)  # sum over genes

class NBVAE(nn.Module):
    def __init__(self, n_genes: int, latent_dim: int, n_tissues: int):
        super().__init__()
        self.encoder = Encoder(n_genes, latent_dim, n_tissues)
        self.decoder = DecoderNB(n_genes, latent_dim, n_tissues)
    
    def forward(self, x, tissue_onehot):
        mu, logvar = self.encoder(x, tissue_onehot)
        z = sample_z(mu, logvar)
        mu_x, r_x = self.decoder(z, tissue_onehot)
        return z, mu_x, r_x, mu, logvar

def elbo_loss(x, tissue_onehot, model: NBVAE, beta_kl=1.0):
    z, mu_x, r_x, mu_z, logvar_z = model(x, tissue_onehot)
    log_px = nb_log_likelihood(x, mu_x, r_x)   # (B,)
    kl = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp(), dim=-1)
    loss = -(log_px - beta_kl * kl).mean()
    return loss, z.detach()
```

---

## 3. E-distance 计算模块

```python
# utils/edistance.py
import torch

def pairwise_distances(x: torch.Tensor, y: torch.Tensor):
    """
    x: (n, d), y: (m, d)
    return: (n, m) pairwise L2 distances
    """
    x2 = (x**2).sum(-1, keepdim=True)  # (n,1)
    y2 = (y**2).sum(-1, keepdim=True).T  # (1,m)
    xy = x @ y.T  # (n,m)
    dist2 = x2 + y2 - 2*xy
    dist2 = torch.clamp(dist2, min=0.)
    return torch.sqrt(dist2 + 1e-8)

def energy_distance(x: torch.Tensor, y: torch.Tensor):
    """
    x: (n, d), y: (m, d)
    返回 E-distance 的无偏估计
    """
    n, m = x.size(0), y.size(0)
    if n == 0 or m == 0:
        return torch.tensor(0., device=x.device)
    
    d_xy = pairwise_distances(x, y)
    d_xx = pairwise_distances(x, x)
    d_yy = pairwise_distances(y, y)
    
    term_xy = 2.0 / (n * m) * d_xy.sum()
    term_xx = 1.0 / (n * n) * d_xx.sum()
    term_yy = 1.0 / (m * m) * d_yy.sum()
    
    ed2 = term_xy - term_xx - term_yy
    return ed2
```

---

## 4. 局部算子拟合（per-θ 拟合 A, b，用于初始化）

这里先给出一个简化版：对每个 θ，我们用线性回归从 z_0 → z_1（不直接用 E-distance，以便初始化），后面在全局训练中再用 E-distance 微调。

```python
# operators/local_fit.py
import torch
from torch import nn
from typing import Tuple

def fit_local_operator(z0: torch.Tensor, z1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对给定条件 θ 下的细胞集合，拟合仿射变换：
    z1 ≈ A z0 + b
    z0: (N, d), z1: (N, d)
    返回 A: (d,d), b: (d,)
    """
    N, d = z0.shape
    device = z0.device
    
    # 构造带偏置的输入 [z0, 1]
    X = torch.cat([z0, torch.ones(N, 1, device=device)], dim=1)  # (N, d+1)
    
    # 最小二乘 A_ext: (d+1, d) 使得 X A_ext ≈ z1
    # 直接用 pseudo-inverse，注意N比较大时要小心内存
    XTX = X.T @ X + 1e-6 * torch.eye(d+1, device=device)
    XTy = X.T @ z1
    A_ext = torch.linalg.solve(XTX, XTy)  # (d+1, d)
    
    A = A_ext[:-1, :]        # (d,d)
    b = A_ext[-1, :]         # (d,)
    return A, b
```

你可以对每个 (dataset, cell type, perturb, tissue) 条件 θ 调用这个函数，得到一批 (\tilde{A}*\theta,\tilde{b}*\theta)，作为下一步低秩分解的“观察值”。

---

## 5. 全局低秩算子模型

### 5.1 OperatorModel 结构

```python
# models/operator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class OperatorModel(nn.Module):
    """
    实现 K_θ(z) = A_θ z + b_θ 的低秩结构:
      A_θ = A_t^(0) + sum_k alpha_k(θ) B_k
      b_θ = b_t^(0) + sum_k beta_k(θ) u_k
    """
    def __init__(self, latent_dim: int, n_tissues: int, n_response_bases: int, cond_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_tissues = n_tissues
        self.K = n_response_bases
        self.cond_dim = cond_dim
        
        # 每个组织的基线算子和偏置
        self.A0_tissue = nn.Parameter(torch.zeros(n_tissues, latent_dim, latent_dim))
        self.b0_tissue = nn.Parameter(torch.zeros(n_tissues, latent_dim))
        
        # 全局响应基 B_k, u_k
        self.B = nn.Parameter(torch.zeros(self.K, latent_dim, latent_dim))
        self.u = nn.Parameter(torch.zeros(self.K, latent_dim))
        
        # 用小网络从条件向量 θ 预测 α_k(θ), β_k(θ)
        hidden_dim = 64
        self.alpha_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.K)
        )
        self.beta_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.K)
        )
    
    def forward(self, z: torch.Tensor, tissue_idx: torch.Tensor, cond_vec: torch.Tensor):
        """
        z: (B, d)
        tissue_idx: (B,) long, 每个细胞所属组织的 index
        cond_vec: (B, cond_dim), 该细胞对应条件 θ 的编码向量
        """
        B = z.size(0)
        d = self.latent_dim
        
        # 计算 α_k(θ) 和 β_k(θ)
        alpha = self.alpha_mlp(cond_vec)  # (B, K)
        beta  = self.beta_mlp(cond_vec)   # (B, K)
        
        # 取对应组织的基线算子 A0_t, b0_t
        A0 = self.A0_tissue[tissue_idx]    # (B, d, d)
        b0 = self.b0_tissue[tissue_idx]    # (B, d)
        
        # 组合响应基：A_theta = A0 + sum_k alpha_k B_k
        # B_k: (K, d, d) -> (B, K, d, d)
        B_expand = self.B.unsqueeze(0).expand(B, -1, -1, -1)
        alpha_expand = alpha.view(B, self.K, 1, 1)  # (B, K,1,1)
        A_res = (alpha_expand * B_expand).sum(1)    # (B, d, d)
        A_theta = A0 + A_res
        
        # 平移项: b_theta = b0 + sum_k beta_k u_k
        u_expand = self.u.unsqueeze(0).expand(B, -1, -1)   # (B, K, d)
        beta_expand = beta.view(B, self.K, 1)              # (B, K,1)
        b_res = (beta_expand * u_expand).sum(1)            # (B, d)
        b_theta = b0 + b_res
        
        # 应用算子
        z_out = torch.bmm(A_theta, z.unsqueeze(-1)).squeeze(-1) + b_theta
        return z_out, A_theta, b_theta
    
    def spectral_penalty(self, max_allowed: float = 1.05):
        """
        对 A_theta 的谱范数做稳定性正则。
        这里只对 B_k 和 A0_tissue 做约束，简化计算。
        """
        penalty = 0.
        # 计算每个 A0_tissue 的谱范数近似
        for A0 in self.A0_tissue:
            # power iteration
            v = torch.randn(A0.size(0), device=A0.device)
            for _ in range(5):
                v = A0 @ v
                v = v / (v.norm() + 1e-8)
            spec = (v @ (A0 @ v)).abs()
            if spec > max_allowed:
                penalty = penalty + (spec - max_allowed)**2
        
        # 对各 B_k 做同样的约束
        for Bk in self.B:
            v = torch.randn(Bk.size(0), device=Bk.device)
            for _ in range(5):
                v = Bk @ v
                v = v / (v.norm() + 1e-8)
            spec = (v @ (Bk @ v)).abs()
            if spec > max_allowed:
                penalty = penalty + (spec - max_allowed)**2
        
        return penalty
```

> 说明：`cond_vec` 的构造非常关键，它应该包含 perturb / tissue / mLOY / batch 等 one-hot 或嵌入。下面在实验 pipeline 里会示意。

---

## 6. 端到端训练循环骨架

### 6.1 训练潜空间（Phase 1）

```python
# train/train_embed.py
import torch
from torch.utils.data import DataLoader

def train_embedding(model: NBVAE,
                    dataloader: DataLoader,
                    n_epochs: int,
                    lr: float,
                    device: str = "cuda"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.
        for batch in dataloader:
            # batch 应该提供 x_raw, tissue_onehot
            x = batch["x"].to(device)  # (B,G)
            tissue = batch["tissue_onehot"].to(device)  # (B, n_tissues)
            
            loss, _ = elbo_loss(x, tissue, model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
        
        print(f"[Embed] Epoch {epoch+1}/{n_epochs}, loss={total_loss/len(dataloader.dataset):.4f}")
```

### 6.2 训练算子模型（Phase 3 / 端到端）

```python
# train/train_operator.py
from torch.utils.data import DataLoader
from utils.edistance import energy_distance

def train_operator(operator_model: OperatorModel,
                   embed_model: NBVAE,
                   dataloader_pairs: DataLoader,
                   train_config: TrainingConfig,
                   device: str = "cuda"):
    """
    dataloader_pairs 输出的 batch 结构示例：
      {
        "x0": (B,G),
        "x1": (B,G),
        "tissue_onehot": (B, n_tissues),
        "tissue_idx": (B,),
        "cond_vec": (B, cond_dim)
      }
    """
    operator_model.to(device)
    embed_model.to(device)
    embed_model.eval()  # 通常冻结 embedding, 也可以微调
    
    opt = torch.optim.Adam(operator_model.parameters(), lr=train_config.lr_operator)
    
    for epoch in range(train_config.n_epochs_operator):
        total_e = 0.
        total_stab = 0.
        n_cells = 0
        
        for batch in dataloader_pairs:
            x0 = batch["x0"].to(device)
            x1 = batch["x1"].to(device)
            tissue_onehot = batch["tissue_onehot"].to(device)
            tissue_idx = batch["tissue_idx"].to(device)
            cond_vec = batch["cond_vec"].to(device)
            
            with torch.no_grad():
                # 编码到潜空间
                mu0, logvar0 = embed_model.encoder(x0, tissue_onehot)
                z0 = mu0  # 也可以 sample_z
                mu1, logvar1 = embed_model.encoder(x1, tissue_onehot)
                z1 = mu1
            
            # 用 operator 预测
            z1_pred, A_theta, b_theta = operator_model(z0, tissue_idx, cond_vec)
            
            # 分布层面的 E-distance 损失
            ed2 = energy_distance(z1_pred, z1)
            
            # 稳定性正则
            stab_penalty = operator_model.spectral_penalty(
                max_allowed=train_config.max_spectral_norm
            )
            
            loss = train_config.lambda_e * ed2 + train_config.lambda_stab * stab_penalty
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_e += ed2.item() * z0.size(0)
            total_stab += stab_penalty.item()
            n_cells += z0.size(0)
        
        print(f"[Op] Epoch {epoch+1}/{train_config.n_epochs_operator}, "
              f"E^2={total_e/n_cells:.4f}, stab={total_stab:.4e}")
```

---

## 7. scPerturb 实验管线骨架

### 7.1 scPerturb 数据 loader 伪代码

```python
# data/scperturb_dataset.py
from torch.utils.data import Dataset
import torch

class SCPerturbEmbedDataset(Dataset):
    """用于训练 VAE 的统一 dataset"""
    def __init__(self, adata, tissue2idx: Dict[str,int]):
        self.adata = adata
        self.tissue2idx = tissue2idx
    
    def __len__(self):
        return self.adata.n_obs
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.adata.X[idx].astype("float32")).squeeze()
        tissue = self.adata.obs["tissue"][idx]
        t_idx = self.tissue2idx[tissue]
        n_tissues = len(self.tissue2idx)
        t_onehot = torch.zeros(n_tissues)
        t_onehot[t_idx] = 1.
        return {
            "x": x,
            "tissue_onehot": t_onehot,
            "tissue_idx": torch.tensor(t_idx, dtype=torch.long)
        }

class SCPerturbPairDataset(Dataset):
    """
    用于训练算子：
      每个样本是一对 (x0, x1) 属于同一 (dataset, cell_type, perturb,tissue)
    """
    def __init__(self, adata, cond_encoder, tissue2idx):
        self.adata = adata
        self.cond_encoder = cond_encoder  # 函数: obs -> cond_vec
        self.tissue2idx = tissue2idx
        
        # 这里需要预先构建所有 (θ) 下的 cell 对，应根据实际数据结构实现
        self.pairs = self._build_pairs()
    
    def _build_pairs(self):
        pairs = []
        # 伪代码：按 (dataset, tissue, cell_type, perturb) 分组，
        # 分别取 t0 和 t1 的细胞做配对/采样。
        # 实现需要看 scPerturb 的注释 & obs 信息。
        # ...
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        idx0, idx1 = self.pairs[idx]
        x0 = torch.from_numpy(self.adata.X[idx0].astype("float32")).squeeze()
        x1 = torch.from_numpy(self.adata.X[idx1].astype("float32")).squeeze()
        
        obs0 = self.adata.obs.iloc[idx0]
        tissue = obs0["tissue"]
        t_idx = self.tissue2idx[tissue]
        n_tissues = len(self.tissue2idx)
        t_onehot = torch.zeros(n_tissues)
        t_onehot[t_idx] = 1.
        
        cond_vec = self.cond_encoder(obs0)  # (cond_dim,)
        
        return {
            "x0": x0,
            "x1": x1,
            "tissue_onehot": t_onehot,
            "tissue_idx": torch.tensor(t_idx, dtype=torch.long),
            "cond_vec": cond_vec
        }
```

### 7.2 条件向量 cond_vec 编码骨架

```python
# utils/cond_encoder.py
import torch
from typing import Dict

class ConditionEncoder:
    def __init__(self,
                 perturb2idx: Dict[str,int],
                 tissue2idx: Dict[str,int],
                 batch2idx: Dict[str,int],
                 cond_dim: int):
        self.perturb2idx = perturb2idx
        self.tissue2idx = tissue2idx
        self.batch2idx = batch2idx
        self.cond_dim = cond_dim
        
        # 简单做：one-hot 拼接，然后用一个线性层降维到 cond_dim
        n_pert = len(perturb2idx)
        n_tissue = len(tissue2idx)
        n_batch = len(batch2idx)
        self.linear = torch.nn.Linear(n_pert + n_tissue + n_batch + 1, cond_dim)  # +1 for mLOY scalar
    
    def _one_hot(self, idx, n):
        v = torch.zeros(n)
        if idx is not None:
            v[idx] = 1.
        return v
    
    def __call__(self, obs_row):
        perturb = obs_row["perturbation"]
        tissue = obs_row["tissue"]
        batch = obs_row.get("batch", "batch0")
        mLOY = float(obs_row.get("mLOY_load", 0.0))  # 对 scPerturb 可以设为0
        
        p_idx = self.perturb2idx.get(perturb, 0)
        t_idx = self.tissue2idx.get(tissue, 0)
        b_idx = self.batch2idx.get(batch, 0)
        
        v_p = self._one_hot(p_idx, len(self.perturb2idx))
        v_t = self._one_hot(t_idx, len(self.tissue2idx))
        v_b = self._one_hot(b_idx, len(self.batch2idx))
        v_m = torch.tensor([mLOY], dtype=torch.float32)
        
        v = torch.cat([v_p, v_t, v_b, v_m], dim=0)  # (n_pert + n_tissue + n_batch + 1,)
        cond_vec = self.linear(v)  # (cond_dim,)
        return cond_vec
```

---

## 8. mLOY kidney + brain 跨组织实验骨架

mLOY 部分的数据加载逻辑会更复杂，这里给骨架：

```python
# data/mloy_dataset.py
from torch.utils.data import Dataset
import torch

class MLOYEmbedDataset(Dataset):
    """
    用于把 mLOY 肾脏/脑数据嵌入到同一潜空间。
    adata 需要包含:
      obs['tissue'] in {'kidney','brain'}
      obs['mLOY_prob'] cell-level P(LOY|x)
      obs['donor_id'], obs['mLOY_load'] donor-level
    """
    def __init__(self, adata, tissue2idx):
        self.adata = adata
        self.tissue2idx = tissue2idx
    
    def __len__(self):
        return self.adata.n_obs
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.adata.X[idx].astype("float32")).squeeze()
        tissue = self.adata.obs["tissue"][idx]
        t_idx = self.tissue2idx[tissue]
        n_tissues = len(self.tissue2idx)
        t_onehot = torch.zeros(n_tissues)
        t_onehot[t_idx] = 1.
        return {
            "x": x,
            "tissue_onehot": t_onehot,
            "tissue_idx": torch.tensor(t_idx, dtype=torch.long)
        }

class MLOYPairDataset(Dataset):
    """
    把 XY vs LOY 看成一种“伪扰动”。
    在同一组织、同一 cell type 内，从 XY 和 LOY 细胞中随机采样成对。
    """
    def __init__(self, adata, cond_encoder, tissue2idx):
        self.adata = adata
        self.cond_encoder = cond_encoder
        self.tissue2idx = tissue2idx
        self.pairs = self._build_pairs()
    
    def _build_pairs(self):
        pairs = []
        # 伪代码：按 tissue, cell_type 分组，在每组内
        #   XY_cells = ...
        #   LOY_cells = ...
        # 然后随机匹配 / 采样索引 
        # ...
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        idx_xy, idx_loy = self.pairs[idx]
        x0 = torch.from_numpy(self.adata.X[idx_xy].astype("float32")).squeeze()
        x1 = torch.from_numpy(self.adata.X[idx_loy].astype("float32")).squeeze()
        
        obs_xy = self.adata.obs.iloc[idx_xy]
        tissue = obs_xy["tissue"]
        t_idx = self.tissue2idx[tissue]
        n_tissues = len(self.tissue2idx)
        t_onehot = torch.zeros(n_tissues)
        t_onehot[t_idx] = 1.
        
        # mLOY_load 可以取 donor-level or cell-level prob
        cond_vec = self.cond_encoder(obs_xy)
        
        return {
            "x0": x0,
            "x1": x1,
            "tissue_onehot": t_onehot,
            "tissue_idx": torch.tensor(t_idx, dtype=torch.long),
            "cond_vec": cond_vec
        }
```

训练方法与 scPerturb 部分一样，只是 dataset 不同、cond_encoder 中 mLOY 分量不为 0。

---

## 9. 反事实模拟与虚拟细胞生成接口

最后给一个“统一虚拟细胞生成函数”，方便你做 mLOY 纠正、药物组合等反事实。

```python
# utils/virtual_cell.py
import torch

@torch.no_grad()
def encode_cells(vae: NBVAE, x: torch.Tensor, tissue_onehot: torch.Tensor, device="cuda"):
    vae.to(device)
    vae.eval()
    x = x.to(device)
    tissue_onehot = tissue_onehot.to(device)
    mu, logvar = vae.encoder(x, tissue_onehot)
    return mu  # (B, latent_dim)

@torch.no_grad()
def decode_cells(vae: NBVAE, z: torch.Tensor, tissue_onehot: torch.Tensor, device="cuda"):
    vae.to(device)
    vae.eval()
    z = z.to(device)
    tissue_onehot = tissue_onehot.to(device)
    mu_x, r_x = vae.decoder(z, tissue_onehot)
    # 返回 mu_x 作为虚拟表达均值
    return mu_x  # (B, G)

@torch.no_grad()
def apply_operator(operator: OperatorModel,
                   z: torch.Tensor,
                   tissue_idx: torch.Tensor,
                   cond_vec: torch.Tensor,
                   device="cuda"):
    operator.to(device)
    operator.eval()
    z = z.to(device)
    tissue_idx = tissue_idx.to(device)
    cond_vec = cond_vec.to(device)
    z_out, _, _ = operator(z, tissue_idx, cond_vec)
    return z_out

@torch.no_grad()
def virtual_cell_scenario(vae: NBVAE,
                          operator: OperatorModel,
                          x0: torch.Tensor,
                          tissue_onehot: torch.Tensor,
                          tissue_idx: torch.Tensor,
                          cond_vec_seq: torch.Tensor,
                          device="cuda"):
    """
    支持多步条件序列:
      cond_vec_seq: (T, cond_dim), 如 [mLOY, drugA+mLOY, ...]
    """
    # encode
    z = encode_cells(vae, x0, tissue_onehot, device=device)
    for t in range(cond_vec_seq.size(0)):
        cond = cond_vec_seq[t].unsqueeze(0).expand(z.size(0), -1)  # (B, cond_dim)
        z = apply_operator(operator, z, tissue_idx, cond, device=device)
    # decode
    x_virtual = decode_cells(vae, z, tissue_onehot, device=device)
    return x_virtual
```

使用示例（mLOY 纠正的反事实）：

```python
# example usage
# x_loy: (B,G) 来自肾脏或脑 LOY 细胞
# tissue_onehot, tissue_idx: 对应组织
# cond_encoder: 用真实 mLOY 信息构造 θ_mLOY, 以及构造 "virt-XY" 条件
obs_template = ...  # 对应一类细胞的元信息行
obs_mLOY = obs_template.copy()
obs_mLOY["perturbation"] = "LOY"
obs_mLOY["mLOY_load"] = 1.0

obs_XY = obs_template.copy()
obs_XY["perturbation"] = "LOY"
obs_XY["mLOY_load"] = 0.0  # 反事实

cond_mLOY = cond_encoder(obs_mLOY)   # (cond_dim,)
cond_XY   = cond_encoder(obs_XY)

cond_seq = torch.stack([cond_mLOY, cond_XY], dim=0)  # 先施加 mLOY, 再虚拟纠正

x_virtual = virtual_cell_scenario(
    vae=nbvae_model,
    operator=operator_model,
    x0=x_loy,
    tissue_onehot=tissue_onehot,
    tissue_idx=tissue_idx,
    cond_vec_seq=cond_seq,
    device="cuda"
)
```

---

## 最后小结

上面已经把整个方案拆成了一套 **完整可实现的代码骨架**：

* 潜空间 NB-VAE（encoder/decoder + ELBO）；
* E-distance 计算模块；
* 局部算子拟合（可选，用于初始化）；
* 全局低秩算子模型（含 α_k(θ)、低秩 B_k 与稳定性谱正则）；
* 两个训练阶段的循环（embedding + operator）；
* scPerturb 和 mLOY kidney/brain 的数据集骨架与条件向量编码；
* 一个统一的虚拟细胞/反事实模拟接口。


