# src/evaluation/metrics.py 修复方案

## 修复1: comprehensive_evaluation - 接口不匹配（严重）

### 位置：第367-394行

### 原始代码：
```python
with torch.no_grad():
    for batch in dataloader:
        x0 = batch["x0"].to(device)
        x1 = batch["x1"].to(device)
        tissue_idx = batch["tissue_idx"].to(device)
        cond_vec = batch["cond_vec"].to(device)

        # 编码 x0 → z0
        mu0, _ = vae_model.encoder(x0, tissue_idx)  # ❌ 错误
        z0 = mu0  # 使用均值

        # 应用算子 z0 → z1_pred
        z1_pred = operator_model(z0, tissue_idx, cond_vec)

        # 解码 z1_pred → x1_pred
        x1_pred = vae_model.decoder.get_mean(z1_pred, tissue_idx)  # ❌ 错误

        # 真实z1（用于分布指标）
        mu1, _ = vae_model.encoder(x1, tissue_idx)  # ❌ 错误
        z1_true = mu1
```

### 修复后代码：
```python
import torch.nn.functional as F

with torch.no_grad():
    for batch in dataloader:
        x0 = batch["x0"].to(device)
        x1 = batch["x1"].to(device)
        tissue_idx = batch["tissue_idx"].to(device)
        cond_vec = batch["cond_vec"].to(device)

        # 将tissue_idx转换为one-hot编码
        tissue_onehot = F.one_hot(tissue_idx, num_classes=vae_model.n_tissues).float()

        # 编码 x0 → z0
        mu0, _ = vae_model.encoder(x0, tissue_onehot)  # ✅ 修复
        z0 = mu0  # 使用均值

        # 应用算子 z0 → z1_pred
        z1_pred = operator_model(z0, tissue_idx, cond_vec)

        # 解码 z1_pred → x1_pred
        mu_x1_pred, _ = vae_model.decoder(z1_pred, tissue_onehot)  # ✅ 修复
        x1_pred = mu_x1_pred

        # 真实z1（用于分布指标）
        mu1, _ = vae_model.encoder(x1, tissue_onehot)  # ✅ 修复
        z1_true = mu1
```

### 修复理由：
1. **encoder需要tissue_onehot**: encoder.forward接受(x, tissue_onehot)参数，不是tissue_idx
2. **decoder没有get_mean方法**: decoder.forward返回(mu, r)元组，需要解包第一个元素
3. **需要导入F.one_hot**: 在文件顶部添加导入

---

## 修复2: distribution_metrics - 协方差除零风险（严重）

### 位置：第136-142行

### 原始代码：
```python
# 协方差距离
z_true_centered = z_true - mean_true
z_pred_centered = z_pred - mean_pred
cov_true = (z_true_centered.T @ z_true_centered) / (z_true.shape[0] - 1)
cov_pred = (z_pred_centered.T @ z_pred_centered) / (z_pred.shape[0] - 1)
cov_dist = torch.norm(cov_true - cov_pred, p='fro').item()
metrics["cov_frobenius_dist"] = cov_dist
```

### 修复后代码：
```python
# 协方差距离
z_true_centered = z_true - mean_true
z_pred_centered = z_pred - mean_pred

# 防止除零：如果样本数为1，分母会是0
n_true = max(z_true.shape[0] - 1, 1)
n_pred = max(z_pred.shape[0] - 1, 1)

cov_true = (z_true_centered.T @ z_true_centered) / n_true
cov_pred = (z_pred_centered.T @ z_pred_centered) / n_pred
cov_dist = torch.norm(cov_true - cov_pred, p='fro').item()
metrics["cov_frobenius_dist"] = cov_dist
```

### 修复理由：
- 当batch size为1时，`z_true.shape[0] - 1 = 0`，导致除零
- 使用`max(..., 1)`确保分母至少为1

---

## 修复3: de_gene_prediction_metrics - pseudocount添加方式（重要）

### 位置：第198-205行

### 原始代码：
```python
# 计算log2 fold change（平均across细胞）
# 添加pseudocount避免log(0)
mean_x0 = x0_np.mean(axis=0) + eps
mean_x1_true = x1_true_np.mean(axis=0) + eps
mean_x1_pred = x1_pred_np.mean(axis=0) + eps

log2fc_true = np.log2(mean_x1_true / mean_x0)
log2fc_pred = np.log2(mean_x1_pred / mean_x0)
```

### 修复方案A（推荐）：在fold change计算时加eps
```python
# 计算log2 fold change（平均across细胞）
# 先计算均值，然后在fold change计算时添加pseudocount
mean_x0 = x0_np.mean(axis=0)
mean_x1_true = x1_true_np.mean(axis=0)
mean_x1_pred = x1_pred_np.mean(axis=0)

# 在分子和分母同时添加eps，避免log(0)和除零
log2fc_true = np.log2((mean_x1_true + eps) / (mean_x0 + eps))
log2fc_pred = np.log2((mean_x1_pred + eps) / (mean_x0 + eps))
```

### 修复方案B（替代）：使用maximum保证最小值
```python
# 计算log2 fold change（平均across细胞）
# 使用maximum确保均值不小于eps
mean_x0 = np.maximum(x0_np.mean(axis=0), eps)
mean_x1_true = np.maximum(x1_true_np.mean(axis=0), eps)
mean_x1_pred = np.maximum(x1_pred_np.mean(axis=0), eps)

log2fc_true = np.log2(mean_x1_true / mean_x0)
log2fc_pred = np.log2(mean_x1_pred / mean_x0)
```

### 修复理由：
- 原始代码：`mean + eps` 会引入bias（尤其对于接近0但非0的值）
- 方案A：在比值计算时加eps，保持比例关系
- 方案B：确保最小值，更简洁

---

## 修复4: reconstruction_metrics - R²计算语义不清晰（改进）

### 位置：第78-81行

### 原始代码：
```python
# R² score
ss_res = ((x_true - x_pred) ** 2).sum()
ss_tot = ((x_true - x_true.mean()) ** 2).sum()
r2 = float(1 - ss_res / (ss_tot + 1e-8))
```

### 修复方案A：明确注释当前是全局R²
```python
# R² score（全局：所有样本和基因的总体拟合度）
# 注意：全局R²会被高表达基因主导
ss_res = ((x_true - x_pred) ** 2).sum()
ss_tot = ((x_true - x_true.mean()) ** 2).sum()
r2 = float(1 - ss_res / (ss_tot + 1e-8))
```

### 修复方案B（推荐）：改为per-gene R²的统计量
```python
# R² score（per-gene，然后取统计量）
# 每个基因单独计算R²，然后取均值和中位数
ss_res_per_gene = ((x_true - x_pred) ** 2).sum(dim=0)  # (G,)
ss_tot_per_gene = ((x_true - x_true.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)  # (G,)
r2_per_gene = 1 - ss_res_per_gene / (ss_tot_per_gene + 1e-8)  # (G,)

# 返回字典中添加：
return {
    "mse": mse,
    "mae": mae,
    "pearson_mean": float(np.mean(pearson_corrs)) if pearson_corrs else 0.0,
    "pearson_median": float(np.median(pearson_corrs)) if pearson_corrs else 0.0,
    "spearman_mean": float(np.mean(spearman_corrs)) if spearman_corrs else 0.0,
    "r2_score_mean": float(r2_per_gene.mean()),  # 新增
    "r2_score_median": float(r2_per_gene.median()),  # 新增
}
```

### 修复理由：
- 方案A：保持原有逻辑，但明确语义
- 方案B：更符合生物学直觉（每个基因独立评估）

---

## 修复5: 添加输入维度验证（改进）

### 在每个函数开头添加验证

#### reconstruction_metrics
```python
def reconstruction_metrics(
    x_true: torch.Tensor,
    x_pred: torch.Tensor
) -> Dict[str, float]:
    """..."""
    # 输入验证
    assert x_true.dim() == 2, f"x_true应为2D张量 (B, G)，实际为{x_true.dim()}D"
    assert x_pred.dim() == 2, f"x_pred应为2D张量 (B, G)，实际为{x_pred.dim()}D"
    assert x_true.shape == x_pred.shape, \
        f"x_true和x_pred维度不匹配：{x_true.shape} vs {x_pred.shape}"

    # ... 原有代码
```

#### distribution_metrics
```python
def distribution_metrics(
    z_true: torch.Tensor,
    z_pred: torch.Tensor,
    use_energy_distance: bool = True
) -> Dict[str, float]:
    """..."""
    # 输入验证
    assert z_true.dim() == 2, f"z_true应为2D张量 (n, d)，实际为{z_true.dim()}D"
    assert z_pred.dim() == 2, f"z_pred应为2D张量 (m, d)，实际为{z_pred.dim()}D"
    assert z_true.shape[1] == z_pred.shape[1], \
        f"z_true和z_pred的特征维度不匹配：{z_true.shape[1]} vs {z_pred.shape[1]}"

    # ... 原有代码
```

#### de_gene_prediction_metrics
```python
def de_gene_prediction_metrics(
    x0: torch.Tensor,
    x1_true: torch.Tensor,
    x1_pred: torch.Tensor,
    top_k: int = 200,
    eps: float = 1e-8
) -> Dict[str, float]:
    """..."""
    # 输入验证
    assert x0.dim() == 2, f"x0应为2D张量 (B, G)，实际为{x0.dim()}D"
    assert x1_true.dim() == 2, f"x1_true应为2D张量 (B, G)，实际为{x1_true.dim()}D"
    assert x1_pred.dim() == 2, f"x1_pred应为2D张量 (B, G)，实际为{x1_pred.dim()}D"
    assert x0.shape == x1_true.shape == x1_pred.shape, \
        f"x0, x1_true, x1_pred维度不匹配：{x0.shape}, {x1_true.shape}, {x1_pred.shape}"

    B, G = x0.shape
    assert top_k <= G, f"top_k ({top_k}) 不能大于基因数量 ({G})"

    # ... 原有代码
```

---

## 完整修复后的文件导入部分

### 在文件顶部添加必要的导入：

```python
# -*- coding: utf-8 -*-
"""
评估指标模块
...
"""

import torch
import torch.nn.functional as F  # ← 新增：用于one-hot编码
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr, pearsonr
from ..utils.edistance import energy_distance
```

---

## 测试建议

### 测试1: 验证接口修复
```python
import torch
import torch.nn.functional as F
from src.models.nb_vae import NBVAE
from src.models.operator import OperatorModel
from src.evaluation.metrics import comprehensive_evaluation

# 创建模型
vae = NBVAE(n_genes=100, latent_dim=16, n_tissues=3)
operator = OperatorModel(latent_dim=16, n_tissues=3, n_response_bases=5, cond_dim=32)

# 创建测试数据
batch = {
    "x0": torch.randn(10, 100),
    "x1": torch.randn(10, 100),
    "tissue_idx": torch.randint(0, 3, (10,)),
    "cond_vec": torch.randn(10, 32)
}

# 测试one-hot转换
tissue_onehot = F.one_hot(batch["tissue_idx"], num_classes=3).float()
print(f"tissue_idx shape: {batch['tissue_idx'].shape}")
print(f"tissue_onehot shape: {tissue_onehot.shape}")

# 测试encoder
mu, logvar = vae.encoder(batch["x0"], tissue_onehot)
print(f"Encoder output: mu={mu.shape}, logvar={logvar.shape}")

# 测试decoder
z = torch.randn(10, 16)
mu_x, r = vae.decoder(z, tissue_onehot)
print(f"Decoder output: mu_x={mu_x.shape}, r={r.shape}")
```

### 测试2: 验证协方差除零修复
```python
from src.evaluation.metrics import distribution_metrics

# 测试单样本情况
z_true = torch.randn(1, 32)
z_pred = torch.randn(1, 32)
metrics = distribution_metrics(z_true, z_pred, use_energy_distance=False)
print(f"单样本协方差距离: {metrics['cov_frobenius_dist']}")
assert not np.isnan(metrics['cov_frobenius_dist']), "协方差计算应该不返回NaN"

# 测试正常情况
z_true = torch.randn(100, 32)
z_pred = torch.randn(100, 32)
metrics = distribution_metrics(z_true, z_pred, use_energy_distance=False)
print(f"正常样本协方差距离: {metrics['cov_frobenius_dist']}")
```

### 测试3: 验证pseudocount修复
```python
from src.evaluation.metrics import de_gene_prediction_metrics

# 创建测试数据（包含零值基因）
x0 = torch.randn(100, 50).abs()
x0[:, 0] = 0  # 第一个基因在所有样本中为0

x1_true = x0 + torch.randn(100, 50) * 0.1
x1_pred = x0 + torch.randn(100, 50) * 0.1

metrics = de_gene_prediction_metrics(x0, x1_true, x1_pred, top_k=10)
print(f"DE metrics: {metrics}")
assert not np.isnan(metrics['mean_log2fc_corr']), "log2FC计算应该不返回NaN"
```

---

## 修复应用顺序

1. **必须立即修复（P0）**：
   - ✅ 修复1: comprehensive_evaluation接口不匹配
   - ✅ 修复2: distribution_metrics协方差除零

2. **强烈建议修复（P1）**：
   - ⚠️ 修复3: de_gene_prediction_metrics的pseudocount

3. **建议改进（P2）**：
   - 📝 修复4: reconstruction_metrics的R²语义
   - 📝 修复5: 添加输入验证

---

## 修复验证清单

修复完成后，请验证：

□ comprehensive_evaluation函数可以正常运行，不抛出AttributeError
□ 单样本batch不会导致NaN或除零错误
□ DE基因预测的log2FC计算合理（零值基因不会dominate）
□ 所有函数都有清晰的语义注释
□ 输入验证可以捕获常见的错误用法
