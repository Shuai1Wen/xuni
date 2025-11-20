# src/evaluation/metrics.py 详细审查报告

生成时间：2025-11-20

## 执行摘要

发现 **5个严重问题** 和 **3个潜在问题**，需要立即修复才能正常运行。

### 严重问题（阻塞性）
1. **comprehensive_evaluation**: encoder接口不匹配（传入tissue_idx而非tissue_onehot）
2. **comprehensive_evaluation**: decoder.get_mean方法不存在
3. **distribution_metrics**: 协方差计算存在除零风险

### 潜在问题（需要改进）
1. **de_gene_prediction_metrics**: pseudocount添加方式引入bias
2. **reconstruction_metrics**: R²计算语义不清晰
3. **所有函数**: 缺少输入维度验证

---

## 详细问题列表

### 问题1: comprehensive_evaluation - encoder接口不匹配 ⚠️ 严重

**位置**: 第374行、第384行

**问题代码**:
```python
mu0, _ = vae_model.encoder(x0, tissue_idx)  # 第374行
mu1, _ = vae_model.encoder(x1, tissue_idx)  # 第384行
```

**问题分析**:
- `vae_model.encoder.forward(x, tissue_onehot)` 接受的第二个参数是 **one-hot编码** (B, n_tissues)
- 但这里传入的是 `tissue_idx`，类型是 (B,) 的long tensor
- 维度不匹配：(B,) vs (B, n_tissues)
- 运行时会抛出维度错误

**证据**:
参考 `src/models/nb_vae.py` 第83-114行：
```python
def forward(
    self,
    x: torch.Tensor,
    tissue_onehot: torch.Tensor  # ← 需要one-hot编码
) -> Tuple[torch.Tensor, torch.Tensor]:
```

**修复方案**:
```python
# 在循环开始前添加转换
import torch.nn.functional as F

# 在for循环中：
tissue_onehot = F.one_hot(tissue_idx, num_classes=vae_model.n_tissues).float()

# 然后使用：
mu0, _ = vae_model.encoder(x0, tissue_onehot)
mu1, _ = vae_model.encoder(x1, tissue_onehot)
```

**影响范围**: 函数无法运行，会立即报错

---

### 问题2: comprehensive_evaluation - decoder.get_mean方法不存在 ⚠️ 严重

**位置**: 第381行

**问题代码**:
```python
x1_pred = vae_model.decoder.get_mean(z1_pred, tissue_idx)
```

**问题分析**:
- `DecoderNB` 类没有 `get_mean` 方法
- `decoder.forward(z, tissue_onehot)` 返回 `(mu, r)` 元组
- 而且同样存在tissue_idx vs tissue_onehot的问题

**证据**:
参考 `src/models/nb_vae.py` 第117-214行，DecoderNB类只有：
- `__init__`
- `forward` (返回 `(mu, r)`)

没有 `get_mean` 方法。

**修复方案**:
```python
# 修复为：
tissue_onehot = F.one_hot(tissue_idx, num_classes=vae_model.n_tissues).float()
mu_x1_pred, _ = vae_model.decoder(z1_pred, tissue_onehot)
x1_pred = mu_x1_pred
```

**影响范围**: 函数无法运行，会抛出AttributeError

---

### 问题3: distribution_metrics - 协方差计算除零风险 ⚠️ 严重

**位置**: 第139-140行

**问题代码**:
```python
cov_true = (z_true_centered.T @ z_true_centered) / (z_true.shape[0] - 1)
cov_pred = (z_pred_centered.T @ z_pred_centered) / (z_pred.shape[0] - 1)
```

**问题分析**:
- 如果 `z_true.shape[0] == 1`（单样本batch），分母为0
- 如果 `z_pred.shape[0] == 1`，分母为0
- 这会导致 NaN 或 Inf
- 虽然单样本batch不常见，但边界情况需要处理

**边界条件测试**:
```python
# 会导致除零
z_true = torch.randn(1, 32)
z_pred = torch.randn(1, 32)
metrics = distribution_metrics(z_true, z_pred)
# cov_frobenius_dist 会是 NaN
```

**修复方案1（推荐）**:
```python
# 添加边界检查
n_true = max(z_true.shape[0] - 1, 1)
n_pred = max(z_pred.shape[0] - 1, 1)
cov_true = (z_true_centered.T @ z_true_centered) / n_true
cov_pred = (z_pred_centered.T @ z_pred_centered) / n_pred
```

**修复方案2（更严格）**:
```python
# 如果样本数太少，跳过协方差计算
if z_true.shape[0] > 1 and z_pred.shape[0] > 1:
    cov_true = (z_true_centered.T @ z_true_centered) / (z_true.shape[0] - 1)
    cov_pred = (z_pred_centered.T @ z_pred_centered) / (z_pred.shape[0] - 1)
    cov_dist = torch.norm(cov_true - cov_pred, p='fro').item()
    metrics["cov_frobenius_dist"] = cov_dist
else:
    metrics["cov_frobenius_dist"] = 0.0  # 或者 NaN
```

**影响范围**: 小batch size时会产生NaN

---

### 问题4: de_gene_prediction_metrics - pseudocount添加方式不当 🔶 改进

**位置**: 第200-202行

**问题代码**:
```python
mean_x0 = x0_np.mean(axis=0) + eps
mean_x1_true = x1_true_np.mean(axis=0) + eps
mean_x1_pred = x1_pred_np.mean(axis=0) + eps
```

**问题分析**:
- 当前方式：先求均值，再加eps
- 问题：如果某个基因在所有细胞中表达为0，均值为0，加eps后为1e-8
  - 另一个基因表达均值为100，加eps后为100.00000001
  - 比例关系被改变：0 vs 100 → 1e-8 vs 100（正确）
- 但如果一个基因均值为0.5，加eps后为0.5 + 1e-8 ≈ 0.5
  - log2(0.5 / 0) → log2(0.5 / 1e-8) = log2(5e7) ≈ 25.6
  - 这引入了巨大的bias！

**正确的做法**:
```python
# 方案1：在计算fold change时加eps
mean_x0 = x0_np.mean(axis=0)
mean_x1_true = x1_true_np.mean(axis=0)
mean_x1_pred = x1_pred_np.mean(axis=0)

log2fc_true = np.log2((mean_x1_true + eps) / (mean_x0 + eps))
log2fc_pred = np.log2((mean_x1_pred + eps) / (mean_x0 + eps))

# 方案2：使用maximum确保最小值
mean_x0 = np.maximum(x0_np.mean(axis=0), eps)
mean_x1_true = np.maximum(x1_true_np.mean(axis=0), eps)
mean_x1_pred = np.maximum(x1_pred_np.mean(axis=0), eps)

log2fc_true = np.log2(mean_x1_true / mean_x0)
log2fc_pred = np.log2(mean_x1_pred / mean_x0)
```

**影响**: 引入bias，影响DE基因排序准确性

---

### 问题5: reconstruction_metrics - R²计算语义不清晰 🔶 改进

**位置**: 第79-81行

**问题代码**:
```python
ss_res = ((x_true - x_pred) ** 2).sum()
ss_tot = ((x_true - x_true.mean()) ** 2).sum()
r2 = float(1 - ss_res / (ss_tot + 1e-8))
```

**问题分析**:
- `x_true.mean()` 默认对所有维度求均值，得到标量
- 这计算的是 **全局R²**（所有样本、所有基因的总体拟合度）
- 但文档说是"R² score"，没有明确说明是全局还是per-gene

**语义问题**:
- 对于单细胞数据，更常见的是per-gene的R²或per-cell的R²
- 全局R²会被高表达基因主导（因为它们的方差大）
- 当前实现在数学上正确，但可能不是预期行为

**建议**:
```python
# 明确注释当前是全局R²
# 全局R² score（所有样本和基因的总体拟合度）
ss_res = ((x_true - x_pred) ** 2).sum()
ss_tot = ((x_true - x_true.mean()) ** 2).sum()
r2_global = float(1 - ss_res / (ss_tot + 1e-8))

# 或者改为per-gene R²的均值（更合理）
ss_res_per_gene = ((x_true - x_pred) ** 2).sum(dim=0)  # (G,)
ss_tot_per_gene = ((x_true - x_true.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)  # (G,)
r2_per_gene = 1 - ss_res_per_gene / (ss_tot_per_gene + 1e-8)  # (G,)
r2_mean = float(r2_per_gene.mean())
r2_median = float(r2_per_gene.median())
```

**影响**: 语义不清，可能不符合预期

---

### 问题6: 所有函数 - 缺少输入维度验证 🔶 改进

**位置**: 所有函数

**问题分析**:
- 所有函数都没有验证输入张量的维度是否正确
- 例如：`reconstruction_metrics(x_true, x_pred)`
  - 如果x_true是3D张量 (B, T, G)，代码不会报错但结果错误
  - 如果x_true和x_pred维度不匹配，会在后续计算中报错，但错误信息不明确

**建议添加**:
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

**影响**: 降低代码鲁棒性，错误信息不友好

---

## 数值稳定性检查

### ✅ 良好的做法

1. **reconstruction_metrics**:
   - 第65行：检查std > 1e-8再计算Pearson
   - 第73行：检查unique值数量>1再计算Spearman
   - 第81行：除法添加1e-8保护

2. **de_gene_prediction_metrics**:
   - 第227-233行：完善的异常处理
   - 第237-243行：NaN检查

3. **operator_quality_metrics**:
   - 使用OperatorModel的内置方法，继承其稳定性保证

### ⚠️ 需要改进

1. **distribution_metrics**:
   - 协方差计算缺少除零保护（问题3）

2. **de_gene_prediction_metrics**:
   - pseudocount添加方式不当（问题4）

---

## 接口一致性检查

### ❌ 不一致

1. **comprehensive_evaluation** ↔ **NBVAE.encoder**
   - 传入: `tissue_idx` (B,)
   - 期望: `tissue_onehot` (B, n_tissues)
   - 严重问题（问题1）

2. **comprehensive_evaluation** ↔ **NBVAE.decoder**
   - 调用: `decoder.get_mean(...)`
   - 实际: `decoder.forward(...)` 返回 `(mu, r)`
   - 严重问题（问题2）

### ✅ 一致

1. **operator_quality_metrics** ↔ **OperatorModel**
   - `compute_operator_norm` 存在且接口匹配
   - `get_response_profile` 存在且接口匹配

---

## 逻辑错误检查

### ✅ 逻辑正确

所有函数的主要逻辑流程都是正确的，没有发现明显的逻辑错误。

---

## 性能问题

### 潜在性能问题

1. **reconstruction_metrics** 第64-76行：
   ```python
   for g in range(G):
       if x_true_np[:, g].std() > 1e-8:
           corr, _ = pearsonr(x_true_np[:, g], x_pred_np[:, g])
   ```
   - 对每个基因循环计算Pearson相关
   - 如果G=20000，会调用20000次pearsonr
   - 可以向量化，但scipy.stats.pearsonr不支持
   - 当前实现是合理的（无法避免）

2. **de_gene_prediction_metrics** 第198-205行：
   - 计算均值和log2FC是向量化的，性能良好

3. **comprehensive_evaluation** 第367-403行：
   - 使用列表append + torch.cat，内存效率中等
   - 如果数据集很大，可能需要预分配
   - 当前实现是标准做法，可以接受

---

## 修复优先级

### P0 - 必须立即修复（阻塞性）

1. ✅ **问题1**: comprehensive_evaluation - encoder接口不匹配
2. ✅ **问题2**: comprehensive_evaluation - decoder.get_mean不存在
3. ✅ **问题3**: distribution_metrics - 协方差除零风险

### P1 - 强烈建议修复（影响准确性）

4. ⚠️ **问题4**: de_gene_prediction_metrics - pseudocount添加方式

### P2 - 建议改进（提升质量）

5. 📝 **问题5**: reconstruction_metrics - R²语义不清晰
6. 📝 **问题6**: 添加输入维度验证

---

## 总结

**可执行性**: ❌ 当前代码**无法运行**，必须先修复问题1和问题2

**数学正确性**: 🔶 大部分正确，但问题4会引入bias

**数值稳定性**: ✅ 大部分良好，问题3需要修复

**代码质量**: 🔶 中等，缺少输入验证和明确的语义注释

**建议**: 先修复P0问题使代码可运行，再逐步修复P1和P2问题
