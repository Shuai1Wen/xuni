# src/evaluation/metrics.py 问题快速摘要

生成时间：2025-11-20

## 🔴 严重问题（必须立即修复，否则代码无法运行）

### 问题1: comprehensive_evaluation - encoder接口不匹配
**行号**: 374, 384
**错误**: 传入`tissue_idx`而不是`tissue_onehot`
**影响**: 运行时抛出维度错误
**修复**: `tissue_onehot = F.one_hot(tissue_idx, num_classes=vae_model.n_tissues).float()`

### 问题2: comprehensive_evaluation - decoder方法不存在
**行号**: 381
**错误**: 调用`decoder.get_mean()`，但该方法不存在
**影响**: 运行时抛出AttributeError
**修复**: `mu_x1_pred, _ = vae_model.decoder(z1_pred, tissue_onehot); x1_pred = mu_x1_pred`

### 问题3: distribution_metrics - 除零风险
**行号**: 139-140
**错误**: 当batch_size=1时，`shape[0]-1=0`，导致除零
**影响**: 产生NaN
**修复**: `n = max(z.shape[0] - 1, 1)`

---

## 🟡 重要问题（影响准确性，强烈建议修复）

### 问题4: de_gene_prediction_metrics - pseudocount位置不当
**行号**: 200-202
**错误**: 先求均值再加eps，引入bias
**影响**: DE基因排序不准确
**修复**: `log2fc = np.log2((mean_x1 + eps) / (mean_x0 + eps))`

---

## 🔵 改进建议（提升代码质量）

### 问题5: reconstruction_metrics - R²语义不清晰
**行号**: 79-81
**问题**: 计算全局R²但未明确说明
**建议**: 改为per-gene R²或添加明确注释

### 问题6: 所有函数缺少输入验证
**问题**: 没有检查输入维度
**建议**: 添加`assert x.dim() == 2`等验证

---

## 逐函数问题列表

| 函数名 | 问题数 | 严重程度 | 详情 |
|--------|--------|----------|------|
| reconstruction_metrics | 1 | 🔵 改进 | R²语义不清晰 |
| distribution_metrics | 1 | 🔴 严重 | 除零风险 |
| de_gene_prediction_metrics | 1 | 🟡 重要 | pseudocount位置 |
| operator_quality_metrics | 0 | ✅ 正常 | - |
| comprehensive_evaluation | 2 | 🔴 严重 | 接口不匹配×2 |

---

## 修复优先级

**第一优先级（必须）**:
1. 修复问题1（comprehensive_evaluation - encoder）
2. 修复问题2（comprehensive_evaluation - decoder）
3. 修复问题3（distribution_metrics - 除零）

**第二优先级（重要）**:
4. 修复问题4（de_gene_prediction_metrics - pseudocount）

**第三优先级（改进）**:
5. 问题5和问题6

---

## 快速修复代码片段

### 修复1+2: comprehensive_evaluation (第367-394行)
```python
# 添加导入（文件顶部）
import torch.nn.functional as F

# 在for循环中添加
tissue_onehot = F.one_hot(tissue_idx, num_classes=vae_model.n_tissues).float()

# 替换3处调用
mu0, _ = vae_model.encoder(x0, tissue_onehot)  # 第374行
mu_x1_pred, _ = vae_model.decoder(z1_pred, tissue_onehot)  # 第381行
x1_pred = mu_x1_pred
mu1, _ = vae_model.encoder(x1, tissue_onehot)  # 第384行
```

### 修复3: distribution_metrics (第139-140行)
```python
# 替换
n_true = max(z_true.shape[0] - 1, 1)
n_pred = max(z_pred.shape[0] - 1, 1)
cov_true = (z_true_centered.T @ z_true_centered) / n_true
cov_pred = (z_pred_centered.T @ z_pred_centered) / n_pred
```

### 修复4: de_gene_prediction_metrics (第204-205行)
```python
# 替换
log2fc_true = np.log2((mean_x1_true + eps) / (mean_x0 + eps))
log2fc_pred = np.log2((mean_x1_pred + eps) / (mean_x0 + eps))
```

---

## 验证测试

修复后运行以下测试验证：

```python
# 测试1: 基本功能
from src.evaluation.metrics import comprehensive_evaluation
# ... 创建模型和数据
metrics = comprehensive_evaluation(vae, operator, dataloader)
print("✅ comprehensive_evaluation运行成功")

# 测试2: 边界条件
from src.evaluation.metrics import distribution_metrics
z_single = torch.randn(1, 32)
metrics = distribution_metrics(z_single, z_single, use_energy_distance=False)
assert not np.isnan(metrics['cov_frobenius_dist'])
print("✅ 单样本不会产生NaN")

# 测试3: 零值处理
from src.evaluation.metrics import de_gene_prediction_metrics
x0 = torch.zeros(10, 50)
x1 = torch.randn(10, 50).abs()
metrics = de_gene_prediction_metrics(x0, x1, x1)
print("✅ 零值基因处理正常")
```

---

## 文件状态

- ✅ 数学逻辑正确
- ❌ 接口一致性（2个严重错误）
- ⚠️ 数值稳定性（1个边界条件问题）
- ⚠️ 数据预处理（1个bias问题）

**总体评估**: 代码框架良好，但存在关键接口错误，**必须修复后才能使用**
