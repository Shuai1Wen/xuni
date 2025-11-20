# 代码审查 - 发现的严重问题汇总

生成时间：2025-11-20
审查范围：所有新创建的实验分析代码

---

## 🚨 关键发现

**总计发现**：**18个问题**
- **P0（阻塞性，必须修复）**：6个
- **P1（严重，强烈建议修复）**：5个
- **P2（改进，提升质量）**：7个

**影响评估**：
- ❌ 当前代码**无法运行**（存在AttributeError、维度不匹配等错误）
- ⚠️ 即使能运行，部分计算结果也会不正确
- 📝 代码质量和鲁棒性需要提升

---

## 📂 文件1: src/evaluation/metrics.py

### P0-1: comprehensive_evaluation - encoder接口不匹配 ⛔ 阻塞

**位置**：第374行、第384行

**问题**：
```python
mu0, _ = vae_model.encoder(x0, tissue_idx)  # ❌ 错误
```

encoder需要`tissue_onehot` (B, n_tissues)，但传入了`tissue_idx` (B,)

**修复**：
```python
import torch.nn.functional as F
tissue_onehot = F.one_hot(tissue_idx, num_classes=vae_model.n_tissues).float()
mu0, _ = vae_model.encoder(x0, tissue_onehot)  # ✅ 正确
```

---

### P0-2: comprehensive_evaluation - decoder方法不存在 ⛔ 阻塞

**位置**：第381行

**问题**：
```python
x1_pred = vae_model.decoder.get_mean(z1_pred, tissue_idx)  # ❌ AttributeError
```

DecoderNB没有`get_mean`方法

**修复**：
```python
tissue_onehot = F.one_hot(tissue_idx, num_classes=vae_model.n_tissues).float()
x1_pred, _ = vae_model.decoder(z1_pred, tissue_onehot)  # ✅ 正确
```

---

### P0-3: distribution_metrics - 协方差计算除零风险 ⛔ 阻塞

**位置**：第139-140行

**问题**：
```python
cov_true = (z_true_centered.T @ z_true_centered) / (z_true.shape[0] - 1)
```

当batch_size=1时，分母为0，产生NaN

**修复**：
```python
n_true = max(z_true.shape[0] - 1, 1)
n_pred = max(z_pred.shape[0] - 1, 1)
cov_true = (z_true_centered.T @ z_true_centered) / n_true
cov_pred = (z_pred_centered.T @ z_pred_centered) / n_pred
```

---

### P1-1: de_gene_prediction_metrics - pseudocount添加方式不当 ⚠️ 严重

**位置**：第200-205行

**问题**：
```python
mean_x0 = x0_np.mean(axis=0) + eps  # ❌ 引入bias
log2fc_true = np.log2(mean_x1_true / mean_x0)
```

先加eps再计算比值，会引入bias

**修复**：
```python
mean_x0 = x0_np.mean(axis=0)
mean_x1_true = x1_true_np.mean(axis=0)
mean_x1_pred = x1_pred_np.mean(axis=0)

log2fc_true = np.log2((mean_x1_true + eps) / (mean_x0 + eps))  # ✅ 正确
log2fc_pred = np.log2((mean_x1_pred + eps) / (mean_x0 + eps))
```

---

### P2-1: 所有函数缺少输入维度验证 📝 改进

**位置**：所有函数

**建议**：
```python
def reconstruction_metrics(x_true, x_pred):
    assert x_true.dim() == 2, f"x_true应为2D，实际{x_true.dim()}D"
    assert x_pred.dim() == 2, f"x_pred应为2D，实际{x_pred.dim()}D"
    assert x_true.shape == x_pred.shape, "维度不匹配"
    # ... 原有代码
```

---

## 📂 文件2: scripts/experiments/eval_perturbation_prediction.py

### P0-4: encoder调用缺少one-hot转换 ⛔ 阻塞

**位置**：第158行、第168行

**问题**：与metrics.py问题相同

**修复**：
```python
# 方案1：直接使用batch中的tissue_onehot（推荐）
tissue_onehot = batch["tissue_onehot"].to(device)
mu0, _ = vae_model.encoder(x0, tissue_onehot)
mu1, _ = vae_model.encoder(x1, tissue_onehot)

# 方案2：手动转换
tissue_onehot = F.one_hot(tissue_idx, num_classes=vae_model.n_tissues).float()
mu0, _ = vae_model.encoder(x0, tissue_onehot)
```

---

### P0-5: decoder方法不存在 ⛔ 阻塞

**位置**：第165行

**问题**：与metrics.py问题相同

**修复**：
```python
tissue_onehot = batch["tissue_onehot"].to(device)
x1_pred, _ = vae_model.decoder(z1_pred, tissue_onehot)
```

---

### P0-6: operator返回值处理错误 ⛔ 阻塞

**位置**：第162行

**问题**：
```python
z1_pred = operator_model(z0, tissue_idx, cond_vec)  # ❌ 返回3个值
```

OperatorModel.forward返回`(z_out, A_theta, b_theta)`，只接收一个会导致类型错误

**修复**：
```python
z1_pred, _, _ = operator_model(z0, tissue_idx, cond_vec)  # ✅ 正确
```

---

## 📂 文件3: scripts/experiments/train_scperturb_baseline.py

### P1-2: VAE checkpoint缺少hidden_dim字段 ⚠️ 严重

**位置**：受影响位置在第257-263行，根本原因在src/train/train_embed_core.py

**问题**：
`train_embed_core.py`保存checkpoint时缺少`hidden_dim`字段

**修复**：
需要修改`src/train/train_embed_core.py`第138-151行：
```python
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": epoch,
    "history": history,
    "model_config": {
        "n_genes": model.n_genes,
        "latent_dim": model.latent_dim,
        "n_tissues": model.n_tissues,
        "hidden_dim": model.hidden_dim,  # ← 添加这行
    },
}
```

---

### P1-3: ConditionEncoder缺少embedding维度参数 ⚠️ 严重

**位置**：第220-224行

**问题**：
配置文件有`perturb_embed_dim`和`tissue_embed_dim`，但没有传递

**修复1** - 修改训练脚本：
```python
cond_encoder = ConditionEncoder.from_anndata(
    adata_train,
    cond_dim=config["model"]["cond_dim"],
    use_embedding=config["cond_encoder"]["use_embedding"],
    perturb_embed_dim=config["cond_encoder"]["perturb_embed_dim"],  # 添加
    tissue_embed_dim=config["cond_encoder"]["tissue_embed_dim"]     # 添加
)
```

**修复2** - 修改src/utils/cond_encoder.py的from_anndata方法：
```python
@classmethod
def from_anndata(
    cls,
    adata,
    cond_dim: int = 64,
    use_embedding: bool = True,
    perturb_embed_dim: int = 16,  # 添加参数
    tissue_embed_dim: int = 8     # 添加参数
) -> "ConditionEncoder":
    # ... 原有代码
    return cls(
        perturb2idx,
        tissue2idx,
        batch2idx,
        cond_dim=cond_dim,
        use_embedding=use_embedding,
        perturb_embed_dim=perturb_embed_dim,  # 传递参数
        tissue_embed_dim=tissue_embed_dim      # 传递参数
    )
```

---

### P1-4: ConditionEncoder checkpoint保存不完整 ⚠️ 严重

**位置**：第308-318行

**问题**：
保存时缺少`perturb_embed_dim`和`tissue_embed_dim`

**修复**：
```python
torch.save({
    "perturb2idx": cond_encoder.perturb2idx,
    "tissue2idx": cond_encoder.tissue2idx,
    "batch2idx": cond_encoder.batch2idx,
    "state_dict": cond_encoder.state_dict(),
    "config": {
        "cond_dim": config["model"]["cond_dim"],
        "use_embedding": config["cond_encoder"]["use_embedding"],
        "perturb_embed_dim": config["cond_encoder"]["perturb_embed_dim"],  # 添加
        "tissue_embed_dim": config["cond_encoder"]["tissue_embed_dim"]     # 添加
    }
}, encoder_path)
```

---

### P2-2: 缺少配置验证 📝 改进

**位置**：train_vae_phase和train_operator_phase开始处

**建议**：
```python
def validate_vae_config(config: dict) -> None:
    """验证VAE配置必需字段"""
    required_fields = {
        "model": ["n_genes", "latent_dim", "n_tissues", "hidden_dim"],
        "training": ["lr_embed", "batch_size", "n_epochs_embed"],
        "experiment": ["seed", "device"],
        "data": ["data_path"]
    }
    for section, fields in required_fields.items():
        if section not in config:
            raise ValueError(f"配置缺少必需部分: {section}")
        for field in fields:
            if field not in config[section]:
                raise ValueError(f"配置缺少必需字段: {section}.{field}")

# 在函数开始处调用
def train_vae_phase(args):
    config = load_config(args.config)
    validate_vae_config(config)  # 添加验证
    # ... 原有代码
```

---

### P2-3: VAE checkpoint加载缺少容错处理 📝 改进

**位置**：第254-264行

**建议**：
```python
vae_config = vae_checkpoint["model_config"]
hidden_dim = vae_config.get("hidden_dim", config["model"].get("hidden_dim", 512))

embed_model = NBVAE(
    n_genes=vae_config["n_genes"],
    latent_dim=vae_config["latent_dim"],
    n_tissues=vae_config["n_tissues"],
    hidden_dim=hidden_dim  # 使用容错后的值
)
```

---

## 📂 文件4: scripts/experiments/analyze_response_axes.py

### P1-5: condition_key解析逻辑有缺陷 ⚠️ 严重

**位置**：第125-140行

**问题**：
```python
adata.obs["condition_key"] = (
    adata.obs["perturbation"].astype(str) + "_" +
    adata.obs["tissue"].astype(str)
)
# ...
parts = cond_key.split("_")  # ❌ 如果perturbation="drug_A"会错误解析
perturbation = parts[0]
tissue = parts[1]
```

如果perturbation本身包含下划线（如"drug_A"），解析会错误：
- condition_key = "drug_A_kidney"
- split("_") = ["drug", "A", "kidney"]
- 结果：perturbation="drug", tissue="A" ❌

**修复方案1**（推荐）：
```python
# 构造时使用更可靠的分隔符
adata.obs["condition_key"] = (
    adata.obs["perturbation"].astype(str) + "||" +
    adata.obs["tissue"].astype(str)
)
# 解析时
parts = cond_key.split("||")
perturbation = parts[0]
tissue = parts[1]
```

**修复方案2**：
```python
# 使用rsplit从右边分割，限制分割次数
parts = cond_key.rsplit("_", 1)  # 只从最右边分割一次
perturbation = parts[0]
tissue = parts[1] if len(parts) > 1 else "unknown"
```

---

### P2-4: K=1时除零错误 📝 改进

**位置**：第190行、第357行

**问题**：
```python
off_diag_mean = (similarity_matrix.sum() - K) / (K * K - K)
```

当K=1时，分母为0

**修复**：
```python
if K > 1:
    off_diag_mean = (similarity_matrix.sum() - K) / (K * K - K)
else:
    off_diag_mean = 0.0
```

---

### P2-5: VAE加载冗余 📝 改进

**位置**：第52-60行

**问题**：
加载了vae_model但从未使用

**修复**：
删除VAE加载代码，或者实际使用它来计算响应基对基因表达的影响（如果需要）

---

### P2-6: OperatorModel缺少hidden_dim参数 📝 改进

**位置**：第64-75行

**建议**：
```python
operator_model = OperatorModel(
    latent_dim=operator_checkpoint["model_config"]["latent_dim"],
    n_tissues=operator_checkpoint["model_config"]["n_tissues"],
    n_response_bases=operator_checkpoint["model_config"]["n_response_bases"],
    cond_dim=operator_checkpoint["model_config"]["cond_dim"],
    max_spectral_norm=operator_checkpoint["model_config"]["max_spectral_norm"],
    hidden_dim=operator_checkpoint["model_config"].get("hidden_dim", 64)  # 添加
)
```

---

### P2-7: 未使用的参数tissue2idx 📝 改进

**位置**：第114行

**问题**：
`compute_activation_matrix`函数接受`tissue2idx`参数但从未使用

**修复**：
删除该参数

---

## 📊 问题统计

### 按严重程度

| 优先级 | 数量 | 说明 |
|--------|------|------|
| P0（阻塞性） | 6 | 必须立即修复，否则代码无法运行 |
| P1（严重） | 5 | 强烈建议修复，影响正确性或可维护性 |
| P2（改进） | 7 | 建议修复，提升代码质量 |
| **总计** | **18** | |

### 按文件

| 文件 | P0 | P1 | P2 | 总计 |
|------|----|----|-----|------|
| src/evaluation/metrics.py | 3 | 1 | 1 | 5 |
| eval_perturbation_prediction.py | 3 | 0 | 0 | 3 |
| train_scperturb_baseline.py | 0 | 3 | 3 | 6 |
| analyze_response_axes.py | 0 | 1 | 3 | 4 |
| **总计** | **6** | **5** | **7** | **18** |

### 按问题类型

| 类型 | 数量 |
|------|------|
| 接口不匹配（encoder/decoder） | 5 |
| 返回值处理错误 | 1 |
| 数值稳定性（除零） | 2 |
| 参数缺失/不完整 | 4 |
| 逻辑错误 | 1 |
| 代码质量（冗余/未使用） | 3 |
| 输入验证缺失 | 2 |

---

## 🔧 修复优先级路线图

### 第1步：修复P0问题（必须，预计1-2小时）

**修复顺序**：
1. **src/evaluation/metrics.py**（3个P0问题）
   - 添加`import torch.nn.functional as F`
   - 修复comprehensive_evaluation中所有encoder/decoder调用
   - 修复distribution_metrics的除零问题

2. **scripts/experiments/eval_perturbation_prediction.py**（3个P0问题）
   - 使用batch中的tissue_onehot
   - 修复operator返回值处理

**验证**：运行一个小batch的评估脚本，确保不报错

---

### 第2步：修复P1问题（强烈建议，预计1-2小时）

**修复顺序**：
1. **src/train/train_embed_core.py**
   - 添加hidden_dim到checkpoint

2. **src/utils/cond_encoder.py**
   - 修改from_anndata支持embedding维度参数

3. **scripts/experiments/train_scperturb_baseline.py**
   - 传递embedding维度参数
   - 完善checkpoint保存

4. **scripts/experiments/analyze_response_axes.py**
   - 修复condition_key解析逻辑

**验证**：运行完整训练流程，确保checkpoint能正确保存和加载

---

### 第3步：修复P2问题（可选，预计30-60分钟）

添加输入验证、删除冗余代码、完善容错处理

---

## 📝 修复检查清单

修复完成后，请逐项检查：

### 代码可运行性
- [ ] metrics.py中的comprehensive_evaluation能正常运行
- [ ] eval_perturbation_prediction.py能完整运行
- [ ] train_scperturb_baseline.py的VAE和Operator阶段都能运行
- [ ] analyze_response_axes.py能正常分析

### 数值正确性
- [ ] DE基因指标计算正确（pseudocount修复）
- [ ] 协方差计算不会产生NaN
- [ ] condition_key解析正确

### 模型一致性
- [ ] VAE checkpoint包含所有必需字段
- [ ] ConditionEncoder参数与配置一致
- [ ] 模型能正确保存和加载

### 代码质量
- [ ] 添加了输入维度验证
- [ ] 删除了冗余代码
- [ ] 添加了配置验证

---

## 🎯 下一步行动

**建议立即执行**：
1. 应用所有P0修复（阻塞性问题）
2. 测试修复后的代码
3. 应用P1修复（严重问题）
4. 重新提交代码

**预计总耗时**：3-5小时（包括测试）

---

**生成时间**：2025-11-20
**审查者**：Claude Code (Sonnet 4.5)
**审查覆盖率**：100%（所有新创建的核心代码文件）
