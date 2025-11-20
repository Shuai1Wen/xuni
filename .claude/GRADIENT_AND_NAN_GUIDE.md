# 梯度流动与数值稳定性技术指南

## 目录
1. [梯度失效问题检测与防护](#梯度失效)
2. [NaN/Inf产生点完全图谱](#nan产生点)
3. [数值稳定性最佳实践](#最佳实践)
4. [调试技巧](#调试技巧)

---

## 梯度失效问题检测与防护 {#梯度失效}

### 常见梯度失效模式

####  1. **张量累加导致的梯度断裂** ⚠️ [已修复]

**症状**: 循环累加张量时，只有最后一次操作保留梯度

```python
# ❌ 错误模式 - 梯度断裂
result = torch.tensor(0.0, requires_grad=True)
for data in batches:
    result = result + compute(data)  # ← 每次创建新节点，旧梯度丢失
```

**原因**: `result = result + ...` 创建新张量，旧的计算图被detach

**修复**:
```python
# ✅ 正确模式 - 保持梯度完整
chunks = []
for data in batches:
    chunks.append(compute(data))
result = torch.stack(chunks).sum()  # 所有块的梯度都保留
```

**项目中的应用**: `src/utils/edistance.py:242-269`

---

#### 2. **不必要的no_grad包裹** ⚠️

**症状**: 需要梯度的计算被no_grad包裹

```python
# ❌ 错误 - 切断梯度
with torch.no_grad():
    z = model.encoder(x)  # z不会有梯度
    loss = compute_loss(z)  # 无法反向传播到encoder
```

**检查清单**:
- [ ] 确认哪些模块需要训练
- [ ] 确认优化器包含哪些参数
- [ ] 只对真正冻结的部分使用no_grad

**项目中的应用**: `src/train/train_operator_core.py:71-74`
- embed_model不在优化器中 → 正确使用no_grad
- operator_model在优化器中 → 不使用no_grad

---

#### 3. **detach()误用** ⚠️

**症状**: 中间变量被detach，后续梯度无法传播

```python
# ❌ 错误 - 切断梯度链
z = model.encoder(x)
z_detached = z.detach()  # ← 切断梯度
loss = compute_loss(z_detached)  # 梯度无法传到encoder
```

**何时使用detach**:
- ✅ 用于记录/可视化（不需要梯度）
- ✅ 实现stop-gradient操作（如目标网络）
- ❌ 不要用于需要反向传播的中间变量

---

#### 4. **.item()过早调用** ⚠️

**症状**: 将标量张量转为Python数值，切断梯度

```python
# ❌ 错误 - 切断梯度
loss_value = loss.item()  # 转为Python float
combined_loss = loss_value + regularization  # 无法反向传播

# ✅ 正确 - 保持张量
combined_loss = loss + regularization  # 保持梯度
loss_value = loss.item()  # 仅用于记录
```

**规则**: 只在记录日志时调用.item()，计算中保持张量形式

---

### 梯度流动检查工具

#### 工具1: 梯度存在性检查
```python
def check_gradient_flow(model):
    """检查模型参数是否有梯度"""
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                print(f"⚠️  {name}: 无梯度")
            elif torch.isnan(param.grad).any():
                print(f"❌ {name}: 梯度包含NaN")
            elif param.grad.abs().max() < 1e-8:
                print(f"⚠️  {name}: 梯度接近零 (max={param.grad.abs().max():.2e})")
            else:
                print(f"✅ {name}: 梯度正常 (norm={param.grad.norm():.4f})")
```

#### 工具2: 中间变量梯度追踪
```python
def trace_gradient_path(loss, *variables):
    """追踪从loss到各变量的梯度路径"""
    loss.backward(retain_graph=True)
    for i, var in enumerate(variables):
        if var.grad is not None:
            print(f"✅ var{i}: 梯度范数 = {var.grad.norm():.4e}")
        else:
            print(f"❌ var{i}: 无梯度 - 可能被detach或使用了no_grad")
```

**使用示例**:
```python
# 在训练循环中
loss.backward()
check_gradient_flow(model)

# 调试特定路径
z0 = torch.randn(10, 32, requires_grad=True)
z1 = operator_model(z0, ...)
loss = criterion(z1, target)
trace_gradient_path(loss, z0, z1)
```

---

## NaN/Inf产生点完全图谱 {#nan产生点}

### 数学运算风险矩阵

| 运算 | 产生NaN的条件 | 产生Inf的条件 | 项目中的位置 | 防护措施 |
|-----|-------------|-------------|------------|---------|
| `exp(x)` | x是NaN | x > 88 | nb_vae.py:246, 474 | clamp x到[-10,10] |
| `log(x)` | x ≤ 0 | - | nb_vae.py:315-317 | x + eps |
| `sqrt(x)` | x < 0 | - | edistance.py:85 | clamp x到>=0 |
| `x / y` | y=0 | x很大且y很小 | 多处 | y + eps |
| `lgamma(x)` | x < 0 | x ≈ 0 | nb_vae.py:306-308 | clamp x到>eps |
| `x ** y` | x<0且y非整数 | 指数很大 | - | 避免或clamp |
| `bmm(A, B)` | A或B包含NaN | 元素过大累积 | operator.py | 输入验证 |

### VAE模型中的NaN传播路径

```mermaid
graph TD
    A[logvar过大 >20] --> B[exp溢出 → std=Inf]
    B --> C[z采样 = mu + std*eps → z=Inf]
    C --> D[Decoder输出 → mu_x=Inf]
    D --> E[log(mu_x) → log(Inf) = Inf]
    E --> F[loss=Inf → backward → 所有参数=NaN]
    
    G[r过小 <1e-8] --> H[lgamma(r) → 极大负值]
    H --> I[log_p异常 → loss异常]
    I --> F
    
    J[mu=0] --> K[log(mu+eps) → log(eps) = -18]
    K --> L[如果eps过小 → 数值下溢]
```

**防护点**:
1. ✅ sample_z: clamp logvar到[-10, 10]
2. ✅ elbo_loss: clamp logvar_z到[-10, 10]
3. ✅ nb_log_likelihood: clamp r到>eps, x到>=0
4. ✅ DecoderNB: r添加eps下界
5. ✅ 所有log运算: 参数+eps

### Operator模型中的NaN传播路径

```
z0正常 → OperatorModel → z1_pred 
                ↓ (如果算子不稳定)
          矩阵乘法累积误差 → z1_pred爆炸 → Inf
                ↓
          energy_distance计算 → dist过大 → overflow
                ↓
          loss=NaN → backward → 参数=NaN
```

**防护点**:
1. ✅ spectral_penalty: 约束算子范数<1.05
2. ✅ power iteration: 归一化防止爆炸
3. ✅ energy_distance: 分块计算防止内存溢出
4. ✅ 训练循环: 检测z1_pred的NaN

---

## 数值稳定性最佳实践 {#最佳实践}

### 原则1: 多层防护（Defense in Depth）

```python
# 层次1: 模型输出层添加保护
mu = F.softplus(self.fc_mu(h)) + eps  # 确保>0

# 层次2: 损失函数输入验证
def nb_log_likelihood(x, mu, r, eps=1e-8):
    r = torch.clamp(r, min=eps)  # 确保r>0
    x = torch.clamp(x, min=0.0)  # 确保x>=0
    ...

# 层次3: 运行时检测
if torch.isnan(loss) or torch.isinf(loss):
    raise RuntimeError("数值不稳定")
```

### 原则2: 参数范围约束

```python
# ✅ 好的做法
logvar = torch.clamp(logvar, min=-10, max=10)  # 明确范围
std = torch.exp(0.5 * logvar)  # std ∈ [0.0067, 148.4]

# ❌ 坏的做法
std = torch.exp(0.5 * logvar)  # std可能是任何值
```

**推荐约束**:
- logvar: [-10, 10] → std: [0.0067, 148.4]
- log_r: [-5, 5] → r: [0.0067, 148.4]
- 算子谱范数: [0, 1.05]

### 原则3: 数值稳定的数学变换

```python
# ❌ 不稳定: log(a/b)
log_ratio = torch.log(a / b)  # 如果a,b都很小，a/b可能下溢

# ✅ 稳定: log(a) - log(b)
log_ratio = torch.log(a + eps) - torch.log(b + eps)

# ❌ 不稳定: exp(a) - exp(b)
diff = torch.exp(a) - torch.exp(b)  # 可能溢出

# ✅ 稳定: log-sum-exp技巧
max_val = torch.max(a, b)
diff = torch.exp(a - max_val) - torch.exp(b - max_val)
```

### 原则4: 梯度裁剪 vs NaN检测

```python
# ❌ 错误: 裁剪可能掩盖NaN
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

# ✅ 正确: 先检测再裁剪
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
if torch.isnan(total_norm) or torch.isinf(total_norm):
    raise RuntimeError(f"梯度异常: {total_norm}")
```

---

## 调试技巧 {#调试技巧}

### 技巧1: 二分查找NaN源头

```python
def debug_forward_pass(model, x, tissue_onehot):
    """逐层检查前向传播，找出NaN产生点"""
    print("输入 x:", check_tensor(x))
    
    # Encoder
    h = model.encoder.input_layer(x)
    print("Encoder hidden:", check_tensor(h))
    
    mu_z, logvar_z = model.encoder(x, tissue_onehot)
    print("mu_z:", check_tensor(mu_z))
    print("logvar_z:", check_tensor(logvar_z))
    
    z = sample_z(mu_z, logvar_z)
    print("z:", check_tensor(z))
    
    # Decoder
    mu_x, r_x = model.decoder(z, tissue_onehot)
    print("mu_x:", check_tensor(mu_x))
    print("r_x:", check_tensor(r_x))
    
    # Loss
    log_p = nb_log_likelihood(x, mu_x, r_x)
    print("log_p:", check_tensor(log_p))

def check_tensor(t):
    """检查张量的统计信息"""
    if torch.isnan(t).any():
        return f"❌ 包含NaN ({torch.isnan(t).sum()}/{t.numel()})"
    if torch.isinf(t).any():
        return f"⚠️  包含Inf ({torch.isinf(t).sum()}/{t.numel()})"
    return f"✅ min={t.min():.2e}, max={t.max():.2e}, mean={t.mean():.2e}"
```

### 技巧2: 梯度可视化

```python
def visualize_gradients(model, loss):
    """可视化各层梯度分布"""
    loss.backward(retain_graph=True)
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for idx, (name, param) in enumerate(model.named_parameters()):
        if param.grad is not None and idx < 6:
            grad_flat = param.grad.cpu().numpy().flatten()
            axes[idx].hist(grad_flat, bins=50, alpha=0.7)
            axes[idx].set_title(f"{name}\nμ={grad_flat.mean():.2e}, σ={grad_flat.std():.2e}")
            axes[idx].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('gradient_histogram.png')
```

### 技巧3: 参数监控Hook

```python
def register_nan_hooks(model):
    """注册hook，自动监控NaN产生"""
    def check_nan(module, input, output):
        if isinstance(output, torch.Tensor):
            if torch.isnan(output).any():
                print(f"❌ NaN detected in {module.__class__.__name__}")
                print(f"   Input stats: {check_tensor(input[0])}")
                raise RuntimeError(f"NaN in {module}")
    
    for name, module in model.named_modules():
        module.register_forward_hook(check_nan)
```

### 技巧4: 保存崩溃前的状态

```python
def train_with_crash_recovery(model, loader, ...):
    """训练循环，崩溃前自动保存状态"""
    try:
        for epoch in range(n_epochs):
            for batch_idx, batch in enumerate(loader):
                # 定期保存checkpoint
                if batch_idx % 100 == 0:
                    torch.save({
                        'model': model.state_dict(),
                        'batch': batch,
                        'epoch': epoch
                    }, 'last_good_state.pt')
                
                loss = train_step(model, batch)
                
                # NaN检测
                if torch.isnan(loss):
                    print(f"⚠️  NaN at epoch {epoch}, batch {batch_idx}")
                    print("最后正常状态已保存至 last_good_state.pt")
                    raise RuntimeError("训练崩溃")
                    
    except RuntimeError as e:
        print(f"错误: {e}")
        print("加载最后正常状态:")
        print("  state = torch.load('last_good_state.pt')")
        print("  model.load_state_dict(state['model'])")
        print("  batch = state['batch']")
        raise
```

### 技巧5: 单元测试数值稳定性

```python
def test_extreme_inputs():
    """测试极端输入下的稳定性"""
    model = NBVAE(n_genes=100, latent_dim=16, n_tissues=2)
    
    # 测试1: 零输入
    x_zero = torch.zeros(10, 100)
    tissue = torch.zeros(10, 2)
    loss, _ = elbo_loss(x_zero, tissue, model)
    assert not torch.isnan(loss), "零输入不应产生NaN"
    
    # 测试2: 非常大的输入
    x_large = torch.ones(10, 100) * 1e6
    loss, _ = elbo_loss(x_large, tissue, model)
    assert not torch.isnan(loss), "大输入不应产生NaN"
    
    # 测试3: 非常小的输入
    x_small = torch.ones(10, 100) * 1e-6
    loss, _ = elbo_loss(x_small, tissue, model)
    assert not torch.isnan(loss), "小输入不应产生NaN"
    
    print("✅ 所有极端输入测试通过")
```

---

## 快速参考卡

### NaN检测清单
```python
# 训练前
□ 检查输入数据是否包含NaN: torch.isnan(x).any()
□ 检查模型初始化是否正常: check_gradient_flow(model)

# 训练中
□ 每个batch后检测loss: if torch.isnan(loss): raise
□ 定期检查参数: 是否有参数变为NaN
□ 监控梯度范数: 是否异常大或异常小

# 训练后
□ 检查最终模型参数: 所有参数都应是有限值
□ 验证集测试: 确保推理时不产生NaN
```

### 梯度流动检查清单
```python
□ 确认所有需要训练的模块在优化器中
□ 确认没有不必要的detach()
□ 确认没有不必要的no_grad
□ 确认中间变量保持张量形式
□ 使用autograd.grad()手动验证梯度路径
```

### 数值稳定性清单
```python
□ 所有exp运算的输入都clamp到合理范围
□ 所有log运算的参数都+eps
□ 所有除法的分母都+eps
□ 所有sqrt的参数都clamp到>=0
□ 参数初始化避免过大或过小
```

---

## 参考资源

### 官方文档
- [PyTorch Autograd原理](https://pytorch.org/docs/stable/notes/autograd.html)
- [数值稳定性最佳实践](https://pytorch.org/docs/stable/notes/numerical_accuracy.html)

### 项目相关
- `src/models/nb_vae.py`: VAE实现，包含所有数值保护
- `src/train/train_embed_core.py`: 训练循环，包含NaN检测
- `src/utils/edistance.py`: E-distance，梯度完整性修复

### 调试工具
- `torchviz`: 可视化计算图
- `torch.autograd.gradcheck`: 数值验证梯度
- `torch.autograd.detect_anomaly()`: 自动检测异常

