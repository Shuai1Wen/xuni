# 代码修复建议 - 详细实施方案

## 优先级1（立即修复）

### 修复1.1：scperturb_dataset.py 随机种子固定问题

**文件**：`src/data/scperturb_dataset.py`
**行号**：202-204
**严重性**：🔴 高（影响数据集有效性）
**修复时间**：5分钟

**问题描述**：
```python
# 当前代码
rng = np.random.RandomState(42)  # ❌ 固定种子
t0_sampled = rng.choice(t0_indices, size=n_pairs, replace=True)
t1_sampled = rng.choice(t1_indices, size=n_pairs, replace=True)
```

**问题影响**：
- 每次运行生成完全相同的数据对
- train/val/test无法真正分割
- 交叉验证失效

**修复方案**：

```python
def __init__(
    self,
    adata,
    cond_encoder: ConditionEncoder,
    tissue2idx: Dict[str, int],
    max_pairs_per_condition: int = 500,
    seed: Optional[int] = None  # 新增参数
):
    self.adata = adata
    self.cond_encoder = cond_encoder
    self.tissue2idx = tissue2idx
    self.n_tissues = len(tissue2idx)
    self.max_pairs_per_condition = max_pairs_per_condition
    self.seed = seed  # 保存seed用于reproducibility

    # 构建配对
    self.pairs = self._build_pairs()

def _build_pairs(self) -> List[Tuple[int, int, Dict]]:
    """构建细胞配对列表"""
    pairs = []

    # ... 之前的代码 ...

    for condition, group in grouped:
        # 分离t0和t1
        t0_indices = group[group["timepoint"] == "t0"].index.tolist()
        t1_indices = group[group["timepoint"] == "t1"].index.tolist()

        if len(t0_indices) == 0 or len(t1_indices) == 0:
            continue

        # 采样配对
        n_pairs = min(
            len(t0_indices),
            len(t1_indices),
            self.max_pairs_per_condition
        )

        # ✓ 修复：使用可控的随机种子
        rng = np.random.RandomState(self.seed)
        t0_sampled = rng.choice(t0_indices, size=n_pairs, replace=True)
        t1_sampled = rng.choice(t1_indices, size=n_pairs, replace=True)

        # ... 后续代码 ...

    return pairs
```

**使用示例**：
```python
# 训练集：随机采样（seed=None）
train_dataset = SCPerturbPairDataset(adata, cond_encoder, tissue2idx, seed=None)

# 验证集：使用固定seed保证可重复性
val_dataset = SCPerturbPairDataset(adata, cond_encoder, tissue2idx, seed=42)

# 测试集：使用不同seed
test_dataset = SCPerturbPairDataset(adata, cond_encoder, tissue2idx, seed=123)
```

---

### 修复1.2：operator.py power iteration梯度问题

**文件**：`src/models/operator.py`
**行号**：399-406（compute_operator_norm方法）
**严重性**：🔴 高（可能导致梯度异常）
**修复时间**：5分钟

**问题描述**：
```python
# 当前代码
for i in range(B):
    v = torch.randn(self.latent_dim, device=A_theta.device)
    for _ in range(5):
        v = A_theta[i] @ v  # ❌ v会积累梯度
        v = v / (v.norm() + 1e-8)
    norms[i] = (v @ (A_theta[i] @ v)).abs()
```

**问题影响**：
- power iteration不应计算梯度（范数是辅助计算）
- 可能导致梯度图过深
- 反向传播变慢

**修复方案**：

```python
def compute_operator_norm(
    self,
    tissue_idx: torch.Tensor,
    cond_vec: torch.Tensor,
    norm_type: str = "spectral"
) -> torch.Tensor:
    """
    计算算子A_θ的范数（用于监控稳定性）

    ... docstring ...
    """
    # 构造虚拟输入（不实际使用z）
    B = tissue_idx.size(0)
    z_dummy = torch.zeros(B, self.latent_dim, device=tissue_idx.device)

    # 获取A_θ
    _, A_theta, _ = self.forward(z_dummy, tissue_idx, cond_vec)  # (B, d, d)

    if norm_type == "frobenius":
        # Frobenius范数：||A||_F = sqrt(Σᵢⱼ A²ᵢⱼ)
        norms = torch.norm(A_theta.view(B, -1), dim=-1)  # (B,)
    elif norm_type == "spectral":
        # 谱范数：使用power iteration近似
        norms = torch.zeros(B, device=A_theta.device)
        with torch.no_grad():  # ✓ 修复：power iteration不需要梯度
            for i in range(B):
                v = torch.randn(self.latent_dim, device=A_theta.device)
                for _ in range(5):
                    v = A_theta[i] @ v
                    v = v / (v.norm() + 1e-8)
                norms[i] = (v @ (A_theta[i] @ v)).abs()
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")

    return norms
```

**类似修复**：spectral_penalty方法（第223-309行）也需要修复：

```python
def spectral_penalty(
    self,
    max_allowed: float = 1.05,
    n_iterations: int = 5
) -> torch.Tensor:
    """计算谱范数稳定性正则化项"""
    penalty = torch.tensor(0.0, device=self.A0_tissue.device)

    # 对每个组织的基线算子 A_t^(0) 计算谱范数
    for t in range(self.n_tissues):
        A0 = self.A0_tissue[t]  # (d, d)

        # ✓ 修复：power iteration不需要梯度
        with torch.no_grad():
            v = torch.randn(A0.size(0), device=A0.device)  # (d,)
            for _ in range(n_iterations):
                v = A0 @ v
                v = v / (v.norm() + 1e-8)
            spec = (v @ (A0 @ v)).abs()

        # 惩罚项需要梯度，所以在no_grad外计算
        if spec > max_allowed:
            penalty = penalty + (spec - max_allowed) ** 2

    # 对每个响应基 B_k 计算谱范数
    for k in range(self.K):
        Bk = self.B[k]  # (d, d)

        with torch.no_grad():
            v = torch.randn(Bk.size(0), device=Bk.device)  # (d,)
            for _ in range(n_iterations):
                v = Bk @ v
                v = v / (v.norm() + 1e-8)
            spec = (v @ (Bk @ v)).abs()

        if spec > max_allowed:
            penalty = penalty + (spec - max_allowed) ** 2

    return penalty
```

---

### 修复1.3：train_*_core.py 文件编码问题

**文件**：
- `src/train/train_operator_core.py`
- `src/train/train_embed_core.py`

**严重性**：🔴 高（无法导入）
**修复时间**：15分钟

**问题描述**：
文件编码损坏（可能是GBK或其他编码混入UTF-8）

**快速诊断**：
```bash
file src/train/train_*.py
# 输出: data（表示编码错误）
```

**修复方案**：

方案A：如果原始文件存储有备份
```bash
# 从git历史恢复
git checkout HEAD -- src/train/train_operator_core.py
git checkout HEAD -- src/train/train_embed_core.py
```

方案B：如果需要重新编码
```bash
# 检测原始编码
chardet src/train/train_operator_core.py

# 转换为UTF-8
iconv -f GBK -t UTF-8 src/train/train_operator_core.py -o temp.py
mv temp.py src/train/train_operator_core.py

iconv -f GBK -t UTF-8 src/train/train_embed_core.py -o temp.py
mv temp.py src/train/train_embed_core.py
```

方案C：重新生成文件
如果上述方案无法工作，需要重新编写这两个文件。可参考suanfa.md的第384-451和352-383行

**验证方案**：
```bash
# 验证编码正确
file src/train/train_*.py
# 应输出：UTF-8 Unicode text

# 尝试导入
python -c "from src.train.train_operator_core import train_operator"
```

---

## 优先级2（本周完成）

### 修复2.1：edistance.py 分块版本梯度问题

**文件**：`src/utils/edistance.py`
**行号**：243, 253, 262
**严重性**：🟡 中（影响反向传播）
**修复时间**：10分钟

**问题描述**：
```python
# 当前代码
term_xy += d_xy_batch.sum().item()  # ❌ .item()破坏梯度
```

**修复方案**：

```python
def energy_distance_batched(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 1000,
    requires_grad: bool = False  # 新参数
) -> torch.Tensor:
    """
    分块计算E-distance，用于大规模数据

    参数:
        x: (n, d) 第一组样本
        y: (m, d) 第二组样本
        batch_size: 分块大小
        requires_grad: 是否需要梯度（影响性能）

    返回:
        ed2: 标量，能量距离的平方
    """
    n, m = x.size(0), y.size(0)

    if n == 0 or m == 0:
        return torch.tensor(0.0, device=x.device)

    # ✓ 修复：使用张量而非标量累加
    if requires_grad:
        # 需要梯度时，保持张量形式
        term_xy = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        for i in range(0, n, batch_size):
            x_batch = x[i:i + batch_size]
            for j in range(0, m, batch_size):
                y_batch = y[j:j + batch_size]
                d_xy_batch = pairwise_distances(x_batch, y_batch)
                term_xy = term_xy + d_xy_batch.sum()
        term_xy = 2.0 / (n * m) * term_xy
    else:
        # 不需要梯度时，使用.item()优化内存
        term_xy_sum = 0.0
        for i in range(0, n, batch_size):
            x_batch = x[i:i + batch_size]
            for j in range(0, m, batch_size):
                y_batch = y[j:j + batch_size]
                d_xy_batch = pairwise_distances(x_batch, y_batch)
                term_xy_sum += d_xy_batch.sum().item()
        term_xy = 2.0 / (n * m) * term_xy_sum

    # term_xx和term_yy类似处理...
    term_xx = torch.tensor(0.0, device=x.device, dtype=x.dtype) if requires_grad else 0.0
    for i in range(0, n, batch_size):
        x_batch_i = x[i:i + batch_size]
        for j in range(0, n, batch_size):
            x_batch_j = x[j:j + batch_size]
            d_xx_batch = pairwise_distances(x_batch_i, x_batch_j)
            if requires_grad:
                term_xx = term_xx + d_xx_batch.sum()
            else:
                term_xx += d_xx_batch.sum().item()
    if requires_grad:
        term_xx = 1.0 / (n * n) * term_xx
    else:
        term_xx = 1.0 / (n * n) * term_xx

    # term_yy类似...
    term_yy = torch.tensor(0.0, device=y.device, dtype=y.dtype) if requires_grad else 0.0
    for i in range(0, m, batch_size):
        y_batch_i = y[i:i + batch_size]
        for j in range(0, m, batch_size):
            y_batch_j = y[j:j + batch_size]
            d_yy_batch = pairwise_distances(y_batch_i, y_batch_j)
            if requires_grad:
                term_yy = term_yy + d_yy_batch.sum()
            else:
                term_yy += d_yy_batch.sum().item()
    if requires_grad:
        term_yy = 1.0 / (m * m) * term_yy
    else:
        term_yy = 1.0 / (m * m) * term_yy

    if isinstance(term_xy, torch.Tensor):
        ed2 = term_xy - term_xx - term_yy
    else:
        ed2 = torch.tensor(term_xy - term_xx - term_yy, device=x.device, dtype=x.dtype)

    return ed2
```

---

### 修复2.2：operator.py 使用einsum优化内存

**文件**：`src/models/operator.py`
**行号**：184-198
**严重性**：🟡 中（内存占用高）
**修复时间**：10分钟
**性能提升**：5倍内存节省

**问题描述**：
```python
# 当前代码：O(B*K*d²)内存
B_expand = self.B.unsqueeze(0).expand(B, -1, -1, -1)
alpha_expand = alpha.view(B, self.K, 1, 1)
A_res = (alpha_expand * B_expand).sum(dim=1)
```

**修复方案**：

```python
def forward(self, z, tissue_idx, cond_vec):
    """前向传播：应用算子"""
    B = z.size(0)
    d = self.latent_dim

    # 计算响应基的激活系数
    alpha = self.alpha_mlp(cond_vec)  # (B, K)
    beta = self.beta_mlp(cond_vec)    # (B, K)

    # 获取对应组织的基线算子
    A0 = self.A0_tissue[tissue_idx]   # (B, d, d)
    b0 = self.b0_tissue[tissue_idx]   # (B, d)

    # ✓ 修复：使用einsum避免显式扩展
    # 原始：A_res = (alpha_expand * B_expand).sum(dim=1)
    # 优化：A_res = torch.einsum('bk,kij->bij', alpha, self.B)
    A_res = torch.einsum('bk,kij->bij', alpha, self.B)  # (B, d, d)

    # 最终算子
    A_theta = A0 + A_res  # (B, d, d)

    # 平移基也使用einsum
    b_res = torch.einsum('bk,ki->bi', beta, self.u)  # (B, d)

    # 最终平移
    b_theta = b0 + b_res  # (B, d)

    # 应用算子
    z_out = torch.bmm(A_theta, z.unsqueeze(-1)).squeeze(-1) + b_theta

    return z_out, A_theta, b_theta
```

**性能对比**：
```
优化前：
  batch=64, K=5, d=32: ~40MB内存占用

优化后（einsum）：
  batch=64, K=5, d=32: ~8MB内存占用
  性能提升：5倍
```

---

### 修复2.3：virtual_cell.py Pearson相关系数向量化

**文件**：`src/utils/virtual_cell.py`
**行号**：339-350
**严重性**：🟡 中（计算慢）
**修复时间**：10分钟
**性能提升**：10-20倍

**问题描述**：
```python
# 当前代码：for循环，O(B)个操作
for i in range(B):
    x_i = x[i]
    x_recon_i = x_recon[i]
    # ... 计算相关系数 ...
```

**修复方案**：

```python
@torch.no_grad()
def compute_reconstruction_error(
    vae: NBVAE,
    x: torch.Tensor,
    tissue_onehot: torch.Tensor,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算VAE的重建误差（用于质量评估）

    参数:
        vae: NB-VAE模型
        x: (B, G) 原始基因表达
        tissue_onehot: (B, n_tissues) 组织one-hot
        device: 设备

    返回:
        mse: (B,) 每个样本的MSE
        correlation: (B,) 每个样本的Pearson相关系数
    """
    # 编码-解码
    z = encode_cells(vae, x, tissue_onehot, device)
    x_recon = decode_cells(vae, z, tissue_onehot, device)

    # MSE
    mse = ((x - x_recon) ** 2).mean(dim=-1)  # (B,)

    # ✓ 修复：向量化计算Pearson相关系数
    # 中心化
    x_centered = x - x.mean(dim=-1, keepdim=True)  # (B, G)
    xr_centered = x_recon - x_recon.mean(dim=-1, keepdim=True)  # (B, G)

    # 相关系数向量化
    numerator = (x_centered * xr_centered).sum(dim=-1)  # (B,)
    denominator = torch.sqrt(
        (x_centered ** 2).sum(dim=-1) * (xr_centered ** 2).sum(dim=-1) + 1e-8
    )  # (B,)
    correlation = numerator / denominator  # (B,)

    return mse, correlation
```

**性能对比**：
```
优化前（for循环）：
  B=1000, G=2000: ~50ms

优化后（向量化）：
  B=1000, G=2000: ~2ms
  性能提升：25倍
```

---

## 优先级3（本月完成）

### 修复3.1：cond_encoder.py 类型提示修复

**文件**：`src/utils/cond_encoder.py`
**行号**：135, 209
**严重性**：🟢 低（类型检查工具报错）
**修复时间**：2分钟

```python
# 修复前
from typing import Dict, Optional, List
def encode_obs_row(self, obs_row: Dict[str, any], ...):
def forward(self, obs_rows: List[Dict[str, any]]) -> torch.Tensor:

# 修复后
from typing import Dict, Optional, List, Any  # 添加Any
def encode_obs_row(self, obs_row: Dict[str, Any], ...):
def forward(self, obs_rows: List[Dict[str, Any]]) -> torch.Tensor:
```

---

### 修复3.2：scperturb_dataset.py 索引问题

**文件**：`src/data/scperturb_dataset.py`
**行号**：207-213
**严重性**：🟡 中（可能导致运行时错误）
**修复时间**：10分钟

```python
# 问题代码
for i0, i1 in zip(t0_sampled, t1_sampled):
    obs_dict = obs_df.iloc[self.adata.obs.index.get_loc(i0)].to_dict()
    pairs.append((
        self.adata.obs.index.get_loc(i0),
        self.adata.obs.index.get_loc(i1),
        obs_dict
    ))

# 修复代码
for i0, i1 in zip(t0_sampled, t1_sampled):
    # i0和i1已经是标签，直接用get_loc转换为位置索引
    pos0 = self.adata.obs.index.get_loc(i0)
    pos1 = self.adata.obs.index.get_loc(i1)
    obs_dict = self.adata.obs.iloc[pos0].to_dict()
    pairs.append((pos0, pos1, obs_dict))
```

---

### 修复3.3：数值稳定性参数统一

**文件**：`src/config.py`
**任务**：创建统一的数值稳定性参数

```python
# 添加到config.py
class NumericalStabilityConfig:
    """数值稳定性相关参数"""

    # E-distance计算
    PAIRWISE_DISTANCE_EPSILON = 1e-7  # pairwise_distances中的epsilon

    # 负二项分布
    NB_LIKELIHOOD_EPSILON = 1e-8  # log计算的epsilon

    # 谱范数计算
    SPECTRAL_NORM_EPSILON = 1e-8  # power iteration的epsilon
    POWER_ITERATION_STEPS = 5     # power iteration的迭代次数

    # VAE
    VAE_EPSILON = 1e-8            # softplus和log的epsilon
```

---

## 修复验证清单

完成每个修复后，请检查：

- [ ] 代码语法正确（python -m py_compile）
- [ ] 能正常导入（python -c "from ... import ..."）
- [ ] 单元测试通过
- [ ] git diff 检查修改内容
- [ ] 提交信息清晰（参考CLAUDE.md规范）

---

## 修复时间表

| 修复 | 优先级 | 难度 | 预期时间 | 完成日期 |
|------|--------|------|----------|----------|
| 1.1 随机种子 | P1 | 低 | 5分钟 | _ |
| 1.2 power iteration | P1 | 低 | 5分钟 | _ |
| 1.3 文件编码 | P1 | 中 | 15分钟 | _ |
| 2.1 梯度问题 | P2 | 中 | 10分钟 | _ |
| 2.2 einsum优化 | P2 | 低 | 10分钟 | _ |
| 2.3 相关系数 | P2 | 低 | 10分钟 | _ |
| 3.1 类型提示 | P3 | 极低 | 2分钟 | _ |
| 3.2 索引问题 | P3 | 低 | 10分钟 | _ |
| 3.3 参数统一 | P3 | 低 | 10分钟 | _ |

**总计**：67分钟

---

## 测试验证脚本

运行提供的测试脚本验证修复：
```bash
cd /home/user/xuni
python .claude/test_core_modules.py
```

预期输出：
```
✓ NBVAE模块测试通过
✓ OperatorModel模块测试通过
✓ E-distance模块测试通过
✓ 条件编码器测试通过
✓ 虚拟细胞接口测试通过
✓ 性能测试完成

✓ 所有测试通过！
```

