# 项目上下文摘要（虚拟细胞算子模型完整实现）
生成时间：2025-11-17

## 1. 相似实现分析

### 实现1：suanfa.md - PyTorch代码骨架
- **位置**：suanfa.md:1-783
- **模式**：完整的模块化架构，包含配置、模型、训练、数据加载
- **可复用组件**：
  - `Encoder`类（第68-89行）：编码器基类，支持组织条件输入
  - `DecoderNB`类（第91-109行）：负二项解码器，包含基因特异离散度参数
  - `NBVAE`类（第132-142行）：完整VAE模型
  - `elbo_loss`函数（第144-149行）：ELBO损失计算
  - `nb_log_likelihood`函数（第116-130行）：负二项对数似然
  - `OperatorModel`类（第238-310行）：低秩算子模型
  - `spectral_penalty`方法（第312-339行）：谱范数正则化
  - `pairwise_distances`函数（第160-170行）：成对距离计算
  - `energy_distance`函数（第172-190行）：E-distance无偏估计
  - `encode_cells`, `decode_cells`, `apply_operator`, `virtual_cell_scenario`函数（第680-734行）
- **需注意**：
  - 代码是骨架级别，需要补充实现细节
  - 数据加载器的`_build_pairs`方法需要根据实际数据结构实现
  - 谱范数计算使用power iteration近似，需要验证精度

### 实现2：model.md - 严格数学定义
- **位置**：model.md:1-525
- **模式**：完整的数学形式化，包含所有公式推导
- **数学对应关系**：
  - **A.2节（潜空间模型）**：
    - 编码器：q_φ(z|x,t)，z ∈ ℝ^{d_z}（第38-44行）
    - 解码器：p_ψ(x|z,t) = ∏_g NB(x_g; μ_g, r_g)（第46-52行）
    - ELBO损失：L_embed = 𝔼[log p] - KL(q||p)（第55-65行）
  - **A.3节（算子定义）**：
    - K_θ(z) = A_θz + b_θ（第77-79行）
  - **A.4节（E-distance）**：
    - Ê²(X,Y) = 2/(nm)Σ|xi-yj| - 1/n²Σ|xi-xi'| - 1/m²Σ|yj-yj'|（第111-119行）
  - **A.5节（低秩分解）**：
    - A_θ = A_t^(0) + Σ_k α_k(θ) B_k（第145-151行）
    - b_θ = b_t^(0) + Σ_k β_k(θ) u_k（第195-200行）
    - α_k(θ) = g_k(θ; ω)（第173-191行）
  - **A.6节（训练目标）**：
    - L = L_embed + λ₁·Σ_θ Ê²(K_θ(z₀), z₁) + λ₂·R_stab（第213-224行）
  - **A.7节（稳定性约束）**：
    - R_stab = Σ_θ max(0, ρ(A_θ) - ρ₀)²（第231-246行）
  - **A.8节（反事实模拟）**：
    - 单步预测：z₁ = K_θ(z₀)（第272-276行）
    - 多步合成：z₂ = K_θ₂(K_θ₁(z₀))（第277-284行）
- **需注意**：
  - 所有实现必须100%对应公式定义
  - 维度变换需要严格验证
  - 数值稳定性是关键考虑因素

### 实现3：details.md - 项目工程化结构
- **位置**：details.md:1-453
- **模式**：完整的目录结构和README模板
- **文件组织**：
  - `configs/`：YAML配置文件（第16-19行）
  - `src/models/`：模型定义（第38-40行）
  - `src/data/`：数据加载器（第41-43行）
  - `src/utils/`：工具函数（第44-48行）
  - `src/train/`：训练循环（第49-51行）
  - `scripts/`：可执行脚本（第23-36行）
- **需注意**：
  - 必须严格遵循目录结构
  - src/下只放可复用核心代码
  - scripts/下只放任务脚本，不放逻辑

### 实现4：开源参考 - scvi-tools
- **来源**：https://docs.scvi-tools.org/
- **可借鉴内容**：
  - 负二项分布的PyTorch实现
  - VAE训练的稳定性技巧
  - AnnData数据加载方式
- **需注意**：
  - 不直接复制代码，仅作为参考
  - 我们的低秩算子结构是原创的

### 实现5：开源参考 - PyTorch负二项VAE
- **来源**：https://github.com/Szym29/ZeroInflatedNegativeBinomial_VAE
- **可借鉴内容**：
  - 负二项对数似然的数值稳定实现
  - 离散度参数的初始化策略
- **需注意**：
  - 我们不需要零膨胀部分（ZINB → NB）
  - 需要适配组织条件输入

## 2. 项目约定

### 命名约定
- **类名**：PascalCase（如`OperatorModel`, `NBVAE`）
- **函数名**：snake_case（如`energy_distance`, `encode_cells`）
- **变量名**：snake_case（如`latent_dim`, `n_tissues`）
- **私有方法**：以单下划线开头（如`_build_pairs`）
- **常量**：全大写（如`DEFAULT_LR`, `MAX_EPOCHS`）

### 文件组织
- **模型文件**：按模型类型分文件（`nb_vae.py`, `operator.py`）
- **工具文件**：按功能分文件（`edistance.py`, `virtual_cell.py`）
- **数据文件**：按数据源分文件（`scperturb_dataset.py`, `mloy_dataset.py`）
- **训练文件**：按训练阶段分文件（`train_embed_core.py`, `train_operator_core.py`）

### 导入顺序
1. 标准库（`from dataclasses import dataclass`）
2. 第三方库（`import torch`, `import numpy`）
3. 项目内模块（`from src.models import NBVAE`）
4. 相对导入最后（`from .utils import helper`）

### 代码风格
- **格式化工具**：遵循Black默认设置
- **行宽**：88字符
- **缩进**：4空格
- **字符串**：优先使用双引号
- **类型提示**：所有公共函数必须有类型提示

## 3. 可复用组件清单

### 3.1 潜空间模型（src/models/nb_vae.py）
- **Encoder**：编码器基类
  - 输入：x (B, G), tissue_onehot (B, n_tissues)
  - 输出：mu (B, d_z), logvar (B, d_z)
  - 数学对应：model.md A.2节，q_φ(z|x,t)
- **DecoderNB**：负二项解码器
  - 输入：z (B, d_z), tissue_onehot (B, n_tissues)
  - 输出：mu (B, G), r (1, G)
  - 数学对应：model.md A.2节，p_ψ(x|z,t)
- **NBVAE**：完整VAE模型
  - 组合Encoder和DecoderNB
  - forward方法返回z, mu_x, r_x, mu_z, logvar_z
- **sample_z**：重参数化采样
  - 输入：mu, logvar
  - 输出：z = mu + eps * std
- **nb_log_likelihood**：负二项对数似然
  - 输入：x, mu, r
  - 输出：log p(x|mu, r)
  - 数学对应：model.md A.2节，NB pmf
- **elbo_loss**：ELBO损失
  - 输入：x, tissue_onehot, model
  - 输出：loss (标量), z (detached)
  - 数学对应：model.md A.2节，L_embed

### 3.2 算子模型（src/models/operator.py）
- **OperatorModel**：低秩算子模型主类
  - 属性：
    - A0_tissue (n_tissues, d, d)：基线算子
    - b0_tissue (n_tissues, d)：基线平移
    - B (K, d, d)：全局响应基
    - u (K, d)：全局平移基
    - alpha_mlp：条件→系数α的MLP
    - beta_mlp：条件→系数β的MLP
  - forward方法：
    - 输入：z (B, d), tissue_idx (B,), cond_vec (B, cond_dim)
    - 输出：z_out (B, d), A_theta (B, d, d), b_theta (B, d)
    - 数学对应：model.md A.3节和A.5节
- **spectral_penalty**：谱范数正则化
  - 使用power iteration计算谱半径近似
  - 数学对应：model.md A.7节

### 3.3 能量距离（src/utils/edistance.py）
- **pairwise_distances**：成对L2距离
  - 输入：x (n, d), y (m, d)
  - 输出：距离矩阵 (n, m)
  - 实现：向量化计算 ||x-y||² = ||x||² + ||y||² - 2x^T y
- **energy_distance**：E-distance无偏估计
  - 输入：x (n, d), y (m, d)
  - 输出：ed2（标量）
  - 数学对应：model.md A.4节
  - 处理边界：空集返回0

### 3.4 条件编码器（src/utils/cond_encoder.py）
- **ConditionEncoder**：条件向量编码器
  - 输入特征：perturbation, tissue, batch, mLOY_load等
  - 输出：cond_vec (cond_dim,)
  - 实现方式：one-hot拼接 + 线性降维

### 3.5 虚拟细胞操作（src/utils/virtual_cell.py）
- **encode_cells**：x → z
  - 使用VAE编码器
  - 返回均值（不采样）
- **decode_cells**：z → x
  - 使用VAE解码器
  - 返回mu_x（负二项均值）
- **apply_operator**：z → K_θ(z)
  - 应用算子模型
  - 数学对应：model.md A.3节
- **virtual_cell_scenario**：多步反事实模拟
  - 循环应用算子序列
  - 数学对应：model.md A.8节

### 3.6 数据加载器（src/data/scperturb_dataset.py）
- **SCPerturbEmbedDataset**：VAE训练数据集
  - 返回：x, tissue_onehot, tissue_idx
- **SCPerturbPairDataset**：算子训练数据集
  - 返回：x0, x1, tissue_onehot, tissue_idx, cond_vec
  - 需要实现_build_pairs方法

### 3.7 训练循环（src/train/）
- **train_embedding**（train_embed_core.py）：
  - 训练NB-VAE
  - 损失：ELBO
  - 优化器：Adam
- **train_operator**（train_operator_core.py）：
  - 训练算子模型
  - 损失：E-distance + 谱范数惩罚
  - 冻结VAE

## 4. 测试策略

### 测试框架
- **框架**：pytest
- **测试目录**：tests/
- **命名约定**：test_*.py

### 测试模式
#### 单元测试
- 每个核心函数都有对应测试
- 覆盖正常情况、边界条件、错误处理
- 示例：
  ```python
  def test_energy_distance():
      # 正常情况
      x = torch.randn(100, 10)
      y = torch.randn(100, 10) + 1.0
      ed2 = energy_distance(x, y)
      assert ed2 > 0

      # 边界情况：相同分布
      ed2_same = energy_distance(x, x)
      assert torch.abs(ed2_same) < 1e-6

      # 边界情况：空集
      x_empty = torch.randn(0, 10)
      ed2_empty = energy_distance(x_empty, y)
      assert ed2_empty == 0
  ```

#### 数值精度测试
- 验证与model.md的数学定义一致
- 容忍误差：1e-6（float32）
- 示例：
  ```python
  def test_nb_likelihood_consistency():
      # 手动计算与函数结果对比
      x = torch.tensor([[5.0, 10.0]])
      mu = torch.tensor([[5.0, 10.0]])
      r = torch.tensor([[1.0, 1.0]])

      log_p_computed = nb_log_likelihood(x, mu, r)
      log_p_expected = compute_nb_manually(x, mu, r)

      assert torch.abs(log_p_computed - log_p_expected) < 1e-6
  ```

#### 集成测试
- 端到端流程测试
- 使用小数据集（100个细胞）
- 验证训练可以收敛

### 测试覆盖要求
- **正常流程**：所有公共函数的典型用例
- **边界条件**：空输入、单样本、极大输入
- **错误处理**：非法输入、维度不匹配、数值溢出
- **数值稳定性**：NaN检测、Inf检测、梯度爆炸

### 参考测试文件
- 暂无（新项目）
- 需要创建：
  - tests/test_nb_vae.py
  - tests/test_edistance.py
  - tests/test_operator.py
  - tests/test_virtual_cell.py

## 5. 依赖和集成点

### 外部依赖
- **核心依赖**：
  - torch >= 2.0.0：深度学习框架
  - numpy >= 1.20.0：数值计算
  - anndata >= 0.9.0：单细胞数据结构
  - scanpy >= 1.9.0：单细胞数据分析
  - pandas >= 1.5.0：数据框操作
- **可选依赖**：
  - matplotlib >= 3.5.0：可视化
  - seaborn >= 0.12.0：高级可视化
  - wandb：实验跟踪
  - tqdm：进度条

### 内部依赖关系
```
scripts/train_embed.py
  → src/train/train_embed_core.py
    → src/models/nb_vae.py (NBVAE, elbo_loss)
      → torch.nn.Module
    → src/data/scperturb_dataset.py (SCPerturbEmbedDataset)
      → anndata
    → src/config.py (ModelConfig, TrainingConfig)

scripts/train_operator_scperturb.py
  → src/train/train_operator_core.py
    → src/models/operator.py (OperatorModel)
      → torch.nn.Module
    → src/utils/edistance.py (energy_distance)
      → torch
    → src/data/scperturb_dataset.py (SCPerturbPairDataset)
      → src/utils/cond_encoder.py (ConditionEncoder)
    → src/models/nb_vae.py (NBVAE，冻结用于编码)

scripts/run_counterfactuals.py
  → src/utils/virtual_cell.py (virtual_cell_scenario)
    → src/models/nb_vae.py (encode, decode)
    → src/models/operator.py (apply)
```

### 集成方式
- **直接导入调用**：模块间通过Python import直接调用
- **配置驱动**：超参数通过YAML配置文件传递
- **检查点加载**：模型通过torch.save/load保存和加载

### 配置来源
- **主配置**：configs/default.yaml
- **任务配置**：configs/scperturb.yaml, configs/mloy_kidney_brain.yaml
- **格式**：YAML
- **解析方式**：使用dataclass + yaml库

## 6. 技术选型理由

### 为什么用PyTorch
- **优势**：
  - 自动微分：自动计算梯度
  - GPU加速：支持CUDA和多GPU
  - 生态丰富：有大量单细胞分析工具基于PyTorch
  - 动态图：方便调试和实验
- **劣势**：
  - 内存占用相对较高
  - 某些操作比TensorFlow慢

### 为什么用E-distance
- **数学严格性**：model.md公式(A.4)明确定义
- **优势**：
  - 无需OT匹配：不需要逐细胞匹配，计算简单
  - 分布级度量：直接度量两个细胞群的分布差异
  - 无偏估计：有严格的理论保证
- **劣势**：
  - O(n²)复杂度：大批次时内存占用高
  - 缺乏可解释性：不像OT那样有细胞对应关系

### 为什么用负二项分布
- **数据特性匹配**：
  - 单细胞RNA-seq数据是离散计数
  - 存在过离散现象（方差 > 均值）
  - 零膨胀（但我们简化为NB，不用ZINB）
- **优势**：
  - 理论基础扎实：有明确的概率解释
  - 参数化灵活：通过μ和r控制均值和离散度
  - 成熟工具：scVI等工具已验证有效性
- **劣势**：
  - 计算复杂：涉及lgamma函数
  - 数值稳定性：需要careful实现

### 为什么用低秩分解
- **参数效率**：
  - 不低秩：需要存储 |Θ| × d² 个参数
  - 低秩后：只需 n_tissues × d² + K × d² + MLP参数
  - 示例：1000个条件，d=32，K=5 → 1M参数 → 10K参数
- **泛化能力**：
  - 共享响应基B_k：学到跨条件的共性
  - 条件特异系数α_k(θ)：捕捉条件特异性
  - 支持零样本泛化：新条件通过MLP预测系数
- **可解释性**：
  - 每个B_k代表一个"虚拟细胞响应轴"
  - 可以通过α_k(θ)分析哪些轴被激活
  - 支持跨组织、跨扰动的比较分析

## 7. 关键风险点

### 7.1 并发问题
- **风险等级**：无
- **原因**：单线程训练，无并发操作
- **注意事项**：DataLoader可以设置多进程，但torch已处理好同步

### 7.2 边界条件
- **风险等级**：中
- **关键边界**：
  1. 空batch：E-distance需要检查n=0或m=0
  2. 单样本batch：batch norm可能失效
  3. 极端值：x=0（计数为0）时log(x)会出现-inf
- **缓解措施**：
  - 所有log计算添加epsilon：log(x + 1e-8)
  - softplus输出添加epsilon：softplus(x) + 1e-8
  - E-distance显式处理空集情况

### 7.3 性能瓶颈
- **风险等级**：高
- **瓶颈点**：
  1. **E-distance的O(n²)复杂度**：
     - 问题：n=10000时需要计算1亿次距离
     - 缓解：分块计算、降低batch size、使用近似算法
  2. **谱范数计算**：
     - 问题：每次都要power iteration
     - 缓解：只对B_k和A0计算，不对每个样本的A_θ计算
  3. **AnnData加载**：
     - 问题：大型h5ad文件加载慢
     - 缓解：预先切分为小文件、使用backed模式
- **监控指标**：
  - 训练速度：samples/sec
  - 内存占用：peak GPU memory
  - 每个epoch时间

### 7.4 数值稳定性
- **风险等级**：高
- **风险点**：
  1. **负二项对数似然**：
     - μ过小时log(μ/(r+μ))可能下溢
     - lgamma(x)在x<0时未定义
  2. **谱范数计算**：
     - power iteration可能不收敛
     - 除以v.norm()时v可能为零向量
  3. **梯度爆炸**：
     - 算子连续应用可能导致z的norm爆炸
- **缓解措施**：
  - 所有除法添加epsilon
  - 所有log计算添加epsilon
  - 使用gradient clipping
  - 定期检查NaN和Inf：torch.isnan(), torch.isinf()

### 7.5 维度一致性
- **风险等级**：中
- **易错点**：
  1. 批量矩阵乘法：bmm要求(B, n, m) @ (B, m, k) → (B, n, k)
  2. 广播语义：(B, d, d) @ (B, d) 需要先unsqueeze(-1)
  3. 组织索引：A0_tissue[tissue_idx]会自动广播batch维度
- **缓解措施**：
  - 所有张量操作后立即检查shape
  - 使用类型提示标注shape：# (B, d, d)
  - 编写维度一致性单元测试

## 8. 数学-代码对应关系速查表

| model.md位置 | 数学公式 | 代码位置 | 函数/类名 |
|-------------|---------|---------|----------|
| A.2节（第38-44行） | q_φ(z\|x,t) | src/models/nb_vae.py | Encoder |
| A.2节（第46-52行） | p_ψ(x\|z,t) = ∏_g NB(...) | src/models/nb_vae.py | DecoderNB |
| A.2节（第55-65行） | L_embed = 𝔼[log p] - KL | src/models/nb_vae.py | elbo_loss |
| A.3节（第77-79行） | K_θ(z) = A_θz + b_θ | src/models/operator.py | OperatorModel.forward |
| A.4节（第111-119行） | Ê²(X,Y) = 2/(nm)Σ\|xi-yj\| - ... | src/utils/edistance.py | energy_distance |
| A.5.1节（第145-151行） | A_θ = A_t^(0) + Σ α_k B_k | src/models/operator.py | OperatorModel.forward（低秩组合） |
| A.5.2节（第173-191行） | α_k(θ) = g_k(θ; ω) | src/models/operator.py | alpha_mlp |
| A.5.3节（第195-200行） | b_θ = b_t^(0) + Σ β_k u_k | src/models/operator.py | OperatorModel.forward（平移组合） |
| A.6节（第213-224行） | L = L_embed + λ₁Ê² + λ₂R | src/train/train_operator_core.py | train_operator（综合损失） |
| A.7.1节（第231-246行） | R_stab = Σ max(0, ρ-ρ₀)² | src/models/operator.py | spectral_penalty |
| A.8节（第272-276行） | z₁ = K_θ(z₀) | src/utils/virtual_cell.py | apply_operator |
| A.8节（第277-284行） | z₂ = K_θ₂(K_θ₁(z₀)) | src/utils/virtual_cell.py | virtual_cell_scenario |

## 9. 实现优先级

### P0（最高优先级，第1-2天）
1. 创建目录结构（按details.md）
2. 实现src/config.py（配置系统）
3. 实现src/data/scperturb_dataset.py（仅EmbedDataset）
4. 编写environment.yml和requirements.txt

### P1（高优先级，第3-5天）
1. 实现src/models/nb_vae.py（NB-VAE核心）
2. 实现src/utils/edistance.py（E-distance）
3. 实现src/train/train_embed_core.py（潜空间训练）
4. 实现scripts/train_embed.py（训练脚本）
5. 在小数据集上验证VAE

### P2（中优先级，第6-10天）
1. 实现src/models/operator.py（算子模型）
2. 实现src/utils/cond_encoder.py（条件编码器）
3. 扩展src/data/scperturb_dataset.py（PairDataset）
4. 实现src/train/train_operator_core.py（算子训练）
5. 实现scripts/train_operator_scperturb.py（训练脚本）
6. 在小数据集上验证算子模型

### P3（中低优先级，第11-14天）
1. 实现src/utils/virtual_cell.py（虚拟细胞操作）
2. 实现scripts/eval_scperturb.py（评估脚本）
3. 与baseline对比

### P4（低优先级，第15+天）
1. 实现mLOY相关功能
2. 实现反事实模拟
3. 完善文档和测试

## 10. 下一步行动

按照优先级，建议立即开始：

**第一步**：创建完整的目录结构
```bash
mkdir -p src/{models,data,utils,train}
mkdir -p scripts configs tests notebooks results/{logs,checkpoints,figures}
touch src/__init__.py src/models/__init__.py src/data/__init__.py
touch src/utils/__init__.py src/train/__init__.py
```

**第二步**：实现src/config.py（配置数据类）

**第三步**：实现src/models/nb_vae.py（NB-VAE核心）

**验收标准**：
- VAE可以在小数据集上训练
- ELBO收敛
- 重建相关系数>0.7

---

## 附录：常用命令速查

### 环境管理
```bash
# 创建环境
conda env create -f environment.yml
conda activate vcell-operator

# 更新环境
conda env update -f environment.yml
```

### 训练命令
```bash
# 训练潜空间模型
python scripts/train_embed.py --config configs/scperturb.yaml

# 训练算子模型
python scripts/train_operator_scperturb.py --config configs/scperturb.yaml
```

### 测试命令
```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_nb_vae.py -v

# 测试覆盖率
pytest --cov=src tests/
```

### 代码格式化
```bash
# 格式化代码
black src/ scripts/ tests/

# 检查格式
black --check src/

# 类型检查
mypy src/
```

---

**摘要生成完成。准备进入实现阶段。**
