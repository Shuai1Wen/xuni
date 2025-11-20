# 实验分析代码实现操作记录

生成时间：2025-11-20
任务：虚拟细胞算子模型完整实验分析代码设计

---

## 任务规划 - 实验分析代码实现

### 接口规格

**输入**：
- 已有核心模型代码（VAE、Operator、E-distance等）
- 用户需求：设计完整实验分析流程

**输出**：
- 完整的实验设计分析报告
- 项目上下文摘要
- 缺失模块清单
- 实现优先级建议

**依赖**：
- 已有模块：src/models/, src/utils/, src/train/, src/data/, src/config.py
- 外部库：PyTorch, AnnData, Scanpy, scikit-learn

### 数学对应关系

本任务不涉及新的数学公式，主要是组织和应用已有模型：
- **VAE训练**：对应 model.md A.2节（ELBO优化）
- **算子训练**：对应 model.md A.4-A.6节（E-distance优化+稳定性约束）
- **虚拟细胞生成**：对应 model.md A.8节（多步算子应用）
- **评估指标**：对应 model.md中的分布度量和生物学验证

### 已分析的关键模块

#### 1. 核心模型（已完成）
- **src/models/nb_vae.py** (487行)
  - 复用组件：`Encoder`, `DecoderNB`, `NBVAE`, `elbo_loss`
  - 数学对应：model.md A.2节
  - 关键特性：负二项建模、组织条件、数值稳定

- **src/models/operator.py** (473行)
  - 复用组件：`OperatorModel`, `spectral_penalty`, `get_response_profile`, `compute_operator_norm`
  - 数学对应：model.md A.3节（算子）+ A.5节（低秩）+ A.7节（稳定性）
  - 关键特性：全局响应基、条件特异系数、谱范数约束

#### 2. 工具模块（已完成）
- **src/utils/edistance.py** (354行)
  - 复用函数：`pairwise_distances`, `energy_distance`, `energy_distance_batched`, `check_edistance_properties`
  - 数学对应：model.md A.4节
  - 优势：无需OT、数值稳定、支持大规模

- **src/utils/virtual_cell.py** (412行)
  - 复用函数：`encode_cells`, `decode_cells`, `apply_operator`, `virtual_cell_scenario`, `compute_reconstruction_error`, `interpolate_conditions`
  - 数学对应：model.md A.8节
  - 应用：mLOY纠正、药物组合、跨组织预测

#### 3. 训练模块（已完成）
- **src/train/train_embed_core.py** (161行)
  - 主函数：`train_embedding`, `validate_embedding`, `save_checkpoint`, `load_checkpoint`
  - 特性：tqdm、验证早停、历史记录

- **src/train/train_operator_core.py** (211行)
  - 主函数：`train_operator`, `validate_operator`, `save_operator_checkpoint`, `load_operator_checkpoint`
  - 特性：E-distance损失、稳定性正则、NaN/Inf检查

#### 4. 数据模块（已完成）
- **src/data/scperturb_dataset.py** (314行)
  - 数据集：`SCPerturbEmbedDataset`, `SCPerturbPairDataset`
  - 配对策略：按(dataset_id, tissue, cell_type, perturbation)分组
  - 要求字段：tissue, perturbation, timepoint, dataset_id

#### 5. 配置模块（已完成）
- **src/config.py** (273行)
  - 配置类：`NumericalConfig`, `ModelConfig`, `TrainingConfig`, `ConditionMeta`, `DataConfig`, `ExperimentConfig`
  - 工具：`set_seed`

### 缺失模块清单

#### 高优先级（必须实现）

1. **src/utils/cond_encoder.py**（条件编码器）
   - 功能：将obs_dict编码为固定维度的cond_vec
   - 实现策略：perturbation/tissue用learned embedding，mLOY_load直接拼接
   - 接口：`ConditionEncoder.encode_obs_row(obs_dict) -> torch.Tensor`
   - 参考：已在experiment-design-analysis.md提供完整实现

2. **src/evaluation/metrics.py**（评估指标）
   - 功能：计算重建质量、DE基因预测、算子质量指标
   - 核心函数：
     - `reconstruction_metrics(x_true, x_pred) -> Dict`
     - `de_gene_prediction_metrics(x0, x1_true, x1_pred) -> Dict`
     - `operator_quality_metrics(operator_model, ...) -> Dict`
   - 参考：已在experiment-design-analysis.md提供完整实现

3. **src/visualization/plotting.py**（可视化工具）
   - 功能：潜空间UMAP、训练曲线、响应轮廓热图、基因表达对比
   - 核心函数：
     - `plot_latent_space_umap(z, labels) -> Figure`
     - `plot_training_curves(history) -> Figure`
     - `plot_response_heatmap(alpha_matrix) -> Figure`
     - `plot_gene_expression_comparison(x_true, x_pred) -> Figure`

4. **scripts/preprocessing/preprocess_scperturb.py**（数据预处理）
   - 功能：标准化scPerturb数据格式
   - 输入：原始h5ad文件
   - 输出：
     - scperturb_merged_train.h5ad
     - scperturb_merged_val.h5ad
     - scperturb_merged_test.h5ad
     - tissue2idx.json
     - perturbation2idx.json
     - condition_metadata.csv

5. **scripts/experiments/train_scperturb_baseline.py**（scPerturb基准实验）
   - 功能：完整的训练+评估流程
   - 阶段：VAE训练 + 算子训练
   - 输出：checkpoints + 评估报告
   - 参考：已在experiment-design-analysis.md提供完整实现

#### 中优先级（推荐实现）

6. **scripts/experiments/eval_perturbation_prediction.py**（扰动预测评估）
   - 功能：在测试集上评估扰动预测性能
   - 指标：E-distance、DE gene AUROC、Pearson相关

7. **scripts/experiments/analyze_response_axes.py**（响应基分析）
   - 功能：提取响应基、通路富集、激活模式分析
   - 输出：响应基对应的top基因、通路、热图

8. **configs/*.yaml**（配置文件）
   - scperturb_vae.yaml
   - scperturb_operator.yaml
   - mloy_kidney.yaml
   - mloy_brain.yaml
   - 参考：已在experiment-design-analysis.md提供模板

#### 低优先级（可选）

9. **scripts/experiments/run_counterfactual_mloy.py**（mLOY反事实）
   - 功能：mLOY纠正、虚拟mLOY生成
   - 需要：mLOY数据准备

10. **scripts/baselines/linear_regression_baseline.py**（基线方法）
    - 功能：简单线性回归基线
    - 用于：对比验证

### 实现决策

#### 决策1：条件编码器使用Hybrid方案
**原因**：
- One-hot可解释性强，但维度高
- Learned embedding泛化好，但需预训练
- Hybrid平衡两者优势

**实现**：
- perturbation, tissue: Learned embedding（16维、8维）
- batch: One-hot或embedding（4维）
- mLOY_load, age: 直接拼接

**依据**：
- 参考CPA（Compositional Perturbation Autoencoder）的设计
- 实验表明learned embedding在零样本泛化上表现更好

#### 决策2：评估指标优先使用E-distance
**原因**：
- 与训练目标一致
- scPerturb基准标准
- 无需OT匹配

**补充指标**：
- DE gene AUROC：生物学验证
- Pearson相关：重建质量
- 通路富集一致性：高层验证

#### 决策3：实验流程采用阶段式设计
**阶段划分**：
1. Phase I: scPerturb基准（必须）
2. Phase II: 响应基分析（必须）
3. Phase III: mLOY实验（可选）

**理由**：
- 逐步验证，降低风险
- 每阶段产出独立价值
- 资源不足时可以只完成I+II

### 向量化实现

所有已有代码均已向量化，新模块也应遵循：

**示例**（条件编码器批量编码）：
```python
def encode_batch(self, obs_list: List[Dict]) -> torch.Tensor:
    """批量编码多个obs
    
    输入: List of obs_dict
    输出: (B, cond_dim) 批量条件向量
    """
    # 收集所有索引
    pert_indices = [self.perturbation_vocab.index(obs["perturbation"]) for obs in obs_list]
    tissue_indices = [self.tissue_vocab.index(obs["tissue"]) for obs in obs_list]
    mLOY_loads = [obs.get("mLOY_load", 0.0) for obs in obs_list]
    
    # 批量embedding查询
    pert_vecs = self.pert_embedding(torch.tensor(pert_indices))  # (B, 16)
    tissue_vecs = self.tissue_embedding(torch.tensor(tissue_indices))  # (B, 8)
    mLOY_vecs = torch.tensor(mLOY_loads).unsqueeze(-1)  # (B, 1)
    
    # 拼接和投影（批量操作）
    concat = torch.cat([pert_vecs, tissue_vecs, mLOY_vecs], dim=-1)  # (B, 25)
    cond_vecs = self.linear(concat)  # (B, 64)
    
    return cond_vecs
```

### 复用的组件（当前任务）

**从已有模块复用**：
1. `src/models/nb_vae.NBVAE` - VAE模型
2. `src/models/operator.OperatorModel` - 算子模型
3. `src/utils/edistance.energy_distance` - E-distance计算
4. `src/utils/virtual_cell.*` - 虚拟细胞生成函数
5. `src/train/train_embed_core.train_embedding` - VAE训练循环
6. `src/train/train_operator_core.train_operator` - 算子训练循环
7. `src/data/scperturb_dataset.*` - 数据加载器
8. `src/config.*` - 配置数据类

**新模块将提供**：
1. `ConditionEncoder` - 统一的条件编码接口
2. `reconstruction_metrics`, `de_gene_prediction_metrics` - 评估指标
3. `plot_*` 系列函数 - 可视化工具
4. `preprocess_scperturb.py` - 数据预处理脚本
5. `train_scperturb_baseline.py` - 端到端训练脚本

### 遵循的项目约定

#### 命名约定（确认遵守）
- 类名：PascalCase ✅
- 函数名：snake_case ✅
- 变量名：snake_case ✅
- 常量：UPPER_SNAKE_CASE ✅

#### 文件组织（确认遵守）
- 模型定义 → `src/models/` ✅
- 工具函数 → `src/utils/` ✅
- 训练循环 → `src/train/` ✅
- 数据加载 → `src/data/` ✅
- 评估指标 → `src/evaluation/`（待创建）
- 可视化 → `src/visualization/`（待创建）
- 脚本 → `scripts/`

#### 注释规范（确认遵守）
- 所有函数有完整中文docstring ✅
- 复杂逻辑有中文行内注释 ✅
- 数学公式引用model.md位置 ✅

### 未重复造轮子的证明

**检查的模块**：
1. ✅ `src/models/` - 已有VAE和Operator，不需要重新实现
2. ✅ `src/utils/` - 已有edistance和virtual_cell，可直接复用
3. ✅ `src/train/` - 已有完整训练循环，只需调用
4. ✅ `src/data/` - 已有scPerturb数据加载器

**新增模块的必要性**：
1. `src/utils/cond_encoder.py` - **必要**，现有代码引用但未实现
2. `src/evaluation/metrics.py` - **必要**，评估需要标准化指标
3. `src/visualization/` - **必要**，论文需要图表
4. `scripts/preprocessing/` - **必要**，数据预处理是实验基础
5. `scripts/experiments/` - **必要**，端到端流程的封装

**结论**：所有新增模块都是填补空白，不存在重复造轮子。

---

## 编码后声明

### 1. 本次任务产出

本次任务为分析和设计任务，未编写实际代码，产出以下文档：

1. **上下文摘要**（`.claude/context-summary-experiment-analysis.md`）
   - 已有实现的详细分析
   - 缺失组件清单
   - 项目约定和代码风格
   - 关键风险点

2. **实验设计分析报告**（`.claude/experiment-design-analysis.md`）
   - 实验目标和数据需求
   - 推荐的实验组织结构
   - 关键技术挑战和解决方案
   - 完整实验流程设计
   - 代码实现关键点
   - 文件模板

3. **操作记录**（本文件：`.claude/operations-log.md`）
   - 任务规划
   - 已分析的关键模块
   - 缺失模块清单
   - 实现决策
   - 复用组件声明

### 2. 遵循的项目约定

- ✅ **简体中文**：所有文档和注释使用简体中文
- ✅ **数学对应**：明确标注与model.md的对应关系
- ✅ **文件结构**：遵循details.md的目录规范
- ✅ **命名规范**：类用PascalCase，函数用snake_case
- ✅ **向量化优先**：所有建议都采用批量操作

### 3. 对比的相似实现

**对比1**：CPA（Compositional Perturbation Autoencoder）
- **差异**：我们的算子模型使用低秩分解B_k，CPA使用embedding
- **借鉴**：条件编码器设计（learned embedding策略）

**对比2**：scGen（Style transfer VAE）
- **差异**：我们使用E-distance优化，scGen使用KL散度
- **借鉴**：VAE架构和评估指标

**对比3**：scVI（Deep generative model for scRNA-seq）
- **差异**：我们的算子是可解释的线性映射，scVI是黑盒神经网络
- **借鉴**：负二项建模、数值稳定性处理

### 4. 关键设计决策总结

| 决策点 | 选择方案 | 理由 | 权衡 |
|--------|----------|------|------|
| 条件编码 | Hybrid (embedding + 直接拼接) | 平衡泛化和可解释性 | 实现稍复杂，但效果最好 |
| 评估指标 | E-distance为主 | 与训练一致、scPerturb标准 | 计算开销大，需优化 |
| 实验流程 | 阶段式（I→II→III） | 降低风险、逐步验证 | 总时间较长 |
| 数据加载 | 预处理+密集矩阵 | 训练速度快 | 前期准备时间长 |
| 可视化 | 独立模块 | 便于复用和维护 | 需额外实现工作 |

---

## 下一步行动建议

### 立即行动（第1周）

1. **实现条件编码器**
   ```bash
   touch src/utils/cond_encoder.py
   # 使用experiment-design-analysis.md中的模板
   ```

2. **实现评估指标**
   ```bash
   mkdir -p src/evaluation
   touch src/evaluation/__init__.py
   touch src/evaluation/metrics.py
   # 使用experiment-design-analysis.md中的模板
   ```

3. **实现基础可视化**
   ```bash
   mkdir -p src/visualization
   touch src/visualization/__init__.py
   touch src/visualization/plotting.py
   ```

4. **编写预处理脚本**
   ```bash
   mkdir -p scripts/preprocessing
   touch scripts/preprocessing/preprocess_scperturb.py
   ```

### 短期目标（第2周）

5. **准备scPerturb数据**
   - 下载数据（使用scPerturb官方资源）
   - 运行预处理脚本
   - 验证数据格式

6. **实现训练脚本**
   ```bash
   mkdir -p scripts/experiments
   touch scripts/experiments/train_scperturb_baseline.py
   # 使用experiment-design-analysis.md中的模板
   ```

7. **创建配置文件**
   ```bash
   mkdir -p configs
   touch configs/scperturb_vae.yaml
   touch configs/scperturb_operator.yaml
   ```

8. **运行基准实验**
   - 训练VAE
   - 训练算子
   - 评估性能

### 中期目标（第3-4周）

9. **响应基分析**
   - 实现`analyze_response_axes.py`
   - 通路富集分析
   - 生成论文图表

10. **mLOY实验**（可选）
    - 准备mLOY数据
    - 联合训练
    - 反事实模拟

### 检查点

**第1周末检查点**：
- [ ] 条件编码器实现并通过单元测试
- [ ] 评估指标实现并通过单元测试
- [ ] 预处理脚本能成功处理样例数据

**第2周末检查点**：
- [ ] 成功在scPerturb数据上训练VAE
- [ ] VAE重建Pearson相关系数 > 0.7
- [ ] 训练脚本稳定运行无错误

**第3周末检查点**：
- [ ] 成功训练算子模型
- [ ] E-distance低于baseline（线性回归）
- [ ] 响应基分析产出可解释结果

---

## 总结

本次分析为虚拟细胞算子模型项目提供了完整的实验分析代码设计方案。主要产出包括：

1. **全面的项目上下文分析**：识别已有模块和缺失组件
2. **详细的实验设计方案**：从数据准备到结果分析的完整流程
3. **具体的实现建议**：包括代码模板、配置文件、脚本结构
4. **清晰的优先级和时间表**：分阶段实施，降低风险

**关键成功因素**：
- 充分复用已有高质量代码
- 遵循项目既有约定和规范
- 采用阶段式实验设计
- 重视数值稳定性和可重复性

**推荐的实施路径**：
1. 先实现基础设施（编码器、指标、可视化）
2. 再运行scPerturb基准实验验证方法有效性
3. 最后扩展到mLOY等高级应用

按照本方案实施，预计4周可完成核心实验，8周可完成完整论文所需的全部实验和分析。
