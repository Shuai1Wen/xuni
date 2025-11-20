# 项目上下文摘要（实验分析任务）
生成时间：2025-11-20

## 1. 已有实现分析

### 1.1 核心模型模块（已完成）
**文件**: src/models/nb_vae.py (487行)
- 模式：编码器-解码器架构
- 可复用：
  - `Encoder`: 基因表达→潜空间（支持组织条件）
  - `DecoderNB`: 潜空间→负二项分布参数
  - `NBVAE`: 完整VAE模型
  - `elbo_loss`: ELBO损失函数（支持β-VAE）
- 数学对应：model.md A.2节（第38-65行）
- 关键特性：负二项建模、组织条件输入、数值稳定性保证

**文件**: src/models/operator.py (473行)
- 模式：低秩算子族
- 可复用：
  - `OperatorModel`: 扰动响应算子主类
  - `spectral_penalty`: 谱范数正则化（power iteration）
  - `get_response_profile`: 获取响应轮廓（α_k, β_k）
  - `compute_operator_norm`: 计算算子范数
- 数学对应：model.md A.3节（算子定义）+ A.5节（低秩分解）+ A.7节（稳定性）
- 关键特性：全局响应基B_k、条件特异系数α_k(θ)、谱范数约束

### 1.2 工具模块（已完成）
**文件**: src/utils/edistance.py (354行)
- 可复用函数：
  - `pairwise_distances`: 成对L2距离（向量化实现）
  - `energy_distance`: E-distance计算（无偏估计）
  - `energy_distance_batched`: 分块计算（用于大规模数据）
  - `check_edistance_properties`: 数学性质验证
- 数学对应：model.md A.4节（第111-119行）
- 优势：无需OT匹配、数值稳定、支持大规模

**文件**: src/utils/virtual_cell.py (412行)
- 可复用函数：
  - `encode_cells`: x→z编码
  - `decode_cells`: z→x解码
  - `apply_operator`: K_θ(z)应用
  - `virtual_cell_scenario`: 多步反事实模拟
  - `compute_reconstruction_error`: VAE质量评估
  - `interpolate_conditions`: 条件插值轨迹
- 数学对应：model.md A.8节（虚拟细胞生成）
- 应用场景：mLOY纠正、药物组合、跨组织预测

### 1.3 训练模块（已完成）
**文件**: src/train/train_embed_core.py (161行)
- 主要函数：
  - `train_embedding`: VAE训练循环（支持warmup、梯度裁剪）
  - `validate_embedding`: 验证集评估
  - `save_checkpoint` / `load_checkpoint`: 模型保存加载
- 特性：tqdm进度条、验证集早停、训练历史记录

**文件**: src/train/train_operator_core.py (211行)
- 主要函数：
  - `train_operator`: 算子训练循环（支持冻结VAE）
  - `validate_operator`: 验证集评估
  - `save_operator_checkpoint` / `load_operator_checkpoint`
- 特性：E-distance损失、稳定性正则、数值检查（NaN/Inf）

### 1.4 数据模块（已完成）
**文件**: src/data/scperturb_dataset.py (314行)
- 数据集类：
  - `SCPerturbEmbedDataset`: VAE训练数据集（统一加载所有细胞）
  - `SCPerturbPairDataset`: 算子训练数据集（t0→t1配对）
  - `collate_fn_embed` / `collate_fn_pair`: 自定义批处理
- 配对策略：按(dataset_id, tissue, cell_type, perturbation)分组，随机采样
- 数据要求：
  - adata.obs["tissue"]
  - adata.obs["perturbation"]
  - adata.obs["timepoint"] ∈ {"t0", "t1"}
  - adata.obs["dataset_id"]

### 1.5 配置模块（已完成）
**文件**: src/config.py (273行)
- 配置类：
  - `NumericalConfig`: 数值稳定性参数（epsilon、容差）
  - `ModelConfig`: 模型超参数（latent_dim、n_response_bases等）
  - `TrainingConfig`: 训练超参数（学习率、损失权重等）
  - `ConditionMeta`: 条件元信息结构
  - `DataConfig`: 数据预处理配置
  - `ExperimentConfig`: 完整实验配置
- 工具函数：`set_seed`: 设置全局随机种子

## 2. 缺失的实验分析组件

### 2.1 评估指标模块（**需要创建**）
**建议路径**: `src/evaluation/metrics.py`
**必需功能**：
- 重建质量指标：
  - MSE / Pearson相关系数（基因层面）
  - 分布级别：E-distance、KL散度
- 扰动预测指标：
  - 差异基因预测：AUROC、AUPRC
  - 通路富集一致性
- 算子质量指标：
  - 谱范数统计
  - 低秩近似误差
  - 响应轴可解释性

### 2.2 可视化模块（**需要创建**）
**建议路径**: `src/visualization/plotting.py`
**必需功能**：
- 潜空间可视化：UMAP/t-SNE（真实vs虚拟细胞混合）
- 轨迹可视化：多步算子应用的状态演化
- 响应轮廓热图：α_k(θ)矩阵
- 基因表达对比：真实vs预测（小提琴图、散点图）
- 损失曲线：训练/验证损失变化

### 2.3 条件编码器（**部分实现**）
**现状**: src/data/scperturb_dataset.py中引用`ConditionEncoder`，但未找到实现文件
**建议路径**: `src/utils/cond_encoder.py`
**必需功能**：
- 将obs_dict编码为固定维度的cond_vec
- 支持：perturbation、tissue、mLOY_load、batch等字段
- 实现方式：one-hot拼接 + 线性投影/MLP

### 2.4 实验脚本（**需要创建**）
**建议路径**: `scripts/experiments/`
**必需脚本**：
1. `train_scperturb_baseline.py`: scPerturb基准实验
2. `eval_perturbation_prediction.py`: 扰动预测评估
3. `analyze_response_axes.py`: 响应基分析
4. `run_counterfactual_mloy.py`: mLOY反事实模拟
5. `cross_tissue_comparison.py`: 跨组织效应分析

### 2.5 数据预处理脚本（**需要创建**）
**建议路径**: `scripts/preprocessing/`
**必需脚本**：
1. `preprocess_scperturb.py`: scPerturb数据预处理
2. `preprocess_mloy_kidney.py`: 肾脏mLOY数据预处理
3. `prepare_condition_metadata.py`: 构建条件元信息表

## 3. 项目约定和代码风格

### 3.1 命名约定
- 类名：PascalCase（如`OperatorModel`、`NBVAE`）
- 函数名：snake_case（如`train_embedding`、`energy_distance`）
- 变量名：snake_case（如`latent_dim`、`cond_vec`）
- 常量：UPPER_SNAKE_CASE（如`_NUM_CFG`）
- 私有方法：前缀下划线（如`_build_pairs`）

### 3.2 文件组织
- `src/models/`: 神经网络模型定义
- `src/utils/`: 工具函数（距离计算、虚拟细胞等）
- `src/data/`: 数据加载器
- `src/train/`: 训练循环
- `src/evaluation/`: 评估指标（**待创建**）
- `src/visualization/`: 可视化工具（**待创建**）
- `scripts/`: 可执行脚本
- `tests/`: 单元测试

### 3.3 注释规范
- 所有函数必须有完整的中文docstring
- docstring包含：简短描述、数学对应、参数说明、返回值、示例
- 复杂逻辑必须有中文行内注释
- 数学公式必须引用model.md的具体位置

### 3.4 数学-代码一致性
- 所有实现必须100%对应model.md的数学定义
- 变量命名尽量与数学符号对应：
  - `z`: 潜变量
  - `A_theta`: 算子矩阵
  - `alpha`, `beta`: 响应系数
  - `B`: 响应基
- 关键公式在注释中标注对应的model.md位置

## 4. 测试策略

### 4.1 已有测试
- `tests/test_nb_vae.py`: VAE模型测试
- `tests/test_operator.py`: 算子模型测试
- `tests/test_edistance.py`: E-distance计算测试
- `tests/conftest.py`: pytest配置

### 4.2 测试模式
- 使用pytest框架
- 测试覆盖：正常流程 + 边界条件 + 数值精度
- 数值测试：容差_NUM_CFG.tol_test
- 模拟数据：torch.randn生成随机输入

## 5. 依赖和集成点

### 5.1 外部依赖
- PyTorch: 自动微分、GPU加速
- AnnData: 单细胞数据格式
- Scanpy: 预处理工具（未在核心代码中使用，但用于脚本）
- NumPy/Pandas: 数值计算和数据处理
- tqdm: 进度条
- pathlib: 路径操作

### 5.2 内部依赖关系
```
scripts/
  ├── 使用 src/train/train_embed_core.py
  ├── 使用 src/train/train_operator_core.py
  └── 使用 src/data/scperturb_dataset.py

src/train/
  ├── 使用 src/models/nb_vae.py
  ├── 使用 src/models/operator.py
  └── 使用 src/utils/edistance.py

src/utils/virtual_cell.py
  ├── 使用 src/models/nb_vae.py
  └── 使用 src/models/operator.py

所有模块
  └── 使用 src/config.py
```

## 6. 关键风险点

### 6.1 数值稳定性
- **风险**: E-distance计算的O(n²)复杂度，大batch时内存爆炸
- **缓解**: 使用`energy_distance_batched`分块计算
- **检查点**: train_operator_core.py第82-98行有NaN/Inf检查

### 6.2 数据格式要求
- **风险**: scPerturb数据格式不统一
- **缓解**: 预处理脚本标准化obs字段
- **必需字段**: tissue, perturbation, timepoint, dataset_id

### 6.3 GPU内存限制
- **风险**: 大模型 + 大batch导致OOM
- **缓解**: 使用梯度累积、混合精度训练
- **监控**: 跟踪batch_size、latent_dim、n_genes的组合

### 6.4 条件编码维度
- **风险**: cond_vec维度设计不当（过大/过小）
- **建议**: cond_dim=64（足够表达，不过参数化）
- **依据**: operator.py默认hidden_dim=64

## 7. 实验设计关键考虑

### 7.1 数据划分
- **训练集**: 70%条件（按θ分组，不按细胞）
- **验证集**: 15%条件
- **测试集**: 15%条件（用于最终评估）
- **零样本测试**: 单独保留未见扰动组合

### 7.2 评估任务
1. **In-distribution预测**: 同细胞类型、已见扰动
2. **跨细胞类型泛化**: 训练时未见的细胞类型
3. **零样本扰动组合**: 预测A+B（仅见过单独A和B）
4. **跨组织效应**: 比较肾脏vs脑的响应基loading

### 7.3 基准对比
- **简单基线**: 线性回归（每基因独立）
- **深度基线**: scGen（VAE风格迁移）
- **可选**: CPA、diffusion模型（如果算力允许）
- **对比指标**: E-distance、AUROC、计算效率

## 8. 技术选型理由

### 8.1 为什么用E-distance而非Wasserstein distance？
- **优势**: 计算高效（无需OT求解）、数学严格、scPerturb基准一致
- **缺点**: 对大batch内存敏感
- **适用场景**: 潜空间维度不高（d_z=32）、batch适中（<2000）

### 8.2 为什么用负二项分布？
- **数据特性**: scRNA-seq计数数据过离散、零膨胀
- **数学优势**: 比泊松更灵活（可学习离散度r）
- **参考**: scVI、totalVI等主流方法

### 8.3 为什么用低秩分解？
- **参数效率**: K个基 << |Θ|个独立算子
- **泛化能力**: 新条件可通过α_k(θ)预测
- **可解释性**: 响应基B_k对应生物学模式
