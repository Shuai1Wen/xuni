# 实验分析使用指南

本文档详细说明如何使用新创建的实验分析代码进行完整的模型训练、评估和分析。

## 📋 目录

1. [环境准备](#环境准备)
2. [数据预处理](#数据预处理)
3. [模型训练](#模型训练)
4. [模型评估](#模型评估)
5. [响应基分析](#响应基分析)
6. [完整实验流程](#完整实验流程)
7. [故障排除](#故障排除)

---

## 环境准备

### 依赖库

确保已安装以下Python库：

```bash
pip install torch scanpy anndata numpy pandas matplotlib seaborn scikit-learn scipy pyyaml umap-learn tqdm
```

### 目录结构检查

运行实验前，确保以下目录结构存在：

```
virtual-cell-operator-mLOY/
├── configs/                  # ✓ 配置文件
├── data/
│   ├── raw/                  # 原始数据（需要手动准备）
│   └── processed/            # 预处理后数据
├── src/                      # ✓ 源代码
│   ├── evaluation/           # ✓ 评估模块
│   ├── visualization/        # ✓ 可视化模块
│   └── ...
├── scripts/                  # ✓ 实验脚本
│   ├── experiments/          # ✓ 训练和评估脚本
│   └── preprocessing/        # ✓ 预处理脚本
└── results/                  # 结果输出（自动创建）
```

---

## 数据预处理

### 第1步：准备原始数据

将scPerturb原始数据放置在 `data/raw/scperturb/` 目录：

```bash
mkdir -p data/raw/scperturb/
# 将您的 scPerturb h5ad 文件放置在此目录
```

**必需的数据字段**：
- `adata.obs["perturbation"]`: 扰动类型（如 "drug_A", "KO_geneX", "control"）
- `adata.obs["tissue"]`: 组织类型（如 "blood", "kidney", "brain"）
- `adata.obs["timepoint"]`: 时间点（"t0" 或 "t1"）
- `adata.obs["dataset_id"]`: 数据集标识
- `adata.X`: 基因表达矩阵（原始计数或归一化）

### 第2步：运行预处理脚本

```bash
python scripts/preprocessing/preprocess_scperturb.py \
    --input data/raw/scperturb/your_data.h5ad \
    --output data/processed/scperturb/ \
    --n_top_genes 2000 \
    --min_cells 100 \
    --min_genes 200 \
    --test_split 0.15 \
    --val_split 0.15
```

**参数说明**：
- `--n_top_genes`: 选择的高变基因数量（推荐2000）
- `--min_cells`: 基因至少在多少个细胞中表达（推荐100）
- `--min_genes`: 细胞至少表达多少个基因（推荐200）
- `--test_split`: 测试集比例（推荐0.15）
- `--val_split`: 验证集比例（推荐0.15）

**输出文件**：
```
data/processed/scperturb/
├── scperturb_merged_train.h5ad    # 训练集
├── scperturb_merged_val.h5ad      # 验证集
├── scperturb_merged_test.h5ad     # 测试集
└── metadata.json                   # 数据集元信息
```

---

## 模型训练

### 第3步：训练VAE（潜空间嵌入）

```bash
python scripts/experiments/train_scperturb_baseline.py \
    --phase vae \
    --config configs/scperturb_vae.yaml
```

**配置文件** (`configs/scperturb_vae.yaml`)：
- 调整 `model.latent_dim` 控制潜空间维度（推荐32）
- 调整 `training.batch_size` 根据GPU内存（推荐512）
- 调整 `training.n_epochs_embed` 控制训练轮数（推荐100）

**输出**：
```
results/checkpoints/scperturb_vae/
├── best_model.pt              # 最佳VAE模型
├── last_model.pt              # 最后一个epoch的模型
├── config.yaml                # 保存的配置
└── ...
results/logs/scperturb_vae/
└── training_history.json      # 训练历史
```

**预期训练时间**：
- 单GPU（RTX 3090）：4-6小时
- 双GPU（A100）：2-3小时

### 第4步：训练算子模型

```bash
python scripts/experiments/train_scperturb_baseline.py \
    --phase operator \
    --config configs/scperturb_operator.yaml \
    --vae_checkpoint results/checkpoints/scperturb_vae/best_model.pt
```

**配置文件** (`configs/scperturb_operator.yaml`)：
- 调整 `model.n_response_bases` 控制响应基数量（推荐5）
- 调整 `model.cond_dim` 控制条件向量维度（推荐64）
- 调整 `training.lambda_e` 控制E-distance损失权重（推荐1.0）
- 调整 `training.lambda_stab` 控制稳定性正则化（推荐0.001）

**输出**：
```
results/checkpoints/scperturb_operator/
├── best_operator.pt           # 最佳算子模型
├── cond_encoder.pt            # 条件编码器
├── config.yaml                # 保存的配置
└── ...
results/logs/scperturb_operator/
└── training_history.json      # 训练历史
```

**预期训练时间**：
- 单GPU（RTX 3090）：6-8小时
- 双GPU（A100）：3-4小时

---

## 模型评估

### 第5步：评估扰动预测性能

```bash
python scripts/experiments/eval_perturbation_prediction.py \
    --vae_checkpoint results/checkpoints/scperturb_vae/best_model.pt \
    --operator_checkpoint results/checkpoints/scperturb_operator/best_operator.pt \
    --data_path data/processed/scperturb/scperturb_merged_test.h5ad \
    --output_dir results/experiments/scperturb_evaluation/ \
    --device cuda
```

**参数说明**：
- `--no_de_metrics`: 跳过差异基因指标计算（加快评估速度）
- `--batch_size`: 批次大小（默认256）

**输出文件**：
```
results/experiments/scperturb_evaluation/
├── metrics.json                        # 评估指标摘要
├── predictions/
│   ├── x0.npy                          # 对照表达
│   ├── x1_true.npy                     # 真实扰动表达
│   ├── x1_pred.npy                     # 预测扰动表达
│   ├── z1_true.npy                     # 真实潜变量
│   ├── z1_pred.npy                     # 预测潜变量
│   └── spectral_norms.npy              # 谱范数
└── figures/
    ├── evaluation_summary.png          # 评估摘要
    ├── latent_space_umap.png           # 潜空间UMAP
    ├── de_genes_scatter.png            # 差异基因散点图
    └── spectral_norm_histogram.png     # 谱范数直方图
```

**关键评估指标**：

| 指标类别 | 指标名称 | 含义 | 期望值 |
|---------|---------|------|-------|
| 重建质量 | Pearson (mean) | 基因表达重建相关性 | > 0.7 |
| 分布匹配 | E-distance | 潜空间分布距离 | < 0.3 |
| 差异基因 | AUROC | 差异基因预测准确性 | > 0.8 |
| 算子质量 | Spectral norm (mean) | 算子稳定性 | < 1.05 |

---

## 响应基分析

### 第6步：分析响应基和激活模式

```bash
python scripts/experiments/analyze_response_axes.py \
    --operator_checkpoint results/checkpoints/scperturb_operator/best_operator.pt \
    --vae_checkpoint results/checkpoints/scperturb_vae/best_model.pt \
    --data_path data/processed/scperturb/scperturb_merged_train.h5ad \
    --output_dir results/experiments/response_axes_analysis/ \
    --device cuda
```

**输出文件**：
```
results/experiments/response_axes_analysis/
├── analysis_summary.json                   # 分析摘要
├── response_bases.npy                      # 响应基矩阵 (K, d_z, d_z)
├── activation_matrix.npy                   # 激活矩阵 (n_conditions, K)
├── condition_names.txt                     # 条件名称列表
├── basis_similarity_matrix.npy             # 响应基相似度矩阵
└── figures/
    ├── response_heatmap.png                # 响应系数热图
    ├── basis_similarity_matrix.png         # 响应基相似度矩阵
    └── condition_clustering_dendrogram.png # 条件聚类树状图
```

**分析洞察**：
1. **响应系数热图**：展示不同条件下各响应基的激活强度
2. **响应基相似度**：检查响应基是否正交（低冗余）
3. **条件聚类**：识别具有相似响应模式的扰动

---

## 完整实验流程

### 端到端实验示例

```bash
#!/bin/bash
# 完整实验流程脚本

# 1. 数据预处理
echo "=== 步骤1: 数据预处理 ==="
python scripts/preprocessing/preprocess_scperturb.py \
    --input data/raw/scperturb/raw_data.h5ad \
    --output data/processed/scperturb/ \
    --n_top_genes 2000

# 2. 训练VAE
echo "=== 步骤2: 训练VAE ==="
python scripts/experiments/train_scperturb_baseline.py \
    --phase vae \
    --config configs/scperturb_vae.yaml

# 3. 训练算子
echo "=== 步骤3: 训练算子 ==="
python scripts/experiments/train_scperturb_baseline.py \
    --phase operator \
    --config configs/scperturb_operator.yaml \
    --vae_checkpoint results/checkpoints/scperturb_vae/best_model.pt

# 4. 评估模型
echo "=== 步骤4: 评估模型 ==="
python scripts/experiments/eval_perturbation_prediction.py \
    --vae_checkpoint results/checkpoints/scperturb_vae/best_model.pt \
    --operator_checkpoint results/checkpoints/scperturb_operator/best_operator.pt \
    --data_path data/processed/scperturb/scperturb_merged_test.h5ad \
    --output_dir results/experiments/scperturb_evaluation/

# 5. 响应基分析
echo "=== 步骤5: 响应基分析 ==="
python scripts/experiments/analyze_response_axes.py \
    --operator_checkpoint results/checkpoints/scperturb_operator/best_operator.pt \
    --vae_checkpoint results/checkpoints/scperturb_vae/best_model.pt \
    --data_path data/processed/scperturb/scperturb_merged_train.h5ad \
    --output_dir results/experiments/response_axes_analysis/

echo "=== 实验完成! ==="
```

保存为 `run_full_experiment.sh` 并执行：
```bash
chmod +x run_full_experiment.sh
./run_full_experiment.sh
```

---

## 故障排除

### 常见问题

#### 问题1：CUDA内存不足

**症状**：
```
RuntimeError: CUDA out of memory
```

**解决方案**：
1. 减小批次大小：
   ```yaml
   # configs/scperturb_vae.yaml
   training:
     batch_size: 256  # 从512减小到256
   ```

2. 使用混合精度训练（需修改训练代码）

3. 使用更小的模型：
   ```yaml
   model:
     latent_dim: 16    # 从32减小到16
     hidden_dim: 256   # 从512减小到256
   ```

#### 问题2：训练不收敛

**症状**：
- 损失不下降
- 验证指标很差

**解决方案**：
1. 检查数据质量：
   ```python
   import scanpy as sc
   adata = sc.read_h5ad("data/processed/scperturb/scperturb_merged_train.h5ad")
   print(adata.obs["perturbation"].value_counts())  # 检查条件分布
   print(adata.X.min(), adata.X.max())  # 检查表达范围
   ```

2. 调整学习率：
   ```yaml
   training:
     lr_embed: 0.0001  # 减小学习率
   ```

3. 增加warmup：
   ```yaml
   training:
     warmup_epochs: 20  # 从10增加到20
   ```

#### 问题3：评估指标低于预期

**症状**：
- Pearson correlation < 0.5
- E-distance > 0.5

**解决方案**：
1. 检查VAE重建质量是否良好
2. 增加响应基数量：
   ```yaml
   model:
     n_response_bases: 10  # 从5增加到10
   ```

3. 调整损失权重：
   ```yaml
   training:
     lambda_e: 2.0      # 增加E-distance权重
     lambda_stab: 0.01  # 增加稳定性正则化
   ```

#### 问题4：文件路径错误

**症状**：
```
FileNotFoundError: [Errno 2] No such file or directory
```

**解决方案**：
使用绝对路径或检查当前工作目录：
```bash
# 在项目根目录运行所有脚本
cd /path/to/virtual-cell-operator-mLOY/
python scripts/experiments/train_scperturb_baseline.py ...
```

---

## 高级用法

### 自定义评估指标

在 `src/evaluation/metrics.py` 中添加自定义指标：

```python
def custom_metric(x_true, x_pred):
    """您的自定义指标"""
    # 实现您的指标计算
    return metric_value
```

### 自定义可视化

在 `src/visualization/plotting.py` 中添加自定义绘图函数：

```python
def plot_custom_visualization(data, save_path):
    """您的自定义可视化"""
    # 实现您的绘图逻辑
    plt.savefig(save_path)
```

---

## 参考文档

- **数学原理**：查看 `model.md`
- **代码实现**：查看 `suanfa.md`
- **项目结构**：查看 `details.md`
- **开发准则**：查看 `CLAUDE.md`

---

## 支持与反馈

如有问题或建议，请查阅项目文档或联系项目维护者。
