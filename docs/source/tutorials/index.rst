教程
====

本节提供完整的使用教程，从基础到高级逐步介绍虚拟细胞算子模型的使用。

.. toctree::
   :maxdepth: 2

   tutorial_01_basics
   tutorial_02_training
   tutorial_03_inference
   tutorial_04_advanced

教程概览
--------

教程1：基础概念
~~~~~~~~~~~~~~~

**时长**: 15分钟

**内容**:

- 理解VAE潜空间嵌入
- 理解算子理论
- 数据准备和预处理
- 基本模型使用

**适合对象**: 初学者

.. note::
   建议先阅读 :doc:`../quick_start` 了解基本安装和配置。

教程2：模型训练
~~~~~~~~~~~~~~~

**时长**: 30分钟

**内容**:

- 训练VAE模型
- 训练Operator模型
- 监控训练进度
- 模型检查点和早停

**适合对象**: 有基础的用户

教程3：推理和预测
~~~~~~~~~~~~~~~~~

**时长**: 20分钟

**内容**:

- 单步扰动预测
- 多步反事实模拟
- 结果可视化
- 性能优化

**适合对象**: 研究人员

教程4：高级应用
~~~~~~~~~~~~~~~

**时长**: 45分钟

**内容**:

- 跨组织效应分析
- 自定义损失函数
- 超参数调优
- 大规模数据处理

**适合对象**: 高级用户

完整示例
--------

端到端工作流程
~~~~~~~~~~~~~~

以下是一个完整的工作流程示例，展示如何从原始数据到最终预测：

.. code-block:: python

   # 1. 导入必要的库
   import torch
   import anndata
   from pathlib import Path

   from src.config import ModelConfig, TrainingConfig, set_seed, ConditionMeta
   from src.models.nb_vae import NBVAE
   from src.models.operator import OperatorModel
   from src.utils.cond_encoder import ConditionEncoder
   from src.data.scperturb_dataset import SCPerturbPairDataset, create_dataloaders
   from src.train.train_embed_core import train_embedding
   from src.train.train_operator_core import train_operator
   from src.utils.virtual_cell import virtual_cell_scenario

   # 2. 设置环境
   set_seed(42)
   device = "cuda" if torch.cuda.is_available() else "cpu"

   # 3. 加载数据
   adata = anndata.read_h5ad("data/scperturb.h5ad")

   # 4. 准备元数据
   cond_meta = ConditionMeta(
       perturbation_names=list(adata.obs["perturbation"].unique()),
       tissue_names=list(adata.obs["tissue"].unique()),
       timepoint_names=list(adata.obs["timepoint"].unique()),
       batch_names=list(adata.obs["batch"].unique())
   )

   cond_encoder = ConditionEncoder(cond_meta)
   tissue2idx = {t: i for i, t in enumerate(cond_meta.tissue_names)}

   # 5. 创建数据集
   dataset = SCPerturbPairDataset(
       adata=adata,
       cond_encoder=cond_encoder,
       tissue2idx=tissue2idx,
       max_pairs_per_condition=500,
       seed=42
   )

   train_loader, val_loader, test_loader = create_dataloaders(
       dataset,
       train_ratio=0.7,
       val_ratio=0.15,
       batch_size=64,
       num_workers=4
   )

   # 6. 训练VAE
   vae_model = NBVAE(
       n_genes=adata.n_vars,
       latent_dim=32,
       n_tissues=len(tissue2idx),
       hidden_dims=[256, 128]
   ).to(device)

   vae_config = TrainingConfig(
       n_epochs=50,
       learning_rate=1e-3,
       beta=1.0
   )

   vae_history = train_embedding(
       model=vae_model,
       train_loader=train_loader,
       config=vae_config,
       val_loader=val_loader,
       checkpoint_dir="checkpoints/vae",
       device=device
   )

   # 7. 训练Operator
   operator_model = OperatorModel(
       latent_dim=32,
       n_tissues=len(tissue2idx),
       n_response_bases=4,
       cond_dim=cond_encoder.get_dim()
   ).to(device)

   operator_config = TrainingConfig(
       n_epochs=100,
       learning_rate=1e-3,
       lambda_edist=1.0,
       lambda_spectral=0.1
   )

   operator_history = train_operator(
       operator_model=operator_model,
       embed_model=vae_model,
       train_loader=train_loader,
       config=operator_config,
       val_loader=val_loader,
       checkpoint_dir="checkpoints/operator",
       device=device
   )

   # 8. 虚拟细胞模拟
   # 选择一些初始细胞
   test_cells = adata[adata.obs["perturbation"] == "control"][:100]
   x0 = torch.tensor(test_cells.X.toarray(), dtype=torch.float32).to(device)

   # 准备组织信息
   tissue_labels = [tissue2idx[t] for t in test_cells.obs["tissue"]]
   tissue_onehot = torch.zeros(100, len(tissue2idx)).to(device)
   for i, t in enumerate(tissue_labels):
       tissue_onehot[i, t] = 1
   tissue_idx = torch.tensor(tissue_labels).to(device)

   # 定义扰动序列
   cond_vec_seq = []
   for timepoint in ["t1", "t2", "t3"]:
       cond_vec = cond_encoder.encode(
           perturbation="drug_A",
           tissue="kidney",
           timepoint=timepoint,
           batch="batch1"
       )
       cond_vec_seq.append(cond_vec.unsqueeze(0).expand(100, -1).to(device))

   # 运行模拟
   results = virtual_cell_scenario(
       vae_model=vae_model,
       operator_model=operator_model,
       x0=x0,
       tissue_onehot=tissue_onehot,
       tissue_idx=tissue_idx,
       cond_vec_seq=cond_vec_seq,
       device=device
   )

   # 9. 分析结果
   print("潜变量轨迹形状:", results["z_trajectory"].shape)  # (4, 100, 32)
   print("表达轨迹形状:", results["x_trajectory"].shape)    # (4, 100, n_genes)

   # 10. 保存预测
   torch.save(results, "results/virtual_cell_predictions.pt")

数据集示例
----------

scPerturb数据集
~~~~~~~~~~~~~~~

本项目主要使用scPerturb数据集进行验证。数据集包含：

- 多种扰动类型（药物、基因敲除等）
- 多个组织类型
- 时间序列数据
- 对照-扰动配对样本

**数据结构**:

.. code-block:: python

   adata = anndata.read_h5ad("data/scperturb.h5ad")

   # 基因表达矩阵
   adata.X  # (n_cells, n_genes)

   # 元数据
   adata.obs  # 包含:
   # - tissue: 组织类型
   # - perturbation: 扰动类型
   # - timepoint: 时间点
   # - batch: 批次信息
   # - dataset_id: 数据集ID

   # 基因信息
   adata.var  # 包含:
   # - gene_name: 基因名称
   # - highly_variable: 是否为高变基因

mLOY数据集
~~~~~~~~~~

Y染色体马赛克缺失（mLOY）跨组织分析数据：

- 肾脏组织样本
- 脑组织样本
- mLOY vs 正常对照
- 跨组织效应分析

**使用示例**:

.. code-block:: python

   # 分别加载肾脏和脑组织数据
   kidney_adata = anndata.read_h5ad("data/mLOY_kidney.h5ad")
   brain_adata = anndata.read_h5ad("data/mLOY_brain.h5ad")

   # 合并
   import anndata
   adata_combined = anndata.concat([kidney_adata, brain_adata], label="tissue")

   # 训练模型进行跨组织分析
   ...

评估指标
--------

模型性能评估
~~~~~~~~~~~~

**VAE重建质量**:

.. code-block:: python

   from src.utils.virtual_cell import compute_reconstruction_metrics

   with torch.no_grad():
       mu_x, r_x, mu_z, logvar_z = vae_model(x, tissue_onehot)

       mse, correlation = compute_reconstruction_metrics(x, mu_x)

   print(f"平均MSE: {mse.mean().item():.4f}")
   print(f"平均Pearson相关系数: {correlation.mean().item():.4f}")

**Operator预测准确性**:

.. code-block:: python

   from src.utils.edistance import energy_distance

   # 真实扰动后的细胞
   z_true = encode_cells(vae_model, x_perturbed, tissue_onehot)

   # 预测扰动后的细胞
   z_pred = apply_operator(operator_model, z_control, tissue_idx, cond_vec)

   # E-distance
   ed = energy_distance(z_pred, z_true)
   print(f"E-distance: {ed.item():.4f}")

可视化
------

UMAP可视化
~~~~~~~~~~

.. code-block:: python

   import umap
   import matplotlib.pyplot as plt

   # 编码到潜空间
   with torch.no_grad():
       z = encode_cells(vae_model, x, tissue_onehot, device)

   # UMAP降维
   reducer = umap.UMAP(n_components=2, random_state=42)
   z_umap = reducer.fit_transform(z.cpu().numpy())

   # 绘制
   plt.figure(figsize=(10, 8))
   scatter = plt.scatter(z_umap[:, 0], z_umap[:, 1],
                        c=tissue_labels, cmap='tab10', alpha=0.6)
   plt.colorbar(scatter, label='Tissue')
   plt.xlabel('UMAP 1')
   plt.ylabel('UMAP 2')
   plt.title('潜空间UMAP可视化')
   plt.savefig('results/umap_latent_space.png', dpi=300)

轨迹可视化
~~~~~~~~~~

.. code-block:: python

   # 多步模拟轨迹
   z_traj = results["z_trajectory"]  # (T+1, B, d_z)

   # 对每个时间点降维
   z_all = z_traj.reshape(-1, z_traj.shape[-1]).cpu().numpy()
   z_umap_all = reducer.fit_transform(z_all)

   # 重塑回轨迹形状
   T, B, _ = z_traj.shape
   z_umap_traj = z_umap_all.reshape(T, B, 2)

   # 绘制轨迹
   plt.figure(figsize=(12, 8))
   for i in range(min(10, B)):  # 绘制前10个细胞的轨迹
       plt.plot(z_umap_traj[:, i, 0], z_umap_traj[:, i, 1],
                marker='o', alpha=0.6, label=f'Cell {i}')

   plt.xlabel('UMAP 1')
   plt.ylabel('UMAP 2')
   plt.title('虚拟细胞轨迹')
   plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
   plt.tight_layout()
   plt.savefig('results/virtual_cell_trajectories.png', dpi=300)

训练曲线
~~~~~~~~

.. code-block:: python

   # 绘制训练历史
   plt.figure(figsize=(15, 5))

   # VAE损失
   plt.subplot(1, 3, 1)
   plt.plot(vae_history["train_loss"], label='Train')
   if "val_loss" in vae_history:
       plt.plot(vae_history["val_loss"], label='Validation')
   plt.xlabel('Epoch')
   plt.ylabel('ELBO Loss')
   plt.title('VAE训练曲线')
   plt.legend()

   # Operator损失
   plt.subplot(1, 3, 2)
   plt.plot(operator_history["train_edist_loss"], label='E-distance')
   plt.xlabel('Epoch')
   plt.ylabel('Loss')
   plt.title('Operator E-distance损失')

   # 谱范数惩罚
   plt.subplot(1, 3, 3)
   plt.plot(operator_history["train_spectral_penalty"], label='Spectral')
   plt.xlabel('Epoch')
   plt.ylabel('Penalty')
   plt.title('谱范数惩罚')

   plt.tight_layout()
   plt.savefig('results/training_curves.png', dpi=300)

常见工作流程
------------

新扰动预测
~~~~~~~~~~

预测一个新的、未见过的扰动对细胞的影响：

.. code-block:: python

   # 1. 选择初始细胞（对照组）
   control_cells = adata[adata.obs["perturbation"] == "control"][:100]

   # 2. 编码新扰动条件
   new_drug_cond = cond_encoder.encode(
       perturbation="new_drug_X",  # 新药物
       tissue="kidney",
       timepoint="t1",
       batch="batch1"
   )

   # 3. 预测
   z_control = encode_cells(vae_model, x_control, tissue_onehot, device)
   z_predicted = apply_operator(operator_model, z_control, tissue_idx,
                                 new_drug_cond.unsqueeze(0).expand(100, -1), device)
   x_predicted = decode_cells(vae_model, z_predicted, tissue_onehot, device)

   # 4. 分析预测的细胞状态
   # 找出差异表达基因
   fold_change = x_predicted / (x_control + 1e-8)
   top_genes = torch.topk(fold_change.mean(dim=0), k=50)

组合扰动
~~~~~~~~

预测多个扰动的组合效应：

.. code-block:: python

   # 编码单个扰动
   drug_A_cond = cond_encoder.encode(perturbation="drug_A", ...)
   drug_B_cond = cond_encoder.encode(perturbation="drug_B", ...)

   # 组合（简单平均）
   combination_cond = (drug_A_cond + drug_B_cond) / 2

   # 或学习组合系数
   alpha, beta = 0.6, 0.4
   combination_cond = alpha * drug_A_cond + beta * drug_B_cond

   # 预测组合效应
   z_combo = apply_operator(operator_model, z_control, tissue_idx,
                            combination_cond.unsqueeze(0).expand(100, -1), device)

跨组织比较
~~~~~~~~~~

比较同一扰动在不同组织中的效应：

.. code-block:: python

   # 相同的初始细胞状态和扰动
   perturbation_cond = cond_encoder.encode(perturbation="drug_A", ...)

   # 不同组织
   tissues = ["kidney", "brain", "blood"]
   results_by_tissue = {}

   for tissue in tissues:
       tissue_idx_curr = torch.tensor([tissue2idx[tissue]] * 100).to(device)
       tissue_onehot_curr = torch.zeros(100, len(tissue2idx)).to(device)
       tissue_onehot_curr[:, tissue2idx[tissue]] = 1

       z_pert = apply_operator(operator_model, z_control, tissue_idx_curr,
                               perturbation_cond.unsqueeze(0).expand(100, -1), device)
       x_pert = decode_cells(vae_model, z_pert, tissue_onehot_curr, device)

       results_by_tissue[tissue] = x_pert

   # 比较组织特异性响应
   for t1, t2 in [("kidney", "brain"), ("kidney", "blood"), ("brain", "blood")]:
       diff = (results_by_tissue[t1] - results_by_tissue[t2]).abs().mean()
       print(f"{t1} vs {t2}: 平均差异 = {diff.item():.4f}")

下一步
------

- 🎓 完成教程1-4的详细学习
- 💡 探索自己的研究问题
- 🔬 尝试自定义扰动和组织
- 📊 进行大规模预测实验
