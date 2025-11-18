快速开始
========

本指南将帮助您快速上手虚拟细胞算子模型。

安装
----

依赖要求
~~~~~~~~

- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (可选，用于GPU加速)

安装步骤
~~~~~~~~

1. 克隆仓库：

.. code-block:: bash

   git clone https://github.com/your-repo/virtual-cell-operator.git
   cd virtual-cell-operator

2. 安装依赖：

.. code-block:: bash

   pip install -r requirements.txt

3. 验证安装：

.. code-block:: python

   import torch
   from src.models.nb_vae import NBVAE
   from src.models.operator import OperatorModel

   print("✓ 安装成功！")

5分钟快速示例
-------------

训练VAE模型
~~~~~~~~~~~

.. code-block:: python

   import torch
   from src.models.nb_vae import NBVAE, elbo_loss
   from src.config import set_seed
   from torch.utils.data import DataLoader

   # 设置随机种子
   set_seed(42)

   # 创建模型
   model = NBVAE(
       n_genes=2000,
       latent_dim=32,
       n_tissues=3
   )

   # 准备数据 (假设已有adata)
   # X = torch.tensor(adata.X, dtype=torch.float32)
   # ... 创建DataLoader ...

   # 训练循环
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

   for epoch in range(10):
       for x_batch, tissue_batch in train_loader:
           loss, loss_dict = elbo_loss(x_batch, tissue_batch, model)

           optimizer.zero_grad()
           loss.backward()
           optimizer.step()

       print(f"Epoch {epoch}: loss={loss.item():.4f}")

训练Operator模型
~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.models.operator import OperatorModel
   from src.train.train_operator_core import train_operator

   # 创建Operator
   operator = OperatorModel(
       latent_dim=32,
       n_tissues=3,
       n_response_bases=4,
       cond_dim=64
   )

   # 使用训练循环
   from src.config import TrainingConfig

   config = TrainingConfig(
       n_epochs=50,
       learning_rate=1e-3,
       lambda_edist=1.0,
       lambda_spectral=0.1
   )

   history = train_operator(
       operator_model=operator,
       embed_model=vae_model,  # 已训练的VAE
       train_loader=train_loader,
       config=config
   )

虚拟细胞生成
~~~~~~~~~~~~

.. code-block:: python

   from src.utils.virtual_cell import virtual_cell_scenario

   # 定义扰动序列
   cond_vec_seq = [
       drug_A_vector,   # t=1
       drug_B_vector,   # t=2
       combination_vector  # t=3
   ]

   # 运行模拟
   results = virtual_cell_scenario(
       vae_model=vae_model,
       operator_model=operator_model,
       x0=initial_cells,  # (B, G) 初始基因表达
       tissue_onehot=tissue_labels,
       tissue_idx=tissue_indices,
       cond_vec_seq=cond_vec_seq
   )

   # 查看结果
   z_trajectory = results["z_trajectory"]  # (T+1, B, d_z)
   x_trajectory = results["x_trajectory"]  # (T+1, B, G)

常见任务
--------

加载预训练模型
~~~~~~~~~~~~~~

.. code-block:: python

   import torch

   # 加载VAE
   vae = NBVAE(n_genes=2000, latent_dim=32, n_tissues=3)
   vae.load_state_dict(torch.load("checkpoints/vae_best.pt"))
   vae.eval()

   # 加载Operator
   operator = OperatorModel(latent_dim=32, n_tissues=3, n_response_bases=4, cond_dim=64)
   operator.load_state_dict(torch.load("checkpoints/operator_best.pt"))
   operator.eval()

保存模型检查点
~~~~~~~~~~~~~~

.. code-block:: python

   # 保存
   torch.save({
       'epoch': epoch,
       'model_state_dict': model.state_dict(),
       'optimizer_state_dict': optimizer.state_dict(),
       'loss': loss,
   }, f'checkpoints/checkpoint_epoch_{epoch}.pt')

   # 加载
   checkpoint = torch.load('checkpoints/checkpoint_epoch_50.pt')
   model.load_state_dict(checkpoint['model_state_dict'])
   optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
   epoch = checkpoint['epoch']

使用GPU
~~~~~~~

.. code-block:: python

   device = "cuda" if torch.cuda.is_available() else "cpu"

   # 模型到GPU
   model = model.to(device)

   # 数据到GPU
   x = x.to(device)
   tissue_onehot = tissue_onehot.to(device)

   # 训练
   loss, _ = elbo_loss(x, tissue_onehot, model)

数据准备
--------

从AnnData加载
~~~~~~~~~~~~~

.. code-block:: python

   import anndata
   import numpy as np
   import torch

   # 读取h5ad文件
   adata = anndata.read_h5ad("data/scperturb_data.h5ad")

   # 提取表达矩阵
   X = torch.tensor(adata.X.toarray() if sparse.issparse(adata.X) else adata.X,
                    dtype=torch.float32)

   # 组织标签
   tissue2idx = {"kidney": 0, "brain": 1, "blood": 2}
   tissue_labels = [tissue2idx[t] for t in adata.obs["tissue"]]

   # One-hot编码
   tissue_onehot = torch.zeros(len(tissue_labels), len(tissue2idx))
   for i, t in enumerate(tissue_labels):
       tissue_onehot[i, t] = 1

创建配对数据集
~~~~~~~~~~~~~~

.. code-block:: python

   from src.data.scperturb_dataset import SCPerturbPairDataset
   from src.utils.cond_encoder import ConditionEncoder
   from src.config import ConditionMeta

   # 定义条件元数据
   cond_meta = ConditionMeta(
       perturbation_names=["control", "drug_A", "drug_B"],
       tissue_names=["kidney", "brain", "blood"],
       timepoint_names=["t0", "t1", "t2"],
       batch_names=["batch1", "batch2"]
   )

   # 创建条件编码器
   cond_encoder = ConditionEncoder(cond_meta)

   # 创建数据集
   dataset = SCPerturbPairDataset(
       adata=adata,
       cond_encoder=cond_encoder,
       tissue2idx=tissue2idx,
       max_pairs_per_condition=500,
       seed=42
   )

   # 创建DataLoader
   from torch.utils.data import DataLoader

   train_loader = DataLoader(
       dataset,
       batch_size=64,
       shuffle=True,
       num_workers=4  # 多进程加载
   )

下一步
------

- 📚 阅读 :doc:`tutorials/index` 了解详细教程
- 📖 查看 :doc:`api/index` 获取完整API参考
- 🧮 学习 :doc:`mathematical_foundation` 理解数学原理
- ⚡ 查看 :doc:`optimization_tips` 获取性能优化建议

故障排除
--------

常见问题
~~~~~~~~

**Q: ImportError: No module named 'src'**

A: 确保在项目根目录运行，或将项目根目录添加到Python路径：

.. code-block:: python

   import sys
   sys.path.insert(0, '/path/to/virtual-cell-operator')

**Q: CUDA out of memory**

A: 减小batch size或使用梯度累积：

.. code-block:: python

   # 减小batch size
   train_loader = DataLoader(dataset, batch_size=32)  # 原来64

   # 或使用梯度累积
   accumulation_steps = 4
   for i, batch in enumerate(train_loader):
       loss = compute_loss(batch)
       loss = loss / accumulation_steps
       loss.backward()

       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()

**Q: 训练损失不下降**

A: 检查以下几点：

1. 学习率是否合适（尝试1e-4到1e-3）
2. 数据是否正确归一化
3. Beta参数是否过大（KL项权重）
4. 是否有梯度爆炸/消失（检查梯度范数）

获取帮助
~~~~~~~~

- 📧 邮件: support@virtual-cell.org
- 💬 Issues: https://github.com/your-repo/virtual-cell-operator/issues
- 📖 文档: https://virtual-cell-operator.readthedocs.io
