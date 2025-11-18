API参考
=======

本节提供所有模块、类和函数的详细API文档。

.. toctree::
   :maxdepth: 2

   models
   utils
   data
   train
   config

核心模块结构
------------

项目的代码组织结构如下：

.. code-block:: text

   src/
   ├── models/          # 模型定义
   │   ├── nb_vae.py    # 负二项VAE
   │   └── operator.py  # 算子模型
   ├── utils/           # 工具函数
   │   ├── virtual_cell.py   # 虚拟细胞模拟
   │   ├── edistance.py      # E-distance计算
   │   └── cond_encoder.py   # 条件编码器
   ├── data/            # 数据加载
   │   └── scperturb_dataset.py
   ├── train/           # 训练循环
   │   ├── train_embed_core.py
   │   └── train_operator_core.py
   └── config.py        # 配置管理

使用约定
--------

参数命名
~~~~~~~~

本项目遵循以下命名约定：

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - 符号
     - 代码变量
     - 含义
   * - :math:`x`
     - ``x``
     - 基因表达向量 :math:`\\in \\mathbb{R}^G`
   * - :math:`z`
     - ``z``
     - 潜变量 :math:`\\in \\mathbb{R}^{d_z}`
   * - :math:`A_\\theta`
     - ``A_theta``
     - 算子矩阵 :math:`\\in \\mathbb{R}^{d_z \\times d_z}`
   * - :math:`\\mu`
     - ``mu``
     - NB分布均值参数
   * - :math:`r`
     - ``r``
     - NB分布离散参数
   * - :math:`t`
     - ``tissue_idx``
     - 组织索引
   * - :math:`c`
     - ``cond_vec``
     - 条件向量

张量形状注释
~~~~~~~~~~~~

代码中使用以下形状注释约定：

- ``B``: batch size
- ``G``: 基因数量
- ``d_z``: 潜空间维度
- ``T``: 组织类型数量
- ``K``: 响应基数量

示例：

.. code-block:: python

   def forward(self, x, tissue_onehot):
       """
       参数:
           x: (B, G) 基因表达
           tissue_onehot: (B, T) 组织one-hot编码

       返回:
           mu_x: (B, G) 重建均值
           r_x: (B, G) NB参数
           mu_z: (B, d_z) 潜变量均值
           logvar_z: (B, d_z) 潜变量对数方差
       """

数学对应关系
~~~~~~~~~~~~

代码实现与 ``model.md`` 中的数学公式对应关系：

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - model.md公式
     - 代码位置
     - 函数/类
   * - 公式 (A.2.1) - VAE编码器
     - ``src/models/nb_vae.py``
     - ``Encoder.forward()``
   * - 公式 (A.2.2) - NB解码器
     - ``src/models/nb_vae.py``
     - ``DecoderNB.forward()``
   * - 公式 (A.3.1) - ELBO损失
     - ``src/models/nb_vae.py``
     - ``elbo_loss()``
   * - 公式 (A.4) - E-distance
     - ``src/utils/edistance.py``
     - ``energy_distance()``
   * - 公式 (A.5.1) - 低秩分解
     - ``src/models/operator.py``
     - ``OperatorModel.forward()``
   * - 公式 (A.6) - 谱范数约束
     - ``src/models/operator.py``
     - ``spectral_penalty()``
   * - 公式 (A.8.2) - 多步模拟
     - ``src/utils/virtual_cell.py``
     - ``virtual_cell_scenario()``

类型注解
~~~~~~~~

本项目使用Python类型注解提升代码可读性：

.. code-block:: python

   from typing import Tuple, Optional, Dict, List
   import torch

   def encode_cells(
       vae: NBVAE,
       x: torch.Tensor,
       tissue_onehot: torch.Tensor,
       device: str = "cuda"
   ) -> torch.Tensor:
       ...

配置管理
~~~~~~~~

所有数值稳定性参数通过 ``NumericalConfig`` 集中管理：

.. code-block:: python

   from src.config import NumericalConfig

   cfg = NumericalConfig()
   # cfg.eps_distance = 1e-8
   # cfg.eps_division = 1e-8
   # cfg.eps_log = 1e-8
   # cfg.tol_test = 1e-6
