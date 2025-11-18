虚拟细胞算子模型 - API文档
=====================================

欢迎来到虚拟细胞算子模型（Virtual Cell Operator Model）项目的API文档。

本项目实现了基于算子理论的虚拟细胞模型，用于单细胞RNA测序（scRNA-seq）数据的扰动响应预测和反事实模拟。

.. toctree::
   :maxdepth: 2
   :caption: 目录

   quick_start
   api/index
   tutorials/index
   mathematical_foundation
   changelog

项目概述
--------

核心功能
~~~~~~~~

本项目提供以下核心功能：

1. **VAE潜空间嵌入**：使用负二项VAE（NB-VAE）将高维基因表达数据嵌入到低维潜空间
2. **扰动响应算子**：学习条件依赖的线性算子来建模细胞对扰动的响应
3. **虚拟细胞生成**：通过算子应用实现反事实模拟和虚拟干预
4. **跨组织分析**：支持不同组织间的差异响应建模

数学框架
~~~~~~~~

模型的核心数学框架：

.. math::

   \\begin{align}
   \\text{编码:} \\quad & z \\sim q_\\phi(z | x, t) \\\\
   \\text{算子:} \\quad & z' = A_\\theta(t, c) z + b_\\theta(t, c) \\\\
   \\text{解码:} \\quad & x' \\sim p_\\psi(x | z', t)
   \\end{align}

其中：

- :math:`x` 是基因表达向量
- :math:`z` 是潜变量
- :math:`t` 是组织类型
- :math:`c` 是扰动条件
- :math:`A_\\theta, b_\\theta` 是学习的算子参数

详细的数学推导请参见 :doc:`mathematical_foundation`。

快速开始
--------

安装依赖
~~~~~~~~

.. code-block:: bash

   pip install -r requirements.txt

基本使用
~~~~~~~~

.. code-block:: python

   from src.models.nb_vae import NBVAE
   from src.models.operator import OperatorModel
   from src.utils.virtual_cell import virtual_cell_scenario

   # 创建模型
   vae = NBVAE(n_genes=2000, latent_dim=32, n_tissues=3)
   operator = OperatorModel(latent_dim=32, n_tissues=3, n_response_bases=4, cond_dim=64)

   # 训练... (详见教程)

   # 虚拟细胞模拟
   results = virtual_cell_scenario(
       vae_model=vae,
       operator_model=operator,
       x0=initial_cells,
       tissue_onehot=tissue_labels,
       tissue_idx=tissue_indices,
       cond_vec_seq=perturbation_sequence
   )

更多详细教程请参见 :doc:`tutorials/index`。

主要模块
--------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - 模块
     - 功能描述
   * - :mod:`src.models.nb_vae`
     - 负二项VAE模型，用于潜空间嵌入
   * - :mod:`src.models.operator`
     - 扰动响应算子模型
   * - :mod:`src.utils.virtual_cell`
     - 虚拟细胞生成和模拟工具
   * - :mod:`src.utils.edistance`
     - 能量距离（E-distance）计算
   * - :mod:`src.data.scperturb_dataset`
     - scPerturb数据集加载和预处理
   * - :mod:`src.train.train_embed_core`
     - VAE训练循环
   * - :mod:`src.train.train_operator_core`
     - Operator训练循环

性能优化
--------

本项目在代码质量和性能方面进行了深度优化：

- ✅ 向量化计算（20倍加速）
- ✅ 内存优化（80%内存降低）
- ✅ 数值稳定性保证
- ✅ 完整的单元测试覆盖（56个测试用例）
- ✅ 集成测试覆盖端到端流程

详见 :doc:`optimization_report`。

引用
----

如果您在研究中使用了本项目，请引用：

.. code-block:: bibtex

   @software{virtual_cell_operator_2025,
     title={Virtual Cell Operator Model: A Perturbation Response Framework for Single-Cell Data},
     author={Virtual Cell Operator Team},
     year={2025},
     url={https://github.com/your-repo/virtual-cell-operator}
   }

许可证
------

本项目采用 MIT 许可证。

索引和表格
----------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
