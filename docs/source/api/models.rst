模型模块 (models)
==================

本模块包含核心的深度学习模型实现。

nb_vae - 负二项VAE
-------------------

.. automodule:: src.models.nb_vae
   :members:
   :undoc-members:
   :show-inheritance:

核心类
~~~~~~

NBVAE
^^^^^

.. autoclass:: src.models.nb_vae.NBVAE
   :members:
   :special-members: __init__
   :show-inheritance:

   负二项变分自编码器（Negative Binomial Variational Autoencoder）

   **数学原理**

   NBVAE使用负二项分布建模单细胞RNA测序数据的计数特性：

   .. math::

      \\begin{align}
      q_\\phi(z | x, t) &= \\mathcal{N}(z | \\mu_z, \\text{diag}(\\sigma_z^2)) \\\\
      p_\\psi(x | z, t) &= \\prod_{g=1}^G \\text{NB}(x_g | \\mu_g, r_g) \\\\
      \\mathcal{L} &= \\mathbb{E}_{q_\\phi}[\\log p_\\psi(x|z,t)] - \\beta \\cdot D_{KL}(q_\\phi(z|x,t) || p(z))
      \\end{align}

   **使用示例**

   .. code-block:: python

      from src.models.nb_vae import NBVAE

      # 创建模型
      model = NBVAE(
          n_genes=2000,
          latent_dim=32,
          n_tissues=3,
          hidden_dim=256  # 隐藏层维度（整数）
      )

      # 前向传播
      mu_x, r_x, mu_z, logvar_z = model(x, tissue_onehot)

      # 训练
      loss, loss_dict = elbo_loss(x, tissue_onehot, model, beta=1.0)
      loss.backward()

Encoder
^^^^^^^

.. autoclass:: src.models.nb_vae.Encoder
   :members:
   :special-members: __init__
   :show-inheritance:

   VAE编码器，将基因表达映射到潜空间。

   **架构**

   .. code-block:: text

      x ∈ ℝ^G
        ↓ [Linear + ReLU]
      h ∈ ℝ^hidden_dim
        ↓ [concat tissue]
      h_cat ∈ ℝ^(hidden_dim+T)
        ├→ [Linear] → μ_z ∈ ℝ^d_z
        └→ [Linear] → log σ²_z ∈ ℝ^d_z

DecoderNB
^^^^^^^^^

.. autoclass:: src.models.nb_vae.DecoderNB
   :members:
   :special-members: __init__
   :show-inheritance:

   负二项解码器，生成基因表达重建。

   **输出约束**

   - :math:`\\mu > 0`: 使用 ``F.softplus(·) + \\epsilon``
   - :math:`r > 0`: 使用 ``F.softplus(·)``

损失函数
~~~~~~~~

nb_log_likelihood
^^^^^^^^^^^^^^^^^

.. autofunction:: src.models.nb_vae.nb_log_likelihood

   负二项对数似然函数。

   **数学定义**

   .. math::

      \\log p(x|\\mu, r) = \\sum_g \\left[
         \\log\\Gamma(x_g + r_g) - \\log\\Gamma(r_g) - \\log\\Gamma(x_g+1)
         + r_g \\log\\frac{r_g}{r_g + \\mu_g}
         + x_g \\log\\frac{\\mu_g}{r_g + \\mu_g}
      \\right]

   **数值稳定性**

   使用 ``torch.lgamma`` 和 epsilon 保护防止数值下溢。

elbo_loss
^^^^^^^^^

.. autofunction:: src.models.nb_vae.elbo_loss

   ELBO（Evidence Lower Bound）损失函数。

   **组成**

   1. **重建损失**: :math:`-\\mathbb{E}_{q(z|x)}[\\log p(x|z)]`
   2. **KL散度**: :math:`D_{KL}(q(z|x) || p(z))`

   **返回值**

   - ``loss``: 标量，总损失
   - ``loss_dict``: 字典，包含 ``recon_loss`` 和 ``kl_loss``

---

operator - 算子模型
--------------------

.. automodule:: src.models.operator
   :members:
   :undoc-members:
   :show-inheritance:

核心类
~~~~~~

OperatorModel
^^^^^^^^^^^^^

.. autoclass:: src.models.operator.OperatorModel
   :members:
   :special-members: __init__
   :show-inheritance:

   扰动响应算子模型。

   **数学框架**

   算子采用低秩分解结构：

   .. math::

      \\begin{align}
      A_\\theta(t, c) &= A_t^{(0)} + \\sum_{k=1}^K \\alpha_k(c) B_k \\\\
      b_\\theta(t, c) &= \\sum_{k=1}^K \\beta_k(c) u_k \\\\
      z' &= A_\\theta(t, c) z + b_\\theta(t, c)
      \\end{align}

   其中：

   - :math:`A_t^{(0)} \\in \\mathbb{R}^{d_z \\times d_z}`: 组织 ``t`` 的基线算子
   - :math:`B_k \\in \\mathbb{R}^{d_z \\times d_z}`: 第 ``k`` 个响应基
   - :math:`\\alpha_k(c)`: 由条件 ``c`` 决定的系数

   **使用示例**

   .. code-block:: python

      from src.models.operator import OperatorModel

      # 创建模型
      operator = OperatorModel(
          latent_dim=32,
          n_tissues=3,
          n_response_bases=4,
          cond_dim=64
      )

      # 应用算子
      z_out, A_theta, b_theta = operator(z, tissue_idx, cond_vec)

      # 计算谱范数惩罚
      penalty = operator.spectral_penalty(max_allowed=1.05)

方法详解
~~~~~~~~

forward
^^^^^^^

.. automethod:: src.models.operator.OperatorModel.forward

   执行算子变换。

   **实现细节**

   1. 根据组织索引获取基线算子 :math:`A_t^{(0)}`
   2. 通过MLP将条件向量映射到系数 :math:`\\alpha, \\beta`
   3. 计算响应项：:math:`\\sum_k \\alpha_k B_k`
   4. 批量矩阵乘法：:math:`(A_t^{(0)} + \\sum_k \\alpha_k B_k) z + \\sum_k \\beta_k u_k`

   **向量化实现**

   使用 ``torch.einsum`` 实现高效计算：

   .. code-block:: python

      A_res = torch.einsum('bk,kij->bij', alpha, self.B)
      b_res = torch.einsum('bk,ki->bi', beta, self.u)

spectral_penalty
^^^^^^^^^^^^^^^^

.. automethod:: src.models.operator.OperatorModel.spectral_penalty

   计算谱范数正则化惩罚。

   **算法**: 幂迭代法（Power Iteration）

   .. math::

      \\begin{align}
      v^{(0)} &\\sim \\mathcal{N}(0, I) \\\\
      v^{(t+1)} &= \\frac{A v^{(t)}}{||A v^{(t)}||} \\\\
      \\rho(A) &\\approx v^{(T)\\top} A v^{(T)}
      \\end{align}

   **惩罚函数**

   .. math::

      \\text{penalty} = \\max(0, \\rho(A_t^{(0)}) - \\rho_{\\text{max}})^2

compute_operator_norm
^^^^^^^^^^^^^^^^^^^^^

.. automethod:: src.models.operator.OperatorModel.compute_operator_norm

   计算批量算子的谱范数。

   **返回**: 每个样本的谱范数 :math:`(B,)`

---

性能注意事项
------------

向量化计算
~~~~~~~~~~

所有模型都使用了向量化实现以提升性能：

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - 操作
     - 避免使用
     - 推荐使用
   * - 批量矩阵乘法
     - ``for`` 循环
     - ``torch.bmm`` / ``torch.einsum``
   * - 元素级运算
     - 逐元素循环
     - 广播操作
   * - 归一化
     - 手动计算
     - ``dim=-1, keepdim=True``

内存优化
~~~~~~~~

- **Operator模型**: 使用 ``einsum`` 替代 ``expand()`` 降低80%内存占用
- **VAE解码**: 使用 ``inplace`` 操作减少中间张量

数值稳定性
~~~~~~~~~~

所有数值计算都通过 ``NumericalConfig`` 进行epsilon保护：

.. code-block:: python

   from src.config import NumericalConfig
   _NUM_CFG = NumericalConfig()

   # 距离计算
   dist = torch.sqrt(dist2 + _NUM_CFG.eps_distance)

   # 除法运算
   result = numerator / (denominator + _NUM_CFG.eps_division)
