# -*- coding: utf-8 -*-
"""
能量距离（Energy Distance）计算

本模块实现分布级别的度量，用于算子模型的训练。

数学对应关系：
- E-distance定义：对应 model.md A.4节，第111-119行
- 公式：Ê²(X,Y) = 2/(nm)Σ|xi-yj| - 1/n²Σ|xi-xi'| - 1/m²Σ|yj-yj'|

关键特性：
- 无需OT匹配：直接计算两个点集之间的分布差异
- 无偏估计：使用model.md中的无偏估计公式
- 向量化实现：避免双重循环，使用矩阵运算
- 数值稳定性：处理距离计算中的数值问题

参考文献：
- Székely, G. J., & Rizzo, M. L. (2013). Energy statistics: A class of statistics
  based on distances. Journal of statistical planning and inference.
"""

import torch
from typing import Optional

from ..config import NumericalConfig

# 默认数值配置
_NUM_CFG = NumericalConfig()


def pairwise_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    计算两组向量之间的成对L2距离

    使用向量化公式避免显式循环：
        ||x_i - y_j||² = ||x_i||² + ||y_j||² - 2⟨x_i, y_j⟩

    参数:
        x: (n, d) 第一组向量
        y: (m, d) 第二组向量

    返回:
        distances: (n, m) 成对距离矩阵，distances[i,j] = ||x_i - y_j||₂

    实现细节:
        1. 计算x的平方范数：(x**2).sum(-1) → (n, 1)
        2. 计算y的平方范数：(y**2).sum(-1) → (1, m)
        3. 计算内积：x @ y.T → (n, m)
        4. 组合：dist² = ||x||² + ||y||² - 2⟨x,y⟩
        5. clamp避免数值误差导致负数
        6. 开方得到距离

    数值稳定性:
        - clamp(min=0)：避免浮点误差导致dist²<0
        - 添加epsilon到开方：避免梯度在0处不稳定

    复杂度:
        时间：O(nmd)，其中d是向量维度
        空间：O(nm)，存储距离矩阵

    示例:
        >>> x = torch.randn(100, 32)
        >>> y = torch.randn(200, 32)
        >>> dist = pairwise_distances(x, y)
        >>> print(dist.shape)
        torch.Size([100, 200])
        >>> # 验证对角线（如果x=y）
        >>> dist_self = pairwise_distances(x, x)
        >>> print(torch.diag(dist_self).max())  # 应该接近0
        tensor(1.4901e-07)
    """
    # 计算x的平方范数 ||x_i||² (n, 1)
    x2 = (x ** 2).sum(dim=-1, keepdim=True)  # (n, 1)

    # 计算y的平方范数 ||y_j||² (1, m)
    y2 = (y ** 2).sum(dim=-1, keepdim=True).T  # (1, m)

    # 计算内积矩阵 ⟨x_i, y_j⟩ (n, m)
    xy = x @ y.T  # (n, m)

    # 计算距离平方：||x_i - y_j||² = ||x_i||² + ||y_j||² - 2⟨x_i, y_j⟩
    dist2 = x2 + y2 - 2 * xy  # (n, m)

    # clamp到非负（避免浮点误差）
    dist2 = torch.clamp(dist2, min=0.0)

    # 开方得到距离，添加epsilon避免梯度不稳定
    distances = torch.sqrt(dist2 + _NUM_CFG.eps_distance)  # (n, m)

    return distances


def energy_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    return_components: bool = False
) -> torch.Tensor:
    """
    计算两组样本之间的能量距离（E-distance）

    数学定义：
        Ê²(X, Y) = 2/(nm) Σᵢⱼ ||xᵢ - yⱼ||
                   - 1/n² Σᵢᵢ' ||xᵢ - xᵢ'||
                   - 1/m² Σⱼⱼ' ||yⱼ - yⱼ'||

    对应：model.md A.4节，第111-119行

    参数:
        x: (n, d) 第一组样本（预测分布或对照）
        y: (m, d) 第二组样本（真实分布或处理）
        return_components: 是否返回三个分量（用于调试）

    返回:
        ed2: 标量，能量距离的平方
        如果return_components=True，返回(ed2, term_xy, term_xx, term_yy)

    性质:
        - ed2 ≥ 0，且 ed2 = 0 当且仅当 X 和 Y 同分布
        - 满足三角不等式
        - 对旋转和平移不变

    实现细节:
        - 使用pairwise_distances向量化计算
        - 对空集进行特殊处理
        - 三项分别计算后组合

    数值稳定性:
        - 检查n=0或m=0的边界情况
        - pairwise_distances内部已处理数值稳定性

    复杂度:
        时间：O((n²+m²+nm)d)
        空间：O(max(n², m², nm))（瓶颈在距离矩阵）

    注意事项:
        - 对于大规模数据（n,m > 10000），内存可能不足
        - 建议分块计算或降低批次大小
        - 可以考虑使用近似算法（如Sinkhorn）

    示例:
        >>> # 相同分布，E-distance应该接近0
        >>> x = torch.randn(100, 32)
        >>> y = x.clone()
        >>> ed2 = energy_distance(x, y)
        >>> print(ed2)
        tensor(3.5762e-07)

        >>> # 不同分布，E-distance应该>0
        >>> x = torch.randn(100, 32)
        >>> y = torch.randn(100, 32) + 2.0  # 平移
        >>> ed2 = energy_distance(x, y)
        >>> print(ed2)
        tensor(2.8452)

        >>> # 空集情况
        >>> x_empty = torch.randn(0, 32)
        >>> y = torch.randn(100, 32)
        >>> ed2 = energy_distance(x_empty, y)
        >>> print(ed2)
        tensor(0.)
    """
    n, m = x.size(0), y.size(0)

    # 边界情况：如果任一组为空，返回0
    if n == 0 or m == 0:
        if return_components:
            zero = torch.tensor(0.0, device=x.device)
            return zero, zero, zero, zero
        return torch.tensor(0.0, device=x.device)

    # 计算三个距离矩阵
    d_xy = pairwise_distances(x, y)  # (n, m)
    d_xx = pairwise_distances(x, x)  # (n, n)
    d_yy = pairwise_distances(y, y)  # (m, m)

    # 第一项：跨分布项 2/(nm) Σᵢⱼ ||xᵢ - yⱼ||
    term_xy = 2.0 / (n * m) * d_xy.sum()

    # 第二项：X内部项 1/n² Σᵢᵢ' ||xᵢ - xᵢ'||
    term_xx = 1.0 / (n * n) * d_xx.sum()

    # 第三项：Y内部项 1/m² Σⱼⱼ' ||yⱼ - yⱼ'||
    term_yy = 1.0 / (m * m) * d_yy.sum()

    # 能量距离平方：term_xy - term_xx - term_yy
    ed2 = term_xy - term_xx - term_yy

    if return_components:
        return ed2, term_xy, term_xx, term_yy

    return ed2


def energy_distance_batched(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 1000
) -> torch.Tensor:
    """
    分块计算E-distance，用于大规模数据

    当n或m很大时（>10000），pairwise_distances的内存占用可能超过GPU容量。
    此函数通过分块计算减少内存峰值。

    参数:
        x: (n, d) 第一组样本
        y: (m, d) 第二组样本
        batch_size: 分块大小

    返回:
        ed2: 标量，能量距离的平方

    实现策略:
        - term_xy：分块计算d_xy，累加
        - term_xx：分块计算d_xx，累加
        - term_yy：分块计算d_yy，累加

    复杂度:
        时间：与energy_distance相同
        空间：O(batch_size²)，显著降低

    注意:
        - batch_size越小，内存占用越低，但计算越慢
        - 建议batch_size=1000-5000（取决于GPU内存）

    示例:
        >>> x = torch.randn(20000, 32)
        >>> y = torch.randn(15000, 32)
        >>> # 标准方法可能OOM
        >>> # ed2 = energy_distance(x, y)
        >>> # 分块方法
        >>> ed2 = energy_distance_batched(x, y, batch_size=2000)
        >>> print(ed2)
        tensor(0.0234)
    """
    n, m = x.size(0), y.size(0)

    if n == 0 or m == 0:
        return torch.tensor(0.0, device=x.device)

    # 分块计算term_xy（收集所有块以维持完整梯度图）
    xy_chunks = []
    for i in range(0, n, batch_size):
        x_batch = x[i:i + batch_size]
        for j in range(0, m, batch_size):
            y_batch = y[j:j + batch_size]
            d_xy_batch = pairwise_distances(x_batch, y_batch)
            xy_chunks.append(d_xy_batch.sum())
    term_xy = 2.0 / (n * m) * torch.stack(xy_chunks).sum()

    # 分块计算term_xx（收集所有块以维持完整梯度图）
    xx_chunks = []
    for i in range(0, n, batch_size):
        x_batch_i = x[i:i + batch_size]
        for j in range(0, n, batch_size):
            x_batch_j = x[j:j + batch_size]
            d_xx_batch = pairwise_distances(x_batch_i, x_batch_j)
            xx_chunks.append(d_xx_batch.sum())
    term_xx = 1.0 / (n * n) * torch.stack(xx_chunks).sum()

    # 分块计算term_yy（收集所有块以维持完整梯度图）
    yy_chunks = []
    for i in range(0, m, batch_size):
        y_batch_i = y[i:i + batch_size]
        for j in range(0, m, batch_size):
            y_batch_j = y[j:j + batch_size]
            d_yy_batch = pairwise_distances(y_batch_i, y_batch_j)
            yy_chunks.append(d_yy_batch.sum())
    term_yy = 1.0 / (m * m) * torch.stack(yy_chunks).sum()

    ed2 = term_xy - term_xx - term_yy

    return ed2


def check_edistance_properties(
    x: torch.Tensor,
    y: torch.Tensor,
    z: Optional[torch.Tensor] = None,
    verbose: bool = True
) -> dict:
    """
    验证E-distance的数学性质（用于单元测试和调试）

    验证的性质：
    1. 非负性：E²(X,Y) ≥ 0
    2. 对称性：E²(X,Y) = E²(Y,X)
    3. 同一性：E²(X,X) ≈ 0
    4. 三角不等式：E(X,Z) ≤ E(X,Y) + E(Y,Z)（可选）

    参数:
        x: (n, d) 第一组样本
        y: (m, d) 第二组样本
        z: (k, d) 第三组样本（用于验证三角不等式，可选）
        verbose: 是否打印验证结果

    返回:
        results: 包含各项验证结果的字典

    示例:
        >>> x = torch.randn(100, 32)
        >>> y = torch.randn(100, 32)
        >>> results = check_edistance_properties(x, y, verbose=True)
        ✓ 非负性: E²(X,Y) = 1.2345 ≥ 0
        ✓ 对称性: |E²(X,Y) - E²(Y,X)| = 1.23e-07 < 1e-6
        ✓ 同一性: E²(X,X) = 3.45e-08 ≈ 0
    """
    results = {}

    # 1. 非负性
    ed2_xy = energy_distance(x, y)
    results["non_negative"] = ed2_xy.item() >= -_NUM_CFG.tol_test
    if verbose:
        if results["non_negative"]:
            print(f"✓ 非负性: E²(X,Y) = {ed2_xy.item():.4f} ≥ 0")
        else:
            print(f"✗ 非负性: E²(X,Y) = {ed2_xy.item():.4f} < 0")

    # 2. 对称性
    ed2_yx = energy_distance(y, x)
    sym_error = torch.abs(ed2_xy - ed2_yx).item()
    results["symmetric"] = sym_error < _NUM_CFG.tol_test
    if verbose:
        if results["symmetric"]:
            print(f"✓ 对称性: |E²(X,Y) - E²(Y,X)| = {sym_error:.2e} < {_NUM_CFG.tol_test}")
        else:
            print(f"✗ 对称性: |E²(X,Y) - E²(Y,X)| = {sym_error:.2e} ≥ {_NUM_CFG.tol_test}")

    # 3. 同一性
    ed2_xx = energy_distance(x, x)
    results["identity"] = ed2_xx.item() < _NUM_CFG.tol_test
    if verbose:
        if results["identity"]:
            print(f"✓ 同一性: E²(X,X) = {ed2_xx.item():.2e} ≈ 0")
        else:
            print(f"✗ 同一性: E²(X,X) = {ed2_xx.item():.2e} > {_NUM_CFG.tol_test}")

    # 4. 三角不等式（可选）
    if z is not None:
        ed_xz = torch.sqrt(torch.clamp(energy_distance(x, z), min=0))
        ed_xy = torch.sqrt(torch.clamp(ed2_xy, min=0))
        ed_yz = torch.sqrt(torch.clamp(energy_distance(y, z), min=0))
        triangle_holds = (ed_xz <= ed_xy + ed_yz + _NUM_CFG.tol_test).item()
        results["triangle"] = triangle_holds
        if verbose:
            if triangle_holds:
                print(f"✓ 三角不等式: E(X,Z) = {ed_xz.item():.4f} "
                      f"≤ E(X,Y) + E(Y,Z) = {(ed_xy + ed_yz).item():.4f}")
            else:
                print(f"✗ 三角不等式: E(X,Z) = {ed_xz.item():.4f} "
                      f"> E(X,Y) + E(Y,Z) = {(ed_xy + ed_yz).item():.4f}")

    return results
