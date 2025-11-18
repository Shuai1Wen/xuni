# -*- coding: utf-8 -*-
"""
E-distance模块单元测试

测试内容：
1. pairwise_distances的正确性
2. energy_distance的数学性质
3. energy_distance_batched与标准版本的等价性
4. 梯度流动正确性
"""

import torch
import pytest
from src.utils.edistance import (
    pairwise_distances,
    energy_distance,
    energy_distance_batched,
    check_edistance_properties
)
from src.config import NumericalConfig

_NUM_CFG = NumericalConfig()


class TestPairwiseDistances:
    """测试成对距离计算"""

    def test_正常情况_两组不同样本(self):
        """测试两组不同样本的距离计算"""
        x = torch.randn(50, 10)
        y = torch.randn(30, 10)

        dist = pairwise_distances(x, y)

        assert dist.shape == (50, 30), "输出形状应为 (n, m)"
        assert (dist >= 0).all(), "所有距离应非负"
        assert not torch.isnan(dist).any(), "不应包含NaN"

    def test_边界情况_相同样本(self):
        """测试相同样本的距离应为0"""
        x = torch.randn(20, 10)

        dist = pairwise_distances(x, x)

        # 对角线应接近0
        diag = torch.diag(dist)
        assert (diag < 1e-6).all(), "自身距离应接近0"

    def test_边界情况_单样本(self):
        """测试单样本情况"""
        x = torch.randn(1, 10)
        y = torch.randn(1, 10)

        dist = pairwise_distances(x, y)

        assert dist.shape == (1, 1)
        assert dist.item() >= 0

    def test_数值稳定性_零向量(self):
        """测试零向量的数值稳定性"""
        x = torch.zeros(5, 10)
        y = torch.zeros(5, 10)

        dist = pairwise_distances(x, y)

        assert not torch.isnan(dist).any(), "零向量不应产生NaN"
        assert (dist < 1e-6).all(), "零向量距离应接近0"

    def test_梯度流动(self):
        """测试梯度能够正确反向传播"""
        x = torch.randn(10, 5, requires_grad=True)
        y = torch.randn(8, 5, requires_grad=True)

        dist = pairwise_distances(x, y)
        loss = dist.sum()
        loss.backward()

        assert x.grad is not None, "x应有梯度"
        assert y.grad is not None, "y应有梯度"
        assert not torch.isnan(x.grad).any(), "x梯度不应有NaN"
        assert not torch.isnan(y.grad).any(), "y梯度不应有NaN"


class TestEnergyDistance:
    """测试能量距离计算"""

    def test_正常情况_两组不同分布(self):
        """测试两组明显不同分布的E-distance"""
        x = torch.randn(100, 10)
        y = torch.randn(100, 10) + 2.0  # 平移分布

        ed2 = energy_distance(x, y)

        assert ed2.item() > 0, "不同分布的E-distance应大于0"
        assert not torch.isnan(ed2), "不应产生NaN"

    def test_数学性质_非负性(self):
        """测试E-distance的非负性"""
        x = torch.randn(50, 8)
        y = torch.randn(50, 8)

        ed2 = energy_distance(x, y)

        assert ed2.item() >= -_NUM_CFG.tol_test, "E-distance应非负"

    def test_数学性质_同一性(self):
        """测试相同分布的E-distance应接近0"""
        x = torch.randn(100, 10)

        ed2 = energy_distance(x, x)

        assert ed2.item() < _NUM_CFG.tol_test, "相同分布的E-distance应接近0"

    def test_数学性质_对称性(self):
        """测试E-distance的对称性"""
        x = torch.randn(50, 10)
        y = torch.randn(60, 10)

        ed2_xy = energy_distance(x, y)
        ed2_yx = energy_distance(y, x)

        assert torch.abs(ed2_xy - ed2_yx) < _NUM_CFG.tol_test, "E-distance应对称"

    def test_边界情况_空集(self):
        """测试空集的处理"""
        x = torch.randn(0, 10)
        y = torch.randn(50, 10)

        ed2 = energy_distance(x, y)

        assert ed2.item() == 0, "空集的E-distance应为0"

    def test_边界情况_单样本(self):
        """测试单样本情况"""
        x = torch.randn(1, 10)
        y = torch.randn(1, 10)

        ed2 = energy_distance(x, y)

        assert not torch.isnan(ed2), "单样本不应产生NaN"

    def test_梯度流动(self):
        """测试E-distance的梯度流动"""
        x = torch.randn(30, 5, requires_grad=True)
        y = torch.randn(30, 5, requires_grad=True)

        ed2 = energy_distance(x, y)
        ed2.backward()

        assert x.grad is not None, "x应有梯度"
        assert y.grad is not None, "y应有梯度"
        assert not torch.isnan(x.grad).any(), "梯度不应有NaN"


class TestEnergyDistanceBatched:
    """测试分块E-distance计算"""

    def test_等价性_与标准版本对比(self):
        """测试分块版本与标准版本的等价性"""
        x = torch.randn(200, 10)
        y = torch.randn(150, 10)

        ed2_standard = energy_distance(x, y)
        ed2_batched = energy_distance_batched(x, y, batch_size=50)

        diff = torch.abs(ed2_standard - ed2_batched)
        assert diff < 1e-4, f"分块版本与标准版本差异过大: {diff.item()}"

    def test_不同batch_size的一致性(self):
        """测试不同batch_size得到相同结果"""
        x = torch.randn(100, 8)
        y = torch.randn(100, 8)

        ed2_bs20 = energy_distance_batched(x, y, batch_size=20)
        ed2_bs50 = energy_distance_batched(x, y, batch_size=50)

        diff = torch.abs(ed2_bs20 - ed2_bs50)
        assert diff < 1e-5, "不同batch_size应得到相同结果"

    def test_边界情况_batch_size大于样本数(self):
        """测试batch_size大于样本数的情况"""
        x = torch.randn(30, 5)
        y = torch.randn(20, 5)

        ed2_batched = energy_distance_batched(x, y, batch_size=100)
        ed2_standard = energy_distance(x, y)

        diff = torch.abs(ed2_batched - ed2_standard)
        assert diff < 1e-5, "batch_size过大不应影响结果"

    def test_梯度流动_修复验证(self):
        """测试修复后的梯度流动（P3-3修复验证）"""
        x = torch.randn(50, 5, requires_grad=True)
        y = torch.randn(50, 5, requires_grad=True)

        ed2 = energy_distance_batched(x, y, batch_size=20)
        ed2.backward()

        assert x.grad is not None, "分块版本应有梯度（修复后）"
        assert y.grad is not None, "分块版本应有梯度（修复后）"
        assert not torch.isnan(x.grad).any(), "梯度不应有NaN"


class TestCheckProperties:
    """测试E-distance性质检查函数"""

    def test_完整性质检查(self):
        """测试所有性质检查"""
        x = torch.randn(100, 10)
        y = torch.randn(100, 10)
        z = torch.randn(100, 10)

        results = check_edistance_properties(x, y, z)

        assert results["non_negative"], "应满足非负性"
        assert results["symmetric"], "应满足对称性"
        assert results["identity"], "应满足同一性"
        assert results["triangle"] is not None, "应检查三角不等式"

    def test_无第三组样本(self):
        """测试只有两组样本的情况"""
        x = torch.randn(50, 8)
        y = torch.randn(50, 8)

        results = check_edistance_properties(x, y, z=None)

        assert "non_negative" in results
        assert "symmetric" in results
        assert "identity" in results
        assert results["triangle"] is None, "无第三组样本时三角不等式应为None"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
