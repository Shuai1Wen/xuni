# -*- coding: utf-8 -*-
"""
Operator模块单元测试

测试内容：
1. OperatorModel的前向传播
2. 低秩分解结构 A_θ = A_t^(0) + Σ_k α_k(θ) B_k
3. 谱范数正则化
4. 梯度流动
5. 数值稳定性
"""

import torch
import pytest
from src.models.operator import OperatorModel
from src.config import NumericalConfig

_NUM_CFG = NumericalConfig()


class TestOperatorModel:
    """测试算子模型基本功能"""

    def test_初始化_参数形状(self):
        """测试模型初始化后参数形状正确"""
        model = OperatorModel(
            latent_dim=32,
            n_tissues=3,
            n_response_bases=4,
            cond_dim=64
        )

        assert model.A0_tissue.shape == (3, 32, 32), "基线算子形状错误"
        assert model.B.shape == (4, 32, 32), "响应基形状错误"
        assert model.u.shape == (4, 32), "偏置基形状错误"

    def test_前向传播_输出形状(self):
        """测试前向传播输出形状正确"""
        model = OperatorModel(
            latent_dim=32,
            n_tissues=3,
            n_response_bases=4,
            cond_dim=64
        )

        z = torch.randn(16, 32)
        tissue_idx = torch.randint(0, 3, (16,))
        cond_vec = torch.randn(16, 64)

        z_out, A_theta, b_theta = model(z, tissue_idx, cond_vec)

        assert z_out.shape == (16, 32), "输出潜变量形状错误"
        assert A_theta.shape == (16, 32, 32), "算子矩阵形状错误"
        assert b_theta.shape == (16, 32), "偏置向量形状错误"

    def test_前向传播_数学形式(self):
        """测试前向传播符合数学定义：z_out = A_θ @ z + b_θ"""
        model = OperatorModel(
            latent_dim=16,
            n_tissues=2,
            n_response_bases=3,
            cond_dim=32
        )

        z = torch.randn(8, 16)
        tissue_idx = torch.zeros(8, dtype=torch.long)
        cond_vec = torch.randn(8, 32)

        z_out, A_theta, b_theta = model(z, tissue_idx, cond_vec)

        # 手动计算：z_out_manual = A_theta @ z + b_theta
        z_out_manual = torch.bmm(
            A_theta,
            z.unsqueeze(-1)
        ).squeeze(-1) + b_theta

        diff = (z_out - z_out_manual).abs().max()
        assert diff < 1e-5, f"前向传播不符合数学定义，差异: {diff}"

    def test_低秩分解_结构(self):
        """测试低秩分解 A_θ = A_t^(0) + Σ_k α_k(θ) B_k"""
        model = OperatorModel(
            latent_dim=16,
            n_tissues=2,
            n_response_bases=3,
            cond_dim=32
        )

        z = torch.randn(8, 16)
        tissue_idx = torch.ones(8, dtype=torch.long)  # tissue 1
        cond_vec = torch.randn(8, 32)

        # 获取alpha和beta系数
        alpha, beta = model.condition_to_coefficients(cond_vec)
        assert alpha.shape == (8, 3), "alpha系数形状错误"
        assert beta.shape == (8, 3), "beta系数形状错误"

        # 验证A_theta的分解
        z_out, A_theta, b_theta = model(z, tissue_idx, cond_vec)

        # 手动构建 A_theta = A_t^(0) + Σ_k α_k B_k
        A0 = model.A0_tissue[tissue_idx]  # (8, 16, 16)
        A_res = torch.einsum('bk,kij->bij', alpha, model.B)  # (8, 16, 16)
        A_theta_manual = A0 + A_res

        diff = (A_theta - A_theta_manual).abs().max()
        assert diff < 1e-5, f"低秩分解结构不正确，差异: {diff}"

    def test_不同组织_不同基线(self):
        """测试不同组织使用不同基线算子"""
        model = OperatorModel(
            latent_dim=16,
            n_tissues=3,
            n_response_bases=2,
            cond_dim=32
        )

        z = torch.randn(6, 16)
        tissue_idx = torch.tensor([0, 0, 1, 1, 2, 2])
        cond_vec = torch.zeros(6, 32)  # 零条件，只看基线

        _, A_theta, _ = model(z, tissue_idx, cond_vec)

        # 相同组织的算子应相同
        assert torch.allclose(A_theta[0], A_theta[1], atol=1e-5)
        assert torch.allclose(A_theta[2], A_theta[3], atol=1e-5)
        assert torch.allclose(A_theta[4], A_theta[5], atol=1e-5)

        # 不同组织的算子应不同
        assert not torch.allclose(A_theta[0], A_theta[2], atol=1e-3)
        assert not torch.allclose(A_theta[0], A_theta[4], atol=1e-3)

    def test_条件向量_影响(self):
        """测试条件向量对算子的影响"""
        model = OperatorModel(
            latent_dim=16,
            n_tissues=2,
            n_response_bases=3,
            cond_dim=32
        )

        z = torch.randn(2, 16)
        tissue_idx = torch.zeros(2, dtype=torch.long)

        # 两个不同的条件
        cond_vec1 = torch.randn(1, 32).expand(2, -1)
        cond_vec2 = torch.randn(1, 32).expand(2, -1)

        _, A_theta1, _ = model(z, tissue_idx, cond_vec1)
        _, A_theta2, _ = model(z, tissue_idx, cond_vec2)

        # 不同条件应产生不同算子
        diff = (A_theta1 - A_theta2).abs().max()
        assert diff > 1e-3, "不同条件应产生明显不同的算子"


class TestSpectralPenalty:
    """测试谱范数正则化"""

    def test_谱范数_计算(self):
        """测试谱范数能够计算"""
        model = OperatorModel(
            latent_dim=16,
            n_tissues=2,
            n_response_bases=3,
            cond_dim=32,
            max_spectral_norm=1.05
        )

        penalty = model.spectral_penalty(max_allowed=1.0)

        assert isinstance(penalty, torch.Tensor)
        assert penalty.shape == ()  # 标量
        assert penalty.item() >= 0, "谱范数惩罚应非负"
        assert not torch.isnan(penalty), "谱范数不应为NaN"

    def test_谱范数_阈值效果(self):
        """测试不同阈值的效果"""
        model = OperatorModel(
            latent_dim=16,
            n_tissues=2,
            n_response_bases=3,
            cond_dim=32
        )

        # 宽松阈值
        penalty_loose = model.spectral_penalty(max_allowed=10.0)
        # 严格阈值
        penalty_strict = model.spectral_penalty(max_allowed=0.5)

        # 严格阈值通常会产生更大的惩罚
        assert penalty_strict.item() >= penalty_loose.item(), \
            "严格阈值应产生更大惩罚"

    def test_幂迭代_收敛(self):
        """测试幂迭代法能够收敛"""
        model = OperatorModel(
            latent_dim=32,
            n_tissues=3,
            n_response_bases=4,
            cond_dim=64
        )

        # 计算多次应得到稳定结果
        penalty1 = model.spectral_penalty(n_iterations=10)
        penalty2 = model.spectral_penalty(n_iterations=20)
        penalty3 = model.spectral_penalty(n_iterations=30)

        # 增加迭代次数，结果应趋于稳定
        diff_12 = (penalty1 - penalty2).abs()
        diff_23 = (penalty2 - penalty3).abs()

        assert diff_23 < diff_12 or diff_23 < 0.1, \
            "更多迭代应使结果更稳定"

    def test_梯度流动_P1修复验证(self):
        """测试谱范数惩罚的梯度流动（P1修复验证）"""
        model = OperatorModel(
            latent_dim=16,
            n_tissues=2,
            n_response_bases=3,
            cond_dim=32
        )

        # 清空梯度
        model.zero_grad()

        penalty = model.spectral_penalty(max_allowed=1.0)
        penalty.backward()

        # 检查基线算子有梯度
        assert model.A0_tissue.grad is not None, \
            "谱范数应对A0产生梯度（P1修复）"
        assert not torch.isnan(model.A0_tissue.grad).any(), \
            "梯度不应有NaN"


class TestComputeOperatorNorm:
    """测试算子范数计算"""

    def test_范数_计算(self):
        """测试能够计算算子范数"""
        model = OperatorModel(
            latent_dim=16,
            n_tissues=2,
            n_response_bases=3,
            cond_dim=32
        )

        z = torch.randn(8, 16)
        tissue_idx = torch.zeros(8, dtype=torch.long)
        cond_vec = torch.randn(8, 32)

        _, A_theta, _ = model(z, tissue_idx, cond_vec)

        norms = model.compute_operator_norm(A_theta, n_iterations=20)

        assert norms.shape == (8,), "应返回每个样本的范数"
        assert (norms >= 0).all(), "范数应非负"
        assert not torch.isnan(norms).any(), "范数不应有NaN"

    def test_单位矩阵_范数(self):
        """测试单位矩阵的谱范数接近1"""
        model = OperatorModel(
            latent_dim=16,
            n_tissues=2,
            n_response_bases=3,
            cond_dim=32
        )

        # 构造单位矩阵
        I = torch.eye(16).unsqueeze(0).expand(4, -1, -1)

        norms = model.compute_operator_norm(I, n_iterations=50)

        # 单位矩阵的谱范数应为1
        assert torch.allclose(norms, torch.ones_like(norms), atol=0.01), \
            "单位矩阵的谱范数应接近1"

    def test_零矩阵_范数(self):
        """测试零矩阵的谱范数接近0"""
        model = OperatorModel(
            latent_dim=16,
            n_tissues=2,
            n_response_bases=3,
            cond_dim=32
        )

        # 零矩阵
        Z = torch.zeros(4, 16, 16)

        norms = model.compute_operator_norm(Z, n_iterations=20)

        # 零矩阵的谱范数应为0
        assert (norms < 0.01).all(), "零矩阵的谱范数应接近0"


class TestNumericalStability:
    """测试数值稳定性"""

    def test_极端条件向量(self):
        """测试极端条件向量下的稳定性"""
        model = OperatorModel(
            latent_dim=16,
            n_tissues=2,
            n_response_bases=3,
            cond_dim=32
        )

        z = torch.randn(8, 16)
        tissue_idx = torch.zeros(8, dtype=torch.long)

        # 极大条件向量
        cond_large = torch.randn(8, 32) * 100

        z_out, A_theta, b_theta = model(z, tissue_idx, cond_large)

        assert not torch.isnan(z_out).any(), "极大条件不应产生NaN"
        assert not torch.isinf(z_out).any(), "极大条件不应产生Inf"

    def test_极端潜变量(self):
        """测试极端潜变量输入"""
        model = OperatorModel(
            latent_dim=16,
            n_tissues=2,
            n_response_bases=3,
            cond_dim=32
        )

        # 极大潜变量
        z_large = torch.randn(8, 16) * 100
        tissue_idx = torch.zeros(8, dtype=torch.long)
        cond_vec = torch.randn(8, 32)

        z_out, _, _ = model(z_large, tissue_idx, cond_vec)

        assert not torch.isnan(z_out).any(), "极大潜变量不应产生NaN"

    def test_批次大小_一致性(self):
        """测试不同批次大小的一致性"""
        model = OperatorModel(
            latent_dim=16,
            n_tissues=2,
            n_response_bases=3,
            cond_dim=32
        )

        z = torch.randn(16, 16)
        tissue_idx = torch.zeros(16, dtype=torch.long)
        cond_vec = torch.randn(16, 32)

        # 完整批次
        z_out_full, _, _ = model(z, tissue_idx, cond_vec)

        # 分批计算
        z_out_batch1, _, _ = model(z[:8], tissue_idx[:8], cond_vec[:8])
        z_out_batch2, _, _ = model(z[8:], tissue_idx[8:], cond_vec[8:])
        z_out_split = torch.cat([z_out_batch1, z_out_batch2], dim=0)

        # 应该得到相同结果
        assert torch.allclose(z_out_full, z_out_split, atol=1e-5), \
            "不同批次大小应得到相同结果"


class TestGradientFlow:
    """测试梯度流动"""

    def test_端到端_梯度(self):
        """测试端到端梯度流动"""
        model = OperatorModel(
            latent_dim=16,
            n_tissues=2,
            n_response_bases=3,
            cond_dim=32
        )

        z = torch.randn(8, 16, requires_grad=True)
        tissue_idx = torch.zeros(8, dtype=torch.long)
        cond_vec = torch.randn(8, 32, requires_grad=True)

        z_out, _, _ = model(z, tissue_idx, cond_vec)
        loss = z_out.sum()
        loss.backward()

        assert z.grad is not None, "z应有梯度"
        assert cond_vec.grad is not None, "cond_vec应有梯度"
        assert not torch.isnan(z.grad).any(), "梯度不应有NaN"

    def test_参数_梯度(self):
        """测试模型参数的梯度"""
        model = OperatorModel(
            latent_dim=16,
            n_tissues=2,
            n_response_bases=3,
            cond_dim=32
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        z = torch.randn(8, 16)
        tissue_idx = torch.zeros(8, dtype=torch.long)
        cond_vec = torch.randn(8, 32)

        z_out, _, _ = model(z, tissue_idx, cond_vec)
        loss = z_out.pow(2).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 检查参数有梯度
        assert model.A0_tissue.grad is not None
        assert model.B.grad is not None
        assert not torch.isnan(model.A0_tissue.grad).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
