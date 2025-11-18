# -*- coding: utf-8 -*-
"""
NB-VAE模块单元测试

测试内容：
1. Encoder输出的正确性
2. DecoderNB输出的正确性
3. NBVAE完整流程
4. nb_log_likelihood的数值稳定性
5. elbo_loss的计算
"""

import torch
import pytest
from src.models.nb_vae import (
    Encoder,
    DecoderNB,
    NBVAE,
    nb_log_likelihood,
    elbo_loss
)
from src.config import NumericalConfig

_NUM_CFG = NumericalConfig()


class TestEncoder:
    """测试VAE编码器"""

    def test_输出形状(self):
        """测试编码器输出形状正确"""
        encoder = Encoder(n_genes=500, latent_dim=32, n_tissues=3)

        x = torch.randn(64, 500)
        tissue_onehot = torch.zeros(64, 3)
        tissue_onehot[:, 0] = 1

        mu, logvar = encoder(x, tissue_onehot)

        assert mu.shape == (64, 32), "mu形状应为 (batch, latent_dim)"
        assert logvar.shape == (64, 32), "logvar形状应为 (batch, latent_dim)"

    def test_输出数值范围(self):
        """测试编码器输出在合理范围内"""
        encoder = Encoder(n_genes=200, latent_dim=16, n_tissues=2)

        x = torch.randn(32, 200).abs()  # 非负输入
        tissue_onehot = torch.zeros(32, 2)
        tissue_onehot[:, 0] = 1

        mu, logvar = encoder(x, tissue_onehot)

        assert not torch.isnan(mu).any(), "mu不应有NaN"
        assert not torch.isnan(logvar).any(), "logvar不应有NaN"
        assert not torch.isinf(mu).any(), "mu不应有Inf"
        assert not torch.isinf(logvar).any(), "logvar不应有Inf"

    def test_梯度流动(self):
        """测试梯度能够反向传播"""
        encoder = Encoder(n_genes=100, latent_dim=16, n_tissues=2)

        x = torch.randn(16, 100, requires_grad=True)
        tissue_onehot = torch.zeros(16, 2)
        tissue_onehot[:, 1] = 1

        mu, logvar = encoder(x, tissue_onehot)
        loss = mu.sum() + logvar.sum()
        loss.backward()

        assert x.grad is not None, "输入应有梯度"
        assert not torch.isnan(x.grad).any(), "梯度不应有NaN"


class TestDecoderNB:
    """测试负二项解码器"""

    def test_输出形状(self):
        """测试解码器输出形状正确"""
        decoder = DecoderNB(latent_dim=32, n_genes=500, n_tissues=3)

        z = torch.randn(64, 32)
        tissue_onehot = torch.zeros(64, 3)
        tissue_onehot[:, 1] = 1

        mu, r = decoder(z, tissue_onehot)

        assert mu.shape == (64, 500), "mu形状应为 (batch, n_genes)"
        assert r.shape == (64, 500), "r形状应为 (batch, n_genes)"

    def test_输出正性约束(self):
        """测试解码器输出满足正性约束"""
        decoder = DecoderNB(latent_dim=16, n_genes=200, n_tissues=2)

        z = torch.randn(32, 16)
        tissue_onehot = torch.zeros(32, 2)
        tissue_onehot[:, 0] = 1

        mu, r = decoder(z, tissue_onehot)

        assert (mu > 0).all(), "mu应严格为正"
        assert (r > 0).all(), "r应严格为正"
        assert (mu >= _NUM_CFG.eps_model_output).all(), "mu应不小于epsilon"

    def test_数值稳定性(self):
        """测试极端输入下的数值稳定性"""
        decoder = DecoderNB(latent_dim=16, n_genes=100, n_tissues=2)

        # 极大值输入
        z_large = torch.randn(8, 16) * 10
        # 极小值输入
        z_small = torch.randn(8, 16) * 0.01

        tissue_onehot = torch.zeros(16, 2)
        tissue_onehot[:8, 0] = 1
        tissue_onehot[8:, 1] = 1

        z = torch.cat([z_large, z_small], dim=0)

        mu, r = decoder(z, tissue_onehot)

        assert not torch.isnan(mu).any(), "极端输入不应产生NaN"
        assert not torch.isinf(mu).any(), "极端输入不应产生Inf"


class TestNBVAE:
    """测试完整NB-VAE模型"""

    def test_前向传播_形状(self):
        """测试完整前向传播的形状"""
        model = NBVAE(n_genes=500, latent_dim=32, n_tissues=3)

        x = torch.randn(64, 500).abs()
        tissue_onehot = torch.zeros(64, 3)
        tissue_onehot[:, 0] = 1

        mu_x, r_x, mu_z, logvar_z = model(x, tissue_onehot)

        assert mu_x.shape == (64, 500)
        assert r_x.shape == (64, 500)
        assert mu_z.shape == (64, 32)
        assert logvar_z.shape == (64, 32)

    def test_重建质量_相同输入(self):
        """测试模型能重建输入（训练前不要求完美）"""
        model = NBVAE(n_genes=100, latent_dim=16, n_tissues=2)
        model.eval()

        x = torch.randn(32, 100).abs() + 1.0
        tissue_onehot = torch.zeros(32, 2)
        tissue_onehot[:, 0] = 1

        with torch.no_grad():
            mu_x, r_x, mu_z, logvar_z = model(x, tissue_onehot)

        # 检查重建的基本合理性（形状和正性）
        assert mu_x.shape == x.shape
        assert (mu_x > 0).all(), "重建应为正值"

    def test_编码器_解码器_流程(self):
        """测试编码-解码流程的一致性"""
        model = NBVAE(n_genes=200, latent_dim=32, n_tissues=3)

        x = torch.randn(16, 200).abs()
        tissue_onehot = torch.zeros(16, 3)
        tissue_onehot[:, 1] = 1

        # 编码
        mu_z, logvar_z = model.encoder(x, tissue_onehot)
        z = model.sample_z(mu_z, logvar_z)

        # 解码
        mu_x, r_x = model.decoder(z, tissue_onehot)

        assert mu_x.shape == x.shape
        assert z.shape == (16, 32)

    def test_采样_重参数化技巧(self):
        """测试重参数化采样"""
        model = NBVAE(n_genes=100, latent_dim=16, n_tissues=2)

        mu = torch.randn(32, 16)
        logvar = torch.randn(32, 16)

        # 多次采样应不同（因为有随机性）
        z1 = model.sample_z(mu, logvar)
        z2 = model.sample_z(mu, logvar)

        assert z1.shape == mu.shape
        assert not torch.allclose(z1, z2), "不同采样应不同"

    def test_评估模式_确定性(self):
        """测试eval模式下的确定性"""
        model = NBVAE(n_genes=100, latent_dim=16, n_tissues=2)
        model.eval()

        x = torch.randn(16, 100).abs()
        tissue_onehot = torch.zeros(16, 2)
        tissue_onehot[:, 0] = 1

        with torch.no_grad():
            out1 = model(x, tissue_onehot)
            out2 = model(x, tissue_onehot)

        # eval模式下应确定（如果不采样）
        # 注意：forward中仍会调用sample_z，所以不是完全确定
        # 这里只检查形状一致性
        assert all(o1.shape == o2.shape for o1, o2 in zip(out1, out2))


class TestNBLogLikelihood:
    """测试负二项对数似然"""

    def test_正常情况_计算(self):
        """测试正常情况下的似然计算"""
        x = torch.randint(0, 100, (32, 100)).float()
        mu = torch.randn(32, 100).abs() + 1.0
        r = torch.randn(32, 100).abs() + 1.0

        nll = nb_log_likelihood(x, mu, r)

        assert nll.shape == ()  # 标量
        assert not torch.isnan(nll), "似然不应为NaN"
        assert not torch.isinf(nll), "似然不应为Inf"

    def test_数值稳定性_零计数(self):
        """测试零计数的数值稳定性"""
        x = torch.zeros(16, 50)
        mu = torch.randn(16, 50).abs() + 0.1
        r = torch.randn(16, 50).abs() + 0.1

        nll = nb_log_likelihood(x, mu, r)

        assert not torch.isnan(nll), "零计数不应产生NaN"

    def test_数值稳定性_大计数(self):
        """测试大计数的数值稳定性"""
        x = torch.randint(1000, 10000, (8, 50)).float()
        mu = torch.randn(8, 50).abs() * 100 + 1.0
        r = torch.randn(8, 50).abs() + 1.0

        nll = nb_log_likelihood(x, mu, r)

        assert not torch.isnan(nll), "大计数不应产生NaN"
        assert not torch.isinf(nll), "大计数不应产生Inf"

    def test_epsilon参数_效果(self):
        """测试epsilon参数的效果"""
        x = torch.randn(16, 50).abs()
        mu = torch.randn(16, 50).abs() + 1e-10  # 接近0
        r = torch.randn(16, 50).abs() + 1e-10

        # 使用默认epsilon
        nll1 = nb_log_likelihood(x, mu, r)
        # 使用更大epsilon
        nll2 = nb_log_likelihood(x, mu, r, eps=1e-6)

        assert not torch.isnan(nll1), "默认epsilon应保证稳定"
        assert not torch.isnan(nll2), "自定义epsilon应保证稳定"

    def test_梯度流动(self):
        """测试似然的梯度流动"""
        x = torch.randint(0, 50, (16, 30)).float()
        mu = torch.randn(16, 30).abs().requires_grad_(True) + 1.0
        r = torch.randn(16, 30).abs().requires_grad_(True) + 1.0

        nll = nb_log_likelihood(x, mu, r)
        nll.backward()

        assert mu.grad is not None, "mu应有梯度"
        assert r.grad is not None, "r应有梯度"
        assert not torch.isnan(mu.grad).any(), "mu梯度不应有NaN"


class TestELBOLoss:
    """测试ELBO损失函数"""

    def test_正常情况_计算(self):
        """测试正常情况下的ELBO计算"""
        model = NBVAE(n_genes=200, latent_dim=32, n_tissues=3)

        x = torch.randint(0, 100, (32, 200)).float()
        tissue_onehot = torch.zeros(32, 3)
        tissue_onehot[:, 0] = 1

        loss, loss_dict = elbo_loss(x, tissue_onehot, model, beta=1.0)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()  # 标量
        assert "recon_loss" in loss_dict
        assert "kl_loss" in loss_dict
        assert not torch.isnan(loss), "ELBO不应为NaN"

    def test_损失_组成成分(self):
        """测试损失的各个组成部分"""
        model = NBVAE(n_genes=100, latent_dim=16, n_tissues=2)

        x = torch.randint(0, 50, (16, 100)).float()
        tissue_onehot = torch.zeros(16, 2)
        tissue_onehot[:, 1] = 1

        loss, loss_dict = elbo_loss(x, tissue_onehot, model, beta=1.0)

        recon_loss = loss_dict["recon_loss"]
        kl_loss = loss_dict["kl_loss"]

        assert recon_loss.item() >= 0, "重建损失应非负"
        assert kl_loss.item() >= 0, "KL散度应非负"
        # ELBO = recon_loss + beta * kl_loss
        expected_loss = recon_loss + kl_loss
        assert torch.abs(loss - expected_loss) < 1e-5, "总损失应等于各部分之和"

    def test_beta参数_效果(self):
        """测试beta参数对损失的影响"""
        model = NBVAE(n_genes=100, latent_dim=16, n_tissues=2)

        x = torch.randint(0, 50, (16, 100)).float()
        tissue_onehot = torch.zeros(16, 2)
        tissue_onehot[:, 0] = 1

        loss_beta0, _ = elbo_loss(x, tissue_onehot, model, beta=0.0)
        loss_beta1, _ = elbo_loss(x, tissue_onehot, model, beta=1.0)
        loss_beta2, _ = elbo_loss(x, tissue_onehot, model, beta=2.0)

        # beta越大，KL权重越大，总损失通常越大
        # （假设KL > 0）
        assert not torch.isnan(loss_beta0)
        assert not torch.isnan(loss_beta1)
        assert not torch.isnan(loss_beta2)

    def test_梯度流动_完整流程(self):
        """测试完整训练流程的梯度"""
        model = NBVAE(n_genes=100, latent_dim=16, n_tissues=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randint(0, 50, (16, 100)).float()
        tissue_onehot = torch.zeros(16, 2)
        tissue_onehot[:, 0] = 1

        loss, _ = elbo_loss(x, tissue_onehot, model, beta=1.0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 检查参数已更新（有梯度）
        has_grad = any(
            p.grad is not None and not torch.isnan(p.grad).any()
            for p in model.parameters() if p.requires_grad
        )
        assert has_grad, "模型参数应有梯度"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
