#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NBVAE 修复验证脚本

测试修复的问题：
1. hidden_dims → hidden_dim 参数错误
2. 缺失的实例属性（n_genes, latent_dim, n_tissues, hidden_dim）
"""

import torch
from src.models.nb_vae import NBVAE
from src.models.nb_vae import elbo_loss


def test_nbvae_attributes():
    """测试1: 验证NBVAE实例属性"""
    print("=" * 60)
    print("测试1: 验证NBVAE实例属性")
    print("=" * 60)

    model = NBVAE(
        n_genes=100,
        latent_dim=16,
        n_tissues=2,
        hidden_dim=64
    )

    # 检查所有必需的属性
    required_attrs = ['n_genes', 'latent_dim', 'n_tissues', 'hidden_dim']
    for attr in required_attrs:
        assert hasattr(model, attr), f"模型缺失属性: {attr}"
        print(f"✓ {attr}: {getattr(model, attr)}")

    # 验证属性值正确
    assert model.n_genes == 100
    assert model.latent_dim == 16
    assert model.n_tissues == 2
    assert model.hidden_dim == 64

    print("✅ 所有属性检查通过！\n")
    return model


def test_nbvae_forward():
    """测试2: 验证前向传播"""
    print("=" * 60)
    print("测试2: 验证前向传播")
    print("=" * 60)

    model = NBVAE(
        n_genes=100,
        latent_dim=16,
        n_tissues=2,
        hidden_dim=64
    )

    # 创建测试数据
    batch_size = 8
    x = torch.randn(batch_size, 100).abs() * 10  # 模拟计数数据
    tissue_onehot = torch.zeros(batch_size, 2)
    tissue_onehot[:, 0] = 1  # 所有样本属于第一个组织

    # 前向传播
    z, mu_x, r_x, mu_z, logvar_z = model(x, tissue_onehot)

    # 验证输出形状
    assert z.shape == (batch_size, 16), f"z shape错误: {z.shape}"
    assert mu_x.shape == (batch_size, 100), f"mu_x shape错误: {mu_x.shape}"
    assert r_x.shape == (1, 100), f"r_x shape错误: {r_x.shape}"
    assert mu_z.shape == (batch_size, 16), f"mu_z shape错误: {mu_z.shape}"
    assert logvar_z.shape == (batch_size, 16), f"logvar_z shape错误: {logvar_z.shape}"

    print(f"✓ z shape: {z.shape}")
    print(f"✓ mu_x shape: {mu_x.shape}")
    print(f"✓ r_x shape: {r_x.shape}")
    print(f"✓ mu_z shape: {mu_z.shape}")
    print(f"✓ logvar_z shape: {logvar_z.shape}")

    print("✅ 前向传播检查通过！\n")
    return model, x, tissue_onehot


def test_elbo_loss():
    """测试3: 验证ELBO损失计算"""
    print("=" * 60)
    print("测试3: 验证ELBO损失计算")
    print("=" * 60)

    model = NBVAE(
        n_genes=100,
        latent_dim=16,
        n_tissues=2,
        hidden_dim=64
    )

    # 创建测试数据
    batch_size = 8
    x = torch.randn(batch_size, 100).abs() * 10
    tissue_onehot = torch.zeros(batch_size, 2)
    tissue_onehot[:, 0] = 1

    # 计算损失
    loss, loss_dict = elbo_loss(x, tissue_onehot, model, beta=1.0)

    # 验证返回值
    assert isinstance(loss, torch.Tensor), "loss应该是torch.Tensor"
    assert isinstance(loss_dict, dict), "loss_dict应该是字典"
    assert 'recon_loss' in loss_dict, "loss_dict缺少recon_loss"
    assert 'kl_loss' in loss_dict, "loss_dict缺少kl_loss"
    assert 'z' in loss_dict, "loss_dict缺少z"

    print(f"✓ loss: {loss.item():.4f}")
    print(f"✓ recon_loss: {loss_dict['recon_loss'].item():.4f}")
    print(f"✓ kl_loss: {loss_dict['kl_loss'].item():.4f}")
    print(f"✓ z shape: {loss_dict['z'].shape}")

    # 验证损失可以反向传播
    loss.backward()
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "损失没有产生梯度"
    print("✓ 梯度反向传播成功")

    print("✅ ELBO损失检查通过！\n")


def test_model_config_access():
    """测试4: 验证模型配置访问（模拟train_embed_core.py中的使用）"""
    print("=" * 60)
    print("测试4: 验证模型配置访问")
    print("=" * 60)

    model = NBVAE(
        n_genes=2000,
        latent_dim=32,
        n_tissues=3,
        hidden_dim=256
    )

    # 模拟 train_embed_core.py:146-148 中的访问
    model_config = {
        "n_genes": model.n_genes,
        "latent_dim": model.latent_dim,
        "n_tissues": model.n_tissues,
    }

    print(f"✓ 成功访问模型配置:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")

    assert model_config['n_genes'] == 2000
    assert model_config['latent_dim'] == 32
    assert model_config['n_tissues'] == 3

    print("✅ 模型配置访问检查通过！\n")


def test_parameter_names():
    """测试5: 验证参数名称（确认使用hidden_dim而非hidden_dims）"""
    print("=" * 60)
    print("测试5: 验证参数名称")
    print("=" * 60)

    # 应该成功
    try:
        model = NBVAE(
            n_genes=100,
            latent_dim=16,
            n_tissues=2,
            hidden_dim=64  # 正确：单数
        )
        print("✓ 使用 hidden_dim 参数成功")
    except TypeError as e:
        print(f"❌ 使用 hidden_dim 失败: {e}")
        raise

    # 应该失败
    try:
        model = NBVAE(
            n_genes=100,
            latent_dim=16,
            n_tissues=2,
            hidden_dims=[64, 32]  # 错误：复数
        )
        print("❌ 使用 hidden_dims 参数意外成功（应该失败）")
        raise AssertionError("hidden_dims参数不应该被接受")
    except TypeError:
        print("✓ 使用 hidden_dims 参数正确地失败了")

    print("✅ 参数名称检查通过！\n")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("NBVAE 修复验证测试套件")
    print("=" * 60 + "\n")

    try:
        test_nbvae_attributes()
        test_nbvae_forward()
        test_elbo_loss()
        test_model_config_access()
        test_parameter_names()

        print("=" * 60)
        print("✅✅✅ 所有测试通过！修复成功！ ✅✅✅")
        print("=" * 60)
        return 0

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌❌❌ 测试失败: {e} ❌❌❌")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
