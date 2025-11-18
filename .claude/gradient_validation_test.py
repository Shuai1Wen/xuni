# -*- coding: utf-8 -*-
"""
梯度传播验证测试

验证spectral_penalty和elbo_loss的梯度传播是否正确
"""

import torch
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.operator import OperatorModel
from src.models.nb_vae import NBVAE, elbo_loss


def test_spectral_penalty_gradient():
    """测试spectral_penalty的梯度传播"""
    print("=" * 60)
    print("测试1: spectral_penalty梯度传播")
    print("=" * 60)

    model = OperatorModel(
        latent_dim=16,
        n_tissues=2,
        n_response_bases=3,
        cond_dim=8
    )

    # 计算谱范数惩罚
    penalty = model.spectral_penalty(max_allowed=1.0, n_iterations=10)

    # 验证可微性
    print(f"✓ penalty.requires_grad: {penalty.requires_grad}")
    assert penalty.requires_grad, "penalty应该可微"

    # 反向传播
    penalty.backward()

    # 验证梯度
    assert model.A0_tissue.grad is not None, "A0_tissue应该有梯度"
    assert model.B.grad is not None, "B应该有梯度"

    print(f"✓ A0_tissue.grad 非空: {model.A0_tissue.grad is not None}")
    print(f"✓ A0_tissue.grad 范数: {model.A0_tissue.grad.norm():.6f}")
    print(f"✓ B.grad 非空: {model.B.grad is not None}")
    print(f"✓ B.grad 范数: {model.B.grad.norm():.6f}")

    print("✅ spectral_penalty梯度传播正确\n")


def test_elbo_loss_gradient():
    """测试elbo_loss的梯度传播"""
    print("=" * 60)
    print("测试2: elbo_loss梯度传播")
    print("=" * 60)

    model = NBVAE(
        n_genes=100,
        latent_dim=16,
        n_tissues=2,
        hidden_dim=64
    )

    # 模拟数据
    batch_size = 8
    x = torch.randn(batch_size, 100).abs() * 10  # 模拟计数数据
    tissue_onehot = torch.zeros(batch_size, 2)
    tissue_onehot[:, 0] = 1

    # 计算损失
    loss, loss_dict = elbo_loss(x, tissue_onehot, model, beta=1.0)

    # 验证返回值
    print(f"✓ loss.requires_grad: {loss.requires_grad}")
    print(f"✓ loss_dict['recon_loss'].requires_grad: {loss_dict['recon_loss'].requires_grad}")
    print(f"✓ loss_dict['kl_loss'].requires_grad: {loss_dict['kl_loss'].requires_grad}")
    print(f"✓ loss_dict['z'].requires_grad: {loss_dict['z'].requires_grad}")

    assert loss.requires_grad, "loss应该可微"
    assert not loss_dict['recon_loss'].requires_grad, "记录值应该detached"
    assert not loss_dict['kl_loss'].requires_grad, "记录值应该detached"
    assert not loss_dict['z'].requires_grad, "记录值应该detached"

    # 反向传播
    loss.backward()

    # 验证梯度
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            print(f"✓ {name} 有梯度，范数: {param.grad.norm():.6f}")

    assert has_grad, "至少有一个参数应该有梯度"
    print("✅ elbo_loss梯度传播正确\n")


def test_compute_operator_norm_no_grad():
    """测试compute_operator_norm不产生梯度"""
    print("=" * 60)
    print("测试3: compute_operator_norm不产生梯度（预期行为）")
    print("=" * 60)

    model = OperatorModel(
        latent_dim=16,
        n_tissues=2,
        n_response_bases=3,
        cond_dim=8
    )

    # 模拟数据
    batch_size = 4
    tissue_idx = torch.zeros(batch_size, dtype=torch.long)
    cond_vec = torch.randn(batch_size, 8)

    # 计算范数
    norms = model.compute_operator_norm(
        tissue_idx, cond_vec,
        norm_type="spectral",
        n_iterations=10
    )

    # 验证不可微（预期行为）
    print(f"✓ norms.requires_grad: {norms.requires_grad}")
    assert not norms.requires_grad, "norms不应该可微（因为@torch.no_grad()）"

    # 验证无法反向传播（预期行为）
    try:
        norms.sum().backward()
        print("❌ 不应该能够反向传播")
        assert False, "norms应该无法反向传播"
    except RuntimeError as e:
        print(f"✓ 预期的错误: {str(e)[:50]}...")
        print("✅ compute_operator_norm正确地使用@torch.no_grad()\n")


def test_spectral_penalty_vs_compute_operator_norm():
    """对比spectral_penalty和compute_operator_norm的区别"""
    print("=" * 60)
    print("测试4: spectral_penalty vs compute_operator_norm职责对比")
    print("=" * 60)

    model = OperatorModel(
        latent_dim=16,
        n_tissues=2,
        n_response_bases=3,
        cond_dim=8
    )

    batch_size = 4
    tissue_idx = torch.zeros(batch_size, dtype=torch.long)
    cond_vec = torch.randn(batch_size, 8)

    # spectral_penalty: 用于训练
    penalty = model.spectral_penalty(max_allowed=1.05, n_iterations=5)
    print(f"spectral_penalty:")
    print(f"  - requires_grad: {penalty.requires_grad} (应该为True，用于损失计算)")
    print(f"  - 用途: 训练时的稳定性正则化")
    print(f"  - 值: {penalty.item():.6f}")

    # compute_operator_norm: 用于监控
    norms = model.compute_operator_norm(
        tissue_idx, cond_vec,
        norm_type="spectral",
        n_iterations=10
    )
    print(f"\ncompute_operator_norm:")
    print(f"  - requires_grad: {norms.requires_grad} (应该为False，用于监控)")
    print(f"  - 用途: 验证/测试时的范数监控")
    print(f"  - 值: mean={norms.mean().item():.6f}, max={norms.max().item():.6f}")

    print("\n✅ 职责分离清晰\n")


def main():
    """运行所有测试"""
    print("\n" + "🧪 梯度传播验证测试套件".center(60, "="))
    print()

    try:
        test_spectral_penalty_gradient()
        test_elbo_loss_gradient()
        test_compute_operator_norm_no_grad()
        test_spectral_penalty_vs_compute_operator_norm()

        print("=" * 60)
        print("🎉 所有测试通过！梯度传播完全正确".center(60))
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        raise
    except Exception as e:
        print(f"\n❌ 意外错误: {e}")
        raise


if __name__ == "__main__":
    main()
