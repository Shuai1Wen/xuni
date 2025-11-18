#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
核心模块功能验证脚本

本脚本测试以下模块的正确性：
1. src/models/nb_vae.py - NBVAE编码/解码/损失
2. src/models/operator.py - 算子应用/谱范数
3. src/utils/edistance.py - E-distance计算
4. src/utils/virtual_cell.py - 虚拟细胞接口
5. src/utils/cond_encoder.py - 条件编码

运行方式：
    python test_core_modules.py

预期输出：
    - 维度检查：✓ 或 ✗
    - 数值检查：✓ 或 ✗
    - 性能指标：运行时间、内存占用
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
import time
import numpy as np

# 添加src目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.nb_vae import NBVAE, elbo_loss, sample_z, nb_log_likelihood
from src.models.operator import OperatorModel
from src.utils.edistance import energy_distance, check_edistance_properties
from src.utils.virtual_cell import (
    encode_cells, decode_cells, apply_operator, virtual_cell_scenario
)
from src.utils.cond_encoder import ConditionEncoder


def test_nb_vae():
    """测试NBVAE模块"""
    print("\n" + "="*60)
    print("测试1：NBVAE模块")
    print("="*60)

    # 参数
    n_genes = 2000
    latent_dim = 32
    n_tissues = 3
    batch_size = 64

    # 创建模型
    vae = NBVAE(n_genes=n_genes, latent_dim=latent_dim, n_tissues=n_tissues)

    # 创建测试数据
    x = torch.randn(batch_size, n_genes)
    tissue_onehot = torch.zeros(batch_size, n_tissues)
    tissue_onehot[:, 0] = 1.0

    print(f"\n输入维度：x={x.shape}, tissue_onehot={tissue_onehot.shape}")

    # 测试编码
    mu_z, logvar_z = vae.encoder(x, tissue_onehot)
    assert mu_z.shape == (batch_size, latent_dim), f"编码器输出维度错误：{mu_z.shape}"
    assert logvar_z.shape == (batch_size, latent_dim), f"编码器logvar维度错误：{logvar_z.shape}"
    print(f"✓ 编码器输出正确：mu_z={mu_z.shape}, logvar_z={logvar_z.shape}")

    # 测试采样
    z = sample_z(mu_z, logvar_z)
    assert z.shape == (batch_size, latent_dim), f"采样输出维度错误：{z.shape}"
    print(f"✓ 重参数化采样正确：z={z.shape}")

    # 测试解码
    mu_x, r_x = vae.decoder(z, tissue_onehot)
    assert mu_x.shape == (batch_size, n_genes), f"解码器μ维度错误：{mu_x.shape}"
    assert r_x.shape == (1, n_genes), f"解码器r维度错误：{r_x.shape}"
    print(f"✓ 解码器输出正确：mu_x={mu_x.shape}, r_x={r_x.shape}")

    # 测试负二项似然度
    log_p = nb_log_likelihood(x, mu_x, r_x)
    assert log_p.shape == (batch_size,), f"似然度维度错误：{log_p.shape}"
    assert not torch.isnan(log_p).any(), "似然度包含NaN"
    print(f"✓ 负二项似然度正确：shape={log_p.shape}, 平均值={log_p.mean():.4f}")

    # 测试完整前向传播
    z, mu_x, r_x, mu_z, logvar_z = vae(x, tissue_onehot)
    assert z.shape == (batch_size, latent_dim), f"VAE z维度错误"
    print(f"✓ VAE前向传播正确：z={z.shape}, mu_x={mu_x.shape}")

    # 测试ELBO损失
    loss, z_detached = elbo_loss(x, tissue_onehot, vae)
    assert loss.dim() == 0, f"损失应该是标量，但得到{loss.shape}"
    assert loss > 0, f"损失应该为正，但得到{loss}"
    print(f"✓ ELBO损失正确：loss={loss:.4f}")

    print("\n✓ NBVAE模块测试通过")


def test_operator():
    """测试OperatorModel模块"""
    print("\n" + "="*60)
    print("测试2：OperatorModel模块")
    print("="*60)

    # 参数
    latent_dim = 32
    n_tissues = 3
    n_response_bases = 5
    cond_dim = 64
    batch_size = 64

    # 创建模型
    operator = OperatorModel(
        latent_dim=latent_dim,
        n_tissues=n_tissues,
        n_response_bases=n_response_bases,
        cond_dim=cond_dim
    )

    # 创建测试数据
    z = torch.randn(batch_size, latent_dim)
    tissue_idx = torch.randint(0, n_tissues, (batch_size,))
    cond_vec = torch.randn(batch_size, cond_dim)

    print(f"\n输入维度：z={z.shape}, tissue_idx={tissue_idx.shape}, cond_vec={cond_vec.shape}")

    # 测试前向传播
    z_out, A_theta, b_theta = operator(z, tissue_idx, cond_vec)
    assert z_out.shape == (batch_size, latent_dim), f"输出z维度错误：{z_out.shape}"
    assert A_theta.shape == (batch_size, latent_dim, latent_dim), f"A_theta维度错误：{A_theta.shape}"
    assert b_theta.shape == (batch_size, latent_dim), f"b_theta维度错误：{b_theta.shape}"
    print(f"✓ 算子前向传播正确：z_out={z_out.shape}, A_theta={A_theta.shape}, b_theta={b_theta.shape}")

    # 测试响应轮廓
    alpha, beta = operator.get_response_profile(cond_vec)
    assert alpha.shape == (batch_size, n_response_bases), f"alpha维度错误"
    assert beta.shape == (batch_size, n_response_bases), f"beta维度错误"
    print(f"✓ 响应轮廓正确：alpha={alpha.shape}, beta={beta.shape}")

    # 测试谱范数约束
    penalty = operator.spectral_penalty(max_allowed=1.05)
    assert penalty.dim() == 0, f"惩罚应该是标量"
    assert penalty >= 0, f"惩罚应该非负"
    print(f"✓ 谱范数约束正确：penalty={penalty:.6f}")

    # 测试算子范数计算
    norms = operator.compute_operator_norm(tissue_idx, cond_vec, norm_type="frobenius")
    assert norms.shape == (batch_size,), f"范数维度错误：{norms.shape}"
    assert (norms > 0).all(), f"范数应该为正"
    print(f"✓ 算子范数计算正确：mean={norms.mean():.4f}, std={norms.std():.4f}")

    print("\n✓ OperatorModel模块测试通过")


def test_edistance():
    """测试E-distance模块"""
    print("\n" + "="*60)
    print("测试3：E-distance模块")
    print("="*60)

    # 参数
    n_x, n_y = 100, 80
    d = 32

    # 创建测试数据
    x = torch.randn(n_x, d)
    y = torch.randn(n_y, d)

    print(f"\n输入维度：x={x.shape}, y={y.shape}")

    # 计算E-distance
    ed2 = energy_distance(x, y)
    assert ed2.dim() == 0, f"E-distance应该是标量，但得到{ed2.shape}"
    assert ed2 >= -1e-6, f"E-distance应该非负，但得到{ed2}"
    print(f"✓ E-distance计算正确：ed2={ed2:.6f}")

    # 验证E-distance性质
    results = check_edistance_properties(x, y, verbose=True)
    if all(results.values()):
        print(f"✓ E-distance数学性质验证通过")
    else:
        print(f"✗ E-distance性质验证失败：{results}")

    # 测试对称性
    ed2_xy = energy_distance(x, y)
    ed2_yx = energy_distance(y, x)
    sym_error = abs(ed2_xy - ed2_yx).item()
    assert sym_error < 1e-6, f"对称性失败：{sym_error}"
    print(f"✓ 对称性验证通过：error={sym_error:.2e}")

    # 测试同一性
    ed2_xx = energy_distance(x, x)
    assert ed2_xx < 1e-6, f"同一性失败：{ed2_xx}"
    print(f"✓ 同一性验证通过：ed2(X,X)={ed2_xx:.2e}")

    print("\n✓ E-distance模块测试通过")


def test_virtual_cell():
    """测试虚拟细胞接口"""
    print("\n" + "="*60)
    print("测试4：虚拟细胞接口")
    print("="*60)

    # 参数
    n_genes = 2000
    latent_dim = 32
    n_tissues = 3
    n_response_bases = 5
    cond_dim = 64
    batch_size = 32

    # 创建模型
    vae = NBVAE(n_genes=n_genes, latent_dim=latent_dim, n_tissues=n_tissues)
    operator = OperatorModel(latent_dim, n_tissues, n_response_bases, cond_dim)

    # 创建测试数据
    x = torch.randn(batch_size, n_genes)
    tissue_onehot = torch.zeros(batch_size, n_tissues)
    tissue_onehot[:, 0] = 1.0
    tissue_idx = torch.zeros(batch_size, dtype=torch.long)

    print(f"\n输入维度：x={x.shape}, tissue_onehot={tissue_onehot.shape}")

    # 测试编码
    z = encode_cells(vae, x, tissue_onehot, device="cpu")
    assert z.shape == (batch_size, latent_dim), f"编码维度错误：{z.shape}"
    print(f"✓ 编码正确：z={z.shape}")

    # 测试解码
    x_recon = decode_cells(vae, z, tissue_onehot, device="cpu")
    assert x_recon.shape == (batch_size, n_genes), f"解码维度错误：{x_recon.shape}"
    print(f"✓ 解码正确：x_recon={x_recon.shape}")

    # 测试算子应用
    cond_vec = torch.randn(batch_size, cond_dim)
    z_out = apply_operator(operator, z, tissue_idx, cond_vec, device="cpu")
    assert z_out.shape == (batch_size, latent_dim), f"算子应用维度错误：{z_out.shape}"
    print(f"✓ 算子应用正确：z_out={z_out.shape}")

    # 测试多步模拟
    cond_vec_seq = torch.randn(2, cond_dim)  # 2步
    x_virtual = virtual_cell_scenario(
        vae, operator, x, tissue_onehot, tissue_idx, cond_vec_seq,
        device="cpu", return_trajectory=False
    )
    assert x_virtual.shape == (batch_size, n_genes), f"虚拟细胞维度错误：{x_virtual.shape}"
    print(f"✓ 多步模拟正确：x_virtual={x_virtual.shape}")

    # 测试轨迹返回
    x_virtual, z_traj, x_traj = virtual_cell_scenario(
        vae, operator, x, tissue_onehot, tissue_idx, cond_vec_seq,
        device="cpu", return_trajectory=True
    )
    assert z_traj.shape == (3, batch_size, latent_dim), f"z轨迹维度错误：{z_traj.shape}"
    assert x_traj.shape == (3, batch_size, n_genes), f"x轨迹维度错误：{x_traj.shape}"
    print(f"✓ 轨迹返回正确：z_traj={z_traj.shape}, x_traj={x_traj.shape}")

    print("\n✓ 虚拟细胞接口测试通过")


def test_cond_encoder():
    """测试条件编码器"""
    print("\n" + "="*60)
    print("测试5：条件编码器")
    print("="*60)

    # 参数
    cond_dim = 64

    # 创建映射
    perturb2idx = {"drug_A": 0, "drug_B": 1, "control": 2}
    tissue2idx = {"blood": 0, "kidney": 1, "brain": 2}
    batch2idx = {"batch1": 0, "batch2": 1}

    # 创建编码器
    encoder = ConditionEncoder(
        perturb2idx=perturb2idx,
        tissue2idx=tissue2idx,
        batch2idx=batch2idx,
        cond_dim=cond_dim,
        use_embedding=True,
        perturb_embed_dim=16,
        tissue_embed_dim=8
    )

    print(f"\n创建编码器：cond_dim={cond_dim}")

    # 测试单个样本编码
    obs_row = {
        "perturbation": "drug_A",
        "tissue": "kidney",
        "batch": "batch1",
        "mLOY_load": 0.15
    }
    cond_vec = encoder.encode_obs_row(obs_row)
    assert cond_vec.shape == (cond_dim,), f"编码维度错误：{cond_vec.shape}"
    print(f"✓ 单个样本编码正确：cond_vec={cond_vec.shape}")

    # 测试批量编码
    obs_rows = [obs_row for _ in range(32)]
    cond_vecs = encoder(obs_rows)
    assert cond_vecs.shape == (32, cond_dim), f"批量编码维度错误：{cond_vecs.shape}"
    print(f"✓ 批量编码正确：cond_vecs={cond_vecs.shape}")

    # 测试OOV处理
    obs_row_oov = {
        "perturbation": "unknown_drug",
        "tissue": "unknown_tissue",
        "batch": "batch1",
        "mLOY_load": 0.0
    }
    cond_vec_oov = encoder.encode_obs_row(obs_row_oov)
    assert cond_vec_oov.shape == (cond_dim,), f"OOV编码维度错误"
    print(f"✓ OOV处理正确")

    print("\n✓ 条件编码器测试通过")


def test_performance():
    """性能测试"""
    print("\n" + "="*60)
    print("测试6：性能测试")
    print("="*60)

    # 参数
    latent_dim = 32
    n_response_bases = 5
    cond_dim = 64

    # 测试算子应用性能
    print("\n算子应用性能测试：")
    operator = OperatorModel(latent_dim, 3, n_response_bases, cond_dim)

    batch_sizes = [32, 64, 128, 256]
    for batch_size in batch_sizes:
        z = torch.randn(batch_size, latent_dim)
        tissue_idx = torch.randint(0, 3, (batch_size,))
        cond_vec = torch.randn(batch_size, cond_dim)

        start = time.time()
        for _ in range(10):
            z_out, _, _ = operator(z, tissue_idx, cond_vec)
        elapsed = (time.time() - start) / 10

        print(f"  batch_size={batch_size:3d}: {elapsed*1000:.2f}ms")

    # 测试E-distance性能
    print("\nE-distance计算性能测试：")
    from src.utils.edistance import pairwise_distances

    dims = [100, 500, 1000]
    for n in dims:
        x = torch.randn(n, 32)
        y = torch.randn(n, 32)

        start = time.time()
        ed2 = energy_distance(x, y)
        elapsed = time.time() - start

        print(f"  n={n:4d}: {elapsed*1000:.2f}ms, ed2={ed2:.6f}")

    print("\n✓ 性能测试完成")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("虚拟细胞算子模型 - 核心模块验证")
    print("="*60)

    try:
        test_nb_vae()
        test_operator()
        test_edistance()
        test_cond_encoder()
        test_virtual_cell()
        test_performance()

        print("\n" + "="*60)
        print("✓ 所有测试通过！")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n✗ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
