# -*- coding: utf-8 -*-
"""
集成测试 - 端到端训练流程

测试内容：
1. 完整VAE训练流程
2. 完整Operator训练流程
3. 虚拟细胞生成
4. 跨组织效应模拟
5. 模型保存和加载
"""

import torch
import numpy as np
import pytest
import tempfile
from pathlib import Path
from torch.utils.data import DataLoader

# 导入项目模块
from src.config import ModelConfig, TrainingConfig, set_seed, ConditionMeta
from src.models.nb_vae import NBVAE, elbo_loss
from src.models.operator import OperatorModel
from src.utils.cond_encoder import ConditionEncoder
from src.data.scperturb_dataset import SCPerturbPairDataset, create_dataloaders
from src.train.train_embed_core import train_embedding
from src.train.train_operator_core import train_operator
from src.utils.virtual_cell import (
    encode_cells,
    decode_cells,
    apply_operator,
    virtual_cell_scenario,
    compute_reconstruction_metrics
)


def create_test_adata(n_cells=200, n_genes=100):
    """创建测试用的AnnData对象"""
    import anndata
    import pandas as pd

    # 模拟基因表达数据
    mu = np.random.gamma(2, 2, (n_cells, n_genes))
    X = np.random.negative_binomial(n=5, p=5/(5+mu)).astype(np.float32)

    # 创建obs元数据
    obs_data = {
        "tissue": np.random.choice(["kidney", "brain"], n_cells),
        "perturbation": np.random.choice(["control", "drug_A"], n_cells),
        "timepoint": np.random.choice(["t0", "t1"], n_cells),
        "dataset_id": ["dataset_001"] * n_cells,
        "batch": np.random.choice(["batch1", "batch2"], n_cells),
    }

    adata = anndata.AnnData(X=X, obs=pd.DataFrame(obs_data))
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]

    return adata


class TestCompleteVAETraining:
    """测试完整VAE训练流程"""

    def test_vae训练_完整流程(self):
        """测试VAE从初始化到训练完成的完整流程"""
        set_seed(42)
        device = "cpu"

        # 创建测试数据
        adata = create_test_adata(n_cells=200, n_genes=100)
        tissue2idx = {"kidney": 0, "brain": 1}

        # 准备数据
        X = torch.tensor(adata.X, dtype=torch.float32)
        tissue_labels = [tissue2idx[t] for t in adata.obs["tissue"]]
        tissue_onehot = torch.zeros(len(tissue_labels), len(tissue2idx))
        for i, t in enumerate(tissue_labels):
            tissue_onehot[i, t] = 1

        # 创建DataLoader
        dataset = torch.utils.data.TensorDataset(X, tissue_onehot)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        # 创建模型
        model = NBVAE(
            n_genes=100,
            latent_dim=16,
            n_tissues=2,
            hidden_dims=[64, 32]
        ).to(device)

        # 训练配置
        config = TrainingConfig(
            n_epochs=5,  # 少量epoch用于测试
            learning_rate=1e-3,
            beta=1.0
        )

        # 训练模型
        history = train_embedding(
            model=model,
            train_loader=train_loader,
            config=config,
            val_loader=None,
            checkpoint_dir=None,
            device=device
        )

        # 验证训练结果
        assert "train_loss" in history, "应记录训练损失"
        assert len(history["train_loss"]) == 5, "应有5个epoch的记录"

        # 验证损失下降
        first_loss = history["train_loss"][0]
        last_loss = history["train_loss"][-1]
        assert last_loss < first_loss, "训练损失应下降"

        # 验证重建损失和KL散度记录
        assert "train_recon_loss" in history
        assert "train_kl_loss" in history

        # 验证模型可以推理
        model.eval()
        with torch.no_grad():
            x_batch = X[:16].to(device)
            tissue_batch = tissue_onehot[:16].to(device)
            mu_x, r_x, mu_z, logvar_z = model(x_batch, tissue_batch)

            assert mu_x.shape == x_batch.shape, "重建形状应匹配输入"
            assert (mu_x > 0).all(), "重建值应为正"
            assert not torch.isnan(mu_x).any(), "重建不应有NaN"

    def test_vae训练_验证集评估(self):
        """测试VAE训练时的验证集评估"""
        set_seed(42)
        device = "cpu"

        adata = create_test_adata(n_cells=200, n_genes=100)
        tissue2idx = {"kidney": 0, "brain": 1}

        X = torch.tensor(adata.X, dtype=torch.float32)
        tissue_labels = [tissue2idx[t] for t in adata.obs["tissue"]]
        tissue_onehot = torch.zeros(len(tissue_labels), len(tissue2idx))
        for i, t in enumerate(tissue_labels):
            tissue_onehot[i, t] = 1

        # 划分训练集和验证集
        dataset = torch.utils.data.TensorDataset(X, tissue_onehot)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        model = NBVAE(n_genes=100, latent_dim=16, n_tissues=2).to(device)
        config = TrainingConfig(n_epochs=3, learning_rate=1e-3)

        history = train_embedding(
            model=model,
            train_loader=train_loader,
            config=config,
            val_loader=val_loader,
            device=device
        )

        # 验证有验证集损失记录
        assert "val_loss" in history, "应记录验证损失"
        assert len(history["val_loss"]) == 3, "每个epoch应有验证损失"
        assert all(loss > 0 for loss in history["val_loss"]), "验证损失应为正"


class TestCompleteOperatorTraining:
    """测试完整Operator训练流程"""

    def test_operator训练_完整流程(self):
        """测试Operator从初始化到训练完成的完整流程"""
        set_seed(42)
        device = "cpu"

        # 创建测试数据
        adata = create_test_adata(n_cells=200, n_genes=100)

        # 创建条件编码器
        cond_meta = ConditionMeta(
            perturbation_names=["control", "drug_A"],
            tissue_names=["kidney", "brain"],
            timepoint_names=["t0", "t1"],
            batch_names=["batch1", "batch2"]
        )
        cond_encoder = ConditionEncoder(cond_meta)

        # 准备组织映射
        tissue2idx = {"kidney": 0, "brain": 1}

        # 创建数据集
        dataset = SCPerturbPairDataset(
            adata=adata,
            cond_encoder=cond_encoder,
            tissue2idx=tissue2idx,
            max_pairs_per_condition=20,
            seed=42
        )

        # 创建DataLoader
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

        # 首先训练一个简单的VAE
        vae_model = NBVAE(n_genes=100, latent_dim=16, n_tissues=2).to(device)
        vae_model.eval()  # 用于编码，不训练

        # 创建Operator模型
        operator_model = OperatorModel(
            latent_dim=16,
            n_tissues=2,
            n_response_bases=3,
            cond_dim=cond_encoder.get_dim()
        ).to(device)

        # 训练配置
        config = TrainingConfig(
            n_epochs=3,
            learning_rate=1e-3,
            lambda_edist=1.0,
            lambda_spectral=0.1
        )

        # 训练Operator
        history = train_operator(
            operator_model=operator_model,
            embed_model=vae_model,
            train_loader=train_loader,
            config=config,
            val_loader=None,
            checkpoint_dir=None,
            device=device
        )

        # 验证训练结果
        assert "train_loss" in history, "应记录训练损失"
        assert "train_edist_loss" in history, "应记录E-distance损失"
        assert "train_spectral_penalty" in history, "应记录谱范数惩罚"

        # 验证记录长度
        assert len(history["train_loss"]) == 3, "应有3个epoch的记录"

        # 验证所有损失为正
        assert all(loss >= 0 for loss in history["train_edist_loss"])
        assert all(loss >= 0 for loss in history["train_spectral_penalty"])

    def test_operator训练_谱范数约束(self):
        """测试Operator训练时谱范数约束生效"""
        set_seed(42)
        device = "cpu"

        # 创建简单数据
        adata = create_test_adata(n_cells=100, n_genes=50)

        cond_meta = ConditionMeta(
            perturbation_names=["control", "drug_A"],
            tissue_names=["kidney", "brain"],
            timepoint_names=["t0", "t1"],
            batch_names=["batch1"]
        )
        cond_encoder = ConditionEncoder(cond_meta)
        tissue2idx = {"kidney": 0, "brain": 1}

        dataset = SCPerturbPairDataset(
            adata, cond_encoder, tissue2idx, max_pairs_per_condition=10, seed=42
        )
        train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

        vae_model = NBVAE(n_genes=50, latent_dim=16, n_tissues=2).to(device)
        vae_model.eval()

        # 使用严格的谱范数约束
        operator_model = OperatorModel(
            latent_dim=16,
            n_tissues=2,
            n_response_bases=2,
            cond_dim=cond_encoder.get_dim(),
            max_spectral_norm=1.0  # 严格约束
        ).to(device)

        config = TrainingConfig(
            n_epochs=5,
            learning_rate=1e-3,
            lambda_spectral=1.0  # 高权重
        )

        history = train_operator(
            operator_model, vae_model, train_loader, config, device=device
        )

        # 验证谱范数惩罚随训练增加（初期）或趋于稳定（后期）
        spectral_penalties = history["train_spectral_penalty"]
        assert all(p >= 0 for p in spectral_penalties), "谱范数惩罚应非负"

        # 检查算子的实际谱范数
        operator_model.eval()
        with torch.no_grad():
            # 随机条件
            cond_vec = torch.randn(4, cond_encoder.get_dim())
            tissue_idx = torch.zeros(4, dtype=torch.long)
            z = torch.randn(4, 16)

            _, A_theta, _ = operator_model(z, tissue_idx, cond_vec)
            norms = operator_model.compute_operator_norm(A_theta, n_iterations=50)

            # 谱范数应在合理范围内（由于惩罚作用）
            assert (norms < 5.0).all(), "训练后的谱范数应受到约束"


class TestVirtualCellGeneration:
    """测试虚拟细胞生成"""

    def test_虚拟细胞_编码解码循环(self):
        """测试虚拟细胞的编码-解码循环"""
        set_seed(42)
        device = "cpu"

        # 创建模型
        vae_model = NBVAE(n_genes=100, latent_dim=16, n_tissues=2).to(device)
        vae_model.eval()

        # 创建测试数据
        x = torch.randn(32, 100).abs() * 10  # 模拟表达数据
        tissue_onehot = torch.zeros(32, 2)
        tissue_onehot[:16, 0] = 1  # 前16个是tissue 0
        tissue_onehot[16:, 1] = 1  # 后16个是tissue 1

        # 编码
        z = encode_cells(vae_model, x, tissue_onehot, device=device)

        assert z.shape == (32, 16), "潜变量维度应正确"
        assert not torch.isnan(z).any(), "编码不应产生NaN"

        # 解码
        x_recon = decode_cells(vae_model, z, tissue_onehot, device=device)

        assert x_recon.shape == x.shape, "重建应与输入形状相同"
        assert (x_recon > 0).all(), "重建值应为正"
        assert not torch.isnan(x_recon).any(), "解码不应产生NaN"

        # 计算重建质量
        mse, correlation = compute_reconstruction_metrics(
            x, x_recon, device=device
        )

        assert mse.shape == (32,), "MSE应为每个样本一个值"
        assert correlation.shape == (32,), "相关系数应为每个样本一个值"
        assert (correlation >= -1).all() and (correlation <= 1).all(), \
            "Pearson相关系数应在[-1, 1]"

    def test_虚拟细胞_算子应用(self):
        """测试虚拟细胞的算子应用"""
        set_seed(42)
        device = "cpu"

        # 创建模型
        vae_model = NBVAE(n_genes=100, latent_dim=16, n_tissues=2).to(device)
        operator_model = OperatorModel(
            latent_dim=16,
            n_tissues=2,
            n_response_bases=3,
            cond_dim=32
        ).to(device)

        vae_model.eval()
        operator_model.eval()

        # 创建测试数据
        x = torch.randn(16, 100).abs() * 10
        tissue_onehot = torch.zeros(16, 2)
        tissue_onehot[:, 0] = 1
        tissue_idx = torch.zeros(16, dtype=torch.long)
        cond_vec = torch.randn(16, 32)

        # 编码
        z0 = encode_cells(vae_model, x, tissue_onehot, device=device)

        # 应用算子
        z1 = apply_operator(
            operator_model, z0, tissue_idx, cond_vec, device=device
        )

        assert z1.shape == z0.shape, "算子输出应与输入维度相同"
        assert not torch.isnan(z1).any(), "算子应用不应产生NaN"

        # 验证算子确实改变了潜变量
        diff = (z1 - z0).abs().mean()
        assert diff > 1e-4, "算子应改变潜变量"

        # 解码虚拟细胞
        x_virtual = decode_cells(vae_model, z1, tissue_onehot, device=device)

        assert x_virtual.shape == x.shape
        assert (x_virtual > 0).all()

    def test_虚拟细胞_多步场景模拟(self):
        """测试虚拟细胞的多步反事实场景模拟"""
        set_seed(42)
        device = "cpu"

        vae_model = NBVAE(n_genes=50, latent_dim=16, n_tissues=2).to(device)
        operator_model = OperatorModel(
            latent_dim=16, n_tissues=2, n_response_bases=2, cond_dim=32
        ).to(device)

        vae_model.eval()
        operator_model.eval()

        # 初始细胞
        x0 = torch.randn(8, 50).abs() * 10
        tissue_onehot = torch.zeros(8, 2)
        tissue_onehot[:, 0] = 1
        tissue_idx = torch.zeros(8, dtype=torch.long)

        # 多步条件序列
        cond_vec_seq = [
            torch.randn(8, 32),  # t1
            torch.randn(8, 32),  # t2
            torch.randn(8, 32),  # t3
        ]

        # 执行多步模拟
        results = virtual_cell_scenario(
            vae_model=vae_model,
            operator_model=operator_model,
            x0=x0,
            tissue_onehot=tissue_onehot,
            tissue_idx=tissue_idx,
            cond_vec_seq=cond_vec_seq,
            device=device
        )

        # 验证结果
        assert "z_trajectory" in results, "应包含潜变量轨迹"
        assert "x_trajectory" in results, "应包含表达轨迹"

        z_traj = results["z_trajectory"]
        x_traj = results["x_trajectory"]

        # 轨迹长度应为n_steps + 1（包含初始状态）
        assert z_traj.shape == (4, 8, 16), "潜变量轨迹形状错误"
        assert x_traj.shape == (4, 8, 50), "表达轨迹形状错误"

        # 验证初始状态
        z0_encoded = encode_cells(vae_model, x0, tissue_onehot, device=device)
        assert torch.allclose(z_traj[0], z0_encoded, atol=1e-5), \
            "轨迹起点应为初始编码"

        # 验证每步都不同
        for i in range(len(z_traj) - 1):
            diff = (z_traj[i+1] - z_traj[i]).abs().mean()
            assert diff > 1e-5, f"第{i}步应改变潜变量"


class TestCrossTissueEffects:
    """测试跨组织效应"""

    def test_跨组织_不同基线算子(self):
        """测试不同组织使用不同的基线算子"""
        set_seed(42)
        device = "cpu"

        operator_model = OperatorModel(
            latent_dim=16,
            n_tissues=3,  # 3个组织
            n_response_bases=2,
            cond_dim=32
        ).to(device)

        operator_model.eval()

        # 相同的潜变量和条件，不同的组织
        z = torch.randn(3, 16)
        cond_vec = torch.randn(3, 32)
        tissue_idx = torch.tensor([0, 1, 2])  # 三个不同组织

        with torch.no_grad():
            z_out, A_theta, b_theta = operator_model(z, tissue_idx, cond_vec)

        # 验证不同组织的算子矩阵不同
        A0 = A_theta[:, :, :]  # (3, 16, 16)

        # 组织0和组织1的算子应不同
        diff_01 = (A0[0] - A0[1]).abs().mean()
        assert diff_01 > 1e-3, "不同组织的算子应明显不同"

        # 组织0和组织2的算子应不同
        diff_02 = (A0[0] - A0[2]).abs().mean()
        assert diff_02 > 1e-3, "不同组织的算子应明显不同"

    def test_跨组织_相同扰动不同响应(self):
        """测试相同扰动在不同组织中产生不同响应"""
        set_seed(42)
        device = "cpu"

        vae_model = NBVAE(n_genes=50, latent_dim=16, n_tissues=2).to(device)
        operator_model = OperatorModel(
            latent_dim=16, n_tissues=2, n_response_bases=2, cond_dim=32
        ).to(device)

        vae_model.eval()
        operator_model.eval()

        # 相同的初始细胞状态
        x0_kidney = torch.randn(8, 50).abs() * 10
        x0_brain = x0_kidney.clone()  # 相同的基因表达

        # 不同的组织
        tissue_onehot_kidney = torch.zeros(8, 2)
        tissue_onehot_kidney[:, 0] = 1
        tissue_idx_kidney = torch.zeros(8, dtype=torch.long)

        tissue_onehot_brain = torch.zeros(8, 2)
        tissue_onehot_brain[:, 1] = 1
        tissue_idx_brain = torch.ones(8, dtype=torch.long)

        # 相同的扰动条件
        cond_vec = torch.randn(8, 32)

        # 在两个组织中应用相同扰动
        with torch.no_grad():
            # Kidney
            z0_kidney = encode_cells(vae_model, x0_kidney, tissue_onehot_kidney, device)
            z1_kidney = apply_operator(operator_model, z0_kidney, tissue_idx_kidney, cond_vec, device)
            x1_kidney = decode_cells(vae_model, z1_kidney, tissue_onehot_kidney, device)

            # Brain
            z0_brain = encode_cells(vae_model, x0_brain, tissue_onehot_brain, device)
            z1_brain = apply_operator(operator_model, z0_brain, tissue_idx_brain, cond_vec, device)
            x1_brain = decode_cells(vae_model, z1_brain, tissue_onehot_brain, device)

        # 验证不同组织产生不同响应
        # 潜空间响应差异
        delta_z_kidney = z1_kidney - z0_kidney
        delta_z_brain = z1_brain - z0_brain
        diff_z = (delta_z_kidney - delta_z_brain).abs().mean()

        assert diff_z > 1e-3, "相同扰动在不同组织应产生不同的潜空间响应"

        # 表达空间响应差异
        diff_x = (x1_kidney - x1_brain).abs().mean()
        assert diff_x > 1e-2, "相同扰动在不同组织应产生不同的表达响应"


class TestModelSaveLoad:
    """测试模型保存和加载"""

    def test_vae模型_保存和加载(self):
        """测试VAE模型的保存和加载"""
        set_seed(42)
        device = "cpu"

        # 创建模型
        model_original = NBVAE(
            n_genes=100,
            latent_dim=16,
            n_tissues=2,
            hidden_dims=[64, 32]
        ).to(device)

        # 随机输入
        x = torch.randn(8, 100).abs()
        tissue_onehot = torch.zeros(8, 2)
        tissue_onehot[:, 0] = 1

        # 获取原始输出
        model_original.eval()
        with torch.no_grad():
            output_original = model_original(x, tissue_onehot)

        # 保存模型
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "vae_model.pt"
            torch.save(model_original.state_dict(), save_path)

            # 加载模型
            model_loaded = NBVAE(
                n_genes=100,
                latent_dim=16,
                n_tissues=2,
                hidden_dims=[64, 32]
            ).to(device)
            model_loaded.load_state_dict(torch.load(save_path))
            model_loaded.eval()

            # 获取加载后的输出
            with torch.no_grad():
                output_loaded = model_loaded(x, tissue_onehot)

        # 验证输出完全相同
        for orig, loaded in zip(output_original, output_loaded):
            assert torch.allclose(orig, loaded, atol=1e-6), \
                "加载的模型应产生相同输出"

    def test_operator模型_保存和加载(self):
        """测试Operator模型的保存和加载"""
        set_seed(42)
        device = "cpu"

        model_original = OperatorModel(
            latent_dim=16,
            n_tissues=2,
            n_response_bases=3,
            cond_dim=32
        ).to(device)

        z = torch.randn(8, 16)
        tissue_idx = torch.zeros(8, dtype=torch.long)
        cond_vec = torch.randn(8, 32)

        model_original.eval()
        with torch.no_grad():
            output_original = model_original(z, tissue_idx, cond_vec)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "operator_model.pt"
            torch.save(model_original.state_dict(), save_path)

            model_loaded = OperatorModel(
                latent_dim=16,
                n_tissues=2,
                n_response_bases=3,
                cond_dim=32
            ).to(device)
            model_loaded.load_state_dict(torch.load(save_path))
            model_loaded.eval()

            with torch.no_grad():
                output_loaded = model_loaded(z, tissue_idx, cond_vec)

        for orig, loaded in zip(output_original, output_loaded):
            assert torch.allclose(orig, loaded, atol=1e-6), \
                "加载的模型应产生相同输出"

    def test_完整pipeline_保存和恢复(self):
        """测试完整训练pipeline的检查点保存和恢复"""
        set_seed(42)
        device = "cpu"

        # 训练一个epoch
        adata = create_test_adata(n_cells=100, n_genes=50)
        tissue2idx = {"kidney": 0, "brain": 1}

        X = torch.tensor(adata.X, dtype=torch.float32)
        tissue_labels = [tissue2idx[t] for t in adata.obs["tissue"]]
        tissue_onehot = torch.zeros(len(tissue_labels), 2)
        for i, t in enumerate(tissue_labels):
            tissue_onehot[i, t] = 1

        dataset = torch.utils.data.TensorDataset(X, tissue_onehot)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

        model = NBVAE(n_genes=50, latent_dim=16, n_tissues=2).to(device)
        config = TrainingConfig(n_epochs=2, learning_rate=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            # 训练并保存检查点
            history = train_embedding(
                model=model,
                train_loader=train_loader,
                config=config,
                checkpoint_dir=tmpdir,
                device=device
            )

            # 验证检查点文件存在
            checkpoint_files = list(Path(tmpdir).glob("*.pt"))
            assert len(checkpoint_files) > 0, "应保存检查点文件"

            # 加载最后的检查点
            last_checkpoint = sorted(checkpoint_files)[-1]
            checkpoint = torch.load(last_checkpoint)

            # 验证检查点内容
            assert "model_state_dict" in checkpoint
            assert "epoch" in checkpoint
            assert "loss" in checkpoint


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
