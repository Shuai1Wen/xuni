#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能分析脚本

使用torch.profiler分析训练和推理的性能瓶颈

运行方式：
    python scripts/profile_performance.py --mode vae
    python scripts/profile_performance.py --mode operator
    python scripts/profile_performance.py --mode inference

输出：
    - 性能分析报告（文本格式）
    - Chrome trace文件（可在chrome://tracing查看）
    - 性能优化建议
"""

import torch
import numpy as np
import argparse
from pathlib import Path
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入项目模块
from src.config import ModelConfig, TrainingConfig, set_seed, ConditionMeta
from src.models.nb_vae import NBVAE, elbo_loss
from src.models.operator import OperatorModel
from src.utils.cond_encoder import ConditionEncoder
from src.data.scperturb_dataset import SCPerturbPairDataset
from src.utils.virtual_cell import encode_cells, decode_cells, apply_operator


def create_test_data(n_cells=1000, n_genes=2000):
    """创建测试数据"""
    import anndata
    import pandas as pd

    mu = np.random.gamma(2, 2, (n_cells, n_genes))
    X = np.random.negative_binomial(n=5, p=5/(5+mu)).astype(np.float32)

    obs_data = {
        "tissue": np.random.choice(["kidney", "brain", "blood"], n_cells),
        "perturbation": np.random.choice(["control", "drug_A", "drug_B"], n_cells),
        "timepoint": np.random.choice(["t0", "t1"], n_cells),
        "dataset_id": ["dataset_001"] * n_cells,
        "batch": np.random.choice(["batch1", "batch2"], n_cells),
    }

    adata = anndata.AnnData(X=X, obs=pd.DataFrame(obs_data))
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]

    return adata


def profile_vae_training(device="cpu", n_steps=10):
    """性能分析：VAE训练"""
    logger.info("=" * 60)
    logger.info("VAE训练性能分析")
    logger.info("=" * 60)

    set_seed(42)

    # 创建数据
    adata = create_test_data(n_cells=500, n_genes=1000)
    tissue2idx = {"kidney": 0, "brain": 1, "blood": 2}

    X = torch.tensor(adata.X, dtype=torch.float32)
    tissue_labels = [tissue2idx[t] for t in adata.obs["tissue"]]
    tissue_onehot = torch.zeros(len(tissue_labels), len(tissue2idx))
    for i, t in enumerate(tissue_labels):
        tissue_onehot[i, t] = 1

    dataset = torch.utils.data.TensorDataset(X, tissue_onehot)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 创建模型
    model = NBVAE(
        n_genes=1000,
        latent_dim=32,
        n_tissues=3,
        hidden_dims=[256, 128]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 性能分析
    logger.info(f"开始profiling {n_steps}个训练步骤...")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if device == "cuda" else [ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function("vae_training"):
            step = 0
            for x_batch, tissue_batch in train_loader:
                if step >= n_steps:
                    break

                x_batch = x_batch.to(device)
                tissue_batch = tissue_batch.to(device)

                with record_function("forward_pass"):
                    loss, loss_dict = elbo_loss(x_batch, tissue_batch, model, beta=1.0)

                with record_function("backward_pass"):
                    optimizer.zero_grad()
                    loss.backward()

                with record_function("optimizer_step"):
                    optimizer.step()

                step += 1
                prof.step()

    # 输出报告
    output_dir = Path("results/profiling")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存详细报告
    report_path = output_dir / "vae_training_profile.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("VAE训练性能分析报告\n")
        f.write("="*80 + "\n\n")

        f.write("【按CPU时间排序】\n")
        f.write("-"*80 + "\n")
        f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
        f.write("\n\n")

        if device == "cuda":
            f.write("【按CUDA时间排序】\n")
            f.write("-"*80 + "\n")
            f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
            f.write("\n\n")

        f.write("【按内存使用排序】\n")
        f.write("-"*80 + "\n")
        f.write(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=20))

    logger.info(f"✓ 详细报告已保存至: {report_path}")

    # 保存Chrome trace
    trace_path = output_dir / "vae_training_trace.json"
    prof.export_chrome_trace(str(trace_path))
    logger.info(f"✓ Chrome trace已保存至: {trace_path}")
    logger.info(f"  可在 chrome://tracing 中打开查看")

    # 分析瓶颈
    analyze_bottlenecks(prof, "VAE训练")


def profile_operator_training(device="cpu", n_steps=10):
    """性能分析：Operator训练"""
    logger.info("=" * 60)
    logger.info("Operator训练性能分析")
    logger.info("=" * 60)

    set_seed(42)

    # 创建数据
    adata = create_test_data(n_cells=500, n_genes=1000)

    cond_meta = ConditionMeta(
        perturbation_names=["control", "drug_A", "drug_B"],
        tissue_names=["kidney", "brain", "blood"],
        timepoint_names=["t0", "t1"],
        batch_names=["batch1", "batch2"]
    )
    cond_encoder = ConditionEncoder(cond_meta)
    tissue2idx = {"kidney": 0, "brain": 1, "blood": 2}

    dataset = SCPerturbPairDataset(
        adata, cond_encoder, tissue2idx, max_pairs_per_condition=50, seed=42
    )
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 创建模型
    vae_model = NBVAE(n_genes=1000, latent_dim=32, n_tissues=3).to(device)
    vae_model.eval()

    operator_model = OperatorModel(
        latent_dim=32,
        n_tissues=3,
        n_response_bases=4,
        cond_dim=cond_encoder.get_dim()
    ).to(device)

    optimizer = torch.optim.Adam(operator_model.parameters(), lr=1e-3)

    # 性能分析
    logger.info(f"开始profiling {n_steps}个训练步骤...")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if device == "cuda" else [ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function("operator_training"):
            step = 0
            for batch in train_loader:
                if step >= n_steps:
                    break

                x_ctrl = batch["x_ctrl"].to(device)
                x_pert = batch["x_pert"].to(device)
                tissue_onehot = batch["tissue_onehot"].to(device)
                tissue_idx = batch["tissue_idx"].to(device)
                cond_vec_pert = batch["cond_vec_pert"].to(device)

                with record_function("encode_cells"):
                    with torch.no_grad():
                        z_ctrl = encode_cells(vae_model, x_ctrl, tissue_onehot, device)
                        z_pert_true = encode_cells(vae_model, x_pert, tissue_onehot, device)

                with record_function("operator_forward"):
                    z_pert_pred, A_theta, b_theta = operator_model(
                        z_ctrl, tissue_idx, cond_vec_pert
                    )

                with record_function("compute_loss"):
                    # E-distance
                    from src.utils.edistance import energy_distance
                    edist_loss = energy_distance(z_pert_pred, z_pert_true)

                    # Spectral penalty
                    spectral_penalty = operator_model.spectral_penalty(max_allowed=1.05)

                    loss = edist_loss + 0.1 * spectral_penalty

                with record_function("backward_pass"):
                    optimizer.zero_grad()
                    loss.backward()

                with record_function("optimizer_step"):
                    optimizer.step()

                step += 1
                prof.step()

    # 输出报告
    output_dir = Path("results/profiling")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "operator_training_profile.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("Operator训练性能分析报告\n")
        f.write("="*80 + "\n\n")

        f.write("【按CPU时间排序】\n")
        f.write("-"*80 + "\n")
        f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
        f.write("\n\n")

        if device == "cuda":
            f.write("【按CUDA时间排序】\n")
            f.write("-"*80 + "\n")
            f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
            f.write("\n\n")

        f.write("【按内存使用排序】\n")
        f.write("-"*80 + "\n")
        f.write(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=20))

    logger.info(f"✓ 详细报告已保存至: {report_path}")

    trace_path = output_dir / "operator_training_trace.json"
    prof.export_chrome_trace(str(trace_path))
    logger.info(f"✓ Chrome trace已保存至: {trace_path}")

    analyze_bottlenecks(prof, "Operator训练")


def profile_inference(device="cpu", n_iterations=100):
    """性能分析：推理性能"""
    logger.info("=" * 60)
    logger.info("推理性能分析")
    logger.info("=" * 60)

    set_seed(42)

    # 创建模型
    vae_model = NBVAE(n_genes=2000, latent_dim=32, n_tissues=3).to(device)
    operator_model = OperatorModel(
        latent_dim=32, n_tissues=3, n_response_bases=4, cond_dim=64
    ).to(device)

    vae_model.eval()
    operator_model.eval()

    # 测试数据
    x = torch.randn(128, 2000).abs().to(device)
    tissue_onehot = torch.zeros(128, 3).to(device)
    tissue_onehot[:, 0] = 1
    tissue_idx = torch.zeros(128, dtype=torch.long).to(device)
    cond_vec = torch.randn(128, 64).to(device)

    logger.info(f"开始profiling {n_iterations}次推理...")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if device == "cuda" else [ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True
    ) as prof:
        with record_function("inference_pipeline"):
            for _ in range(n_iterations):
                with torch.no_grad():
                    with record_function("encode"):
                        z = encode_cells(vae_model, x, tissue_onehot, device)

                    with record_function("operator_apply"):
                        z_out = apply_operator(operator_model, z, tissue_idx, cond_vec, device)

                    with record_function("decode"):
                        x_out = decode_cells(vae_model, z_out, tissue_onehot, device)

                prof.step()

    # 输出报告
    output_dir = Path("results/profiling")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "inference_profile.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("推理性能分析报告\n")
        f.write("="*80 + "\n\n")

        f.write(f"测试配置：\n")
        f.write(f"  批次大小: 128\n")
        f.write(f"  基因数: 2000\n")
        f.write(f"  潜空间维度: 32\n")
        f.write(f"  迭代次数: {n_iterations}\n\n")

        f.write("【按CPU时间排序】\n")
        f.write("-"*80 + "\n")
        f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
        f.write("\n\n")

        if device == "cuda":
            f.write("【按CUDA时间排序】\n")
            f.write("-"*80 + "\n")
            f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
            f.write("\n\n")

        # 计算平均时间
        events = prof.key_averages()
        encode_time = sum(e.cpu_time_total for e in events if "encode" in e.key.lower())
        operator_time = sum(e.cpu_time_total for e in events if "operator" in e.key.lower())
        decode_time = sum(e.cpu_time_total for e in events if "decode" in e.key.lower())
        total_time = encode_time + operator_time + decode_time

        f.write("【平均单次推理时间】\n")
        f.write("-"*80 + "\n")
        f.write(f"  编码 (encode):      {encode_time/n_iterations/1000:.2f} ms\n")
        f.write(f"  算子应用 (operator): {operator_time/n_iterations/1000:.2f} ms\n")
        f.write(f"  解码 (decode):      {decode_time/n_iterations/1000:.2f} ms\n")
        f.write(f"  总计:               {total_time/n_iterations/1000:.2f} ms\n")
        f.write(f"\n  吞吐量: {128*n_iterations/(total_time/1e6):.0f} cells/sec\n")

    logger.info(f"✓ 详细报告已保存至: {report_path}")

    trace_path = output_dir / "inference_trace.json"
    prof.export_chrome_trace(str(trace_path))
    logger.info(f"✓ Chrome trace已保存至: {trace_path}")

    analyze_bottlenecks(prof, "推理")


def analyze_bottlenecks(prof, mode_name):
    """分析性能瓶颈并给出优化建议"""
    logger.info("\n" + "="*60)
    logger.info(f"{mode_name}性能瓶颈分析")
    logger.info("="*60)

    events = prof.key_averages()

    # 找出最耗时的操作
    top_events = sorted(events, key=lambda e: e.cpu_time_total, reverse=True)[:10]

    logger.info("\n【Top 10 最耗时操作】")
    for i, event in enumerate(top_events, 1):
        logger.info(f"{i}. {event.key[:50]:<50} {event.cpu_time_total/1000:.2f} ms")

    # 内存使用分析
    memory_events = sorted(events, key=lambda e: e.cpu_memory_usage, reverse=True)[:5]
    logger.info("\n【Top 5 内存占用操作】")
    for i, event in enumerate(memory_events, 1):
        mem_mb = event.cpu_memory_usage / (1024*1024)
        logger.info(f"{i}. {event.key[:50]:<50} {mem_mb:.2f} MB")

    # 优化建议
    logger.info("\n【优化建议】")
    suggestions = []

    # 检查矩阵乘法
    matmul_time = sum(e.cpu_time_total for e in events if "matmul" in e.key.lower() or "mm" in e.key.lower())
    total_time = sum(e.cpu_time_total for e in events)

    if matmul_time / total_time > 0.3:
        suggestions.append("• 矩阵乘法占比超过30%，考虑：")
        suggestions.append("  - 使用更高效的BLAS库（MKL, OpenBLAS）")
        suggestions.append("  - 在GPU上运行（如果当前是CPU）")
        suggestions.append("  - 使用混合精度训练（FP16）")

    # 检查数据移动
    data_movement_time = sum(
        e.cpu_time_total for e in events
        if "to(" in e.key.lower() or "copy" in e.key.lower()
    )
    if data_movement_time / total_time > 0.1:
        suggestions.append("• 数据移动占比超过10%，考虑：")
        suggestions.append("  - 提前将所有数据移到GPU")
        suggestions.append("  - 使用pin_memory=True加速数据传输")
        suggestions.append("  - 减少CPU-GPU之间的数据拷贝")

    # 检查归一化操作
    norm_time = sum(e.cpu_time_total for e in events if "norm" in e.key.lower())
    if norm_time / total_time > 0.15:
        suggestions.append("• 归一化操作占比超过15%，考虑：")
        suggestions.append("  - 使用inplace操作")
        suggestions.append("  - 批归一化替代层归一化")

    if not suggestions:
        suggestions.append("• 性能分布较为均衡，无明显瓶颈")
        suggestions.append("• 可以考虑：")
        suggestions.append("  - 增大batch size充分利用并行")
        suggestions.append("  - 使用torch.compile()进行图优化（PyTorch 2.0+）")

    for suggestion in suggestions:
        logger.info(suggestion)

    logger.info("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="性能分析脚本")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["vae", "operator", "inference", "all"],
        default="all",
        help="分析模式：vae（VAE训练）、operator（Operator训练）、inference（推理）、all（全部）"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="运行设备"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="训练步数（用于profiling）"
    )

    args = parser.parse_args()

    logger.info("虚拟细胞算子模型 - 性能分析工具")
    logger.info(f"设备: {args.device}")
    logger.info(f"模式: {args.mode}")

    if args.mode in ["vae", "all"]:
        profile_vae_training(device=args.device, n_steps=args.steps)

    if args.mode in ["operator", "all"]:
        profile_operator_training(device=args.device, n_steps=args.steps)

    if args.mode in ["inference", "all"]:
        profile_inference(device=args.device, n_iterations=100)

    logger.info("\n✓ 性能分析完成！")
    logger.info("详细报告位于: results/profiling/")


if __name__ == "__main__":
    main()
