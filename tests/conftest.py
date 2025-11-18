# -*- coding: utf-8 -*-
"""
pytest配置文件

全局fixture和配置
"""

import pytest
import torch
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def device():
    """全局设备fixture"""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(autouse=True)
def reset_random_seed():
    """每个测试前重置随机种子"""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


@pytest.fixture
def small_batch():
    """小批次数据fixture"""
    return {
        "batch_size": 16,
        "n_genes": 100,
        "latent_dim": 16,
        "n_tissues": 2
    }


@pytest.fixture
def medium_batch():
    """中等批次数据fixture"""
    return {
        "batch_size": 64,
        "n_genes": 500,
        "latent_dim": 32,
        "n_tissues": 3
    }


def pytest_configure(config):
    """pytest配置"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
