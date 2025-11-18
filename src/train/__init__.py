# -*- coding: utf-8 -*-
"""
训练模块

本模块提供VAE和算子模型的训练循环。
"""

from .train_embed_core import train_embedding
from .train_operator_core import train_operator

__all__ = [
    "train_embedding",
    "train_operator",
]
