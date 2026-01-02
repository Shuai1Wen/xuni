# -*- coding: utf-8 -*-
"""
扰动标签标准化工具

用于统一control相关别名，避免训练与评测阶段因标签不一致导致的逻辑偏差。
"""

from typing import Iterable


DEFAULT_CONTROL_ALIASES = {
    "control",
    "ctrl",
    "vehicle",
    "untreated",
    "baseline",
    "dmso",
    "mock",
    "negative_control",
    "neg_control",
    "wt",
    "wildtype",
}


def normalize_perturbation_label(
    label: str,
    control_aliases: Iterable[str] = DEFAULT_CONTROL_ALIASES,
    control_token: str = "control"
) -> str:
    """
    标准化扰动标签

    参数:
        label: 原始扰动标签
        control_aliases: control别名集合
        control_token: 标准control标签

    返回:
        标准化后的扰动标签
    """
    if label is None:
        return control_token
    normalized = str(label).strip()
    if normalized == "":
        return control_token
    if normalized.lower() in {alias.lower() for alias in control_aliases}:
        return control_token
    return normalized
