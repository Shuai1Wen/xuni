# -*- coding: utf-8 -*-
"""
Sphinx配置文件 - 虚拟细胞算子模型项目
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# -- 项目信息 --
project = '虚拟细胞算子模型'
copyright = '2025, Virtual Cell Operator Team'
author = 'Virtual Cell Operator Team'

# 版本信息
release = '0.1.0'
version = '0.1'

# -- 通用配置 --

extensions = [
    'sphinx.ext.autodoc',        # 自动从docstring生成文档
    'sphinx.ext.napoleon',       # 支持Google/NumPy风格的docstring
    'sphinx.ext.viewcode',       # 添加源代码链接
    'sphinx.ext.intersphinx',    # 链接到其他项目的文档
    'sphinx.ext.mathjax',        # 数学公式支持
    'sphinx.ext.todo',           # TODO支持
    'sphinx.ext.coverage',       # 文档覆盖率检查
    'sphinx_rtd_theme',          # Read the Docs主题
]

# 模板路径
templates_path = ['_templates']

# 源文件编码
source_encoding = 'utf-8'

# 主文档
master_doc = 'index'

# 语言
language = 'zh_CN'

# 排除的文件模式
exclude_patterns = []

# Pygments语法高亮样式
pygments_style = 'sphinx'

# -- HTML输出选项 --

# HTML主题
html_theme = 'sphinx_rtd_theme'

# 主题选项
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
}

# 静态文件路径
html_static_path = ['_static']

# -- autodoc配置 --

# autodoc选项
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# 自动导入模块
autodoc_mock_imports = ['torch', 'numpy', 'pandas', 'anndata', 'scanpy', 'scipy']

# -- Napoleon配置 (Google/NumPy风格docstring) --

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Intersphinx配置 --

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}

# -- Math配置 --

# MathJax配置（支持中文）
mathjax_config = {
    'TeX': {
        'extensions': ['AMSmath.js', 'AMSsymbols.js', 'noErrors.js', 'noUndefined.js'],
    }
}
