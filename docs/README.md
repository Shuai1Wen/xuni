# 虚拟细胞算子模型 - 文档

本目录包含项目的完整Sphinx文档。

## 快速开始

### 安装Sphinx

```bash
pip install sphinx sphinx_rtd_theme
```

### 构建文档

```bash
cd docs
make html
```

文档将生成在 `build/html/` 目录下。

### 查看文档

```bash
make serve
```

然后在浏览器中访问 http://localhost:8000

## 文档结构

```
docs/
├── source/
│   ├── index.rst              # 主页
│   ├── quick_start.rst        # 快速开始
│   ├── api/                   # API参考
│   │   ├── index.rst
│   │   ├── models.rst
│   │   ├── utils.rst
│   │   ├── data.rst
│   │   └── train.rst
│   ├── tutorials/             # 教程
│   │   └── index.rst
│   ├── conf.py               # Sphinx配置
│   └── _static/              # 静态文件
├── build/                    # 构建输出
└── Makefile                  # 构建脚本
```

## 文档类型

- **快速开始**: 5分钟上手指南
- **API参考**: 完整的模块、类、函数文档
- **教程**: 从基础到高级的详细教程
- **数学基础**: 模型的数学原理

## 更新文档

当修改了源代码中的docstring后，重新构建文档：

```bash
make clean  # 清理旧文档
make html   # 重新构建
```

## 文档规范

### Docstring格式

使用Google风格的docstring：

```python
def function_name(arg1: int, arg2: str) -> bool:
    """
    一句话简短描述

    详细描述内容，可以多行。

    参数:
        arg1: 第一个参数的描述
        arg2: 第二个参数的描述

    返回:
        返回值的描述

    示例:
        >>> result = function_name(1, "test")
        >>> print(result)
        True

    注意:
        特殊注意事项

    引发:
        ValueError: 错误条件描述
    """
```

### 数学公式

在文档中使用LaTeX格式的数学公式：

```rst
.. math::

   \\mathcal{L} = \\mathbb{E}_{q(z|x)}[\\log p(x|z)] - \\beta D_{KL}(q(z|x) || p(z))
```

### 代码示例

使用代码块：

```rst
.. code-block:: python

   from src.models.nb_vae import NBVAE

   model = NBVAE(n_genes=2000, latent_dim=32)
```

## 贡献文档

欢迎贡献文档！请遵循以下步骤：

1. 在 `source/` 下创建或修改 `.rst` 文件
2. 在docstring中添加详细说明
3. 构建并检查文档
4. 提交Pull Request

## 自动生成

自动生成API文档：

```bash
sphinx-apidoc -o source/api ../src
```

## 在线文档

文档托管在：
- Read the Docs: https://virtual-cell-operator.readthedocs.io (待配置)
- GitHub Pages: https://your-username.github.io/virtual-cell-operator (待配置)

## 问题反馈

如果发现文档问题，请在GitHub Issues中报告：
https://github.com/your-repo/virtual-cell-operator/issues
