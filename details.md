好，那我们现在就把这个项目“工程化”——给出**完整目录结构 + 每个文件的职责说明 + 一份可以直接用的 README 草稿**。你后面只要按这个骨架往里填代码和数据就行。

下面我会统一用 **英文命名代码 & 文件**，**中文解释**，方便将来投稿和组内协作。

---

## 一、推荐项目目录结构

建议仓库名类似：`virtual-cell-operator-mLOY`

```bash
virtual-cell-operator-mLOY/
├── README.md
├── environment.yml        # 或 requirements.txt
├── setup.py               # 可选，打包用
├── configs/
│   ├── default.yaml       # 通用超参数
│   ├── scperturb.yaml     # scPerturb 实验配置
│   └── mloy_kidney_brain.yaml
├── data/
│   ├── raw/
│   │   ├── scperturb/     # 原始或下载后的 scPerturb 数据（h5ad 等）
│   │   └── mloy/          # 肾脏/脑 mLOY 单细胞数据
│   └── processed/
│       ├── scperturb/
│       └── mloy/
├── scripts/
│   ├── download_scperturb.py    # 可选：自动下载/整理 scPerturb
│   ├── preprocess_scperturb.py  # 预处理到 AnnData
│   ├── preprocess_mloy.py       # 预处理肾脏 + 脑数据
│   ├── train_embed.py           # 训练 NB-VAE
│   ├── train_operator_scperturb.py
│   ├── train_operator_mloy.py
│   ├── eval_scperturb.py        # Benchmark 指标
│   ├── eval_mloy_cross_tissue.py
│   └── run_counterfactuals.py   # 反事实模拟：mLOY 纠正、药物组合
├── src/
│   ├── __init__.py
│   ├── config.py                # dataclass 配置
│   ├── models/
│   │   ├── __init__.py
│   │   ├── nb_vae.py            # Encoder/Decoder + ELBO
│   │   └── operator.py          # OperatorModel
│   ├── data/
│   │   ├── __init__.py
│   │   ├── scperturb_dataset.py
│   │   └── mloy_dataset.py
│   ├── utils/
│   │   ├── edistance.py
│   │   ├── cond_encoder.py
│   │   ├── virtual_cell.py      # encode/decode/apply_operator 接口
│   │   └── logging_utils.py
│   └── train/
│       ├── __init__.py
│       ├── train_embed_core.py      # 通用 embed 训练循环
│       └── train_operator_core.py   # 通用 operator 训练循环
├── notebooks/
│   ├── 00_explore_scperturb.ipynb
│   ├── 01_explore_mloy_kidney.ipynb
│   ├── 02_explore_mloy_brain.ipynb
│   └── 03_visualize_virtual_cells.ipynb
└── results/
    ├── logs/
    ├── checkpoints/
    │   ├── embed/
    │   └── operator/
    └── figures/
        ├── scperturb/
        └── mloy/
```

> 这个结构的逻辑是：
>
> * `src/` 放“可复用核心代码”；
> * `scripts/` 放“具体任务脚本”（读配置 → 调用 `src`）；
> * `data/`、`results/`、`notebooks/` 分别对应数据、模型结果、可视化/探索。

---

## 二、环境配置骨架（environment.yml）

你可以用 conda 管理，示例：

```yaml
name: vcell-operator
channels:
  - conda-forge
  - pytorch
  - nvidia
dependencies:
  - python=3.10
  - pytorch
  - pytorch-cuda=12.1  # 如果有 GPU
  - cudatoolkit
  - numpy
  - pandas
  - scikit-learn
  - anndata
  - scanpy
  - matplotlib
  - seaborn
  - pip
  - pip:
      - pyyaml
      - tqdm
      - wandb           # 可选，用于日志
      - rich            # 可选，美化 log
```

你也可以改成 `requirements.txt`，内容类似。

---

## 三、README 草稿（可以直接放到 README.md 里再润色）

下面给你一版 **英文 README 草稿**，方便后续开源或投稿附录使用；你可以再加一份中文版本。

````markdown
# Virtual Cell Operator Model for Cross-Tissue mLOY Effects

This repository implements a mathematically grounded **operator-based virtual cell model** for single-cell perturbation data and cross-tissue analysis of **mosaic loss of chromosome Y (mLOY)**.

The core idea is to represent "virtual cells" not as a single black-box function, but as a **family of linear operators** acting in a shared latent space:
\[
K_\theta: \mathbb{R}^{d_z} \to \mathbb{R}^{d_z},
\]
where each operator encodes the expected state transition of a cell under a given **condition** \(\theta = (\text{perturbation}, \text{tissue}, \text{mLOY state}, \ldots)\).
These operators are **factorized** into a small number of global response bases, allowing us to:

- Generalize across **perturbations, cell types, and tissues**;
- Quantify cross-tissue **mLOY response axes** (e.g., blood–kidney–brain);
- Perform **counterfactual simulations** (e.g., "mLOY correction", drug + mLOY combinations),
  while keeping the model lightweight and numerically stable.

---

## 1. Repository structure

```bash
virtual-cell-operator-mLOY/
├── README.md
├── environment.yml
├── configs/
│   ├── default.yaml
│   ├── scperturb.yaml
│   └── mloy_kidney_brain.yaml
├── data/
│   ├── raw/
│   │   ├── scperturb/
│   │   └── mloy/
│   └── processed/
│       ├── scperturb/
│       └── mloy/
├── scripts/
│   ├── download_scperturb.py
│   ├── preprocess_scperturb.py
│   ├── preprocess_mloy.py
│   ├── train_embed.py
│   ├── train_operator_scperturb.py
│   ├── train_operator_mloy.py
│   ├── eval_scperturb.py
│   ├── eval_mloy_cross_tissue.py
│   └── run_counterfactuals.py
├── src/
│   ├── config.py
│   ├── models/
│   │   ├── nb_vae.py
│   │   └── operator.py
│   ├── data/
│   │   ├── scperturb_dataset.py
│   │   └── mloy_dataset.py
│   ├── utils/
│   │   ├── edistance.py
│   │   ├── cond_encoder.py
│   │   ├── virtual_cell.py
│   │   └── logging_utils.py
│   └── train/
│       ├── train_embed_core.py
│       └── train_operator_core.py
├── notebooks/
│   ├── 00_explore_scperturb.ipynb
│   ├── 01_explore_mloy_kidney.ipynb
│   ├── 02_explore_mloy_brain.ipynb
│   └── 03_visualize_virtual_cells.ipynb
└── results/
    ├── logs/
    ├── checkpoints/
    │   ├── embed/
    │   └── operator/
    └── figures/
        ├── scperturb/
        └── mloy/
````

---

## 2. Installation

```bash
git clone <this-repo-url>
cd virtual-cell-operator-mLOY

# using conda
conda env create -f environment.yml
conda activate vcell-operator
```

---

## 3. Data preparation

### 3.1 scPerturb data

We use a subset of the **scPerturb** collection of harmonized single-cell perturbation datasets (CRISPR, drugs, etc.).
You have two options:

1. **Manual download** following the instructions from the scPerturb resource (e.g., h5ad files).
2. **Semi-automatic download** using `scripts/download_scperturb.py` (if implemented).

Then preprocess:

```bash
python scripts/preprocess_scperturb.py \
  --input data/raw/scperturb/ \
  --output data/processed/scperturb/ \
  --min_cells 2000 \
  --min_genes 1000
```

The script should:

* Perform basic QC and gene selection,
* Annotate `obs` fields such as `tissue`, `cell_type`, `perturbation`, `timepoint`, `batch`,
* Save a unified `AnnData` object (e.g., `scperturb_merged.h5ad`).

### 3.2 mLOY kidney and brain datasets

We use public single-cell / single-nucleus datasets that report **mosaic loss of Y (mLOY)** in kidney and brain (e.g., microglia):

* Kidney: single-cell RNA/ATAC data quantifying LOY across nephron segments and immune cells in aging and CKD.
* Brain: single-nucleus data reporting LOY in microglia in aged / neurodegenerative conditions.

You need to download the raw or processed matrices (e.g., from GEO, ArrayExpress, or provided portals),
and then run:

```bash
python scripts/preprocess_mloy.py \
  --input data/raw/mloy/ \
  --output data/processed/mloy/
```

The script should:

* Harmonize gene IDs;
* Annotate `obs["tissue"]` ∈ {"kidney", "brain"};
* Annotate `obs["mLOY_prob"]` (cell-level) and `obs["mLOY_load"]` (donor-level);
* Annotate `obs["cell_type"]`, `obs["batch"]`, etc.

---

## 4. Model overview

We first train a **shared NB-VAE** to embed all cells into a latent space (z \in \mathbb{R}^{d_z}),
then learn a **family of linear operators** (K_\theta(z) = A_\theta z + b_\theta)
that capture how cells move in this latent space under perturbations, mLOY, and across tissues.

Formally:

* Encoder: (q_\phi(z\mid x,t))
* Decoder: (p_\psi(x\mid z,t)) with a negative-binomial likelihood
* Operator:
  [
  A_\theta = A_{t}^{(0)} + \sum_{k=1}^K \alpha_k(\theta) B_k, \quad
  b_\theta = b_{t}^{(0)} + \sum_{k=1}^K \beta_k(\theta) u_k
  ]
* Condition vector (\theta) encodes:

  * perturbation (drug / KO / "LOY"),
  * tissue (blood / kidney / brain),
  * mLOY state (cell-level or donor-level),
  * batch, age, disease, etc.

Operators are trained to minimize an **energy distance** between predicted and observed endpoint distributions in latent space, plus a **spectral stability penalty** to avoid exploding trajectories.

---

## 5. Training pipeline

### 5.1 Train the NB-VAE (embedding)

```bash
python scripts/train_embed.py \
  --config configs/scperturb.yaml \
  --adata data/processed/scperturb/scperturb_merged.h5ad \
  --output results/checkpoints/embed/
```

This script will:

* Load the processed AnnData;
* Train the NB-VAE (`src/models/nb_vae.py`) using ELBO loss;
* Save encoder/decoder weights.

### 5.2 Train the operator on scPerturb

```bash
python scripts/train_operator_scperturb.py \
  --config configs/scperturb.yaml \
  --adata data/processed/scperturb/scperturb_merged.h5ad \
  --embed_ckpt results/checkpoints/embed/best_model.pt \
  --output results/checkpoints/operator_scperturb/
```

This script will:

* Build pairwise datasets `(x0, x1)` for each `(dataset, tissue, cell_type, perturbation)` with timepoints (t0 → t1);
* Encode them to latent space;
* Train the operator model (`src/models/operator.py`) using:

  * Energy distance between predicted and true endpoints;
  * Spectral penalty for stability.

### 5.3 Train the operator on mLOY kidney + brain

```bash
python scripts/train_operator_mloy.py \
  --config configs/mloy_kidney_brain.yaml \
  --adata data/processed/mloy/mloy_merged.h5ad \
  --embed_ckpt results/checkpoints/embed/best_model.pt \
  --output results/checkpoints/operator_mloy/
```

This script treats LOY vs XY within each tissue / cell type as a "pseudo-perturbation"
and fits corresponding operators, sharing response bases with the scPerturb operator.

---

## 6. Evaluation and analysis

### 6.1 scPerturb benchmark

```bash
python scripts/eval_scperturb.py \
  --config configs/scperturb.yaml \
  --adata data/processed/scperturb/scperturb_merged.h5ad \
  --embed_ckpt results/checkpoints/embed/best_model.pt \
  --operator_ckpt results/checkpoints/operator_scperturb/best_model.pt \
  --output results/figures/scperturb/
```

This script evaluates:

* Energy distance (E-distance) on held-out perturbations / cell types;
* Gene-level DE prediction (AUROC / auPR);
* UMAP visualization of real vs. virtual perturbed cells.

### 6.2 Cross-tissue mLOY axes

```bash
python scripts/eval_mloy_cross_tissue.py \
  --config configs/mloy_kidney_brain.yaml \
  --adata data/processed/mloy/mloy_merged.h5ad \
  --embed_ckpt results/checkpoints/embed/best_model.pt \
  --operator_ckpt results/checkpoints/operator_mloy/best_model.pt \
  --output results/figures/mloy/
```

This script:

* Identifies global response bases (axes) most strongly activated by mLOY;
* Quantifies their loading across tissues (kidney vs brain) and cell types;
* Performs gene/pathway enrichment analysis for each axis.

---

## 7. Counterfactual simulations

### 7.1 mLOY "correction"

```bash
python scripts/run_counterfactuals.py \
  --mode mloy_correction \
  --adata data/processed/mloy/mloy_merged.h5ad \
  --embed_ckpt results/checkpoints/embed/best_model.pt \
  --operator_ckpt results/checkpoints/operator_mloy/best_model.pt \
  --output results/figures/mloy_correction/
```

This will:

* Take LOY cells as input;
* Construct a "virtual XY" condition (setting mLOY component to zero);
* Apply the corresponding operator to map LOY cells back toward XY-like states in latent space;
* Decode to gene space and compare with true XY cells at the level of distributions, DE genes, and pathways.

### 7.2 mLOY + drug combination

```bash
python scripts/run_counterfactuals.py \
  --mode mloy_drug_combo \
  --adata_scperturb data/processed/scperturb/scperturb_merged.h5ad \
  --adata_mloy data/processed/mloy/mloy_merged.h5ad \
  --embed_ckpt results/checkpoints/embed/best_model.pt \
  --operator_ckpt results/checkpoints/operator_combined.pt \
  --output results/figures/mloy_drug_combo/
```

This will:

* Identify perturbations whose response profiles are similar or opposite to mLOY axes;
* Simulate virtual cells under mLOY + drug combinations;
* Visualize and quantify "corrective" vs "synergistic" effects at the pathway level.

---

## 8. Reproducibility

* Random seeds are controlled at the level of:
。
  * numpy / torch / dataloader;
* All hyperparameters are stored in YAML under `configs/`;
* Checkpoints and logs are stored in `results/checkpoints/` and `results/logs/`;
* Main figures can be reproduced by:

  * Rerunning `eval_scperturb.py`, `eval_mloy_cross_tissue.py`,
  * And visualizing with notebooks under `notebooks/`.

---

## 9. License and citation

(Here you can add license information and a citation once the manuscript is available.)

```

---

## 三、接下来你可以怎么用这些骨架？

1. **先建仓库 + 目录**：可以直接复制上面的树，用 `mkdir` 和空文件先搭起来；
2. **把我之前给你的核心代码**分别塞进：
   - `src/models/nb_vae.py`
   - `src/models/operator.py`
   - `src/utils/edistance.py`
   - `src/utils/cond_encoder.py`
   - `src/utils/virtual_cell.py`
   - `src/train/train_embed_core.py`
   - `src/train/train_operator_core.py`
3. **在 scripts 里写轻量的 main**：  
   每个脚本就是：解析命令行参数 → 读 config → 调用 `src/train/...` 或 `src/utils/...`；
4. **用 notebooks 做最初的 debug & 可视化**：先在小 subset 上跑通流程。

```
