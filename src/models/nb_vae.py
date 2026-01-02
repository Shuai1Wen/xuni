# -*- coding: utf-8 -*-
"""
负二项变分自编码器（NB-VAE）

本模块实现潜空间模型，将高维单细胞计数数据映射到低维潜空间。

数学对应关系：
- 编码器：q_φ(z|x,t)，对应 model.md A.2节第38-44行
- 解码器：p_ψ(x|z,t) = ∏_g NB(x_g; μ_g, r_g)，对应 model.md A.2节第46-52行
- ELBO损失：L_embed = 𝔼[log p_ψ(x|z,t)] - KL(q_φ||p)，对应 model.md A.2节第55-65行

关键特性：
- 使用负二项分布建模单细胞RNA-seq的计数数据
- 支持组织条件输入（tissue-specific参数）
- 数值稳定性：所有log计算添加epsilon，softplus输出添加下界
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from ..config import NumericalConfig

# 默认数值配置
_NUM_CFG = NumericalConfig()


class Encoder(nn.Module):
    """
    编码器网络

    实现 q_φ(z|x,t)，将观测x和组织t编码为潜变量z的分布参数。

    数学定义：
        q_φ(z|x,t) ~ N(μ(x,t), diag(σ²(x,t)))
        其中 μ, log(σ²) 由神经网络参数化

    对应：model.md A.2节，第38-44行

    参数:
        n_genes: 基因数量 G
        latent_dim: 潜空间维度 d_z
        n_tissues: 组织类型数量
        hidden_dim: 隐藏层维度

    架构:
        input_layer: x → hidden (G → hidden_dim)
        拼接组织one-hot: [hidden, tissue_onehot]
        fc_mean: → μ (latent_dim)
        fc_logvar: → log(σ²) (latent_dim)

    示例:
        >>> encoder = Encoder(n_genes=2000, latent_dim=32, n_tissues=3)
        >>> x = torch.randn(64, 2000)  # (batch, genes)
        >>> tissue_onehot = torch.zeros(64, 3)
        >>> tissue_onehot[:, 0] = 1  # 第一种组织
        >>> mu, logvar = encoder(x, tissue_onehot)
        >>> print(mu.shape, logvar.shape)
        torch.Size([64, 32]) torch.Size([64, 32])
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int,
        n_tissues: int,
        hidden_dim: int = 512
    ):
        super().__init__()
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        self.n_tissues = n_tissues
        self.hidden_dim = hidden_dim

        # 输入层：基因表达 → 隐藏层
        self.input_layer = nn.Linear(n_genes, hidden_dim)

        # 输出层：[隐藏层 + 组织] → 潜空间参数
        self.fc_mean = nn.Linear(hidden_dim + n_tissues, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim + n_tissues, latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        tissue_onehot: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        参数:
            x: (B, G) 基因表达计数，通常经过log1p和标准化
            tissue_onehot: (B, n_tissues) 组织类型one-hot编码

        返回:
            mu: (B, latent_dim) 潜变量均值
            logvar: (B, latent_dim) 潜变量对数方差

        实现细节:
            1. 非线性变换：x → ReLU(Wx + b)
            2. 拼接组织信息：[h, tissue_onehot]
            3. 输出均值和对数方差（不是方差，用于数值稳定）
        """
        # 编码到隐藏层
        h = F.relu(self.input_layer(x))  # (B, hidden_dim)

        # 拼接组织one-hot
        h_cat = torch.cat([h, tissue_onehot], dim=-1)  # (B, hidden_dim + n_tissues)

        # 输出潜变量分布参数
        mu = self.fc_mean(h_cat)         # (B, latent_dim)
        logvar = self.fc_logvar(h_cat)   # (B, latent_dim)

        return mu, logvar


class DecoderNB(nn.Module):
    """
    负二项解码器

    实现 p_ψ(x|z,t) = ∏_g NB(x_g; μ_g(z,t), r_g(t))

    数学定义：
        μ_g(z,t) = softplus(w_g^T z + b_{g,t}) + ε
        r_g(t) = exp(log_r_g)，基因特异的离散度参数

    对应：model.md A.2节，第46-52行

    参数:
        n_genes: 基因数量 G
        latent_dim: 潜空间维度 d_z
        n_tissues: 组织类型数量
        hidden_dim: 隐藏层维度

    架构:
        fc: [z, tissue_onehot] → hidden
        fc_mu: hidden → μ (n_genes)
        log_dispersion: 可学习参数，shape (n_genes,)

    关键实现:
        - 使用softplus激活函数保证μ > 0
        - 添加epsilon (1e-8) 避免μ=0导致log(0)
        - 离散度参数 r 通过 exp(log_r) 保证 > 0

    示例:
        >>> decoder = DecoderNB(n_genes=2000, latent_dim=32, n_tissues=3)
        >>> z = torch.randn(64, 32)
        >>> tissue_onehot = torch.zeros(64, 3)
        >>> tissue_onehot[:, 0] = 1
        >>> mu, r = decoder(z, tissue_onehot)
        >>> print(mu.shape, r.shape)
        torch.Size([64, 2000]) torch.Size([1, 2000])
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int,
        n_tissues: int,
        hidden_dim: int = 512
    ):
        super().__init__()
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        self.n_tissues = n_tissues
        self.hidden_dim = hidden_dim

        # 解码网络：[z + 组织] → 隐藏层 → 基因表达
        self.fc = nn.Linear(latent_dim + n_tissues, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, n_genes)

        # 基因特异的离散度参数（gene-wise dispersion）
        # 初始化为0，对应r=1（泊松分布的起点）
        self.log_dispersion = nn.Parameter(torch.zeros(n_genes))

    def forward(
        self,
        z: torch.Tensor,
        tissue_onehot: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        参数:
            z: (B, latent_dim) 潜变量
            tissue_onehot: (B, n_tissues) 组织类型one-hot编码

        返回:
            mu: (B, G) 负二项分布的均值参数
            r: (1, G) 负二项分布的离散度参数（基因特异，不依赖样本）

        实现细节:
            1. 拼接z和组织信息
            2. 通过隐藏层
            3. 输出μ = softplus(...) + ε
            4. r = exp(log_r)（基因特异，广播到batch）

        数值稳定性:
            - softplus自然保证输出>0
            - 额外添加1e-8避免极端情况下μ=0
            - r通过exp(log_r)保证>0，log_r可学习
        """
        # 解码到隐藏层
        h = F.relu(self.fc(torch.cat([z, tissue_onehot], dim=-1)))  # (B, hidden_dim)

        # 输出负二项分布的均值参数μ
        # softplus(x) = log(1 + exp(x)) 保证输出>0
        mu = F.softplus(self.fc_mu(h)) + _NUM_CFG.eps_model_output  # (B, G)

        # 离散度参数r（基因特异）
        # shape: (B, G)
        r = torch.exp(self.log_dispersion).unsqueeze(0) + _NUM_CFG.eps_model_output  # (1, G)
        r = r.expand(z.size(0), -1).contiguous()

        return mu, r


def sample_z(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    重参数化采样（reparameterization trick）

    实现：z = μ + σ ⊙ ε，其中 ε ~ N(0,I)

    数学依据：
        如果 z ~ N(μ, σ²)，可以表示为：
        z = μ + σ * ε，ε ~ N(0,1)
        这样梯度可以通过μ和σ反向传播

    参数:
        mu: (B, latent_dim) 均值
        logvar: (B, latent_dim) 对数方差

    返回:
        z: (B, latent_dim) 采样的潜变量

    示例:
        >>> mu = torch.zeros(64, 32)
        >>> logvar = torch.zeros(64, 32)  # log(1) = 0
        >>> z = sample_z(mu, logvar)
        >>> print(z.std())  # 应该接近1
        tensor(1.0123)
    """
    # σ = exp(0.5 * log(σ²)) = exp(log(σ)) = σ
    # 裁剪logvar防止指数溢出：logvar∈[-10,10] → std∈[0.0067, 148.4]
    # 这确保数值稳定性，同时允许足够的方差范围
    logvar = torch.clamp(logvar, min=-10.0, max=10.0)
    std = torch.exp(0.5 * logvar)  # (B, latent_dim)

    # ε ~ N(0,1)
    eps = torch.randn_like(std)    # (B, latent_dim)

    # z = μ + σ ⊙ ε
    return mu + eps * std


def nb_log_likelihood(
    x: torch.Tensor,
    mu: torch.Tensor,
    r: torch.Tensor,
    eps: float = None
) -> torch.Tensor:
    """
    负二项分布的对数似然

    数学定义：
        NB(x; μ, r) = Γ(x+r) / (Γ(r) · x!) · (r/(r+μ))^r · (μ/(r+μ))^x

        log p(x) = log Γ(x+r) - log Γ(r) - log Γ(x+1)
                   + r·log(r/(r+μ)) + x·log(μ/(r+μ))

    对应：model.md A.2节，负二项pmf定义

    参数:
        x: (B, G) 观测计数
        mu: (B, G) 均值参数
        r: (1, G) 或 (B, G) 离散度参数
        eps: 数值稳定性参数

    返回:
        log_p: (B,) 每个样本的对数似然（对基因求和）

    实现细节:
        - 使用torch.lgamma计算log Γ(x)
        - 所有log计算添加epsilon避免log(0)
        - 返回shape (B,) 而非 (B, G)，因为已对基因求和

    数值稳定性:
        - log(r/(r+μ)) = log(r) - log(r+μ)
        - 添加eps避免μ=0时log(0)

    示例:
        >>> x = torch.tensor([[5.0, 10.0]])
        >>> mu = torch.tensor([[5.0, 10.0]])
        >>> r = torch.tensor([[1.0, 1.0]])
        >>> log_p = nb_log_likelihood(x, mu, r)
        >>> print(log_p.shape)
        torch.Size([1])
    """
    # 使用配置的epsilon值
    if eps is None:
        eps = _NUM_CFG.eps_log

    x = x.float()  # 确保为float类型

    # 输入验证：确保参数在有效范围内，防止lgamma产生NaN
    # r必须>0（负二项分布的定义域要求）
    r_input = r
    r = torch.clamp(r_input, min=eps)
    # x必须>=0（计数数据的自然约束）
    x = torch.clamp(x, min=0.0)
    if mu.requires_grad:
        mu.retain_grad()
    if r_input.requires_grad:
        r_input.retain_grad()

    # log Γ(x+r) - log Γ(r) - log Γ(x+1)
    log_coef = (
        torch.lgamma(x + r)
        - torch.lgamma(r)
        - torch.lgamma(x + 1.0)
    )  # (B, G)

    # r·log(r/(r+μ)) + x·log(μ/(r+μ))
    # 为了数值稳定，使用对数减法性质：
    # log(a/b) = log(a) - log(b)
    # 避免直接计算除法和小数
    log_r = torch.log(r + eps)
    log_mu = torch.log(mu + eps)
    log_r_plus_mu = torch.log(r + mu + eps)

    log_r_over_r_plus_mu = log_r - log_r_plus_mu     # (B, G)
    log_mu_over_r_plus_mu = log_mu - log_r_plus_mu   # (B, G)

    log_p = (
        log_coef
        + r * log_r_over_r_plus_mu
        + x * log_mu_over_r_plus_mu
    )  # (B, G)

    # 对基因维度求和，再对batch求均值，返回标量
    return log_p.sum(dim=-1).mean()  # 标量


class NBVAE(nn.Module):
    """
    负二项变分自编码器（完整模型）

    组合Encoder和DecoderNB，实现端到端的VAE。

    模型流程：
        x, tissue → Encoder → (μ_z, logvar_z)
        → sample z = μ_z + σ_z ⊙ ε
        → Decoder → (μ_x, r_x)
        → NB likelihood p(x|μ_x, r_x)

    损失函数：ELBO = 𝔼[log p(x|z)] - KL(q(z|x)||p(z))

    对应：model.md A.2节，完整的潜空间模型

    参数:
        n_genes: 基因数量 G
        latent_dim: 潜空间维度 d_z
        n_tissues: 组织类型数量
        hidden_dim: 隐藏层维度

    示例:
        >>> model = NBVAE(n_genes=2000, latent_dim=32, n_tissues=3)
        >>> x = torch.randn(64, 2000)
        >>> tissue_onehot = torch.zeros(64, 3)
        >>> tissue_onehot[:, 0] = 1
        >>> z, mu_x, r_x, mu_z, logvar_z = model(x, tissue_onehot)
        >>> print(z.shape, mu_x.shape)
        torch.Size([64, 32]) torch.Size([64, 2000])
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int,
        n_tissues: int,
        hidden_dim: int = 512
    ):
        super().__init__()
        # 保存模型配置参数为实例属性
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        self.n_tissues = n_tissues
        self.hidden_dim = hidden_dim
        self.likelihood = "nb"

        # 创建编码器和解码器
        self.encoder = Encoder(n_genes, latent_dim, n_tissues, hidden_dim)
        self.decoder = DecoderNB(n_genes, latent_dim, n_tissues, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        tissue_onehot: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        参数:
            x: (B, G) 基因表达计数
            tissue_onehot: (B, n_tissues) 组织one-hot编码

        返回:
            z: (B, latent_dim) 采样的潜变量
            mu_x: (B, G) 重建的表达均值
            r_x: (1, G) 离散度参数
            mu_z: (B, latent_dim) 潜变量均值（用于计算KL散度）
            logvar_z: (B, latent_dim) 潜变量对数方差（用于计算KL散度）

        流程:
            1. 编码：x → (μ_z, σ_z)
            2. 采样：z ~ N(μ_z, σ_z)
            3. 解码：z → (μ_x, r_x)
        """
        # 编码
        mu_z, logvar_z = self.encoder(x, tissue_onehot)

        # 使用均值作为确定性潜变量
        z = mu_z

        # 解码
        mu_x, r_x = self.decoder(z, tissue_onehot)

        return mu_x, r_x, mu_z, logvar_z

    def sample_z(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """包装重参数化采样，供测试调用"""
        return sample_z(mu, logvar)


class DecoderGaussian(nn.Module):
    """
    高斯解码器

    实现 p_ψ(x|z,t) = N(μ(z,t), diag(σ²))
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int,
        n_tissues: int,
        hidden_dim: int = 512
    ):
        super().__init__()
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        self.n_tissues = n_tissues
        self.hidden_dim = hidden_dim

        self.fc = nn.Linear(latent_dim + n_tissues, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, n_genes)
        self.logvar = nn.Parameter(torch.zeros(n_genes))

    def forward(
        self,
        z: torch.Tensor,
        tissue_onehot: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.fc(torch.cat([z, tissue_onehot], dim=-1)))
        mu = self.fc_mu(h)
        logvar = self.logvar.unsqueeze(0).expand_as(mu)
        return mu, logvar


class GaussianVAE(nn.Module):
    """
    高斯变分自编码器（用于log-normalized数据）
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int,
        n_tissues: int,
        hidden_dim: int = 512
    ):
        super().__init__()
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        self.n_tissues = n_tissues
        self.hidden_dim = hidden_dim
        self.likelihood = "gaussian"

        self.encoder = Encoder(n_genes, latent_dim, n_tissues, hidden_dim)
        self.decoder = DecoderGaussian(n_genes, latent_dim, n_tissues, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        tissue_onehot: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_z, logvar_z = self.encoder(x, tissue_onehot)
        z = sample_z(mu_z, logvar_z)
        mu_x, logvar_x = self.decoder(z, tissue_onehot)
        return z, mu_x, logvar_x, mu_z, logvar_z


def elbo_loss(
    x: torch.Tensor,
    tissue_onehot: torch.Tensor,
    model: NBVAE,
    beta: float = 1.0,
    x_target: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, dict]:
    """
    ELBO损失函数

    数学定义：
        L_ELBO = 𝔼_{q(z|x)}[log p(x|z)] - β·KL(q(z|x)||p(z))
        其中 p(z) = N(0,I) 是标准高斯先验

    对应：model.md A.2节，第55-65行

    参数:
        x: (B, G) 基因表达计数（模型输入）
        tissue_onehot: (B, n_tissues) 组织one-hot编码
        model: NBVAE模型
        beta: KL散度权重（β-VAE），默认1.0
        x_target: 重建目标（默认与x一致，用于去噪训练）

    返回:
        loss: 标量，负ELBO（需要最小化）
        loss_dict: 损失分量字典，包含以下键：
            - "recon_loss": 重建损失（负对数似然）
            - "kl_loss": KL散度
            - "z": 采样的潜变量（detached，用于下游任务）

    ELBO分解：
        - 重建项：log p(x|z) = Σ_g log NB(x_g; μ_g, r_g)
        - KL散度项：KL(q(z|x)||N(0,I))
                  = -0.5 * Σ_d (1 + log σ²_d - μ²_d - σ²_d)

    实现细节：
        - 返回 -ELBO，因为优化器执行最小化
        - loss_dict中的各分量都已detach，用于记录和监控

    示例:
        >>> model = NBVAE(n_genes=2000, latent_dim=32, n_tissues=3)
        >>> x = torch.randn(64, 2000)
        >>> tissue_onehot = torch.zeros(64, 3)
        >>> tissue_onehot[:, 0] = 1
        >>> loss, loss_dict = elbo_loss(x, tissue_onehot, model)
        >>> print(loss.shape, loss_dict.keys())
        torch.Size([]) dict_keys(['recon_loss', 'kl_loss', 'z'])
    """
    # 前向传播
    if x_target is None:
        x_target = x

    mu_x, r_x, mu_z, logvar_z = model(x, tissue_onehot)
    z = sample_z(mu_z, logvar_z)

    # 重建项：log p(x|z)
    log_px = nb_log_likelihood(x_target, mu_x, r_x)  # 标量
    recon_loss = -log_px  # 负对数似然

    # KL散度：KL(q(z|x)||N(0,I))
    # 解析解：-0.5 * Σ (1 + log σ² - μ² - σ²)
    # 裁剪logvar防止指数溢出（与sample_z中的限制一致）
    logvar_z_clamped = torch.clamp(logvar_z, min=-10.0, max=10.0)
    kl = -0.5 * torch.sum(
        1 + logvar_z_clamped - mu_z.pow(2) - logvar_z_clamped.exp(),
        dim=-1
    )  # (B,)
    kl_loss = kl.mean()

    # 总损失：重建损失 + β·KL散度
    loss = recon_loss + beta * kl_loss

    # 返回损失和分量字典
    loss_dict = {
        "recon_loss": recon_loss.detach(),
        "kl_loss": kl_loss.detach(),
        "z": z.detach()  # 用于下游任务
    }

    return loss, loss_dict


def gaussian_elbo_loss(
    x: torch.Tensor,
    tissue_onehot: torch.Tensor,
    model: GaussianVAE,
    beta: float = 1.0
) -> Tuple[torch.Tensor, dict]:
    """
    高斯VAE的ELBO损失
    """
    z, mu_x, logvar_x, mu_z, logvar_z = model(x, tissue_onehot)

    recon = 0.5 * (logvar_x + (x - mu_x) ** 2 / torch.exp(logvar_x + _NUM_CFG.eps_log))
    recon_loss = recon.sum(dim=-1).mean()

    logvar_z_clamped = torch.clamp(logvar_z, min=-10.0, max=10.0)
    kl = -0.5 * torch.sum(
        1 + logvar_z_clamped - mu_z.pow(2) - logvar_z_clamped.exp(),
        dim=-1
    )
    kl_loss = kl.mean()
    loss = recon_loss + beta * kl_loss

    loss_dict = {
        "recon_loss": recon_loss.detach(),
        "kl_loss": kl_loss.detach(),
        "z": z.detach()
    }
    return loss, loss_dict
