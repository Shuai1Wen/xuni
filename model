好，那我们现在认真把这个方案“落地成数学 + 实验设计”，你后面几乎可以直接往 Methods 里搬。下面分两大部分：

* **A. 严格数学符号的 Methods 草稿（模型部分）**
* **B. 实验路线图：scPerturb → mLOY 肾脏 + 脑 → 反事实预测 & 生物学验证**

---

# A. Methods 草稿（数学形式）

下面假设你会在正式稿里用英文写 Methods，这里我用中英文混排 + 数学符号，方便你直接改写。

## A.1 数据与符号约定

* 细胞水平观测：
  对任意单细胞或单核记为
  [
  x \in \mathbb{R}^G_{\ge 0}
  ]
  是长度为 (G) 的计数向量（UMI counts），服从**过离散、零膨胀**的分布。

* 每个细胞附带元信息：

  * 组织 (t \in \mathcal{T})（如 blood, kidney, brain）；
  * 扰动 (p \in \mathcal{P})（CRISPR KO、药物、mLOY vs XY 等）；
  * 时间索引 (s\in{0,1})（例如 (s=0) 为对照或 (t=0)，(s=1) 为 (t=T) snapshot）；
  * 可能的额外协变量 (c)（年龄、疾病状态、批次等）。

* **条件向量**
  把所有条件编码到一个向量
  [
  \theta = \theta(p,t,m,c,\Delta t) \in \mathbb{R}^{d_\theta},
  ]
  其中 (m) 表示 **mLOY 状态**（可以是 cell-level 概率 (\in[0,1]) 或 donor-level 负荷），(\Delta t) 是从 (s=0) 到 (s=1) 的时间间隔（对大多数 scPerturb 数据可视为 1）。

## A.2 低维潜空间与计数观测模型

我们采用一个**负二项 VAE / 因子模型**把高维计数映射到低维潜空间：

* 编码器
  [
  q_\phi(z\mid x,t) \quad\text{with}\quad z\in\mathbb{R}^{d_z}.
  ]
  其中 (d_z \ll G)，(q_\phi) 可以是高斯或其它简单分布；
  编码器参数 (\phi) 可以依赖组织 (t)（如加入 one-hot）。

* 解码器（计数观测）
  给定潜变量 (z)，生成表达
  [
  p_\psi(x\mid z,t) = \prod_{g=1}^G \text{NB}\Big(x_g;,\mu_g(z,t), r_g(t)\Big),
  ]
  其中 (\mu_g(z,t)=\mathrm{softplus}(w_g^\top z + b_{g,t}))，(r_g(t)) 为离散度参数。

> 这里可以直接引用 scVI/总VI 的 NB 设定，不需要大网络，只要两层 MLP 或线性层即可：这保证了**针对离散、零膨胀**数据的合理性。

* **潜空间训练目标**
  在所有细胞上最大化 ELBO：

  [
  \mathcal{L}*{\text{embed}}(\phi,\psi)=
  \sum*{i} \mathbb{E}*{q*\phi(z_i\mid x_i,t_i)}
  \left[\log p_\psi(x_i\mid z_i,t_i)\right]

  * \mathrm{KL}\left(q_\phi(z_i\mid x_i,t_i),|,p(z)\right),
    ]
    其中 (p(z)=\mathcal{N}(0,I))。

训练完成后记编码出的潜空间表示为
[
z_i = \mathbb{E}*{q*\phi(z\mid x_i,t_i)}[z].
]

## A.3 扰动响应算子族 (K_\theta)

**核心定义：**
在潜空间 (\mathbb{R}^{d_z}) 上，定义一族**仿射算子**（离散时间）：

[
K_\theta(z) = A_\theta z + b_\theta,\quad A_\theta\in\mathbb{R}^{d_z\times d_z},\ b_\theta\in\mathbb{R}^{d_z}.
]

含义：在条件 (\theta) 下，从时间 (s=0) 的潜状态 (z^{(0)}) 演化一步到近似终点状态 (z^{(1)})：

[
z^{(1)}\approx K_\theta(z^{(0)}).
]

若需要连续时间视角，可以把 (A_\theta) 理解为近似指数映射：

[
A_\theta \approx e^{\Delta t L_\theta},\quad L_\theta\ \text{为生成元（generator）},
]
但在实现中我们只显式拟合 (A_\theta,b_\theta)（离散时间）。

---

## A.4 局部算子估计：基于 E-distance 的分布拟合

对每个有配对时间点的条件 (\theta)，例如 scPerturb 中某个数据集、某个细胞类型 c、某个扰动 p：

* 取对照（或 (t=0)）细胞集合的潜表示：
  [
  Z^{(0)}_\theta = {z^{(0)}*i}*{i=1}^{n_0}
  ]
* 取扰动终点（或 (t=T)）细胞集合的潜表示：
  [
  Z^{(1)}_\theta = {z^{(1)}*j}*{j=1}^{n_1}
  ]

为了在**分布层面**拟合算子，我们使用 scPerturb 中提出的 **能量距离（E-distance）** 作为高维分布间的度量：

给定两组样本 (X={x_i}*{i=1}^n)、(Y={y_j}*{j=1}^m)，E-distance 的无偏估计为：

[
\widehat{\mathcal{E}}^2(X,Y)
= \frac{2}{nm} \sum_{i=1}^n\sum_{j=1}^m |x_i-y_j|_2

* \frac{1}{n^2}\sum_{i=1}^n\sum_{i'=1}^n |x_i-x_{i'}|_2
* \frac{1}{m^2}\sum_{j=1}^m\sum_{j'=1}^m |y_j-y_{j'}|_2.
  ]

我们令

* 真实终点分布：(P^{(1)}*\theta \approx Z^{(1)}*\theta)
* 预测终点分布：(\tilde{P}^{(1)}*\theta \approx {A*\theta z^{(0)}*i + b*\theta}_i)

**局部算子拟合目标：**

[
\mathcal{L}*{\text{local}}(\theta;A*\theta,b_\theta)
= \widehat{\mathcal{E}}^2\left( {A_\theta z^{(0)}*i + b*\theta}*i,\ Z^{(1)}*\theta\right).
]

在最简单的两阶段方案中，可以先**对每个 (\theta) 独立**拟合一个局部算子 (\tilde{A}*\theta,\tilde{b}*\theta)，作为后续低秩分解的“数据”。

> 注意：这里完全不需要 OT/匹配每个细胞，只是在潜空间里做分布对齐，计算复杂度在可接受范围内，并且与 scPerturb 基准保持一致。

---

## A.5 低秩分解：全局响应基 (B_k) 与系数 (\alpha_k(\theta))

我们不希望为每个 (\theta) 存一套高维矩阵，而是引入低秩和**共享响应基**。

### A.5.1 线性部分的低秩结构

假设每个线性部分 (A_\theta) 可以分解为：

[
A_\theta = A^{(0)}_{t(\theta)}

* \sum_{k=1}^K \alpha_k(\theta), B_k,
  ]

其中

* (A^{(0)}_{t})：组织 (t) 的**基线算子**（例如接近单位阵，表示无扰动时的自然演化）；
* (B_k\in\mathbb{R}^{d_z\times d_z})：**全局共享的响应基算子**（“虚拟细胞响应轴”）；
* (\alpha_k(\theta)\in\mathbb{R})：在条件 (\theta) 下第 k 个响应轴的激活强度。

对所有局部估计 (\tilde{A}_\theta)，我们拟合 ({A^{(0)}_t},{B_k},{\alpha_k(\theta)}) 以近似：

[
\min_{{A^{(0)}*t},{B_k},{\alpha_k}}
\sum*{\theta\in\Theta_{\text{train}}}
\left| \tilde{A}*\theta - A^{(0)}*{t(\theta)} - \sum_{k=1}^K \alpha_k(\theta),B_k \right|_F^2

* \lambda_{\text{reg}}\left(\sum_t|A^{(0)}_t|_F^2+\sum_k|B_k|_F^2\right).
  ]

这一步本质上是一个**矩阵族的低秩因子分析**。

### A.5.2 系数 (\alpha_k(\theta)) 的参数化（含 mLOY）

为了泛化到新扰动/新组织/新 mLOY 状态，我们不直接存储每个 (\theta) 的系数，而是用一个小函数族 (g_k)：

[
\alpha_k(\theta) = g_k(\theta;,\omega),\quad k=1,\dots,K.
]

* (\theta) 的输入维度包括：

  * 扰动 p 的 embedding（如 one-hot + learned embedding）；
  * 组织 t 的 embedding；
  * mLOY 相关变量：

    * donor-level LOY 负荷 (L)（例如 kidney/brain LOY 研究中的个体级 mLOY 比例）
    * cell-level LOY 概率 (P(\text{LOY}\mid x))（来自 Y 染色体表达/CNV 推断）；
  * 其它 covariate：年龄、疾病状态、数据集 ID 等。

* (g_k) 可以是简单的 MLP 或甚至线性模型（根据算力决定），(\omega) 为全部 (g_k) 的参数。

这样，当我们在新的 (p,t,m) 上做预测时，只要能构造出 (\theta)，就可以生成对应的 (A_\theta)。

### A.5.3 平移项 (b_\theta)

平移项也可用类似结构建模：

[
b_\theta = b^{(0)}*{t(\theta)} + \sum*{k=1}^K \beta_k(\theta),u_k,
]
其中 (u_k\in\mathbb{R}^{d_z}) 为共享平移基，(\beta_k(\theta)) 用类似 (g_k) 的小网络参数化。

---

## A.6 端到端训练目标

在实践中，可以采用**两阶段 + 端到端微调**：

1. Phase 1：训练 encoder/decoder（(\phi,\psi)），得到 (z_i)；
2. Phase 2：估计局部 (\tilde{A}*\theta,\tilde{b}*\theta)；
3. Phase 3：用低秩结构初始化 (A^{(0)}_t,B_k,g_k)，然后在 E-distance 损失下**端到端微调**算子族。

综合 loss 可写为：

[
\mathcal{L} =
\underbrace{\mathcal{L}*{\text{embed}}}*{\text{潜空间重建}}

* \lambda_1 \underbrace{\sum_{\theta\in\Theta_{\text{train}}}
  \widehat{\mathcal{E}}^2\Big({K_\theta(z_i^{(0)})}*i,\ Z^{(1)}*\theta\Big)}_{\text{分布级算子拟合}}
* \lambda_2 \underbrace{\sum_{\theta}
  \left| K_\theta - \tilde{K}*\theta\right|*{\text{approx}}^2}_{\text{低秩近似约束（可选）}}
* \lambda_3 \mathcal{R}*{\text{stab}},
  ]
  其中 (\mathcal{R}*{\text{stab}}) 是稳定性正则（见下一节）。

---

## A.7 稳定性约束（谱半径/生成元）

你前面特别强调**数值稳定性**，尤其是多步合成时不希望潜空间发散。我们可以用两类约束：

### A.7.1 谱半径/谱范数约束（离散时间）

对每个 (A_\theta) 令 (\rho(A_\theta)) 表示其谱半径（最大特征值模）：

* 硬约束：强制 (\rho(A_\theta)\le \rho_{\max}\approx 1)
* 实现上更常见的是**软约束**（惩罚项）：

  [
  \mathcal{R}*{\text{stab}}^{(1)} =
  \sum*{\theta\in\Theta_{\text{train}}}
  \max\big(0,,\rho(A_\theta)-\rho_0\big)^2,
  ]
  例如 (\rho_0=1) 或稍小。

在实际优化时可以用谱范数（(|A_\theta|_2)）作为 upper bound，使用 power iteration 近似它。

### A.7.2 生成元实部约束（连续时间视角，可选）

若希望解释为连续时间生成元 (L_\theta)，我们可以通过矩阵对数近似：

[
L_\theta \approx \frac{1}{\Delta t}\log(A_\theta).
]

并对 (L_\theta) 的特征值实部施加非正约束，保证系统是**非发散/耗散**的：

[
\mathcal{R}^{(2)}*{\text{stab}} =
\sum*{\theta} \sum_{\lambda\in\mathrm{spec}(L_\theta)}
\max\big(0,\ \mathrm{Re}(\lambda)\big)^2.
]

在实现中可以简化为对对称部分 ((L_\theta + L_\theta^\top)/2) 的最大特征值加惩罚，而不必显式求矩阵对数，以降低计算开销。

---

## A.8 预测与反事实模拟

训练好的模型允许我们在以下场景下生成“虚拟细胞”：

1. **单步预测**
   给定初始潜状态 (z_0) 和条件 (\theta)，
   [
   z_1 = K_\theta(z_0),\quad x_1\sim p_\psi(x\mid z_1,t).
   ]

2. **多步合成/组合扰动**
   例如先施加 mLOY 状态，再施加药物 p：
   [
   z_1 = K_{\theta_{\mathrm{mLOY}}}(z_0),\quad
   z_2 = K_{\theta_{\mathrm{drug+mLOY}}}(z_1),
   ]
   其中 (\theta_{\mathrm{drug+mLOY}}) 包含 mLOY + 药物；
   或者比较两种顺序对应的 (z_2)，做反事实分析。

3. **跨组织虚拟细胞**
   对给定“虚拟系统状态”（例如某个 (\theta) 只差组织变量），
   对 blood/kidney/brain 分别构造
   [
   z^{\text{blood}}*1 = K*{\theta_{\text{blood}}}(z^{\text{blood}}*0),\quad
   z^{\text{kidney}}*1 = K*{\theta*{\text{kidney}}}(z^{\text{kidney}}*0),\quad
   z^{\text{brain}}*1 = K*{\theta*{\text{brain}}}(z^{\text{brain}}_0),
   ]
   再解码得到三个组织的虚拟细胞群，用于比较 mLOY 或药物在不同组织的效应。

---

# B. 实验路线图（完整方案）

下面是一个可实施的路线，分三阶段：

1. **scPerturb 上的小规模 proof-of-concept（方法正确性 & baseline 对比）**
2. **mLOY 肾脏 + 脑上的跨组织分析（展示虚拟细胞 + mLOY 轴）**
3. **反事实预测 + 生物学验证场景设计（支撑论文亮点）**

---

## B.1 Phase I：scPerturb 上的 proof-of-concept

### B.1.1 数据选择与预处理

* 使用 **scPerturb 资源**中 1–3 个经典 RNA-seq 数据集：例如包含 CRISPR KO + 小分子药物、具有 t=0 和 t=T 两个时间点的实验。
* 处理步骤：

  1. 标准 QC（细胞过滤、基因过滤、去除高线粒体比例细胞）；
  2. 选择高变基因集（如 2k–3k）；
  3. 每个细胞保留：扰动 ID、细胞类型/cluster、时间点。
* 对所有选定数据集联合训练 encoder/decoder，得到共享潜空间 (z)。

### B.1.2 算子拟合与低秩分解

1. **局部算子估计**：

   * 对每个 (dataset, 细胞类型 c, 扰动 p) 组合，构建条件 (\theta)，用 E-distance 最小化得到 (\tilde{A}*\theta,\tilde{b}*\theta)。

2. **低秩分解与 (\alpha_k(\theta))**：

   * 选一个小的 K（如 3–5），对所有 (\tilde{A}_\theta) 做最小二乘 + 正则的低秩拟合，得到初始 (A^{(0)}_t,B_k,\alpha_k(\theta))；
   * 再用小 MLP 把 (\alpha_k(\theta)) 参数化为 (g_k(\theta;\omega))，端到端微调。

3. **稳定性正则**：

   * 在训练过程中加入谱范数惩罚，确保 (\rho(A_\theta)) 不超过一个阈值，为后续多步组合做准备。

### B.1.3 评价指标与 baseline

**任务设计：**

* 任务 1：in-distribution 预测（同细胞类型、同实验系统，预留一部分扰动做测试）；
* 任务 2：跨细胞类型/上下文泛化：在训练中不使用某些细胞类型对应的 ((p,c)) 组合，测试时用这些组合评估；
* 任务 3：零样本扰动组合：训练只见到单扰动 A、B，测试预测组合 A+B。

**评价指标：**

* **E-distance**：对每个 (p,c) 在潜空间上比较预测分布与真实分布的 E-distance（越小越好）；
* **基因层面指标**：

  * DE genes 的 AUROC / auPR（预测 vs 真实差异）；
  * pathway enrichment 一致性（比如对 MSigDB 或 KEGG 通路做富集，两边排名的相关性）；
* **细胞状态层面**：在嵌入空间中，对真实 vs 虚拟扰动细胞混合 UMAP，观察 cluster overlap。

**Baselines：**

* 简单线性模型：对每个基因做 linear regression / Ridge，预测扰动下表达变化；
* scGen / CPA 这类基于 VAE 的“风格迁移”方法（如果算力允许，至少在小数据集上做对比）；
* 如有精力，可选一个 diffusion/SB 模型（例如 DC-DSB）在小规模上对比。

目标不是完全打败所有 baseline，而是证明：
**在相似参数规模下，算子模型在“跨扰动组合 / 跨细胞类型泛化”的任务上有优势，并且结构更清晰。**

---

## B.2 Phase II：mLOY 肾脏 + 脑跨组织分析

### B.2.1 数据集选择

**肾脏 mLOY：**

* 使用 Genome Biology 2024 的研究：在慢性肾病患者肾脏单细胞 RNA + ATAC 数据中，系统分析了 LOY 在不同肾单位位置的分布及其随年龄、病程变化。
* GEO/OmicsDI 中有对应的 scRNA/snATAC 数据（如 GSE232222 等）。

**脑 microglia LOY / AD 相关：**

* Genome Research “Mosaic loss of Chromosome Y in aged human microglia” + 相关 biorxiv/AD 风险研究，提供了 microglia 中 LOY 的单核转录组证据；
* 可以结合更大的微胶质细胞 atlas（如 Cell 上人类 AD 进展中的 microglia dynamics，或 Nature Communications 的 microglia aging/AD atlas）做背景参照。

### B.2.2 预处理与潜空间对齐

* 将肾脏 mLOY 数据和脑 microglia 数据与 Phase I 的潜空间模型联合/迁移学习：

  * 共享一部分编码器参数（或通过 scArches 式的 adapter 迁移），确保不同组织的细胞可以嵌入到**同一个潜空间**；
  * 用组织标签 (t) 作为 encoder 的条件输入，允许组织特异性偏移。

* 对每个细胞估计 LOY 状态：

  * 根据肾脏/脑研究中的方法，用 Y 染色体基因表达 + scATAC Y 峰 + CNV 来推断 LOY 概率；
  * 得到 (P(\text{LOY}\mid x)\in[0,1])，作为 mLOY cell-level 变量。

### B.2.3 把 LOY 视作“伪扰动”，拟合算子

* 在肾脏数据中：
  对每个主要细胞类型（近端小管上皮、集合管、各类免疫细胞等），构造：
  [
  \theta = (\text{p = LOY vs XY},\ t=\text{kidney},\ m=L_{\text{donor}},\ c=\text{CKD stage},\Delta t = 1)
  ]

  * 在潜空间内，将 XY 细胞视作“起点”，LOY 细胞视作“终点”，用和 scPerturb 同样的方式拟合 (\tilde{K}_\theta)。

* 在脑 microglia 数据中：
  类似地对 LOY vs XY microglia 构造算子，协变量包括 AD 病理负担等。

### B.2.4 共同低秩分解：提取“mLOY 系统性响应轴”

将 Phase I 的扰动算子 + Phase II 的 LOY 算子**一起**做低秩分解：

* 这样，(B_k) 同时编码：

  * scPerturb 中 CRISPR/药物对细胞状态的主导响应模式；
  * mLOY 在肾脏/脑 microglia 中诱导的响应模式。

分析步骤：

1. 识别那些在 LOY 条件下 (\alpha_k(\theta)) 绝对值显著的轴，定义为“mLOY 轴”；
2. 对这些轴做基因层面的贡献分析（通过解码器的 Jacobian 或沿 (f_k) 方向的局部线性近似），进行 pathway 富集（炎症应答、应激、衰老、血管损伤等）；
3. 比较肾脏与脑 microglia 在这些 mLOY 轴上的 loading (\lambda_{tkc})（等效地，看 (B_k) 在组织特异基上 (B_k^{\text{kidney}},B_k^{\text{brain}}) 的差异），回答：

   * 哪些响应轴是**跨组织共享的**（例如系统性炎症轴）；
   * 哪些是**肾脏特异**（上皮损伤/代谢）或**脑特异**（microglia 激活、髓鞘相关轴）。

这一步给出一个**跨组织 mLOY 虚拟细胞效应图谱**，是文章的一个核心结果。

---

## B.3 Phase III：反事实预测 + 生物学验证场景

最后，为了让这篇工作在一区层面更有说服力，需要精心设计 1–2 个“反事实 + 验证”场景。这里给出几个可选方案，实际可以结合你们掌握的数据与实验条件选择。

### 场景 1：mLOY “纠正”反事实 —— LOY 细胞回推到“虚拟 XY 状态”

**目标：**
验证：算子模型是否能在虚拟空间内，把 LOY 细胞映射回接近 XY 细胞的分布，这既是数值测试，也是“虚拟干预”。

**操作：**

1. 选择肾脏或脑 microglia 数据中的 LOY 细胞，得到其潜表示 (z^{\text{LOY}})；
2. 构造一个“反 mLOY 算子”：

   * 在系数空间中将 mLOY 变量 (m) 从 1 设为 0（或把 (\alpha_k(\theta)) 中与 mLOY 强相关的部分取反/置零）；
   * 得到一个“虚拟 XY 条件” (\theta_{\text{virt-XY}}) 的算子 (K_{\theta_{\text{virt-XY}}})。
3. 对 LOY 细胞应用这个算子：
   [
   \tilde{z}^{\text{XY}} = K_{\theta_{\text{virt-XY}}}\left(z^{\text{LOY}}\right),
   ]
   然后解码到基因层面。
4. 评价：

   * 用 E-distance 比较 (\tilde{z}^{\text{XY}}) 的分布与真实 XY 细胞分布（越接近越好）；
   * 检查关键通路（如肾脏上皮损伤标志、microglia 激活标志、补体/炎症通路）的表达是否回落到接近 XY 水平。

**意义：**
如果模型成功，这将证明你们的“mLOY 算子”不仅能解释差异，还可以在潜空间中反转效应，为后续**药物筛选/靶点预测**提供基础。

---

### 场景 2：mLOY 与药物/基因扰动的“等效性”与“协同/拮抗”分析

**目标：**
利用 scPerturb 中的扰动（例如抗炎药物、JAK/STAT 通路抑制剂、TGF-β 通路相关药物/KO），分析这些扰动在算子轴上与 mLOY 的关系：

* 有没有某些药物的响应轴与 mLOY 轴高度接近（“虚拟矫正药物”候选）？
* 有没有某些药物与 mLOY 在同一轴上同向（潜在加重风险）？

**操作：**

1. 在 Phase I + II 共同低秩分解中，对于每个扰动 p（来自 scPerturb）和 mLOY，得到 (\alpha_k(\theta)) vector。
2. 定义一个“响应轮廓”向量
   [
   r(\theta) = (\alpha_1(\theta),\dots,\alpha_K(\theta)).
   ]
3. 计算 mLOY 对应条件 (\theta_{\text{mLOY}}) 与各药物条件 (\theta_p) 的相似度（余弦相似度、欧氏距离等）：
   [
   \mathrm{sim}(\theta_{\text{mLOY}},\theta_p)
   = \frac{\langle r(\theta_{\text{mLOY}}), r(\theta_p)\rangle}{|r(\theta_{\text{mLOY}})|\cdot|r(\theta_p)|}.
   ]
4. 选择几类：

   * 与 mLOY 轮廓相反（相似度为负）且在肾/脑相关通路上有已知保护作用的药物/靶点 → **候选“虚拟矫正药物”**；
   * 与 mLOY 同向的药物 → 可能在 mLOY 高风险个体中应慎用。

**虚拟组合实验：**

* 在潜空间中，对 mLOY 细胞应用组合算子：
  [
  z' = K_{\theta_p}\big( K_{\theta_{\text{mLOY}}}(z_0)\big),
  ]
  与只施加 mLOY 的 (K_{\theta_{\text{mLOY}}}(z_0)) 比较，观察关键通路变化。

**意义：**

* 从“virtual cell operator space”的角度，提出对 mLOY 有潜在矫正效应的 perturbation 组合；
* 即使不做 wet-lab，也可以用已有转录组/功能研究来部分验证这些候选，例如与肾脏/脑 mLOY 文献中提到的通路吻合度。

---

### 场景 3（可选）：mLOY 风险梯度上的虚拟脑微胶质图谱

如果数据允许（例如有不同 AD 风险/病理负荷、不同 mLOY 程度的个体）：

* 把 donor-level mLOY 负荷 (L)、AD 病理指标一起喂给 (\theta)，
* 在虚拟空间中构造一个“mLOY 风险梯度”（(\gamma) 从低到高），模拟脑 microglia 在这个梯度上的状态变化；
* 对比真实数据中不同 mLOY/AD 分组的 microglia 状态，从而展示算子模型对**疾病进展轨迹**的拟合能力。

---

## 小结

* 在 **方法层面**，我们已经用严格的符号把：

  * 潜空间构建（NB-VAE），
  * 扰动响应算子族 (K_\theta)，
  * 低秩分解 (A_\theta = A^{(0)}_t + \sum_k \alpha_k(\theta) B_k)，
  * 基于 E-distance 的训练目标，
  * 稳定性约束（谱半径/生成元）
    系统写出来了。

* 在 **实验层面**，提出了一个从：

  1. scPerturb proof-of-concept（方法正确性 + baseline 对比）；
  2. mLOY 肾脏 + 脑跨组织效应图谱（展示“系统性 mLOY 轴”）；
  3. 反事实 + 虚拟干预场景（mLOY “纠正”、mLOY–药物协同/拮抗）
     逐步升级的路线。


