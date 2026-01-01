# 蓝图落地进展与简化缺陷清单

本文件用于记录当前代码库相对“最终优化方案蓝图”的落地进展与仍存简化点，便于后续迭代与评审。

## 一、已落地的关键改动（阶段性成果）

### 1. 数据—模型契约（部分落实）
- **NB-VAE** 保留作为 counts 输入的主线。
- **Gaussian VAE** 作为 log-normalized 输入兜底。
- 训练脚本强制检查 `data.input_representation` 与 `model.likelihood` 一致性，避免统计假设混用。

### 2. 扰动算子结构升级
- **Residual + Gated** 结构：`z_out = z + g(θ) ⊙ (A_θ z + b_θ)`。
- **系数 softmax** 约束，确保基矩阵的凸组合，提升稳定性与可解释性。

### 3. 训练目标与稳定性
- 分布级损失：**E-distance** 支持自动 exact/batched 切换。
- 新增 **Δ-loss** 与 **组内一致性正则**（condition-wise variance）。
- 新增 **样本级谱范数约束**（batch-wise power iteration）。
- 控制条件加入 **gate 正则**，鼓励对照条件接近恒等映射。

### 4. 训练策略
- 已具备 **冻结/部分解冻/全解冻** 能力（`finetune_scope`）。
- 支持设定 `finetune_start_epoch` 进行阶段式解冻。

### 5. 评测与对标
- 引入 **PCA E-distance**。
- 引入 **perturbation-specific shift 指标**：Pearson(Δ) 与 Pearson(Δ20)。
- 增加 **简单基线对照**：no-change / mean / matching-mean。

---

## 二、仍存在的简化缺陷（待消除）

### A. 数据切分与评测闭环（P0）
- **Systema 的 unseen split / 组合扰动分层**尚未实现。
  - 目前仍依赖外部准备的 train/val/test 文件，未在代码中内置 perturbation-level split。

### B. Δ20 的严格防泄漏（P0）
- **Δ20 需要只基于训练集定义 top-20 DE genes**。
- 目前仅在传入 `--train_data_path` 时执行该逻辑；若未提供则可能退回测试集估计（存在信息泄漏风险）。

### C. 评测基线不足（P1）
- 目前仅实现 `no-change / mean / matching-mean`。
- **线性/加性基线**（如简单线性回归或扰动均值模型）尚未纳入。

### D. VAE Denoising（P1）
- 蓝图要求的 **gene masking / denoising 训练**尚未加入。
- 当前训练使用原始输入，缺乏噪声鲁棒性增强。

### E. scPerturb 评测流程细节（P1/P2）
- PCA E-distance 已实现，但未包含 **HVG 筛选 / subsampling / batch correction** 等 scPerturb 规范流程。

### F. 对照条件门控（P1）
- 目前使用 **gate 正则**；但 **control 条件识别**仍依赖标签一致性（无统一映射/标准化）。

---

## 三、推荐后续推进顺序（执行优先级）

1. **Systema split + 组合扰动分层评测**（P0）
2. **Δ20 防泄漏**（强制使用训练集统计）（P0）
3. **基线扩展：线性/加性基线**（P1）
4. **VAE denoising / gene masking**（P1）
5. **scPerturb 评测完整流程复刻**（P1/P2）
6. **control 标签标准化机制**（P1）

---

## 四、备注

- 本文件仅记录**当前落地状态**与**简化缺陷**，并不改变代码逻辑。
- 后续每次迭代建议更新本文件，以保持评审闭环与复现友好。
