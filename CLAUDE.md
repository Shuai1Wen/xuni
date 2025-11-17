# CLAUDE.md 开发准则 - 虚拟细胞算子模型项目

## 概览

本文件是针对 **虚拟细胞算子模型（Virtual Cell Operator Model）+ mLOY跨组织分析** 项目的开发准则，用于指导在当前仓库内进行的全部开发与文档工作，确保输出遵循强制性标准并保持可审计性。

### 项目核心目标
构建基于算子理论的虚拟细胞模型，实现：
1. 在scPerturb数据集上验证扰动响应预测能力
2. 分析mLOY（Y染色体马赛克缺失）在肾脏和脑组织中的跨组织效应
3. 实现反事实模拟和虚拟干预预测

---

## 强制性全局规范

### 1. 语言使用强制规范

⚠️ **绝对强制使用简体中文**：
- ✅ 所有 AI 回复、文档、注释、日志、提交信息、代码注释
- ✅ README.md、配置说明、实验报告、审查报告
- ✅ 变量/函数的文档字符串（docstring）
- ❌ **唯一例外**：代码标识符（变量名、函数名、类名）使用英文

**违反此规范的任何输出必须立即重写为简体中文。**

### 2. 强制验证机制

🔒 **本地AI自动验证原则**：
- ✅ 必须拒绝一切 CI、远程流水线或人工外包验证
- ✅ 所有验证均由本地 AI 自动执行
- ✅ 每次改动必须提供可重复的本地验证步骤
- ✅ 验证失败时立即终止提交

**验证清单**：
```markdown
□ 代码可执行性验证（无语法错误）
□ 数学公式与model.md一致性验证
□ 文档完整性验证（README + requirements.txt）
□ 命名规范验证（遵循项目既有约定）
□ 注释充分性验证（所有核心逻辑有中文注释）
```

### 3. 质量审查规范

**审查职责（Claude Code 独立执行）**：
1. 使用 `sequential-thinking` 进行深度审查分析
2. 输出技术维度评分（代码质量、测试覆盖、规范遵循）
3. 输出战略维度评分（需求匹配、架构一致、风险评估）
4. 输出综合评分（0-100）和明确建议（通过/退回/需讨论）
5. 生成 `.claude/verification-report.md` 审查报告

**决策规则**：
- 综合评分≥90分且建议"通过" → 确认通过
- 综合评分<80分且建议"退回" → 确认退回
- 80-89分或建议"需讨论" → 仔细审阅后决策

---

## 项目特定上下文要求

### 强制上下文检索机制

在编码前，**必须**完成以下检索（按复杂度分级执行）：

#### 复杂度分级标准
- **简单任务**（单文件、<50行、无依赖）：执行步骤1-3
- **中等任务**（多文件、<200行、少量依赖）：执行完整7步
- **复杂任务**（架构级、>200行、复杂依赖）：执行完整7步+增强验证

#### 7步强制检索清单

**□ 步骤1：文件名搜索（必须）**
```bash
desktop-commander.start_search searchType="files" pattern="关键词"
```
- 目标：找到5-10个候选文件
- 记录：找到X个相关文件，重点关注 [列出文件路径]

**□ 步骤2：内容搜索（必须）**
```bash
desktop-commander.start_search searchType="content" pattern="函数名|类名" literalSearch=true contextLines=5
```
- 目标：找到关键实现位置
- 记录：找到X处实现，重点分析 [file:line]

**□ 步骤3：阅读相似实现（必须≥3个）**
- 目标：理解实现模式和设计理由
- 记录：分析了 [file1:line, file2:line, file3:line]
- 关注点：实现模式、可复用组件、需注意事项

**□ 步骤4：开源实现搜索（通用功能必做）**
```bash
github.search_code query="具体功能实现" language:"Python"
```
- 目标：学习最佳实践
- 触发条件：算法实现、数据结构、设计模式

**□ 步骤5：官方文档查询（涉及库/框架必做）**
```bash
context7 resolve-library-id libraryName="PyTorch"
context7 get-library-docs context7CompatibleLibraryID="pytorch/pytorch"
```
- 目标：掌握最佳实践，避免错误用法
- 优先级：PyTorch、AnnData、Scanpy、NumPy

**□ 步骤6：测试代码分析（必须）**
```bash
desktop-commander.start_search searchType="content" pattern="test_|pytest" filePattern=".test.py"
```
- 目标：理解测试策略和覆盖标准

**□ 步骤7：模式提取和分析（必须）**
```bash
sequential-thinking # 分析检索结果，提取项目模式
```
- 输出：生成 `.claude/context-summary-[任务名].md`

### 上下文充分性验证

编码前必须全部回答"是"：

```markdown
□ 我能说出至少3个相似实现的文件路径吗？
  ✅ 是：[file1:line, file2:line, file3:line]
  ❌ 否 → 返回步骤1重新搜索

□ 我理解项目中这类功能的实现模式吗？
  ✅ 是：模式是 [具体描述]，因为 [理由]
  ❌ 不确定 → 返回步骤3深度阅读

□ 我知道项目中有哪些可复用的工具函数/类吗？
  ✅ 是：[列出具体函数/类名和路径]
  ❌ 不知道 → 强制搜索utils/helpers模块

□ 我理解项目的命名约定和代码风格吗？
  ✅ 是：命名约定是 [具体说明]
  ❌ 不清楚 → 阅读更多代码

□ 我知道如何测试这个功能吗？
  ✅ 是：参考 [测试文件]，我会 [具体测试策略]
  ❌ 不知道 → 搜索并阅读相关测试代码

□ 我确认没有重复造轮子吗？
  ✅ 是：检查了 [具体模块/文件]
  ❌ 不确定 → 扩大搜索范围

□ 我理解这个功能的依赖和集成点吗？
  ✅ 是：依赖 [具体依赖]，集成点是 [具体位置]
  ❌ 不清楚 → 分析import语句和调用链
```

---

## 项目特定架构原则

### 1. 数学保真度优先级（最高）

**核心原则**：代码实现必须100%忠实于 `model.md` 的数学定义

**强制检查**：
```markdown
□ 所有公式符号与model.md完全一致
□ 所有算子定义（K_θ、A_θ、B_k）严格遵循数学形式
□ 所有维度变换有明确数学依据
□ 所有损失函数与model.md中的目标函数对应
```

**禁止行为**：
- ❌ 简化或省略model.md中的任何数学细节
- ❌ 在未记录的情况下修改公式
- ❌ 使用与数学定义不一致的实现方式

### 2. 向量化优先原则

**强制要求**：
- ✅ 禁止在可向量化的场景使用for循环
- ✅ 优先使用PyTorch/NumPy的批量操作
- ✅ 所有张量操作必须支持批处理

**示例**：
```python
# ❌ 禁止
for i in range(batch_size):
    z_out[i] = A_theta[i] @ z[i] + b_theta[i]

# ✅ 正确
z_out = torch.bmm(A_theta, z.unsqueeze(-1)).squeeze(-1) + b_theta
```

### 3. 组件复用强制原则

**项目核心组件清单**（必须复用）：
```python
# src/models/nb_vae.py
- Encoder: 编码器基类
- DecoderNB: 负二项解码器
- NBVAE: 完整VAE模型
- elbo_loss: ELBO损失函数

# src/models/operator.py
- OperatorModel: 算子模型主类
- spectral_penalty: 谱范数正则化

# src/utils/edistance.py
- pairwise_distances: 成对距离计算
- energy_distance: E-distance计算

# src/utils/virtual_cell.py
- encode_cells: 细胞编码
- decode_cells: 细胞解码
- apply_operator: 算子应用
- virtual_cell_scenario: 多步反事实模拟
```

**强制验证**：编码后必须在 `.claude/operations-log.md` 中声明复用的组件

### 4. 文件结构强制规范

**项目目录结构**（来自details.md，必须严格遵循）：
```
virtual-cell-operator-mLOY/
├── configs/          # YAML配置文件
├── data/             # 数据目录
│   ├── raw/          # 原始数据
│   └── processed/    # 处理后数据
├── scripts/          # 可执行脚本
├── src/              # 核心代码库
│   ├── models/       # 模型定义
│   ├── data/         # 数据加载器
│   ├── utils/        # 工具函数
│   └── train/        # 训练循环
├── notebooks/        # Jupyter笔记本
└── results/          # 结果输出
    ├── logs/
    ├── checkpoints/
    └── figures/
```

**禁止行为**：
- ❌ 在src/外编写核心功能代码
- ❌ 在scripts/中编写可复用逻辑
- ❌ 混淆data/raw和data/processed

---

## 强制工作流程

### 总原则
1. **强制深度思考**：任何时候必须首先使用 `sequential-thinking` 工具梳理问题
2. **连续执行**：非必要问题不询问用户，必须自动连续执行
3. **问题驱动**：追求充分性而非完整性，动态调整而非僵化执行

### 工具链执行顺序（必须）
```
sequential-thinking → desktop-commander(检索) → 直接执行
```

### 标准工作流 6 步骤

#### 阶段0：需求理解与上下文收集

**简单任务快速通道**：
- 条件：<30字描述，单一目标，无架构影响
- 动作：直接进入上下文收集

**复杂任务深度分析**：
```bash
sequential-thinking "分析任务需求，识别关键疑问"
```

**上下文收集**：
- 执行7步强制检索清单
- 生成 `.claude/context-summary-[任务名].md`
- 通过充分性验证（7项检查）

#### 阶段1：任务规划

```bash
sequential-thinking "基于上下文摘要分析实现策略"
```

**规划输出**（写入 `.claude/operations-log.md`）：
```markdown
## 任务规划 - [任务名称]
时间：[YYYY-MM-DD HH:mm:ss]

### 接口规格
- 输入：[类型、形状、约束]
- 输出：[类型、形状、约束]
- 依赖：[外部库、内部模块]

### 数学对应关系
- model.md 公式：[公式编号或描述]
- 代码实现：[函数名、文件路径]
- 关键参数：[参数含义与数学符号对应]

### 边界条件
- [列出需要处理的边界情况]

### 性能要求
- 时间复杂度：O(...)
- 空间复杂度：O(...)
- 批处理大小：[建议值]

### 测试标准
- 单元测试：[测试内容]
- 数值精度：[容忍误差]
- 与model.md一致性：[验证方法]
```

#### 阶段2：代码执行

**实时记录**（写入 `.claude/operations-log.md`）：
```markdown
## 编码记录 - [功能名称]
时间：[YYYY-MM-DD HH:mm:ss]

### 复用的组件
- [组件1]：路径 [src/xxx.py:line]，用途 [...]
- [组件2]：路径 [src/xxx.py:line]，用途 [...]

### 实现决策
- 决策1：[为什么这样实现]，依据 [model.md公式X / 相似实现Y]
- 决策2：[...]

### 向量化实现
- 原始逻辑：[数学表达]
- 向量化代码：[代码片段]
- 性能提升：[预期加速比]
```

**编码规范**：
```python
# 强制注释格式（简体中文）
def energy_distance(x: torch.Tensor, y: torch.Tensor):
    """
    计算两组样本之间的能量距离（E-distance）

    对应 model.md 公式 (A.4) 节

    参数:
        x: (n, d) 第一组样本
        y: (m, d) 第二组样本

    返回:
        ed2: 能量距离的平方（标量）

    实现细节:
        - 使用向量化计算避免双重循环
        - 数值稳定性：距离平方添加clamp和epsilon
    """
    # 实现代码...
```

#### 阶段3：质量验证

**验证步骤**：
1. 使用 `sequential-thinking` 进行深度审查
2. 执行本地测试脚本
3. 生成 `.claude/verification-report.md`

**验证报告模板**：
```markdown
# 验证报告 - [功能名称]
生成时间：[YYYY-MM-DD HH:mm:ss]

## 技术维度评分
- 代码质量：[0-100分]
  - 可读性：[评分] - [说明]
  - 向量化程度：[评分] - [说明]
  - 注释完整性：[评分] - [说明]

- 测试覆盖：[0-100分]
  - 单元测试：[是/否] - [说明]
  - 边界条件：[覆盖率%] - [说明]
  - 数值精度：[是/否] - [说明]

- 规范遵循：[0-100分]
  - 命名规范：[评分] - [说明]
  - 文件结构：[评分] - [说明]
  - 中文注释：[评分] - [说明]

## 战略维度评分
- 需求匹配：[0-100分]
  - 与model.md一致性：[评分] - [说明]
  - 功能完整性：[评分] - [说明]

- 架构一致：[0-100分]
  - 组件复用：[评分] - [列出复用的组件]
  - 模块解耦：[评分] - [说明]

- 风险评估：[低/中/高]
  - 数值稳定性：[风险等级] - [说明]
  - 性能瓶颈：[风险等级] - [说明]
  - 维护成本：[风险等级] - [说明]

## 综合评分
总分：[0-100]

## 建议
- [通过 / 退回 / 需讨论]
- 理由：[...]
- 改进建议：[...]
```

---

## 代码质量强制标准

### 1. 注释规范

**强制要求**：
- ✅ 所有函数必须有完整的中文docstring
- ✅ 所有复杂逻辑必须有中文行内注释
- ✅ 所有数学公式必须引用model.md的对应位置

**模板**：
```python
class OperatorModel(nn.Module):
    """
    扰动响应算子模型

    实现 model.md 公式 (A.5.1)：
    A_θ = A_t^(0) + Σ_k α_k(θ) B_k

    参数:
        latent_dim: 潜空间维度 d_z
        n_tissues: 组织类型数量
        n_response_bases: 响应基数量 K
        cond_dim: 条件向量维度

    属性:
        A0_tissue: 组织基线算子 (n_tissues, d_z, d_z)
        B: 全局响应基 (K, d_z, d_z)
        alpha_mlp: 条件→系数的映射网络
    """
```

### 2. 测试规范

**强制要求**：
- ✅ 每个核心函数必须有对应的测试
- ✅ 测试必须验证与model.md的一致性
- ✅ 测试必须覆盖边界条件

**测试模板**：
```python
def test_energy_distance():
    """测试E-distance计算的正确性"""
    # 1. 正常情况：两个不同分布
    x = torch.randn(100, 10)
    y = torch.randn(100, 10) + 1.0
    ed2 = energy_distance(x, y)
    assert ed2 > 0, "不同分布的E-distance应大于0"

    # 2. 边界情况：相同分布
    ed2_same = energy_distance(x, x)
    assert torch.abs(ed2_same) < 1e-6, "相同分布的E-distance应接近0"

    # 3. 数值稳定性：空集
    x_empty = torch.randn(0, 10)
    ed2_empty = energy_distance(x_empty, y)
    assert ed2_empty == 0, "空集的E-distance应为0"
```

### 3. 性能规范

**强制要求**：
- ✅ 禁止不必要的循环
- ✅ 禁止重复计算
- ✅ 禁止内存泄漏

**性能检查清单**：
```markdown
□ 所有批量操作都使用了向量化
□ 所有中间结果都使用了.detach()（当不需要梯度时）
□ 所有大矩阵乘法都使用了torch.bmm或torch.einsum
□ 所有循环都经过必要性验证（无法向量化才允许）
```

---

## 项目特定工具集成

### 优先级排序

1. **desktop-commander**（最高优先级）
   - 所有本地文件操作
   - 所有数据分析（CSV/JSON/h5ad）
   - 所有代码搜索

2. **context7**（编程文档查询）
   - PyTorch官方文档
   - AnnData/Scanpy文档
   - NumPy/Pandas文档

3. **github.search_code**（开源实现搜索）
   - 算法实现参考
   - 最佳实践学习

4. **sequential-thinking**（深度分析）
   - 需求分析
   - 代码审查
   - 问题诊断

### 禁止行为

❌ **绝对禁止使用bash进行以下操作**：
- 文件读写（应使用desktop-commander.read_file / write_file）
- 代码搜索（应使用desktop-commander.start_search）
- 数据分析（应使用desktop-commander + Python REPL）

---

## 懒惰检测与防护机制

### 检测点1：编码前检查

**必须在 `.claude/operations-log.md` 中记录**：
```markdown
## 编码前检查 - [功能名称]
时间：[YYYY-MM-DD HH:mm:ss]

□ 已查阅上下文摘要文件：.claude/context-summary-[任务名].md
□ 将使用以下可复用组件：
  - [组件1]: src/xxx.py:line - [用途]
  - [组件2]: src/xxx.py:line - [用途]
□ 将遵循命名约定：[具体说明]
□ 将遵循代码风格：[具体说明]
□ 确认不重复造轮子，证明：[检查了哪些模块]
```

### 检测点2：编码后验证

**完整声明**：
```markdown
## 编码后声明 - [功能名称]
时间：[YYYY-MM-DD HH:mm:ss]

1. 复用了以下既有组件
   - [组件1]: 用于 [用途]，位于 src/xxx.py:line

2. 遵循了以下项目约定
   - 命名约定：[对比说明，举例证明]
   - 代码风格：[对比说明，举例证明]

3. 对比了以下相似实现
   - [实现1]: 我的方案与其差异是 [...]，理由是 [...]

4. 未重复造轮子的证明
   - 检查了 [模块列表]，确认不存在相同功能
```

### 三级惩罚体系

**Level 1 - 警告**（首次检测）：
- 立即暂停编码
- 记录警告到operations-log.md
- 要求立即修正
- 重新对比上下文摘要

**Level 2 - 强制退回**（二次检测）：
- 删除已编写的代码
- 强制返回检索阶段
- 重新生成上下文摘要
- 记录"二次懒惰"

**Level 3 - 任务失败**（三次检测）：
- 标记任务为"失败"
- 生成失败报告
- 需要用户介入

---

## 项目特定禁止事项

### 绝对禁止

1. **数学方面**
   - ❌ 简化model.md中的任何公式
   - ❌ 省略model.md中的任何细节
   - ❌ 在未记录的情况下修改数学定义

2. **实现方面**
   - ❌ 使用for循环进行可向量化的批量操作
   - ❌ 重复实现已有的工具函数
   - ❌ 在src/外编写核心逻辑

3. **文档方面**
   - ❌ 使用英文注释（代码标识符除外）
   - ❌ 省略docstring
   - ❌ 不引用model.md公式

4. **流程方面**
   - ❌ 跳过上下文检索
   - ❌ 跳过充分性验证
   - ❌ 不记录operations-log.md

---

## 文件结构规范

### 强制工作文件位置

所有工作文件必须写入项目本地 `.claude/` 目录：

```
<project>/.claude/
├── context-summary-[任务名].md    # 上下文摘要
├── operations-log.md               # 决策和操作记录
└── verification-report.md          # 验证报告
```

### 上下文摘要模板

```markdown
# 项目上下文摘要（[任务名称]）
生成时间：[YYYY-MM-DD HH:mm:ss]

## 1. 相似实现分析
**实现1**: src/models/nb_vae.py:68-89
- 模式：编码器-解码器架构
- 可复用：Encoder基类、sample_z函数
- 需注意：组织条件输入拼接方式

## 2. 项目约定
- 命名约定：类用PascalCase，函数用snake_case
- 文件组织：models/下按模型类型分文件
- 导入顺序：标准库 → 第三方库 → 项目内模块
- 代码风格：遵循Black格式化

## 3. 可复用组件清单
- src/utils/edistance.py: E-distance计算
- src/models/nb_vae.py: VAE编码解码器
- src/utils/virtual_cell.py: 虚拟细胞操作

## 4. 测试策略
- 测试框架：pytest
- 测试模式：单元测试 + 数值精度测试
- 参考文件：tests/test_edistance.py
- 覆盖要求：正常流程 + 边界条件 + 数值稳定性

## 5. 依赖和集成点
- 外部依赖：torch, numpy, anndata
- 内部依赖：models.nb_vae → utils.edistance
- 集成方式：直接导入调用
- 配置来源：configs/default.yaml

## 6. 技术选型理由
- 为什么用PyTorch：自动微分 + GPU加速
- 为什么用E-distance：model.md公式(A.4)，无需OT匹配
- 优势：计算高效、数学严格
- 劣势和风险：大batch时内存占用高

## 7. 关键风险点
- 并发问题：无（单线程训练）
- 边界条件：空batch、单样本batch
- 性能瓶颈：E-distance的O(n²)复杂度
- 数值稳定性：距离计算需要clamp和epsilon
```

---

## 交付物强制标准

### 每次功能实现必须包含

1. **源代码**
   - 位置：`src/` 下的正确子目录
   - 格式：符合Black标准
   - 注释：完整的中文docstring和行内注释
   - 测试：对应的测试文件

2. **文档更新**
   - README.md：更新使用说明（如有新功能）
   - requirements.txt：更新依赖（如有新库）
   - 配置文件：更新configs/（如有新参数）

3. **验证报告**
   - 路径：`.claude/verification-report.md`
   - 内容：完整的技术和战略维度评分
   - 结论：明确的通过/退回建议

4. **操作日志**
   - 路径：`.claude/operations-log.md`
   - 内容：完整的决策记录和复用声明

---

## Git操作规范

### 提交前强制检查

```markdown
□ 所有文件使用UTF-8编码
□ 所有注释使用简体中文
□ 所有测试通过
□ 验证报告综合评分≥80分
□ operations-log.md已更新
```

### 提交信息规范

**格式**（简体中文）：
```
<类型>: <简短描述>

详细说明：
- 实现了什么功能
- 对应model.md的哪个部分
- 复用了哪些组件

验证：
- 测试覆盖率：X%
- 综合评分：X分
```

**类型标签**：
- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `refactor`: 代码重构
- `test`: 测试相关
- `perf`: 性能优化

### 分支策略

**当前分支**：`claude/generate-claude-md-guidelines-014Li1oNeYYZ7wYmTTZANbhL`

**强制要求**：
- ✅ 所有开发必须在指定分支上进行
- ✅ 提交前必须通过本地验证
- ✅ 推送使用：`git push -u origin <branch-name>`

---

## 数学-代码一致性检查清单

### 强制对应关系验证

在实现任何数学公式时，必须在注释中明确标注：

```python
def low_rank_decomposition(A_theta_list, K):
    """
    低秩分解算子族

    对应 model.md 公式 (A.5.1):
    A_θ = A_t^(0) + Σ_{k=1}^K α_k(θ) B_k

    参数:
        A_theta_list: List[Tensor] - 局部算子列表，对应 {Ã_θ}
        K: int - 响应基数量

    返回:
        A0: Tensor (n_tissues, d, d) - 基线算子 A_t^(0)
        B: Tensor (K, d, d) - 响应基 B_k
        alpha: Tensor (n_conditions, K) - 系数 α_k(θ)

    实现逻辑:
        1. 对所有 Ã_θ 减去对应组织的均值 → 获得偏差
        2. 对偏差矩阵族进行SVD分解
        3. 取前K个主成分作为 B_k
        4. 最小二乘拟合 α_k(θ)
    """
```

### 维度一致性验证

**强制检查**：
```markdown
□ 所有张量维度与model.md中的符号定义一致
  - z ∈ ℝ^{d_z} → shape: (batch, latent_dim)
  - A_θ ∈ ℝ^{d_z×d_z} → shape: (batch, latent_dim, latent_dim)
  - x ∈ ℝ^G → shape: (batch, n_genes)

□ 所有批量操作正确处理了batch维度
□ 所有矩阵乘法使用了正确的运算（bmm/matmul/einsum）
```

---

## 反事实模拟特定规范

### 多步算子应用规范

**实现要求**：
```python
def multi_step_operator(z0, operator_model, cond_vec_seq, tissue_idx):
    """
    多步算子序列应用

    对应 model.md 公式 (A.8.2):
    z_1 = K_{θ_1}(z_0)
    z_2 = K_{θ_2}(z_1)
    ...

    参数:
        z0: 初始潜状态 (batch, d_z)
        operator_model: 算子模型
        cond_vec_seq: 条件序列 (n_steps, cond_dim)
        tissue_idx: 组织索引 (batch,)

    返回:
        z_final: 最终潜状态 (batch, d_z)
        z_trajectory: 完整轨迹 (n_steps+1, batch, d_z)

    数值稳定性:
        - 每步检查谱范数是否<阈值
        - 检测潜空间是否发散（norm是否爆炸）
        - 如发散则提前终止并警告
    """
```

**稳定性检查**：
```markdown
□ 每步算子的谱范数 ρ(A_θ) ≤ 1.05
□ 潜状态的范数不超过初始值的10倍
□ 梯度不包含NaN或Inf
```

---

## 最终检查清单

### 功能实现完成前必须验证

```markdown
## 功能完成检查清单

### 数学正确性
□ 与model.md公式100%对应
□ 所有维度变换有数学依据
□ 所有损失函数与目标函数一致

### 代码质量
□ 所有函数有完整中文docstring
□ 所有复杂逻辑有中文注释
□ 无不必要的for循环
□ 所有批量操作已向量化

### 组件复用
□ 复用了至少1个既有组件
□ 在operations-log.md中声明了复用
□ 未重复实现已有功能

### 测试覆盖
□ 编写了单元测试
□ 测试覆盖正常情况
□ 测试覆盖边界条件
□ 测试验证数值精度

### 文档完整
□ README.md已更新（如需要）
□ requirements.txt已更新（如需要）
□ .claude/verification-report.md已生成
□ .claude/operations-log.md已记录

### 性能优化
□ 无内存泄漏
□ 无重复计算
□ 关键路径已优化

### 规范遵循
□ 所有文本使用简体中文（代码标识符除外）
□ 文件位于正确目录
□ 符合项目命名约定
□ 通过Black格式化
```

---

## 结语

本开发准则是针对虚拟细胞算子模型项目的强制性规范，所有开发工作必须严格遵守。违反任何强制性条款都将导致代码退回。

**核心原则总结**：
1. **数学保真度**：100%忠实于model.md
2. **简体中文**：所有可读文本必须中文
3. **向量化优先**：禁止不必要的循环
4. **组件复用**：绝不重复造轮子
5. **本地验证**：AI自动执行，拒绝外包
6. **完整记录**：所有决策可追溯

**三个关键文件**：
- `model.md` - 数学真理来源
- `suanfa.md` - 代码骨架参考
- `details.md` - 工程结构蓝图

遵循本准则，确保项目的科学严谨性、工程质量和可维护性。
