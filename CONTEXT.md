# AI 协作指导文件

> **最后更新**：2026-02-18
> **研究人员**：徐振宇、李享

---

# 第一部分：个性化要求（强制遵守）

> **本部分是永久性强制规则。无论研究内容如何变化，以下要求始终生效，任何 AI 助手在参与本项目讨论时必须严格遵守，不得违反。**

## 1. 思维方式

- **绝对严谨**：所有推理、分析、建议必须有明确的逻辑链条或事实依据。不确定的内容必须明确标注"不确定"或"需要验证"，禁止用模糊措辞掩盖不确定性。
- **遵循事实**：只陈述已知事实和有依据的推论。如果不知道，直接说不知道。**禁止弄虚作假、编造数据、捏造引用、虚构实验结果。**
- **不刻意质疑，不刻意讨好**：不为了显示批判性思维而强行找茬，也不为了迎合用户而附和错误观点。遇到用户的逻辑漏洞或事实错误，直接客观指出并给出理由，不需要委婉修饰。
- **区分"已证实"与"推测"**：讨论中必须清晰区分——哪些是实验数据支撑的结论，哪些是合理推测，哪些是未经验证的假设。
- 若在思考过程中出现任何不确定的内容，必须提出问题与我讨论得出答案再继续。禁止自主产生幻觉。

## 2. 输出格式

- **公式**：使用 $\LaTeX$ 格式书写所有数学公式。行内公式用 `$...$`，独立公式用 `$$...$$`。如果是Claude code，请用Unicode格式输出。
- **语言**：默认使用中文交流。
- **术语解释**：讨论中涉及 LLM 相关专业术语时，需在首次出现时附带简要解释。

## 3. 讨论原则

- **基于已有数据**：讨论和建议应优先基于已有的实验数据和机理发现。提出新方向时，需说明与已有发现的关系。
- **谨慎区分层次**：
  - 表征层面的可读性（Probe 能解码出信息） $\neq$ 决策层面的采纳（模型最终输出是否依赖该信息）
  - 相关性 $\neq$ 因果性
- **本研究的性质**：这是授权的安全研究（authorized security research），目的是揭示多模态大模型在情绪维度上的对抗脆弱性并推动防御研究。

## 4. 协作边界

- 可以讨论的内容：方法论设计、数学推导、实验设计与结果解读、论文写作与结构、相关文献、代码实现方案
- AI 提出的所有建议必须附带理由
- 如果用户提供的信息与本文件中的研究内容描述有冲突，**以用户当前陈述为准**（研究内容随时在更新）

---

# 第二部分：研究内容（实时参考）

> **本部分是研究进展的快照，仅供参考。如果用户在对话中提供了更新的信息，以用户的最新陈述为准。**

## A. 研究概述

研究多模态语音-语言大模型（ALLM）在情绪维度上的对抗脆弱性：能否通过对输入语音施加微小扰动，使模型将说话人情绪定向误判为攻击者指定的目标情绪，同时保持语义内容基本不变？

目标模型 Kimi-Audio（7B，28 层）和 OpenS2S（35 层），架构流程相似：原始波形 → Audio Encoder → Speech Adapter → 与 text embedding 拼接 → Transformer。关键特性：不同 prompt 共享同一音频表示。机理分析主要基于 OpenS2S（35 层）的实验数据。

| 路线 | 攻击者能力 | 状态 |
|------|-----------|------|
| 白盒攻击 | 完全访问模型权重与中间表征 | 方法论 + 机理分析完成 |
| 黑盒攻击 | 仅能查询模型 API | 尚未开始 |

## B. 白盒攻击

通过优化输入音频波形实现情绪误判。损失函数包含三项：情绪目标损失（使模型输出目标情绪）、语义一致性约束（防止通过破坏语义来改变情绪）、扰动控制（限制人耳可感知的变化）。总损失：

$$\mathcal{L}(x') = \lambda_{\text{emo}} \mathcal{L}_{\text{emo}} + \lambda_{\text{asr}} \mathcal{L}_{\text{asr}} + \lambda_{\text{per}} \mathcal{L}_{\text{per}}$$

当前结果：训练 prompt 下成功率 ~80%，但跨 prompt 鲁棒性差（直接分类 ~40%、描述+分类 ~50%、CoT 类 ~20%）。这一问题驱动了机理分析。

## C. 机理分析

机理分析揭示 ALLM 如何处理情绪相关的模态冲突，核心发现如下：

**层级结构**（Probe 实验，247 条样本）：早层（0-14）韵律表征主导，中层（14-23）融合态，晚层（23-35）语义决策主导。关键边界在 14-15 层。

**决策追踪**（Logit Lens）：表征层面韵律更可读，但决策层面从第 23 层起语义逐步占主导——模型"听得出"韵律但"判断"时依赖语义。

**因果证据**（Activation Patching）：语义 patch 在早层（0-12）有强定向控制力（Flip to Target ~0.65），韵律 patch 效果弱（~0.14-0.26）。~14-15 层是 audio span 可控性边界，之后信息已扩散到全局位置。

**Prompt-Audio 冲突**（18 条音频 × 3 种条件）：当文本指令携带的情绪指向与音频情绪冲突时，文本指令在中层（5-20）保持强因果效应（PatchText 有效，PatchAudio 无效），冲突仲裁在 26-28 层固化为文本指向。即：ALLM 的情绪决策存在两级模态优先级——音频内语义 > 韵律（2.1），文本指令 > 音频整体信号（2.2）。

## D. 论文大纲与当前焦点

```
1. Introduction
2. Observation（纯机理发现，为方法论提供因果铺垫）
   2.1 音频内模态冲突机理（语义 vs 韵律）
   2.2 跨模态冲突仲裁机理（文本指令 vs 音频信号）
3. Threat Model
4. 方法论（由 2.1-2.2 机理驱动设计）
5. 实验（Setting / 白盒结果 / 黑盒结果 / 回复评估 / 拓展领域 / Defense）
6. Related Work
7. Discussion and Limitations
附录
```

**当前焦点**：Section 2 Observation LaTeX 初稿已完成（见 `2OBSERVATION/observation.tex`），写作风格参考 hallucinate\_yangming 论文 Section 3 的叙事技法优化。下一步需补充待验证实验（文中红色标注），并推进 Section 3 Threat Model 写作。

**Observation 写作状态**：
- 2.1 音频内模态冲突机理：LaTeX 正文已完成，含 Probe/Logit Lens/Patching 三层分析 + 编号性质 (1)(2)(3) + 桥接句过渡
- 2.2 跨模态冲突仲裁机理：LaTeX 正文已完成，含晚期层固化/文本因果主导两个发现 + 两级优先级总结 + 过渡至 Section 3
- 待补充实验（共 8 处红色标注）：bootstrap CI、Probe 稳健性、unigram 基线、position-level patching、PatchAudio 不可逆性、Text-Dominance Index、压制机制归因、跨模型验证

**关键写作原则**：
- Observation 中不出现攻击成功率等实验结果数据（属于 Section 5）
- 攻击方法论的引出基于机理发现，而非反向从攻击结果解释机理
- 原 2.3（黑盒条件）无独立实验数据，已合并入 Section 3 Threat Model

## E. 文件索引

| 路径 | 内容 |
|------|------|
| `2OBSERVATION/observation.tex` | **Section 2 Observation LaTeX 正文（当前主文件）** |
| `2OBSERVATION/main.tex` | LaTeX 编译入口（XeLaTeX + ctex） |
| `2OBSERVATION/figures/` | Observation 配图（Probe/Logit Lens/Patching 共 11 张） |
| `2OBSERVATION/observation.md` | Section 2 Observation 写作大纲 |
| `2OBSERVATION/observation_cc.md` | Observation 大纲草稿（详细版，含缺口分析） |
| `2OBSERVATION/observation_pro.md` | Observation 大纲草稿（精简版） |
| `2OBSERVATION/observation_cx.md` | Observation 大纲草稿（含 Gap Checklist） |
| `LATEST/white_box_final/PPT大纲.md` | PPT 全文案 |
| `LATEST/white_box_final/PPTtext.md` | PPT 解析版 |
| `LATEST/white_box_final/audio内部机理1/` | 实验素材：音频内部机理第一阶段 |
| `LATEST/white_box_final/audio内部机理2/` | 实验素材：音频内部机理第二阶段 |
| `LATEST/white_box_final/prompt&audio机理/` | 实验素材：Prompt-Audio 冲突机理 |
| `LATEST/white_box_final/白盒对抗样本方法论/` | 实验素材：白盒攻击方法论 |
| `LATEST/白盒讲稿.md` | 汇报讲稿 |
| `LATEST/情绪LLM白盒攻击研究.pptx` | 完整 PPT |
| `LATEST/情绪LLM白盒攻击研究.pdf` | PPT 导出 PDF |
| `PPT.pptx` | 演示文稿 |
| `框架.png` | 论文大纲图 |
| `paper/` | 参考文献（~20 篇） |
| `PREVIOUS/` | 归档的早期探索（一般不相关） |

服务器：`/home/xuzhenyu/Kimi-Audio/`（代码）、`/data3/xuzhenyu/dataset from TTS/`（音频）

## F. 更新日志

| 日期 | 更新内容 |
|------|----------|
| 2026-02-06 | 创建本文件 |
| 2026-02-08 | 更新论文大纲，Observation 前置 |
| 2026-02-14 | 精简第二部分（删公式推导、删早期探索、合并分块），聚焦第三部分至 Observation |
| 2026-02-15 | Observation 从三节合并为两节（去掉 2.3）；2.2 重定位为纯机理（去除攻击数据）；更新文件索引与执行计划 |

---

# 第三部分：执行计划

## H. 完成论文 Section 2 Observation

目标：形成纯机理驱动的叙事链 `音频内冲突机理 → 跨模态冲突仲裁机理 → （过渡至 Threat Model 与方法论）`。

C 节发现与论文小节的映射：
- C 节 Probe 层级结构 + Logit Lens + Activation Patching → 支撑 2.1（ALLM 如何处理语义-韵律冲突）
- C 节 Prompt-Audio 冲突实验 → 支撑 2.2（文本指令如何覆盖音频情绪信号）
- 2.1–2.2 的机理发现 → 自然过渡至 Section 3 Threat Model 与 Section 4 方法论设计

**2.1 音频内模态冲突机理（语义 vs 韵律）**
- 已有数据：Probe 层级结构、Logit Lens 决策轨迹、Activation Patching 因果证据
- 待办：Probe 稳健性验证（交叉验证、随机标签对照）、Logit Lens 词表敏感性测试、负对照实验（非冲突样本）、统计显著性（bootstrap CI）
- 产出：可写入论文正文的图表与结论

**2.2 跨模态冲突仲裁机理（文本指令 vs 音频信号）**
- 已有数据：Prompt-Audio 冲突实验（三组对照的 Logit Lens 差分 + Activation Patching）
- 待办：文本主导性量化指标（Text-Dominance Index）、不同指令复杂度对中层主导性的影响、仲裁固化不可逆性验证、跨模型复现
- 产出：两级模态优先级的因果证据链 + 过渡至 Section 3

**完成判据**：两节内容形成递进（音频内→跨模态），每条结论有实验证据支撑，不包含攻击方法或攻击结果的引用，可直接写入论文正文。
