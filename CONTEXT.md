# AI 协作指导文件

> **最后更新**：2026-03-28
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

研究主线分三个阶段：（1）**机理分析**——观察 ALLM 处理情绪信息的内部机制，揭示可利用的结构性特点；（2）**白盒攻击**——基于机理发现设计针对性攻击方法，攻击者可完全访问模型权重与中间表征；（3）**黑盒攻击**——在仅有 API 访问的条件下验证攻击的泛化能力。

目标模型：OpenS2S、Kimi-Audio、Voxtral。

| 阶段     | 状态                                                                                                    |
| -------- | ------------------------------------------------------------------------------------------------------- |
| 机理分析 | v3 四问结构；Q1/Q2/Q3 已完成，Q4 待执行                                                                 |
| 白盒攻击 | 实验完成（OpenS2S / Voxtral）；方法论 v1，待据 Q4 启示迭代                                              |
| 黑盒攻击 | 尚未开始                                                                                                |
| 论文写作 | `finalpaper/` Observation 基本完成（Q1/Q2/Q3 已写入）；本周优先：Threat Model + 方法论重构 + 白盒实验 |

## B. 机理分析（Observation v3，Q1-Q4 四问结构）

围绕四个递进问题展开，从可行性→后果→传统方法局限→内部机理，逐步为攻击设计提供动机与约束。

### Q1: 音频情绪可以翻转吗？  已完成  已写入论文

> Qwen2-Audio-7B-Instruct, audio-only, 400 样本（4 情绪 × 100）

统计 clean 音频的 emotion token logit distribution，量化翻转可行性。包含 logit 层面的全套分析：token probability distribution、稳定性统计、prototype posterior、prototype similarity（JS/KL/Cosine）、prediction confusion matrix、first-step logit 三分版（目标/其他情绪/无关 token）。

**关键发现**：

- 模型情绪输出不是 one-hot，而是 structured preference distribution，各情绪 token 间存在竞争
- 各情绪稳定性高度不对称：angry/sad 较稳定（候选集准确率 61%/63%），**happy 极脆弱**（准确率仅 15%，85% 样本目标分数低于最强 competitor）
- **angry 是万能 competitor**：happy、neutral、sad 的主要竞争者均为 angry
- Prototype 相似性：angry↔happy JS 散度最小（0.0195）→ 翻转阻力最小
- Prototype posterior：happy 偏向 angry（angry 0.460 > happy 自身 0.140）；neutral 不以自身为中心（angry 0.355 > neutral 0.251）
- Prediction confusion：happy 50% 被判为 angry，neutral 在 angry/sad 间分裂 → 存在 **anger-oriented overlap region**
- 对攻击设计的启示：不宜只用 hard target CE → 需 hybrid margin + soft distribution steering + pair-specific 策略

**结论**：翻转可行，且不同方向难度高度不对称。clean posterior 空间存在结构性偏向（anger-oriented overlap），为攻击方法论设计提供约束。

- 交付物：`observation_v3/Q1&Q4/emotion_token_analysis_reference_zh.pdf`（含小提琴图、稳定性表、prototype posterior、confusion matrix、JS similarity、first-step logit 三分版）
- 论文：`finalpaper/2.observation.tex` §2.1 "Clean emotion recognition is governed by structured non-target preference"（含 Fig 1 candidate probability + Fig 2 first-step logit histograms）

### Q2: 翻转了有什么后果？  已完成  已写入论文

> Voxtral-Mini-3B, Aligned vs Conflict prompt, 20 样本（5 情绪 × 4）, DeepSeek V3 LLM Judge

通过文本 Prompt 操纵情绪感知（不依赖攻击方法），验证情绪误感知对下游回复的系统性影响。

**关键发现**：

- 三维度系统性退化：Faithfulness Δ+1.30, Empathy Δ+1.20, Relevance Δ+1.05
- **happy 最脆弱**（Empathy Δ+3.00, Relevance Δ+2.50）/ **sad 完全鲁棒**（三维度 Δ=0）/ **neutral 是幻觉温床**（Faithfulness 低至 1.50/5）
- **surprised** 也高度脆弱（Faithfulness Δ+2.50, Relevance Δ+2.00）
- 强负面情绪音频信号可抵抗 Prompt 误导，但 happy/neutral/surprised 无法抵抗

**结论**：情绪翻转不只是标签变化，会系统性污染下游回复，为攻击提供 threat justification。

- 交付物：`observation_v3/Q2/experiment/result/ob3_analysis_report.md`
- 论文：`finalpaper/2.observation.tex` §2.2（含 Table 1）

### Q3: 传统 SER 攻击方法对 ALLM 有效吗？  已完成  部分写入论文

> Voxtral-Mini-3B, 四种 SER 攻击方法迁移评估, ESD 数据集

将 SER 领域的四种代表性对抗攻击方法迁移到 ALLM（Voxtral），评估翻转效果。所有方法均为 untargeted 攻击（比 targeted 更容易），给予白盒 access。

**四种方法及结果**：

| 方法         | 类型                            | 扰动预算    | SER surrogate ASR | Voxtral 预测变化率        |
| ------------ | ------------------------------- | ----------- | ----------------- | ------------------------- |
| STAA-Net     | 生成器 (Wave-U-Net) + C&W loss  | ε=0.03 L∞ | 77% (train)       | **2.5%** (200 样本) |
| PGD          | 白盒迭代梯度攻击                | ε=0.03 L∞ | 100%              | ≤9.3%                    |
| GAO (3 变体) | 黑盒语音畸变 (VTLN/McAdams/MSS) | —          | —                | 4.2–9.3%                 |
| REN          | CNN 生成器 (Atrous)             | ε=0.03 L∞ | 70%               | ≤9.3%                    |

**关键发现**：

- 尽管 PGD 在 SER surrogate 上达到 100% ASR，四种方法迁移到 Voxtral 后实际预测变化率均 ≤9.3%
- STAA-Net 在 4× 扰动预算（0.03 vs 0.008）+ 更容易的 untargeted 任务下，仅使 2.5% 样本预测改变
- 失败原因非扰动不足，而是架构不兼容：SER 攻击针对独立分类器，ALLM 通过 encoder-adapter-LLM pipeline 自回归生成情绪判断，扰动信号在 LLM 的十亿参数中耗散

**结论**：传统 SER 攻击方法论即使直接适配 ALLM（给予白盒 access），也几乎无法改变 ALLM 的情绪预测。→ 论证 ALLM-native 攻击方法的必要性。

- 交付物：`observation_v3/Q3/`（STAA/PGD/GAO/REN 四个子目录，各含代码、结果、报告）
- 论文：`finalpaper/2.observation.tex` §2.3（STAA-Net 已写入含 Table 2；其余 3 种方法待整合，计划用一张图展示）

### Q4: 具体的机理怎么实现？ ⬜ 待执行

> 计划使用之前已完成的 Probing 实验框架（`code/modal_conflict/`）

通过 Probing 分析 hidden state 中不同情绪的可分离性，揭示 ALLM 内部情绪表征的形成机理，解释 Q1 观察到的不对称性的深层原因。

**实验目标**：

1. **Representation Probing**：对各层 hidden state 训练线性探针，测量不同情绪在表征空间中的可分性
2. **情绪对难度分析**：量化不同情绪对（如 happy↔angry vs sad↔angry）在表征空间中的分离难度差异

**预期价值**：为 Q1 的 logit 层面发现（anger-oriented overlap、happy 脆弱性）提供表征层面的因果解释——不仅知道"翻转容易"，还要解释"为什么容易"。

- 交付物：Representation Probing Results + 不同情绪对难度分析

## C. 白盒攻击

通过对输入音频波形施加扰动（L∞ ≤ 0.008，Adam + 每步 L∞ 投影）实现情绪定向误判。损失函数：

$$
\mathcal{L}(x') = \lambda_{\text{emo}} \mathcal{L}_{\text{emo}} + \lambda_{\text{asr}} \mathcal{L}_{\text{asr}} + \lambda_{\text{per}} \mathcal{L}_{\text{per}}
$$

- **L_emo**：对模型情绪输出 token 做 teacher-forcing CE（直接优化模型自身 logits，非外部分类器）
- **L_asr**：同一模型在转写 Prompt 下做自一致约束，以干净音频基准转写为目标，保持语义
- **L_per**：多分辨率 STFT 幅度差，控制感知可闻性

**两阶段调度**：Stage A（20 步）λ_emo 主导先翻转情绪；Stage B（40 步）逐步增大 λ_asr / λ_per 兼顾语义与可闻性。**多 Prompt 集成**（3 个情绪 Prompt 损失求平均）提升跨 Prompt 鲁棒性；**EoT**（随机时移 + 增益扰动）增强稳健性。

**实验结果**：

| 指标                            | OpenS2S（ESD, 8949 条） | Voxtral（ESD EN+CN, 2000 条） |
| ------------------------------- | ----------------------- | ----------------------------- |
| 攻击成功率（3-prompt 多数投票） | 78.44%                  | **93.80%**              |
| 单 Prompt 最高                  | 94.66%                  | 99.90%                        |
| 语义保留率                      | 19.43%（LLM Judge）     | 39.75%（Cosine Sim）          |
| 联合成功率（情绪✓ ∧ 语义✓）  | 15.07%                  | 36.40%                        |
| 平均 SNR                        | 16.43 dB                | 20.60 dB                      |

**当前瓶颈**：情绪翻转成功率高（78–94%），但语义保持率是联合成功率的限制因素。Voxtral 显著优于 OpenS2S，可能与架构差异有关。

> **注**：当前为 v1 方法论。Q4 实验提示需引入 soft distribution steering / pair-specific 策略，方法论将据此迭代。

## D. 论文写作进度

论文 LaTeX 源码位于 `finalpaper/`。Observation 部分基本完成，本周优先推进 Threat Model、方法论重构、白盒实验部分。

```
1. Introduction                                          ⬜ 未开始
2. Background                                            ⬜ 未开始（
3. Observation（三个递进问题，为攻击方法提供动机与约束）
   3.1 Q1: structured non-target preference              ✅ 已完成（含 Fig 1 + Fig 2）
   3.2 Q2: 情绪误感知的下游传导与幻觉                  ✅ 已完成（含 Table 1）
   3.3 Q3: 传统 SER 攻击方法的不充分性                  🔧 STAA-Net 已写入（含 Table 2），其余 3 方法待整合
   (Q4 Probing 实验待执行，完成后决定是否作为独立小节)
4. Threat Model                                          🔧 本周优先
5. 方法论（由 Observation 推导的设计约束驱动）           🔧 本周优先（方法论重构）
6. 实验（Setting / 白盒结果 / 黑盒结果 / 回复评估 / Defense）  🔧 本周优先（白盒实验部分）
7. Related Work                                          ⬜ 未开始
8. Discussion and Limitations                            ⬜ 未开始
附录                                                     ⬜ 未开始
```

## E. 文件索引

| 路径                                           | 内容                                                                                                                               |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `observation/` (v1)                          | **已替代**：v1 Observation（按工具组织：Probe/Logit Lens/Patching），含 LaTeX 初稿、配图、实验脚本与结果                     |
| `observation_v2/`                            | **已替代**：v2 Observation（R1-R4 映射），过渡版本                                                                           |
| `observation_v3/`                            | **当前版本**：v3 Observation（Q1-Q4 四问结构）；Q1/Q2/Q3 已完成，Q4 待执行                                                   |
| `observation_v3/Q1&Q4/`                      | Q1 实验交付物：`emotion_token_analysis_reference_zh.pdf`（logit 分析、prototype、confusion matrix）；Q4（Probing）待执行         |
| `observation_v3/Q2/`                         | Q2 实验：Voxtral Aligned vs Conflict 推理对比，含推理脚本、评估脚本、分析报告                                                      |
| `observation_v3/Q3/`                         | Q3 实验目录（已完成）：四种 SER 方法迁移评估                                                                                       |
| `observation_v3/Q3/STAA/`                    | STAA-Net 方法迁移：Wave-U-Net 生成器 + C&W loss，200 样本，2.5% 预测变化率                                                         |
| `observation_v3/Q3/PGD/`                     | PGD 白盒迭代攻击迁移：surrogate SER 100% ASR，Voxtral ≤9.3% 变化率                                                                |
| `observation_v3/Q3/GAO/`                     | GAO 黑盒语音畸变（VTLN/McAdams/MSS）：4.2–9.3% 变化率                                                                             |
| `observation_v3/Q3/REN/`                     | REN CNN 生成器 (Atrous) 攻击迁移：surrogate 70% ASR，Voxtral ≤9.3% 变化率                                                         |
| `code/white_box_voxtral/result/`             | **Voxtral 白盒批量实验结果**：`report_all.md`（2000 条，ASR 93.80%）                                                       |
| `finalpaper/`                                | **论文 LaTeX 源码（当前版本）**：`main.tex`（主文件）+ `2.observation.tex`（Section 2）                                  |
| `PREVIOUS/2OBSERVATION/`                     | **已归档**：旧版 Observation 写作（P0+P1 修订版）                                                                            |
| `LATEST/white_box_final/PPT大纲.md`          | PPT 全文案                                                                                                                         |
| `LATEST/white_box_final/PPTtext.md`          | PPT 解析版                                                                                                                         |
| `LATEST/white_box_final/audio内部机理1/`     | 实验素材：音频内部机理第一阶段                                                                                                     |
| `LATEST/white_box_final/audio内部机理2/`     | 实验素材：音频内部机理第二阶段                                                                                                     |
| `LATEST/white_box_final/prompt&audio机理/`   | 实验素材：Prompt-Audio 冲突机理                                                                                                    |
| `LATEST/white_box_final/白盒对抗样本方法论/` | 实验素材：白盒攻击方法论                                                                                                           |
| `LATEST/白盒讲稿.md`                         | 汇报讲稿                                                                                                                           |
| `LATEST/情绪LLM白盒攻击研究.pptx`            | 完整 PPT                                                                                                                           |
| `LATEST/情绪LLM白盒攻击研究.pdf`             | PPT 导出 PDF                                                                                                                       |
| `code/modal_conflict/`                       | **Probe 实验**：音频内模态冲突机理（语义 vs 韵律），对应 2.1 节。250 样本 × 36 层 × 5-fold GroupKFold，输出 dominance 曲线 |
| `code/logit_lens/`                           | **Logit Lens 实验**：逐层决策追踪，对应 2.1 节。197 冲突样本，输出 margin 曲线与 win-rate 曲线                               |
| `code/activation_patching/`                  | **Activation Patching 实验**：因果干预，对应 2.1–2.2 节。100 韵律对 + 100 语义对，输出 flip rate 与 delta logit 曲线        |
| `code/white_box_v2/`                         | **白盒攻击代码（当前版本）**：PGD+EoT 对抗攻击框架。`codex/` 为通用实验模板，`experiment/` 为特定数据集实验版本          |
| `code/white_box_v2/result/ESDfinal/`         | **OpenS2S 白盒批量实验结果**                                                                                                 |
| `code/white_box_v1/`                         | ~~已废弃，旧版方法论，请勿阅读~~                                                                                                  |
| `meeting/3.19.md`                            | 3.19 汇报 PPT 大纲（Q1-Q4 四问 + 白盒结果）                                                                                        |
| `meeting/3.19_text.md`                       | 3.19 汇报讲稿（逐页口述内容）                                                                                                      |
| `meeting/3.26.md`                            | 3.26 周会汇报（Q2/Q3 论文写作进展 + Q3 补充实验）                                                                                  |
| `PPT.pptx`                                   | 演示文稿                                                                                                                           |
| `框架.png`                                   | 论文大纲图                                                                                                                         |
| `paper/`                                     | 参考文献（~20 篇）                                                                                                                 |
| `PREVIOUS/`                                  | 归档的早期探索（一般不相关）                                                                                                       |

## F. 更新日志

| 日期       | 更新内容                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-02-06 | 创建本文件                                                                                                                                                                                                                                                                                                                                                                                                                       |
| 2026-02-08 | 更新论文大纲，Observation 前置                                                                                                                                                                                                                                                                                                                                                                                                   |
| 2026-02-14 | 精简第二部分（删公式推导、删早期探索、合并分块），聚焦第三部分至 Observation                                                                                                                                                                                                                                                                                                                                                     |
| 2026-02-15 | Observation 从三节合并为两节（去掉 2.3）；2.2 重定位为纯机理（去除攻击数据）；更新文件索引与执行计划                                                                                                                                                                                                                                                                                                                             |
| 2026-02-21 | 新增实验代码文件索引（modal_conflict / logit_lens / activation_patching / white_box_v2）                                                                                                                                                                                                                                                                                                                                         |
| 2026-02-24 | Observation 完成 P0+P1 修订：新叙事框架（表征解耦与因果不对称）、claim 收缩、过渡段重写、方法论声明、中英文统一                                                                                                                                                                                                                                                                                                                  |
| 2026-03-01 | Observation 初稿完成（OPUS→observation）；旧版 2OBSERVATION 归档至 PREVIOUS；OpenS2S 批量实验完成（结果在 code/white_box_v2/result/ESDfinal/）；下一步目标：Voxtral 模型批量实验                                                                                                                                                                                                                                                |
| 2026-03-11 | 重构第二部分（A-D 节全面更新）：新增 Voxtral 至目标模型、B/C 节对调并更新内容（B 对齐 observation_final.tex，C 更新为当前方法论与 ESDfinal 实验结果）、删除当前焦点与执行计划                                                                                                                                                                                                                                                    |
| 2026-03-18 | Observation 重构至 v3（Q1/Q2/Q3 + 4 个 Observation）；Q2 实验完成；B 节重写为问题导向架构；论文大纲与文件索引同步更新                                                                                                                                                                                                                                                                                                            |
| 2026-03-19 | B 节重写为 Q1-Q4 四问结构：Q1 合并全部 logit 层面分析（含 prototype/confusion），Q4 恢复为 Probing hidden state 计划（待执行）；C 节补充 Voxtral 实验结果；D 节论文大纲新增 Q4；E 节新增文件索引                                                                                                                                                                                                                                 |
| 2026-03-28 | Q3 实验完成（四种 SER 方法迁移：STAA-Net/PGD/GAO/REN，预测变化率均 ≤9.3%）；Q1 完整写入论文（标题 "structured non-target preference"，含 2 图）；Q2 论文版更新（N=20，5 情绪含 surprised）；新增 `finalpaper/` 论文 LaTeX 目录；论文结构新增 Background 节（在 Observation 前）；D 节更新为论文写作进度（Observation 基本完成，本周优先 Threat Model / 方法论重构 / 白盒实验）；E 节新增 Q3 子目录、finalpaper、3.26 会议索引 |
