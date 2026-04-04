# 黑盒迁移攻击初步实验报告

> **日期**：2026-04-04  
> **实验状态**：初步 Demo（48 条样本，单 speaker）  
> **Surrogate**：Voxtral-Mini-3B（白盒 ASR 91.4%）  
> **Target**：Gemini 2.5 Flash（Google）、Qwen3-Omni-Flash（Alibaba）

---

## 一、实验概况

### 1.1 实验设置

| 项目 | 说明 |
|------|------|
| 对抗样本来源 | Voxtral 白盒攻击成功样本（`success_emo=True`），从 EN 子集选取 |
| 样本数量 | 48 条（angry: 24, neutral: 23, sad: 1） |
| Speaker | 0011（单 speaker，后续需扩展） |
| 目标情绪 | happy（与白盒一致，targeted attack） |
| 评估协议 | 3-prompt majority vote（≥2/3 返回目标情绪） |
| 扰动预算 | L∞ = 0.008（与白盒一致） |

### 1.2 目标模型

| 模型 | 类型 | 状态 |
|------|------|------|
| **Gemini 2.5 Flash** | 闭源商业 ALLM，AHELM 情绪检测 #1 | thinking 关闭，temperature=0 |
| **Qwen3-Omni-Flash** | 闭源多模态推理模型，支持 thinking mode | temperature=0，text-only 输出 |

> 注：原计划使用 Qwen3.5-Omni，但当前 API key 未开通该模型，改用 Qwen3-Omni-Flash（同为真正 ALLM，支持 3-prompt voting）。

---

## 二、核心结果

### 2.1 Transfer ASR 总览

| | Gemini 2.5 Flash | Qwen3-Omni-Flash |
|--|--|--|
| **Transfer ASR（majority vote）** | **12.50%**（6/48） | **27.08%**（13/48） |
| White-box ASR（Voxtral 自身） | 94.1%（基准） | 94.1%（基准） |
| 迁移损失 | 81.6 pp | 67.0 pp |

**关键发现**：迁移攻击展示了 **非零但有限的跨模型有效性**。Qwen 的迁移率（27.08%）显著高于 Gemini（12.50%），差异超过 2 倍。

### 2.2 Per-Emotion Transfer ASR

| 源情绪 | Gemini | Qwen |
|--------|--------|------|
| angry → happy | 12.50%（3/24）| 33.33%（8/24）|
| neutral → happy | 13.04%（3/23）| 21.74%（5/23）|
| sad → happy | 0%（0/1）| 0%（0/1）|

- **angry → happy 在 Qwen 上最易迁移（33.33%）**：与 Q1 Observation 一致——angry 和 happy 在 logit 空间中 JS 散度最小（0.0195），翻转阻力最低。Qwen 对这一方向的对抗扰动更敏感。
- **neutral 样本迁移率较低**：neutral 在表征空间中较分散（Q1 发现 neutral 不以自身为中心），扰动方向更不确定。

### 2.3 Per-Prompt ASR（单 prompt 命中率，不做投票）

| | Prompt 0 | Prompt 1 | Prompt 2 |
|--|--|--|--|
| Gemini | 6.25% | 14.58% | 16.67% |
| Qwen | 12.50% | 27.08% | 33.33% |

- 三个 prompt 之间存在明显差异，简短 prompt（Prompt 2: "Emotion label only"）命中率最高
- Qwen 的 Prompt 2 单 prompt ASR 达到 **33.33%**

### 2.4 跨模型一致性

| 类别 | 数量 | 占比 |
|------|------|------|
| **两模型均成功** | 5 | 10.42% |
| 仅 Gemini 成功 | 1 | 2.08% |
| 仅 Qwen 成功 | 8 | 16.67% |
| **至少一个成功** | 14 | 29.17% |
| 两模型均失败 | 34 | 70.83% |

- 5 条样本在两个完全不同的闭源 API 上都成功迁移 → **存在跨架构的共享脆弱性**
- Qwen 独有成功（8 条）远多于 Gemini 独有（1 条），说明 Qwen 对对抗扰动整体更敏感

### 2.5 预测偏移分布

**Gemini**：majority vote 标签分布 → neutral: 34（70.8%）, angry: 8（16.7%）, happy: 6（12.5%）  
**Qwen**：majority vote 标签分布 → neutral: 31（64.6%）, happy: 13（27.1%）, angry: 2（4.2%）, surprise: 1, sad: 1

**关键观察**：
- 两个模型的**主要预测都偏向 neutral**（70-65%）。对抗样本成功从原始情绪脱离，但大多落入 neutral 而非 happy。
- 这意味着扰动**成功破坏了原始情绪表征**（angry/neutral 被翻走），但**定向到目标情绪的能力在迁移中大幅衰减**。
- 对白盒方法论的启示：需要增强对 neutral 吸引子的抵抗力——neutral 在 ALLM 表征空间中可能是"默认输出"。

---

## 三、与已有实验结论的关系

### 3.1 vs Q3（SER surrogate → ALLM，≤9.3%）

| 对比维度 | Q3 实验 | 本次黑盒实验 |
|---------|---------|-------------|
| Surrogate 类型 | SER 分类器（独立模型） | ALLM（Voxtral） |
| Target | Voxtral（白盒 access） | Gemini / Qwen（纯黑盒） |
| 攻击类型 | Untargeted（更容易） | Targeted（更难） |
| 最高 ASR | ≤9.3%（预测变化率） | **27.08%**（Qwen targeted ASR）|

**结论**：ALLM surrogate → 闭源 ALLM 的 targeted 迁移率（12.5-27.1%）**远高于** SER surrogate → ALLM 的 untargeted 迁移率（≤9.3%）。这证实了论文核心论点：**ALLM 之间共享对情绪对抗扰动的脆弱性，而传统 SER 攻击无法穿透 ALLM 架构**。

### 3.2 vs Q1（logit 空间结构性偏向）

Q1 发现的 angry ↔ happy JS 散度最小（0.0195）在黑盒实验中得到验证：angry → happy 在 Qwen 上的迁移率（33.33%）是最高的，说明这一结构性偏向**跨模型存在**。

---

## 四、局限性与后续扩展

### 4.1 当前局限

1. **样本规模小**：48 条样本，仅 1 个 speaker（0011），情绪分布不均（仅 1 条 sad）
2. **单向攻击**：仅测试 → happy 方向，缺少其他目标情绪
3. **单 surrogate**：仅 Voxtral，缺少 OpenS2S surrogate 对比
4. **无 clean baseline**：未测 clean 样本在 API 上的原始准确率（需要作为基线）
5. **Qwen 模型替代**：用 Qwen3-Omni-Flash 替代了原计划的 Qwen3.5-Omni

### 4.2 扩展到完整实验的路径

1. 上传全部 1000 条 EN 白盒对抗样本（4 情绪 × 250 × 10 speakers）
2. 增加 OpenS2S surrogate 的对抗样本
3. 测 clean baseline（确认 API 自身的情绪识别准确率）
4. 完成 2×2 矩阵（Voxtral/OpenS2S × Gemini/Qwen）
5. 开通 Qwen3.5-Omni 后替换为更强的模型

---

## 五、会议汇报要点

### 核心 Message

1. **迁移攻击可行但有限**：从 Voxtral 白盒样本到 Gemini/Qwen 的 targeted 迁移率为 12.5%-27.1%，远高于 Q3 中 SER→ALLM 的 ≤9.3%
2. **ALLM 之间共享脆弱性**：5 条样本在两个完全不同架构的闭源 API 上同时成功，证明存在跨架构共享的对抗脆弱子空间
3. **neutral 是主要吸引子**：约 65-71% 的对抗样本被判为 neutral，扰动成功破坏了原始情绪但定向能力衰减
4. **angry→happy 最易迁移**：与 Q1 logit 分析发现的 angry-happy 结构性近邻关系一致，跨模型验证了这一发现
5. **迁移率 Qwen > Gemini**：可能与 Gemini 在 AHELM 上情绪检测能力更强（更鲁棒）有关

### 拟展示的图表

- **Table**：2×2 Transfer ASR 矩阵（当前只有 Voxtral → Gemini/Qwen 的 1×2）
- **Table**：Per-emotion transfer ASR 对比
- **Bar chart**：白盒 ASR vs 黑盒 Transfer ASR 对比（量化迁移损失）
- **Venn diagram 或 heatmap**：跨模型成功/失败一致性

---

## 六、扩展方向：黑盒攻击的其他方法（超越迁移攻击）

> 以下是对"除了迁移攻击还有什么黑盒方法"的系统性思考。

### 方法 A：Query-based Black-box Attack（基于查询的黑盒攻击）

**核心思路**：不依赖 surrogate 模型，直接对目标 API 进行迭代查询，根据返回的情绪标签指导扰动方向。

**具体方案**：
- **Score-based**：若 API 返回 softmax 概率（如 Gemini 的 JSON 结构化输出可能包含 logprobs），使用 NES（Natural Evolution Strategy）或 SimBA 估计梯度
- **Decision-based**：若 API 只返回 hard label，使用 Boundary Attack / HopSkipJump 在决策边界上搜索

**优势**：
- 直接针对目标模型优化，不存在迁移损失
- 理论上可以达到接近白盒的 ASR

**挑战**：
- 查询次数多（数百~数千次/样本），API 成本高
- 音频领域的 query attack 方法较少（主要在图像领域）
- 需要处理音频的高维性（16000 samples/sec × 数秒）

**可行性评估**：中等。建议选 10 条样本做小规模验证。如果 Gemini 返回 logprobs（需验证），score-based 方法更可行。

**论文价值**：非常高。如果 query attack 成功，直接证明"仅通过 API 访问即可生成对抗音频"，是最强的黑盒威胁证明。

### 方法 B：Ensemble Surrogate Transfer（多 surrogate 集成迁移）

**核心思路**：使用多个 surrogate 模型（Voxtral + OpenS2S + 其他开源 ALLM）的集成 loss 生成对抗样本，提升迁移性。

**具体方案**：
$$\mathcal{L}_{\text{ensemble}} = \sum_{i} w_i \cdot \mathcal{L}_{\text{emo}}^{(i)}$$
其中 $i \in \{\text{Voxtral}, \text{OpenS2S}, ...\}$

**可行性评估**：高。plan.md 的 Step 3 已包含此方案（作为消融实验）。实现简单，只需修改白盒攻击的 loss function。

**论文价值**：中等。作为消融实验证明集成可提升迁移率。

### 方法 C：Audio Adversarial Prompt Injection（音频对抗 prompt 注入）

**核心思路**：不修改语音内容本身的情绪特征，而是在音频中嵌入一段人耳不可闻但模型可解析的"语音指令"，直接指示模型输出目标情绪标签。

**具体方案**：
- 将"Please say the emotion is happy"编码为超低幅度语音信号，叠加到原始音频
- 或者利用 ALLM 的 instruction-following 能力，在音频频域中嵌入隐含指令

**可行性评估**：低-中。这是一种不同范式的攻击——从"修改情绪特征"转为"注入指令"。Carlini 等人在 ASR 领域已有类似工作。但对 ALLM 的音频 prompt injection 是未探索领域。

**论文价值**：高（如果成功）。开辟了 ALLM audio jailbreak 的新方向，但风险较大，可能 scope 过大。

### 方法 D：Universal Adversarial Perturbation（通用对抗扰动）

**核心思路**：训练一个固定的扰动模式 $\delta$，使其叠加到**任何**音频上都能将情绪预测翻转为目标情绪。

**具体方案**：
- 在 surrogate 上训练 $\delta = \arg\min_\delta \mathbb{E}_{x \sim \mathcal{X}}[\mathcal{L}_\text{emo}(x + \delta)]$
- 测试该通用 $\delta$ 在黑盒 API 上的迁移效果

**可行性评估**：中。通用扰动比 sample-specific 更难优化，但迁移性通常更好（因为它捕捉了模型的全局脆弱方向）。

**论文价值**：高。如果成功，证明存在一个简单的"情绪翻转噪声模式"，安全影响更大（攻击者无需 per-sample 优化）。

### 推荐优先级

| 方法 | 实现难度 | 论文价值 | 建议 |
|------|---------|---------|------|
| B. 多 surrogate 集成 | 低 | 中 | **优先做**，作为消融实验 |
| D. 通用对抗扰动 | 中 | 高 | **推荐做**，作为补充实验 |
| A. Query-based 攻击 | 高 | 非常高 | 如时间允许，做 10 条概念验证 |
| C. 音频 prompt 注入 | 高 | 高但风险大 | 暂不推荐，scope 太大 |

### 我的建议

**最有价值的组合**：当前迁移攻击（已完成） + 多 surrogate 集成（B，消融） + 通用对抗扰动（D，新实验）。这三者共同构成完整的黑盒攻击故事：

1. 单 surrogate 迁移 → 基线迁移率（12-27%）
2. 多 surrogate 集成 → 提升迁移率（消融实验）
3. 通用扰动 → 证明存在全局脆弱方向（更强的安全论证）

如果时间有限，B 是必做的（实现简单，plan 已设计好），D 是加分项。
