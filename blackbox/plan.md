# 黑盒攻击实验计划

> **状态**：Demo 代码已完成  
> **最后更新**：2026-04-04  
> **关联论文节**：§6 Black-box Transferability（待写）

---

## 修改记录（2026-04-04）

### 修改 1：Qwen3-ASR-Flash → Qwen3.5-Omni

**原方案**：使用 Qwen3-ASR-Flash（结构化 7 类情绪标签）

**修改为**：使用 Qwen3.5-Omni-Flash（真正的 ALLM 多模态推理模型，当前免费 preview）

**理由**：
- Qwen3-ASR-Flash 本质上是 ASR 模型附加情绪元数据，不是端到端音频推理模型，与论文"攻击 ALLM"主题不一致
- Qwen3-ASR-Flash 的情绪标签不受 prompt 控制，3-prompt majority vote 方案失去意义（每次调用返回相同结果）
- Qwen3.5-Omni 是真正的多模态推理模型，支持 prompt 控制输出，且当前免费

### 修改 2：Step 1 简化

**原方案**：对 500 条 ESD-EN 样本重新运行白盒攻击

**修改为**：直接从 `code/white_box_voxtral/result/Voxtral_EN/` 已有 1000 条白盒结果中选取子集

**理由**：Voxtral EN 白盒实验已完成（1000 条，ASR 91.4%），对抗 WAV 文件已生成，无需重复劳动

### 修改 3：标签映射

Qwen3.5-Omni 返回自由文本而非结构化标签，通过 prompt 约束输出格式 + regex 解析提取情绪词

### Demo 代码结构

```
blackbox/
├── config.py          # 配置（API key 环境变量、路径、prompts、标签映射）
├── gemini_client.py   # Gemini 2.5 Flash REST API 客户端
├── qwen_client.py     # Qwen3.5-Omni OpenAI-compatible 客户端
├── sample_loader.py   # 从白盒结果加载对抗样本元数据 + 选取子集
├── evaluate.py        # 主评估脚本（3-prompt voting + 结果汇总）
├── plan.md            # 本文件
├── api_research.md    # API 调研报告
└── results/           # 输出目录
```

### 运行方式

```bash
# 安装依赖
pip install requests openai

# 设置 API key
export GEMINI_API_KEY="your-key"
export DASHSCOPE_API_KEY="your-key"

# 预实验（dry run，不调用 API）
python evaluate.py --target gemini --num_samples 10 --dry_run

# 运行 Gemini 评估（10 条样本）
python evaluate.py --target gemini --num_samples 10

# 运行 Qwen 评估
python evaluate.py --target qwen --num_samples 10

# 完整 500 条评估
python evaluate.py --target gemini --per_emotion 125
```

---

## 一、定位与动机

### 在论文中的角色

白盒攻击（§5）已证明：当攻击者可访问模型权重与梯度时，能以 93.80%（Voxtral）和 78.44%（OpenS2S）的成功率实现定向情绪翻转。然而，现实威胁场景中，攻击者通常只能通过 API 提交音频并观察输出标签，无法获取任何梯度信息。

黑盒实验需要回答：**白盒攻击生成的对抗样本，能否在完全封闭的商业 API 上保持攻击效果？**

这一问题的答案直接决定了本研究的现实威胁等级。如果迁移攻击有效，则意味着攻击者无需接触目标模型内部即可实施攻击，安全威胁被显著放大。

### 与 Q3 实验的区分

Q3（§2.3）已经展示了"SER 分类器作为 surrogate → ALLM 目标"的迁移失败（≤9.3%）。本实验的关键区别是：**surrogate 本身也是 ALLM**（Voxtral / OpenS2S）。这一设定下的迁移率反映的是 ALLM 之间对情绪对抗扰动的共享脆弱性，而非 SER→ALLM 架构跨越带来的失败。

---

## 二、实验设计

### 2.1 目标模型（黑盒端）

选择**两个真正闭源的商业 API**，覆盖国际与国内主流生态：

| 目标模型 | 提供方 | 选择理由 |
|----------|--------|----------|
| **Gemini 2.5 Flash** | Google | AHELM 情绪检测 benchmark #1（win rate 0.803）；官方明确支持结构化情绪 JSON 输出；价格最低（~$1/1M tokens）；全球 REST API |
| **Qwen3-ASR-Flash** | Alibaba | 直接返回 7 类结构化情绪标签（最易解析）；REST + WebSocket 全球可访问；覆盖国内主流生态 |

**不选 GPT-4o Audio**：成本为 Gemini 的 40 倍（~$40/1M tokens），情绪输出为自由文本而非结构化标签，不适合大规模批量实验。

**不选 Kimi-Audio**：权重开源，不构成真正黑盒场景，论文说服力不足。

### 2.2 Surrogate 模型（白盒端）

使用已有的两个本地白盒模型：

| Surrogate | 代码路径 | 白盒 ASR |
|-----------|----------|----------|
| Voxtral-Mini-3B | `code/white_box_voxtral/` | 93.80% |
| OpenS2S | `code/white_box_opens2s_v2/` | 78.44% |

### 2.3 数据集

- **来源**：ESD（Emotional Speech Dataset）
- **子集**：英文（EN）子集，保证 Gemini 和 Qwen 均可正常处理
- **规模**：500 条，覆盖 4 种源情绪（angry / neutral / sad / surprise），每类 125 条
- **选取原则**：从白盒 Voxtral 实验已跑过的 EN 样本中取子集，确保白盒 ASR 有对应基准可比较

> **待确认**：白盒 Voxtral 实验（EN+CN 各 1000 条）中 EN 子集具体分布，确认每类情绪可取到 ≥125 条。

### 2.4 情绪标签映射

ESD 使用 5 类标签（happy / angry / neutral / sad / surprise），但白盒实验排除了 happy（`esd_exclude_emotion: "happy"`），实际攻击源情绪为 4 类。

各目标 API 的返回标签与 ESD 标签的映射关系：

| ESD 标签 | Gemini 返回 | Qwen3-ASR 返回 | 备注 |
|----------|-------------|----------------|------|
| angry | Angry | anger | 直接映射 |
| neutral | Neutral | calm | calm → neutral |
| sad | Sad | sadness | 直接映射 |
| surprise | Surprised | surprise | 直接映射 |

> **待验证**：以上映射基于文档推断，需用 5-10 条 clean 样本预实验确认实际返回格式。

### 2.5 评估协议

与白盒实验保持一致，确保结果可直接比较：

- **情绪 Prompt**：复用白盒实验的 3 个情绪 elicitation prompts（来自 `config.py`）
- **成功判定**：3 个 Prompt 多数投票（≥2/3 返回目标情绪）
- **指标**：
  - **Transfer ASR**：迁移攻击成功率（主指标）
  - **Per-emotion Transfer ASR**：按源情绪分类的迁移成功率
  - **Surrogate ASR vs Transfer ASR**：两者相关性分析

语义保持不在黑盒评估范围内（黑盒 API 不提供转写，且对抗样本的语义已在白盒阶段评估）。

---

## 三、实验矩阵

主实验为 **2 surrogate × 2 target** 的全交叉矩阵，共 4 组：

| | Gemini 2.5 Flash | Qwen3-ASR-Flash |
|--|--|--|
| **Voxtral surrogate** | 组 A | 组 B |
| **OpenS2S surrogate** | 组 C | 组 D |

每组 500 条样本 × 3 Prompt = 1500 次 API 调用。总调用次数：4 组 × 1500 = **6000 次**。

### 成本估算

**Gemini 2.5 Flash**：
- ESD 英文样本平均时长约 3–5 秒
- 500 条 × 4 秒 × 32 tokens/秒 = 64,000 audio tokens/组
- 加上 prompt text tokens（约 50 tokens × 3 prompts）= 可忽略
- 2 组（A + C）× 64,000 tokens × $1/1M = **< $0.15**

**Qwen3-ASR-Flash**：
- 按官方计费，规模相近，估计 **< $1**

总成本估计：**< $2**（不考虑预实验）

---

## 四、执行步骤

### Step 0：环境确认与预实验

- [ ] **S0.1** 确认 EN 子集数据分布：从白盒 Voxtral 实验结果中统计 EN 子集各情绪样本数，确认 4 类各 ≥125 条
- [ ] **S0.2** Gemini API 预实验：取 10 条 clean ESD-EN 样本，调用 `generateContent`，确认：
  - 返回格式（JSON 结构 vs 文本）
  - 情绪标签枚举值
  - clean 样本上的情绪识别准确率（基线）
- [ ] **S0.3** Qwen3-ASR-Flash API 预实验：取同 10 条样本，确认：
  - 返回 JSON 中情绪字段名与值
  - clean 样本基线准确率
- [ ] **S0.4** 整理 500 条样本列表（文件路径 + 源情绪 + 目标情绪），确保两个 surrogate 用完全相同的样本集

### Step 1：生成对抗样本（本地，无 API 费用）

**Voxtral surrogate（组 A + B 所用）**：
- [ ] **S1.1** 基于已有 `code/white_box_voxtral/` 代码，对 500 条 ESD-EN 样本运行白盒攻击
- [ ] **S1.2** 记录每条样本的 surrogate ASR（Voxtral 上的白盒攻击成功与否），作为后续相关性分析的基准
- [ ] **S1.3** 保存对抗音频到 `blackbox/adv_audio/voxtral_surrogate/`

**OpenS2S surrogate（组 C + D 所用）**：
- [ ] **S1.4** 基于 `code/white_box_opens2s_v2/`，对同 500 条样本运行白盒攻击
- [ ] **S1.5** 记录 surrogate ASR
- [ ] **S1.6** 保存对抗音频到 `blackbox/adv_audio/opens2s_surrogate/`

### Step 2：黑盒 API 评估

**组 A（Voxtral surrogate → Gemini）**：
- [ ] **S2.1** 批量上传 `blackbox/adv_audio/voxtral_surrogate/` 中的 500 条对抗音频至 Gemini API
- [ ] **S2.2** 每条音频使用 3 个 emotion prompt，记录返回情绪标签
- [ ] **S2.3** 多数投票，计算 Transfer ASR

**组 B（Voxtral surrogate → Qwen）**：
- [ ] **S2.4–S2.6** 同上，目标改为 Qwen3-ASR-Flash

**组 C（OpenS2S surrogate → Gemini）**：
- [ ] **S2.7–S2.9** 同上，使用 OpenS2S 对抗音频

**组 D（OpenS2S surrogate → Qwen）**：
- [ ] **S2.10–S2.12** 同上

### Step 3：消融实验（视 Step 2 结果决定是否执行）

若任一组 Transfer ASR < 40%，执行多 surrogate 集成：

- [ ] **S3.1** 修改白盒攻击代码，支持双模型集成 loss：$\mathcal{L} = \mathcal{L}_{\text{Voxtral}} + \mathcal{L}_{\text{OpenS2S}}$
- [ ] **S3.2** 重新生成 500 条集成对抗样本
- [ ] **S3.3** 重测 Gemini 和 Qwen 迁移率，与单 surrogate 对比

### Step 4：数据分析

- [ ] **S4.1** 整理 2×2 Transfer ASR 矩阵（表格）
- [ ] **S4.2** 计算 per-emotion 迁移率，与白盒 per-emotion 结果（§5 Table 2）做对比
- [ ] **S4.3** Surrogate ASR 与 Transfer ASR 相关性分析（scatter plot）：验证"白盒收敛越快的样本迁移率越高"假设
- [ ] **S4.4** Gemini vs Qwen 迁移率差异分析（两个目标模型架构差异的讨论）

### Step 5：论文写作

- [ ] **S5.1** 新建 `finalpaper/6.blackbox.tex`，撰写 §6 Black-box Transferability
- [ ] **S5.2** 修改 `finalpaper/3.threat_model.tex`，新增黑盒 attacker knowledge 段落
- [ ] **S5.3** 在 `finalpaper/main.tex` 中 `\input{6.blackbox}`

---

## 五、论文节结构草案

```
§ 6  Black-box Transferability

6.1  Setup
     - 黑盒场景定义：attacker 仅观察 API 输出情绪标签，无权重、无梯度
     - 目标模型：Gemini 2.5 Flash（国际）、Qwen3-ASR-Flash（国内）
     - Surrogate：Voxtral / OpenS2S
     - 评估协议：500 条 ESD-EN，3-prompt majority vote

6.2  Transfer Attack Results
     - Table：2×2 Transfer ASR 矩阵 [Surrogate × Target]
     - 与白盒 ASR 的对比（迁移损失量化）
     - 与 Q3 SER surrogate 失败（≤9.3%）的对比（ALLM surrogate 的优越性）

6.3  Analysis
     - Per-emotion 迁移率表（与 §5 Table 2 对照）
     - Surrogate 选择对迁移率的影响（Voxtral vs OpenS2S as surrogate）
     - 多 surrogate 集成消融（若执行 Step 3）

6.4  Summary
     - 攻击在真实黑盒场景下的有效性结论
     - 对 threat model 中黑盒场景的论证支撑
```

---

## 六、关键风险与预案

| 风险 | 概率 | 预案 |
|------|------|------|
| Gemini API 情绪标签非结构化（自由文本） | 低（官方文档有 JSON 示例） | 改用 regex 解析情绪词；若仍不稳定改用 Gemini 2.5 Pro |
| Qwen3-ASR 标签映射与 ESD 不对齐 | 中 | Step 0 预实验验证；备选 Qwen3.5-Omni（free preview） |
| 整体迁移率过低（<20%）写不进论文 | 中 | 执行多 surrogate 集成（Step 3）；降低结论强度，改为"有限但非零的迁移性" |
| ESD-EN 英文子集样本数不足（<500） | 低 | 放宽至每类 100 条（共 400 条）；或混入部分 CN 样本（Qwen 支持中文） |
| API 调用限速 / 封号 | 低 | 加 rate limiting（每请求间隔 1s）；分批跑 |

---

## 七、待确认事项（需在执行前解决）

1. **ESD-EN 子集分布**：白盒 Voxtral 实验（EN 1000 条）中，4 类情绪各有多少条？可从 `code/white_box_voxtral/result/` 中统计。

2. **Gemini 情绪 Prompt 设计**：是否需要修改 Voxtral 的 3 个 emotion prompt 以适配 Gemini？（Voxtral prompt 要求"exactly one word from: happy, sad, angry, neutral, surprise"，Gemini 可能更适合结构化 JSON 请求。）

3. **Qwen3-ASR vs Qwen3.5-Omni 选择**：Qwen3-ASR 返回结构化标签但本质是 ASR 附加情绪，Qwen3.5-Omni 是真正的多模态推理模型（当前免费）——是否两者都测？从论文角度，Omni 模型更能代表"ALLM"的范畴，与本研究目标更吻合。

4. **Threat Model 修改范围**：目前 `3.threat_model.tex` 仅描述白盒设定。黑盒段落是作为 §3 的新 paragraph 还是在 §6.1 内自描述？
