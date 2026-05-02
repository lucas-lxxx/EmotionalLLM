# 研究进度同步文档

> **最后更新**：2026-04-18
> **研究人员**：徐振宇、李享

---

## A. 研究概述

本研究揭示 Audio Large Language Models（ALLM）在情绪感知维度上的对抗脆弱性。核心问题：能否通过对输入语音施加人耳不可察觉的微小扰动（L∞ ≤ 0.008），使 ALLM 将说话人情绪**定向误判**为攻击者指定的目标情绪，同时保持语义内容不变？

研究发现，ALLM 的情绪预测并非 one-hot 分类，而是由内部 soft competition field 主导——各情绪 token 之间存在结构性竞争关系，且这一机制跨模型普遍存在（Qwen2-Audio / Voxtral / SALMONN 均如此），但各模型的 bias center 不同（Qwen→angry, Voxtral→neutral, SALMONN→happy）。基于此发现，我们提出 **TEMPO**（Topology-Guided Emotion Manipulation by Pairwise Optimization），将情绪翻转建模为模型内部情绪空间中的拓扑引导状态转移，而非简单的 logit 操纵。TEMPO 在白盒设定下对三个架构迥异的 ALLM 均取得高 targeted ASR（3-prompt majority vote, 4-dataset averaged）：**Voxtral-Mini-3B 95.24%、OpenS2S 89.20%、MERaLiON-2-3B 100.00%**，扰动预算仅为传统 SER 攻击方法的 1/3.75，而后者在 ALLM 上的 untargeted 预测变化率不超过 10.5%。跨模型迁移矩阵（3×3 全实测）揭示 audio encoder 家族是对抗扰动可迁移性的主导因素——MERaLiON 同时是最强源攻击（out avg 53.52%）与最抗迁移目标（in avg 19.58%），Whisper 家族内部互迁移显著高于跨到 Qwen2-Audio 系。初步黑盒迁移实验表明，白盒对抗样本对闭源商业 API（Gemini / Qwen）具有非零但有限的迁移能力（12.5%–27.1%），远高于 SER→ALLM 的迁移率。

论文采用 observation-driven 结构：四个递进的 Observation（§2）揭示 ALLM 情绪机制的结构性弱点 → Threat Model（§3）形式化攻击目标与约束 → TEMPO 方法论（§4）由 Observation 推导的设计原则驱动 → 白盒实验（§5）验证攻击有效性。目标会议：安全/AI 顶会（ACM sigconf 格式）。

**涉及模型**：

| 角色 | 模型 |
| ---- | ---- |
| Observation 分析 | Qwen2-Audio-7B, Voxtral-Mini-3B, SALMONN-7B |
| 白盒攻击目标 | Voxtral-Mini-3B, OpenS2S（Qwen-based）, MERaLiON-2-3B（Whisper-family + paralinguistic decoder） |
| 跨模型迁移评估 | 3×3 Voxtral/OpenS2S/MERaLiON 全矩阵 + SALMONN/Kimi-Audio 占位 |
| 黑盒攻击目标 | Gemini 2.5 Flash, Qwen3-Omni-Flash |

**各阶段状态**：

| 阶段 | 状态 |
| ---- | ---- |
| 机理分析（Observation） | ✅ Q1-Q4 全部完成，论文初稿已写入 |
| 白盒攻击 | ✅ Voxtral + OpenS2S + MERaLiON 全部完成；TEMPO 方法论 + 3×3 跨模型迁移矩阵已写入 §5 |
| 跨模型迁移评估 | ✅ 6 个 off-diagonal cell 全部实测完成（每 cell 60×4 样本，majority vote） |
| 黑盒攻击 | 🔧 Demo 完成（48 条，Gemini 12.5% / Qwen 27.1%），完整实验与论文未开始 |
| 论文写作 | §2-§5 初稿完成（§5 包含 3-model 主表 + 跨模型迁移矩阵）；待写：§1 Introduction / §6 黑盒 / §7 Related Work / §8 Discussion / Abstract |

## B. 机理分析（Observation，§2）

围绕四个递进问题展开：翻转可行性 → 翻转后果 → 传统方法局限 → 跨模型普遍性，逐步为 TEMPO 方法论提供动机与设计约束。

### Q1: ALLM 的情绪预测是 structured competition 而非 one-hot  ✅ 已写入 §2.1

> Qwen2-Audio-7B-Instruct, audio-only, 400 样本（4 情绪 × 100）

分析 clean 音频的 emotion token logit/probability distribution，揭示情绪预测的内部竞争结构。

**关键发现**：
- 模型情绪输出是 structured preference distribution，各情绪 token 间存在稳定的竞争关系
- 稳定性高度不对称：angry/sad 较稳定（候选集准确率 61%/63%），happy 极脆弱（准确率仅 15%）
- angry 是万能 competitor：happy/neutral/sad 的主要竞争者均为 angry
- Prototype 相似性：angry↔happy JS 散度最小（0.0195）→ 翻转阻力最低
- First-step logit 证据：模型不确定性集中在情绪 token 子空间内，而非全词表

**论文**：§2.1 "Clean emotion recognition is governed by structured non-target preference"（Fig 1 candidate probability + Fig 2 first-step logit histograms）

### Q2: 情绪误感知系统性污染下游回复  ✅ 已写入 §2.2

> Voxtral-Mini-3B, Aligned vs Conflict prompt, 20 样本（5 情绪 × 4）, DeepSeek V3 LLM Judge

通过文本 Prompt 操纵情绪感知（不依赖攻击方法），验证情绪误感知对下游回复的影响。

**关键发现**：
- 三维度系统性退化：Faithfulness Δ+1.30, Empathy Δ+1.20, Relevance Δ+1.05
- happy 最脆弱（Empathy Δ+3.00）/ sad 完全鲁棒（Δ=0）/ neutral 是幻觉温床（Faithfulness 低至 1.50/5）
- 强负面情绪音频信号可抵抗 Prompt 误导，但 happy/neutral/surprised 无法抵抗
- neutral 语音下模型会凭空编造情绪叙事 → 现实部署中最常见的语音类型恰恰是最脆弱的攻击面

**论文**：§2.2 "Downstream Impact of Emotion Misperception"（Table 1 三维度退化）

### Q3: 传统 SER 攻击方法对 ALLM 无效  ✅ 已写入 §2.3

> Voxtral-Mini-3B, 四种 SER 攻击方法（6 个变体）迁移评估, ESD 数据集

将 SER 领域四种代表性对抗攻击（STAA-Net / PGD / Gao et al. / Ren et al.）迁移到 ALLM。所有方法均为 untargeted（比 targeted 更容易），给予白盒 access + 3.75× 扰动预算（ε=0.03 vs 我们的 0.008）。

**结果**：全部失败，预测变化率 2.5%–10.5%。PGD 在 SER surrogate 上 100% ASR，迁移到 Voxtral 仅 10.5%。失败原因是架构不兼容：SER 攻击针对独立分类器的紧凑决策边界，扰动信号在 ALLM 的 encoder-adapter-LLM pipeline 中耗散。

**论文**：§2.3 "Insufficiency of Traditional SER Attack Methods"（Fig 3 六种方法迁移失败对比图）

### Q4: 跨模型普遍竞争结构与模型特异性拓扑  ✅ 已写入 §2.4

> Qwen2-Audio-7B、Voxtral-Mini-3B、SALMONN-7B，共享 400 样本 manifest，clean probing protocol

将 Q1 的单模型发现扩展到三个架构不同的 ALLM，同时测量 candidate-level emotion competition 和 layerwise linear separability。

**关键发现**：
- **跨模型不变量 1**：三个模型均展现 soft competition field，情绪预测由内部竞争主导而非 one-hot
- **跨模型不变量 2**：情绪可分性均在中间层达到峰值（Qwen layer 6 / Voxtral layer 12 / SALMONN layer 6），而非最终层
- **模型特异性**：bias center 完全不同——Qwen 偏向 angry（happy 给 angry 0.46 > 自身 0.14），Voxtral 回落到 neutral（angry 给 neutral 0.53），SALMONN 吸引向 happy（angry 给 happy 0.32）
- **对攻击设计的三条设计原则**：competitor-aware（因为 bias center 不同）、pair-specific（因为最易/最难情绪对因模型而异）、layer-adaptive（因为最佳控制层在中间而非输出）→ 直接驱动 TEMPO 方法论

**论文**：§2.4 "Universal Competitive Structure but Model-Specific Emotion Topology"（Fig 4 cross-model competition heatmaps + Fig 5 mid-layer separability peaks）

## C. 白盒攻击（TEMPO 框架，§3-§5）

### Threat Model（§3）

白盒设定：攻击者完全访问 ALLM 的架构、权重与中间表征，可计算端到端梯度。仅控制输入波形，不修改模型参数或 prompt。

约束三重：(i) targeted emotion flip：f(x', p_emo) = e_t；(ii) semantic preservation：对抗音频转写与 clean 转写高度相似；(iii) imperceptibility：‖δ‖∞ ≤ 0.008 + 多分辨率 STFT 幅度损失。

### TEMPO 方法论（§4）

方法论从 v1（简单三项损失加权）升级为 **TEMPO**（Topology-Guided Emotion Manipulation by Pairwise Optimization）。核心思想：情绪翻转不是 logit 操纵，而是模型内部情绪空间中的**拓扑引导状态转移**。

**四个核心组件**：
1. **Topology Bank**：为每个模型构建情绪拓扑先验（bias center b_m、dominant competitor k_m(e)、boundary/steering layers、prototypes μ_{m,l,e}、pair hardness H_m(s,t)）
2. **Pair-Specific Route Planning**：根据 pair hardness 选择直接翻转（easy/bias-aligned pairs）或经由中间情绪 waypoint 的间接路由（hard non-bias pairs），避免坍缩到 dominant competitor
3. **Layer-Adaptive Steering**：三层目标分工——
   - L_exit（中间层）：将表征推出源情绪区域
   - L_steer（深层）：沿 source→target 方向引导表征移动
   - L_comp（输出层）：竞争巩固，确保 target 同时压制 source 和 dominant competitor
4. **Margin-Aware Optimization**：根据 pair hardness 和 clean source margin 动态分配扰动预算 ε(x,s,t) 与优化步数

**三阶段优化**：Phase I 边界逃逸（L_exit 主导）→ Phase II 路由对齐引导（L_steer + L_exit）→ Phase III 竞争巩固（L_comp + L_steer）

### 白盒实验结果（§5）

主表（IEMOCAP / RAVDESS / ESD-EN / ESD-CN 的 dataset-averaged 指标，基于 60 样本/数据集的 matched subset）：

| 指标 | Voxtral-Mini-3B | OpenS2S | MERaLiON-2-3B |
| ---- | ---- | ---- | ---- |
| Targeted ASR（3-prompt majority vote, 4-dataset avg） | 95.24% | 89.20% | **100.00%** |
| 单 Prompt 最高 ASR | 99.90% | 94.66% | 100.00%（所有 prompt） |
| 语义保留率 Sem. | 39.76% | 21.23% | **50.28%** |
| 联合成功率（情绪✓ ∧ 语义✓）Joint | 36.46% | 18.35% | **50.28%** |
| 平均 SNR | **20.14 dB** | 17.04 dB | 13.95 dB |
| 收敛率 Conv.（emo_loss<1.0 @ 60 steps） | 99.91% | 50.00% | **100.00%** |

**Per-emotion 分析**：Voxtral 上四种情绪 ASR 均 >92%（sad 最高 96%）；OpenS2S 上 angry 最难翻转（71.21%），与 Q1 发现的 angry 作为 dominant competitor 的结构性锚定一致；MERaLiON 将 per-emotion 分布完全拉平（4 种 source emotion 均 100%），反映其 paralinguistic-supervised decoder 的局部尖锐、全局脆弱特性。

**瓶颈**：语义保持率仍是联合成功率的限制因素。Voxtral 与 MERaLiON 收敛极快（少数步内完成），OpenS2S 收敛慢（47.85% 收敛，平均 39.5 步），长时间优化导致扰动漂移到内容承载的声学维度。MERaLiON 的高语义保留（50.28%）正是其 near-instant convergence 的副产品——优化器没机会 drift 到 content-bearing dimensions。

**与 Q3 的对比**：在 3.75× 更小的扰动预算下，ALLM-native 攻击的 targeted ASR（95.24% / 89.20% / 100.00%）远超传统 SER 攻击的 untargeted 预测变化率（≤10.5%），证实 ALLM-native 攻击设计的必要性与有效性。

### 跨模型迁移矩阵（§5.3）

完整 3×3 矩阵（行=source attacker，列=target evaluator，每 cell = 60×4 样本的 dataset-averaged targeted ASR %）：

| Source ↓ \ Target → | Voxtral | OpenS2S | MERaLiON | **Row avg** |
| ---- | ---- | ---- | ---- | ---- |
| Voxtral | *95.24* (WB) | 34.17 | 29.58 | 31.88 |
| OpenS2S | 35.42 | *89.20* (WB) | 9.58 | 22.50 |
| MERaLiON | **67.79** | **39.25** | *100.00* (WB) | **53.52** |
| **Col avg (excl. diag)** | 51.60 | 36.71 | **19.58** | — |

**三条核心观察**（已写入 §5.3）：

1. **MERaLiON 的双重角色**：同时是最强源攻击（out avg 53.52%）与最抗迁移目标（in avg 19.58%），反映其 paralinguistic-supervised decoder 带来的「局部尖锐、全局脆弱」几何。
2. **Encoder 家族主导迁移**：MERaLiON→Voxtral（67.79%）是最强单 cell，两者共享 Whisper 系编码器；跨到 OpenS2S（Qwen2-Audio 系）时降到 39.25%，一致的对称模式在反向方向（OS→V 35.42% vs OS→M 9.58%）同样成立。
3. **OpenS2S→MERaLiON = 9.58%**：整个矩阵最弱的 cell，意味着 Qwen2-Audio 系代理模型难以攻击 Whisper 家族 + 情绪训练的目标。对黑盒攻击的启示：代理模型的 audio front-end 应匹配目标模型。

## D. 黑盒攻击

验证白盒对抗样本能否在闭源商业 API 上保持攻击效果。代码与结果位于 `blackbox/`。

### Demo 实验结果（48 条样本，Voxtral surrogate，单 speaker 0011）

| 目标模型 | Transfer ASR（majority vote） | 迁移损失 |
| -------- | ----------------------------- | -------- |
| Gemini 2.5 Flash | 12.50%（6/48） | 81.6 pp |
| Qwen3-Omni-Flash | 27.08%（13/48） | 67.0 pp |

**关键发现**：
- ALLM→ALLM 的 targeted 迁移率（12.5%–27.1%）远高于 Q3 中 SER→ALLM 的 untargeted 迁移率（≤9.3%），证实 ALLM 之间共享对情绪对抗扰动的脆弱性
- 5 条样本在 Gemini 和 Qwen 两个完全不同架构的闭源 API 上同时成功 → 存在跨架构共享的对抗脆弱子空间
- angry→happy 在 Qwen 上迁移率最高（33.33%），与 Q1 发现的 angry-happy JS 散度最小一致，跨模型验证了结构性偏向
- 约 65-71% 的对抗样本被判为 neutral → neutral 是 ALLM 的主要吸引子（与 Q4 发现的 Voxtral neutral fallback 一致）
- Qwen 迁移率（27.1%）显著高于 Gemini（12.5%），可能与 Gemini 在 AHELM 情绪检测 benchmark 上更强（更鲁棒）有关

### 完整实验计划（见 `blackbox/plan.md`）

- **实验矩阵**：4 surrogate（Voxtral EN/CN + OpenS2S EN/CN） × 6 target（Gemini Flash/Pro, GPT-4o Audio, Qwen-Omni-Turbo/Qwen2.5-Omni-7B, ERNIE）
- **评估协议**：3-prompt majority vote，与白盒一致
- **Baseline**：Clean + Random Noise（已生成 1718 条噪声 WAV）
- **消融实验**：多 surrogate 集成（Voxtral + OpenS2S 联合 loss）、通用对抗扰动（UAP）
- **论文节**：§6 Black-box Transferability

**当前状态**（2026-04-06）：
- ✅ 完整实验基础设施已搭建（6 个 API 客户端、评估流水线、分析脚本、图表生成、一键编排器）
- ✅ 样本准备完成：manifest.csv (3594 条)、noise baseline WAV (1718 条)
- ✅ 论文 §6 初稿已写入 `finalpaper/6.blackbox.tex` 并编译通过
- ⏳ 等待 API Key 配置后运行实验（OpenS2S EN/CN 数据就绪，Voxtral 需从服务器下载 WAV）
- 运行方式：`cd blackbox && set GEMINI_API_KEY=xxx && python run_all.py`

## E. 论文写作进度

论文 LaTeX 源码位于 `finalpaper/`。ACM sigconf 格式。

```
1. Introduction                                          ⬜ 未开始
2. Observation（四个递进问题，为攻击方法提供动机与约束）
   2.1 Q1: structured non-target preference              ✅ 已完成（Fig 1 + Fig 2）
   2.2 Q2: 情绪误感知的下游传导与幻觉                  ✅ 已完成（Table 1）
   2.3 Q3: 传统 SER 攻击方法的不充分性                  ✅ 已完成（Fig 3 SER transfer）
   2.4 Q4: 跨模型普遍性分析                             ✅ 已完成（Fig 4 competition heatmaps + Fig 5 mid-layer peaks）
3. Threat Model                                          ✅ 已完成（formal problem statement）
4. 方法论 TEMPO                                          ✅ 已完成（4 subsections + Fig paradigm/route/layer）
5. 白盒实验                                              ✅ 已完成（Setup / Targeted Flipping / Semantic Preservation / Summary）
6. 黑盒实验                                              🔧 初稿已写（基于 Demo 数据，待完整实验数据更新）
7. Related Work                                          ⬜ 未开始
8. Discussion and Limitations                            ⬜ 未开始
Abstract / Title                                         ⬜ 未开始
附录                                                     ⬜ 未开始
```

## F. 文件索引

> **2026-04-18 目录重构**：顶层已收敛为 9 个目录，详细检索见 `README.md`。此处仅列研究高频用到的数据/代码入口。

| 路径 | 内容 |
| ---- | ---- |
| **论文与汇报** | |
| `paper/` | **论文 LaTeX 源码**：`main.tex` + `2.observation.tex` + `3.threat_model.tex` + `4.methodology.tex` + `5.whitebox.tex` + `6.blackbox.tex`（原 `finalpaper/`） |
| `paper/figure/` | 论文配图：Q1 概率分布图、logit 直方图、Q3 SER 迁移图、Q4 competition heatmaps + mid-layer peaks、TEMPO 框架图 |
| `reports/` | 汇报材料（PPT、讲稿、实验素材，原 `LATEST/`） |
| `refs/` | 参考文献 PDF（~35 篇，原 `paper/`） |
| `docs/` | 框架图、相关工作调研、会议记录 |
| **Observation（§2）** | |
| `results/observation_v3/Q1&Q4/` | Q1/Q4 实验交付物：logit 分析、prototype、confusion matrix（原 `observation_v3/Q1&Q4/`） |
| `results/observation_v3/Q2/` | Q2 实验：Aligned vs Conflict 推理对比 |
| `results/observation_v3/Q3/` | Q3 实验：四种 SER 方法迁移评估（STAA/PGD/GAO/REN 子目录） |
| `code/observation/modal_conflict/` | Probe 实验：音频内模态冲突机理 |
| `code/observation/logit_lens/` | Logit Lens 实验：逐层决策追踪 |
| `code/observation/activation_patching/` | Activation Patching 实验：因果干预 |
| **白盒攻击（§5）** | |
| `code/white_box_voxtral/` | Voxtral 白盒攻击代码（含 cross_eval.py） |
| `code/white_box_voxtral/result/Voxtral_{EN,CN,IEMOCAP,RAVDESS}/` | Voxtral 对抗样本 WAV + 每样本 JSON（EN/CN 各 1000 条，IEMOCAP/RAVDESS 各 60 条） |
| `code/white_box_voxtral/result/cross_eval/` | Voxtral 作为 target 评估 OpenS2S/MERaLiON 对抗样本的 summary_{TAG}.json（仅服务器） |
| `code/white_box_opens2s_v2/ver2.0/` | OpenS2S 白盒攻击代码（含 cross_eval.py） |
| `code/white_box_opens2s_v2/result/{IEMOCAP,RAVDESS}/` | OpenS2S 对抗样本（IEMOCAP/RAVDESS 各 60 条） |
| `code/white_box_opens2s_v2/result/blackbox/{EN,CN}/` | OpenS2S 对抗样本（ESD EN/CN 各 1000 条，命名历史遗留，下次重命名为 `ESD_{EN,CN}_full`） |
| `code/white_box_opens2s_v2/result/cross_eval/` | OpenS2S 作为 target 评估 Voxtral/MERaLiON 对抗样本的结果（仅服务器） |
| `code/white_box_meralion/` | MERaLiON 白盒攻击代码（config.py / meralion_io.py / attack_core.py / run_attack.py / cross_eval.py） |
| `code/white_box_meralion/result/MERaLiON_{EN,CN,IEMOCAP,RAVDESS}/` | MERaLiON 对抗样本（每数据集 60 条，CN 因一条 loader skip 为 59 条；仅服务器） |
| `code/white_box_meralion/result/cross_eval/` | MERaLiON 作为 target 的 cross-eval 结果 + all_summaries.json 汇总（仅服务器） |
| **黑盒攻击（§6）** | |
| `blackbox/plan.md` | 黑盒实验详细计划（2×2 矩阵、执行步骤、论文节结构草案） |
| `blackbox/report.md` | 黑盒 Demo 实验报告（48 条样本） |
| `blackbox/results/{gemini,qwen}/` | API 评估结果 |
| **数据与归档** | |
| `data/ESD/` | ESD 数据集（原 `dataset/ESD/`） |
| `archive/` | 早期探索、废弃代码（原 `PREVIOUS/`），含 `white_box_opens2s_v1/`、`observation_early_docs/`、`pipid_paper_template/` 等 |

## G. 更新日志

| 日期 | 更新内容 |
| ---- | ---- |
| 2026-02-06 | 创建本文件 |
| 2026-02-08 | 更新论文大纲，Observation 前置 |
| 2026-02-14 | 精简第二部分，聚焦 Observation |
| 2026-02-15 | Observation 从三节合并为两节；更新文件索引与执行计划 |
| 2026-02-21 | 新增实验代码文件索引 |
| 2026-02-24 | Observation P0+P1 修订完成 |
| 2026-03-01 | Observation 初稿完成；OpenS2S 批量实验完成 |
| 2026-03-11 | 重构第二部分（A-D 节全面更新）；新增 Voxtral 至目标模型 |
| 2026-03-18 | Observation 重构至 v3（Q1-Q3）；Q2 实验完成 |
| 2026-03-19 | B 节重写为 Q1-Q4 四问结构；C 节补充 Voxtral 实验结果 |
| 2026-03-28 | Q3 实验完成；Q1/Q2 写入论文；新增 `finalpaper/` 目录 |
| 2026-04-05 | 文档重构为进度同步文档（删除 AI 协作规则部分）；论文 §2-§5 初稿全部完成；方法论升级为 TEMPO 框架；Q4 跨模型普遍性分析完成并写入论文；黑盒 Demo 完成（Gemini 12.5% / Qwen 27.1%）；新增 blackbox 目录 |
| 2026-04-18 | 新增第三个白盒目标 MERaLiON-2-3B（4 数据集 × 60 样本，ASR 100%，Sem 50.28%，SNR 13.95 dB）；完成 3×3 跨模型迁移矩阵（6 个 off-diagonal cell 全部实测）；论文 §5 整体更新为 3-model 版本 + 新增 §5.3 Cross-Model Transferability；白盒攻击代码目录 `code/white_box_meralion/`；主表格新增 SALMONN-7B/Kimi-Audio-7B 占位行待后续实验 |
| 2026-04-18 | **目录重构**：顶层从 18 项收敛到 9 项。`PREVIOUS/`→`archive/`、`LATEST/`→`reports/`、`paper/`→`refs/`、`finalpaper/`→`paper/`、`dataset/`→`data/`、`observation_v3/`→`results/observation_v3/`、`meeting/`→`docs/meeting/`、`框架.png`→`docs/framework.png`；归档 `observation/`+`observation_v2/`+`white_box_opens2s_v1/`+`finalpaper2/` 到 `archive/`；清理 `code/` 根下两个冗余 cross_eval.py；新增 `README.md` 作项目入口与文件检索；服务器端同步命令见 `sync_to_server.sh` |
