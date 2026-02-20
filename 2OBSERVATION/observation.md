## Section 2 Observation

### 2.1 How ALLMs Resolve Prosody–Semantic Conflicts

> 核心论点：ALLM 内部呈现「早层韵律表征主导 → 中层融合竞争 → 晚层语义决策主导」的层级结构；韵律在表征层"可读"但在决策层不被"采纳"。

**2.1.1 Probe 层级结构**

* 早层（0–14）韵律主导，$D(\ell)>0$，均值 ≈ 0.146，峰值 Layer 5（$D≈0.215$）（已完善
* 中层（14–23）融合态，$D≈0$，Layer 14–15 符号翻转（已完善）
* 晚层（23–35）语义占优，$Acc_{semantic}(27)≈0.830$（已完善）
* Layer 29–34 韵律回弹现象，需判定真机制 vs 统计伪影（待补充：bootstrap CI 显著性检验 + 非冲突对照）

**2.1.2 Logit Lens 决策轨迹**

* Layer 0–22 margin ≈ 0，大量 "other" 输出，决策未成形（已完善）
* Layer 23 起 margin 显著转负，语义 win-rate 持续上升——早层韵律表征优势不转化为决策偏好（已完善）
* 词表偏差可能影响 Layer 23 转折的可靠性（待补充：5 词 unigram 基线 + 校正后重绘）

**2.1.3 Activation Patching 因果确认**

* 语义 patch（Layer 0–12）：Flip-to-Target ≈ 0.65，Delta Logit 5–6，强定向控制（已完善）
* 韵律 patch（峰值 ~Layer 9）：Flip-to-Target ≈ 0.14–0.26，Delta Logit ≈ 2.2–2.3，弱扰动效应（已完善）
* Layer 14–15 为 audio span 可控性边界，此后局部替换难影响输出（已完善）
* "信息扩散至全局位置"的解释缺乏直接证据（待补充：position-level patching，对比 audio/text/random span 在 Layer 15+ 的控制力变化）

**小结**：Probe（what is encoded）→ Logit Lens（what drives output）→ Patching（what causally determines output），三工具共同确认「语义优先仲裁」策略。关键边界：表征转折 ~Layer 14–15，决策转折 ~Layer 23。（已完善）

**待补充验证汇总**

* Probe 稳健性：K-fold 交叉验证 + 随机标签对照 + 非冲突负对照（待补充）
* 统计显著性：所有指标 bootstrap 95% CI + 多重比较校正（待补充）
* 三组对照对齐：Audio-only / Conflict / Consistent 的 Probe·Lens·Patching 并列呈现（待补充）
* Layer 14–15 vs Layer 23 边界落差的解释：是"表征先变、决策滞后"还是 Logit Lens 中间层不可靠？（待补充）

---

### 2.2 How Text Instructions Override Audio Emotion Signals

> 核心论点：当文本指令携带的情绪指向与音频情绪冲突时，ALLM 呈现「中层文本因果主导 + 晚层决策固化」的仲裁模式——文本指令在 Layer 5–20 建立强因果控制，音频情绪信号在此区间被结构性压制，最终决策在 Layer 26–28 不可逆地固化为文本指向。

**2.2.1 实验设计**（18 音频 × 3 条件，TTS 控制语义恒定）

* 三组对照：Audio-only（仅判断音频情绪）/ Conflict（文本指令指向与音频不同的情绪）/ Consistent（文本指令与音频一致）（已完善）
* 固定语义文本内容，排除语义内容干扰，隔离"指令情绪指向 vs 音频韵律情绪"的纯冲突（已完善）

**2.2.2 Finding 1：仲裁发生在晚期层（Late-Layer Consolidation）**

* Logit Lens 差分（$\Delta logit_\ell = logit_\ell(T) - logit_\ell(A)$）显示三组在前 20 层差异接近 0，Layer 26–28 出现明显分化与固化（已完善）
* 一旦决策在 Layer 26–28 固化，后续层无法再逆转——决策窗口是有限的（已完善）

**2.2.3 Finding 2：文本因果主导，音频中层被结构性压制**

* PatchText 在 Layer 5–20 有效：替换文本指令的隐状态可显著改变最终情绪输出（已完善）
* PatchAudio 在相同层段无效/不稳定：替换音频隐状态对输出影响微弱（已完善）
* 文本主导的因果不对称性：文本指令在中层"覆盖"音频信号，而非简单的加权竞争（已完善）

**2.2.4 与 2.1 的关系**

* 2.1 揭示音频内部语义-韵律冲突中语义胜出；2.2 进一步发现当外部文本指令参与竞争时，文本的因果主导性更强、作用区间更广（Layer 5–20 vs 2.1 中语义的决策转折 Layer 23）（已完善——定性层面）
* 两级主导层级：**音频内语义 > 音频内韵律**（2.1）；**文本指令 > 音频整体信号**（2.2）（已完善——定性层面）
* Layer 14–15（2.1 中 audio span 可控性边界）落在 2.2 文本主导区间（5–20）内——暗示一旦文本在中层建立控制，音频侧的任何扰动都难以穿越此区间（已完善——推导层面）

**2.2.5 过渡至 Section 3**：2.1–2.2 的机理发现表明，ALLM 的情绪决策存在清晰的层级结构和模态优先级。这些发现为理解音频对抗扰动的作用空间与限制提供了基础，自然引出攻击可行性分析与方法论设计。（未做）

**待补充验证汇总**

* 文本主导性的量化指标（Text-Dominance Index）：在 Layer 5–20 对 text token 做 mean-ablation 测量输出变化，定义为标量指标（未做）
* 不同指令复杂度对中层主导性的影响：简单指令 vs 多步指令（如 CoT 格式）的 PatchText 因果强度对比（未做）
* 音频信号在中层被"压制"的具体机制：是 attention 权重重分配？还是 MLP 层的非线性消解？（未做）
* 仲裁固化的不可逆性验证：在 Layer 26–28 之后做 PatchAudio，确认对输出无影响（待补充）
* 跨模型验证：Prompt-Audio 冲突实验在 Kimi-Audio 等其他 ALLM 上的复现（未做）
