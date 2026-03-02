## Section 2 Observation

### 2.1 Representational Decoupling and Causal Asymmetry within Audio

> 核心论点：ALLM 内部呈现**表征解耦与因果不对称**——早层同时编码丰富的韵律与语义表征，但因果路由在极早期即选择语义路径；韵律在表征层"可读"但在因果决策中不被"采纳"。

**2.1.1 表征层面的韵律-语义分离（Probe 分析）**

* 早层（0–14）**韵律表征优势**（注意：非因果主导），$D(\ell) = Acc_{prosody}(\ell) - Acc_{semantic}(\ell) > 0$，均值 ≈ 0.146，峰值 Layer 5（$D≈0.215$）。Probe 结构：线性分类器，GroupKFold 交叉验证，chance level = 0.20（5 类）。（已完善）
* 中层（14–23）融合态，$D≈0$，Layer 14–15 符号翻转（已完善）
* 晚层（23–35）语义占优，$Acc_{semantic}(27)≈0.830$（已完善）
* Layer 29–34 存在韵律准确率可能回弹的迹象，但尚未通过显著性检验，作为待验证现象置于附录（待补充：bootstrap 95% CI + 多重比较校正 + 非冲突对照）
* **方法论声明**：Probe 测量的是表征层面的信息富集程度（encodability），并不等同于因果驱动力。因果关系需由 Activation Patching 验证（见 2.1.3）。

**2.1.2 决策轨迹中的语义锁定（Logit Lens 分析）**

* **方法论声明**：Logit Lens 在早层的可靠性受限于隐状态与输出词表空间的对齐程度。因此本文重点关注中晚层（Layer 15+）的 margin 趋势，而非早层绝对值。参见 Belrose et al. (2023, Tuned Lens) 对该方法在较深层的可靠性验证。
* Layer 0–22：LM-head 投影在此区间产生低置信度、弥散分布，大量样本的 argmax 为非情绪 token。尽管 Probe 已在此区间检测到强韵律可读性，输出机制尚未做出承诺。（已完善）
* Layer 23 起 margin 显著转负，语义 win-rate 持续上升——早层韵律表征优势**未转化为决策偏好**（已完善）
* 词表偏差控制（待补充：5 词 unigram 基线 + synonym set 多 verbalizer 验证 + 校正后重绘 margin 曲线）

**2.1.3 因果干预确认语义主导（Activation Patching）**

* 语义 patch（Layer 0–12）：Flip-to-Target ≈ 0.65，Delta Logit 5–6，强定向因果控制（已完善）
* 韵律 patch（峰值 ~Layer 9）：Flip-to-Target ≈ 0.14–0.26，Delta Logit ≈ 2.2–2.3，弱扰动效应（已完善）
* **Patch 可比性说明**：语义 patch 与韵律 patch 的替换对象分别为语义内容不同但韵律相同的样本对、韵律不同但语义相同的样本对。两类 patch 在隐状态空间中引入的扰动幅度（L2 范数）具有可比性（$\|h_{sem}^{patch} - h_{sem}^{base}\|_2$ 与 $\|h_{pro}^{patch} - h_{pro}^{base}\|_2$ 分布重叠），排除了因扰动强度差异导致的系统偏差（待补充：完整的 KL 散度 / L2 范数归一化对比数据）
* Layer 14–15 为 audio span 可控性边界，此后局部替换难影响输出（已完善）
* 中层后局部干预失效的可能解释（假说）：情绪相关信息在中层已从局部 audio span 扩散至更广泛的位置表示，导致局部替换的边际效应趋零（待补充：position-level patching，对比 audio/text/random span 在 Layer 15+ 的控制力变化，以验证该假说）

**小结**：三种分析工具提供了互补的证据层次——Probe *suggests* 表征层面的信息分布模式；Logit Lens *reveals* 与决策对齐的隐状态转变趋势；Activation Patching *provides causal evidence* 确认语义路径的因果主导地位。**核心发现：表征解耦与因果不对称**——早层同时编码韵律和语义（Probe），但因果路由在极早期（Layer 0–12）即将语义设定为核心驱动力（Patching），韵律仅为伴随表征。关键边界：表征交叉 ~Layer 14–15，决策转折 ~Layer 23，两者约 8 层的间隔反映了表征变化与决策承诺之间的结构性滞后。（已完善）

**待补充验证汇总**

* Probe 稳健性：K-fold 交叉验证 + 随机标签对照（$D(\ell) \to 0$？）+ 非冲突负对照（Layer 14–15 翻转是否消失？）（待补充）
* 统计显著性：所有指标 bootstrap 95% CI + 多重比较校正（待补充）
* 三组对照对齐：Audio-only / Conflict / Consistent 的 Probe·Lens·Patching 并列呈现（待补充）
* Patch 可比性定量验证：L2 范数 / KL 散度归一化对比数据（待补充）
* Layer 14–15 vs Layer 23 边界落差的归因实验（待补充）

---

### 2.2 Text Instruction Dominance over Audio Prosody Signals

> 核心论点：当文本指令携带的情绪指向与音频韵律情绪冲突时（语义内容保持中性），ALLM 呈现**因果贡献的跨模态不对称**——文本指令在 Layer 5–20 建立显著更强的因果控制，音频韵律信号的因果贡献在此区间被边缘化。决策在 Layer 26–28 附近变得可读（consolidation），此后干预效力急剧下降。
>
> **Claim 边界声明**：本节实验固定了音频语义为中性内容，冲突仅存在于"文本指令情绪 vs 音频韵律情绪"之间。因此结论严格适用于"文本指令 > 音频韵律（在语义中性条件下）"。当音频语义本身携带强情绪时，文本指令的优势效力有待进一步验证（见待补充实验）。

**2.2.1 实验设计**（18 音频 × 3 条件，TTS 控制语义恒定；controlled pilot study）

* 三组对照：Audio-only（仅判断音频情绪）/ Conflict（文本指令指向与音频不同的情绪，$T \neq A$）/ Consistent（$T = A$，正对照）（已完善）
* 固定语义文本内容（中性句），排除音频语义的情绪性干扰，隔离"指令情绪指向 vs 音频韵律情绪"的纯冲突（已完善）
* 每类情绪样本数分布：neutral/happy/sad/angry/surprised 各 3–4 条基础音频（已完善）

**2.2.2 Finding 1：决策可读性在晚期层涌现（Late-Layer Decision Readability）**

* Logit Lens 差分（$\Delta logit_\ell = logit_\ell(T) - logit_\ell(A)$）显示三组在前 20 层差异接近 0，Layer 26–28 出现明显分化（已完善）
* 我们观察到 Layer 26–28 之后决策方向保持稳定（consolidation phase）。该区间的分化更可能反映隐状态最终对齐到词表空间后的"可读性涌现"，而非仲裁发生的真正时点——结合 2.2.3 的 Patching 数据，因果决策实际上在中层（Layer 5–20）已基本完成（见下文"决策完成与决策可读的区分"）。（已完善）
* 不可逆性待验证：Layer 26–28 后 PatchAudio 是否确认对输出无影响（待补充实验）

**2.2.3 Finding 2：因果贡献的跨模态不对称（Causal Asymmetry across Modalities）**

* PatchText 在 Layer 5–20 有效：替换文本指令的隐状态可显著改变最终情绪输出（已完善）
* PatchAudio 在相同层段无效/不稳定：替换音频隐状态对输出影响微弱（已完善）
* **因果不对称性**：当前证据支持"文本模态的因果贡献显著强于音频韵律"。该不对称性的具体机制（Attention 权重重分配？MLP 层的非线性消解？抑或表征竞争？）仍是开放问题，作为假说（hypothesis）标注，有待后续归因实验验证。（已完善）

**2.2.4 决策完成与决策可读的区分**

* **因果决策完成时点**（由 Patching 确定）：PatchText 在 Layer 5–20 有效，表明文本通过因果路径在中层即已建立对决策的控制。
* **决策可读性涌现时点**（由 Logit Lens 确定）：$\Delta logit_\ell$ 的分化直到 Layer 26–28 才出现，这反映的是隐状态对齐到词表空间的固有延迟，而非仲裁的真正发生地。
* 两个时点之间约 6–8 层的间隔，与 2.1 中表征交叉（Layer 14–15）到决策转折（Layer 23）的滞后模式一致，进一步支持"因果决策先于可读性涌现"的解释。

**2.2.5 与 2.1 的关系：两级模态优先级层级**

* 2.1 揭示音频内部语义-韵律冲突中语义通过因果路由胜出；2.2 进一步发现当外部文本指令参与竞争时，文本的因果贡献更强、作用区间更广（Layer 5–20 vs 2.1 中语义的决策转折 Layer 23）（已完善——定性层面）
* 两级主导层级（在当前实验条件下）：**音频内语义 > 音频内韵律**（2.1）；**文本指令 > 音频韵律**（2.2，语义中性条件下）（已完善）
* Layer 14–15（2.1 中 audio span 可控性边界）落在 2.2 文本主导区间（5–20）内——暗示一旦文本在中层建立控制，音频侧的局部扰动难以穿越此区间影响最终决策（待补充：在文本冲突条件下，对比 Layer <14 vs Layer 14–20 的 audio patch effect size，以将此推导转化为可复现的约束）

**2.2.6 过渡至 Section 3：从机理约束到攻击设计**

上述机理分析为定向情绪攻击的设计空间提供了两条硬约束和一个关键假说：

* **Constraint 1（可控窗口）**：音频侧的因果干预仅在 Layer 14–15 之前有效；超过此边界后，局部 audio span 的修改对最终情绪输出的影响可忽略不计。
* **Constraint 2（模态优先级）**：文本指令在 Layer 5–20 建立了主导性的因果控制，限制了仅靠音频扰动在标准推理条件下实现情绪操纵的可迁移性。

* **Vulnerability Window 假说**：尽管正常音频情绪信号在中层的因果贡献被文本边缘化，早层（0–14）对音频表征的高度依赖（Probe 所示的丰富编码）为对抗扰动提供了一个极窄但可利用的劫持窗口。关键洞察在于：对抗扰动无需沿正常情绪信号的通道传播——梯度优化可找到绕过中层模态不对称的非直觉路径，在可控窗口内制造语义级别的表征偏移，从而借用语义路径的因果优先权。

* **Design Implication**：因此，定向情绪攻击方法必须满足：(i) 扰动需在 Layer 14–15 可控性边界之前注入并生效；(ii) 扰动应模拟语义级特征以借用语义的因果优先权，而非单纯操纵韵律维度；(iii) 为对抗文本指令的中层因果主导，攻击需包含 prompt-robust 的设计策略。这些约束如何被具体转化为攻击方法，将在 Section~3 中予以回答。

**待补充验证汇总**

* 文本主导性的量化指标（Text-Dominance Index + Audio-Dominance 对偶指标）：在 Layer 5–20 对 text/audio token 做 mean-ablation 测量输出变化，定义为标量指标并报告置信区间（未做）
* 不同指令复杂度对中层主导性的影响：简单指令 vs 多步指令（如 CoT 格式）的 PatchText 因果强度对比（未做）
* 因果不对称性的机制归因（hypothesis testing）：Attention 信息流分析 / MLP gate activation 对比 / Knockout 实验（未做）
* 不可逆性验证：在 Layer 26–28 之后做 PatchAudio，确认对输出无影响（待补充）
* 桥句闭环实验：在文本冲突条件下对比 Layer <14 vs Layer 14–20 的 audio patch effect size（待补充）
* 完整对照实验：语义+韵律一致的音频 vs 冲突文本指令，验证文本能否压制完整的音频情绪信号（未做）
* 不同 TTS voice / speaker 的小规模复现，排除 TTS 系统偏差（未做）
* 跨模型验证：Prompt-Audio 冲突实验在 Kimi-Audio 等其他 ALLM 上的最小复现（未做；在复现完成前，正文使用 "In OpenS2S, we observe …" 的限定表述）
