---
# PPT Content (Markdown) — 情绪LLM白盒攻击研究
> 说明：本文件仅包含“可直接复制进PPT/交给生成器”的具体文案与图片占位，不包含排版细节。图片请用占位标注位置替换为你本地对应路径。

---

## Slide 0 — 封面
- 标题：情绪LLM白盒攻击研究
- 副标题：白盒定向情绪攻击 + 模态冲突机理分析（草稿）
- 汇报人 / 单位 / 日期：按模板填写

---

## Slide 1 — 目录
1. 白盒攻击方法论（闭环验证）
2. Audio内部冲突机理（语义 vs 韵律）
3. Prompt-Audio冲突机理（指令 vs 音频）
4. 机理引导攻击优化（进行中）

---

# Part 1 白盒攻击方法论（闭环验证）

## Slide 2 — 研究问题与目标
**标题：问题定义与核心目标**

- 研究问题：是否可以通过对输入语音进行微小扰动，使多模态语音-语言大模型在理解该语音时，将说话人的情绪稳定判断为目标情绪（如 happy），同时保持语义内容基本不变？
- 攻击约束：不改模型参数、不改 prompt，只优化输入音频
- 输出形式：情绪判断任务下输出单词情绪标签（如 happy / sad / angry / neutral）

**图片占位（放在本页右侧或下方）：**
- [FIG: 方法论闭环示意图]
  - 原始音频 `x` → 加扰动 `δ` → 对抗音频 `x'` → 模型 → 输出目标情绪
  - 同时：`x` 与 `x'` 在 ASR 任务上尽量一致

---

## Slide 3 — 模型抽象（条件生成）
**标题：模型抽象与基本表征**

- 将多模态模型抽象为条件生成模型：  
  \[
  p_\theta(y \mid x, p)
  \]
- 变量含义：
  - \(x\)：输入语音信号
  - \(p\)：prompt（离散且固定）
  - \(y\)：生成 token 序列
  - \(\theta\)：模型参数（固定，不参与优化）
- 内部两步：
  1. 音频编码（共享）：  
     \[
     h = f_{\text{audio}}(x)
     \]
  2. 条件生成（依赖 prompt）：  
     \[
     p_\theta(y \mid x, p) = p_\theta(y \mid h, p)
     \]
- 重要事实（结构前提）：对同一段音频 \(x\)，不同 prompt 下使用同一个音频表示 \(h\)

---

## Slide 4 — 情绪的内部表征
**标题：情绪不是显式变量，而是条件分布的结构性偏好**

- 模型内部不存在显式“情绪变量”
- 情绪体现在：在特定任务条件下，对情绪相关 token 的概率分布偏好
- 当 prompt 要求模型输出情绪标签时，在某个生成位置形成分布：  
  \[
  p_\theta(y_1 = v \mid h, p_{\text{emo}}), 
  \quad v \in \{\text{happy}, \text{sad}, \text{angry}, \text{neutral}\}
  \]
- “模型认为说话人是 happy”  
  ⟺ 在该条件下 happy token 的概率（或 logit）显著更高/为最大

---

## Slide 5 — 攻击目标：能量最小化
**标题：攻击核心思想：将情绪判断转化为优化问题**

- 定义目标情绪损失（以 happy 为例）：  
  \[
  \mathcal{L}_{\text{emo}}(x')
  =
  -\log p_\theta(y_1=\text{happy} \mid h(x'), p_{\text{emo}})
  \]
- 直观含义：
  - 若模型不倾向输出 happy → 损失大
  - 若模型高度确信 happy → 损失趋近 0
- “Happy 区域”概念（用于解释，不是显式变量）：  
  \[
  \mathcal{H}_{\text{happy}}
  =
  \left\{
  h \mid
  p_\theta(\text{happy} \mid h, p_{\text{emo}})
  >
  p_\theta(v \mid h, p_{\text{emo}}),
  \;\forall v\neq \text{happy}
  \right\}
  \]
- 攻击目标：通过优化扰动，使 \(h(x')\) 推入 \(\mathcal{H}_{\text{happy}}\)

---

## Slide 6 — 语义一致性约束
**标题：防止退化解：语义自一致约束**

- 问题：仅优化情绪损失可能出现退化（通过破坏语义“投机”改变情绪输出）
- 先定义基准转写：  
  \[
  y^{\text{asr}}(x)
  =
  \arg\max_y p_\theta(y \mid h(x), p_{\text{asr}})
  \]
- 引入语义一致性约束：  
  \[
  \mathcal{L}_{\text{asr}}(x')
  =
  -\log p_\theta(y^{\text{asr}}(x) \mid h(x'), p_{\text{asr}})
  \]
- 作用：要求在转写任务上，对抗音频与原始音频具有同一高概率解释，从而强制模型继续“认真听音频”，避免破坏语义

---

## Slide 7 — 初步结果（方法论闭环）
**标题：实验结果（初步）**

- 攻击闭环已跑通：输入音频 → 优化扰动 → 输出指定情绪
- 在 prompt 为“直接输出音频情绪标签”的设置下：
  - 情绪攻击成功率约 **80%**（10条样本）
- 评估限制（当前阶段）：
  - WER 等参数设置过于严格，后续批量实验准备接入商业 API 判断
- 现象：攻击效果对 prompt 敏感
  - 直接分类提示：约 40% 识别为 happy
  - 描述+分类提示：约 50% 识别为 happy
  - 逐步分析提示：约 20% 识别为 happy
- 结论（方法论层面）：定向情绪攻击可行，但跨 prompt 鲁棒性需要机理分析支撑

**图片占位：**
- [FIG: 结果示例（emo_pred_clean vs emo_pred_adv）]
- [FIG: 成功率统计卡片/小表格（num_samples=10, emo_success_rate=0.8）]

---

# Part 2 Audio内部冲突机理（语义 vs 韵律）

## Slide 8 — 动机与核心问题
**标题：Audio内部冲突：语义情绪 vs 韵律情绪**

- 音频内部两种情绪线索：
  - 语义情绪（semantic）：文本内容本身表达的情绪
  - 韵律情绪（prosody）：语调/节奏/强度等副语言线索
- 核心问题：
  - 在模型不同层里，哪类信息更强、更可读？
  - 尤其在语义与韵律冲突时，表征更偏向哪一边？

---

## Slide 9 — 数据与设置（Probe）
**标题：实验设计：逐层表示 + Probe 可读性**

- 情绪类别（5类）：neutral / happy / sad / angry / surprised
- 数据：
  - 50条文本（每类10条）
  - 每条文本生成5种韵律版本 → 理论250条，实际可用247条
  - 冲突样本 197；一致样本 50
- Prompt 固定为中性判断（不引导听谁）：
  - “What is the emotion of this audio? Answer with exactly one word: neutral, happy, sad, angry, surprised.”
- 表示提取方式：
  - 一次 forward 获取 0–35 层 hidden states
  - 取 audio span 做 pooling 得到向量
  - 每层训练两个 probe：
    - 预测 `text_emotion`（语义）
    - 预测 `prosody_emotion`（韵律）
- 主导性指标：
  \[
  D(\ell) = Acc_{\text{prosody}}(\ell) - Acc_{\text{semantic}}(\ell)
  \]
  - \(D>0\)：韵律更“可读”
  - \(D<0\)：语义更“可读”

---

## Slide 10 — Probe结果：层级结构
**标题：Probe结果：韵律早强 → 中层融合 → 语义窗口 → 晚层回潮**

关键数值（按你记录的结果）：
- overall dominance：prosody，平均 \(D \approx 0.0526\)
- 韵律最强：layer 0 的 prosody_acc \(\approx 0.842\)
- 语义最强：layer 27 的 semantic_acc \(\approx 0.830\)
- 韵律主导峰值：layer 5，\(D \approx 0.2146\)（冲突子集 \(D_{\text{conf}} \approx 0.2182\)）
- 语义占优最明显：layer 26，\(D \approx -0.0414\)

层级结构总结：
- 早层（0–14）：显著韵律主导（连续 \(D>0\)，早层平均 \(D\approx 0.146\)）
- 中层（约12–23）：接近融合态（平均 \(D\approx 0\)，14–15 层出现符号翻转）
- 26–28 层：小幅语义占优窗口
- 晚层（29–34）：韵律再次回潮（\(D>0\)）

**图片占位：**
- [FIG: Probe Accuracy on Conflict Samples]
- [FIG: Modality Dominance Curve]

---

## Slide 11 — 冲突子集统计（197条）
**标题：冲突样本上整体更偏韵律（优势不大）**

- 冲突子集（semantic ≠ prosody）：
  - avg semantic_acc \(\approx 0.7299\)
  - avg prosody_acc \(\approx 0.7895\)
  - avg dominance \(\approx 0.0596\)
- 解读（保持谨慎口径）：
  - 层内表征层面：韵律线索整体更“可读”
  - 但最终输出是否采用韵律，需要进一步看“决策轨迹/因果证据”（引出 Logit Lens & Patching）

---

## Slide 12 — Logit Lens：决策倾向
**标题：Logit Lens：逐层“如果现在就输出，会选谁？”（冲突样本197条）**

方法细节（精简但可复现的描述）：
- 只取冲突样本（semantic_label ≠ prosody_label）
- 一次前向拿到所有层表示：hidden_states[l][t]
- Readout position：生成第一个输出 token 的倾向 → 取最后一个输入 token：
  - readout_pos = T − 1
  - \(h_\ell = hidden\_states[\ell][readout\_pos]\)
- 使用真实输出路径读出 logits（保证层间可比）：
  - logits_\(\ell\) = LMHead(FinalNorm(\(h_\ell\)))
- Restricted 5-way：只看 neutral/happy/sad/angry/surprised 的 logits
- 指标：
  - Win-rate：赢家为语义/韵律/其他的比例
  - Margin：
    \[
    margin(\ell) = logit_{\text{prosody}}(\ell) - logit_{\text{semantic}}(\ell)
    \]
    - margin>0：偏韵律；margin<0：偏语义

主要观察：
- 前半段层（0–22左右）：margin 接近 0（语义 vs 韵律混合）
- 后半段层（约23–35）：margin 明显下降为负（逐层更偏语义）
- Win-rate：前半段“其他”占比高（不稳定），后半段语义赢率明显上升

**图片占位：**
- [FIG: Logit Lens Margin Curve (Conflict Samples)]
- [FIG: Logit Lens Win-Rate Curve (Conflict Samples)]

---

## Slide 13 — Activation Patching：因果证据
**标题：Activation Patching：语义强定向控制，韵律偏弱**

基本原理：
- 对样本对 (A,B)，在某层 \(\ell\) 把 A 的 audio 区域表示替换成 B 的，再继续前向到输出
- 若替换后输出显著朝 B 的目标标签变化，说明该层 audio 表示对该标签具有因果控制力

指标：
- Flip to Target：pred_patch(\(\ell\)) == target 的比例
- Flip from Base：pred_patch(\(\ell\)) != pred_base 的比例
- Delta Logit(Target)：
  \[
  \Delta(\ell) = logit_{patch}(target) - logit_{base}(target)
  \]

结果（按你记录的结论与数值）：
- Semantic patch（只换语义）：
  - 早期层（0–12）：flip_to_target 很高（最高约 0.65，多层维持 0.5+）
  - delta_logit(target) 早期很大（约 5–6），随后逐步衰减
- Prosody patch（只换韵律）：
  - flip_to_target 整体较低（约 0.14–0.26），峰值在较早层（~9）
  - delta_logit(target) 早期较小（~2.2–2.3），随后逐步衰减
- 关键边界：
  - ~14–15 层是“audio span 可控性”的边界
  - 从这里开始仅替换 audio span 难以影响最终输出（信息已扩散到全局/其他位置）

**图片占位：**
- [FIG: Flip Rate Curve (prosody)]
- [FIG: Flip Rate Curve (semantic)]
- [FIG: Delta Logit Curve (prosody)]
- [FIG: Delta Logit Curve (semantic)]

---

# Part 3 Prompt-Audio冲突机理（指令 vs 音频）

## Slide 14 — 动机与实验设置
**标题：Prompt与Audio情绪冲突：对齐过程如何发生？**

动机（两问）：
- 问题1：情感信息的消解发生在模型何处？（where）
- 问题2：谁具有主导性因果关系？（who）

数据与控制（按你文档口径）：
- 数据集：2组中英文，5种情绪的对照音频（TTS生成）
- 固定语义文本（避免语义干扰）：
  - “The package is scheduled to arrive...”

实验分组（表格直接放PPT）：
| 组别 | 音频 | 文本指令 |
|---|---|---|
| Audio-only | A | 请判断音频情绪 |
| Conflict | A | 请以T判断（T不等于A） |
| Consistent | A | 请以T判断（T等于A） |

方法（两类）：
1) Logit Lens 差分定位 where：逐层比较 T 与 A 的 logit 差值  
\[
\Delta logit_\ell = logit_\ell(T) - logit_\ell(A)
\]
2) Activation patching 定位 who：不同层替换文本/音频 hidden states，对比 PatchText vs PatchAudio

---

## Slide 15 — Finding 1：仲裁发生在晚期层（26–28）
**标题：Finding 1：Late-Layer Consolidation of Text Dominance**

- 通过对三组（Audio-only / Conflict / Consistent）做 Logit Lens 差分观察：
  - 临界层基本发生在 **26–28 层**
- 现象描述：
  - 前20层：\(\Delta logit\) 变化接近0（尚未明显仲裁）
  - 晚期层：出现明显分化与“固化”（consolidation）

**图片占位：**
- [FIG: OB1 — Late-Layer Consolidation of Text Dominance（Δlogit vs layer，三组曲线）]

---

## Slide 16 — Finding 2：文本因果主导，音频中层被wash-out
**标题：Finding 2：文本占据因果主导地位**

- 结论（按你文档原话/含义表达）：
  - Patch Text 在中层有效：文本指令在决策窗口（5–20）保持强因果效应
  - Patch Audio 无效/不稳定：音频信号在中层被 structural wash-out
- 一句话总结：
  - 当 prompt 与 audio 情绪冲突时，最终决策过程呈现“文本主导”的因果结构

**图片占位：**
- [FIG: Patch实验 — Logit shift to target emotion（Patch Text vs Patch Audio）]
- [FIG: Patch实验 — Flip rate（Patch Text vs Patch Audio）]

---

# Part 4 机理引导攻击优化（进行中）

## Slide 17 — 占位页：机理→方法论闭环
**标题：机理引导攻击优化（进行中）**

（仅写方向标题，最终版你再补内容）
- 方向1：设计特殊 text prompt，在中层产生强控制信号（探索翻转文本主导的 prompt 设计）
- 方向2：在早层注入“能保持到中层”的音频特征（用于对抗样本增强）
- 方向3：结合层级主导区间设计分段/分层损失（让方法论与机理产生直接联系）

**图片占位：**
- [FIG: “Methodology ↔ Mechanism” 双向箭头示意图（可选）]

---

## Slide 18 — 结束页（可选）
**标题：Q&A**

- 收束一句话（可选）：
  - 已验证白盒定向情绪攻击可行；通过 probe / logit lens / activation patching 给出模态冲突的层级结构与因果线索；下一步用机理反哺提升攻击鲁棒性（进行中）。


---
