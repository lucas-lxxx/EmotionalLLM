# 实验报告（实验一 + 实验二）

> 日期：2026-01-31
> 目标：用**实验一（Logit Lens）**看“逐层倾向”，用**实验二（Activation Patching）**给出“因果干预证据”。

---

## 1. 实验设置（简要）
- 任务：识别音频情绪（neutral/happy/sad/angry/surprised）
- Prompt：固定为
  > What is the emotion of this audio? Answer with exactly one word: neutral, happy, sad, angry, or surprised.
- 样本：冲突样本（semantic ≠ prosody）197 条
- 模型：OpenS2S

---

## 2. 实验一：Logit Lens（趋势）

### 2.1 结果图
- `logit_lens/outputs/margin_curve_conflict.png`
- `logit_lens/outputs/winrate_curve_conflict.png`

### 2.2 主要观察（直观描述）
- **前半段层（0–22左右）**：
  - margin 接近 0（没有明显偏向），说明“语义 vs 韵律”在这一段比较混合。
- **后半段层（约 23–35）**：
  - margin 明显下降到负值，代表模型逐层更偏向“语义标签”。
- **Win-rate 曲线**：
  - 前半段“其他”占比高（模型不稳定地输出非语义/韵律目标）
  - 后半段语义赢率明显上升，韵律赢率上升较小

**一句话总结（实验一）**：
> 模型在后半层对“语义标签”的倾向明显增强，韵律优势不突出。

---

## 3. 实验二：Activation Patching（因果证据）

### 3.1 Pair 构造概况
- Prosody-only pairs：从 291 个候选中采样 100 对
- Semantic-only pairs：从 2911 个候选中采样 100 对

### 3.2 结果图
- Prosody：
  - `activation_patching/outputs/flip_rate_curve_prosody.png`
  - `activation_patching/outputs/delta_logit_curve_prosody.png`
- Semantic：
  - `activation_patching/outputs/flip_rate_curve_semantic.png`
  - `activation_patching/outputs/delta_logit_curve_semantic.png`

### 3.3 主要观察（直观描述）
**Prosody patch（只换韵律）**
- flip_to_target 整体较低（约 0.14–0.26），最高出现在较早层（~9 层）。
- delta_logit(target) 在早期层很高（>2），之后逐步下降，后期接近 0。
- 说明：**韵律信息更偏“早层”有效**，但整体“翻转到目标”的能力有限。

**Semantic patch（只换语义）**
- flip_to_target 早期层很高（最高约 0.65），0–12 层都维持在 0.5 以上。
- delta_logit(target) 在早期层非常大（约 5–6），之后逐步下降。
- 说明：**语义信息在早期层有更强、更稳定的因果影响**。

**一句话总结（实验二）**：
> 语义 patch 在早期层能显著拉动输出，韵律 patch 的因果影响较弱且更局部。

---

## 4. 两个实验对照结论
- **实验一（趋势）**显示：后期层更偏语义输出；
- **实验二（因果）**显示：语义 patch 的因果影响主要集中在早期层，而韵律 patch 效果较弱。

**直观解释**：
- 模型可能在早期层就把语义相关信息写入表示，后期层更多是在“确认/稳定”语义结果；
- 韵律信息对最终决策影响有限，且更难触发明显翻转。

---

## 5. 限制与注意事项
- Semantic-only pair 虽然控制了 prosody 标签，但实际声学差异仍可能存在（混淆因素）。
- Activation patch 可能带来分布外激活，故需结合 delta_logit 曲线一起看趋势。
- 本次实验仅做 layer-level patch，未深入到 attention head/MLP。

---

## 6. 输出文件列表（便于复现）
- 实验一：`/data1/lixiang/lx_code/logit_lens/outputs/*`
- 实验二：`/data1/lixiang/lx_code/activation_patching/outputs/*`

---

## 7. 一句话总结
**语义信息的因果影响更强、更早出现；韵律信息影响较弱且更局部。**

