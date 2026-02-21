```markdown
# Logit Lens 实验：方法论与核心实现文档

> 版本：v1.0
> 目的：供后续审查者理解实验设计与核心实现逻辑
> 前置依赖：已完成的 Probe 实验（语义 vs 韵律逐层可读性分析）

---

## 1. 实验目标

**将 Probe 的"可读性"升级为"决策倾向"。**

| 实验 | 回答的问题 | 方法 |
|------|-----------|------|
| Probe | 某层的音频表示里，能读出语义/韵律信息吗？ | 外部线性分类器 |
| **Logit Lens** | 某层时，模型更倾向输出哪个情绪标签？ | 模型自己的 LM head |

Logit Lens 不训练任何分类器，直接用模型的输出头"偷看"每一层的临时决策倾向。

---

## 2. 方法论核心

### 2.1 核心公式

对每一层 $l$，在**决策位置**提取 hidden state，经过模型的 final norm 和 LM head 投影到词表：

$$
\text{logits}_l = \text{LM\_head}(\text{Norm}(h_l))
$$

然后计算 margin：

$$
\text{margin}(l) = \text{logit}(\text{prosody\_label}) - \text{logit}(\text{semantic\_label})
$$

- $\text{margin} > 0$：该层倾向输出韵律标签
- $\text{margin} < 0$：该层倾向输出语义标签

### 2.2 决策位置（Readout Position）

使用**最后一个输入 token 位置**：

```
readout_pos = input_ids.shape[1] - 1
```

**原因**：Causal LM 中，位置 $T-1$ 的 hidden state 决定位置 $T$ 的输出（即第一个生成 token）。

**与 Probe 的区别**：
- Probe 使用 audio span 的 mean pooling（测"感知"）
- Logit Lens 使用最后一个 token（测"决策"）

这是有意设计，两者互补。

### 2.3 Layer Index 对齐

HuggingFace 的 `hidden_states` 长度为 `n_layers + 1`：
- `hidden_states[0]` = embedding 输出
- `hidden_states[l+1]` = Transformer 第 $l$ 层输出

为与 Probe 一致：
- 本实验的 layer 0 → `hidden_states[1]`
- 本实验的 layer $L$ → `hidden_states[L+1]`

---

## 3. 核心实现

### 3.1 Logit Lens 投影函数

```python
def logit_lens_projection(hidden_state, model):
    """
    对单层 hidden state 做 logit lens 投影
  
    Args:
        hidden_state: [batch, seq_len, hidden_dim]
        model: OmniSpeech 模型
  
    Returns:
        logits: [batch, seq_len, vocab_size]
    """
    normed = model.llm_model.model.norm(hidden_state)
    logits = model.llm_model.lm_head(normed)
    return logits.float()  # 避免 fp16 精度问题
```

### 3.2 标签 Tokenization

情绪标签：`neutral, happy, sad, angry, surprised`

策略：
- 单 token → 直接使用
- 多 token → 使用 first-token 近似，并记录 warning

### 3.3 指标计算

**Margin 曲线**（主指标）：
```python
margin_l = logits_l[prosody_token_id] - logits_l[semantic_token_id]
```

**5-way Win-rate**（辅助指标）：
```python
label_logits = logits_l[label_token_ids]  # 5 个标签
pred_l = argmax(label_logits)
# 统计 win_semantic, win_prosody, win_other
```

---

## 4. 输入输出规范

### 4.1 输入
- 数据集：与 Probe 实验相同的 247 条样本
- 筛选：默认使用冲突样本（semantic ≠ prosody，共 197 条）
- Prompt：与 Probe 完全一致
  ```
  What is the emotion of this audio? Answer with exactly one word: 
  neutral, happy, sad, angry, or surprised.
  ```

### 4.2 输出

| 文件 | 说明 |
|------|------|
| `tokenization_report.json` | 标签分词情况 |
| `logit_lens_metrics_sample.csv` | sample-level 逐层统计 |
| `logit_lens_metrics_group.csv` | group-by-text 逐层统计 |
| `margin_curve_conflict.png` | 主图：margin 随层变化 |
| `winrate_curve_conflict.png` | 辅助图：win-rate 随层变化 |

---

## 5. 验证检查（Sanity Checks）

| 检查项 | 预期结果 |
|--------|----------|
| `generate(1 token)` vs `logits[:,-1].argmax()` | 必须一致 |
| `len(hidden_states)` | 必须等于 `n_layers + 1` (37) |
| 标签 tokenization | 单 token 优先；多 token 需记录 |

---

## 6. 期望结果

若与 Probe 结构一致，margin 曲线应呈现：
- **0–14 层**：margin > 0（偏 prosody）
- **12–23 层**：margin ≈ 0（融合区）
- **26–28 层**：margin < 0（语义小窗口）
- **29–34 层**：margin > 0（prosody 回潮）

**重要提示**：Probe 测的是"感知可读性"，Logit Lens 测的是"输出倾向"，两者趋势可能相似但不必严格一致。

---

## 7. 审查要点

审查者需重点确认：

1. **输入构造**：是否与 Probe 实验使用相同的 prompt 和 chat template？
2. **Readout 位置**：是否正确使用最后一个 token 位置？
3. **Layer 对齐**：layer 0 是否对应 `hidden_states[1]`？
4. **Norm + LM head**：路径是否正确（`model.llm_model.model.norm` / `model.llm_model.lm_head`）？
5. **Sanity checks**：三项检查是否通过？

---

## 附录：与 Probe 实验的对比

| 维度 | Probe | Logit Lens |
|------|-------|------------|
| 目标 | 测信息可读性 | 测决策倾向 |
| 分类器 | 外部线性分类器 | 模型自己的 LM head |
| 位置 | audio span mean pooling | 最后一个 token |
| 训练 | 需要训练 probe | 无需训练 |
| 输出 | 准确率 / F1 | logit margin / win-rate |
