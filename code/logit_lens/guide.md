# 实验一（必须实现）：Decision‑level Tracing / Logit Lens

> 目标：将“probe 可读性”升级为“模型逐层决策倾向”。
> 本实验不是因果，但能回答：模型在生成情绪标签时，偏向语义/韵律的趋势是如何逐层形成的？

---

## 1) 通俗解释：这个实验在干什么？
我们不训练任何外部分类器，直接用模型自己的 **LM head** 去“偷看”每一层的临时倾向。

具体做法：
- 对每层的 hidden state（在**情绪标签输出位置**）做一次 `LM head` 投影
- 看这一层更像要输出 `semantic_label` 还是 `prosody_label`
- 画出随层变化的 margin 曲线

最终得到一条：
> **margin(layer) = logit(prosody_label) − logit(semantic_label)**

若该曲线和你 probe 的 D(l) 结构一致，说明“层级可读性”正在影响模型的决策轨迹。

---

## 2) 方法论核心（必须严格对齐）

### 2.1 选样本
- 默认分析 **冲突样本**（semantic != prosody）
- 可选输出 all‑samples 对照

### 2.2 决策位置（readout position）
我们要追踪的是“模型准备生成情绪标签的那一步”。

**要求**：
- 输入构造必须与现有 OpenS2S inference 完全一致
- 使用 `add_generation_prompt=True` 的 chat template
- **readout_pos = input_ids.shape[1] - 1**
  即“生成开始前的最后一个输入 token 位置”

**必须做的 sanity check**：
- `generate(max_new_tokens=1, temperature=0)` 的第一个 token
  必须与 `logits[:, -1, :].argmax()` 一致（抽样检查）

### 2.3 Logit Lens 投影方式（必须与模型一致）
通常模型输出 logits 是：
```
logits = lm_head( final_norm(hidden_state) )
```
所以在每层应当：
```
logits_l = lm_head( final_norm(h_l) )
```

**实现要求**：
- 写 `apply_final_norm(model, x)`
  优先尝试 `model.model.norm`, `model.norm`, `model.transformer.ln_f` 等
- 若找不到 norm，退化成 identity，但必须 **warning**

### 2.4 标签 tokenization（必须处理）
情绪标签：
`neutral, happy, sad, angry, surprised`

**默认策略**：
- `tokenizer.encode(label, add_special_tokens=False)`
- 若多 token → 使用 **first-token 近似**
- 必须输出 `tokenization_report.json`（含 token ids / token strings / 是否多 token）
- 若多 token，打印 warning（不中断）

> 可选升级版：multi-token logprob tracing（本实验不做）

---

## 3) 指标与输出（必须）

### 3.1 Margin 曲线（主图）
对每个样本、每层：
```
margin_l = logit(prosody_label) - logit(semantic_label)
```

输出：
- sample-level mean/std 曲线
- group-by-text mean/std 曲线（避免同句重复计权）

### 3.2 Restricted 5‑way win‑rate（强烈建议）
在 5 个情绪标签内 softmax：
- `pred_l = argmax(softmax(logits_l[label_ids]))`
- 统计：
  - win_rate_semantic
  - win_rate_prosody
  - win_rate_other

这张图更直观展示“逐层站队”。

---

## 4) 代码实现大纲（给 Codex）

### 4.1 新增模块建议（逻辑，不限定路径）
- `logit_lens.py`（核心逻辑）
- `run_logit_lens.py`（入口脚本）

关键函数建议：
- `LabelTokenizerHelper`
- `get_readout_position(input_ids)`
- `apply_final_norm(model, x)`
- `compute_logit_lens_metrics(sample, outputs, ...)`
- `aggregate_metrics(records, group_by_text=...)`
- `plot_margin_curve(...)` / `plot_win_rates(...)`

### 4.2 数据流（必须按序）
1) load config
2) load dataset
3) filter conflict_only（可配置）
4) for each sample (建议 batch_size=1)：
   - build inputs（必须复用你现有的输入构造逻辑）
   - `model.eval()`, `torch.no_grad()`
   - forward with `output_hidden_states=True`
   - readout_pos = input_len-1
   - 对每层 l：
     - 取 `h_l = hidden_states[layer_idx][:, readout_pos, :]`
     - `logits_l = lm_head(final_norm(h_l)).float()`
     - 取出 5 个标签 logits
     - 计算 margin / win‑rate
     - 只保存标量（避免显存爆炸）
5) 聚合 sample‑level + group‑level
6) 保存 csv/json + plots

### 4.3 Layer index 对齐（必须）
HF 的 `hidden_states` 通常长度是 `n_layers + 1`：
- `hidden_states[0]` = embedding 输出
- `hidden_states[1]` = Transformer 第 0 层输出

为了和你 probe 实验的层编号一致：
- 你的 layer 0 → `hidden_states[1]`
- 你的 layer L → `hidden_states[L+1]`

若发现 `len(hidden_states) != n_layers + 1` 必须报错并停止。

---

## 5) 输出文件（建议固定命名）
- `tokenization_report.json`
- `logit_lens_records.jsonl`（逐样本逐层 margin，可选）
- `logit_lens_metrics_sample.csv`
- `logit_lens_metrics_group.csv`
- `margin_curve_conflict.png`（主图）
- `winrate_curve_conflict.png`（辅助图）
- 可选：`margin_curve_all.png` / `winrate_curve_all.png`

---

## 6) 必须做的 sanity checks
1. `generate(1 token)` 与 `logits[:, -1].argmax()` 一致
2. label tokenization 报告输出并检查
3. 层数与 config 对齐
4. conflict-only 与 all-samples 的趋势对照

---

## 7) 期望趋势（用于结果自检）
若与 probe 结构一致，margin 曲线应呈现：
- 0–14 层偏 prosody（margin>0）
- 12–23 近 0
- 26–28 可能出现 semantic 小窗口（margin<0）
- 29–34 再次回到 prosody

---

**最终提醒**
本实验最大风险是输入构造和 readout_pos 对齐错误。
**必须复用现有 OpenS2S inference 输入构造逻辑，避免重新拼 prompt。**
```
