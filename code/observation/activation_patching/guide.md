# 实验二（必须实现）：Activation Patching（逐层因果"开关"）

> 目标：把"逐层决策倾向（Logit Lens）"进一步升级为 **因果证据**：
> **如果我在第 l 层把 A 的某些内部激活替换为 B 的，对最终情绪输出是否会系统性改变？**
> 若会改变，则说明该层的表示位于决策的因果路径上（causal control / mediation）。

---

## 0. 本实验回答什么问题？（通俗解释）

Logit Lens 只能说"模型在某层看起来更像要输出什么"。Activation patching 做的是"真干预"：

- 取两条音频 A / B（只改变某种因素：韵律或语义）
- 在第 l 层，把 A 的**音频区域激活**换成 B 的
- 让模型继续往后算
- 看最终输出的情绪标签会不会被"拉向B"

这相当于在模型内部做一次"do(activation = B)" 的干预。

---

## 1. 方法论核心（必须严格定义、保证可复现）

### 1.1 任务与Prompt（必须与实验一一致）
- 任务：让模型输出 5 类情绪之一
- Prompt（必须固定，不得改词）：
  ```
  What is the emotion of this audio? Answer with exactly one word: neutral, happy, sad, angry, or surprised.
  ```
- 输入构造：**必须复用opens2s_io.py**（与Probe/Logit Lens 完全一致的 chat template + audio placeholder + generation prompt）

---

### 1.2 决策位置（readout position）（与实验一一致）
- `readout_pos = input_ids.shape[1] - 1`
- 用`logits[:, readout_pos, :]` 来评估"下一个token"（情绪标签的第一个 token）

> 注：本实验不依赖 generate()；仅用 forward logits + restricted 5-way 评估即可（更快、更稳定）。

---

### 1.3 Layer 编号对齐（必须与实验一、Probe 对齐）
- 共有 `n_layers = model.llm_model.config.num_hidden_layers`（例如 36）
- Patch 的 layer 编号采用你们论文中的 `0..n_layers-1`
- 实际hook 的模块：`model.llm_model.model.layers[l]` 对应论文 layer `l`

> 若你需要与 `hidden_states` 对齐：`hidden_states[l+1]` 通常等价于"layer l 输出"。
```

<thinking>
继续输出第二部分，关于成对样本的构造。
</thinking>

---

```markdown
---

## 2. 关键设计：成对样本（Pair）如何构造（必须）

Activation patching 的可信度取决于"只改变一个因素"。本实验必须实现两类pair：

### 2.1 Prosody-only Pair（只变韵律）
- 条件：**同一句文本/同一 text_id**（语义固定），但prosody emotion 不同
- 记录：
  - `semantic_label(A) == semantic_label(B)`（应成立）
  - `prosody_label(A) != prosody_label(B)`（必须成立）

**用途**：验证"韵律信息"在哪些层具有因果影响
**目标 label（target）**：`target = prosody_label(B)`（把 B 的韵律注入 A）

**构造逻辑**：
1. 按 text_id 分组所有样本
2. 在同一 text_id 组内，选取不同 prosody_label 的样本配对
3. 每个组内可产生 C(n,2) 对（n 为该text_id 的样本数）

---

### 2.2 Semantic-only Pair（只变语义）
- 条件：**不同 text_id（语义不同）**，但 prosody emotion 相同（尽量控制韵律一致）
- 记录：
  - `prosody_label(A) == prosody_label(B)`（应成立）
  - `semantic_label(A) != semantic_label(B)`（必须成立）

**用途**：验证"语义/内容情绪"在哪些层具有因果影响
**目标 label（target）**：`target = semantic_label(B)`（把 B 的语义注入 A）

**构造逻辑**：
1. 按 prosody_label 分组所有样本
2. 在同一 prosody_label 组内，选取不同 text_id 且不同 semantic_label 的样本配对
3. 若同一 prosody 组内样本不足，跳过该组（并记录 warning）

> **⚠️ Codex 确认点**：实现时必须输出 `pair_construction_report.json`，记录每类 pair 的数量、来源分布、跳过的组等信息。

---

### 2.3 Pair 采样与计算预算（必须可控）
- Pair 数量可能指数增长（同text_id 多个prosody 版本会产生很多组合）
- 必须提供配置项限制：
  - `max_pairs_prosody`（默认 100）
  - `max_pairs_semantic`（默认 100）
  - `pair_sampling_seed`（默认 42）
- 建议：默认每类先50–200 对（视算力）

### 2.4 计算量预估（必须了解）
- 每个 pair 需要：
  - 1 次 forward A（baseline）
  - 1 次 forward B（收集激活）
  - **n_layers 次 forward A**（每层 patch 一次）
- 若 n_layers=36，每个 pair 需要 **38 次 forward**
- 100 对 prosody pair + 100 对 semantic pair = **7600 次 forward**

> **⚠️ Codex 确认点**：先用少量 pair（如 10-20 对）验证流程正确性，再扩大规模。
```

<thinking>
继续输出第三部分，关于 Patch 的对象与位置。
</thinking>

---

```markdown
---

## 3. Patch 的对象与位置（必须明确）

### 3.1 Patch 的"位置集合"（默认：audio span）
我们只patch **音频 token区域**（audio span）的 residual stream表示，不改prompt text token。

**必须实现函数**：
- `find_audio_span(input_ids) -> audio_indices`
  - 通过 `<|im_audio_start|>` 与 `<|im_audio_end|>` 定位区间（推荐）
  - 或通过 `<|im_audio|>` token id 的连续段定位（备选）
  - 必须 assert：只存在 1 段 audio span（若多段，报错停止）

> **⚠️ Codex 确认点**：实现后必须在 sanity check 中打印几个样本的 audio_indices，确认定位正确。

---

### 3.2 Patch 的"激活点"（默认：layer 输出 hidden_states）
对layer `l`，patch 的对象是该层输出的 residual stream hidden state：

- hook 点：`model.llm_model.model.layers[l]` 的 forward 输出
- 替换张量：`hidden_state[:, audio_indices, :]`

这是最朴素、最稳健的 patch 点（等价于对"该层完成一次信息融合后的表示"做干预）。

---

## 4. Patch 对齐问题（必须处理）：A/B 音频长度可能不同

A 和 B 的 audio token 数可能不一致，必须定义对齐策略（否则 patch 不可复现）。

### 4.1 默认对齐策略：truncate_to_min（推荐作为 v1 默认）
- `lenA = len(audio_indices_A)`
- `lenB = len(audio_indices_B)`
- `L = min(lenA, lenB)`

**详细说明**：
- 若 `lenA > lenB`：A 的前 lenB 个位置被替换，后 (lenA-lenB) 个保持原样
- 若 `lenA < lenB`：B 的前 lenA 个位置被使用，后 (lenB-lenA) 个被丢弃
- 若 `lenA == lenB`：完全替换

优点：实现简单、稳健
缺点：忽略尾部 token

### 4.2 可选对齐策略：resample_linear（建议保留开关）
将 B 的 `[lenB, d]` 沿长度维插值到 `[lenA, d]`：
- 形状变换到 `[1, d, lenB]` → `F.interpolate(..., size=lenA, mode="linear")` → `[lenA, d]`

优点：对齐更"全量"
缺点：引入插值近似

> **⚠️ Codex 确认点**：v1 只实现 truncate_to_min；resample_linear 作为可选配置项保留接口但可不实现。
```

<thinking>
继续输出第四部分，关于指标定义。
</thinking>

---

```markdown
---

## 5. 指标定义（必须输出）

### 5.1 预测方式：restricted 5-way（必须与实验一一致）
只在五个情绪标签 token 上比较 logits（避免 open-vocab噪声）：
- label set：`neutral, happy, sad, angry, surprised`
- `pred = argmax(logits[label_token_ids])`

> label tokenization复用实验一的 `LabelTokenizerHelper`（多token 则first-token 近似 + report）。

---

### 5.2 Flip Rate（主指标，必须）
对每个 pair (A,B)、每层 l：

- baseline：`pred_base = pred(model(A))`
- patched：`pred_patch(l) = pred(model(A) with layer-l patched from B))`
- target label：
  - prosody-pair：`target = prosody_label(B)`
  - semantic-pair：`target = semantic_label(B)`

定义两个flip指标（都要输出）：

1) **Target Flip（主要）**
`flip_to_target(l) = 1[pred_patch(l) == target]`

2) **Any Flip（辅助）**
`flip_from_base(l) = 1[pred_patch(l) != pred_base]`

并统计：
- `flip_to_target_rate[l]`（平均）
- `flip_from_base_rate[l]`（平均）
- 同时输出 `eligible_count[l]`：`pred_base != target` 的样本数（避免"本来就等于 target"导致上限被抬高）

---

### 5.3 Logit Shift（建议输出，帮助解释）
对 target标签的 logits 变化：
- `delta_logit_target(l) = logit_patch(target) - logit_base(target)`

这在 flip稀少时仍能显示"方向性影响"。
```

---

```markdown
---

## 6. 核心算法流程（Codex 必须按此实现）

对每个 pair (A,B)：

1) **构造输入**（必须复用 opens2s_io.py）
   - 得到 `inputs_A`、`inputs_B`（包含音频 features/embeddings + tokenized prompt）

2) **提取 audio span indices**
   - `audio_indices_A = find_audio_span(inputs_A.input_ids)`
   - `audio_indices_B = find_audio_span(inputs_B.input_ids)`

3) **Baseline forward on A**（无 hook）
   - `logits_base = model(**inputs_A, use_cache=False)`
   - `pred_base = restricted_pred(logits_base[readout_pos])`

4) **记录 B 的逐层激活（只存audio span，小存储）**
   - 注册 hooks于所有层 `layers[l]`（或配置的 layers_to_patch）
   - 在每层 hook 中：
     - 取输出 hidden_state
     - 只slice `[:, audio_indices_B, :]`
     - `detach()` 后存到 CPU（或同device但避免累积显存）
   - forward 一次 B，得到 `cache_B[l] = act_B_l`

5) **逐层 patch A并评估**
   对每个l in `layers_to_patch`：
   - 注册一个 patch hook 于 `layers[l]`：
     - 在层输出处将 A 的 `[:, audio_indices_A, :]` 替换为对齐后的 `cache_B[l]`
     - 支持 `patch_alpha`：
       `patched = (1-alpha)*A + alpha*B`（默认 alpha=1.0 即全替换）
   - forward A（use_cache=False）
   - 计算 `pred_patch(l)`, `flip_to_target(l)`, `flip_from_base(l)`, `delta_logit_target(l)`
   - 立刻移除 hook，进入下一层

6) 写入 pair-level 记录（可选 jsonl） + layer-level 累积统计

>强制要求：每次 forward 后及时 `del` 中间变量；必要时 `torch.cuda.empty_cache()`，防止显存碎片化。

### 6.1 layers_to_patch 配置（必须）
- 默认值：`list(range(n_layers))`（所有层）
- 可配置为部分层（如 `[20, 22, 24, 26, 28, 30]`）用于快速测试
- 配置项名称：`layers_to_patch`（列表）或 `patch_all_layers`（布尔）
```

<thinking>
继续输出第五部分，关于实现要点。
</thinking>

---

```markdown
---

## 7. 实现要点（必须）

### 7.1 必须关闭 cache
- 所有 forward 调用必须 `use_cache=False`
否则 KV cache 可能导致 hook 干预不一致（尤其后续扩展到 generate 时更危险）。

### 7.2 必须 eval + inference_mode
- `model.eval()`
- `with torch.inference_mode():`

### 7.3 Hook 输出形态必须兼容 tuple
Transformer layer 输出可能是：
- Tensor（hidden_state）
- 或 tuple（hidden_state, ...）

Patch hook 必须能识别并返回相同结构，只替换第一个元素。

示例代码：
```python
def patch_hook(module, input, output):
    if isinstance(output, tuple):
        hidden_state = output[0]
        # ... 执行 patch ...return (patched_hidden_state,) + output[1:]
    else:
        # ... 执行 patch ...
        return patched_hidden_state
```

### 7.4 Hook 类型（必须）
使用 **forward hook**（`register_forward_hook`），在层输出后执行替换。
不要使用 pre-forward hook。

> **⚠️ Codex 确认点**：实现后必须验证 hook 被正确调用（可在 hook 内加print 或计数器）。
```

---

```markdown
---

## 8. 输出文件（建议固定命名，便于审稿与复现）

在一次 run 的输出目录下保存：

**配置与元信息**：
- `patching_config.json`（本次运行的所有配置）
- `tokenization_report.json`（复用实验一；或软链接）
- `pair_construction_report.json`（pair 构造的统计信息）
- `pair_list_prosody.jsonl`（实际用到的 prosody pair 列表）
- `pair_list_semantic.jsonl`（实际用到的 semantic pair 列表）

**原始记录（可选，体积较大）**：
- `patching_records_prosody.jsonl`（每对、每层的详细结果）
- `patching_records_semantic.jsonl`

**聚合结果（必须）**：
- `patching_metrics_prosody.csv`：每层的 flip_to_target_rate / flip_from_base_rate / delta_logit_target_mean / eligible_count
- `patching_metrics_semantic.csv`：同上

**绘图（必须）**：
- `flip_rate_curve_prosody.png`
- `flip_rate_curve_semantic.png`
- 可选：`delta_logit_curve_prosody.png` / `delta_logit_curve_semantic.png`
```

---

```markdown
---

## 9. Sanity Checks（必须做，否则结果不可信）

### 9.1 Self-patch 不应改变输出（必做）
选若干样本 S：
- 记录 S 的 cache_B
- 用相同 S 作为 A，在任意层 patch（等价于"替换成自己"）
- 结果必须：`pred_patch == pred_base`（允许极少数数值抖动，但应接近 100% 一致）

若不成立：说明 hook 点或readout_pos 或 logits 计算有bug。

> **⚠️ Codex 确认点**：必须在主实验前运行 self-patch 检查，输出 `sanity_check_self_patch.json`。

### 9.2 Random patch 控制（必做）
- 将 B 换成随机样本（不满足"只变一个因素"）
理论上会引入大量噪声，target flip 不应呈现与你的机制假设一致的"层结构峰值"。

### 9.3 Non-audio位置 patch（建议做）
- 在同一层 patch 一段随机文本 token positions（非audio span）
- 观察：target flip 不应与prosody/semantic pair逻辑一致（用于排除"任何 patch 都会翻转"的伪因果）

### 9.4 无冲突样本对照（建议做）
在语义=韵律样本上做同样 patch（或只测少量层），flip 应整体更低（用于排除"任务本身不稳定"）。

### 9.5 Audio span 定位验证（必做）
在实验开始前，打印 5-10 个样本的：
- `input_ids` 长度
- `audio_indices` 起止位置
- 对应的 token 内容（确认是 audio token）

> **⚠️ Codex 确认点**：输出 `sanity_check_audio_span.json`，包含上述信息。
```

<thinking>
继续输出最后部分，关于结果解释边界、不做的事、以及最终提醒。
</thinking>

---

```markdown
---

## 10. 结果解释边界（提前声明，防止误读）

### 10.1 因果解释的边界
- Activation patching 证明的是"该层的被patch 表示对输出有因果影响"，不直接等价于"信息只在该层产生"。
- patch 可能产生 out-of-distribution 表示，建议保留 `patch_alpha`（可做 0.25/0.5/1.0 的剂量响应验证，但 v1 可只实现 alpha=1.0）。

### 10.2 Semantic-only pair 的固有限制（必须在论文中讨论）
即使两个样本的 `prosody_label` 相同，它们的**实际声学特征**可能差异很大（不同说话人、语速、音色等）。

这意味着 semantic-only pair 的 patch 可能同时引入：
- 语义变化（目标）
- 声学特征变化（混淆变量）

**建议**：
- 在论文 limitation 部分讨论此问题
- 可选：在结果分析时，对比 prosody-pair 和 semantic-pair 的flip rate 差异，若semantic-pair 的 flip 更"嘈杂"，可能与此有关

---

## 11. 本实验不做的事（明确边界）
- 不做 head-level/MLP-level 更细粒度的 patch（先把 layer-level 跑通）
- 不做生成多token 的复杂评估（统一用 readout_pos 的 restricted 5-way）
- 不引入任何训练（保持纯分析/干预）
- 不实现 resample_linear 对齐策略（v1 只用 truncate_to_min）

---

## 12. 文件结构建议（给Codex）

```
activation_patching/
├── src/
│   ├── __init__.py
│   ├── patching.py           # 核心 patching 逻辑
│   ├── pair_constructor.py   # pair 构造逻辑
│   ├── hooks.py              # hook 注册与管理
│   └── visualization.py      # 绘图函数
├── scripts/
│   ├── run_patching.py       # 主入口脚本
│   └── sanity_checks.py      # sanity check 脚本
├── configs/
│   └── patching_config.yaml
└── outputs/
```

---

## 13. 最终提醒（最容易踩坑的点）

1) **audio span 定位必须正确且稳定**（强烈建议通过 start/end token 找区间）
2) **A/B 音频长度不一致必须有明确对齐策略**（truncate_to_min 为默认）
3) **hook 输出结构（tuple）必须处理正确**
4) **use_cache 必须关**
5) **self-patch sanity check 必须通过，否则整套patching 无效**
6) **forward hook，不是 pre-forward hook**
7) **每个 pair 需要 38 次 forward，注意计算量**

---

## 14. Codex 实现前必须确认的 Checklist

在开始写代码前，Codex 必须确认以下事项：

- [ ] 已读取 opens2s_io.py，理解输入构造逻辑
- [ ] 已确认 audio span 的定位方式（start/end token 还是连续 audio token）
- [ ] 已确认 `model.llm_model.model.layers` 的结构
- [ ] 已确认 layer输出是tensor 还是 tuple
- [ ] 已规划 sanity check 的实现顺序（先 self-patch，再主实验）
- [ ] 已理解 pair 构造逻辑（prosody-pair vs semantic-pair）

>确认完成后，先实现 sanity_checks.py，验证 hook 和 audio span 定位正确，再实现主流程。
```

