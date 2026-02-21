```markdown
# AUDIO_SEMANTIC_VS_PROSODY_CONFLICT.md
**目的**：构建“语义情绪 vs 韵律情绪（音频内冲突）”的机制实验，并产出逐层主导性证据图（安全四大标准下可复现实验）。

---

## 0. 核心问题（必须清晰）

> 在 OpenS2S 的不同层中，**韵律情绪（prosody）** 与 **语义情绪（semantic）** 哪一个对表示的影响更强？
> 当两者冲突时，模型的表示更偏向哪一侧？

实验不涉及 prompt 冲突，prompt 固定为**中性**，聚焦“音频内部冲突”。

---

## 1. 数据规范（已确定）

### 1.1 `text.jsonl` 格式（50条）
每行包含：
```json
{"id":"t000","text_emotion":"neutral","text":"The package is scheduled to arrive tomorrow afternoon."}
```

- `id`：文本唯一编号（50条）
- `text_emotion`：语义情绪标签（5类）
- `text`：文本内容（英语）

### 1.2 音频文件组织
**prosody 情绪目录**建议如下：
```
audio_root/
  neutral/t000.wav
  happy/t000.wav
  sad/t000.wav
  angry/t000.wav
  surprised/t000.wav
  ...
```

- `prosody_emotion` 由目录名确定
- 每条文本生成 5 种韵律版本，共 250 条音频
- 采样率/音频格式可能不统一，代码需自适应（由 OpenS2S 的 audio processor 负责统一）

---

## 2. 固定 Prompt（中性）

统一使用：
```
PROMPT = "What is the emotion of this audio? Answer with exactly one word: neutral, happy, sad, angry, surprised."
```

> 不做 prompt 冲突，保证实验只反映“音频内部的语义-韵律冲突”。

---

## 3. 实验设计（核心）

### 3.1 样本构造
对每条 `text_id`，有 5 个音频样本：

| text_id | semantic (text_emotion) | prosody (folder) |
|--------|--------------------------|------------------|
| t000   | neutral                  | neutral/happy/sad/angry/surprised |

**冲突样本**：`semantic != prosody`
**一致样本**：`semantic == prosody`

---

### 3.2 表示抽取策略（由 Codex 实现）

建议使用 **audio span mean pooling**（最稳健）：

- 在 LLM 的每一层 hidden state 中，找到 audio span 的 token 位置
- 对 audio span 做 mean pooling 得到该层表示向量
- 这样能最大限度代表“音频信息”，避免 prompt token 干扰

> 如果 audio span 位置无法直接获得，Codex 需从 OpenS2S 代码中定位：
> - `AUDIO_TOKEN_INDEX`
> - `speech_mask`
> - 或 `prepare_inputs_labels_for_llm` 返回的 audio长度信息
> 以最可靠方式定位 audio span。

---

### 3.3 逐层双 Probe（核心指标）

对每一层 ℓ 训练两个线性 probe：

1) **Semantic Probe**
预测 `text_emotion`（语义情绪）

2) **Prosody Probe**
预测 `prosody_emotion`（韵律情绪）

计算：
- `Acc_semantic(ℓ)` / `F1_semantic(ℓ)`
- `Acc_prosody(ℓ)` / `F1_prosody(ℓ)`

定义主导性指标：
```
D(ℓ) = Acc_prosody(ℓ) - Acc_semantic(ℓ)
```

- D(ℓ) > 0：该层韵律主导
- D(ℓ) < 0：该层语义主导

---

### 3.4 冲突子集分析（必须做）

仅在冲突样本 (`semantic != prosody`) 上评估：

- `Acc_semantic_conflict(ℓ)`
- `Acc_prosody_conflict(ℓ)`
- `D_conflict(ℓ)`

这直接回答：“在冲突条件下，该层更偏向语义还是韵律？”

---

## 4. 数据划分原则（必须遵守）

### **Group Split by text_id**

禁止按样本随机划分（会数据泄漏）。
必须按文本分组：

- 同一 `text_id` 的 5 个韵律版本必须在同一侧（train 或 test）

建议：`GroupKFold(n_splits=5)`，group = text_id

---

## 5. 工程化实现要求（Codex 需实现）

### 5.1 脚本结构建议

```
/experiment
  ├── run_experiment.py         # 主入口（CLI）
  ├── opens2s_wrapper.py         # 模型加载与hidden states抽取
  ├── data_loader.py             # jsonl + 音频目录索引
  ├── feature_cache.py           # hidden states 缓存（可选）
  ├── probe.py                   # 线性probe训练与评估
  ├── metrics.py                 # Acc/F1/D计算
  ├── plot_results.py            # 图像生成
  └── outputs/
       ├── metrics_per_layer.csv
       ├── dominance_curve.png
       ├── acc_curves.png
       ├── conflict_curves.png
       └── summary.json
```

---

### 5.2 关键输出（必须产出）

1. **metrics_per_layer.csv**
   - layer, acc_sem, f1_sem, acc_pros, f1_pros, D
   - conflict 版本同列（acc_sem_conflict 等）

2. **dominance_curve.png**
   - D(ℓ) 曲线（全样本 vs 冲突样本）

3. **acc_curves.png**
   - Acc_sem vs Acc_pros 随层变化曲线

4. **summary.json**
   - `dominance_peak_layer`（D 最大层）
   - `crossing_layer`（D 从正到负或负到正的层）
   - `prosody_dominant_range`（连续 D>0 的层区间）

---

## 6. 线性 Probe 的选择（推荐）

- 使用 **Logistic Regression（多类）** 或 **Linear Layer + softmax**
- 建议 `sklearn` 实现，稳定可控：
  - `LogisticRegression(multi_class="multinomial", max_iter=1000, C=1.0)`
- 指标：Accuracy + Macro-F1

---

## 7. 运行入口（建议 CLI）

`run_experiment.py` 参数建议：

```
python run_experiment.py \
  --text_jsonl ./text.jsonl \
  --audio_root ./audio_root \
  --model_path /path/to/opens2s \
  --prompt "What is the emotion..." \
  --batch_size 1 \
  --n_splits 5 \
  --out_dir ./outputs \
  --cache_dir ./cache \
  --device cuda
```

---

## 8. 复现性要求

- 固定随机种子（numpy / torch / sklearn）
- 模型必须 `eval()` + `torch.no_grad()`
- 记录 OpenS2S 版本 & commit hash

---

## 9. 可选加分项（建议）

1) **Permutation Test**：随机打乱 semantic 标签，看 probe 是否仍高，排除“伪相关”
2) **ANOVA / variance decomposition**：量化语义 vs 韵律对每层表示的解释方差
3) **Layer-wise distance**：同一 text_id 不同 prosody 的表示距离曲线（定性验证）

---

## 10. 给 Codex 的核心实现提示

1) **Hidden states**
   - 从 OpenS2S 中取出 LLM hidden states（`output_hidden_states=True`）
   - 不要走 `generate()`，直接 forward 一次即可

2) **Audio span 位置**
   - 优先使用 `AUDIO_TOKEN_INDEX` 或 `speech_mask`
   - 必须保证 audio positions 与 hidden_states 对齐

3) **数据缓存**
   - 250条样本 × 32层 × hidden_dim
   - 建议缓存为 `.pt` 或 `.npz`，避免重复推理

---

## 11. 最终要回答的结论（论文级）

- **Finding 1**：声学情绪在某一层区间具有最高可分性（D>0）。
- **Finding 2**：在语义-韵律冲突样本中，该层区间的表示更偏向韵律。
- **Finding 3**：深层表现出语义主导或仲裁固化趋势（D<0 或下降）。

这些结论将直接连接你们的白盒攻击机制解释。

---

如需补充：我可以继续给出“图像标准模板”、“Figure caption 文案”、“统计显著性检验设计”。