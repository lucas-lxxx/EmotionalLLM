# OB1 实验方法论总结

**目标**：证实"音频-文本情绪冲突时，仲裁在 LLM 高层固化，且文本指令在中层保持因果主导性"

**核心假设**：
- **H1**：仲裁在高层（late-layer consolidation）
- **H2**：文本因果主导（text causal dominance）

---

## 一、实验设计概览

### 1.1 数据准备

#### 音频数据（N=18）
- **语义固定**：同一句话"The package is scheduled to arrive by tomorrow afternoon."（中英文各一套）
- **情绪变化**：5 种情绪（neutral, cheerful, sad, angry, terrified）
- **强度分级**：每种情绪 2 档（normal, strong）
- **语言覆盖**：中文（zh）、英文（en）
- **TTS 生成**：同一说话人音色，只变韵律情绪

**数据位置**：`/data3/xuzhenyu/dataset from TTS/OB1/renamed/`  
**命名规范**：`{lang}_{emotion}_{intensity}.wav`（如 `zh_cheerful_strong.wav`）

#### 文本指令（控制变量）

**三种 prompt 模式**：

1. **Audio-only**（基线）
   ```
   "请判断这段音频的情绪。只输出标签。"
   ```

2. **Text Control（冲突/一致）**
   ```
   "请判断这段音频的情绪。要求：请以「{TARGET_EMOTION}」的语气/情绪来判断并给出标签。"
   ```

3. **Consistent**（控制组）
   - 同 Text Control，但 TARGET_EMOTION = 音频真实情绪

---

### 1.2 实验条件矩阵

对每条音频，构造 **3 组对照**：

| 条件 | 音频情绪 | 文本指令情绪 | 用途 |
|------|---------|------------|------|
| **Audio-only** | A | - | Baseline：验证模型能识别音频 |
| **Conflict** | A | T（T ≠ A） | 冲突：观察仲裁 |
| **Consistent** | A | T（T = A） | 控制组：无冲突 |

**冲突映射表**：
- cheerful ↔ sad
- angry ↔ neutral
- terrified ↔ cheerful
- neutral ↔ angry
- sad ↔ cheerful

**总实验次数**：18 音频 × 3 条件 = **54 次推理**

---

## 二、核心实验方法

### 实验 A：Logit Lens（层级定位）

#### 原理
在每一层 ℓ，用 `lm_head` 把该层的 hidden states 投影到 vocabulary logits，观察"文本情绪 T"和"音频情绪 A"的 logit 差异。

#### 操作步骤

1. **Forward 一次，开启 `output_hidden_states=True`**
   ```python
   outputs = model.alm(
       inputs_embeds=inputs_embeds,
       attention_mask=attention_mask,
       output_hidden_states=True,
       return_dict=True,
   )
   ```

2. **提取所有层的 hidden states**
   ```python
   hidden_states = outputs.hidden_states  # tuple of (layer_0, layer_1, ..., layer_28)
   ```

3. **逐层投影到 vocab**
   ```python
   for layer_idx in range(num_layers):
       hs = hidden_states[layer_idx]  # [batch, seq, hidden_dim]
       logits_at_layer = model.alm.lm_head(hs)  # [batch, seq, vocab_size]
   ```

4. **计算 Δlogit**
   ```python
   last_pos = logits_at_layer.shape[1] - 1
   logit_T = logits_at_layer[0, last_pos, token_id_T]
   logit_A = logits_at_layer[0, last_pos, token_id_A]
   delta_logit = logit_T - logit_A
   ```

5. **得到曲线**
   - 每层一个 delta 值 → 得到 `delta_logit_by_text_layer[0...27]`

#### 关键脚本
**`mech1_text_vs_audio_dominance.py`**
```bash
python -u mech1_text_vs_audio_dominance.py \
  --audio_path "/path/to/audio.wav" \
  --audio_emotion cheerful \
  --text_emotion sad \
  --prompt_mode conflict \
  --out_json "output.json"
```

#### 产出
- **JSON 文件**：包含 `delta_logit_by_text_layer`（28 个数值）
- **图表**：用 `mech1_analyze_ob1.py` 生成平均曲线

---

### 实验 B：Activation Patching（因果验证）

#### 原理
在不同层 ℓ，**强行替换**某个模态的内部表征，观察输出是否翻转。

#### 操作步骤

**Step 1：准备两个条件**

- **Source**：原始冲突
  - 音频：cheerful
  - 文本：sad 指令
  - Forward → 得到 `hidden_states_source[0...27]`
  - 输出：预测 "sad"

- **Donor**：用于替换的条件
  - **同一音频**：cheerful
  - 文本：cheerful 指令
  - Forward → 得到 `hidden_states_donor[0...27]`
  - 输出：预测 "cheerful"

**Step 2：在第 ℓ 层做 Patching**

**Patch Text（替换文本指令部分）**：
```python
# 用 PyTorch hook 在 layer ℓ 的输出上操作
def patch_hook(module, input, output):
    hidden_states = output[0]  # [batch, seq, hidden]
    donor_states = hidden_states_donor[layer_idx]
    
    # 只替换"文本 token 位置"的 states
    patched_states = hidden_states.clone()
    patched_states[:, text_positions, :] = donor_states[:, text_positions, :]
    
    return (patched_states,) + output[1:]

# 注册 hook 到 layer ℓ
model.alm.model.layers[layer_idx].register_forward_hook(patch_hook)

# Forward（会触发 patching）
outputs_patched = model.alm(inputs_embeds_source, ...)
```

**Step 3：观察输出是否翻转**
```python
pred_baseline = "sad"  # 原始输出
pred_patched = ?       # patching 后的输出

if pred_patched != pred_baseline:
    flip = True  # 翻转了 → 该层有因果影响
```

**Step 4：对多个层重复（0, 5, 10, 15, 20, 25, 27）**
- 得到每层的 flip rate
- 画出曲线：flip rate vs layer

#### 关键脚本
**`mech1_patching.py`**
```bash
python -u mech1_patching.py \
  --audio_path "/path/to/audio.wav" \
  --audio_emotion cheerful \
  --text_source sad \
  --text_target cheerful \
  --patch_layers "0,5,10,15,20,25,27" \
  --out_json "patching_result.json"
```

#### 产出
- **JSON 文件**：每层的 flip 结果 + logit shift
- **图表**：双子图（Flip Rate + Logit Shift）

---

### 实验 C：Attention Mass 重构（补充）

#### 原理
用 Q/K 重构 last-token 对 audio prefix 的注意力权重，观察注意力是否随层变化。

#### 操作（已集成在 mech1 脚本里）
```bash
python -u mech1_text_vs_audio_dominance.py \
  --attn_probe_layers "0,5,10,15,20,25,27" \
  ...
```

#### 产出
- JSON 里增加 `attn_probe_last_token` 字段
- 包含每层的 `mass_audio` 和 `mass_text`

---

## 三、完整执行流程（可复现）

### Step 1：数据准备与重命名
```bash
cd /home/xuzhenyu/Kimi-Audio
python rename_ob1_files.py
```

**产出**：`/data3/.../OB1/renamed/` 下 18 条标准命名音频

---

### Step 2：批量运行 Logit Lens（实验 A）
```bash
conda activate kimiaudio
cd /home/xuzhenyu/Kimi-Audio

python -u mech1_runner_ob1_batch.py 2>&1 | tee ob1_batch_run.log
```

**耗时**：约 20-30 分钟  
**产出**：`mech1_outputs/ob1_experiment/*.json`（54 个文件）

---

### Step 3：分析 Logit Lens 结果
```bash
# 生成平均曲线
python -u mech1_analyze_ob1.py

# 找临界层
python -u mech1_find_critical_layer.py
```

**产出**：
- `mech1_figs/ob1_analysis/delta_logit__combined.png`（Finding 1 核心图）
- `mech1_figs/ob1_analysis/critical_layers.csv`（统计表）
- `mech1_figs/ob1_analysis/ob1_flip_summary.csv`（翻转率表）

---

### Step 4：运行 Patching 实验（实验 B）

**选 3-4 个代表性样本**：
```bash
# 样本 1
python -u mech1_patching.py \
  --audio_path "/data3/xuzhenyu/dataset from TTS/OB1/renamed/en_cheerful_strong.wav" \
  --audio_emotion cheerful \
  --text_source sad \
  --text_target cheerful \
  --patch_layers "0,5,10,15,20,25,27" \
  --out_json "/home/xuzhenyu/Kimi-Audio/mech1_outputs/patching/en_cheerful_strong_sad_to_cheerful.json"

# 样本 2-4 同理...
```

**耗时**：每个样本约 5 分钟（7 层 × 2 类 patching）  
**产出**：`mech1_outputs/patching/*.json` + 单样本图

---

### Step 5：汇总 Patching 结果
```bash
python -u mech1_summarize_patching.py
```

**产出**：
- `mech1_figs/ob1_patching_summary/patching_summary.png`（Finding 2 核心图）
- `mech1_figs/ob1_patching_summary/patching_summary.csv`（统计表）

---

## 四、关键脚本说明

### 4.1 核心分析脚本

| 脚本 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `mech1_text_vs_audio_dominance.py` | 单样本 logit lens | 音频 + 情绪标注 | JSON（delta_logit_by_layer, probs, MAR 等） |
| `mech1_runner_ob1_batch.py` | 批量运行 logit lens | 音频目录 | 54 个 JSON |
| `mech1_analyze_ob1.py` | 汇总分析 logit lens | 54 个 JSON | 平均曲线图 + CSV 表 |
| `mech1_find_critical_layer.py` | 找临界层 ℓ\* | Conflict JSON | critical_layers.csv |
| `mech1_patching.py` | 单样本 patching | 音频 + 两种文本情绪 | JSON（逐层 flip rate） + 图 |
| `mech1_summarize_patching.py` | 汇总 patching | 多个 patching JSON | 平均 patching 曲线 + CSV |

---

### 4.2 可视化脚本

| 脚本 | 功能 | 产出 |
|------|------|------|
| `mech1_plot.py` | 通用绘图（旧版） | 按音频分组的图表 |
| `mech1_analyze_ob1.py` | OB1 专用绘图 | delta_logit 曲线、flip-rate 热力图 |
| `mech1_summarize_patching.py` | Patching 绘图 | patching_summary.png |

---

## 五、关键技术细节

### 5.1 Logit Lens 的实现

#### 为什么不直接用 model.generate()？
- `generate()` 是自回归生成，每步只返回最终 token
- 我们需要**每一层的中间表征**，所以直接调用 `model.alm.forward()`

#### 如何获取每层的 logits？
```python
# 1. Forward 一次，拿到所有层的 hidden states
outputs = model.alm(
    inputs_embeds=inputs_embeds,
    output_hidden_states=True,
)

# 2. outputs.hidden_states 是一个 tuple，包含 (layer_0, layer_1, ..., layer_28)
#    但 Kimi-Audio 还有 mimo 分支，所以需要只取 text decoder 部分
num_text_layers = model.alm.config.num_hidden_layers  # 28
text_hidden_states = outputs.hidden_states[:num_text_layers + 1]

# 3. 对每层，用 lm_head 投影到 vocab
for layer_idx, hs in enumerate(text_hidden_states):
    logits_at_layer = model.alm.lm_head(hs)  # [batch, seq, vocab]
    
    # 4. 取最后一个位置（next-token prediction）
    last_pos = logits_at_layer.shape[1] - 1
    logit_T = logits_at_layer[0, last_pos, token_id_T]
    logit_A = logits_at_layer[0, last_pos, token_id_A]
    delta = logit_T - logit_A
```

---

### 5.2 Activation Patching 的实现

#### 核心挑战
- 需要在**中间某一层**替换表征
- 不能改变模型结构，只能在推理时"动态替换"
- → 使用 **PyTorch Forward Hook**

#### Hook 机制
```python
def patch_hook(module, input, output):
    """
    module: 某一层（如 model.alm.model.layers[10]）
    output: 该层的输出 tuple: (hidden_states, ...)
    """
    hidden_states = output[0]  # [batch, seq, hidden]
    
    # 从 donor 拿到同一层的 states
    donor_states = hidden_states_donor[layer_idx]
    
    # 只替换指定位置（文本 token 或音频 token）
    patched_states = hidden_states.clone()
    patched_states[:, patch_positions, :] = donor_states[:, patch_positions, :]
    
    return (patched_states,) + output[1:]

# 注册 hook
hook_handle = model.alm.model.layers[layer_idx].register_forward_hook(patch_hook)

# Forward（会自动触发 hook）
outputs_patched = model.alm(inputs_embeds_source, ...)

# 用完移除 hook
hook_handle.remove()
```

#### 如何区分"文本 token"和"音频 token"？

在 Kimi-Audio 里：
- `text_input_ids` 中，非 `blank_id` 的位置 = 文本 token
- 其余位置 = 音频 token（或 padding）

```python
blank_id = model.prompt_manager.extra_tokens.kimia_text_blank
is_text_pos = (text_input_ids != blank_id)  # bool tensor [seq_len]
is_audio_pos = ~is_text_pos
```

---

### 5.3 为什么 Layer 27 的结果特殊？

#### 观察
- Patch Text 在 layer 27 失效（flip rate = 0）
- Patch Audio 在 layer 27 反而有效（flip rate ≈ 0.67）

#### 可能解释

**解释 1：最后一层的"传播瓶颈"**
- Layer 27 是最后一层，patching 后**没有更多层**来"传播"这个改变
- 决策已经在 layer 26 固化，layer 27 只是"输出准备"
- 所以 patch layer 27 的效果不稳定

**解释 2：Mimo 分支的影响**
- Kimi-Audio 在 layer 21 之后有一个 mimo 分支（用于音频生成）
- Layer 27 可能涉及 mimo 和 text decoder 的交互
- Patching 可能影响了这个交互（需要进一步验证）

**当前处理**：
- 我们把 layer 27 作为"特殊情况"单独讨论
- **核心结论仍基于 layer 5-25 的数据**（那里的模式很清晰）

---

## 六、数据与结果文件位置

### 输入数据
```
/data3/xuzhenyu/dataset from TTS/OB1/renamed/
  ├── zh_cheerful_strong.wav
  ├── zh_cheerful_normal.wav
  ├── ...（共 18 个）
```

### 中间结果
```
/home/xuzhenyu/Kimi-Audio/mech1_outputs/
  ├── ob1_experiment/          # Logit lens 结果（54 个 JSON）
  │   ├── zh_cheerful_strong__audio_only.json
  │   ├── zh_cheerful_strong__conflict.json
  │   └── ...
  └── patching/                # Patching 结果（3-4 个 JSON）
      ├── en_cheerful_strong_sad_to_cheerful.json
      └── ...
```

### 最终图表
```
/home/xuzhenyu/Kimi-Audio/mech1_figs/
  ├── ob1_analysis/
  │   ├── delta_logit__combined.png       # Finding 1 核心图
  │   ├── critical_layers.csv             # 临界层统计
  │   └── ob1_flip_summary.csv            # 翻转率汇总
  └── ob1_patching_summary/
      ├── patching_summary.png            # Finding 2 核心图
      └── patching_summary.csv            # Patching 统计
```

---

## 七、复现指南（完整命令）

### 环境要求
```bash
conda activate kimiaudio
cd /home/xuzhenyu/Kimi-Audio
```

### 完整执行序列
```bash
# 1. 数据准备（如果需要）
python rename_ob1_files.py

# 2. 批量 Logit Lens（约 20-30 分钟）
python -u mech1_runner_ob1_batch.py 2>&1 | tee ob1_batch_run.log

# 3. 分析 Logit Lens
python -u mech1_analyze_ob1.py
python -u mech1_find_critical_layer.py

# 4. 运行 Patching（选 3-4 个样本，每个约 5 分钟）
python -u mech1_patching.py --audio_path "..." --audio_emotion ... --text_source ... --text_target ... --patch_layers "0,5,10,15,20,25,27" --out_json "..."

# 5. 汇总 Patching
python -u mech1_summarize_patching.py

# Done!
```

---

## 八、关键参数说明

### Logit Lens 参数
- `--audio_emotion`：音频的真实情绪（A）
- `--text_emotion`：文本指令要求的情绪（T）
- `--prompt_mode`：
  - `audio_only`：只看音频
  - `text_control`：加文本指令（冲突或一致）
- `--attn_probe_layers`（可选）：重构注意力的层

### Patching 参数
- `--audio_emotion`：音频真实情绪（A）
- `--text_source`：原始文本指令（Source，被 patch 的）
- `--text_target`：目标文本指令（Donor，用于 patch 的）
- `--patch_layers`：要测试的层列表

---

## 九、常见问题（FAQ）

### Q1：为什么 Consistent 条件下 Δlogit ≈ 0？
**A**：因为 T=A，所以 logit(T) − logit(A) = logit(A) − logit(A) ≈ 0（理论上）。实际会有小波动，因为 tokenizer 编码的微小差异。

### Q2：Patching 时 Source 和 Donor 序列长度不同怎么办？
**A**：我们的脚本会自动处理，只 patch 公共长度部分（`min(len_source, len_donor)`）。

### Q3：临界层 ℓ\* 是怎么算出来的？
**A**：用二阶导数（加速度）找曲线的"拐点"。详见 `mech1_find_critical_layer.py` 的 `find_critical_layer()` 函数。

### Q4：为什么 Layer 27 的 patching 结果特殊？
**A**：可能因为是最后一层，patching 后没有更多层来传播改变。我们主要看 layer 5-25 的数据。

### Q5：能在其他 ALLM 上复现吗？
**A**：可以，但需要适配：
- 不同模型的 `hidden_states` 结构可能不同
- 需要找到对应的 `lm_head` 和层数
- Patching hook 的位置可能需要调整

---

## 十、OB1 实验设计的核心要点（总结）

### 控制变量
✅ **语义固定**（同一句话）  
✅ **说话人固定**（同一 TTS 音色）  
✅ **只变情绪韵律**（5 种 × 2 强度）  
✅ **文本指令标准化**（固定模板）

### 对照完整
✅ **Audio-only vs Conflict vs Consistent**（三组对照）  
✅ **多样本统计**（18 样本，能画均值±CI）  
✅ **多层测量**（28 层，找临界点）

### 因果证据
✅ **观察（Logit Lens）** → 发现 late-layer consolidation  
✅ **干预（Patching）** → 证实 text causal dominance  
✅ **补充（Attn Mass）** → 结构性证据

### 可复现性
✅ **完整脚本**（从数据到图表）  
✅ **标准化数据**（明确的 TTS 规格）  
✅ **文档齐全**（方法论 + 总结 + PPT）

---

## 十一、致谢与引用

### 使用的工具
- **模型**：Kimi-Audio (moonshotai/Kimi-Audio-7B-Instruct)
- **框架**：PyTorch, Transformers, kimia_infer
- **可视化**：matplotlib, pandas, numpy

### 参考方法
- **Logit Lens**：nostalgebraist (2020), "Interpreting GPT: the logit lens"
- **Activation Patching**：Meng et al. (2022), "Locating and Editing Factual Associations in GPT"

---

**文档版本**：v1.0  
**最后更新**：2026-01-16  
**联系人**：[你的信息]

---

## 附录：完整文件树

```
Kimi-Audio/
├── mech1_text_vs_audio_dominance.py    # 核心：单样本 logit lens
├── mech1_runner_ob1_batch.py           # 批量运行
├── mech1_analyze_ob1.py                # 汇总分析
├── mech1_find_critical_layer.py        # 找临界层
├── mech1_patching.py                   # 因果实验
├── mech1_summarize_patching.py         # Patching 汇总
├── OB1_SUMMARY.md                      # 结果总结
├── OB1_METHODOLOGY.md                  # 本文档
├── OB1_PPT_OUTLINE.md                  # PPT 大纲
├── OB1_SLIDES.md                       # Marp slides
├── mech1_outputs/
│   ├── ob1_experiment/                 # 54 个 logit lens JSON
│   └── patching/                       # 3-4 个 patching JSON
└── mech1_figs/
    ├── ob1_analysis/                   # Finding 1 图表
    │   ├── delta_logit__combined.png
    │   ├── critical_layers.csv
    │   └── ob1_flip_summary.csv
    └── ob1_patching_summary/           # Finding 2 图表
        ├── patching_summary.png
        └── patching_summary.csv
```

