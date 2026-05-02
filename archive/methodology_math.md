# OpenS2S 情绪白盒对抗攻击数学框架

> 完整数学表述：从原始音频输入到对抗样本生成的全流程

---

## 1️⃣ 问题定义

### 输入
给定一段**原始音频** $\mathbf{x} \in \mathbb{R}^T$（$T$ 为采样点数）

### 目标
找到一个**微小扰动** $\boldsymbol{\delta} \in \mathbb{R}^T$，使得：

$$\mathbf{x}_{\text{adv}} = \text{clip}(\mathbf{x} + \boldsymbol{\delta}, -1, 1)$$

满足：
1. **情绪攻击成功**：模型输出目标情绪 token（如 "happy"）
2. **语义保持**：转写文本与原始一致（低 WER）
3. **扰动受限**：$\|\boldsymbol{\delta}\|_\infty \leq \epsilon$（如 $\epsilon = 0.008$）

---

## 2️⃣ 前向传播链路（完全可微）

### Step 1: 音频 → 声学特征
$$\mathbf{M}_{\text{adv}} = \log\text{-}\text{Mel}(\mathbf{x}_{\text{adv}}) \in \mathbb{R}^{n_{\text{mel}} \times n_{\text{frames}}}$$

**代码位置**：`opens2s_io.py:101-120` (`_torch_log_mel()`)

**关键**：使用 PyTorch 的 `torchaudio.transforms.MelSpectrogram` 保证可微

$$
\begin{aligned}
\text{STFT}: \quad & S(\omega, t) = \sum_{\tau} \mathbf{x}_{\text{adv}}[\tau] \cdot w[\tau - t] \cdot e^{-j\omega\tau} \\
\text{Mel-filter}: \quad & \mathbf{M}[m, t] = \sum_{\omega} |S(\omega, t)|^2 \cdot H_m(\omega) \\
\text{Log-scale}: \quad & \mathbf{M}_{\text{adv}}[m, t] = \log(\mathbf{M}[m, t] + 10^{-6})
\end{aligned}
$$

### Step 2: 特征 → OpenS2S 模型
$$\mathbf{h} = f_{\text{encoder}}(\mathbf{M}_{\text{adv}}) \in \mathbb{R}^{d_{\text{hidden}}}$$

**代码位置**：`opens2s_io.py:199-212` (`forward_logits()`)

OpenS2S 包含：
- **Audio Encoder**：将 log-Mel 编码为音频表征
- **LLM Backbone**：与文本 prompt token 融合处理
- **输出层**：生成词表上的 logits

### Step 3: 模型 → 输出 Logits
给定 prompt $p$（如"What is the emotion? Answer: happy/sad/angry/neutral."）：

$$\mathbf{z} = f_{\text{OpenS2S}}(\mathbf{M}_{\text{adv}}, p) \in \mathbb{R}^{L \times V}$$

其中：
- $L$：序列长度
- $V$：词表大小（~32000）
- $\mathbf{z}[i, :]$：第 $i$ 个 token 位置的 logits

---

## 3️⃣ 损失函数设计

### 总损失（两阶段）

$$\mathcal{L}_{\text{total}} = \lambda_{\text{emo}} \cdot \mathcal{L}_{\text{emo}} + \lambda_{\text{asr}} \cdot \mathcal{L}_{\text{asr}} + \lambda_{\text{per}} \cdot \mathcal{L}_{\text{per}}$$

**代码位置**：`attack_core.py:219`

**两阶段权重调度**（`attack_core.py:143-147`）：

| 阶段 | 步数 | $\lambda_{\text{emo}}$ | $\lambda_{\text{asr}}$ | $\lambda_{\text{per}}$ | 策略 |
|------|------|------------------------|------------------------|------------------------|------|
| Stage A | 0-19 | 1.0 | $10^{-4}$ | 0.0 | **优先攻击情绪** |
| Stage B | 20-59 | 1.0 | $10^{-2}$ | $10^{-5}$ | **增强语义/感知约束** |

---

### 3.1 情绪损失 $\mathcal{L}_{\text{emo}}$（核心）

**代码位置**：`attack_core.py:56-73`

$$\mathcal{L}_{\text{emo}} = \frac{1}{|\mathcal{P}_{\text{emo}}|} \sum_{p \in \mathcal{P}_{\text{emo}}} \mathcal{L}_{\text{CE}}(\mathbf{z}_p, \mathbf{y}_{\text{target}})$$

其中：
- $\mathcal{P}_{\text{emo}}$：情绪 prompt 集合（3 个等价 prompts，ensemble）
- $\mathbf{y}_{\text{target}}$：目标情绪 token IDs（如 `tokenizer.encode("happy")`）
- $\mathcal{L}_{\text{CE}}$：**Token-level 交叉熵损失**

**关键细节**（`attack_core.py:21-40`）：

1. **构造监督标签**：
   ```python
   input_ids = [prompt_tokens, <audio>, text_tokens]  # 输入
   labels = [-100, -100, ..., -100, target_token_1, target_token_2]  # 标签
   ```
   - 只监督目标 token 位置（assistant 输出位置）
   - 其他位置用 `IGNORE_INDEX = -100`（标准 HuggingFace 约定）

2. **Shifted Causal LM Loss**（`attack_core.py:43-53`）：
   $$\mathcal{L}_{\text{CE}} = -\frac{1}{|\mathcal{T}|} \sum_{t \in \mathcal{T}} \log P(y_t | \mathbf{z}_{t-1})$$

   其中 $\mathcal{T}$ 是目标 token 位置集合。

**数学本质**：
- **不使用外部分类器**（无 surrogate）
- **直接优化 OpenS2S 输出分布**，使目标 token 的 logit 最大化

---

### 3.2 语义保持损失 $\mathcal{L}_{\text{asr}}$（Self-Consistency）

**代码位置**：`attack_core.py:76-90`

$$\mathcal{L}_{\text{asr}} = \mathcal{L}_{\text{CE}}(\mathbf{z}_{\text{asr}}, \mathbf{y}_{\text{ref}})$$

其中：
- $\mathbf{z}_{\text{asr}}$：用转写 prompt 得到的 logits
- $\mathbf{y}_{\text{ref}}$：**OpenS2S 自身对原始音频 $\mathbf{x}$ 的转写结果**

**关键步骤**（`run_attack.py:141-154`）：

1. **预先获取基准转写**：
   $$\mathbf{y}_{\text{ref}} = \arg\max_{\mathbf{y}} P(\mathbf{y} | \mathbf{x}, p_{\text{asr}})$$

   ```python
   asr_text_clean = decode_text(model, tokenizer, x, ...)
   asr_target_token_ids = tokenizer.encode(asr_text_clean)
   ```

2. **Teacher Forcing**：
   $$\mathcal{L}_{\text{asr}} = -\sum_{t=1}^{|\mathbf{y}_{\text{ref}}|} \log P(y_{\text{ref}, t} | \mathbf{x}_{\text{adv}}, p_{\text{asr}}, \mathbf{y}_{\text{ref}, <t})$$

**数学意义**：
- 确保 $\mathbf{x}_{\text{adv}}$ 的转写与 $\mathbf{x}$ 一致
- **使用同一模型**（OpenS2S），而非外部 ASR
- 体现 "Self-Consistency" 原则

---

### 3.3 感知损失 $\mathcal{L}_{\text{per}}$（频域约束）

**代码位置**：`attack_core.py:106-113`

$$\mathcal{L}_{\text{per}} = \frac{1}{|\mathcal{R}|} \sum_{(n, h, w) \in \mathcal{R}} \left\| |\text{STFT}_{n,h,w}(\mathbf{x}_{\text{adv}})| - |\text{STFT}_{n,h,w}(\mathbf{x})| \right\|_1$$

其中 $\mathcal{R}$ 是多分辨率 STFT 参数集：

| FFT Size ($n$) | Hop Size ($h$) | Window Length ($w$) |
|----------------|----------------|---------------------|
| 256 | 64 | 256 |
| 512 | 128 | 512 |
| 1024 | 256 | 1024 |

**数学形式**：
$$|\text{STFT}_{n,h,w}(\mathbf{x})|[f, t] = \left| \sum_{\tau=0}^{w-1} \mathbf{x}[th + \tau] \cdot \text{Hann}[\tau] \cdot e^{-j2\pi f\tau / n} \right|$$

**作用**：
- 约束频谱差异（人耳感知更依赖频域）
- 避免纯 $L_2$ 范数（易导致高频噪声）
- **全程可微**（PyTorch `torch.stft` + `.abs()`）

---

## 4️⃣ EoT（Expectation over Transformations）

**代码位置**：`attack_core.py:123-141`, `attack_core.py:187-191`

### 数学形式

$$\mathcal{L}_{\text{total}} = \mathbb{E}_{T \sim \mathcal{T}} \left[ \mathcal{L}\left( f_{\text{OpenS2S}}(T(\mathbf{x}_{\text{adv}})), \mathbf{y}_{\text{target}} \right) \right]$$

**实践近似**（Monte Carlo）：
$$\mathcal{L}_{\text{total}} \approx \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}\left( f_{\text{OpenS2S}}(T_i(\mathbf{x}_{\text{adv}})), \mathbf{y}_{\text{target}} \right)$$

### 可微变换 $T$

**代码位置**：`attack_core.py:130-140` (`apply_eot()`)

1. **时域平移**（Time Shift）：
   $$T_1(\mathbf{x})[t] = \mathbf{x}[t + s], \quad s \sim \text{Uniform}(-160, 160)$$

   **实现**：`torch.roll(waveform, shifts=s)`

2. **增益调整**（Gain）：
   $$T_2(\mathbf{x}) = g \cdot \mathbf{x}, \quad g \sim \text{Uniform}(0.8, 1.2)$$

3. **可选高斯噪声**：
   $$T_3(\mathbf{x}) = \mathbf{x} + \sigma \cdot \mathcal{N}(0, I), \quad \sigma = 0.0 \text{ (默认关闭)}$$

**关键**：所有变换都是**可微的 PyTorch 操作**，不破坏梯度链。

---

## 5️⃣ 优化算法

**代码位置**：`attack_core.py:150-261`

### 伪代码

```
输入：x, model, tokenizer, target_emotion, asr_ref, ε, steps
输出：x_adv

1. δ ← 0 ∈ ℝ^T, requires_grad = True
2. optimizer ← Adam([δ], lr=0.003)

3. FOR step = 1 TO steps:
4.     optimizer.zero_grad()
5.
6.     // 两阶段权重调度
7.     IF step < 20:
8.         λ_emo, λ_asr, λ_per = 1.0, 1e-4, 0.0
9.     ELSE:
10.        λ_emo, λ_asr, λ_per = 1.0, 1e-2, 1e-5
11.
12.    // EoT 采样
13.    L_total = 0
14.    FOR i = 1 TO eot_samples:
15.        T_i ← sample_random_transform()
16.        x_adv ← clip(x + δ, -1, 1)
17.        x_adv_t ← T_i(x_adv)
18.        x_t ← T_i(x)
19.
20.        L_emo ← emotion_loss(x_adv_t, target_emotion)
21.        L_asr ← asr_loss(x_adv_t, asr_ref)
22.        L_per ← perceptual_loss(x_adv_t, x_t)
23.
24.        L_total += λ_emo * L_emo + λ_asr * L_asr + λ_per * L_per
25.    END FOR
26.
27.    L_total ← L_total / eot_samples
28.    L_total.backward()
29.
30.    // 梯度检查
31.    IF ‖∇_δ L_total‖_2 < 1e-8 for 3 consecutive steps:
32.        RAISE ERROR "Gradient chain broken"
33.
34.    optimizer.step()  // δ ← δ - lr * ∇_δ L_total
35.
36.    // L∞ 投影
37.    δ ← clip(δ, -ε, ε)
38. END FOR

39. RETURN clip(x + δ, -1, 1)
```

---

## 6️⃣ 关键数学性质

### ✅ 完整梯度链

$$\frac{\partial \mathcal{L}_{\text{total}}}{\partial \boldsymbol{\delta}} = \frac{\partial \mathcal{L}_{\text{total}}}{\partial \mathbf{z}} \cdot \frac{\partial \mathbf{z}}{\partial \mathbf{h}} \cdot \frac{\partial \mathbf{h}}{\partial \mathbf{M}_{\text{adv}}} \cdot \frac{\partial \mathbf{M}_{\text{adv}}}{\partial \mathbf{x}_{\text{adv}}} \cdot \frac{\partial \mathbf{x}_{\text{adv}}}{\partial \boldsymbol{\delta}}$$

**每一项都是可微的**：
- $\frac{\partial \mathbf{x}_{\text{adv}}}{\partial \boldsymbol{\delta}} = I$ （线性）
- $\frac{\partial \mathbf{M}_{\text{adv}}}{\partial \mathbf{x}_{\text{adv}}}$：STFT + log 的雅可比矩阵（PyTorch 自动微分）
- $\frac{\partial \mathbf{h}}{\partial \mathbf{M}_{\text{adv}}}$：神经网络反向传播
- $\frac{\partial \mathcal{L}_{\text{total}}}{\partial \mathbf{z}}$：CE loss 梯度

### ✅ 约束投影

在每次梯度更新后：
$$\boldsymbol{\delta}^{(t+1)} = \text{Proj}_{\mathcal{B}_\infty(\epsilon)} \left( \boldsymbol{\delta}^{(t)} - \alpha \nabla_{\boldsymbol{\delta}} \mathcal{L}_{\text{total}} \right)$$

其中：
$$\mathcal{B}_\infty(\epsilon) = \{ \boldsymbol{\delta} : \|\boldsymbol{\delta}\|_\infty \leq \epsilon \}$$

$$\text{Proj}_{\mathcal{B}_\infty(\epsilon)}(\boldsymbol{\delta}) = \text{clip}(\boldsymbol{\delta}, -\epsilon, \epsilon)$$

---

## 7️⃣ 成功判定

### 情绪攻击成功

$$\text{Success}_{\text{emo}} = \mathbb{1}\left[ \arg\max_y P(y | \mathbf{x}_{\text{adv}}, p_{\text{emo}}) = y_{\text{target}} \right], \quad \forall p_{\text{emo}} \in \mathcal{P}_{\text{emo}}$$

**代码位置**：`run_attack.py:197`
```python
success_emo = all(p == cfg.target_emotion for p in emo_pred_adv)
```

### 语义保持

$$\text{WER}(\mathbf{x}_{\text{adv}}, \mathbf{x}) = \frac{\text{edit\_distance}(S_{\text{adv}}, S_{\text{ref}})}{|S_{\text{ref}}|}$$

其中：
- $S_{\text{ref}} = \arg\max P(\mathbf{y} | \mathbf{x}, p_{\text{asr}})$
- $S_{\text{adv}} = \arg\max P(\mathbf{y} | \mathbf{x}_{\text{adv}}, p_{\text{asr}})$

### 联合成功

$$\text{Success}_{\text{joint}} = \text{Success}_{\text{emo}} \land (\text{WER} \leq \tau)$$

典型阈值：$\tau \in \{0.0, 0.05\}$

---

## 8️⃣ 方法论核心创新点

| 创新点 | 数学体现 | 代码位置 |
|--------|----------|----------|
| **Token-level 优化** | 直接最小化 $\mathcal{L}_{\text{CE}}(\mathbf{z}, y_{\text{target}})$，无 surrogate | `attack_core.py:56-73` |
| **Self-Consistency** | $\mathbf{y}_{\text{ref}}$ 来自同一 OpenS2S 模型 | `run_attack.py:141-154` |
| **Prompt Ensemble** | $\frac{1}{|\mathcal{P}_{\text{emo}}|} \sum_{p} \mathcal{L}_p$ | `attack_core.py:68` |
| **两阶段策略** | 动态权重 $\lambda_{\text{asr}}^{(t)}, \lambda_{\text{per}}^{(t)}$ | `attack_core.py:143-147` |
| **完整梯度链** | 全程 PyTorch，无 `.detach()` 断链 | 所有文件 |
| **多分辨率 STFT** | $\sum_{r \in \mathcal{R}} \|\|\text{STFT}_r(\mathbf{x}_{\text{adv}})\| - \|\text{STFT}_r(\mathbf{x})\|\|_1$ | `attack_core.py:106-113` |
| **EoT 鲁棒性** | $\mathbb{E}_{T \sim \mathcal{T}} \mathcal{L}(f(T(\mathbf{x}_{\text{adv}})))$ | `attack_core.py:187-221` |

---

## 9️⃣ 参数配置总结

### 核心参数（`config.py`）

| 参数 | 默认值 | 数学符号 | 作用 |
|------|--------|----------|------|
| `epsilon` | 0.008 | $\epsilon$ | L∞ 扰动上界 |
| `total_steps` | 60 | $T$ | 总优化步数 |
| `stage_a_steps` | 20 | $T_a$ | 阶段 A 步数 |
| `lr` | 0.003 | $\alpha$ | Adam 学习率 |
| `lambda_emo` | 1.0 | $\lambda_{\text{emo}}$ | 情绪损失权重 |
| `lambda_asr_stage_a` | 1e-4 | $\lambda_{\text{asr}}^{(a)}$ | 阶段 A ASR 权重 |
| `lambda_asr_stage_b` | 1e-2 | $\lambda_{\text{asr}}^{(b)}$ | 阶段 B ASR 权重 |
| `lambda_per_stage_a` | 0.0 | $\lambda_{\text{per}}^{(a)}$ | 阶段 A 感知权重 |
| `lambda_per_stage_b` | 1e-5 | $\lambda_{\text{per}}^{(b)}$ | 阶段 B 感知权重 |
| `eot_samples` | 1 | $N$ | EoT 采样次数 |
| `temperature` | 0.0 | - | 解码温度（greedy） |

---

## 🔟 完整数据流图

```
原始音频 x ∈ ℝ^T
    ↓ [初始化]
扰动 δ ← 0, requires_grad=True
    ↓
┌─────────────── 优化循环 (60 步) ───────────────┐
│                                                 │
│  x_adv = clip(x + δ, -1, 1)                   │
│      ↓                                          │
│  [EoT 采样] T_i ~ Uniform(transforms)          │
│      ↓                                          │
│  x_adv_t = T_i(x_adv)  [可微变换]             │
│  x_t = T_i(x)                                  │
│      ↓                                          │
│  M_adv = log-Mel(x_adv_t)  [可微 STFT]        │
│      ↓                                          │
│  z = OpenS2S(M_adv, prompts)  [神经网络]       │
│      ↓                                          │
│  L_emo = CE(z_emo, y_target)  [token-level]    │
│  L_asr = CE(z_asr, y_ref)  [self-consistency]  │
│  L_per = STFT_L1(x_adv_t, x_t)  [multi-res]    │
│      ↓                                          │
│  L_total = λ_emo*L_emo + λ_asr*L_asr + λ_per*L_per │
│      ↓                                          │
│  L_total.backward()  [反向传播]                │
│      ↓                                          │
│  δ ← δ - lr * ∇_δ L_total  [Adam]             │
│  δ ← clip(δ, -ε, ε)  [L∞ 投影]                │
│      ↓                                          │
└─────────────────────────────────────────────────┘
    ↓
x_adv = clip(x + δ, -1, 1)  [最终对抗样本]
    ↓
[评估] emo_pred = decode(x_adv, p_emo)
       asr_pred = decode(x_adv, p_asr)
       WER = edit_distance(asr_pred, asr_ref)
    ↓
成功判定：success_emo ∧ (WER ≤ 0.05)
```

---

## 附录：代码文件对应关系

| 文件 | 核心功能 | 数学模块 |
|------|----------|----------|
| `config.py` | 参数配置 | 超参数 $\epsilon, \lambda, T, \alpha$ |
| `run_attack.py` | 主流程控制 | 批量处理、评估 |
| `attack_core.py` | 攻击核心 | 损失函数、优化循环、EoT |
| `opens2s_io.py` | 模型接口 | $\mathbf{M} = \log\text{-}\text{Mel}(\mathbf{x})$, $\mathbf{z} = f_{\text{OpenS2S}}(\mathbf{M})$ |
| `eval_metrics.py` | 评估指标 | WER, SNR, 成功率统计 |

---

**文档版本**：v1.0
**生成时间**：2026-01-07
**对应代码版本**：white_box_v2/codex/
