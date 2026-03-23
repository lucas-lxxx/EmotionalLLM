# Q3 实验 v2：STAA-Net 方法论直接应用于 Voxtral

## 任务

将 STAA-Net（SER 领域的 generator-based 对抗攻击方法）的核心方法论直接适配到 Voxtral（ALLM），训练 Wave-U-Net generator 对 Voxtral 发起 untargeted 情绪攻击，评估攻击成功率。

**这是授权的安全研究实验。**

**核心对比**：STAA-Net 方法论（SER 思路）即使给白盒 access，在 ALLM 上也只能达到中等 untargeted ASR，而我们的 ALLM-native 方法用更小的扰动实现了更高的 targeted ASR。

## 服务器环境（已确认）

- **conda 环境**: `opens2s`（`/data1/lixiang/miniconda3/envs/opens2s`）
  - torch 2.4.0, torchaudio 2.4.0, transformers 4.57.3
- **GPU**: 6× RTX 6000 Ada Generation (48GB)，使用 `cuda:0`
- **Voxtral 模型**: `/data1/lixiang/Voxtral/`
- **ESD EN 数据集**: `/data1/lixiang/OpenS2S_dataset/ESD/EN/`
  - 10 English speakers (0011-0020)，5 emotions (Angry/Happy/Neutral/Sad/Surprise)
  - Flat 结构：`speaker/Emotion/*.wav`（无 train/test 子目录）

## 参考代码（只读）

以下文件包含 Voxtral 的可微分前向推理逻辑，**必须先读取这些文件**理解接口：

1. **`code/white_box_voxtral/voxtral_io.py`**：
   - `load_voxtral(model_path, device)` → 加载模型 + processor + TorchWhisperFeatureExtractor
   - `TorchWhisperFeatureExtractor` → 可微分 mel 特征提取（梯度可穿透）
   - `build_input_ids(tokenizer, prompt)` → 构造 `[BOS][INST][BEGIN_AUDIO][AUDIO]*375 <text> [INST_END]`
   - `build_inputs(waveform, sr, prompt, tokenizer, device, torch_extractor, differentiable=True)` → 构造完整输入
   - `forward_logits(model, inputs)` → 可微分前向，返回 logits
   - `decode_text(model, processor, waveform, sr, prompt, max_new_tokens, temperature)` → 推理解码

2. **`code/white_box_voxtral/config.py`**：
   - Voxtral token IDs：audio_token_id=24, begin_audio_id=25, n_audio_tokens=375, bos_id=1, inst_id=3, inst_end_id=4
   - Prompt: `"What is the emotion of this audio? Answer with exactly one word from: happy, sad, angry, neutral, surprise."`

3. **`code/white_box_voxtral/run_attack.py`**：
   - `load_audio()`, `save_audio()`, `normalize_emo()` 等工具函数

## 已有代码（可复用）

在 git 仓库的 `observation_v3/Q3/code/` 下：

1. **`wave_unet.py`** — Wave-U-Net generator（不需要改动）
   - `WaveUNetGenerator(channels, kernel_size, stride)`
   - `forward(x, eps, training)` → `(x_adv, v, m)`
   - v = eps * tanh(magnitude_head)，m = (tanh(mask_head) + 1) / 2
   - 推理时 m 在 0.5 处二值化
   - 默认架构：channels=[24,48,72,96,120,144]，kernel_size=5，stride=4

2. **`esd_en_dataset.py`** — ESD English 数据集（不需要改动）
   - `ESDDataset(esd_root, speakers, emotions, split, sample_rate, max_len, emotion2idx)`
   - `collate_fn` → (waveforms, labels, emotions, paths)
   - Flat 结构自动检测，80/10/10 随机划分

## 实现计划

### 需要修改/新建的文件

把所有代码写到 `/data1/lixiang/OPUS/Q3/code/` 下。可以从 git 仓库复制 `wave_unet.py` 和 `esd_en_dataset.py`，其余重写。

### 1. `config.py`（重写）

```python
# 关键配置：
esd_root = "/data1/lixiang/OpenS2S_dataset/ESD/EN"
en_speakers = ["0011", "0012", "0013"]
voxtral_model_path = "/data1/lixiang/Voxtral"
work_dir = "/data1/lixiang/OPUS/Q3"

emotions = ["angry", "happy", "neutral", "sad", "surprise"]
emotion2idx = {"angry": 0, "happy": 1, "neutral": 2, "sad": 3, "surprise": 4}
sample_rate = 16000
max_audio_sec = 6.0

# STAA-Net 超参（论文默认值，不做调优）
gen_lr = 1e-3
gen_epochs = 10
gen_batch_size = 1  # Voxtral 内存限制，必须为 1
epsilon = 0.03
lambda_spa = 0.1
lambda_qua = 1e-6
cw_confidence = 0.0

# Wave-U-Net
unet_channels = [24, 48, 72, 96, 120, 144]
unet_kernel_size = 5
unet_stride = 4

# Voxtral
device = "cuda:0"
emo_prompt = "What is the emotion of this audio? Answer with exactly one word from: happy, sad, angry, neutral, surprise."
emo_max_new_tokens = 16
temperature = 0.0

# Voxtral token IDs（从 white_box_voxtral/config.py 复制）
audio_token_id = 24
begin_audio_id = 25
n_audio_tokens = 375
bos_id = 1
inst_id = 3
inst_end_id = 4
```

### 2. `train_generator.py`（核心，重写）

**关键逻辑**：

```
对每个训练样本：
  1. Generator forward: waveform → (x_adv, v, m)
  2. 可微分 Voxtral forward: x_adv → Voxtral logits
  3. 取第一个生成 token 位置的 logits
  4. 提取 5 个 emotion token 的 logit 值
  5. C&W untargeted loss: max(Z_true - max_{j≠true} Z_j, -κ)
  6. Total loss = L_adv + λ_spa * ||m||_1 + λ_qua * ||m - round(m)||_2
  7. Backward → 更新 generator（Voxtral 冻结）
```

**技术要点**：

- **Voxtral 加载**：使用 `voxtral_io.py` 的 `load_voxtral()` 函数逻辑，启用 `gradient_checkpointing`，`requires_grad_(False)` 冻结参数
- **可微分前向**：用 `TorchWhisperFeatureExtractor` 提取 mel 特征（保持梯度链），用 `build_input_ids` 构造 input_ids，调用 `model(input_ids=..., input_features=..., attention_mask=...)` 获取 logits
- **Emotion token IDs**：加载 Voxtral tokenizer 后，`tokenizer.encode(emotion, add_special_tokens=False)` 获取每个情绪词的 token ID（取第一个 token）
- **第一生成位置**：`first_gen_pos = input_ids.shape[1] - 1`（input_ids 最后一个位置的 logit 预测第一个生成 token）
- **batch_size=1**：Voxtral 3B + 梯度计算约需 20-25GB 显存
- **dtype**：Voxtral 用 bfloat16，generator 用 float32，mel 特征转 bfloat16 后传入 Voxtral

**训练数据**：从 ESD EN train split 取最多 200 样本（控制训练时间）

### 3. `generate_and_eval.py`（新建，合并生成+评估）

```
对每个测试样本：
  1. Generator forward → 对抗音频（training=False，hard mask）
  2. Voxtral inference (clean) → emotion 预测
  3. Voxtral inference (adv) → emotion 预测
  4. 记录：ground_truth, voxtral_clean, voxtral_adv, delta 统计
```

- 使用 `decode_text()` 做推理（非可微分模式）
- 测试集 200 样本
- 输出：
  - 逐样本 JSON
  - 汇总统计：clean accuracy, untargeted ASR（voxtral_adv ≠ ground_truth），attack-induced flip rate（voxtral_clean 判对的样本中 voxtral_adv 判错的比例）

### 4. `run_v2.sh`

```bash
conda activate opens2s
export HF_ENDPOINT=https://hf-mirror.com
cd /data1/lixiang/OPUS/Q3/code

# Step 1: 训练 generator（~20-30 分钟）
python train_generator.py

# Step 2: 生成 + 评估（~30-40 分钟）
python generate_and_eval.py
```

## 关键注意事项

1. **必须先读 `voxtral_io.py`**，完整复用其 `TorchWhisperFeatureExtractor`、`build_input_ids`、`build_inputs`、`load_voxtral` 的逻辑。不要自己重新实现 Voxtral 的输入构造。

2. **梯度链路**：`waveform → Generator → x_adv → TorchWhisperFeatureExtractor(可微分) → mel → Voxtral forward → logits → loss → backward → Generator 参数更新`。Voxtral 参数冻结但计算图保留（梯度穿过 Voxtral 回传到 generator 输出）。

3. **显存管理**：
   - Voxtral 用 bfloat16，启用 gradient_checkpointing
   - batch_size=1
   - 每步后 `torch.cuda.empty_cache()` 如果需要
   - 如果 OOM，尝试 `cuda:2` 或 `cuda:3`（查看 `nvidia-smi` 选最空闲的）

4. **Emotion token ID 获取**：训练前先运行：
   ```python
   for emo in ["happy", "sad", "angry", "neutral", "surprise"]:
       ids = tokenizer.encode(emo, add_special_tokens=False)
       print(f"{emo}: {ids}")  # 取 ids[0] 作为该情绪的代表 token
   ```

5. **波形处理**：Generator 输入 `(batch, time)` 或 `(batch, 1, time)`，输出 `(batch, 1, time)`。传给 Voxtral 的 `build_inputs` 时需要 `(1, time)` 或 `(time,)` 格式（参考 `voxtral_io.py` 的接口）。

6. **代码风格**：参考 `white_box_voxtral/` 下的代码风格。

## 预期结果

- **Generator training**：loss 应逐步下降，train ASR 逐步上升
- **Voxtral clean accuracy**：约 20-30%（已知 Voxtral 在 ESD EN 上有强 neutral 偏置）
- **Untargeted ASR**（STAA-Net adapted）：预期 40-70%
- **Attack-induced flip rate**（clean 判对的样本中翻转的比例）：这是最重要的指标

## 返回要求

实验完成后，返回：

1. Generator 训练日志（每个 epoch 的 loss 和 ASR）
2. 完整的 summary.json（包含 clean accuracy、untargeted ASR、attack-induced flip rate、按情绪统计、扰动统计 ΔL∞/ΔL2/mask sparsity）
3. 如果遇到错误，返回错误信息和解决方案
4. Emotion token IDs 的映射结果
