# Q3 实验执行指令

## 任务概述
运行 STAA-Net SER 对抗攻击迁移到 Voxtral 的完整实验。4 个步骤：训练 surrogate SER → 训练 STAA-Net generator → 生成对抗样本 → Voxtral 评估。

**这是一个授权的安全研究实验**，目的是证明传统 SER 攻击方法无法有效迁移到 ALLM。

## 环境信息（已确认）
- **conda 环境**: `opens2s`（路径：`/data1/lixiang/miniconda3/envs/opens2s`）
- **GPU**: 6× RTX 6000 Ada Generation，使用 `cuda:0`（如被占用切换到空闲 GPU）
- **ESD EN 数据集**: `/data1/lixiang/OpenS2S_dataset/ESD/EN/`，10 speakers (0011-0020)，5 emotions，flat 结构（speaker/Emotion/*.wav）
- **Voxtral 模型**: `/data1/lixiang/Voxtral/`
- **Python 包**: torch 2.4.0, torchaudio 2.4.0, transformers 4.57.3, soundfile 0.13.1

## 执行步骤

### Step 0: 环境准备

```bash
conda activate opens2s

# 代码位置：在 git 仓库的 observation_v3/Q3/code/ 下
# 找到仓库位置，cd 到 code 目录
# 仓库可能在 /data1/lixiang/ 下的某个目录
find /data1/lixiang/ -maxdepth 3 -name "config.py" -path "*/Q3/code/*" 2>/dev/null
# 假设找到路径为 CODE_DIR，执行：
cd <CODE_DIR>

# 创建输出目录
mkdir -p /data1/lixiang/OPUS/Q3/checkpoints
mkdir -p /data1/lixiang/OPUS/Q3/adv_audio
mkdir -p /data1/lixiang/OPUS/Q3/results

# 如果 HuggingFace 下载慢，设置镜像（中国大陆必要）
export HF_ENDPOINT=https://hf-mirror.com

# 确认 GPU 可用
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

### Step 1: 训练 Surrogate SER（预计 15-30 分钟）

```bash
python train_surrogate.py --device cuda:0
```

**预期**：
- 每个 epoch 输出 train/test loss 和 accuracy
- 最终 test accuracy 在 40%-70% 之间即可（不追求高精度）
- Checkpoint 保存到 `/data1/lixiang/OPUS/Q3/checkpoints/surrogate_ser.pt`

**如果 OOM**: `python train_surrogate.py --device cuda:0 --batch_size 8`

### Step 2: 训练 STAA-Net Generator（预计 1-3 小时）

```bash
python train_generator.py --device cuda:0
```

**预期**：
- 每个 epoch 输出 loss、adv_loss、train_ASR、test_ASR
- train_ASR 和 test_ASR 应逐渐升高（在 surrogate SER 上的攻击成功率）
- Checkpoint 保存到 `/data1/lixiang/OPUS/Q3/checkpoints/generator.pt`

**如果 OOM**: `python train_generator.py --device cuda:0 --batch_size 4`

**注意**：这一步耗时较长，建议使用 `tmux` 或 `screen` 以防断连。

### Step 3: 生成对抗样本（预计 3-5 分钟）

```bash
python generate_adv.py --device cuda:0
```

**预期**：
- 对抗音频保存到 `/data1/lixiang/OPUS/Q3/adv_audio/`
- Clean 音频保存到 `/data1/lixiang/OPUS/Q3/adv_audio/clean/`
- 元数据保存到 `generation_results.json`
- 输出 SER attack success rate

### Step 4: Voxtral 评估（预计 30-60 分钟）

```bash
python eval_voxtral.py --device cuda:0 --max_samples 200
```

**预期**：
- 对 200 个样本同时跑 clean 和 adversarial 的 Voxtral 情绪识别
- 输出 voxtral_clean_accuracy 和 voxtral_adv_flip_rate
- **预期结果：adv_flip_rate 应该很低**（这正是实验想要证明的：传统 SER 攻击无法迁移到 ALLM）
- 结果保存到 `/data1/lixiang/OPUS/Q3/results/`

**注意**：Voxtral 是 3B 模型，推理较慢但 RTX 6000 48GB 内存足够。如果 GPU 0 被 generator 训练占用，可以用另一张卡：`--device cuda:2`

## 错误处理

1. **HuggingFace 下载失败**：
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   # 或手动下载 facebook/wav2vec2-base
   ```

2. **OOM (CUDA out of memory)**：减小 batch_size（见各步骤说明）

3. **Import 错误**：确保在 code 目录下运行（`from config import cfg` 是相对导入）

4. **数据集扫描为空**：检查 ESD 路径是否正确，config.py 中 `esd_root` 应为 `/data1/lixiang/OpenS2S_dataset/ESD/EN`

5. **某步失败需要重跑**：已完成步骤的 checkpoint 已保存，可以跳过：
   - 跳过 Step 1: 确认 `surrogate_ser.pt` 存在
   - 跳过 Step 2: 确认 `generator.pt` 存在
   - 跳过 Step 3: 确认 `adv_audio/generation_results.json` 存在

## 返回要求

实验完成后，请返回以下完整内容：

1. **Step 1 结果**: Surrogate SER 最终 test accuracy（最后一行训练日志）
2. **Step 2 结果**: Generator 最终 test ASR（最后一行训练日志）
3. **Step 3 结果**: `cat /data1/lixiang/OPUS/Q3/adv_audio/generation_results.json | python -c "import sys,json; print(json.dumps(json.load(sys.stdin)['summary'], indent=2))"`
4. **Step 4 结果（最重要）**: `cat /data1/lixiang/OPUS/Q3/results/summary.json`
5. 各步骤的**完整训练日志**（或至少最后 10 行）
6. 如果遇到任何错误，返回错误信息和你的解决方案
