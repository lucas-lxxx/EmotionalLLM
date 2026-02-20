# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个针对 OpenS2S 多模态语音模型的白盒对抗性攻击实验框架。主要目标是生成对抗性音频样本,使模型输出特定的情感标签(如 "happy"),同时保持语音转录的一致性和音频的感知质量。

## 核心架构

### 主要模块

1. **config.py** - 集中式配置管理
   - 所有超参数、路径、模型设置都在 `Config` 类中定义
   - 使用 `cfg = Config()` 全局实例
   - 关键配置:
     - `opens2s_root`: OpenS2S 代码库路径
     - `model_path`: 预训练模型路径
     - `epsilon`: 对抗扰动的 L∞ 约束
     - `total_steps`: 攻击迭代步数
     - `stage_a_steps`: 第一阶段步数(两阶段攻击策略)

2. **run_attack.py** - 主入口脚本
   - 加载音频样本列表
   - 对每个样本执行攻击
   - 评估攻击前后的情感分类和转录质量
   - 保存对抗样本和结果 JSON
   - 支持分片处理: `--shard_id` 和 `--num_shards`

3. **attack_core.py** - 核心攻击逻辑
   - `attack_one_sample()`: 主攻击函数,使用 PGD 优化
   - `loss_emo()`: 情感目标损失(交叉熵)
   - `loss_asr()`: ASR 自一致性损失
   - `loss_per()`: 感知损失(多分辨率 STFT)
   - 两阶段权重调度: Stage A 侧重情感攻击,Stage B 增加感知约束
   - EoT (Expectation over Transformation): 时间偏移、增益变化增强鲁棒性

4. **opens2s_io.py** - OpenS2S 模型接口
   - `load_opens2s()`: 加载模型、tokenizer 和特征提取器
   - `TorchWhisperFeatureExtractor`: 可微分的 Whisper 特征提取(用于梯度反向传播)
   - `build_inputs()`: 构建模型输入(input_ids + speech_values + masks)
   - `decode_text()`: 推理解码(贪婪或采样)
   - 关键: 使用 `torch_extractor` 进行可微分特征提取,使用 `audio_extractor` 进行推理

5. **eval_metrics.py** - 评估指标
   - `compute_wer()`: 词错误率(使用 jiwer 或回退到编辑距离)
   - `signal_metrics()`: L∞、L2、SNR 计算
   - `aggregate_results()`: 汇总成功率和 WER 统计
   - `aggregate_results_by_speaker()`: 按说话人聚合结果
   - `aggregate_results_by_emotion()`: 按源情绪聚合结果

6. **esd_dataset.py** - ESD/CN 数据集加载模块（新增）
   - `AudioSample`: 音频样本元数据类（包含路径、说话人ID、情绪标签）
   - `scan_esd_dataset()`: 扫描 ESD/CN 目录结构
   - `create_experiment_samples()`: 为所有说话人创建实验样本
   - 支持随机采样（每种情绪指定数量）
   - 处理情绪标签大小写不一致（Happy vs happy）

### 数据流

```
音频文件 → load_audio() → waveform (torch.Tensor)
                              ↓
                    attack_one_sample()
                    (PGD 优化 + EoT)
                              ↓
                    waveform_adv (对抗样本)
                              ↓
                    decode_text() × 多个 prompt
                              ↓
                    评估: 情感成功率 + WER + 信号指标
```

## 运行命令

### ESD/CN 数据集模式（推荐）

```bash
# 处理所有说话人（每种情绪 100 条）
python run_attack.py --mode esd

# 处理单个说话人
python run_attack.py --mode esd --speaker_id 0001

# 指定 ESD 数据集路径
python run_attack.py --mode esd --esd_root /path/to/ESD/CN

# 分片并行处理（多 GPU）
python run_attack.py --mode esd --shard_id 0 --num_shards 4
```

**ESD 模式特点**：
- 自动扫描 ESD/CN 数据集（10 个说话人，5 种情绪）
- 从除 Happy 外的 4 种情绪中随机采样（angry, neutral, sad, surprise）
- 按说话人分目录输出结果
- 结果包含完整元数据（speaker_id, ground_truth_emotion, target_emotion）

### 传统 sample_list 模式（兼容旧版）

```bash
# 运行攻击(使用默认配置)
python run_attack.py --mode sample_list

# 指定样本列表和输出目录
python run_attack.py --sample_list path/to/sample_list.txt --results_dir path/to/output

# 处理特定范围的样本
python run_attack.py --start_idx 0 --end_idx 10

# 分片并行处理(例如在 4 个 GPU 上)
python run_attack.py --shard_id 0 --num_shards 4  # GPU 0
python run_attack.py --shard_id 1 --num_shards 4  # GPU 1
# ... 以此类推
```

### GPU 设置

```bash
# 使用特定 GPU(通过环境变量映射)
CUDA_VISIBLE_DEVICES=6 python run_attack.py  # GPU 6 映射到 cuda:0

# 在 config.py 中设置设备
# device: str = "cuda:0"  # 或 "cpu"
```

## 依赖项

- PyTorch (需要 CUDA 支持以获得最佳性能)
- transformers (Hugging Face)
- torchaudio 或 soundfile (音频加载)
- jiwer (WER 计算,可选)
- OpenS2S 代码库(必须在 `cfg.opens2s_root` 路径下)

## 配置修改指南

### 修改攻击参数

编辑 `config.py` 中的 `Config` 类:

```python
# 扰动约束
epsilon: float = 0.008  # L∞ 范围 [-ε, +ε]

# 优化设置
total_steps: int = 60
lr: float = 0.003
optimizer: str = "adam"  # 或 "sgd"

# 损失权重(两阶段)
lambda_emo: float = 1.0
lambda_asr_stage_a: float = 1e-4
lambda_asr_stage_b: float = 1e-2
lambda_per_stage_a: float = 0.0
lambda_per_stage_b: float = 1e-5

# 目标情感
target_emotion: str = "happy"  # 或 "sad", "angry", "neutral"
```

### 修改路径

```python
# 在 config.py 中
opens2s_root: Path = Path("/path/to/OpenS2S")
model_path: Path = Path("/path/to/OpenS2S/models/OpenS2S")

# 传统模式路径
sample_list_path: Path = Path(__file__).resolve().parent / "testN10" / "sample_list.txt"
results_dir: Path = Path(__file__).resolve().parent / "testN10"

# ESD 数据集配置（新增）
esd_dataset_root: Path = Path("/data1/lixiang/ESD/CN")
esd_samples_per_emotion: int = 100  # 每种情绪采样数量
esd_exclude_emotion: str = "happy"  # 排除的情绪（不作为源情绪）
results_by_speaker: bool = True  # 按说话人分目录输出
speaker_results_dir: Path = Path(__file__).resolve().parent / "results_esd"
```

### 情绪标签配置

当前支持 **5 种情绪标签**：happy, sad, angry, neutral, surprise

```python
emo_labels: list[str] = ["happy", "sad", "angry", "neutral", "surprise"]
emo_prompts: list[str] = [
    "What is the emotion of this audio? Answer with exactly one word from: happy, sad, angry, neutral, surprise.",
    "Classify the emotion. Output exactly one word: happy/sad/angry/neutral/surprise.",
    "Emotion label only (one word): happy, sad, angry, neutral, or surprise.",
]
```

### GPU 配置

```python
# 推荐使用 H100 (80GB) 以避免内存不足
device: str = "cuda:6"  # H100
# 或使用 RTX 6000 (48GB)
# device: str = "cuda:0"  # RTX 6000
```

## 输出格式

### ESD 模式输出结构

```
results_esd/
├── 0001/                          # 说话人 0001
│   ├── 00000_0001_angry_xxx.json # 样本结果（包含元数据）
│   ├── 00000_0001_angry_xxx.wav  # 对抗样本
│   ├── 00001_0001_sad_yyy.json
│   ├── 00001_0001_sad_yyy.wav
│   └── ...
├── 0002/                          # 说话人 0002
│   └── ...
├── summary_all.json               # 全局汇总
├── summary_by_speaker.json        # 按说话人汇总（可选）
└── summary_by_emotion.json        # 按源情绪汇总（可选）
```

**JSON 格式（包含元数据）**：
```json
{
  "sample_id": "00042_0001_sad_xxx",
  "path": "/data1/lixiang/ESD/CN/0001/sad/xxx.wav",
  "speaker_id": "0001",
  "ground_truth_emotion": "sad",
  "target_emotion": "happy",
  "emo_text_clean": ["sad", "sad", "sad"],
  "emo_text_adv": ["happy", "happy", "happy"],
  "emo_pred_clean": ["sad", "sad", "sad"],
  "emo_pred_adv": ["happy", "happy", "happy"],
  "asr_text_clean": "原始转录",
  "asr_text_adv": "对抗转录",
  "success_emo": true,
  "wer": 0.05,
  "delta_linf": 0.008,
  "snr_db": 25.3,
  "loss_trace": [...],
  "grad_trace": [...]
}
```

### 传统模式输出

每个样本生成两个文件:

1. **{sample_id}.json** - 详细结果
   ```json
   {
     "sample_id": "00000_sample_name",
     "emo_text_clean": ["happy", ...],
     "emo_text_adv": ["happy", ...],
     "asr_text_clean": "original transcript",
     "asr_text_adv": "adversarial transcript",
     "success_emo": true,
     "wer": 0.05,
     "delta_linf": 0.008,
     "snr_db": 25.3,
     "loss_trace": [...],
     "grad_trace": [...]
   }
   ```

2. **{sample_id}.wav** - 对抗音频样本

3. **summary.json** 和 **summary.csv** - 汇总统计

## 关键实现细节

### 梯度链检查

代码包含梯度链完整性检查(attack_core.py:270-278):
- 如果 `delta.grad` 为 None,抛出错误
- 如果梯度范数连续 `grad_norm_patience` 步小于 `grad_norm_min`,抛出错误
- 这确保了从模型输出到音频波形的梯度流畅通

### 内存优化

- 使用梯度累积减少内存峰值(attack_core.py:223-268)
- 每个 EoT 样本立即反向传播并清理中间张量
- 启用梯度检查点(opens2s_io.py:179-193)
- 在 CUDA 上使用 bfloat16 精度

### 两阶段攻击策略

- **Stage A** (前 `stage_a_steps` 步): 专注于情感攻击,低 ASR/感知权重
- **Stage B** (剩余步数): 增加 ASR 和感知权重以保持质量

### EoT 增强

对抗样本在优化过程中应用随机变换:
- 时间偏移: ±`eot_max_shift` 样本
- 增益变化: [`eot_gain_min`, `eot_gain_max`]
- 可选: 高斯噪声、带限滤波

## 故障排除

### CUDA 内存不足

- 减少 `eot_samples`(默认 1)
- 减少 `total_steps`
- 使用更小的 `epsilon`
- 确保启用了梯度检查点

### 梯度链断裂

- 检查 OpenS2S 模型是否正确加载
- 确认 `torch_extractor` 用于可微分特征提取
- 验证 `speech_values.requires_grad = True`

### 攻击不成功

- 增加 `total_steps` 或 `lr`
- 调整损失权重(增加 `lambda_emo`)
- 检查目标情感 token 是否正确编码
- 尝试不同的 `target_emotion`

## 安全与伦理

此代码用于**授权的安全研究和学术实验**。对抗性攻击研究有助于:
- 理解多模态模型的脆弱性
- 开发更鲁棒的防御机制
- 推进可信 AI 研究

**禁止**将此代码用于恶意目的、未经授权的系统测试或任何非法活动。
