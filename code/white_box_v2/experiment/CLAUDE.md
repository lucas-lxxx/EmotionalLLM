# ESD 全量白盒攻击实验 — 执行指南

## 实验目标

对 ESD/CN 数据集全部音频（排除 happy 作为源情绪）执行白盒对抗攻击。
目标情绪：happy。结果输出至 `result/ESDfinal/`。

## 前置条件

- **GPU**: CUDA 可用，建议 ≥48GB 显存（H100/A100/RTX 6000）
- **依赖**: torch, transformers, torchaudio, sentence-transformers, jiwer
- **模型**: OpenS2S 在 `/data1/lixiang/Opens2s/OpenS2S/models/OpenS2S`
- **数据**: ESD/CN 在 `/data1/lixiang/ESD/CN`

## 执行流程

### Step 0: 探测可用 GPU

**执行前必须先检查 GPU 状态**，这是公共服务器，需要找到空闲卡：

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits
```

**选卡标准**：
- 显存 >= 48GB（RTX 6000 Ada = 48GB，H100 = 80GB）
- 空闲显存 >= 40GB（已用显存 < 8GB 视为空闲）
- GPU 利用率低（< 10%）

**示例输出与判断**：
```
0, NVIDIA RTX 6000, 1200, 49140, 47940, 0     ← 空闲，可用
1, NVIDIA RTX 6000, 45000, 49140, 4140, 95    ← 被占，跳过
2, NVIDIA RTX 6000, 500, 49140, 48640, 0      ← 空闲，可用
3, NVIDIA RTX 6000, 48000, 49140, 1140, 98    ← 被占，跳过
```

→ 可用卡: GPU 0, 2 → `num_shards=2`

### Step 1: 启动攻击（尽量多卡并行）

根据 Step 0 探测到的空闲卡数量，**尽可能多卡并行**以加快速度。

每张卡各自加载一份模型，处理不同的样本子集（按索引取模分配），互不干扰。

**单卡运行**（只有 1 张空闲卡时）：

```bash
CUDA_VISIBLE_DEVICES=0 python run_attack.py --mode esd
```

**多卡并行**（假设 GPU 0,2,5,6 空闲，共 4 张）：

```bash
CUDA_VISIBLE_DEVICES=0 python run_attack.py --mode esd --shard_id 0 --num_shards 4 &
CUDA_VISIBLE_DEVICES=2 python run_attack.py --mode esd --shard_id 1 --num_shards 4 &
CUDA_VISIBLE_DEVICES=5 python run_attack.py --mode esd --shard_id 2 --num_shards 4 &
CUDA_VISIBLE_DEVICES=6 python run_attack.py --mode esd --shard_id 3 --num_shards 4 &
wait
```

**规则**：`num_shards` = 空闲卡总数，`shard_id` 从 0 递增，`CUDA_VISIBLE_DEVICES` 填实际空闲的物理 GPU 编号。

### Step 2: 汇总结果

分片模式下各进程只写每样本 JSON，不写汇总。全部跑完后统一生成：

```bash
python run_attack.py --aggregate_only
```

### 单说话人快速测试

先跑一个说话人验证流程，再全量运行：

```bash
CUDA_VISIBLE_DEVICES=0 python run_attack.py --mode esd --speaker_id 0001
```

## 断点续跑

`config.py` 中 `skip_existing=True`（默认），已存在的 JSON 会被跳过。
直接重新运行相同命令即可从断点继续。

## 监控进度

```bash
# 已完成样本数
find result/ESDfinal/ -name "*.json" ! -name "summary_*" | wc -l

# 各说话人完成数
for d in result/ESDfinal/*/; do echo "$(basename $d): $(ls $d/*.json 2>/dev/null | wc -l)"; done
```

## 预期数据量与耗时

- 10 说话人 x 4 情绪（angry/neutral/sad/surprise）x ~350 条 ≈ 14,000 样本
- 每样本产出：1 个 JSON（~17KB）+ 1 个 WAV（~100KB）
- 预计总输出 ~1.6GB
- 单卡耗时 ~1.5 分钟/样本 → 14,000 样本约 14 天
- N 卡并行 → 约 14/N 天（4 卡 ≈ 3.5 天）

## 输出结构

```
result/ESDfinal/
├── 0001/                    # 说话人目录
│   ├── 00000_0001_angry_000399.json
│   ├── 00000_0001_angry_000399.wav
│   └── ...
├── 0002/
│   └── ...
├── summary_all.json         # 全局汇总
├── summary_by_speaker.json  # 按说话人
└── summary_by_emotion.json  # 按源情绪
```

## 单样本 JSON 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `sample_id` | string | 样本唯一标识 |
| `path` | string | 原始音频路径 |
| `speaker_id` | string | 说话人 ID |
| `ground_truth_emotion` | string | 源情绪标签 |
| `target_emotion` | string | 目标情绪（happy） |
| `emo_text_clean` | list[string] | 原始音频情绪识别结果（3 个 prompt） |
| `emo_text_adv` | list[string] | 对抗音频情绪识别结果 |
| `emo_pred_clean` | list[string] | 归一化后的原始情绪标签 |
| `emo_pred_adv` | list[string] | 归一化后的对抗情绪标签 |
| `asr_text_clean` | string | 原始音频转录文本 |
| `asr_text_adv` | string | 对抗音频转录文本 |
| `success_emo` | bool | 3 个 prompt 是否全部预测为 happy |
| `wer` | float | 词错误率（clean vs adv 转录） |
| `semantic_sim` | float | 语义余弦相似度（clean vs adv 转录） |
| `semantic_preserved` | bool | semantic_sim >= 0.8 |
| `delta_linf` | float | 扰动 L∞ 范数 |
| `delta_l2` | float | 扰动 L2 范数 |
| `snr_db` | float | 信噪比 (dB) |
| `grad_norm_trace` | list[float] | 每步梯度范数 |
| `loss_trace` | list[dict] | 每步损失值明细 |

## 汇总指标说明

| 指标 | 说明 |
|------|------|
| `emo_success_rate` | 情绪攻击成功率 |
| `semantic_preserve_rate` | 语义保持率（semantic_sim >= 0.8） |
| `joint_success_semantic` | 情绪成功 AND 语义保持 |
| `wer_le_0.0` | WER = 0 的比例 |
| `joint_success_le_0.0` | 情绪成功 AND WER = 0 |

## 故障处理

- **CUDA OOM — 显存优化手段**（按优先级依次尝试）：
  1. **SDPA 注意力**：在 `opens2s_io.py` 的 `from_pretrained()` 中添加 `attn_implementation="sdpa"`。PyTorch 2.0+ 内置，无需额外安装，省 3-5GB
  2. **SGD 替代 Adam**：修改 `config.py` 中 `optimizer: str = "sgd"`。Adam 额外存储 2 倍参数量的动量/方差状态，SGD 无此开销，省约 5-8GB。注意 lr 可能需要调大（如 0.01）
  3. **Flash Attention**：`from_pretrained()` 中添加 `attn_implementation="flash_attention_2"`，需 `pip install flash-attn`（编译安装，可能失败）
  4. **8-bit 量化**：`pip install bitsandbytes`，然后 `from_pretrained()` 中添加 `load_in_8bit=True`，去掉 `torch_dtype` 参数。省约 50% 模型显存。注意：需验证梯度链是否仍然完整（8-bit 下梯度可能断裂）
  5. 以上手段可叠加使用（如 SDPA + SGD）。模型已默认启用 bfloat16 + gradient checkpointing
- **梯度链断裂**: 检查 OpenS2S 模型路径和依赖
- **sentence-transformers 未安装**: `pip install sentence-transformers`
- **断点续跑失败**: 检查 `config.py` 中 `skip_existing=True`

## 核心模块说明

| 文件 | 功能 |
|------|------|
| `config.py` | 集中配置（路径、超参、阈值） |
| `run_attack.py` | 主入口（加载数据→攻击→评估→保存） |
| `attack_core.py` | PGD 两阶段攻击算法 |
| `opens2s_io.py` | OpenS2S 模型加载和可微分特征提取 |
| `eval_metrics.py` | WER、语义相似度、信号指标、汇总统计 |
| `esd_dataset.py` | ESD/CN 数据集扫描和采样 |
