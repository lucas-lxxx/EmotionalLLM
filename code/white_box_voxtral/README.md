# Voxtral 白盒对抗攻击实验

对 Voxtral-Mini-3B-2507 执行白盒对抗攻击，验证攻击方法从 OpenS2S 到 Voxtral 的跨模型泛化能力。

## 环境依赖

```
transformers>=4.54.0
torch>=2.0
torchaudio
jiwer
sentence-transformers
```

## 环境检查

```bash
# 确认 transformers 版本
python -c "import transformers; print(transformers.__version__)"

# 确认模型路径
ls /data1/lixiang/Voxtral/

# 确认 GPU
nvidia-smi --query-gpu=index,name,memory.free,utilization.gpu --format=csv,noheader,nounits
```

## 文件结构

| 文件 | 功能 |
|------|------|
| `config.py` | 集中配置（路径、Voxtral token ID、攻击超参） |
| `voxtral_io.py` | Voxtral 模型加载、可微分特征提取、推理解码 |
| `attack_core.py` | PGD 两阶段攻击（EoT + 梯度优化） |
| `run_attack.py` | 主入口（加载→攻击→评估→保存） |
| `eval_metrics.py` | WER、语义相似度、信号指标、汇总统计 |
| `esd_dataset.py` | ESD/CN 数据集扫描和采样 |

## 运行

### 单说话人测试

```bash
CUDA_VISIBLE_DEVICES=0 python run_attack.py --speaker_id 0001
```

### 全量单卡

```bash
CUDA_VISIBLE_DEVICES=0 python run_attack.py
```

### 多卡并行（以 4 卡为例）

```bash
CUDA_VISIBLE_DEVICES=0 python run_attack.py --shard_id 0 --num_shards 4 &
CUDA_VISIBLE_DEVICES=2 python run_attack.py --shard_id 1 --num_shards 4 &
CUDA_VISIBLE_DEVICES=5 python run_attack.py --shard_id 2 --num_shards 4 &
CUDA_VISIBLE_DEVICES=6 python run_attack.py --shard_id 3 --num_shards 4 &
wait
```

### 汇总结果

```bash
python run_attack.py --aggregate_only
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--esd_root` | `/data1/lixiang/OpenS2S_dataset/ESD/CN` | ESD 数据集路径 |
| `--speaker_id` | None | 指定说话人（如 `0001`） |
| `--results_dir` | `../result/Voxtral/` | 结果输出目录 |
| `--shard_id` | None | 分片 ID（多卡并行） |
| `--num_shards` | 1 | 总分片数 |
| `--start_idx` / `--end_idx` | None | 样本索引范围 |
| `--aggregate_only` | False | 仅汇总已有结果 |

## 输出结构

```
result/Voxtral/
├── 0001/
│   ├── 00000_0001_angry_000399.json
│   ├── 00000_0001_angry_000399.wav
│   └── ...
├── summary_all.json
├── summary_by_speaker.json
└── summary_by_emotion.json
```

## 断点续跑

`config.py` 中 `skip_existing=True`（默认），已存在的 JSON 会被跳过。直接重新运行相同命令即可。

## 与 OpenS2S 实验的差异

- 模型: Voxtral-Mini-3B (~4.7B) vs OpenS2S (~7B+)
- 无 system prompt（Voxtral 不支持）
- Chat 模板: Mistral Instruct `[INST]...[/INST]` vs OpenS2S 自定义
- Audio token: 375 个 `[AUDIO]` token vs 单个占位符
- Generate 返回完整序列，需 slice 去掉 input
- 攻击超参数完全一致，便于对比
