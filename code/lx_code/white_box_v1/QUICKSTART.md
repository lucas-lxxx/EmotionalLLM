# 快速开始指南

## 前置条件

1. 已安装 OpenS2S 模型（需要 `src` 模块可导入）
2. 已安装 Python 依赖：`pip install -r requirements.txt`
3. 准备好音频数据（格式：`{emotion}/{age}/{gender}/*.wav`）

## 快速运行

### 1. 设置环境变量

```bash
export PYTHONPATH=/path/to/OpenS2S:$PYTHONPATH
cd /path/to/OpenS2S_white_box
```

### 2. 准备数据

```bash
python scripts/sad2happy_batch_data_prep.py \
    --data-root /path/to/your/audio/data \
    --output-dir /path/to/output \
    --seed 2025
```

### 3. 运行实验

```bash
python scripts/sad2happy_batch_experiment.py \
    --omnispeech-path /path/to/OpenS2S/models/OpenS2S_ckpt \
    --checkpoint /path/to/emotion_classifier.pt \
    --data-root /path/to/your/audio/data \
    --output-dir /path/to/output \
    --device cuda:0
```

### 4. 评估结果

```bash
python scripts/evaluate_sad2happy_results.py \
    --results-csv /path/to/output/results.csv \
    --output-dir /path/to/output/eval \
    --device cuda:0
```

## 常见问题

### Q: 导入错误 `ModuleNotFoundError: No module named 'src'`

**A**: 需要将 OpenS2S 根目录添加到 PYTHONPATH：
```bash
export PYTHONPATH=/path/to/OpenS2S:$PYTHONPATH
```

### Q: 找不到 `OmniSpeechModel`

**A**: 确保 OpenS2S 已正确安装，并且 `src.modeling_omnispeech` 模块可以导入。

### Q: 数据路径格式

**A**: 数据应按以下格式组织：
```
data_root/
├── Sad/
│   ├── adult/
│   │   ├── female/
│   │   │   └── *.wav
│   │   └── male/
│   │       └── *.wav
│   └── child/
│       └── ...
└── Happy/
    └── ...
```

### Q: 如何训练情绪分类器？

**A**: 参考 `scripts/train_sad_happy_classifier.py`，需要 ESD 数据集。

## 输出说明

- `results.csv`: 每个样本的详细结果
- `config.json`: 实验配置
- `audio/clean/`: 原始音频
- `audio/adv/`: 对抗音频
- `text/clean/`: Clean 文本结果
- `text/adv/`: Attack 文本结果
- `eval/results_eval.csv`: 评估结果
- `eval/stats_summary.json`: 统计摘要

