# OpenS2S White-Box Attack Experiments

本项目包含 OpenS2S 模型的白盒对抗攻击实验代码，主要用于情绪翻转攻击（Sad → Happy）的批量实验。

## 项目结构

```
OpenS2S_white_box/
├── README.md                          # 项目说明文档
├── requirements.txt                   # Python 依赖
├── constants.py                       # 常量定义
├── utils_audio.py                     # 音频工具函数
├── scripts/                          # 实验脚本
│   ├── sad2happy_batch_data_prep.py   # 数据准备脚本
│   ├── sad2happy_batch_experiment.py  # 主实验脚本
│   ├── evaluate_sad2happy_results.py # 评估脚本
│   └── train_sad_happy_classifier.py # 训练情绪分类器
├── attack/                           # 攻击相关代码
│   ├── objectives.py                 # 攻击目标函数
│   ├── transforms.py                 # EOT 变换
│   └── optimizers/                   # 优化器
│       └── pgd.py                    # PGD 优化器
├── utils/                            # 工具模块
│   ├── emotion_classifier.py         # 情绪分类器
│   └── esd_data_loader.py            # ESD 数据加载器
└── docs/                             # 文档
    ├── 实验设置说明.md
    └── 实验结果评价指标.md
```

## 环境要求

1. **OpenS2S 模型**：需要安装 OpenS2S 模型和相关依赖
2. **Python 版本**：Python 3.8+
3. **CUDA**：支持 CUDA 的 GPU（推荐）

## 安装

### 1. 安装 OpenS2S

首先需要安装 OpenS2S 模型和相关依赖。请参考 OpenS2S 官方文档。

### 2. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 3. 设置环境变量

确保 OpenS2S 的 `src` 模块可以被导入。通常需要将 OpenS2S 根目录添加到 `PYTHONPATH`：

```bash
export PYTHONPATH=/path/to/OpenS2S:$PYTHONPATH
```

## 使用说明

### 1. 训练情绪分类器（可选）

如果需要训练自己的情绪分类器：

```bash
cd /path/to/OpenS2S_white_box
export PYTHONPATH=/path/to/OpenS2S:$PYTHONPATH

python scripts/train_sad_happy_classifier.py \
    --omnispeech-path /path/to/OpenS2S/models/OpenS2S_ckpt \
    --output checkpoints/sad_happy_classifier.pt \
    --emotions Sad Happy \
    --split train \
    --max-samples-per-emotion 100 \
    --max-speakers 3 \
    --svd-rank 20 \
    --epochs 50 \
    --device cuda:0
```

### 2. 准备数据

```bash
python scripts/sad2happy_batch_data_prep.py \
    --data-root /path/to/audio/data \
    --output-dir /path/to/output \
    --seed 2025
```

### 3. 运行批量实验

```bash
python scripts/sad2happy_batch_experiment.py \
    --omnispeech-path /path/to/OpenS2S/models/OpenS2S_ckpt \
    --checkpoint /path/to/emotion_classifier.pt \
    --data-root /path/to/audio/data \
    --output-dir /path/to/output \
    --prompt "What is the emotion of this audio? Please answer with only the emotion label (e.g., happy, sad, neutral)." \
    --epsilon 0.002 \
    --steps 30 \
    --lambda-emo 1.0 \
    --lambda-sem 1e-2 \
    --lambda-per 1e-4 \
    --device cuda:0 \
    --seed 2025
```

**注意**：默认 prompt 要求 OpenS2S 直接输出音频的情绪标签，这样评估时可以直接从输出文本中提取情绪，而不是使用外部文本分类器。这更科学，因为直接评估音频的情绪识别结果。

### 4. 评估结果

```bash
python scripts/evaluate_sad2happy_results.py \
    --results-csv /path/to/output/results.csv \
    --output-dir /path/to/output/eval \
    --device cuda:0
```

## 实验参数说明

### 攻击参数

- `epsilon`: 扰动上界（L∞ 范数），默认 0.002
- `steps`: 攻击迭代步数，默认 30
- `alpha`: 步长，默认 `epsilon / 10`
- `lambda_emo`: 情绪翻转损失权重，默认 1.0
- `lambda_sem`: 语义保持损失权重，默认 1e-2
- `lambda_per`: 感知约束损失权重，默认 1e-4

### 数据划分

- 训练集：80%
- 测试集：20%
- 随机种子：2025（固定种子保证可复现）

## 输出文件

实验完成后会生成以下文件：

- `results.csv`: 每个样本的详细结果
- `config.json`: 实验配置信息
- `audio/clean/`: 原始音频（软链接）
- `audio/adv/`: 对抗音频
- `text/clean/`: Clean 推理的文本结果
- `text/adv/`: Attack 推理的文本结果

评估完成后会生成：

- `results_eval.csv`: 评估结果（包含情绪、语义、音频扰动指标）
- `stats_summary.json`: 统计摘要

## 评估方法

### 评估策略
- **Prompt 设计**：要求 OpenS2S 直接输出音频的情绪标签（如 "happy", "sad"）
- **情绪提取**：从 OpenS2S 的输出文本中直接提取情绪关键词，而不是使用外部文本分类器
- **优势**：直接评估音频的情绪识别结果，更科学、更符合实际应用场景

详细说明请参考 `docs/评估方法说明.md`。

## 注意事项

1. **模型路径**：确保 `--omnispeech-path` 指向正确的 OpenS2S 模型目录
2. **数据格式**：音频数据应按照 `{emotion}/{age}/{gender}/*.wav` 的目录结构组织
3. **GPU 内存**：批量实验需要较大的 GPU 内存，建议使用至少 24GB 显存的 GPU
4. **依赖关系**：本项目依赖 OpenS2S 的 `src` 模块，需要确保 OpenS2S 已正确安装
5. **Prompt 设计**：建议使用明确的 prompt，要求模型输出情绪标签，便于评估

## 引用

如果使用本代码，请引用相关论文。

## 许可证

请参考 OpenS2S 项目的许可证。

## 联系方式

如有问题，请联系项目维护者。

