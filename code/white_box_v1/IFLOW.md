# IFLOW 上下文指南

本文档为 iFlow CLI 提供 OpenS2S 白盒攻击实验项目的上下文信息，用于指导未来的交互和任务执行。

## 项目概述

**项目名称**：OpenS2S White-Box Attack Experiments  
**项目类型**：Python 代码项目（研究/实验性质）  
**主要目的**：针对 OpenS2S 语音-文本模型的白盒对抗攻击实验，专注于 Sad → Happy 情绪翻转攻击的批量实验。  
**核心技术**：
- OpenS2S（OmniSpeechModel）音频-文本多模态模型
- PGD（Projected Gradient Descent）白盒攻击算法
- 情绪分类器（FrozenEmotionClassifier）用于攻击目标
- SVD（奇异值分解）降维技术

**架构特点**：
- 模块化设计：攻击目标函数、优化器、数据加载器分离
- 批量实验支持：自动数据划分、并行处理、结果收集
- 可复现性：固定随机种子、配置记录、Git 信息跟踪

## 项目结构

```
/data1/lixiang/lx_code/white_box_v1/
├── __init__.py
├── constants.py                       # 常量定义（音频 token、模型常量）
├── utils_audio.py                     # 音频工具函数（模型加载、特征提取）
├── requirements.txt                   # Python 依赖
├── README.md                          # 项目主说明文档
├── QUICKSTART.md                      # 快速开始指南
├── PROJECT_SUMMARY.md                 # 项目整理总结
├── 情绪反转攻击实验详解.md            # 详细实验说明
├── IFLOW.md                           # 本文件（iFlow 上下文）
├── scripts/                           # 实验脚本目录
│   ├── __init__.py
│   ├── sad2happy_batch_data_prep.py   # 数据准备脚本
│   ├── sad2happy_batch_experiment.py  # 主实验脚本（完整 pipeline）
│   ├── evaluate_sad2happy_results.py  # 评估脚本
│   └── train_sad_happy_classifier.py  # 训练情绪分类器脚本
├── attack/                            # 攻击相关代码
│   ├── __init__.py
│   ├── objectives.py                  # 攻击目标函数（EmotionAttackObjective）
│   ├── transforms.py                  # EOT 变换（未使用）
│   └── optimizers/
│       ├── __init__.py
│       └── pgd.py                     # PGD 优化器
├── utils/                             # 工具模块
│   ├── __init__.py
│   ├── emotion_classifier.py          # 情绪分类器（FrozenEmotionClassifier）
│   └── esd_data_loader.py             # ESD 数据加载器
└── docs/                              # 文档
    ├── 实验设置说明.md                # 详细实验设置说明
    └── 评估方法说明.md                # 评估方法说明
```

## 环境要求

### 系统要求
- **操作系统**：Linux（推荐）
- **Python 版本**：3.8+
- **CUDA**：支持 CUDA 的 GPU（推荐至少 24GB 显存）

### 外部依赖
- **OpenS2S 模型**：需要安装 OpenS2S 模型和相关依赖
- **PYTHONPATH**：必须包含 OpenS2S 根目录以便导入 `src` 模块

### Python 依赖
核心依赖（见 `requirements.txt`）：
- `torch>=2.0.0`
- `torchaudio>=2.0.0`
- `transformers>=4.30.0`
- `numpy>=1.21.0`
- `scipy>=1.9.0`
- `soundfile>=0.12.0`
- `tqdm>=4.65.0`
- `sentence-transformers>=2.2.0`（语义相似度评估）
- `scikit-learn>=1.0.0`（SVD 和分类器训练）
- `gitpython>=3.1.0`（可选，用于 Git 信息）

## 安装与设置

### 1. 安装 OpenS2S
按照 OpenS2S 官方文档安装模型和相关依赖。

### 2. 安装 Python 依赖
```bash
pip install -r requirements.txt
```

### 3. 设置环境变量
```bash
export PYTHONPATH=/path/to/OpenS2S:$PYTHONPATH
```

## 使用说明

### 关键命令

#### 训练情绪分类器（可选）
```bash
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

#### 准备数据
```bash
python scripts/sad2happy_batch_data_prep.py \
    --data-root /path/to/audio/data \
    --output-dir /path/to/output \
    --seed 2025
```

#### 运行批量实验（主脚本）
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

#### 评估结果
```bash
python scripts/evaluate_sad2happy_results.py \
    --results-csv /path/to/output/results.csv \
    --output-dir /path/to/output/eval \
    --device cuda:0
```

### 数据格式要求
音频数据应按照以下目录结构组织：
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

### 实验参数说明
- **epsilon**: 扰动上界（L∞ 范数），默认 0.002
- **steps**: 攻击迭代步数，默认 30
- **alpha**: 步长，默认 `epsilon / 10`
- **lambda_emo**: 情绪翻转损失权重，默认 1.0
- **lambda_sem**: 语义保持损失权重，默认 1e-2
- **lambda_per**: 感知约束损失权重，默认 1e-4

## 开发约定

### 代码风格
- **导入路径**：已从原项目 `emotion_editing_v6` 修改为本地相对导入
- **模块组织**：功能模块化，攻击、工具、脚本分离
- **类型提示**：关键函数使用 Python 类型提示
- **错误处理**：使用 try-except 处理可能的外部依赖失败

### 命名约定
- **Python 文件**：小写字母加下划线（snake_case）
- **类名**：大驼峰（CamelCase）
- **函数/变量**：小写字母加下划线（snake_case）
- **常量**：全大写加下划线（UPPER_CASE）

### 测试与验证
- **可复现性**：固定随机种子（2025）
- **配置记录**：自动生成 `config.json` 记录实验参数
- **Git 跟踪**：自动记录 Git 提交信息（如果可用）
- **错误处理**：记录失败样本并继续处理

## 关键文件说明

### `scripts/sad2happy_batch_experiment.py`
**功能**：主实验脚本，执行完整 pipeline：
1. 数据准备（调用数据准备脚本）
2. Clean 推理（原始音频 → 文本）
3. 白盒攻击（PGD 生成对抗音频）
4. Attack 推理（对抗音频 → 文本）
5. 结果保存（CSV、JSON、音频、文本文件）

**关键函数**：
- `prepare_data()`: 准备数据并划分
- `clean_inference()`: 执行原始推理
- `run_attack()`: 执行 PGD 攻击
- `run_batch_experiment()`: 批量实验主函数

### `attack/objectives.py`
**功能**：定义攻击目标函数 `EmotionAttackObjective`
- 情绪翻转损失（使用 emotion classifier）
- 语义保持损失（hidden states cosine similarity）
- 感知约束损失（L2、L∞、SNR）
- 支持 SVD 降维变换

### `utils/emotion_classifier.py`
**功能**：情绪分类器模块
- `FrozenEmotionClassifier`: 冻结的线性分类器
- 训练函数：`train_emotion_classifier()`
- 决策边界计算：`compute_boundary_direction()`

### `utils_audio.py`
**功能**：音频工具函数
- 模型加载：`load_model()`, `load_audio_extractor()`
- 特征提取：`get_audio_encoder_layers()`
- 音频处理：`load_waveform()`

## 评估方法

### 评估策略
- **Prompt 设计**：要求 OpenS2S 直接输出音频的情绪标签
- **情绪提取**：从输出文本中提取情绪关键词（happy, sad, neutral 等）
- **优势**：直接评估音频的情绪识别结果，更科学、更符合实际应用

### 关键指标
1. **情绪翻转率 (EFR)**：成功翻转的样本比例
2. **Delta Happy**：Happy 概率的变化幅度
3. **语义相似度**：Clean 和 Attack 文本的语义相似度（0-1）
4. **音频扰动指标**：
   - L∞ 范数（最大扰动幅度）
   - L2 范数（总扰动能量）
   - SNR（信噪比，dB）

### 评估脚本
`scripts/evaluate_sad2happy_results.py` 提供：
- 情绪分类（使用独立文本分类模型）
- 语义相似度计算（使用 Sentence-BERT）
- 统计摘要生成

## 注意事项

### 依赖管理
1. **OpenS2S 依赖**：项目依赖于 OpenS2S 的 `src` 模块，必须正确设置 `PYTHONPATH`
2. **GPU 内存**：批量实验需要较大显存（建议 ≥24GB）
3. **音频格式**：支持 WAV 格式，单声道或立体声（自动转换为单声道）

### 实验设置
1. **随机种子**：固定为 2025 保证可复现性
2. **数据划分**：80% 训练集 / 20% 测试集
3. **攻击参数**：默认参数经过实验验证，修改需谨慎

### 错误处理
1. **模型加载失败**：检查 OpenS2S 安装和 `PYTHONPATH`
2. **音频读取失败**：检查文件格式和权限
3. **GPU 内存不足**：减少批量大小或使用更小模型

### 输出文件
实验生成的文件结构：
```
output_dir/
├── audio/
│   ├── clean/          # 原始音频（软链接）
│   └── adv/            # 对抗音频
├── text/
│   ├── clean/          # Clean 推理文本
│   └── adv/            # Attack 推理文本
├── results.csv         # 详细结果（每样本一行）
├── config.json         # 实验配置
└── eval/               # 评估结果（评估后生成）
    ├── results_eval.csv
    └── stats_summary.json
```

## 扩展与定制

### 修改攻击目标
1. **编辑 `attack/objectives.py`**：修改损失函数权重或添加新损失项
2. **调整特征层**：修改 `target_layers` 参数选择不同编码器层
3. **更改情绪目标**：修改 `target_emotion` 参数实现其他情绪翻转

### 添加新实验
1. **创建新脚本**：参考现有脚本结构
2. **修改数据加载**：调整 `utils/esd_data_loader.py` 支持新数据集
3. **自定义评估**：修改评估脚本以支持新指标

### 性能优化
1. **批量处理**：现有脚本支持批量处理，可进一步并行化
2. **内存优化**：使用梯度检查点或更小的批大小
3. **分布式训练**：可扩展为多 GPU 训练

## 故障排除

### 常见问题
1. **导入错误 `ModuleNotFoundError: No module named 'src'`**
   - 解决方案：正确设置 `PYTHONPATH` 环境变量

2. **GPU 内存不足**
   - 解决方案：减少批量大小，使用 `--device cpu` 测试，或使用更大显存 GPU

3. **音频读取错误**
   - 解决方案：检查文件格式是否为 WAV，确保文件未损坏

4. **情绪分类器加载失败**
   - 解决方案：检查 checkpoint 路径，确保文件存在且格式正确

### 调试建议
1. **启用详细日志**：修改脚本添加更多打印语句
2. **小规模测试**：使用少量样本（如 10 个）测试完整流程
3. **检查中间文件**：验证生成的音频和文本文件是否正确

## 版本与兼容性

### 已知兼容版本
- **OpenS2S**：需要特定版本（参考原项目文档）
- **PyTorch**：≥2.0.0
- **Transformers**：≥4.30.0

### 版本记录
- **项目版本**：基于 `emotion_editing_v6` 代码库迁移
- **迁移日期**：根据文件内容推断为近期迁移
- **修改内容**：导入路径重构，结构调整

---

*本文件最后更新：2026年1月5日*  
*基于项目文件分析生成，用于 iFlow CLI 上下文指导*