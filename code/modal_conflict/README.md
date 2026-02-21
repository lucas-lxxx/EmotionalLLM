# OpenS2S 模态冲突实验

## 实验简介

这个实验研究 OpenS2S 语音对话模型中 **语义情绪** 和 **韵律情绪** 之间的冲突关系。

### 什么是模态冲突？

想象一下：一段音频，文本内容是"今天天气真好"（中性/开心的语义），但说话人用悲伤的语调来念（悲伤的韵律）。这就产生了**语义情绪**和**韵律情绪**的冲突。

本实验想回答：
1. 模型在不同层中，哪种情绪信息更强？
2. 当两者冲突时，模型更倾向于哪一种？

## 实验原理

### 核心思路

1. **准备冲突数据**：同一句话用不同情绪的语调来读
2. **提取特征**：从模型每一层提取音频的表示向量
3. **训练探针**：用简单的分类器（Probe）预测情绪
4. **对比分析**：比较预测语义情绪和韵律情绪的准确率

### 主导性指标

```
D(层) = 韵律准确率 - 语义准确率

D > 0: 该层韵律信息更强
D < 0: 该层语义信息更强
```

## 目录结构

```
modal_conflict/
├── configs/
│   └── experiment_config.yaml   # 实验配置
├── src/
│   ├── data/
│   │   └── dataset.py          # 数据加载
│   ├── models/
│   │   ├── feature_extractor.py # 特征提取
│   │   └── probes.py           # 情绪分类器
│   ├── evaluation/
│   │   └── cross_validation.py  # 交叉验证评估
│   └── visualization/
│       └── plotting.py          # 绑图
├── scripts/
│   ├── run_experiment.py        # 主程序
│   └── run_all.sh              # 一键运行
├── outputs/
│   ├── hidden_states/          # 特征缓存
│   └── results/                # 实验结果
└── requirements.txt
```

## 各模块说明

### 1. 数据模块 (`src/data/dataset.py`)

加载实验数据：
- 读取 `text.jsonl` 中的50条文本及其语义情绪标签
- 遍历5个情绪目录找到对应的音频文件
- 每条文本有5个不同韵律版本，共约250个样本
  - 如遇缺失音频文件，将自动跳过该样本

```python
# 样本结构
sample.text           # "今天天气真好"
sample.semantic_emotion  # "neutral" (文本的情绪)
sample.prosody_emotion   # "sad" (说话的语调)
sample.is_conflict    # True (语义≠韵律)
```

### 2. 特征提取 (`src/models/feature_extractor.py`)

从 OpenS2S 模型提取 hidden states：
1. 加载音频，转换为模型输入格式
2. 通过 chat template 构造固定中性 prompt + audio token 输入（与 OpenS2S worker 一致）
3. 前向传播，获取所有36层的隐藏状态
4. 定位音频对应的 token 位置
5. 对音频区域做平均池化，得到固定维度向量

```python
# 输出: [36层, 4096维]
hidden_states = extractor.extract_hidden_states(sample)
```

### 3. 情绪探针 (`src/models/probes.py`)

默认使用线性 Probe（Logistic Regression）：
- 输入：某一层的 4096 维向量
- 输出：5类情绪的预测概率

可选 MLP Probe（用于对比）：
- AdamW 优化器
- 早停机制防止过拟合
- 验证集监控

### 4. 交叉验证 (`src/evaluation/cross_validation.py`)

使用 GroupKFold 确保公平评估：
- 同一文本的不同韵律版本必须在同一折
- 避免模型"记住"特定文本
- 每层训练两个探针：语义探针 + 韵律探针

### 5. 可视化 (`src/visualization/plotting.py`)

生成多种图表：
- **主导性曲线**：D(层) 随层变化
- **准确率曲线**：两种探针的准确率对比
- **冲突子集分析**：仅看冲突样本的表现

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行实验

```bash
# 方式1: 使用一键脚本
bash scripts/run_all.sh

# 方式2: 直接运行
python scripts/run_experiment.py --config configs/experiment_config.yaml
```

### 3. 查看结果

结果保存在 `outputs/results/run_YYYYMMDD_HHMMSS/` 目录：

| 文件 | 内容 |
|-----|------|
| `dominance_curve.png` | 主导性曲线图 |
| `accuracy_curves.png` | 准确率对比图 |
| `conflict_curves.png` | 冲突子集分析图 |
| `metrics_per_layer.csv` | 逐层详细指标 |
| `summary.json` | 实验摘要 |

## 配置说明

编辑 `configs/experiment_config.yaml` 修改实验参数：

```yaml
# 数据路径
data:
  text_jsonl: /path/to/text.jsonl
  audio_root: /path/to/audio/

# 模型配置
model:
  model_path: /path/to/model/
  device: cuda
  dtype: bfloat16

# 固定Prompt
prompt: "What is the emotion of this audio? Answer with exactly one word: neutral, happy, sad, angry, surprised."
system_prompt: "You are a helpful assistant."

# 探针训练
probe:
  type: linear  # linear 或 mlp
  linear:
    max_iter: 1000
    C: 1.0
    solver: lbfgs
  hidden_dims: [512, 128]  # MLP隐藏层
  epochs: 100
  patience: 10  # 早停

# 评估
evaluation:
  n_splits: 5  # 5折交叉验证
```

## 结果解读

### 主导性曲线

- **绿色区域 (D>0)**：韵律主导，模型更关注"怎么说"
- **橙色区域 (D<0)**：语义主导，模型更关注"说什么"

### 典型发现

- 浅层：通常韵律信息更强（声学特征）
- 中层：两者可能交叉
- 深层：语义信息可能增强（语言理解）

## 常见问题

**Q: 显存不够怎么办？**
A: 修改配置中的 `dtype: float16` 或使用 CPU

**Q: 如何只评估部分层？**
A: 在配置中设置 `layers_to_evaluate: [0, 6, 12, 18, 24, 30, 35]`

**Q: 如何清除缓存重新提取？**
A: 删除 `outputs/hidden_states/` 目录，或使用 `--no-cache` 参数

## 技术细节

### Audio Span 定位

模型输入中有一个特殊的 `<|im_audio|>` token，在前向传播时被替换为音频特征。当前实现使用与 OpenS2S worker 相同的 `<|im_audio_start|><|im_audio|><|im_audio_end|>` 包裹方式，再定位 audio token 位置提取对应的 hidden states。

### 为什么用 GroupKFold？

如果同一文本的不同韵律版本分散在训练集和测试集，探针可能学会识别"这是哪句话"而不是"这是什么情绪"，导致虚高的准确率。GroupKFold 保证同一文本的所有版本要么都在训练集，要么都在测试集。
