# White-Box Adversarial Attack on OpenS2S

## 项目概述

本项目实现了针对OpenS2S多模态语音模型的白盒对抗攻击，目标是在保持语义不变的前提下改变模型的情绪识别结果。

**重要发现**：经过详细测试，我们发现OpenS2S模型在情绪识别任务上存在严重的能力限制，即使在理想条件下（正确的采样率、语言匹配）也无法准确识别情绪。详见 `FINDINGS.md`。

## 核心文件

### 攻击实现
- `attack_core.py` - 核心攻击算法实现（PGD + EoT + 两阶段优化）
- `opens2s_io.py` - OpenS2S模型加载和I/O接口
- `config.py` - 所有攻击参数配置
- `run_attack.py` - 批量攻击执行脚本

### 评估
- `eval_metrics.py` - WER、情绪准确率等指标计算

### 文档
- `README.md` - 快速开始指南
- `RUN_GUIDE.md` - 详细使用说明
- `FINDINGS.md` - 实验发现和问题分析

### 示例
- `sample_list.txt` - 样本列表示例
- `quick_start.sh` - 环境检查脚本

## 快速开始

### 1. 环境要求

```bash
# Python 3.10+
# PyTorch 2.0+ with CUDA
# OpenS2S模型和依赖

# 检查环境
bash quick_start.sh
```

### 2. 配置

编辑 `config.py`：

```python
# 路径配置
opens2s_root = Path("/path/to/OpenS2S")
model_path = Path("/path/to/OpenS2S/models/OpenS2S")
sample_list_path = Path("sample_list.txt")
results_dir = Path("results")

# GPU配置
device = "cuda:0"  # 根据实际GPU调整

# 攻击参数
epsilon = 0.008      # L∞扰动上限
total_steps = 60     # 优化步数
lr = 0.003          # 学习率
target_emotion = "happy"  # 目标情绪
```

### 3. 准备样本列表

创建 `sample_list.txt`，每行一个音频文件路径：

```
/path/to/audio1.wav
/path/to/audio2.wav
/path/to/audio3.wav
```

### 4. 运行攻击

```bash
# 使用OpenS2S的Python环境
/path/to/OpenS2S/venv/bin/python3 run_attack.py

# 或指定参数
/path/to/OpenS2S/venv/bin/python3 run_attack.py \
    --sample_list my_samples.txt \
    --results_dir my_results
```

### 5. 查看结果

结果保存在 `results_dir` 目录：
- `XXXXX_filename.wav` - 对抗样本音频
- `XXXXX_filename.json` - 详细结果（loss trace、预测等）
- `summary.json` - 汇总统计

## 攻击方法

### 算法流程

1. **初始化**：加载OpenS2S模型和目标音频
2. **两阶段优化**：
   - Stage A (步骤0-19)：优先攻击情绪分类
   - Stage B (步骤20-59)：加强语义保持和感知质量
3. **EoT (Expectation over Transformations)**：使用可微分增强提高鲁棒性
4. **约束投影**：每步将扰动投影到L∞球内

### 损失函数

```
L_total = λ_emo * L_emo + λ_asr * L_asr + λ_per * L_per

其中：
- L_emo: 目标情绪的交叉熵损失
- L_asr: 自一致性ASR损失（保持语义）
- L_per: 多分辨率STFT感知损失（保持音质）
```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| epsilon | 0.008 | L∞扰动上限 |
| total_steps | 60 | 总优化步数 |
| stage_a_steps | 20 | Stage A步数 |
| lr | 0.003 | 学习率 |
| lambda_emo | 1.0 | 情绪损失权重 |
| lambda_asr_stage_b | 1e-2 | Stage B ASR权重 |
| lambda_per_stage_b | 1e-5 | Stage B 感知权重 |

## 实验发现

### 关键问题

经过详细测试，我们发现OpenS2S模型存在以下问题：

1. **情绪识别能力极差**
   - 在50个sad样本上测试，0%识别为sad
   - 系统性偏向输出"neutral"
   - 大量无效输出（"the emotion label is"等）

2. **ASR功能失效**
   - 无法转录实际语音内容
   - 只能生成模板化回应（"好的，我来转录..."）

3. **特征提取问题**
   - 24kHz音频 + 16kHz mel参数 = 时间分辨率错误
   - 即使修正采样率，模型仍无法工作

4. **模型定位不匹配**
   - OpenS2S是对话助手，不是语音分析工具
   - 不适合情绪分类和精确转录任务

详细分析见 `FINDINGS.md`。

### 实验结果

**配置1：24kHz音频（原始）**
- 情绪攻击成功率：40%
- WER=0.0：0%
- 问题：特征不匹配 + 模型能力不足

**配置2：16kHz重采样**
- 情绪攻击成功率：100%（但baseline也是错的）
- ASR输出全是中文（错误）
- 问题：重采样破坏音频质量

**配置3：中文ESD数据集（16kHz）**
- 情绪识别：0/5正确
- ASR：0/5正确
- 结论：即使采样率正确，模型仍无法工作

## 代码结构

```
codex/
├── attack_core.py          # 核心攻击算法
├── opens2s_io.py          # 模型I/O接口
├── config.py              # 配置文件
├── run_attack.py          # 主执行脚本
├── eval_metrics.py        # 评估指标
├── README.md              # 本文件
├── RUN_GUIDE.md           # 详细指南
├── FINDINGS.md            # 实验发现
├── quick_start.sh         # 环境检查
└── sample_list.txt        # 样本列表示例
```

## 依赖项

```
torch>=2.0.0
torchaudio>=2.0.0
transformers>=4.30.0
jiwer  # WER计算
```

OpenS2S模型依赖：
- 需要完整的OpenS2S仓库和模型权重
- 参考：https://github.com/ictnlp/OpenS2S

## 注意事项

1. **GPU内存**：需要至少40GB显存（RTX 6000或更高）
2. **采样率**：确保音频采样率与配置匹配
3. **模型限制**：OpenS2S不适合情绪分类任务，实验结果仅供研究参考
4. **路径配置**：修改 `config.py` 中的所有路径为实际路径
