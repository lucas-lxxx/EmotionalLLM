# OpenS2S White-Box Attack Experiment Report (N=10)

## 实验概述

本实验在OpenS2S模型上进行白盒对抗攻击测试，目标是将"Sad"情绪翻转为"Happy"情绪。使用10个音频样本进行初步测试，观察攻击成功率。

**实验日期**: 2026-01-05
**实验位置**: `/data1/lixiang/lx_code/white_box_v1/test/`
**结果目录**: `test/results_n10/`

## 实验配置

### 模型配置
- **OpenS2S模型路径**: `/data1/lixiang/Opens2s/OpenS2S/models/OpenS2S`
- **情绪分类器**: `checkpoints/sad_happy_classifier.pt`
  - 类型: 二分类器 (Sad vs Happy)
  - 输入维度: 20 (SVD降维后)
  - 原始维度: 1280
  - 训练准确率: 93%
  - SVD解释方差: 94.88%
- **设备**: CUDA (cuda:0)
- **提取层**: layer_06, layer_16, layer_25

### 攻击参数
```python
EPSILON = 0.002          # L∞约束范围
STEPS = 30               # PGD迭代步数
ALPHA = 0.0002           # 步长 (epsilon/10)
LAMBDA_EMO = 1.0         # 情绪损失权重
LAMBDA_SEM = 0.01        # 语义保持权重
LAMBDA_PER = 0.0001      # 感知约束权重
```

### 测试提示词
```
What is the emotion of this audio? Please answer with only one word: the emotion label (happy, sad, angry, or neutral).
```

**设计目的**: 让模型仅输出检测到的情绪标签，而不生成长文本回复或音频响应。

## 测试样本

使用10个来自ESD数据集的"Sad"情绪音频样本：

| 样本ID | 文件名 | 音频路径 |
|--------|--------|----------|
| sample_001 | 20683.wav | `/data1/lixiang/OpenS2S_dataset/data/en_query_wav/Sad/adult/female/20683.wav` |
| sample_002 | 24190.wav | `/data1/lixiang/OpenS2S_dataset/data/en_query_wav/Sad/adult/female/24190.wav` |
| sample_003 | 15822.wav | `/data1/lixiang/OpenS2S_dataset/data/en_query_wav/Sad/adult/female/15822.wav` |
| sample_004 | 22344.wav | `/data1/lixiang/OpenS2S_dataset/data/en_query_wav/Sad/adult/female/22344.wav` |
| sample_005 | 20495.wav | `/data1/lixiang/OpenS2S_dataset/data/en_query_wav/Sad/adult/female/20495.wav` |
| sample_006 | 22713.wav | `/data1/lixiang/OpenS2S_dataset/data/en_query_wav/Sad/adult/female/22713.wav` |
| sample_007 | 38804.wav | `/data1/lixiang/OpenS2S_dataset/data/en_query_wav/Sad/adult/female/38804.wav` |
| sample_008 | 27476.wav | `/data1/lixiang/OpenS2S_dataset/data/en_query_wav/Sad/adult/female/27476.wav` |
| sample_009 | 48240.wav | `/data1/lixiang/OpenS2S_dataset/data/en_query_wav/Sad/adult/female/48240.wav` |
| sample_010 | 20827.wav | `/data1/lixiang/OpenS2S_dataset/data/en_query_wav/Sad/adult/female/20827.wav` |

所有样本均为成年女性的悲伤情绪语音。

## 详细实验结果

### 样本级别结果

| 样本ID | 原始检测情绪 | 攻击后检测情绪 | 攻击成功 | L∞ | L2 | SNR (dB) | 攻击时间(s) |
|--------|-------------|---------------|---------|-----|-----|----------|------------|
| sample_001 | "I'm sorry to hear that. Would you like to talk about it?" | "The emotion of the audio is sad." | ❌ | 0.00200 | 0.676 | 11.17 | 3.80 |
| sample_002 | "The emotion of the audio is sad." | "The emotion of this audio is sad." | ❌ | 0.00200 | 0.659 | 16.97 | 3.42 |
| sample_003 | "The emotion of this audio is neutral." | "The emotion of the audio is neutral." | ❌ | 0.00200 | 0.656 | 37.89 | 3.42 |
| sample_004 | "The emotion of the speech is sad." | "The emotion label is Sad." | ❌ | 0.00200 | 0.684 | 18.17 | 3.47 |
| sample_005 | "The emotion of this audio is neutral." | " presentations can be scary. Try practicing deep breaths..." | ❌ | 0.00200 | 0.852 | 24.81 | 3.54 |
| sample_006 | "The emotion of the audio is sad." | "The emotion is sad." | ❌ | 0.00200 | 0.594 | 26.62 | 3.56 |
| sample_007 | "The emotion is sad." | "the emotion is sad" | ❌ | 0.00200 | 0.709 | 16.92 | 3.42 |
| sample_008 | "The emotion label is neutral." | "Neutral." | ❌ | 0.00200 | 0.803 | 37.54 | 3.47 |
| sample_009 | "The emotion of this audio is angry." | "The emotion of the audio is neutral." | ❌ | 0.00200 | 0.585 | 19.18 | 3.60 |
| sample_010 | "The emotion is sad." | "Sad." | ❌ | 0.00200 | 0.736 | 25.95 | 3.47 |

### 模型响应详细记录

#### Sample 001 (20683.wav)
- **Clean Output**: "I'm sorry to hear that. Would you like to talk about it?"
- **Attack Output**: "The emotion of the audio is sad."
- **分析**: 模型原本输出同情性回复，攻击后虽输出情绪标签但仍识别为sad
- **目标情绪概率**: 0.5093 (Happy概率刚超过50%)

#### Sample 002 (24190.wav)
- **Clean Output**: "The emotion of the audio is sad."
- **Attack Output**: "The emotion of this audio is sad."
- **分析**: 攻击前后都正确识别为sad，仅措辞略有变化
- **目标情绪概率**: 0.5229

#### Sample 003 (15822.wav)
- **Clean Output**: "The emotion of this audio is neutral."
- **Attack Output**: "The emotion of the audio is neutral."
- **分析**: 模型将sad识别为neutral，攻击后保持neutral
- **目标情绪概率**: 0.5595 (最高)

#### Sample 004 (22344.wav)
- **Clean Output**: "The emotion of the speech is sad."
- **Attack Output**: "The emotion label is Sad."
- **分析**: 攻击前后都识别为sad，输出格式略有变化
- **目标情绪概率**: 0.5207

#### Sample 005 (20495.wav)
- **Clean Output**: "The emotion of this audio is neutral."
- **Attack Output**: " presentations can be scary. Try practicing deep breaths and remember, it's okay to feel nervous. You'll do great."
- **分析**: 攻击导致模型输出失控，生成了无关的安慰性回复
- **目标情绪概率**: 0.5379
- **特殊情况**: L2扰动较大(0.852)，SNR=24.81dB

#### Sample 006 (22713.wav)
- **Clean Output**: "The emotion of the audio is sad."
- **Attack Output**: "The emotion is sad."
- **分析**: 攻击前后都识别为sad
- **目标情绪概率**: 0.5529

#### Sample 007 (38804.wav)
- **Clean Output**: "The emotion is sad."
- **Attack Output**: "the emotion is sad"
- **分析**: 攻击前后几乎相同，仅大小写变化
- **目标情绪概率**: 0.4781 (最低，低于50%)

#### Sample 008 (27476.wav)
- **Clean Output**: "The emotion label is neutral."
- **Attack Output**: "Neutral."
- **分析**: 模型将sad识别为neutral，攻击后仍为neutral但更简洁
- **目标情绪概率**: 0.5182

#### Sample 009 (48240.wav)
- **Clean Output**: "The emotion of this audio is angry."
- **Attack Output**: "The emotion of the audio is neutral."
- **分析**: 有趣案例 - 从angry变为neutral，有一定情绪变化但非目标happy
- **目标情绪概率**: 0.4818
- **特殊情况**: 原本就误识别为angry而非sad

#### Sample 010 (20827.wav)
- **Clean Output**: "The emotion is sad."
- **Attack Output**: "Sad."
- **分析**: 攻击前后都识别为sad，仅输出更简洁
- **目标情绪概率**: 0.5207

## 统计摘要

### 攻击成功率
- **总样本数**: 10
- **成功翻转数**: 0
- **成功率**: 0% (0/10)
- **处理完成率**: 100% (10/10)

### 平均指标
| 指标 | 平均值 | 最小值 | 最大值 | 标准差 |
|------|--------|--------|--------|--------|
| L∞ (约束) | 0.00200 | 0.00200 | 0.00200 | 0.000 |
| L2 | 0.695 | 0.585 | 0.852 | 0.083 |
| SNR (dB) | 23.52 | 11.17 | 37.89 | 8.73 |
| 攻击时间 (s) | 3.51 | 3.42 | 3.80 | 0.13 |
| 目标情绪概率 | 0.520 | 0.478 | 0.560 | 0.023 |

### 输出情绪分布
**Clean阶段**:
- Sad: 6/10 (60%)
- Neutral: 3/10 (30%)
- Angry: 1/10 (10%)
- Happy: 0/10 (0%)

**Attack阶段**:
- Sad: 7/10 (70%)
- Neutral: 2/10 (20%)
- Other (失控输出): 1/10 (10%)
- Happy: 0/10 (0%)

## 失败原因分析

### 1. 攻击强度不足
- **Epsilon过小**: 0.002的L∞约束可能太小，无法产生足够的模型扰动
- **目标概率低**: Happy情绪概率均值仅0.520，接近随机猜测(0.5)
- **梯度范数**: 观察到梯度范数很小(0.000010-0.000171)，优化困难

### 2. 损失函数权重不平衡
- **Lambda_emo可能过小**: 当前为1.0，相对于其他损失可能不够dominant
- **语义约束过强**: Lambda_sem=0.01可能过度限制了扰动空间
- **感知约束**: Lambda_per=0.0001可能对L2/SNR的约束过于严格

### 3. 优化困难
```
示例优化轨迹 (Sample 001):
Step 10/30: emo_prob_target=0.5092, grad_norm=0.000053
Step 20/30: emo_prob_target=0.5093, grad_norm=0.000053
Step 30/30: emo_prob_target=0.5093, grad_norm=0.000053
```
- 目标概率几乎不变，优化陷入平台期
- 梯度范数极小，难以有效更新

### 4. 情绪分类器泛化性
- **训练数据有限**: 仅100个样本(50 Sad + 50 Happy)
- **SVD降维**: 从1280维降至20维可能损失了关键信息
- **层选择**: layer_06, layer_16, layer_25的组合可能不是最优

### 5. 模型鲁棒性
- OpenS2S作为大型语音模型，对小扰动(epsilon=0.002)具有较强鲁棒性
- 音频域的扰动空间相比图像更受限
- 语音的时序连续性使得局部扰动难以改变整体情绪感知

## 改进建议

### 短期改进 (参数调优)

#### 1. 增加攻击强度
```python
EPSILON = 0.005  # 增加到0.005 (当前的2.5倍)
STEPS = 50       # 增加迭代次数到50
ALPHA = 0.001    # 相应增加步长
```

#### 2. 调整损失权重
```python
LAMBDA_EMO = 10.0     # 大幅增加情绪损失权重
LAMBDA_SEM = 0.005    # 降低语义约束
LAMBDA_PER = 0.00005  # 降低感知约束
```

#### 3. 使用自适应学习率
```python
# 考虑在PGD中加入momentum或使用Adam优化器
# 当梯度范数过小时自动增加学习率
```

### 中期改进 (算法增强)

#### 1. 多目标优化策略
- 分阶段优化: 先最大化情绪损失(前20步)，再考虑语义约束(后10步)
- 自适应权重: 根据目标概率动态调整lambda值

#### 2. 改进情绪分类器
```python
# 增加训练数据到500-1000个样本
# 尝试不同的SVD维度: 50, 100, 200
# 测试不同的层组合
```

#### 3. 集成多个攻击方法
- 尝试C&W攻击
- 结合音频域和频谱域的攻击
- 使用进化算法辅助优化

### 长期改进 (架构改进)

#### 1. 基于迁移的攻击
- 在小模型上生成对抗样本
- 迁移到OpenS2S上测试

#### 2. 黑盒攻击对比
- 实现query-based攻击方法
- 对比白盒和黑盒的成功率

#### 3. 人类感知研究
- 即使模型未被欺骗，对抗样本是否改变人类感知
- 主观评测情绪翻转效果

## 下一步实验计划

### 实验A: 强攻击参数测试 (N=10)
```python
EPSILON = 0.005
STEPS = 50
LAMBDA_EMO = 10.0
LAMBDA_SEM = 0.005
```
**预期**: 成功率提升至30-50%

### 实验B: 分阶段优化 (N=10)
```python
# Stage 1 (steps 1-30): 纯情绪优化
LAMBDA_EMO = 10.0, LAMBDA_SEM = 0.0
# Stage 2 (steps 31-50): 加入语义约束
LAMBDA_EMO = 10.0, LAMBDA_SEM = 0.01
```
**预期**: 成功率提升至40-60%

### 实验C: 大规模测试 (N=100)
- 使用实验A或B的最佳参数
- 在100个样本上测试泛化性能
- 分析成功/失败案例的特征

## 代码配置记录

### 主测试脚本
```python
# test/run_test_n10_v2.py
OMNISPEECH_PATH = "/data1/lixiang/Opens2s/OpenS2S/models/OpenS2S"
CHECKPOINT_PATH = "checkpoints/sad_happy_classifier.pt"
SAMPLE_LIST_PATH = "test/sad_samples_10.txt"
OUTPUT_DIR = "test/results_n10"
DEVICE = "cuda:0"

EPSILON = 0.002
STEPS = 30
LAMBDA_EMO = 1.0
LAMBDA_SEM = 1e-2
LAMBDA_PER = 1e-4

PROMPT = "What is the emotion of this audio? Please answer with only one word: the emotion label (happy, sad, angry, or neutral)."
```

### 关键代码片段

#### SVD组件加载修复
```python
# 确保SVD组件在正确的设备上
svd_components = checkpoint.get('svd_components', None)
svd_mean = checkpoint.get('svd_mean', None)
if svd_components is not None:
    if isinstance(svd_components, np.ndarray):
        svd_components = torch.from_numpy(svd_components).float()
    svd_components = svd_components.to(DEVICE)
if svd_mean is not None:
    if isinstance(svd_mean, np.ndarray):
        svd_mean = torch.from_numpy(svd_mean).float()
    svd_mean = svd_mean.to(DEVICE)
```

#### 攻击调用
```python
waveform_adv, metrics, attack_time, sample_rate = run_attack(
    model=model,
    tokenizer=tokenizer,
    audio_extractor=audio_extractor,
    audio_path=audio_path,
    prompt=PROMPT,
    target_emotion="Happy",
    source_emotion="Sad",
    emotion_classifier=emotion_classifier,
    emotion_label_to_idx=emotion_label_to_idx,
    svd_components=svd_components,
    svd_mean=svd_mean,
    epsilon=EPSILON,
    steps=STEPS,
    alpha=EPSILON / 10.0,
    lambda_emo=LAMBDA_EMO,
    lambda_sem=LAMBDA_SEM,
    lambda_per=LAMBDA_PER,
    device=DEVICE
)
```

## 输出文件清单

### 生成的文件
```
test/results_n10/
├── config.json                          # 实验配置
├── results.json                         # 完整结果JSON
├── audio/
│   ├── clean/                          # 原始音频软链接
│   │   ├── sample_001.wav -> [原始路径]
│   │   └── ... (10个文件)
│   └── adv/                            # 对抗音频
│       ├── sample_001.wav
│       └── ... (10个文件)
└── text/
    ├── sample_001_clean.txt            # 原始推理输出
    ├── sample_001_attack.txt           # 攻击后推理输出
    └── ... (20个文件)
```

### 日志文件
- `test/run_test_n10_final_v2.log`: 完整运行日志
- 包含每个step的loss、metrics、grad_norm等详细信息

## 结论

本次N=10的初步实验表明：

1. **技术可行性**: 白盒攻击框架完整可运行，所有样本处理成功率100%
2. **攻击效果**: 当前参数设置下攻击成功率为0%，需要大幅调整参数
3. **优化挑战**: 梯度范数极小，目标概率停滞，优化陷入局部最优
4. **改进方向**: 需要增加epsilon、提高lambda_emo、增加迭代步数

**下一步行动**: 按照改进建议实施实验A(强攻击参数)，预期可观察到明显的成功率提升。

---

**实验执行者**: Claude Code
**报告生成时间**: 2026-01-05
**实验版本**: white_box_v1
**OpenS2S版本**: v1.0
