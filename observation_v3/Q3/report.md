# Q3：传统 SER 攻击方法对 ALLM 有效吗？

## 问题

语音情感识别（SER）领域已有成熟的对抗攻击方法。如果在 SER 模型上生成的对抗音频直接输入 ALLM，能否骗过 ALLM 的情绪判断？

## 实验设计

采用 STAA-Net（arXiv 2402.01227）——一种 generator-based 的 SER 对抗攻击方法。流程：

1. **训练 surrogate SER**：wav2vec2-base + linear head，在 ESD English 上做 5-class 情绪分类
2. **训练 STAA-Net generator**：冻结 SER，训练 Wave-U-Net 生成对抗扰动（untargeted，让 SER 判错即可）
3. **生成对抗音频**：用 generator 对测试集生成对抗样本
4. **评估 Voxtral**：将 clean 和对抗音频分别输入 Voxtral，比较情绪识别结果

| 项目 | 配置 |
|------|------|
| 数据集 | ESD English，3 speakers，5 emotions，525 测试样本 |
| Surrogate | wav2vec2-base + linear head，test acc = 47% |
| 攻击方法 | STAA-Net，ε=0.03（L∞），论文默认超参 |
| Victim | Voxtral-Mini-3B |

## 结果

### Voxtral Baseline（clean 音频）

Voxtral 在 ESD English 上整体准确率仅 **23%**，存在极强的 neutral 偏置（90/100 样本被预测为 neutral）：

| 情绪 | n | Voxtral 准确率 |
|------|---|----------------|
| neutral | 24 | 91.7% |
| angry | 22 | 4.5% |
| happy / surprise / sad | 54 | 0% |

### 迁移攻击效果

Generator 产生的扰动几乎为零（87% 样本 ΔL∞ < 0.001），99/100 样本 Voxtral 在 clean 和 adversarial 上输出完全一致。

| 指标 | 值 |
|------|------|
| Voxtral clean 与 adv 输出一致 | 99/100 |
| **真实攻击翻转率**（clean 判对 → adv 判错） | **0/23 = 0%** |

### 与白盒攻击对比

| | SER 迁移攻击 | 白盒攻击（直接优化 Voxtral） |
|---|---|---|
| 真实翻转率 | **0%** | **93.8%** |
| 扰动 ΔL∞ | ≈ 0 | ≤ 0.008 |

## 结论

**传统 SER 对抗攻击无法迁移到 ALLM。** SER 模型（wav2vec2 + linear head）的决策边界与 ALLM（Whisper encoder + LLM 自回归生成）的情绪判断路径完全不同，针对前者优化的扰动对后者无效。这证明了设计 ALLM-native 攻击方法的必要性。
