# White-Box Attack Package - 发送说明

## 包信息

**文件名**: `white_box_v2.zip`
**位置**: `/data1/lixiang/lx_code/white_box_v2/codex/white_box_v2.zip`
**大小**: 25 KB
**创建时间**: 2026-01-08

## 包含内容

### 核心代码 (5个文件)
1. `attack_core.py` - 核心攻击算法（PGD + EoT + 两阶段优化）
2. `opens2s_io.py` - OpenS2S模型加载和I/O接口
3. `config.py` - 配置文件（所有参数）
4. `run_attack.py` - 主执行脚本
5. `eval_metrics.py` - 评估指标计算

### 文档 (3个文件)
1. `README.md` - 快速开始指南
2. `RUN_GUIDE.md` - 详细使用说明（7个章节）
3. `FINDINGS.md` - 实验发现和问题分析报告

### 辅助文件 (2个文件)
1. `sample_list.txt` - 样本列表示例
2. `quick_start.sh` - 环境检查脚本

## 重要提醒

### 1. 关键发现（务必阅读 FINDINGS.md）

经过系统性测试，我们发现：

- **OpenS2S模型不适合情绪分类任务**
  - 50个sad样本测试，0%识别为sad
  - 5个angry样本测试，0%识别为angry
  - 系统性偏向输出"neutral"

- **ASR功能也存在问题**
  - 无法转录实际语音内容
  - 只能生成模板化回应

- **根本原因**
  - 模型是对话助手，不是分类/分析工具
  - 特征提取存在问题（采样率不匹配）
  - 训练数据和任务目标不匹配

### 2. 使用建议

**给协作者的建议**：

1. **先阅读 FINDINGS.md**
   - 了解模型的限制
   - 理解为什么实验结果不理想
   - 避免重复相同的错误

2. **考虑更换模型**
   - 使用专门的情绪识别模型（如Wav2Vec2-emotion）
   - 使用专门的ASR模型（如Whisper）
   - 不要使用通用对话模型做分类任务

3. **如果继续使用OpenS2S**
   - 调整研究方向（攻击对话生成而非分类）
   - 降低期望（接受baseline性能差）
   - 关注方法论而非结果

### 3. 配置要求

- Python 3.10+
- PyTorch 2.0+ with CUDA
- GPU: 至少40GB显存（RTX 6000或更高）
- OpenS2S完整仓库和模型权重

### 4. 快速开始

```bash
# 解压
unzip white_box_v2.zip
cd white_box_v2

# 检查环境
bash quick_start.sh

# 修改配置
vim config.py  # 修改路径和GPU设置

# 运行攻击
/path/to/OpenS2S/venv/bin/python3 run_attack.py
```

## 发送清单

发送给协作者时，请确保：

- [x] 附上 `white_box_v2.zip` 文件
- [x] 提醒阅读 `FINDINGS.md`（非常重要！）
- [x] 说明OpenS2S的限制
- [x] 建议考虑更换模型
- [x] 提供配置要求和GPU需求

## 补充说明

### 实验结果摘要

| 配置 | 情绪攻击成功率 | WER=0.0 | 主要问题 |
|------|--------------|---------|---------|
| 24kHz原始 | 40% | 0% | 特征不匹配 + 模型能力不足 |
| 16kHz重采样 | 100% | 80% | ASR baseline错误（输出中文） |
| 中文ESD | 0% | 0% | 模型完全无法工作 |

### 技术细节

- **攻击方法**: PGD + EoT + 两阶段优化
- **损失函数**: L_emo + L_asr + L_per
- **约束**: L∞ = 0.008
- **优化步数**: 60步（Stage A: 20步，Stage B: 40步）

### 联系方式

如有问题，请联系原作者。

---

**创建日期**: 2026-01-08
**版本**: v2.0
**状态**: 已完成测试和分析
