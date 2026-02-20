# OpenS2S White-Box Attack Experiment Report (Test3 N=100)

## 实验概述
- 目标: 在 OpenS2S 上进行白盒对抗攻击, 将 "Sad" 情绪翻转为 "Happy"。
- 规模: 100 条 Sad 情绪音频样本。
- 位置: `/data1/lixiang/lx_code/white_box_v1/test3`
- 结果目录: `/data1/lixiang/lx_code/white_box_v1/test3/results_n100`

## 环境与模型
- OpenS2S 模型路径: `/data1/lixiang/Opens2s/OpenS2S/models/OpenS2S`
- 情绪分类器: `checkpoints/sad_happy_classifier.pt` (Sad vs Happy)
- 设备: `cuda:0`
- 情绪特征提取层: `layer_06`, `layer_16`, `layer_25`

## 样本与输入
- 样本列表: `/data1/lixiang/lx_code/white_box_v1/test3/sad_samples_100.txt`
- Prompt:
  "What is the emotion of this audio? Please answer with only one word: the emotion label (happy, sad, angry, or neutral)."

## 攻击参数 (Aggressive)
- `EPSILON = 0.008`
- `STEPS = 80`
- `LAMBDA_EMO = 30.0`
- `LAMBDA_SEM = 0.001`
- `LAMBDA_PER = 0.00001`

## 实验流程
- 对每个样本依次执行:
  1) Clean inference
  2) PGD 攻击 (Sad -> Happy)
  3) Attack inference
- 输出保存:
  - 对抗音频: `results_n100/audio/adv/`
  - 文本输出: `results_n100/text/`
  - 配置: `results_n100/config.json`
  - 结果总表: `results_n100/results.json`

## 结果概览
- 总样本数: 100
- 成功数: 2
- 成功率: 2.0%
- 失败(错误): 0
- 总耗时: 21.4 分钟
- 单样本平均耗时: 12.84 秒

### 平均指标
- L∞: 0.008000
- L2: 2.627
- SNR: 12.61 dB
- Attack time: 11.80 秒

## 备注与观察
- 攻击在所有样本上完成并保存结果, 未出现未捕获的错误。
- 多数样本攻击后仍保持 sad/neutral 输出, 表明当前强攻击参数仍难稳定翻转。
- 攻击过程中多次出现梯度范数过小的警告 (grad_norm too small), 可能影响优化效率。

## 可复现信息
- 脚本: `/data1/lixiang/lx_code/white_box_v1/test3/run_test_n100.py`
- 日志: `/data1/lixiang/lx_code/white_box_v1/test3/run_test_n100.log`
- 结果: `/data1/lixiang/lx_code/white_box_v1/test3/results_n100/results.json`
