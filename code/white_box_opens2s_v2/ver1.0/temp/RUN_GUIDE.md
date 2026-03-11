# OpenS2S 白盒攻击实验运行指南

> 完整的实验执行步骤，从环境准备到结果分析

---

## 📋 目录

1. [环境准备](#1-环境准备)
2. [配置检查](#2-配置检查)
3. [样本准备](#3-样本准备)
4. [运行实验](#4-运行实验)
5. [结果查看](#5-结果查看)
6. [常见问题](#6-常见问题)
7. [高级用法](#7-高级用法)

---

## 1️⃣ 环境准备

### 1.1 检查 Python 环境

```bash
# 检查 Python 版本（需要 >= 3.9）
python3 --version

# 进入 codex 目录
cd /data1/lixiang/lx_code/white_box_v2/codex
```

### 1.2 安装依赖包

```bash
# 核心依赖
pip install torch torchaudio transformers
pip install numpy soundfile jiwer

# 如果使用 conda 环境
conda install pytorch torchaudio -c pytorch
pip install transformers soundfile jiwer
```

### 1.3 检查 OpenS2S 模型

```bash
# 检查模型路径是否存在
ls -lh /data1/lixiang/Opens2s/OpenS2S/models/OpenS2S

# 应该看到如下文件：
# - config.json
# - pytorch_model.bin (或 model.safetensors)
# - tokenizer_config.json
# - special_tokens_map.json
```

### 1.4 检查 GPU 可用性

```bash
# 检查 CUDA 是否可用
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
python3 -c "import torch; print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 检查 GPU 显存
nvidia-smi
```

**预期输出**：
```
CUDA available: True
GPU count: 1
GPU name: NVIDIA A100-SXM4-80GB (或你的 GPU 型号)
```

---

## 2️⃣ 配置检查

### 2.1 查看当前配置

```bash
# 查看配置文件
cat config.py
```

### 2.2 关键配置项说明

打开 `config.py`，检查以下配置：

```python
# 路径配置（最重要！）
opens2s_root: Path = Path("/data1/lixiang/Opens2s/OpenS2S")  # OpenS2S 代码根目录
model_path: Path = Path("/data1/lixiang/Opens2s/OpenS2S/models/OpenS2S")  # 模型路径
sample_list_path: Path = Path(__file__).resolve().parent / "sample_list.txt"  # 样本列表
results_dir: Path = Path(__file__).resolve().parent / "results"  # 结果保存目录

# 运行配置
device: str = "cuda"  # 使用 GPU（如果没有 GPU，改为 "cpu"）
seed: int = 1234  # 随机种子

# 攻击参数
epsilon: float = 0.008  # L∞ 扰动上界（0.008 ≈ 音频范围的 0.8%）
total_steps: int = 60  # 总优化步数
lr: float = 0.003  # 学习率

# 目标情绪
target_emotion: str = "happy"  # 目标情绪（可选：happy, sad, angry, neutral）
```

### 2.3 修改配置（如果需要）

**方法 1：直接编辑 `config.py`**
```bash
nano config.py  # 或使用 vim/code 编辑
```

**方法 2：在代码中临时修改**（不推荐，应该修改配置文件）

---

## 3️⃣ 样本准备

### 3.1 查看样本列表

```bash
# 查看当前样本列表
cat sample_list.txt
```

**格式说明**：
- 每行一个音频文件的**绝对路径**
- 以 `#` 开头的行是注释，会被忽略
- 空行会被忽略

**示例**：
```
# 情绪攻击测试样本
/data1/lixiang/OpenS2S_dataset/data/en_query_wav/Sad/adult/female/20683.wav
/data1/lixiang/OpenS2S_dataset/data/en_query_wav/Sad/adult/female/24190.wav
/data1/lixiang/OpenS2S_dataset/data/en_query_wav/Sad/adult/female/15822.wav
```

### 3.2 修改样本列表

**方法 1：手动编辑**
```bash
nano sample_list.txt
```

**方法 2：批量添加样本**
```bash
# 添加某个目录下的所有 .wav 文件
find /data1/lixiang/OpenS2S_dataset/data/en_query_wav/Sad/adult/female/ -name "*.wav" | head -10 > sample_list.txt

# 或者从多个情绪类别中采样
find /data1/lixiang/OpenS2S_dataset/data/en_query_wav/Sad/ -name "*.wav" | shuf | head -5 > sample_list.txt
find /data1/lixiang/OpenS2S_dataset/data/en_query_wav/Angry/ -name "*.wav" | shuf | head -5 >> sample_list.txt
```

### 3.3 验证样本文件存在

```bash
# 检查样本列表中的文件是否都存在
while read -r line; do
    # 跳过注释和空行
    [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue
    path=$(echo "$line" | awk '{print $1}')
    if [ ! -f "$path" ]; then
        echo "❌ 文件不存在: $path"
    else
        echo "✅ 文件存在: $path"
    fi
done < sample_list.txt
```

---

## 4️⃣ 运行实验

### 4.1 基础运行（单 GPU，所有样本）

```bash
# 指定 GPU 0
export CUDA_VISIBLE_DEVICES=0

# 运行攻击
python3 run_attack.py
```

**预期输出**：
```
[DEBUG] input_ids shape: torch.Size([1, 234])
[DEBUG] speech_values shape: torch.Size([1, 128, 3000])
[DEBUG] speech_mask shape: torch.Size([1, 3000])
Processing sample 00000_20683...
Step 0: L_total=2.345, L_emo=2.340, L_asr=0.005, L_per=0.000
Step 10: L_total=1.234, L_emo=1.230, L_asr=0.004, L_per=0.000
...
✅ Sample 00000_20683 completed
Emotion: clean=[sad, sad, sad] → adv=[happy, happy, happy] ✓
WER: 0.023 ✓
SNR: 42.3 dB
```

### 4.2 指定样本范围

```bash
# 只处理前 5 个样本（索引 0-4）
python3 run_attack.py --start_idx 0 --end_idx 5

# 处理第 10-20 个样本
python3 run_attack.py --start_idx 10 --end_idx 20
```

### 4.3 分片运行（多 GPU 并行）

假设你有 4 个 GPU，想并行处理：

**终端 1（GPU 0）：**
```bash
export CUDA_VISIBLE_DEVICES=0
python3 run_attack.py --shard_id 0 --num_shards 4
```

**终端 2（GPU 1）：**
```bash
export CUDA_VISIBLE_DEVICES=1
python3 run_attack.py --shard_id 1 --num_shards 4
```

**终端 3（GPU 2）：**
```bash
export CUDA_VISIBLE_DEVICES=2
python3 run_attack.py --shard_id 2 --num_shards 4
```

**终端 4（GPU 3）：**
```bash
export CUDA_VISIBLE_DEVICES=3
python3 run_attack.py --shard_id 3 --num_shards 4
```

**原理**：
- `--shard_id 0 --num_shards 4`：处理索引 % 4 == 0 的样本（0, 4, 8, ...）
- `--shard_id 1 --num_shards 4`：处理索引 % 4 == 1 的样本（1, 5, 9, ...）
- 以此类推

### 4.4 自定义结果目录

```bash
# 将结果保存到其他目录
python3 run_attack.py --results_dir ./results_exp1

# 使用不同的样本列表
python3 run_attack.py --sample_list ./sample_list_test.txt --results_dir ./results_test
```

### 4.5 后台运行（推荐长时间实验）

```bash
# 使用 nohup 后台运行，输出重定向到日志
nohup python3 run_attack.py > attack.log 2>&1 &

# 查看进程
ps aux | grep run_attack

# 实时查看日志
tail -f attack.log

# 停止实验
kill -9 <PID>
```

### 4.6 使用 tmux/screen（推荐）

```bash
# 创建新会话
tmux new -s attack_exp

# 在 tmux 中运行
export CUDA_VISIBLE_DEVICES=0
python3 run_attack.py

# 分离会话：按 Ctrl+B，然后按 D
# 重新连接：tmux attach -t attack_exp
# 查看所有会话：tmux ls
# 杀死会话：tmux kill-session -t attack_exp
```

---

## 5️⃣ 结果查看

### 5.1 查看结果文件

```bash
# 查看结果目录
ls -lh results/

# 应该看到：
# - 00000_20683.json      # 每个样本的详细结果
# - 00000_20683.wav       # 对抗音频
# - 00001_24190.json
# - 00001_24190.wav
# - summary.json          # 汇总统计
# - summary.csv           # 汇总统计（CSV 格式）
```

### 5.2 查看单个样本结果

```bash
# 查看 JSON 结果
cat results/00000_20683.json | jq .

# 或使用 Python 美化输出
python3 -c "import json; print(json.dumps(json.load(open('results/00000_20683.json')), indent=2))"
```

**JSON 结构**：
```json
{
  "sample_id": "00000_20683",
  "path": "/data1/lixiang/OpenS2S_dataset/data/en_query_wav/Sad/adult/female/20683.wav",
  "emo_pred_clean": ["sad", "sad", "sad"],
  "emo_pred_adv": ["happy", "happy", "happy"],
  "asr_text_clean": "I feel very down today",
  "asr_text_adv": "I feel very down today",
  "success_emo": true,
  "wer": 0.0,
  "delta_linf": 0.008,
  "delta_l2": 0.234,
  "snr_db": 42.3,
  "grad_norm_trace": [2.34, 1.23, 0.89, ...],
  "loss_trace": [
    {"step": 0, "total": 2.345, "emo": 2.340, "asr": 0.005, "per": 0.000},
    {"step": 1, "total": 2.123, "emo": 2.118, "asr": 0.005, "per": 0.000},
    ...
  ]
}
```

### 5.3 查看汇总结果

```bash
# 查看 JSON 汇总
cat results/summary.json

# 查看 CSV 汇总
cat results/summary.csv
```

**汇总指标**：
```json
{
  "num_samples": 10,
  "emo_success_rate": 0.90,        // 情绪攻击成功率
  "wer_le_0.0": 0.60,              // WER = 0.0 的比例（完美保持）
  "wer_le_0.05": 0.80,             // WER <= 0.05 的比例
  "joint_success_le_0.0": 0.55,    // 情绪成功 ∧ WER=0.0
  "joint_success_le_0.05": 0.75    // 情绪成功 ∧ WER<=0.05
}
```

### 5.4 播放对抗音频（如果有音频播放器）

```bash
# 使用 ffplay 播放
ffplay results/00000_20683.wav

# 或使用 aplay
aplay results/00000_20683.wav

# 比较原始音频和对抗音频
ffplay /data1/lixiang/OpenS2S_dataset/data/en_query_wav/Sad/adult/female/20683.wav
ffplay results/00000_20683.wav
```

### 5.5 分析损失曲线（Python）

```python
import json
import matplotlib.pyplot as plt

# 加载结果
with open('results/00000_20683.json') as f:
    data = json.load(f)

# 绘制损失曲线
loss_trace = data['loss_trace']
steps = [x['step'] for x in loss_trace]
emo_loss = [x['emo'] for x in loss_trace]
asr_loss = [x['asr'] for x in loss_trace]
per_loss = [x['per'] for x in loss_trace]

plt.figure(figsize=(10, 6))
plt.plot(steps, emo_loss, label='L_emo')
plt.plot(steps, asr_loss, label='L_asr')
plt.plot(steps, per_loss, label='L_per')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Trace')
plt.savefig('loss_trace.png')
```

---

## 6️⃣ 常见问题

### Q1: `RuntimeError: CUDA out of memory`

**原因**：GPU 显存不足

**解决方案**：
1. 减少批次处理（代码已经是 batch=1）
2. 启用梯度检查点（代码已启用）
3. 使用更小的模型或更少的 EoT 采样

```python
# 在 config.py 中修改
eot_samples: int = 1  # 从 3 减少到 1
```

4. 或者使用 CPU（较慢）：
```python
# config.py
device: str = "cpu"
```

### Q2: `FileNotFoundError: OpenS2S imports failed`

**原因**：OpenS2S 模块未找到

**解决方案**：
```bash
# 检查路径
ls /data1/lixiang/Opens2s/OpenS2S/src/modeling_omnispeech.py

# 确保 config.py 中路径正确
opens2s_root: Path = Path("/data1/lixiang/Opens2s/OpenS2S")
```

### Q3: 梯度为 0 或极小

**错误信息**：
```
RuntimeError: Grad norm too small; check gradient chain (Methodology §4.2).
```

**原因**：梯度链断裂

**检查**：
1. 确保音频预处理全程在 torch 中
2. 检查是否有 `.detach()` 操作
3. 检查模型是否在 eval 模式（应该是）

**临时解决**：
```python
# config.py 中放宽检查
grad_norm_min: float = 1e-10  # 从 1e-8 改为 1e-10
grad_norm_patience: int = 5   # 从 3 改为 5
```

### Q4: 攻击成功率很低

**可能原因**：
1. 步数太少
2. 学习率不合适
3. 权重比例不合理

**调试方案**：
```python
# config.py
total_steps: int = 100  # 增加到 100 步
lr: float = 0.005  # 尝试更大的学习率
lambda_emo: float = 2.0  # 增大情绪损失权重
```

### Q5: WER 太高（语义保持差）

**原因**：ASR 损失权重太小

**解决方案**：
```python
# config.py
lambda_asr_stage_b: float = 1e-1  # 从 1e-2 增大到 1e-1
```

### Q6: 音频文件读取失败

**错误信息**：
```
RuntimeError: Error loading audio file
```

**解决方案**：
```bash
# 检查文件是否损坏
ffmpeg -v error -i sample.wav -f null - 2>error.log
cat error.log

# 转换为标准格式
ffmpeg -i input.wav -ar 16000 -ac 1 -sample_fmt s16 output.wav
```

---

## 7️⃣ 高级用法

### 7.1 修改目标情绪

```python
# config.py
target_emotion: str = "angry"  # 改为 angry, sad, neutral
```

### 7.2 使用不同的 Prompt

```python
# config.py
emo_prompts: list[str] = field(
    default_factory=lambda: [
        "What emotion is expressed? Output one word: happy/sad/angry/neutral.",
        "Identify the emotion (one word only): happy, sad, angry, neutral.",
        # 添加更多等价 prompts
    ]
)
```

### 7.3 调整 EoT 变换

```python
# config.py
eot_samples: int = 3  # 增加采样次数
eot_max_shift: int = 320  # 增大时移范围（从 160 到 320）
eot_gain_min: float = 0.7  # 扩大增益范围
eot_gain_max: float = 1.3
eot_noise_std: float = 0.001  # 启用噪声
```

### 7.4 修改扰动约束

```python
# config.py
epsilon: float = 0.01  # 增大扰动上界（从 0.008 到 0.01）
```

### 7.5 调整两阶段策略

```python
# config.py
total_steps: int = 80
stage_a_steps: int = 30  # 增加阶段 A 的步数
lambda_asr_stage_a: float = 1e-5  # 阶段 A 更弱的约束
lambda_asr_stage_b: float = 5e-2  # 阶段 B 更强的约束
```

### 7.6 导出结果到 LaTeX 表格

```bash
# 使用 Python 脚本生成 LaTeX 表格
python3 << 'EOF'
import json

with open('results/summary.json') as f:
    data = json.load(f)

print("\\begin{table}[h]")
print("\\centering")
print("\\begin{tabular}{lc}")
print("\\hline")
print("Metric & Value \\\\")
print("\\hline")
print(f"Samples & {data['num_samples']} \\\\")
print(f"Emotion Success Rate & {data['emo_success_rate']:.2%} \\\\")
print(f"Joint Success (WER$\\leq$0.05) & {data['joint_success_le_0.05']:.2%} \\\\")
print("\\hline")
print("\\end{tabular}")
print("\\caption{Attack Results}")
print("\\end{table}")
EOF
```

### 7.7 批量实验脚本

```bash
#!/bin/bash
# batch_experiment.sh

EMOTIONS=("happy" "sad" "angry" "neutral")

for emo in "${EMOTIONS[@]}"; do
    echo "Running experiment for target emotion: $emo"

    # 修改 config.py 中的 target_emotion
    sed -i "s/target_emotion: str = .*/target_emotion: str = \"$emo\"/" config.py

    # 运行实验
    export CUDA_VISIBLE_DEVICES=0
    python3 run_attack.py --results_dir "./results_${emo}"

    echo "✅ Completed $emo"
done

echo "🎉 All experiments completed!"
```

运行：
```bash
chmod +x batch_experiment.sh
./batch_experiment.sh
```

---

## 📊 实验检查清单

在运行实验前，请确认：

- [ ] Python 版本 >= 3.9
- [ ] CUDA 可用（如果使用 GPU）
- [ ] OpenS2S 模型路径正确
- [ ] sample_list.txt 中的文件都存在
- [ ] config.py 中的路径配置正确
- [ ] results/ 目录有写入权限
- [ ] GPU 显存充足（建议 >= 24GB）

在运行完成后，检查：

- [ ] results/ 目录包含所有样本的 .json 和 .wav
- [ ] summary.json 存在且指标合理
- [ ] 没有样本报错或跳过
- [ ] 情绪攻击成功率 > 0.8
- [ ] 联合成功率（WER <= 0.05）> 0.6

---

## 📞 获取帮助

如果遇到问题：

1. 检查日志输出中的错误信息
2. 查看 `[DEBUG]` 输出了解执行细节
3. 检查 GPU 显存使用情况：`nvidia-smi`
4. 验证音频文件完整性
5. 参考 `methodology_math.md` 理解方法论

---

**祝实验顺利！** 🚀
