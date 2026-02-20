# OpenS2S 白盒攻击实验代码

端到端的情绪对抗攻击实现，完全符合方法论要求。

## 🚀 快速开始

### 方式 1：一键启动（推荐）

```bash
cd /data1/lixiang/lx_code/white_box_v2/codex
./quick_start.sh
```

脚本会自动检查环境并引导你完成运行。

### 方式 2：手动运行

```bash
# 1. 指定 GPU
export CUDA_VISIBLE_DEVICES=0

# 2. 运行攻击（处理所有样本）
python3 run_attack.py

# 3. 或仅测试前 3 个样本
python3 run_attack.py --start_idx 0 --end_idx 3
```

## 📁 文件说明

| 文件 | 功能 |
|------|------|
| `run_attack.py` | 主入口脚本 |
| `config.py` | 所有配置参数 |
| `attack_core.py` | 攻击核心（损失函数、优化） |
| `opens2s_io.py` | OpenS2S 模型接口 |
| `eval_metrics.py` | 评估指标计算 |
| `sample_list.txt` | 样本列表 |
| `RUN_GUIDE.md` | **详细运行指南** ⭐ |
| `quick_start.sh` | 一键启动脚本 |

## 🔧 快速配置

编辑 `config.py` 修改以下常用参数：

```python
# 目标情绪
target_emotion: str = "happy"  # 可选: happy, sad, angry, neutral

# 攻击强度
epsilon: float = 0.008  # L∞ 扰动上界
total_steps: int = 60  # 优化步数

# GPU 设置
device: str = "cuda"  # 或 "cpu"
```

## 📊 查看结果

```bash
# 查看结果文件
ls results/

# 查看汇总统计
cat results/summary.json

# 输出示例：
# {
#   "num_samples": 10,
#   "emo_success_rate": 0.90,
#   "joint_success_le_0.05": 0.75
# }
```

## 🎯 常用命令

```bash
# 处理特定范围样本
python3 run_attack.py --start_idx 0 --end_idx 10

# 自定义结果目录
python3 run_attack.py --results_dir ./my_results

# 使用不同样本列表
python3 run_attack.py --sample_list ./my_samples.txt

# 多 GPU 并行（4 个 GPU）
export CUDA_VISIBLE_DEVICES=0 && python3 run_attack.py --shard_id 0 --num_shards 4 &
export CUDA_VISIBLE_DEVICES=1 && python3 run_attack.py --shard_id 1 --num_shards 4 &
export CUDA_VISIBLE_DEVICES=2 && python3 run_attack.py --shard_id 2 --num_shards 4 &
export CUDA_VISIBLE_DEVICES=3 && python3 run_attack.py --shard_id 3 --num_shards 4 &
```

## 📖 详细文档

- **运行指南**：`RUN_GUIDE.md` - 完整的环境配置、运行、调试指南
- **方法论数学**：`../methodology_math.md` - 完整的数学推导
- **方法论检查**：`../metho_check.md` - 代码审查清单

## ✅ 环境要求

- Python >= 3.9
- PyTorch >= 2.0
- CUDA 11.8+ (可选，但强烈推荐)
- GPU 显存 >= 24GB (推荐)

## 🐛 常见问题

### CUDA OOM (显存不足)
```python
# config.py
eot_samples: int = 1  # 减少 EoT 采样
```

### 梯度为 0
```python
# config.py
grad_norm_min: float = 1e-10  # 放宽阈值
```

### 攻击成功率低
```python
# config.py
total_steps: int = 100  # 增加步数
lr: float = 0.005  # 增大学习率
```

更多问题请查看 `RUN_GUIDE.md` 第 6 节。

## 📞 获取帮助

1. 查看 `RUN_GUIDE.md` 获取详细指南
2. 检查 `[DEBUG]` 输出了解执行细节
3. 运行 `nvidia-smi` 检查 GPU 状态

---

**祝实验顺利！** 🚀
