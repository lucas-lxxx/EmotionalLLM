# OPUS Workflow: Observation Section 全自动撰写工作流

> **创建日期**: 2026-02-26
> **目标**: 独立完成 Section 2 Observation 的完整终稿（大纲 → 实验 → 数据 → LaTeX → 审阅 → 编译）
> **产出目录**: 本地 `OPUS/`，服务器 `/data1/lixiang/OPUS`

---

## 目录结构

```
OPUS/
├── workflow.md              # 本文件：总工作流
├── paper/                   # LaTeX 论文产出
│   ├── observation_final.tex  # 最终 Observation 正文
│   └── main.tex               # 编译入口
├── experiments/             # 实验代码（上传服务器执行）
│   ├── exp_*.py               # 各实验脚本
│   └── utils/                 # 共用工具（模型加载、数据集等）
├── results/                 # 服务器回收的原始实验数据
├── figures/                 # 本地生成的论文图表
├── scripts/                 # 本地辅助脚本（可视化、数据处理）
├── reviews/                 # Reviewer 审阅报告
└── observation_outline.md   # Phase 1 产出：独立逻辑大纲
```

---

## 环境与约束

### 服务器信息

- **SSH**: `ssh -p 22326 lixiang@202.120.38.31`
- **SCP**: `scp -P 22326 <local> lixiang@202.120.38.31:<remote>`
- **工作目录**: `/data1/lixiang/OPUS`（新建）
- **模型路径**: `/data1/lixiang/Opens2s/`
- **数据集路径**: `/data1/lixiang/OpenS2S_dataset/TTS_modal_conflict/`
- **Python 环境**: `conda activate opens2s`
- **GPU**: 运行前用 `nvidia-smi` 查看空闲卡，用 `CUDA_VISIBLE_DEVICES=X` 指定

### 硬约束

- ❌ **禁止**修改 `/home/lixiang/` 下的任何内容
- ❌ **禁止**参考已有 `2OBSERVATION/observation.tex` 的具体文本
- ✅ **可以**参考 `observation_cc.md` 的缺口分析作为灵感来源
- ✅ 大纲和论文必须**完全独立**，质量高于已有版本
- ✅ 所有产出输出到本地 `OPUS/` 对应子目录

### 已有实验材料（参考用）

- `LATEST/white_box_final/audio内部机理1/`: Probe 实验报告 + 图表（dominance_curve, conflict_curves）
- `LATEST/white_box_final/audio内部机理2/`: Logit Lens + Activation Patching 报告 + 图表（margin, winrate, flip_rate, delta_logit 曲线）
- `LATEST/white_box_final/prompt&audio机理/`: Prompt-Audio 冲突实验报告 + 图表（Logit Lens 差分, PatchText vs PatchAudio）
- `LATEST/white_box_final/白盒对抗样本方法论/`: 白盒攻击方法论 + 结果

### 已有实验代码模式（参考用）

- `code/modal_conflict/`: Probe 实验，YAML 配置 + `scripts/run_experiment.py` 入口
- `code/logit_lens/`: Logit Lens，YAML 配置 + `scripts/` 入口
- `code/activation_patching/`: Activation Patching，YAML 配置 + `scripts/run_patching.py` 入口
- 共用模式：`opens2s_io.py` 适配层、`find_audio_span()` 定位、`restricted 5-way` 评估、`GroupKFold` 交叉验证

---

## Phase 1: 独立逻辑大纲

### 目标

从零构建 Observation 的完整叙事逻辑，输出 `OPUS/observation_outline.md`。

### 输入

1. `CONTEXT.md` — 研究框架与约束
2. `observation_cc.md` — 缺口分析（仅作灵感，不复制结构）
3. `LATEST/white_box_final/` — 已有实验数据与图表
4. `LATEST/白盒讲稿.md` — 汇报讲稿
5. 模型自身的深度思考与学术判断

### 执行步骤

1. **阅读所有输入材料**，建立对已有证据链的完整理解
2. **独立构思叙事主线**：
   - 保持 CONTEXT.md 的两节框架（2.1 音频内 + 2.2 跨模态）
   - 但叙事角度、段落逻辑、证据呈现顺序完全独立设计
3. **评估已有证据的充分性**，识别：
   - 哪些 claim 有充分数据支撑，可直接写入
   - 哪些 claim 需要补充实验才能成立
   - 哪些新实验可以显著增强论文质量
4. **设计新增实验清单**（模型自主决策，目标是终稿完善）：
   - 每个实验说明：目的、方法、预期结果、对论文的贡献
   - 评估可行性（服务器资源、代码复杂度、时间成本）
   - 筛选出性价比最高的实验子集
5. **输出完整大纲**到 `OPUS/observation_outline.md`：
   - 每段的核心论点
   - 证据来源（已有 or 新实验）
   - 图表规划（编号、内容、形式）
   - 新增实验清单（含优先级排序）

### 质量要求

- 叙事逻辑链完整、无断裂
- 每条 claim 标注证据等级（已验证 / 待验证 / 假说）
- 与 Section 3 Threat Model 的过渡自然
- 比 `observation_cc.md` 更精炼、更有深度

---

## Phase 2: 编写实验代码

### 目标

为 Phase 1 确定的新增实验编写可执行代码，输出到 `OPUS/experiments/`。

### 执行步骤

1. **分析已有代码结构**：
   - 读取 `code/modal_conflict/src/` 的模型加载、数据集、特征提取逻辑
   - 读取 `code/activation_patching/src/` 的 hook 机制、patch 逻辑
   - 读取 `code/logit_lens/src/` 的 Logit Lens 投影逻辑
   - 提取可复用的工具函数到 `OPUS/experiments/utils/`
2. **为每个新实验编写独立脚本**：
   - 脚本命名: `exp_<实验名>.py`
   - 每个脚本自包含：参数解析、模型加载、实验逻辑、结果保存
   - 结果保存格式：CSV + JSON + PNG
   - 配置通过命令行参数或 YAML 文件
3. **编写配置文件**：
   - `OPUS/experiments/config.yaml`：统一的路径配置
   - 模型路径: `/data1/lixiang/Opens2s/`
   - 数据集路径: `/data1/lixiang/OpenS2S_dataset/TTS_modal_conflict/`
4. **编写本地可视化脚本**到 `OPUS/scripts/`：
   - 读取 `OPUS/results/` 中的实验数据
   - 生成论文级图表到 `OPUS/figures/`

### 代码规范

- 复用已有的 `opens2s_io.py` 输入构造逻辑
- 复用 `find_audio_span()` 定位逻辑
- 所有 forward 调用 `use_cache=False`
- `model.eval()` + `torch.inference_mode()`
- 及时释放显存（`del` + `torch.cuda.empty_cache()`）
- 结果保存路径使用相对路径，便于本地/服务器切换

---

## Phase 3: 远程执行实验 + 数据回收

### 目标

在服务器上执行实验，回收数据到本地。

### 执行步骤

1. **准备服务器环境**：
   ```bash
   ssh -p 22326 lixiang@202.120.38.31
   mkdir -p /data1/lixiang/OPUS/experiments
   mkdir -p /data1/lixiang/OPUS/results
   ```

2. **上传实验代码**：
   ```bash
   scp -P 22326 -r OPUS/experiments/ lixiang@202.120.38.31:/data1/lixiang/OPUS/
   ```

3. **检查 GPU 可用性**：
   ```bash
   ssh -p 22326 lixiang@202.120.38.31 "nvidia-smi"
   ```

4. **逐个执行实验**（在服务器上）：
   ```bash
   cd /data1/lixiang/OPUS/experiments
   conda activate opens2s
   CUDA_VISIBLE_DEVICES=X python exp_<name>.py --config config.yaml
   ```
   - 对每个实验：检查输出是否正确，确认无报错
   - 长时间实验考虑用 `nohup` 或 `tmux`

5. **回收结果到本地**：
   ```bash
   scp -P 22326 -r lixiang@202.120.38.31:/data1/lixiang/OPUS/results/ OPUS/results/
   ```

6. **本地生成图表**：
   ```bash
   python OPUS/scripts/generate_figures.py
   ```
   - 图表保存到 `OPUS/figures/`
   - 图表格式：PDF 或高分辨率 PNG（300 dpi）
   - 风格统一：字体、颜色、标注风格与已有图表一致

### SSH 命令模式

由于无法直接使用 SSH MCP，通过 `run_command` 执行：
```bash
ssh -p 22326 lixiang@202.120.38.31 "<远程命令>"
```

对于交互式操作或长时间任务：
```bash
ssh -p 22326 lixiang@202.120.38.31 "cd /data1/lixiang/OPUS && nohup bash -c 'conda activate opens2s && CUDA_VISIBLE_DEVICES=X python experiments/exp_<name>.py' > logs/<name>.log 2>&1 &"
```

---

## Phase 4: 综合撰写 LaTeX 论文（中文版）

### 目标

基于大纲和所有实验数据，撰写完整的 Observation LaTeX 正文，输出到 `OPUS/paper/observation_final.tex`。

### 输入

1. `OPUS/observation_outline.md` — Phase 1 大纲
2. `LATEST/white_box_final/` — 已有实验数据
3. `OPUS/results/` + `OPUS/figures/` — 新增实验数据与图表
4. `CONTEXT.md` — 写作原则与约束

### 执行步骤

1. **按大纲逐段撰写**：
   - 严格遵循大纲的逻辑链
   - 每段引用具体实验数据和图表
   - 中文正文，英文术语/标题/公式
2. **写作规范**：
   - 每条 claim 有实验证据支撑
   - 区分"已证实"、"推测"、"假说"
   - 不包含攻击成功率等属于 Section 5 的数据
   - 攻击方法论的引出基于机理发现
   - 过渡到 Section 3 自然流畅
3. **图表引用**：
   - 使用 `\ref{fig:xxx}` 交叉引用
   - 图表说明简洁准确
   - 所有图表路径指向 `figures/` 目录
4. **编译入口**：
   - 创建 `OPUS/paper/main.tex`（XeLaTeX + ctex）
   - 确保可独立编译

---

## Phase 5: Reviewer 审阅循环

### 目标

模拟顶会 Reviewer 审阅，发现并修复问题。

### Reviewer Prompt 设计

Reviewer 以独立视角审阅，关注以下维度：

1. **逻辑完整性** (★★★★★)
   - 论点之间是否有逻辑断裂？
   - 每条 claim 是否有充分证据？
   - 叙事链是否连贯？

2. **实验充分性** (★★★★★)
   - 是否存在关键实验缺失？
   - 对照实验是否充分？
   - 样本量是否足够支撑统计结论？

3. **Claim 边界** (★★★★)
   - 结论是否过度泛化？
   - 条件限定是否充分？
   - "已证实" vs "假说" 是否区分清楚？

4. **统计严谨性** (★★★★)
   - 置信区间是否报告？
   - 多重比较是否校正？
   - 效应量是否报告？

5. **写作质量** (★★★)
   - 术语使用是否一致？
   - 行文是否简洁？
   - 图表是否清晰、信息量充足？

### 执行步骤

1. **审阅**：以 Reviewer 身份阅读 `OPUS/paper/observation_final.tex`
2. **输出审阅报告**到 `OPUS/reviews/review_round_N.md`：
   - 按维度打分（1-10）
   - 列出具体问题（Major / Minor）
   - 给出修改建议
3. **决策**：
   - 若需补充实验 → 回到 Phase 2-3
   - 若仅需文本修改 → 直接修改 Phase 4 输出
   - 若质量达标 → 进入 Phase 6
4. **循环**直到 Reviewer 评分全部 ≥ 7/10

---

## Phase 6: 最终编译

### 目标

输出可正确编译的 LaTeX 终稿。

### 执行步骤

1. **整合所有文件**：
   - `OPUS/paper/main.tex` — 编译入口
   - `OPUS/paper/observation_final.tex` — 正文（`\input{}`）
   - `OPUS/figures/` — 所有图表
2. **编译测试**：
   - 使用 XeLaTeX + ctex
   - 确认无编译错误
   - 确认图表正确显示
   - 确认交叉引用正确
3. **最终检查**：
   - 所有 `\textcolor{red}{...}` 标注已清除
   - 无遗留的 TODO / FIXME
   - 图表编号连续
   - 参考文献完整

### 最终产出

- `OPUS/paper/main.tex` — 可编译的完整文档
- `OPUS/paper/observation_final.tex` — Observation 正文
- `OPUS/figures/` — 所有论文图表
- `OPUS/reviews/` — 审阅记录

---

## 执行顺序与依赖

```
Phase 1 (大纲)
    ↓
Phase 2 (代码) ← 依赖 Phase 1 的实验清单
    ↓
Phase 3 (执行) ← 依赖 Phase 2 的代码
    ↓
Phase 4 (撰写) ← 依赖 Phase 1 大纲 + Phase 3 数据
    ↓
Phase 5 (审阅) ← 依赖 Phase 4 文本
    ↓ ↑ (循环: 若需补实验回到 Phase 2-3, 若需改文本回到 Phase 4)
Phase 6 (编译) ← 审阅通过后
```

---

## 检查点

每个 Phase 完成后，暂停并与用户确认：

- **Phase 1 后**: 确认大纲方向、实验清单
- **Phase 2 后**: 确认代码逻辑（可选）
- **Phase 3 后**: 确认实验结果合理性
- **Phase 4 后**: 确认论文内容
- **Phase 5 后**: 确认审阅意见的处理方案
- **Phase 6 后**: 确认最终稿

