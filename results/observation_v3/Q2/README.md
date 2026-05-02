
# Observation 3: Emotion Misperception → Downstream Reasoning Impact

## 目标

验证情绪误感知的实质危害性：当模型对说话人情绪的判断被改变时，其下游回复行为（语气、策略、内容）会发生系统性偏移，甚至产生幻觉。

本 Observation 不依赖任何攻击方法，仅通过文本 Prompt 操纵实现情绪误导，属于 threat justification——为后续攻击方法提供"为什么值得攻击"的动机论证。

## 核心问题（Q2）

> 如果模型的情绪感知被翻转，会产生什么后果？——情绪不只是一个标签，它会传导到模型的实际回复行为中。

## 实验模型

Voxtral-Mini-3B-2507（本地部署，源码位于 `C:\Users\potte\Desktop\research\emotional LLM\model\Voxtral-Mini-3B-2507`）

## 实验设计

对同一段音频，用不同的文本 Prompt 引导模型做出不同的情绪感知，对比回复差异。

| 条件     | 输入音频 | Prompt 情绪引导                | 作用         |
| -------- | -------- | ------------------------------ | ------------ |
| Truthful | 原始音频 | 正确描述说话人情绪（或不描述） | Baseline     |
| Misled   | 同一音频 | 显式注入错误的目标情绪         | 观察回复偏移 |

核心对比：同一段音频、同一个模型，仅改变 Prompt 中的情绪描述 → 回复发生了什么变化？

## 执行步骤

### Step 0: 环境与数据摸底

* 阅读 Voxtral-Mini-3B-2507 源码，搞清推理接口（输入格式、Prompt 模板机制、如何传入音频、输出格式）
* 确认 ESD 数据集中可用的干净音频样本（路径、情绪标签、语言）
* **交付物** ：推理接口摘要、可用音频样本统计

### Step 1: Prompt 设计与样本选取

* 设计两组 Prompt：Truthful（正确/中立情绪描述）vs Misled（注入错误情绪）
* 从 ESD 数据集选取 15 条音频，覆盖不同原始情绪和误导方向（优先跨 valence 对）
* **交付物** ：Prompt 模板对、15 条样本列表（含音频路径、真实情绪、误导目标情绪）

### Step 2: Truthful vs Misled 推理对比

* 对每条样本分别用两组 Prompt 做推理，收集回复
* **交付物** ：每条样本的 Truthful 回复 + Misled 回复（JSON/CSV）

### Step 3: LLM Judge 自动化评估

* 对每对回复做结构化标注：语气偏移、策略变化、内容相关性、幻觉检测
* **交付物** ：带评估分数的扩展表格

### Step 4: 汇总与可视化

* 统计各维度分布、偏移方向与误导情绪的一致率、幻觉触发率
* 挑选 3 个最典型 case 作为论文 figure 候选
* **交付物** ：统计摘要 + 可视化图 + showcase cases

## 文件索引

| 路径 | 内容 |
| --- | --- |
| `observation_v3/voxtral_info.md` | **Step 0 交付物**：Voxtral-Mini-3B-2507 推理接口摘要（加载方式、输入格式、调用示例、输出格式、与本实验的对接要点） |
| `observation_v3/experiment/run_inference.py` | **Step 1-2 推理脚本**：Aligned vs Conflict 情绪 Prompt 推理对比，支持 `--dry-run` 模式 |
| `observation_v3/experiment/config_demo.json` | 示例配置文件（3 条样本，路径需按实际 ESD 目录调整） |
| `observation_v3/experiment/config_15samples.json` | 正式推理配置（15 条样本，覆盖 5 情绪 × 跨 valence 冲突对） |
| `observation_v3/experiment/run_evaluation.py` | **Step 3 评估脚本**：DeepSeek V3.2 LLM Judge，三维度评估（Faithfulness / Empathy / Relevance） |
| `observation_v3/experiment/eval_config.json` | 评估配置（API 地址、模型、文件路径） |
| `observation_v3/experiment/result/ob3_results.json` | Voxtral 推理原始结果 |
| `observation_v3/experiment/result/ob3_eval_results.json` | LLM Judge 评估结果（含分数和理由） |
| `observation_v3/experiment/result/ob3_analysis_report.md` | **Step 3 交付物**：完整分析报告（统计、拆解、发现、论文支撑） |

## 备注

* 当前阶段为框架 demo（15 条样本），验证 pattern 后再扩量
* 代码在本地编写，上传至服务器用 GPU 执行推理
* 关键优势：不依赖攻击方法，论文叙事顺序合法（Observation → Threat Model → Method）
