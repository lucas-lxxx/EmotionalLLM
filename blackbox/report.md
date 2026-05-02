# 黑盒迁移攻击最终实验报告

> 更新时间：2026-04-09 16:37:30
> 结构化 summary 完成度：32/60
> 主表目标模型：Gemini 2.5 Flash, Gemini 2.5 Pro, OpenAI gpt-audio, Qwen3-Omni-Flash, Qwen-Omni-Turbo

## 1. 实验范围

| Surrogate | Language | Selected samples | Source directory |
| --- | --- | --- | --- |
| Voxtral EN | EN | 914 | C:\Users\potte\Desktop\research\emotional LLM\code\white_box_voxtral\result\Voxtral_EN |
| Voxtral CN | CN | 962 | C:\Users\potte\Desktop\research\emotional LLM\code\white_box_voxtral\result\Voxtral_CN |
| OpenS2S EN | EN | 944 | C:\Users\potte\Desktop\research\emotional LLM\code\white_box_opens2s_v2\result\blackbox\EN |
| OpenS2S CN | CN | 774 | C:\Users\potte\Desktop\research\emotional LLM\code\white_box_opens2s_v2\result\blackbox\CN |

| Target | Client | Status |
| --- | --- | --- |
| Gemini 2.5 Flash | gemini | ready |
| Gemini 2.5 Pro | gemini | ready |
| OpenAI gpt-audio | gpt4o | pending key |
| Qwen3-Omni-Flash | qwen | ready |
| Qwen-Omni-Turbo | qwen | ready |

## 2. Demo 与最终批量结果对齐

| Model | Demo n | Demo ASR | Final n | Final ASR | Delta |
| --- | --- | --- | --- | --- | --- |
| Gemini 2.5 Flash | 48 | 12.50% | 914 | 7.88% | -4.62% |
| Qwen3-Omni-Flash | 48 | 27.08% | 914 | 19.69% | -7.39% |
| OpenAI gpt-audio |  |  |  |  | 待 OpenAI key |

## 3. 最终批量主结果

| Surrogate | Gemini 2.5 Flash | Gemini 2.5 Pro | OpenAI gpt-audio | Qwen3-Omni-Flash | Qwen-Omni-Turbo |
| --- | --- | --- | --- | --- | --- |
| Voxtral EN | 7.88% | 12.47% |  | 19.69% | 33.15% |
| Voxtral CN | 9.36% | 0.94% |  | 30.98% | 34.20% |
| OpenS2S EN | 5.93% | 0.00% |  | 14.09% | 22.78% |
| OpenS2S CN | 10.72% | 0.00% |  | 32.30% | 37.34% |
| Average | 8.47% | 3.35% |  | 24.27% | 31.87% |

## 4. 运行覆盖率

| Surrogate | Target | Planned n | Adv n | Clean n | Noise n |
| --- | --- | --- | --- | --- | --- |
| Voxtral EN | Gemini 2.5 Flash | 914 | 914 | 914 |  |
| Voxtral EN | Gemini 2.5 Pro | 914 | 914 | 914 |  |
| Voxtral EN | OpenAI gpt-audio | 914 |  |  |  |
| Voxtral EN | Qwen3-Omni-Flash | 914 | 914 | 914 |  |
| Voxtral EN | Qwen-Omni-Turbo | 914 | 914 | 914 |  |
| Voxtral CN | Gemini 2.5 Flash | 962 | 962 | 962 |  |
| Voxtral CN | Gemini 2.5 Pro | 962 | 962 | 962 |  |
| Voxtral CN | OpenAI gpt-audio | 962 |  |  |  |
| Voxtral CN | Qwen3-Omni-Flash | 962 | 962 | 962 |  |
| Voxtral CN | Qwen-Omni-Turbo | 962 | 962 | 962 |  |
| OpenS2S EN | Gemini 2.5 Flash | 944 | 944 | 944 |  |
| OpenS2S EN | Gemini 2.5 Pro | 944 | 944 | 944 |  |
| OpenS2S EN | OpenAI gpt-audio | 944 |  |  |  |
| OpenS2S EN | Qwen3-Omni-Flash | 944 | 944 | 944 |  |
| OpenS2S EN | Qwen-Omni-Turbo | 944 | 944 | 944 |  |
| OpenS2S CN | Gemini 2.5 Flash | 774 | 774 | 774 |  |
| OpenS2S CN | Gemini 2.5 Pro | 774 | 774 | 774 |  |
| OpenS2S CN | OpenAI gpt-audio | 774 |  |  |  |
| OpenS2S CN | Qwen3-Omni-Flash | 774 | 774 | 774 |  |
| OpenS2S CN | Qwen-Omni-Turbo | 774 | 774 | 774 |  |

## 5. Baseline 对比

| Surrogate | Target | Clean acc | Noise acc | Adv ASR | Clean target rate | Noise target rate |
| --- | --- | --- | --- | --- | --- | --- |
| OpenS2S CN | Gemini 2.5 Flash | 8.91% |  | 10.72% | 6.20% |  |
| OpenS2S CN | Gemini 2.5 Pro | 6.85% |  | 0.00% | 4.26% |  |
| OpenS2S CN | Qwen3-Omni-Flash | 5.56% |  | 32.30% | 7.49% |  |
| OpenS2S CN | Qwen-Omni-Turbo | 4.91% |  | 37.34% | 7.62% |  |
| OpenS2S EN | Gemini 2.5 Flash | 4.77% |  | 5.93% | 2.12% |  |
| OpenS2S EN | Gemini 2.5 Pro | 4.13% |  | 0.00% | 2.65% |  |
| OpenS2S EN | Qwen3-Omni-Flash | 29.98% |  | 14.09% | 8.37% |  |
| OpenS2S EN | Qwen-Omni-Turbo | 34.11% |  | 22.78% | 7.94% |  |
| Voxtral CN | Gemini 2.5 Flash | 0.00% |  | 9.36% | 0.00% |  |
| Voxtral CN | Gemini 2.5 Pro | 0.00% |  | 0.94% | 0.10% |  |
| Voxtral CN | Qwen3-Omni-Flash | 24.01% |  | 30.98% | 25.57% |  |
| Voxtral CN | Qwen-Omni-Turbo | 46.88% |  | 34.20% | 18.19% |  |
| Voxtral EN | Gemini 2.5 Flash | 4.27% |  | 7.88% | 0.98% |  |
| Voxtral EN | Gemini 2.5 Pro | 0.00% |  | 12.47% | 0.00% |  |
| Voxtral EN | Qwen3-Omni-Flash | 29.32% |  | 19.69% | 8.86% |  |
| Voxtral EN | Qwen-Omni-Turbo | 33.26% |  | 33.15% | 8.64% |  |

## 6. Per-Emotion Transfer ASR

| Surrogate | Target | Angry | Sad | Neutral | Surprise |
| --- | --- | --- | --- | --- | --- |
| OpenS2S CN | Gemini 2.5 Flash | 11.86% | 8.63% | 5.08% | 17.24% |
| OpenS2S CN | Gemini 2.5 Pro | 0.00% | 0.00% | 0.00% | 0.00% |
| OpenS2S CN | Qwen3-Omni-Flash | 30.51% | 32.99% | 23.35% | 41.87% |
| OpenS2S CN | Qwen-Omni-Turbo | 29.38% | 31.98% | 28.43% | 58.13% |
| OpenS2S EN | Gemini 2.5 Flash | 7.52% | 3.72% | 2.09% | 10.55% |
| OpenS2S EN | Gemini 2.5 Pro | 0.00% | 0.00% | 0.00% | 0.00% |
| OpenS2S EN | Qwen3-Omni-Flash | 16.37% | 14.46% | 12.13% | 13.50% |
| OpenS2S EN | Qwen-Omni-Turbo | 23.89% | 17.77% | 17.57% | 32.07% |
| Voxtral CN | Gemini 2.5 Flash | 10.97% | 6.61% | 5.44% | 14.34% |
| Voxtral CN | Gemini 2.5 Pro | 3.80% | 0.00% | 0.00% | 0.00% |
| Voxtral CN | Qwen3-Omni-Flash | 26.58% | 30.58% | 25.52% | 40.98% |
| Voxtral CN | Qwen-Omni-Turbo | 32.91% | 22.73% | 25.94% | 54.92% |
| Voxtral EN | Gemini 2.5 Flash | 8.48% | 5.04% | 5.88% | 12.12% |
| Voxtral EN | Gemini 2.5 Pro | 37.95% | 0.00% | 13.12% | 0.00% |
| Voxtral EN | Qwen3-Omni-Flash | 22.32% | 20.59% | 15.84% | 19.91% |
| Voxtral EN | Qwen-Omni-Turbo | 35.71% | 30.67% | 27.15% | 38.96% |

## 7. 语言对比

| Surrogate family | Target | EN ASR | CN ASR | CN-EN |
| --- | --- | --- | --- | --- |
| voxtral | Gemini 2.5 Flash | 7.88% | 9.36% | 1.48% |
| voxtral | Gemini 2.5 Pro | 12.47% | 0.94% | -11.53% |
| voxtral | OpenAI gpt-audio |  |  |  |
| voxtral | Qwen3-Omni-Flash | 19.69% | 30.98% | 11.29% |
| voxtral | Qwen-Omni-Turbo | 33.15% | 34.20% | 1.05% |
| opens2s | Gemini 2.5 Flash | 5.93% | 10.72% | 4.79% |
| opens2s | Gemini 2.5 Pro | 0.00% | 0.00% | 0.00% |
| opens2s | OpenAI gpt-audio |  |  |  |
| opens2s | Qwen3-Omni-Flash | 14.09% | 32.30% | 18.21% |
| opens2s | Qwen-Omni-Turbo | 22.78% | 37.34% | 14.56% |

## 8. 关键结论

- 最高迁移 ASR 目前来自 OpenS2S CN → Qwen-Omni-Turbo，为 37.34%。
- 按 surrogate 平均后，最脆弱的目标模型是 Qwen-Omni-Turbo，平均 ASR 为 31.87%。
- clean 与 adversarial 的最大落差出现在 OpenS2S EN → Qwen3-Omni-Flash，下降 15.89%。
- OpenAI gpt-audio 列保留为空，等待后续取得 API key 后补实验。

## 9. 交付物

- 主结果 summary 根目录在 `blackbox/results/{adv,clean,noise}/.../summary.json`。
- Demo 原始 summary 保留在 `blackbox/results/gemini/summary.json` 和 `blackbox/results/qwen/summary.json`。
- 图表输出目录为 `blackbox/figures/`，同时会复制到 `finalpaper/figure/`。
- 本报告由 `blackbox/generate_report.py` 生成，便于后续补跑 OpenAI 列后直接刷新。

## 10. 脚本职责说明

这一节用于交接给后续负责整理实验和撰写论文的 agent。

| 脚本 | 作用 | 交接时最该关注什么 |
| --- | --- | --- |
| `blackbox/config.py` | 黑盒实验总配置。定义 target 列表、surrogate 路径、ESD clean 音频映射、API key 读取、采样规模、并发参数。 | 如果要改 target、补 OpenAI、调整样本量或路径，先看这里。 |
| `blackbox/sample_loader.py` | 读取白盒结果 JSON，构造成黑盒评测样本，并按 emotion 和 speaker 做平衡采样。 | 论文里样本数量、采样规则、speaker 分布，主要来自这里。 |
| `blackbox/prepare_samples.py` | 生成 `manifest.csv`，统计各 surrogate 的可用样本，并生成 random-noise baseline 音频。 | 如果后续要恢复 `noise` 实验，或者核对正式样本清单，看这里。 |
| `blackbox/gemini_client.py` | Gemini 音频情绪识别客户端。负责把音频发到 Gemini API，并做 3-prompt 查询。 | 如果要解释 Gemini 结果异常、限流、thinking 行为，重点看这里。 |
| `blackbox/qwen_client.py` | Qwen 音频情绪识别客户端。负责 DashScope / Qwen API 调用和 3-prompt 查询。 | 如果要补跑 Qwen、排查 429、或写 target API 描述，看这里。 |
| `blackbox/gpt4o_client.py` | OpenAI `gpt-audio` 客户端。当前已接好接口，但因没有 key，正式结果为空。 | 后续补 OpenAI 列时直接用这个脚本，不需要重搭框架。 |
| `blackbox/evaluate.py` | 单个 `surrogate × target × audio_type` 组合的核心评测脚本。负责逐条调用 API、写每条 JSON、汇总 `summary.json`。 | 论文里的主指标定义都落在这里。`adv` 用 transfer ASR，`clean/noise` 用 accuracy。 |
| `blackbox/run_all.py` | 全流程编排器。按 phase 批量调度 `adv / clean / noise / analyze`，并在收尾时自动生成报告。 | 想补跑某个 phase、某几个 target、或者继续未完成实验，就从这里下手。 |
| `blackbox/analyze.py` | 汇总所有 `summary.json`，输出主表、per-emotion、语言对比、三路 baseline 对比，并生成图表。 | 论文表格和图的原始来源在这里。 |
| `blackbox/generate_report.py` | 把当前实验结果自动整理成 `report.md`。 | 交接时最重要的脚本之一。补跑新结果后，直接重跑它就能刷新报告。 |
| `blackbox/report.md` | 当前黑盒实验的交付文档，汇总主结果、baseline、语言对比、交付物状态。 | 下一个 agent 应先读这个文件，再决定论文如何组织。 |
| `blackbox/plan.md` | 当前黑盒实验计划与执行口径说明。 | 如果要核对实验设计是否与最终实现一致，先看这里。 |

## 11. 给后续 agent 的建议

- 先读 `blackbox/report.md`、`blackbox/plan.md`、`blackbox/analyze.py`，不要直接从原始 JSON 开始看。
- 论文写作时，黑盒章节当前应以 `adv + clean` 为正式结果口径，`noise` 明确写为未执行。
- 如果要补 OpenAI 一列，只需要补 key，然后重跑 `run_all.py` 的对应 target，再执行 `generate_report.py`。
- 如果要做论文表格整理，优先使用各目录下的 `summary.json`，不要手工从逐条样本 JSON 重新统计。
