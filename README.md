# EmotionalLLM

> Audio LLM 情绪感知的对抗脆弱性研究 · 目标：SIGCONF 顶会
> 当前进度请见 `CONTEXT.md`

## 项目一句话

在 L∞ ≤ 0.008 的扰动预算下，对 ALLM（Voxtral / OpenS2S / MERaLiON）的语音情绪感知进行**定向翻转攻击**，同时保持语义内容不变；配套跨模型迁移矩阵与黑盒商业 API 迁移实验。

---

## 仓库总览

```
EmotionalLLM/
├── README.md             本文件，项目入口 + 文件检索
├── CONTEXT.md            研究进度同步文档（长期维护）
├── CLAUDE.md             项目元信息
├── .gitignore
│
├── code/                 所有实验代码（按模块组织）
│   ├── observation/         Q1-Q4 机理分析的 probe 实验代码
│   │   ├── activation_patching/
│   │   ├── logit_lens/
│   │   └── modal_conflict/
│   ├── white_box_voxtral/   Voxtral 白盒攻击
│   ├── white_box_opens2s_v2/ OpenS2S 白盒攻击（ver2.0 为当前版）
│   └── white_box_meralion/   MERaLiON 白盒攻击（含 cross_eval）
│
├── blackbox/             黑盒攻击：API 客户端 + 迁移实验（相对独立，因结果量大）
│
├── results/              大型实验结果（不在 code/ 内部）
│   └── observation_v3/     Q1-Q4 实验产物
│       ├── Q1&Q4/
│       ├── Q2/
│       └── Q3/             STAA / PGD / GAO / REN 四种 SER 攻击迁移结果
│
├── paper/                论文 LaTeX 源码（原 finalpaper/）
│   ├── main.tex
│   ├── 2.observation.tex
│   ├── 3.threat_model.tex
│   ├── 4.methodology.tex
│   ├── 5.whitebox.tex
│   ├── 6.blackbox.tex
│   └── figure/
│
├── data/                 数据集（原 dataset/），目前只有 ESD/
│
├── refs/                 参考文献 PDF（原 paper/），约 35 篇
│
├── docs/                 框架图、调研文档、会议记录
│   ├── framework.png
│   ├── related_work_research.md
│   └── meeting/
│
├── reports/              汇报材料（原 LATEST/）：白盒讲稿、PDF、PPT 素材
│
└── archive/              早期探索、废弃代码、无关模板（原 PREVIOUS/）
    ├── white_box_opens2s_v1/    早期 OpenS2S 攻击，已废弃
    ├── observation_early_docs/  早期 observation 设计文档
    ├── pipid_paper_template/    无关论文模板（仅用作 LaTeX 参考）
    └── ...                      其他早期探索 md
```

---

## 实验原数据检索表

> **服务器路径前缀**：`/data1/lixiang/EmotionalLLM/`（Linux，SSH `lixiang@202.120.38.31 -p 22326`）
> **本地路径前缀**：`c:\Users\potte\Desktop\research\emotional LLM\`
> **双源**表示本地与服务器都有；**服务器**表示本地未同步或量太大不适合本地存放

### Observation（§2 机理分析）

| 实验 | 论文位置 | 代码位置 | 原始结果 |
|---|---|---|---|
| **Q1** ALLM 情绪 token 结构化竞争 | §2.1 | `code/observation/logit_lens/` | `results/observation_v3/Q1&Q4/`（双源） |
| **Q2** 情绪误判的下游后果（aligned vs conflict） | §2.2 | `code/observation/modal_conflict/` | `results/observation_v3/Q2/`（双源） |
| **Q3** SER 攻击迁移到 ALLM 的局限性 | §2.3 | — (用 `results/observation_v3/Q3/` 内各子目录的自带脚本) | `results/observation_v3/Q3/{STAA,PGD,GAO,REN}/`（双源） |
| **Q4** 跨模型普遍性（Qwen/Voxtral/SALMONN 均存在情绪 bias） | §2.4 | `code/observation/activation_patching/` 等 | `results/observation_v3/Q1&Q4/`（双源） |

### White-box attack（§5）

| 模型 | 代码 | 对抗样本（WAV + JSON） | 攻击日志 | 数据来源 |
|---|---|---|---|---|
| **Voxtral-Mini-3B** | `code/white_box_voxtral/` | `code/white_box_voxtral/result/Voxtral_{EN,CN,IEMOCAP,RAVDESS}/` | `code/white_box_voxtral/logs/` | 双源 |
| **OpenS2S** | `code/white_box_opens2s_v2/ver2.0/` | `code/white_box_opens2s_v2/result/{ESDfinal,blackbox/EN,blackbox/CN,IEMOCAP,RAVDESS}/` | 服务器端 logs | 双源 |
| **MERaLiON-2-3B** | `code/white_box_meralion/` | `code/white_box_meralion/result/MERaLiON_{EN,CN,IEMOCAP,RAVDESS}/` | `code/white_box_meralion/logs/` | **仅服务器**（本地无 result/） |

注：`code/white_box_voxtral/result/` 内还包含 `analyze_results.py`、`deepseek_judge.py`、`cleaned_data_all.csv`、`report_all.md` 等后处理脚本与聚合产物。

### Cross-model transferability（§5.3）

3×3 矩阵的每个 off-diagonal cell（源→目标）都有独立的 cross_eval 结果，命名规则 `{SRC}2{TGT}_{DATASET}`：

| Source | Target | 结果 JSON 路径（前缀：对应 target 模型的 `result/cross_eval/`） |
|---|---|---|
| Voxtral → OpenS2S | summary_V2OS_{EN,CN,IEMOCAP,RAVDESS}.json | `code/white_box_opens2s_v2/result/cross_eval/`（仅服务器） |
| Voxtral → MERaLiON | summary_V2M_{...}.json | `code/white_box_meralion/result/cross_eval/`（仅服务器） |
| OpenS2S → Voxtral | summary_OS2V_{...}.json | `code/white_box_voxtral/result/cross_eval/`（仅服务器） |
| OpenS2S → MERaLiON | summary_OS2M_{...}.json | `code/white_box_meralion/result/cross_eval/`（仅服务器） |
| MERaLiON → Voxtral | summary_M2V_{...}.json | `code/white_box_voxtral/result/cross_eval/`（仅服务器） |
| MERaLiON → OpenS2S | summary_M2OS_{...}.json | `code/white_box_opens2s_v2/result/cross_eval/`（仅服务器） |

汇总：`code/white_box_meralion/result/cross_eval/all_summaries.json`（服务器）。

### Black-box attack（§6，demo 阶段）

| 组件 | 位置 |
|---|---|
| 实验代码（客户端+评估） | `blackbox/{gemini,qwen,gpt4o,ernie}_client.py`、`blackbox/evaluate.py`、`blackbox/run_all.py` |
| 样本清单 | `blackbox/manifest.csv` + `blackbox/sample/` |
| 结果 | `blackbox/results/{gemini,qwen}/` |
| 实验计划 | `blackbox/plan.md`、`blackbox/report.md` |

### 参考文献与调研

| 内容 | 位置 |
|---|---|
| 参考文献 PDF（~35 篇） | `refs/` |
| 参考文献摘要 | `refs/summary.md` |
| 相关工作调研 | `docs/related_work_research.md` |
| 框架图 | `docs/framework.png` |
| 会议记录 | `docs/meeting/` |

### 论文源码与汇报

| 内容 | 位置 |
|---|---|
| 当前论文（ACM sigconf） | `paper/main.tex` + `paper/{2..6}.*.tex` |
| 白盒汇报 PDF | `reports/情绪LLM白盒攻击研究.pdf` |
| 白盒汇报讲稿 | `reports/白盒讲稿.md` |
| 白盒汇报素材目录 | `reports/white_box_final/` |

---

## 快速导航

### 我想查看白盒主结果表的数据来源
- 论文位置：`paper/5.whitebox.tex:38-61`（Table 1 `tab:main_results`）
- 各模型原始 JSON：见上方 "White-box attack" 表
- 聚合脚本：`code/white_box_voxtral/result/analyze_results.py`

### 我想复现某个白盒攻击
1. 登录服务器，`cd /data1/lixiang/EmotionalLLM/code/white_box_<model>/`
2. 查看 `config.py` 的数据集路径（默认 `/data1/lixiang/OpenS2S_dataset/ESD/CN`）
3. 运行 `python run_attack.py` 或 `bash run_batch.sh`
4. 结果写入 `result/` 子目录

### 我想查看 Q2 的 aligned-vs-conflict 数据
- 原始数据：`results/observation_v3/Q2/`
- 相关代码：`code/observation/modal_conflict/`

### 我想跑一次跨模型迁移评估
- 代码：各白盒目录下的 `cross_eval.py`（例如 `code/white_box_meralion/cross_eval.py`）
- 启动脚本：`code/white_box_meralion/launch_cross_eval.sh`

---

## 目录变更日志（2026-04-18 整理）

本次整理将原混乱的根目录收敛为 9 个顶层目录。重命名与归档清单：

| 原路径 | 新路径 | 类型 |
|---|---|---|
| `PREVIOUS/` | `archive/` | 重命名 |
| `LATEST/` | `reports/` | 重命名 |
| `paper/` | `refs/` | 重命名（原参考文献目录） |
| `finalpaper/` | `paper/` | 重命名（原论文源码目录） |
| `dataset/` | `data/` | 重命名 |
| `finalpaper2/` | `archive/pipid_paper_template/` | 归档（无关模板） |
| `observation/` | `archive/observation_early_docs/observation_v1/` | 归档 |
| `observation_v2/` | `archive/observation_early_docs/observation_v2/` | 归档 |
| `observation_v3/` | `results/observation_v3/` | 移动（视为结果） |
| `code/white_box_opens2s_v1/` | `archive/white_box_opens2s_v1/` | 归档（已废弃） |
| `code/white_box_opens2s_cross_eval.py` | （删除） | 删除（已在模型目录内） |
| `code/white_box_voxtral_cross_eval.py` | （删除） | 删除（同上） |
| `meeting/` | `docs/meeting/` | 移动 |
| `框架.png` | `docs/framework.png` | 移动+重命名 |
| `temp/research.md` | `docs/related_work_research.md` | 移动+重命名 |
| `temp/` | （删除） | 删除（已空） |

---

## 尚未处理的已知混乱（未来可选）

1. `code/white_box_voxtral/result/` 混放代码与数据（`analyze_results.py` 等应移到 `code/white_box_voxtral/analysis/`）
2. `code/white_box_opens2s_v2/ver1.0/` 保留作参考，可进一步归档
3. `code/white_box_opens2s_v2/result/blackbox/{EN,CN}/` 命名误导（实际是 ESD 大规模白盒结果，非黑盒），建议重命名为 `result/ESD_{EN,CN}_full/`
4. `blackbox/` 在顶层，与 `code/white_box_*/` 不对称；未来可移入 `code/black_box/`
5. 各白盒代码目录的 `result/` 未统一到顶层 `results/white_box/<model>/`

以上均为"代码路径硬编码敏感区"，需谨慎处理。执行前 grep 检查所有 `.py`/`.sh` 中的相关路径字符串。
