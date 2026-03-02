# 安全顶会 Observation 章节写作风格规格书

> 基于一篇 NDSS/S&P/CCS/USENIX Security 级别论文的 observation 章节提取的抽象写作模式。

---

## 1. 章节与段落组织方式

summary: 采用"总领段 + 编号子节"的层级结构，每个子节围绕一个独立发现展开，段落长度差异显著——分析性段落较长、结论性段落简短。

- section_hierarchy: `顶层节（引导段）→ 编号子节（3.x）→ 子节内编号发现（(1)(2)）`
- intro_paragraph_role: 顶层节以 1 个概述段开篇，预告子节内容并提供路线图
- subsection_count: 2 个编号子节，每个子节聚焦一个核心发现
- paragraphs_per_subsection: 3–6 段
- paragraph_length_distribution: 分析性段落 80–150 词，过渡/总结段落 30–60 词
- intra_paragraph_pattern: `[背景/已知信息] → [具体观察/数据] → [推断/影响]`
- numbered_findings_pattern: 子节内部使用 `(1) (2)` 编号条目呈现核心发现，每条发现以**粗体短句**作为标题，后跟 2–4 句展开论述
- finding_title_style: `(N) [主语] + [动词短语描述发现]`，例如 `(1) [System X] [exhibits property Y]`

---

## 2. 小标题命名风格

summary: 子节标题采用名词短语为主的策略，信息密度中等偏高，兼顾概念引入与研究动作。

- naming_strategy: 以核心概念/现象为锚点，辅以分析对象
- grammatical_structure: `名词短语（主） / 动名词短语（辅）`
- typical_pattern: `"[Analysis/Property] of [Target Concept]"` 或 `"[Aggregated/Observed Phenomenon] in [Context]"`
- information_density: medium-high（每个标题包含 4–7 个实词）
- capitalization: Title Case（每个实词首字母大写）
- top_section_title_pattern: `"[Core Concept A] and [Core Concept B]"` — 双概念并列形式

---

## 3. 段落间过渡与衔接手法

summary: 合策略为主，显式过渡词使用频率中等，段间逻辑主要依赖"概念承接"和"证据-推论"链条衔接。

- explicit_transition_frequency: ~30% 的段落以显式过渡词/短语开头
- transition_word_types:
  - 因果类: `As a result`, `Consequently`
  - 递进类: `Furthermore`, `Moreover`, `Additionally`
  - 对比/转折类: `While`, `However`, `Notably`
  - 承接类: `To illustrate`, `Leading by`
- implicit_transition_method: 上一段末句的关键概念在下一段首句中被复现或指代（代词 `this`, `these`, `such` 等）
- cross_subsection_bridge: 子节末段包含对下一子节的预告性语句，形成 `[本节结论] → [引出下一节问题]` 的过渡模式
- numbered_finding_transition: 编号发现之间无额外过渡词，靠编号本身和概念递进维持连贯

---

## 4. 句式特征

summary: 句式以中长复合句为主体，被动语态比例较高但与主动语态间有节奏性交替，偶尔使用短句强调关键发现。

- avg_sentence_length: 22–35 词
- sentence_length_range: 12–50 词
- short_sentence_usage: 偶尔在关键发现处使用 ≤15 词的短句以制造强调效果
- passive_ratio: ~40–45%
- passive_usage_context: 描述实验设置、数据处理流程和已被观察到的现象时倾向使用被动语态
- active_usage_context: 陈述作者的分析推断、提出假设、描述因果关系时使用主动语态（主语常为 `we`）
- rhythm_pattern: `[长分析句(25-40词)] → [中等衔接句(15-25词)] → [短结论/强调句(10-18词)]`，三句一组的节奏循环
- sentence_opener_variety:
  - `We + verb`（~25%）: 作者行为/发现
  - `The/This + noun phrase`（~30%）: 承接上文概念
  - `介词/副词短语前置`（~20%）: 时间/条件限定
  - `动名词/不定式短语前置`（~10%）: 目的/方法说明
  - 其他（~15%）
- parenthetical_usage: 频繁使用圆括号插入补充信息（引用标记、缩写说明、举例），每段 1–3 处

---

## 5. 术语与表达习惯

summary: 术语在首次出现时以"全称 + 括号缩写"方式引入，后续统一使用缩写；技术细节处理采用"先定性描述后定量佐证"的递进策略。

- term_introduction_pattern: `[Full Term Name] ([Abbreviation])` 首次出现时给出完整定义
- subsequent_usage: 首次引入后全文统一使用缩写
- definition_technique: 通过一句功能性描述嵌入术语含义，而非独立定义句；形式为 `[Term], which [functional description],`
- technical_detail_granularity: 中等偏高 — 给出指标名称、计算方法的文字描述，但将完整公式留给后续方法章节
- hedging_expressions: 适度使用学术模糊表达（`may`, `generally`, `tends to`, `suggesting`, `indicating`），约每 2–3 段出现 1–2 处
- specificity_pattern: `[定性描述(1-2句)] → [量化数据/指标佐证(1句)] → [推论(1句)]`
- cross_reference_style: 使用 `(Section X.Y)` 格式的括号内交叉引用，指向同一论文其他章节

---

## 6. 图表引用方式

summary: 图表引用紧跟分析论述之后，采用"先陈述发现、再引图佐证"的模式，引用句式高度统一。

- figure_reference_frequency: 每个子节引用 1–3 个图/表
- reference_position: 几乎总是置于论述该发现的段落**末尾**或**中后部**
- reference_syntax_patterns:
  - `as shown in Fig. X`（最常见，约 40%）
  - `as displayed in Fig. X`
  - `as depicted in Fig. X`
  - `as illustrated in Fig. X`
  - `[Finding statement], as shown in Fig. X (a)-(b).`（带子图标号）
- figure_text_coordination: `[文字先阐述发现的定性含义] → [引用图表提供视觉/定量证据] → [基于图表数据进一步分析]`
- subfigure_reference_style: 使用 `(a)-(b)` 或 `(c)` 指代子图，嵌入引用句中
- figure_caption_role: salience: low — 规格书不分析图注本身的写作风格，但注意到图注在正文中不被重复

---

## 7. 论证与叙事风格

summary: 采用"渐进式揭示"的叙事策略——从宏观数据集现象入手，逐层深入到模型内部机制，最终汇聚为可攻击的脆弱性结论；证据组织遵循"定性观察 → 定量验证 → 机制解释"的三段式。

- argument_presentation_order: `[宏观现象] → [数据验证] → [微观机制] → [因果推断]`
- evidence_organization:
  - 第一层：对训练数据的统计分析（数据集层面）
  - 第二层：对模型输出的指标测量（模型行为层面）
  - 第三层：对内部表示的相似度分析（机制层面）
- causal_chain_construction: `[观察到现象 A] → [测量指标 B 验证] → [提出假设 C 解释成因] → [进一步证据 D 支撑假设] → [得出结论 E]`
- hypothesis_presentation_style: 使用 `We hypothesize that...` 或 `We speculate that...` 显式标记假设，与已验证的发现区分
- finding_emphasis_technique: 核心发现以 `(1) (2)` 编号列表形式突出呈现，每条以**粗体短语**开头作为"迷你标题"
- scope_expansion_pattern: 从单一模型/数据集的发现出发，扩展到多模型验证以增强普适性（`This observation is not only... but also prevalent on...`）
- concluding_move_per_subsection: 子节末尾将观察结果与论文后续章节（攻击方法）建立桥梁，形成 `[发现] → [脆弱性暴露] → [可利用性暗示]` 的叙事弧线
- tone: 客观分析为主体，偶尔使用"发现-值得注意"（`Notably`, `It is worth noting`）的语气强调关键点
- author_presence: 中等 — `we` 作为主语在分析推断时频繁出现，但在描述客观现象时退场

---

> **使用说明**：本规格书描述的是写作模式的抽象特征，所有槽位模板中的占位符（如 `[Core Concept]`、`[Finding statement]`）均为结构性示例，不包含源论文的任何实质内容。
