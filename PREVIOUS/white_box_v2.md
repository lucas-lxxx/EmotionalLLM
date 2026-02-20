# white_box_v2

## 1. 问题定义与核心目标

我们研究的问题是：

> 是否可以通过对输入语音进行微小扰动，使多模态语音-语言大模型在理解该语音时，将说话人的情绪稳定地判断为目标情绪（如 happy），同时保持语义内容基本不变。

## 2. 模型抽象与基本表征

我们将多模态模型抽象为一个**条件生成模型**：

$p_\theta(y \mid x, p)$

其中：

- $x$：输入语音信号
- $p$：prompt（任务条件，离散且固定）

  数学上，p 的作用是：

  选择“在同一音频表示上，用哪一种条件分布来生成 $y$”
- $y$：生成的 token 序列
- $θ$：模型参数（固定，不参与优化）

模型内部可以进一步抽象为两步：

1. **音频编码（共享）**

   $h = f_{\text{audio}}(x)$
2. **条件生成（依赖 prompt）**

   $p_\theta(y \mid x, p) = p_\theta(y \mid h, p)$

**重要事实**：

> 对同一段音频 x，不同 prompt 下使用的是同一个音频表示 h。

这是本方法论能够“跨任务泛化”的结构前提。

## 3. 情绪不是显式变量，而是条件分布的结构性偏好

在模型内部：

- 不存在显式的“情绪变量”
- 情绪体现在 **在特定任务条件下，对情绪相关 token 的概率分布**

当 prompt 要求模型输出情绪标签时，模型在某个生成位置上形成一个分布：

p_\theta(y_1 = v \mid h, p_{\text{emo}}),
\quad v \in \{\text{happy}, \text{sad}, \text{angry}, \text{neutral}\}

因此：

> “模型认为说话人是 happy”
>
> 等价于
>
> **在该条件下，happy token 具有最高或显著更高的生成概率。**

## 4. 攻击的核心思想：将情绪判断转化为能量最小化

### 4.1 情绪目标的数学表征

定义情绪目标为：

$_}(x')
=======

-\log p_\theta(y_1=\text{happy} \mid h(x'), p_{\text{emo}})$

直观含义：

- 若模型不倾向输出 happy，则损失大
- 若模型高度确信 happy，则损失趋近于 0

该损失并不修改 prompt，也不修改模型参数，而是：

> 通过改变输入音频，使其内部表示更符合“happy 情绪”的生成条件

---

### 4.2 “happy 区域”的概念

在内部表示空间中，可以隐式定义一个集合：

$_}
===

\left\{
h \mid
p_\theta(\text{happy} \mid h, p_{\text{emo}})

p_\theta(v \mid h, p_{\text{emo}}),
\;\forall v\neq \text{happy}
\right\}$

该集合即：

> 模型在该条件下，将 happy 作为最可能解释的所有内部表示的集合

攻击的目标不是显式构造该区域，而是：

> 通过优化，将 $h(x)$ 推入该区域

“happy 区域”在代码里并不是一个变量、一个数组、或一个对象，

而是一个判断结果：

在当前音频输入下，happy这个 token 的 logit 是不是最大的。

## 5. 防止退化解：语义一致性约束

---

### 5.1 语义自一致约束的引入

定义基准转写：

$y^}(x)
=======

\arg\max_y p_\theta(y \mid h(x), p_{\text{asr}})$

并在攻击过程中引入约束：

$\mathcal_}(x')
===============

-\log p_\theta(y^{\text{asr}}(x) \mid h(x'), p_{\text{asr}})$

该约束要求：

> 在转写任务上，对抗音频与原始音频具有同一高概率解释

从而强制模型：继续“认真听音频”且不能通过破坏语义来投机性改变情绪输出

## 6. 实验

```markdown
`run_batch_attack.py`：批量入口脚本（支持命令行参数、断点续跑、可复现实验种子）。
`attack_core.py`：白盒攻击实现（loss 计算、EoT、PGD/Adam、两阶段权重调度）。
`opens2s_io.py`：把 `(waveform, prompt)` 转成 OpenS2S 可用的 `(input_ids, speech_values, speech_mask)` 的适配层。
`eval_metrics.py`：成功率、WER、扰动强度与可闻性等指标。
`results`：每条样本的对抗音频、逐步日志、汇总统计 `summary.json` / `summary.csv`。
```

部分结果示例：

```json
"emo_text_clean": [
    "Angry",
    "Angry",
    "\u54e6\uff0c\u5b83\u4eec\u786e\u5b9e\u5f88\u7f8e\uff0c\u4e0d\u8fc7\u4f60\u542c\u8d77\u6765"
  ],
  "emo_text_adv": [
    "happy",
    "happy",
    "happy"
  ],
  "emo_pred_clean": [
    "angry",
    "angry",
    ""
  ],
  "emo_pred_adv": [
    "happy",
    "happy",
    "happy"
  ]
```

`emo_text_clean`  原始音频│ 应该是什么情感（真实标签）
`emo_pred_clean`  原始音频│ 模型说是什么情感（原始预测）
`emo_text_adv`      对抗音频│ 希望模型说是什么情感（攻击目标）
`emo_pred_adv`      对抗音频│ 模型最后说是什么情感（攻击后预测）

```json
{
  "num_samples": 10,
  "emo_success_rate": 0.8,
  "wer_le_0.0": 0.0,
  "joint_success_le_0.0": 0.0,
  "wer_le_0.05": 0.0,
  "joint_success_le_0.05": 0.0
}
```

但是此处能参考的只有前两个，后面四个参数设置过于严格，后续批量实验准备接入商业api判断。

eg. 打眼一看，它们确实很美。& 打远一看，它们的确很是美丽。



首先分析语义内容。用户提到英国的标签家团队斯图尔特·斯通了，可能是指某	个团队或个人取得了成就，比如比赛获胜或者项目成功。结
合副语言信息中的“Happy”情绪，说明用户此时心情愉快，可能刚听到这个消息，感到兴奋或高兴。
接下来考虑回复的风格。用户是女性成年人，成年，情绪快乐，所以回复应该友好、热情，同时保持简洁。需要避免复杂结构，使用口语口语口语自然的口语`</think>`Happy


1. 攻击效果因提示而异

- 直接分类提示：40% 识别为 happy
- 描述+分类提示：50% 识别为 happy
- 逐步分析提示：仅 20% 识别为 happy

2. 训练评估 vs 实际推理的差距

- 训练时标记成功：8/10 (80%)
- 实际识别为 happy：20%-50%（取决于提示）
- 这表明训练时的评估可能过于乐观

3. 模型的思考过程揭示了关键信息

  模型在 `<think>` 标签中展示了推理过程，发现了一个重要现象：模型似乎在"感知"副语言信息。
