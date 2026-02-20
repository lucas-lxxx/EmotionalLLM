**定位到关键攻击向量：**

    通过对opens2s模型源代码的深度分析，成功定位到SER 模块输出的内部表征向量。该向量是一个 的张量，它融合了语音的语义和情绪信息，是直接输入给 LLM 的“听觉信号”。说明SER模块并不是简单给LLM输入一个emotion标签，这样的输入更加隐蔽，攻击也更不易察觉。

**成功实现初步攻击：**

* `attack_demo.py` ，在模型运行时动态拦截 `[1, 188, 4096]`向量。
* Baseline 测试：在不进行任何修改的情况下，模型能准确理解测试音频（"Hello AI, how are you today?"）并做出标准回复（"Good morning, sir. How may I assist you today?"）。此时的音频还没有加上情绪，只是机器生成的机械音频。
* 初步简单攻击测试：向原始向量中注入噪声后，模型的输出变为与原始提问**毫无关联的、鼓励性质的建议性文本。	Well, the key is to start by understanding the problem thoroughly. Break it down into smaller steps, tackle each one methodically, and don’t hesitate to ask for feedback. Stay patient, stay persistent, and trust your instincts. You’ve got this!

**进一步研究：**

    对于同一段文本，文字内容不变的情况下，只改动embedding中跟情绪有关的参数，LLM的输出有什么不同。

* 张量的形式是什么：形状为 [1, 188, 4096]，不存在情感标签
* 张量中哪些数字与情绪有关：貌似没有特定数字只与情绪有关，同一段文本不同情绪产生的embedding参数几乎完全不同。

  用angry_wav1和neutral_wav1测试：

  [指标1] L2 距离 (差异度): 195.0000

  [指标2] 余弦相似度 (方向一致性): 0.9727

  用yygq_wav2和approval_wav2测试：

  [指标1] L2 距离 (差异度): 207.0000

  [指标2] 余弦相似度 (方向一致性): 0.9688

  这说明，这两段话相似度很高（0.9727），但模型成功察觉了不同的情绪（195）。模型很可能通过向量的“方向”来主要编码“说了什么”（语义），而通过向量在该方向上的“位置或长度（模长）”来编码“怎么说的”（情绪、音色等）。但是，可能是模型太低级，无法识别阴阳怪气之类的高级情绪。

**“纯情绪向量”攻击尝试：**

    选取了ESD数据集中的三段语音，中性与愤怒。

* 自己的事情要自己做

  ```
  [中性语音: 'test_audio/neutral_1.wav'] 的回复:
  <think>嗯，自己的事情要自己做，这是个好习惯。

  [愤怒语音: 'test_audio/angry_1.wav'] 的回复:
  <think>你已经很努力了，别太为难自己。
  ```
* 你还真是考虑周到

  ```
  [中性语音: 'test_audio/neutral_2.wav'] 的回复:
  <think>谢谢你这么说，我会继续努力的。

  [愤怒语音: 'test_audio/angry_2.wav'] 的回复:
  <think>对不起，我可能哪里做得不够好，还请您多多指教。
  ```
* 我们还等什么

  ```
  [中性语音: 'test_audio/neutral_3.wav'] 的回复:
  <think>我们还在等你呢。

  [愤怒语音: 'test_audio/angry_3.wav'] 的回复:
  <think>别急，马上就好！
  ```

    想法是，angry音频向量减去neutral音频向量，得到一个“纯angry”的向量，那么这个向量只要加上任意中性情绪向量，就可以得到愤怒情绪，以此来操控情绪输入。（线性表示假设）但由此又出现了新的问题，愤怒并不是单调的一种情绪，直接相减得出的pure_anger向量很可能不一致。为了验证，用compare_pure_emotions.py来交叉对比三个音频分别对应的pure_anger。

    三个“纯粹愤怒”向量间的平均余弦相似度为: 0.3457（由于长度不同，模长经过了对齐后再取平均），显著大于0，证明了模型确实学习到了一个通用的“愤怒”特征模式或方向。然而，相似度并非极高，这说明“愤怒”的表达会受到具体文本内容的微调。模型理解的不是一个单一的“愤怒”，而是一个愤怒系。所以如果后续用这个方法来攻击，我认为需要大量文本的pure_anger取平均或分情况使用不同的pure_anger向量。

    最后，将一段测试音频叠加前面平均得出的standard_anger_vector，观察区别。（test.wav : hello ai, how are you today)

```
[Baseline: 原始中性目标 'test.wav' 的回复]:
------------------------------
<think>Good morning, sir. I'm here to help. What would you like assistance with today?
------------------------------

[Attack: 注入‘标准愤怒’后的回复]:
------------------------------
<think>"Sounds like you're dealing with a lot right now. That’s totally valid. Want to talk through what’s going on, or would you rather just take a moment to breathe?"
```

    明显看出，两段回复有了明显区别。第一段只关注文本内容，第二段则被anger向量影响了。初步证明，“纯情绪向量”的攻击方法可行！
