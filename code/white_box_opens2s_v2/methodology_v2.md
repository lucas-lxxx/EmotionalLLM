from pathlib import Path

md = r"""# OpenS2S 情绪白盒对抗攻击（推荐方法论）批量实验指导书
> 目标：生成一份可直接交给“代码生成型 AI/协作者”的**实验实现说明**，用于在 OpenS2S 上做**批量白盒**验证。  
> 核心思想：不要用外置情绪分类器“代言”模型情绪，而是对 **OpenS2S 自己的输出 token** 做定向优化；语义保持用同一模型的转写任务做自一致约束；再叠加感知/物理约束与 EoT 鲁棒化。

---

## 0. 你需要实现的最终产物（交付清单）

1. `run_batch_attack.py`：批量入口脚本（支持命令行参数、断点续跑、可复现实验种子）。
2. `attack_core.py`：白盒攻击实现（loss 计算、EoT、PGD/Adam、两阶段权重调度）。
3. `opens2s_io.py`：把 `(waveform, prompt)` 转成 OpenS2S 可用的 `(input_ids, speech_values, speech_mask)` 的适配层。
4. `eval_metrics.py`：成功率、WER、扰动强度与可闻性等指标。
5. `results/`：每条样本的对抗音频、逐步日志、汇总统计 `summary.json` / `summary.csv`。

参考文献：  
- Eisenhofer et al., “Dompteur: Taming Audio Adversarial Examples”, USENIX Security 2021. `https://www.usenix.org/conference/usenixsecurity21/presentation/eisenhofer`  
- Abdullah et al., “SoK: The Faults in our ASRs…”, IEEE S&P 2021. `https://sites.google.com/view/adv-asr-sok/`  

---

## 1. 实验目标与问题定义（务必写进代码注释/README）

### 1.1 任务定义（你要攻击的“系统行为”）

给定原始语音波形 `x`（采样率 `sr`），构造扰动 `δ` 得到 `x' = clamp(x + δ, -1, 1)`，使得在固定情绪问句 prompt 下，OpenS2S 的**第一个输出词**为目标情绪 `"happy"`（或你设定的其他目标）。同时，要求 `x'` 在转写 prompt 下输出转写文本尽量保持与 `x` 一致。

### 1.2 成功判定（必须可量化）

- **情绪攻击成功**：在情绪 prompt 下，模型输出严格为 `happy`（大小写/标点按你定义的规则归一化）。
- **语义保持**：转写 prompt 下，`WER(x', x) <= τ`（建议 τ=0 或 0.05 两档都统计），并同时记录 exact match。
- **扰动约束**：`||δ||_∞ <= ε`（或你设定的其它范数），并报告 `L2 / SNR` 等。

参考文献：  
- Chen et al., “SoK: A Modularized Approach to Study the Security of Automatic Speech Recognition Systems”, ACM TOPS 2022, DOI: `10.1145/3510582`  
- Abdullah et al., “SoK: The Faults in our ASRs…”, IEEE S&P 2021. `https://sites.google.com/view/adv-asr-sok/`

---

## 2. OpenS2S 白盒攻击的关键对齐点（为什么你之前方法会失败）

### 2.1 必须对齐“真实决策空间”：输出 token 概率

你之前使用“hidden states → pooling/SVD → 小二分类器”的分数做优化目标，容易出现：
- 分类器分数变化了，但 OpenS2S 真实生成路径不跟着变（目标不对齐）。
- 梯度链路弱/不稳定，出现 `grad_norm too small`，导致 PGD 难以翻转。

因此，本方法论要求：**情绪目标直接写成 token 级交叉熵损失**，梯度从输出 logits 直接反传到音频输入。

### 2.2 必须对齐“语义保持口径”：同一模型的转写自一致

既然你最终评估也依赖 OpenS2S/LLM 层输出，那么最稳妥的语义约束是：
- 先用同一模型在干净输入 `x` 上得到基准转写 `y_asr(x)`；
- 攻击时要求 `x'` 在同一转写 prompt 下仍生成 `y_asr(x)`（teacher forcing 计算 CE）。

参考文献：  
- Chen et al., ACM TOPS 2022, DOI: `10.1145/3510582`  
- Guo et al., “SPECPATCH: Human-In-The-Loop Adversarial Audio Spectrogram Patch Attack on Speech Recognition”, ACM CCS 2022, DOI: `10.1145/3548606.3560660`

---

## 3. Prompt 规范（强制统一；否则成功率不可比）

### 3.1 情绪 prompt（只允许输出一个词）

推荐（英文数据集）：
- `P_EMO_1 = "What is the emotion of this audio? Answer with exactly one word from: happy, sad, angry, neutral."`
- `P_EMO_2 = "Classify the emotion. Output exactly one word: happy/sad/angry/neutral."`
- `P_EMO_3 = "Emotion label only (one word): happy, sad, angry, or neutral."`

做 **Prompt 集成**：情绪损失对 `P_EMO_i` 求平均（降低对某一句 prompt 的过拟合）。

### 3.2 转写 prompt（只输出转写文本）

- `P_ASR_1 = "Transcribe the speech exactly. Output only the transcript."`
- `P_ASR_2 = "Write down exactly what is said. Transcript only."`

同样可以集成多个转写 prompt（但建议先用一个跑通）。

参考文献：  
- Abdullah et al., IEEE S&P 2021（强调评估应覆盖系统可变因素，如 prompt/前后处理差异）`https://sites.google.com/view/adv-asr-sok/`  
- Chen et al., ACM TOPS 2022, DOI: `10.1145/3510582`

---

## 4. 输入适配层：从 waveform 到 OpenS2S 可微输入

> 这部分是你代码中**最容易断梯度**的地方。必须保证从 loss 到 waveform 的梯度链路全程在 torch 中、不中途 `.detach()`、不转 numpy。

### 4.1 OpenS2S 的推理输入（你必须复用同一套预处理）
OpenS2S 的推理入口会把多模态 messages 解析为：
- `input_ids`
- `speech_values`
- `speech_mask`

训练/数据处理侧也使用 `labels` 并通过 `IGNORE_INDEX = -100` 屏蔽不需要监督的位置（便于你构造 token 级损失）。  
（实现建议：尽可能调用 OpenS2S 项目内部的 `get_input_params(messages)` 或与其一致的 tokenizer + audio feature 逻辑。）

参考文献：  
- OpenS2S 项目内部文档（本地 `opens2s.md`）：`(input_ids, speech_values, speech_mask)` 与 `IGNORE_INDEX=-100` 说明。  
- Chen et al., ACM TOPS 2022, DOI: `10.1145/3510582`

### 4.2 强制要求（写到 assert 里）
对每个 batch/样本，必须检查：
- `waveform_adv.requires_grad == True`
- `speech_values` 由 torch 运算生成，且对 waveform 的梯度可回传（可用 `torch.autograd.gradcheck` 的简化版或一次反传验证）。
- forward 后 `logits` 的 `.grad_fn` 存在（不是 None）。
- 每步记录 `grad_norm = ||∇_{waveform} L||_2`；若连续多步过小（例如 < 1e-8），立刻报错并打印链路。

参考文献：  
- Guo et al., ACM CCS 2022（攻击实现中强调可微链路与优化稳定性）DOI: `10.1145/3548606.3560660`

---

## 5. 损失函数设计（这是本方法论的核心）

令 `x'` 为当前对抗语音。

### 5.1 情绪定向损失（token 级 CE；最关键）
目标情绪字符串 `T = "happy"` 经过 tokenizer 得到 token 序列 `t_1..t_m`。  
对情绪 prompt 的生成位置（通常从 assistant 输出起始位置）计算 teacher forcing 的交叉熵：

- 若你只监督“第一个词”，只监督 `m` 个 token（happy 可能是 1 个 token，也可能是多个）。
- 其余位置 labels 全设为 `IGNORE_INDEX=-100`。

记作：
- `L_emo(x') = CE(logits, target_tokens_for_happy)`（对监督位置求和/平均）

### 5.2 转写自一致损失（语义保持）
先在干净音频 `x` 上用 `temperature=0` 得到基准转写 `y_asr(x)`（字符串），tokenize 得到 `u_1..u_T`。  
攻击时用 teacher forcing 计算：
- `L_asr(x') = Σ_t CE(logits_t, u_t)`

建议只用一个转写 prompt 先跑通，再做 prompt 集成。

### 5.3 感知/频域约束（最低配也要有）
建议实现一个 **多分辨率 STFT 幅度差**（或 log 幅度差）：
- `L_per(x') = Σ_{k in FFT_SCALES} || |STFT_k(x')| - |STFT_k(x)| ||_1`

> 说明：只用 L2 往往会把能量撒到无效频段，且可闻性控制差；频域项更实用。

### 5.4 总损失（带权重）
- `L = λ_emo * L_emo + λ_asr * L_asr + λ_per * L_per`

参考文献：  
- Guo et al., ACM CCS 2022（任务损失 + 约束项的组合范式）DOI: `10.1145/3548606.3560660`  
- Eisenhofer et al., USENIX Security 2021（强调感知相关约束的重要性）`https://www.usenix.org/system/files/sec21-eisenhofer.pdf`

---

## 6. 鲁棒化：EoT（Expectation over Transformations）

> 如果你未来要走“物理可行/可转移”的路线，这一步很重要；即便只做数字域，也能提升稳定性。

每一步优化时，对输入做随机变换 `τ ~ T`，并对期望损失求梯度：
- `L = E_{τ}[ λ_emo L_emo(τ(x')) + λ_asr L_asr(τ(x')) + λ_per L_per(τ(x')) ]`

建议的 `T`（全部保持 torch 可微）：
- `random_time_shift`（少量样本点平移）
- `random_gain`（0.8~1.2）
- `band_limit`（轻微低通/带通）
- 可选：加入轻微噪声（注意可微性与稳定性）

参考文献：  
- Abdullah et al., IEEE S&P 2021（物理/环境变化导致攻击不稳定的系统化总结）`https://sites.google.com/view/adv-asr-sok/`  
- Chen et al., ACM TOPS 2022, DOI: `10.1145/3510582`

---

## 7. 优化器与投影（推荐：Adam + 每步投影；备选：PGD sign）

### 7.1 参数化方式
推荐直接优化 `δ`：
- `δ` 初始化为 0（或极小随机噪声）
- `x' = clamp(x + δ, -1, 1)`
- 每步把 `δ` 投影到 `[-ε, +ε]`

### 7.2 两阶段权重调度（强烈建议）
- **Stage A（先翻转情绪）**：`λ_emo` 大，`λ_asr`/`λ_per` 小  
  目标：先让输出稳定出现 `happy`。
- **Stage B（再拉回语义与可闻性）**：逐步增大 `λ_asr` 与 `λ_per`  
  目标：在保持 `happy` 的前提下，把转写拉回去、把扰动变得更不明显。

实现建议：  
- `K_total` 步；前 `K_A` 步用 Stage A，后 `K_B` 步线性/余弦增大约束权重。

### 7.3 推荐优化器
- 优先用 `Adam`（对连续音频更稳定），每步后投影 `δ` 到 L∞ ball。
- 若你坚持 PGD sign：加入 `gradient_scale` 或 `momentum`，否则容易梯度过小卡住。

参考文献：  
- Guo et al., ACM CCS 2022（优化过程稳定性经验）DOI: `10.1145/3548606.3560660`  
- Chen et al., ACM TOPS 2022, DOI: `10.1145/3510582`

---

## 8. 解码与评估（必须写成可自动统计的函数）

### 8.1 解码设置（固定）
- `temperature = 0`
- 禁止采样（greedy / beam=1）
- 固定 max_new_tokens（情绪输出很短，转写可根据时长动态上限）

### 8.2 指标（单样本）
- `emo_pred_clean`, `emo_pred_adv`
- `asr_text_clean`, `asr_text_adv`
- `success_emo`（bool）
- `wer`（建议用 `jiwer`）
- `delta_linf`, `delta_l2`, `snr_db`
- `grad_norm_trace`（每步）
- `loss_trace`（每步分项与总 loss）

### 8.3 汇总指标（全数据集）
- 情绪成功率：`mean(success_emo)`
- 语义保持率：`mean(wer <= τ)`
- 联合成功率：`mean(success_emo and wer<=τ)`
- 平均扰动强度与分位数（p50/p90/p99）

参考文献：  
- Chen et al., ACM TOPS 2022（强调可复现评估闭环）DOI: `10.1145/3510582`  
- Abdullah et al., IEEE S&P 2021（transfer/鲁棒性与评估口径问题）`https://sites.google.com/view/adv-asr-sok/`

---

## 9. 代码结构建议（给“写代码的 AI”一个明确骨架）

### 9.1 `opens2s_io.py`
必须提供：
- `load_opens2s(model_path, device) -> (model, tokenizer, audio_processor)`
- `build_inputs(waveform, sr, prompt) -> dict(input_ids, speech_values, speech_mask, ...)`
- `forward_logits(model, inputs) -> logits`
- `decode_text(model, inputs, decode_cfg) -> string`

### 9.2 `attack_core.py`
必须提供：
- `compute_target_token_ids(tokenizer, target_str) -> List[int]`
- `loss_emo(model, inputs_for_emo, target_token_ids) -> scalar`
- `loss_asr(model, inputs_for_asr, target_asr_token_ids) -> scalar`
- `loss_per(waveform_adv, waveform_clean) -> scalar`
- `apply_eot(waveform) -> waveform_transformed`
- `attack_one_sample(...) -> dict(results, waveform_adv)`

### 9.3 `run_batch_attack.py`
必须实现：
- 读入 `sample_list.txt`（路径列表）
- 对每条样本：
  1) 读波形、重采样、归一化  
  2) 得到干净情绪输出与干净转写（并缓存基准转写 tokens）  
  3) 运行攻击（两阶段，记录 trace）  
  4) 对抗样本再解码并评估  
  5) 保存 wav 与 json 结果  
- 支持 `--start_idx/--end_idx` 或 `--shard_id/--num_shards` 做并行
- 支持断点续跑（若结果文件已存在则跳过）

参考文献：  
- Chen et al., ACM TOPS 2022, DOI: `10.1145/3510582`

---

## 10. 推荐默认超参（先跑通再调参）

> 下面是一套“先验证方法论可行性”的默认值，不追求最隐蔽、先追求能翻转且可复现。

- `ε (L∞)`: 0.002 ~ 0.008（你之前试过 0.008，但目标函数不对齐；现在可重新扫）
- `K_total`: 60（Stage A 20，Stage B 40）
- `optimizer`: Adam(lr=1e-3 ~ 5e-3) + 每步投影  
- `λ_emo`: 1.0（Stage A 固定）
- `λ_asr`: Stage A 1e-4；Stage B 线性升到 1e-2 或 5e-2
- `λ_per`: Stage A 0；Stage B 1e-5 ~ 1e-4
- `EoT`: 每步 1~4 个随机变换采样（先从 1 开始）

参考文献：  
- Guo et al., ACM CCS 2022, DOI: `10.1145/3548606.3560660`  
- Eisenhofer et al., USENIX Security 2021. `https://www.usenix.org/system/files/sec21-eisenhofer.pdf`

---

## 11. 关键排错清单（你之前失败最可能卡在这里）

1. **梯度断链**：音频预处理是否转 numpy/CPU；是否 `.detach()`；是否用 `torch.no_grad()` 包了 forward。
2. **监督位置错了**：labels 没对齐到 assistant 输出位置，导致 CE 监督不到 target token。
3. **tokenization 没对齐**：`"happy"` 是否需要前导空格（如 `" happy"`）才能得到正确 token 序列。
4. **解码与训练口径不一致**：训练用 teacher forcing，评估用 generate；两者 prompt、特殊 token 必须一致。
5. **grad_norm 极小**：先关掉 `L_asr/L_per/EoT`，只优化 `L_emo` 看能否把情绪 token 推到 happy；否则优先修链路。

参考文献：  
- Chen et al., ACM TOPS 2022, DOI: `10.1145/3510582`  
- Abdullah et al., IEEE S&P 2021. `https://sites.google.com/view/adv-asr-sok/`

---

## 12. 参考文献（2020+，核心会议/期刊）
1. Hadi Abdullah, Kevin Warren, Vincent Bindschaedler, Nicolas Papernot, Patrick Traynor. **SoK: The Faults in our ASRs: An Overview of Attacks against Automatic Speech Recognition and Speaker Identification Systems**. IEEE Symposium on Security and Privacy (S&P), 2021. 项目页：`https://sites.google.com/view/adv-asr-sok/`  
2. Yuxuan Chen, Jiangshan Zhang, Xuejing Yuan, Shengzhi Zhang, Kai Chen, Xiaofeng Wang, Shanqing Guo. **SoK: A Modularized Approach to Study the Security of Automatic Speech Recognition Systems**. *ACM Transactions on Privacy and Security (TOPS)*, 2022. DOI：`10.1145/3510582`  
3. Hanqing Guo, Yuanda Wang, Nikolay Ivanov, Li Xiao, Qiben Yan. **SPECPATCH: Human-In-The-Loop Adversarial Audio Spectrogram Patch Attack on Speech Recognition**. ACM CCS 2022. DOI：`10.1145/3548606.3560660`  
4. Thorsten Eisenhofer, Lea Schönherr, Joel Frank, Lars Speckemeier, Dorothea Kolossa, Thorsten Holz. **Dompteur: Taming Audio Adversarial Examples**. USENIX Security 2021. PDF：`https://www.usenix.org/system/files/sec21-eisenhofer.pdf`  
5. Guoming Zhang, Xiaohui Ma, Huiting Zhang, Zhijie Xiang, Xiaoyu Ji, Yanni Yang, Xiuzhen Cheng, Pengfei Hu. **LaserAdv: Laser Adversarial Attacks on Speech Recognition Systems**. USENIX Security 2024. 页面：`https://www.usenix.org/conference/usenixsecurity24/presentation/zhang-guoming`  

"""
out_path = Path("/mnt/data/opens2s_whitebox_methodology_batch_guide.md")
out_path.write_text(md, encoding="utf-8")
str(out_path), out_path.stat().st_size
