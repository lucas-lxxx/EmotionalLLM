# 一、后门攻击（Backdoor Attacks）

## 1. 情绪触发后门（EmoAttack）

- 利用 Emotional Voice Conversion (EVC)
- 将 neutral → angry/happy/surprise
- 训练时把转换后的样本标注成一个“攻击者指定标签”
- 模型学到隐藏规则：**看到某种情绪 → 输出攻击标签**

特点：

- 隐蔽性极强（情绪变化自然）
- 毒样本极少（50 条左右）
- 攻击成功率 ASR ≈ 99%

适用任务：KWS、Speaker Verification

---

# 二、白盒对抗攻击（White-box Adversarial Attacks）

攻击者知道模型结构与梯度。

## 2. FGSM（Fast Gradient Sign Method）

- 一步梯度攻击
- 扰动公式：x_adv = x + ε * sign(∇x L)

## 3. BIM / PGD（迭代攻击）

- 多步 FGSM
- 更强、更稳定的对抗扰动
- PGD 是最强的 L∞ 范数攻击之一

## 4. CW Attack（Carlini–Wagner）

- 基于优化的攻击：最小扰动 → 最大误导
- 难以检测 & 难以防御
- SER 模型在 CW 下表现最差

---

# 三、黑盒对抗攻击（Black-box Attacks）

攻击者无法获得梯度，只能看到输出。

## 5. 迁移攻击（Transfer Attack）

- 在替代模型生成对抗样本
- 再迁移攻击目标模型
- 利用“对抗样本可迁移性”

## 6. 查询攻击（Query-based Attack）

- 攻击者多次输入语音并查看输出
- 使用 NES / SPSA 等优化方法寻找最佳扰动

---

# 四、信号扭曲攻击（Speech Distortion Attacks）

无需梯度，也无需知道模型结构。

## 7. VTLN（Vocal Tract Length Normalization）

- 扭曲频率轴，改变声道长度特征
- 改变共振峰位置，破坏情绪特征

## 8. McAdams Transformation

- 修改 LPC 极点角度（共振峰结构）
- 让说话风格被完全扭曲但内容不变

## 9. MSS（Modulation Spectrum Smoothing）

- 平滑语音能量随时间变化
- 抹掉“情绪韵律特征”

---

# 五、物理攻击（Physical-world Attacks）

可在真实设备中实施，无需数字模型。

## 10. 噪声注入（Noise Injection）

- 工地噪声 / 风声 / 电话噪声 / 白噪声
- SER 准确度急剧下降

## 11. 音高/语速操控（Pitch/Tempo Change）

- 升降 2–4 半音
- 语速加快/放慢 ±10%
- 人类听不出异常，但情绪特征崩溃

## 12. 频带滤波（Band-pass Distortion）

- 削弱某个频段 → 情绪关键频率被破坏

---

## 1. 电话身份验证中的情绪伪装冒充攻击

利用情绪转换（EVC）或频谱扭曲，使银行/客服的说话人验证系统错误识别身份。
攻击自然难察觉，可直接造成欺诈风险，是目前产业最关心的真实安全问题。

## 2. 车载语音助手中的情绪触发后门攻击

对训练数据投毒，使车辆在“愤怒语气”时触发攻击者预设的错误指令（如导航偏移、开窗误触）。
适用于车载语音系统安全评估，风险高、现实性强。

## 3. AI 面试系统的语音情绪欺骗攻击

通过 pitch/tempo/MSS 扭曲隐藏紧张或制造“自信语气”，欺骗企业自动化情绪评估模型。
现实应用需求高，具有清晰伦理问题，适合作为完整研究课题。

## 4. 心理健康筛查系统的抗情绪检测攻击

医院、学校使用 SER 判断抑郁/焦虑，通过频谱扭曲或噪声可隐藏异常情绪。
兼具社会意义与技术挑战，是非常成熟的科研课题方向。

## 5. 老人报警系统的情绪混淆攻击（Elderly Emergency Call SER Attack）

许多老年人报警设备（紧急按钮、远程监控电话）利用情绪识别评估紧急程度。
攻击者可通过对抗扰动让系统误以为“情绪平稳”，导致延迟救援；
或制造“假高紧急情绪”导致系统误判。
场景真实、影响重大，属于高价值公益安全研究方向。
