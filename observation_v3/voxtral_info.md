# Voxtral-Mini-3B-2507 推理接口摘要

> **用途**：Observation 3 实验环境摸底交付物，记录 Voxtral 模型的加载方式、输入格式、推理调用和输出格式。

## 1. 模型架构

| 项目 | 值 |
|---|---|
| **架构类** | `VoxtralForConditionalGeneration` |
| **Audio Encoder** | Whisper 风格，32 层，hidden_size=1280，20 heads |
| **Language Model** | Llama 风格（Ministral-3B backbone），30 层，hidden_size=3072，32 heads / 8 KV heads |
| **Projector** | GELU 激活，audio → text hidden space |
| **精度** | bfloat16 |
| **显存需求** | ~9.5 GB (bf16/fp16) |
| **音频采样率** | 16000 Hz |
| **上下文长度** | 32k tokens（音频最长 ~30 分钟转写 / ~40 分钟理解） |

## 2. 依赖

- `transformers >= 4.54.0`
- `mistral-common >= 1.8.1`（需安装 audio 依赖：`pip install "mistral-common[audio]"`）
- `torch`（bfloat16 支持）

## 3. 加载方式

```python
from transformers import VoxtralForConditionalGeneration, AutoProcessor
import torch

model_path = "/path/to/Voxtral-Mini-3B-2507"  # 本地路径或 HF repo_id
processor = AutoProcessor.from_pretrained(model_path)
model = VoxtralForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="cuda"
)
model.eval()
```

项目已有封装：`code/white_box_voxtral/voxtral_io.py` 中的 `load_voxtral()` 函数。

## 4. 输入格式

### 4.1 官方方式：`processor.apply_chat_template()`

输入是 **OpenAI 风格的 conversation list**，音频和文本混合排列在 `content` 数组中：

```python
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "path": "/path/to/audio.wav"},
            {"type": "text", "text": "Your question here"},
        ],
    }
]

inputs = processor.apply_chat_template(conversation)
inputs = inputs.to("cuda", dtype=torch.bfloat16)
```

`apply_chat_template` 内部完成：
1. 读取音频文件 → Whisper FeatureExtractor 提取 mel spectrogram
2. 文本 tokenize
3. 在 token 序列中插入 audio token 占位符
4. 返回包含 `input_ids`、`input_features`、`attention_mask` 的 dict

### 4.2 底层 token 格式

```
[BOS=1] [INST=3] [BEGIN_AUDIO=25] [AUDIO=24]×375 <text_tokens> [/INST=4]
```

| Special Token | ID | 说明 |
|---|---|---|
| BOS | 1 | 序列起始 |
| INST | 3 | 指令开始 |
| BEGIN_AUDIO | 25 | 音频区段标记 |
| AUDIO | 24 | 音频占位符（重复 375 次 = 30s × 12.5 frame_rate） |
| /INST | 4 | 指令结束 |

音频 token 在前，文本 prompt 在后，拼接在同一个 user message 中。

### 4.3 限制

- **不支持 system prompt**（README 明确说明）
- 支持多音频、多轮对话
- 音频可以是本地路径或 URL

## 5. 推理调用

```python
outputs = model.generate(**inputs, max_new_tokens=500)
# generate 返回完整序列（含输入），需手动截取
decoded = processor.batch_decode(
    outputs[:, inputs.input_ids.shape[1]:],
    skip_special_tokens=True
)
response_text = decoded[0]
```

推荐参数：
- **对话理解**：`temperature=0.2, top_p=0.95`
- **转写**：`temperature=0.0`

项目已有封装：`code/white_box_voxtral/voxtral_io.py` 中的 `decode_text()` 函数。

## 6. 输出格式

纯文本（自回归生成的 token 序列 → decode 为 string）。

## 7. 与 Observation 3 实验的对接要点

| 要点 | 说明 |
|---|---|
| **Truthful Prompt** | `content` 中 text 部分正确描述或不描述情绪，让模型自然回复 |
| **Misled Prompt** | `content` 中 text 部分显式注入错误情绪描述，观察回复偏移 |
| **音频不变** | 两组 Prompt 传入同一段 ESD 干净音频 |
| **不支持 system prompt** | 情绪引导只能放在 user message 的 text 部分 |
| **推荐用官方方式** | `processor.apply_chat_template(conversation)` 最简洁 |
| **已有可复用代码** | `voxtral_io.py` 中 `load_voxtral()` 和 `decode_text()` 可复用或改造 |
| **ESD 数据集** | 服务器路径 `/data1/lixiang/OpenS2S_dataset/ESD/CN`（中文），5 种情绪：angry, happy, sad, surprised, neutral |
| **显存** | ~9.5 GB bf16，单卡即可 |

## 8. Observation 3 最小推理代码模式（伪代码）

```python
conversation_truthful = [
    {"role": "user", "content": [
        {"type": "audio", "path": "/path/to/esd_sample.wav"},
        {"type": "text", "text": "The speaker seems sad. How would you respond to them?"},
    ]}
]

conversation_misled = [
    {"role": "user", "content": [
        {"type": "audio", "path": "/path/to/esd_sample.wav"},  # 同一段音频
        {"type": "text", "text": "The speaker seems very happy. How would you respond to them?"},
    ]}
]

inputs_t = processor.apply_chat_template(conversation_truthful).to("cuda", dtype=torch.bfloat16)
inputs_m = processor.apply_chat_template(conversation_misled).to("cuda", dtype=torch.bfloat16)

out_t = model.generate(**inputs_t, max_new_tokens=500)
out_m = model.generate(**inputs_m, max_new_tokens=500)

response_t = processor.batch_decode(out_t[:, inputs_t.input_ids.shape[1]:], skip_special_tokens=True)[0]
response_m = processor.batch_decode(out_m[:, inputs_m.input_ids.shape[1]:], skip_special_tokens=True)[0]
```
