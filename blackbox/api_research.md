# Audio emotion understanding in commercial LLM APIs as of April 2026

**Five major platforms now offer closed-source commercial APIs capable of perceiving speaker emotions directly from raw audio input: Google Gemini, OpenAI GPT-4o, Alibaba Qwen, ByteDance Doubao, and several Chinese competitors including StepFun and SenseTime.** These models process audio natively — not through an ASR-then-text pipeline — preserving paralinguistic cues like tone, prosody, and vocal affect that transcription destroys. However, academic benchmarks consistently show that even the best models still rely more on lexical content than pure acoustic cues for emotion classification, and audio-only emotion recognition performance lags significantly behind text-assisted performance. This report covers each platform's capabilities, API mechanics, pricing, and suitability for black-box adversarial attack experiments.

---

## Google Gemini leads in documented emotion detection from audio

Google offers the **most explicitly documented audio emotion understanding** among all platforms. The official audio guide at `ai.google.dev/gemini-api/docs/audio` lists "Detect emotion in speech and music" as a first-class use case, complete with working code samples in Python, JavaScript, Go, and REST that return structured JSON emotion labels (Happy/Sad/Angry/Neutral) per audio segment with timestamps and speaker IDs.

**Models supporting audio input** span the full Gemini lineup: Gemini 3.1 Pro Preview, Gemini 3 Flash Preview, Gemini 2.5 Pro (stable), and Gemini 2.5 Flash (stable). The deprecated Gemini 2.0 Flash shuts down June 1, 2026. Audio is tokenized at  **32 tokens per second** , with a maximum input length of **9.5 hours** per prompt. Supported formats include WAV, MP3, AIFF, AAC, OGG Vorbis, and FLAC.

Two complementary pathways exist for emotion work. The **standard generateContent API** accepts uploaded audio files and returns text analysis — ideal for batch emotion classification experiments. The **Live API** with Gemini 2.5 Flash Native Audio offers real-time streaming with an "Affective Dialog" feature (enabled via `enable_affective_dialog=True`), which Google describes as letting the model "interpret subtle acoustic nuances like tone, emotion, and pace" and "automatically de-escalate stressful support calls." Native audio processing means the model reasons directly on audio tokens rather than an intermediate transcript. Google's blog explicitly states: "Unlike traditional voice systems that convert speech to text before processing, native audio models can directly understand and respond to audio input, preserving nuances like tone, emotion, and conversational context."

On the **AHELM benchmark** (2025), Gemini 2.5 Pro ranked **#1 in emotion detection** among all tested audio language models with a mean win rate of 0.803. On the LISTEN benchmark, it achieved the highest overall performance but showed sharp declines in audio-only and paralinguistic-only settings, confirming that even the best models struggle with pure acoustic emotion cues.

| Attribute                           | Details                                                                                         |
| ----------------------------------- | ----------------------------------------------------------------------------------------------- |
| **API endpoint**              | `https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`             |
| **Live API**                  | WebSocket via Google AI Studio or Vertex AI                                                     |
| **Auth**                      | API key (`x-goog-api-key`) or Google Cloud service account                                    |
| **Audio pricing (2.5 Flash)** | **$1.00/1M input tokens**(~$0.002/min audio)                                              |
| **Audio pricing (2.5 Pro)**   | **$1.25/1M input tokens**(~$0.0024/min audio)                                             |
| **Closed-source**             | Yes — proprietary, no model weights available                                                  |
| **Key limitation**            | Affective Dialog only on Gemini 2.5 Flash Native Audio, not yet on 3.x Live models              |
| **Docs**                      | `ai.google.dev/gemini-api/docs/audio`,`ai.google.dev/gemini-api/docs/live-api/capabilities` |

---

## OpenAI GPT-4o processes audio end-to-end with emotion perception

OpenAI's GPT-4o family provides **native end-to-end audio processing** through a single neural network trained jointly across text, vision, and audio. OpenAI explicitly designed this to solve the limitation where "the main source of intelligence, GPT-4, loses a lot of information — it can't directly observe tone, multiple speakers, or background noises, and it can't output laughter, singing, or express emotion."

**Available models** include `gpt-audio` (alias for gpt-4o-audio-preview, snapshot `gpt-4o-audio-preview-2025-06-03`), `gpt-audio-mini`, and `gpt-realtime` / `gpt-realtime-mini` for low-latency WebSocket/WebRTC streaming. Audio input is provided as base64-encoded data within the Chat Completions API using `"type": "input_audio"` content blocks, with the `modalities` parameter set to `["text", "audio"]`.

Microsoft's Azure documentation confirms GPT-4o-Audio-Preview is "tailored for processing and generating audio content... ideal for audio sentiment analysis" and can "analyze recorded audio conversations to detect subtle emotional nuances, vocal characteristics, and mood indicators." The gpt-realtime announcement adds that the model "can capture non-verbal cues (like laughs), switch languages mid-sentence, and adapt tone." The GPT-4o System Card explicitly flagged "emotional perception and anthropomorphism risks," confirming the model's emotion detection capabilities warranted safety evaluation.

On the **PALLM benchmark** (arXiv:2603.15981), GPT-4o Audio achieved **68% tone-understanding appropriateness** in human evaluation across Expresso, IEMOCAP, and RAVDESS emotion datasets. On Dynamic SUPERB, GPT-4o performed comparably to Qwen2-Audio-7B-Instruct on emotion recognition tasks, outperforming Whisper-LLaMA and confirming genuine paralinguistic processing capability.

| Attribute                           | Details                                                                                                           |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Chat Completions endpoint** | `POST https://api.openai.com/v1/chat/completions`with `input_audio`content                                    |
| **Realtime endpoint**         | `wss://api.openai.com/v1/realtime`(WebSocket/WebRTC)                                                            |
| **Supported input formats**   | WAV, MP3; output: WAV, MP3, PCM16, FLAC, Opus, AAC                                                                |
| **Audio input pricing**       | **$40.00/1M tokens**(~$1.55/hr); Realtime: $32.00/1M                                                        |
| **Audio output pricing**      | **$80.00/1M tokens** ; Realtime: $64.00/1M                                                                  |
| **Context window**            | 128K tokens                                                                                                       |
| **Closed-source**             | Yes — fully proprietary                                                                                          |
| **Key limitation**            | Still labeled "preview"; audio not yet supported in newer Responses API; significantly more expensive than Gemini |
| **Docs**                      | `platform.openai.com/docs/guides/audio`,`platform.openai.com/docs/guides/realtime`                            |

---

## Alibaba Qwen offers three tiers of audio emotion capability

Alibaba's DashScope platform provides the  **most granular set of audio emotion tools** , spanning from structured emotion labels (ASR-level) to open-ended audio reasoning (omni-model level).

**Tier 1 — Qwen3-ASR-Flash** is the production-ready ASR model with explicit emotion classification. It recognizes **7 emotion categories: surprise, calm, happiness, sadness, disgust, anger, and fear** alongside transcription output. Three variants exist: `qwen3-asr-flash` (sync, ≤5 min), `qwen3-asr-flash-filetrans` (async, ≤12 hours), and `qwen3-asr-flash-realtime` (streaming WebSocket). Available in Singapore, US, and Beijing regions. This model returns structured emotion labels per utterance — ideal for systematic evaluation, though it is fundamentally an ASR model that attaches emotion metadata rather than a reasoning LLM.

**Tier 2 — Qwen3.5-Omni** is the flagship multimodal model and Alibaba's recommended replacement for the legacy Qwen-Audio series. Three sizes are available: `qwen3.5-omni-plus`, `qwen3.5-omni-flash`, and `qwen3.5-omni-light`. It accepts audio input (URL or Base64) up to  **3 hours** , with a  **256K token context window** . Documentation shows the model describing speakers' "emotional state" including nuanced descriptions like "relaxed, comfortable, full of everyday life" and "affection for his hometown." It supports **113 languages** for speech input. Currently in  **preview and temporarily free** . Expected pricing around $0.32–0.40/1M input tokens once GA. Audio tokenizes at  **25 tokens per second** .

**Tier 3 — Legacy Qwen-Audio** (`qwen-audio-turbo`, `qwen2-audio-instruct`) explicitly supports analyzing "说话人的情绪" (speaker's emotion) via prompting but is **free-trial only** with a 100K token quota and no paid option. Not recommended for production. On the MELD emotion benchmark, Qwen2-Audio-Instruct achieved  **49.9% unweighted accuracy** . Third-party evaluation found >90% accuracy on anger, happiness, neutral, sadness, and surprise but <20% on disgust and fear.

| Attribute                | Qwen3-ASR-Flash                                                   | Qwen3.5-Omni                                                         |
| ------------------------ | ----------------------------------------------------------------- | -------------------------------------------------------------------- |
| **Emotion output** | 7 structured categories                                           | Free-form natural language descriptions                              |
| **Max audio**      | 12 hours (async)                                                  | 3 hours                                                              |
| **API style**      | REST + WebSocket, OpenAI-compatible                               | OpenAI-compatible (streaming required)                               |
| **Regions**        | Singapore, US, Beijing                                            | Singapore, Beijing                                                   |
| **Pricing**        | Commercial (per-token)                                            | Free preview; ~$0.60/1M input tokens (Qwen3-Omni-Flash reference)    |
| **Closed-source**  | Yes (API is proprietary)                                          | Yes (API is proprietary; open-source model weights exist separately) |
| **Docs**           | `alibabacloud.com/help/en/model-studio/qwen-speech-recognition` | `alibabacloud.com/help/en/model-studio/qwen-omni`                  |

---

## ByteDance Doubao focuses on real-time emotion-aware voice dialogue

ByteDance's audio emotion capability lives primarily in the **End-to-End Realtime Voice Model (端到端实时语音大模型)** on Volcano Engine — a true end-to-end speech model, not an ASR→LLM→TTS cascade. It natively understands speaker emotions, tone, and mood from voice input and maintains **emotion continuity (情绪承接)** across conversation turns.

In external user testing, Doubao's voice model scored **4.36/5.0 satisfaction** versus GPT-4o's 3.18/5.0, with only **2% of testers** finding Doubao "too AI-like" compared to 30%+ for GPT-4o. The model was specifically praised for "情绪理解和情感表达" (emotion understanding and emotional expression).

 **Critical limitation for adversarial research** : Doubao's emotion-capable model is accessed exclusively via **WebSocket** (`wss://openspeech.bytedance.com/api/v3/realtime/dialogue`) and is designed for  **interactive real-time dialogue** , not batch audio file analysis. There is no documented "upload audio file → get emotion analysis" REST endpoint. The Seed 1.8/2.0 LLM chat API (`ark.cn-beijing.volces.com`) claims multimodal audio support, but practical evidence of direct audio file upload to the chat completion endpoint is limited — audio capabilities appear routed through the separate speech services. The standard ASR models on Volcano Engine  **do not include emotion recognition** .

| Attribute                | Details                                                                           |
| ------------------------ | --------------------------------------------------------------------------------- |
| **Realtime API**   | `wss://openspeech.bytedance.com/api/v3/realtime/dialogue`                       |
| **Resource ID**    | `volc.speech.dialog`                                                            |
| **Auth**           | App ID + Access Token from Volcano Engine console                                 |
| **SDKs**           | Android, iOS, Go                                                                  |
| **Pricing**        | Token-based (~6.25 tokens/sec input, ~25 tokens/sec output); free trial available |
| **Closed-source**  | Yes — fully proprietary commercial                                               |
| **Registration**   | Typically requires Chinese phone number; third-party gateways available           |
| **Key limitation** | Real-time dialogue only; no batch/file-based audio emotion analysis endpoint      |
| **Docs**           | `volcengine.com/docs/6561/1594356`                                              |

---

## Other platforms with notable audio emotion capabilities

**StepFun Step-1o-Audio** stands out as perhaps the most explicitly emotion-aware Chinese audio LLM. This **100B+ parameter end-to-end speech model** "understands environmental sounds, human paralanguage and emotions in speech, infers user age from voice, and understands music." Its documentation details "emotional intelligence: can recognize emotion information from tone and intonation, understand user emotional needs in context, and provide contextualized responses." Available as a closed-source commercial Realtime API via WebSocket at `platform.stepfun.com`, with OpenAI SDK compatibility. Supports Mandarin, Cantonese, Sichuan dialect, English, and Japanese.

**SenseTime SenseNova V6.5 Omni** offers real-time audio/video input with emotion perception as part of a  **6000B parameter MoE model** . The platform at `platform.sensenova.cn` provides a Realtime API integrated with RTC networks. The model "precisely recognizes user emotions" in voice interactions. Closed-source commercial, with limited-time free access.

**Zhipu GLM-4-Voice** is an end-to-end speech model with explicit "emotional expression and emotional empathy" — it can "simulate different emotions and tones such as happy, sad, angry, scared, and reply with appropriate emotional tone." The base model weights are open-sourced, but Zhipu offers a **closed-source commercial VideoCall API** at `bigmodel.cn` that provides the real-time audio interaction capability.

**iFlytek** offers strong speech emotion recognition through its **超拟人交互技术 (Super Human-like Interaction)** API, which uses "multi-dimensional voice attribute decoupling" to separate content, emotion, language, timbre, and prosody, identifying  **10+ emotion states** . However, this is a specialized pipeline service on the AIUI platform, not an integrated multimodal LLM API.

**Anthropic Claude does not support audio input** via API as of early 2026. Its consumer Voice Mode transcribes speech to text externally before processing, losing all paralinguistic information. **MiniMax** excels at emotionally expressive TTS but does not accept audio input for understanding. **Baidu ERNIE 5.0** supports audio input in a multimodal framework, but specific acoustic emotion perception is not clearly documented. **Tencent Hunyuan** announced a voice model with emotion capabilities but developer API availability remains unclear.

---

## Comparative overview for black-box adversarial attack experiments

For adversarial attack research requiring closed-source commercial APIs with direct audio input and emotion understanding, the platforms rank as follows by suitability:

| Platform                          | Audio file upload     | Real-time stream      | Emotion output          | API accessibility                      | Relative cost                  |
| --------------------------------- | --------------------- | --------------------- | ----------------------- | -------------------------------------- | ------------------------------ |
| **Google Gemini 2.5**       | ✅ Up to 9.5 hrs      | ✅ Affective Dialog   | Structured + free-form  | Global REST API, easy                  | **Low**(~$1/1M tokens)   |
| **Alibaba Qwen3-ASR-Flash** | ✅ Up to 12 hrs       | ✅ WebSocket          | 7 structured categories | Global REST API                        | Low–Medium                    |
| **Alibaba Qwen3.5-Omni**    | ✅ Up to 3 hrs        | ✅ Required streaming | Free-form descriptions  | Global REST API                        | **Free**(preview)        |
| **OpenAI GPT-4o Audio**     | ✅ Base64 in messages | ✅ Realtime WebSocket | Free-form + sentiment   | Global REST API, easy                  | **High**(~$40/1M tokens) |
| **StepFun Step-1o-Audio**   | ⚠️ Realtime only    | ✅ WebSocket          | Free-form, detailed     | Chinese platform                       | Low                            |
| **ByteDance Doubao**        | ❌ Realtime only      | ✅ WebSocket          | Emotion continuity      | Chinese platform, registration barrier | Low                            |
| **SenseTime SenseNova**     | ⚠️ Realtime focus   | ✅ RTC-integrated     | Free-form               | Chinese platform                       | Free (limited-time)            |
| **Zhipu GLM-4-Voice**       | ⚠️ VideoCall API    | ✅ WebSocket/RTC      | Emotional empathy       | Chinese platform                       | Medium                         |

**Gemini and Qwen are the strongest choices for systematic adversarial experiments.** Both support file-based audio upload (essential for controlled experiments), provide global API access, offer structured emotion outputs, and are priced affordably. Gemini's official emotion detection code samples with structured JSON output make it particularly experiment-friendly. OpenAI GPT-4o is the most established but costs **40× more per input token** than Gemini 2.5 Flash for audio — a significant factor at experimental scale. The Chinese platforms (Doubao, StepFun, SenseTime) are primarily oriented toward real-time dialogue rather than batch file analysis, which complicates controlled experimental setups but may still be viable through WebSocket automation.

## Conclusion

The audio emotion understanding landscape in commercial LLMs has matured rapidly. **All five major platforms now process audio natively** rather than through ASR intermediaries, preserving the paralinguistic signals critical for emotion perception. Google Gemini provides the best-documented and most benchmark-validated emotion detection with affordable pricing. Alibaba Qwen uniquely offers both structured emotion labels (ASR-level) and open-ended emotion reasoning (omni-model level) across the broadest set of deployment options. OpenAI GPT-4o delivers strong emotion perception but at premium pricing. ByteDance and Chinese competitors excel in real-time emotionally-aware dialogue but lack the batch file-analysis endpoints most useful for adversarial research.

A key finding across all platforms is that **acoustic-only emotion recognition remains substantially weaker than text-assisted emotion recognition** — the LISTEN benchmark showed sharp performance declines when models could only access audio without text cues. This gap between lexical and paralinguistic emotion understanding represents both a limitation and a potential attack surface for adversarial research: models may be vulnerable to adversarial audio perturbations that preserve text content while altering emotional prosody, or vice versa.
