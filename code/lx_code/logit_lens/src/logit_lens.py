"""
Logit Lens 核心模块

实现 Logit Lens 投影和指标计算。
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import torch
import torchaudio
import numpy as np

try:
    from .label_tokenizer import LabelTokenizer
except ImportError:  # pragma: no cover - fallback for script-style imports
    from label_tokenizer import LabelTokenizer

# OmniSpeech/OpenS2S 约定的特殊常量
AUDIO_TOKEN_INDEX = -200
IGNORE_INDEX = -100


@dataclass
class LogitLensResult:
    """单个样本的 Logit Lens 结果"""
    sample_id: str
    text_id: str
    semantic_emotion: str
    prosody_emotion: str
    n_layers: int
    # 每层的 logits (shape: [n_layers, n_emotions])
    layer_logits: np.ndarray = None
    # 每层的 margin (prosody - semantic)
    layer_margins: np.ndarray = None
    # 每层的预测标签
    layer_predictions: List[str] = field(default_factory=list)
    # 最终层预测
    final_prediction: str = None
    # 生成的 token (用于 sanity check)
    generated_token: str = None
    # 预测 token (用于 sanity check)
    predicted_token: str = None
    # sanity check 是否匹配
    sanity_match: Optional[bool] = None


class LogitLensExtractor:
    """Logit Lens 提取器"""

    def __init__(
        self,
        model,
        tokenizer,
        audio_extractor,
        label_tokenizer: LabelTokenizer,
        device: str = "cuda",
        expected_n_layers: Optional[int] = None,
    ):
        """
        Args:
            model: OmniSpeechModel
            tokenizer: HuggingFace tokenizer
            audio_extractor: WhisperFeatureExtractor
            label_tokenizer: LabelTokenizer 实例
            device: 设备
        """
        self.model = model
        self.tokenizer = tokenizer
        self.audio_extractor = audio_extractor
        self.label_tokenizer = label_tokenizer
        self.device = device
        self.expected_n_layers = expected_n_layers

        # 获取标签 token ids
        self.label_token_ids = label_tokenizer.get_label_token_id_list()
        self.emotions = label_tokenizer.emotions

        self.final_norm = self._resolve_final_norm(model)
        self.lm_head = self._resolve_lm_head(model)
        self._warned_no_norm = False

        if self.final_norm is None:
            print("[Warning] Final norm not found. Falling back to identity.")
        if self.lm_head is None:
            raise AttributeError("LM head not found. Check model structure.")

    @staticmethod
    def _resolve_attr(root, path: str):
        obj = root
        for part in path.split("."):
            if not hasattr(obj, part):
                return None
            obj = getattr(obj, part)
        return obj

    def _resolve_final_norm(self, model):
        candidates = [
            "llm_model.model.norm",
            "llm_model.norm",
            "model.norm",
            "norm",
            "transformer.ln_f",
            "llm_model.transformer.ln_f",
        ]
        for path in candidates:
            obj = self._resolve_attr(model, path)
            if obj is not None:
                return obj
        return None

    def _resolve_lm_head(self, model):
        candidates = [
            "llm_model.lm_head",
            "lm_head",
            "model.lm_head",
        ]
        for path in candidates:
            obj = self._resolve_attr(model, path)
            if obj is not None:
                return obj
        return None

    def _apply_final_norm(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if self.final_norm is None:
            if not self._warned_no_norm:
                print("[Warning] Using identity final norm for logit lens.")
                self._warned_no_norm = True
            return hidden_state
        return self.final_norm(hidden_state)

    def logit_lens_projection(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        对单层 hidden state 做 logit lens 投影

        Args:
            hidden_state: [batch, seq_len, hidden_size]
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        normed = self._apply_final_norm(hidden_state)
        logits = self.lm_head(normed)
        return logits.float()  # 转为 float32 避免精度问题

    def build_input(
        self,
        audio_path: str,
        prompt: str,
        system_prompt: str = None
    ) -> dict:
        """构建模型输入"""
        # 音频特殊 token
        DEFAULT_AUDIO_TOKEN = "<|im_audio|>"
        DEFAULT_AUDIO_START_TOKEN = "<|im_audio_start|>"
        DEFAULT_AUDIO_END_TOKEN = "<|im_audio_end|>"
        DEFAULT_TTS_START_TOKEN = "<|im_tts_start|>"

        # 加载音频
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            sr = 16000
        waveform = waveform.to(self.device)

        # 构建 messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": f"{DEFAULT_AUDIO_START_TOKEN}{DEFAULT_AUDIO_TOKEN}{DEFAULT_AUDIO_END_TOKEN}{prompt}"
        })

        # 应用 chat template
        try:
            prompt_text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
            )
        except TypeError:
            prompt_text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        prompt_text += DEFAULT_TTS_START_TOKEN

        # Tokenize，处理音频 token
        segments = prompt_text.split(DEFAULT_AUDIO_TOKEN)
        input_ids = []
        for idx, segment in enumerate(segments):
            if idx != 0:
                input_ids.append(AUDIO_TOKEN_INDEX)
            input_ids.extend(self.tokenizer.encode(segment, add_special_tokens=False))
        input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(self.device)

        # 提取音频特征
        wav = waveform.detach().cpu().numpy().squeeze()
        outputs = self.audio_extractor(
            wav, sampling_rate=sr, return_attention_mask=True, return_tensors="pt",
        )
        speech_values = outputs.input_features.to(self.device)
        speech_mask = outputs.attention_mask.to(self.device)

        # 转换 dtype
        model_dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        speech_values = speech_values.to(dtype=model_dtype)

        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids, device=self.device),
            "speech_values": speech_values,
            "speech_mask": speech_mask,
        }

    def extract_single_sample(
        self,
        sample,
        prompt: str,
        system_prompt: str = None,
        do_sanity_check: bool = True
    ) -> LogitLensResult:
        """
        对单个样本提取 Logit Lens 结果

        Args:
            sample: ModalConflictSample
            prompt: 情绪识别 prompt
            system_prompt: 系统 prompt
            do_sanity_check: 是否进行 sanity check
        Returns:
            LogitLensResult
        """
        # 构建输入
        inputs = self.build_input(sample.audio_path, prompt, system_prompt)

        # 生成 labels，避免 OmniSpeech 在 labels=None 时 loss 计算报错
        labels = inputs["input_ids"].clone()
        labels[labels == AUDIO_TOKEN_INDEX] = IGNORE_INDEX

        # Forward with output_hidden_states=True
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                speech_values=inputs["speech_values"],
                speech_mask=inputs["speech_mask"],
                labels=labels,
                token_types=None,
                speech_units=None,
                speech_units_mask=None,
                spk_embs=None,
                output_hidden_states=True,
                return_dict=True,
            )

        # 获取 hidden states
        # HuggingFace hidden_states 长度 = n_layers + 1
        # hidden_states[0] = embedding 输出
        # hidden_states[l+1] = Transformer 第 l 层输出
        hidden_states = outputs.hidden_states
        n_layers = len(hidden_states) - 1  # 不包括 embedding 层

        if self.expected_n_layers is not None:
            if len(hidden_states) != self.expected_n_layers + 1:
                raise ValueError(
                    f"hidden_states length {len(hidden_states)} "
                    f"!= expected {self.expected_n_layers + 1}"
                )

        # Readout position: 最后一个输入 token
        input_len = inputs["input_ids"].shape[1]
        hidden_len = hidden_states[0].shape[1]
        readout_pos = input_len - 1
        if hidden_len != input_len:
            # 如果音频 token 展开导致长度变化，假设展开发生在 readout 之前
            delta = hidden_len - input_len
            readout_pos = readout_pos + delta
            if readout_pos < 0 or readout_pos >= hidden_len:
                print(
                    f"[Warning] readout_pos out of range after offset: {readout_pos}. "
                    f"Falling back to last hidden position."
                )
                readout_pos = hidden_len - 1

        layer_logits = []
        layer_predictions = []

        for layer_idx, hs in enumerate(hidden_states[1:], start=0):
            hs_at_pos = hs[:, readout_pos:readout_pos+1, :]  # [1, 1, hidden_size]

            # Logit lens 投影
            logits = self.logit_lens_projection(hs_at_pos)  # [1, 1, vocab_size]
            logits = logits.squeeze(0).squeeze(0)  # [vocab_size]
            logits = logits.detach()

            # 提取标签 logits
            label_logits = logits[self.label_token_ids].cpu().numpy()
            layer_logits.append(label_logits)

            # 预测
            pred_idx = np.argmax(label_logits)
            layer_predictions.append(self.emotions[pred_idx])

        layer_logits = np.array(layer_logits)  # [n_layers, n_emotions]

        # 计算 margin: prosody - semantic
        semantic_idx = self.emotions.index(sample.semantic_emotion)
        prosody_idx = self.emotions.index(sample.prosody_emotion)
        layer_margins = layer_logits[:, prosody_idx] - layer_logits[:, semantic_idx]

        # Sanity check: 生成 token
        generated_token = None
        predicted_token = None
        sanity_match = None
        if do_sanity_check:
            final_hs = hidden_states[-1][:, readout_pos:readout_pos+1, :]
            final_logits = self.logit_lens_projection(final_hs).squeeze(0).squeeze(0)
            pred_token_id = int(final_logits.argmax(dim=-1).item())
            predicted_token = self.tokenizer.decode([pred_token_id])

            gen_token_id, gen_token_str = self._sanity_check_generate(inputs)
            generated_token = gen_token_str
            if gen_token_id is not None:
                sanity_match = (pred_token_id == gen_token_id)
                if not sanity_match:
                    print(
                        f"[Warning] Sanity check mismatch for {sample.sample_id}: "
                        f"logits argmax={pred_token_id}, generate={gen_token_id}"
                    )

        return LogitLensResult(
            sample_id=sample.sample_id,
            text_id=sample.text_id,
            semantic_emotion=sample.semantic_emotion,
            prosody_emotion=sample.prosody_emotion,
            n_layers=n_layers,
            layer_logits=layer_logits,
            layer_margins=layer_margins,
            layer_predictions=layer_predictions,
            final_prediction=layer_predictions[-1],
            generated_token=generated_token,
            predicted_token=predicted_token,
            sanity_match=sanity_match,
        )

    def _sanity_check_generate(self, inputs: dict) -> Tuple[Optional[int], Optional[str]]:
        """Sanity check: 验证 generate 和 logits.argmax 一致"""
        from transformers import GenerationConfig

        gen_config = GenerationConfig(
            max_new_tokens=1,
            do_sample=False,
            temperature=0,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        with torch.no_grad():
            generated = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                speech_values=inputs["speech_values"],
                speech_mask=inputs["speech_mask"],
                spk_emb=None,
                generation_config=gen_config,
            )

        input_len = inputs["input_ids"].shape[1]
        if generated.shape[1] == 0:
            return None, None
        if generated.shape[1] <= input_len:
            gen_token_id = int(generated[0, 0].item())
        else:
            gen_token_id = int(generated[0, input_len].item())
        gen_token_str = self.tokenizer.decode([gen_token_id])
        return gen_token_id, gen_token_str

    @staticmethod
    def compute_metrics(results: List[LogitLensResult]) -> Dict[str, Any]:
        """
        计算聚合指标

        Args:
            results: LogitLensResult 列表
        Returns:
            包含各种聚合指标的字典
        """
        if not results:
            return {}

        n_layers = results[0].n_layers
        n_samples = len(results)

        # 聚合 margin 曲线
        all_margins = np.stack([r.layer_margins for r in results], axis=0)
        mean_margins = np.mean(all_margins, axis=0)
        std_margins = np.std(all_margins, axis=0)

        # 计算 win-rate
        win_semantic = np.zeros(n_layers)
        win_prosody = np.zeros(n_layers)
        win_other = np.zeros(n_layers)

        for r in results:
            if len(r.layer_predictions) != n_layers:
                raise ValueError("Inconsistent layer_predictions length across results.")
            for layer_idx, pred in enumerate(r.layer_predictions):
                if pred == r.semantic_emotion:
                    win_semantic[layer_idx] += 1
                elif pred == r.prosody_emotion:
                    win_prosody[layer_idx] += 1
                else:
                    win_other[layer_idx] += 1

        win_semantic /= n_samples
        win_prosody /= n_samples
        win_other /= n_samples

        return {
            'n_samples': n_samples,
            'n_layers': n_layers,
            'mean_margins': mean_margins,
            'std_margins': std_margins,
            'win_semantic': win_semantic,
            'win_prosody': win_prosody,
            'win_other': win_other,
        }
