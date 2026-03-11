import sys
from pathlib import Path
from typing import Any

import torch

from config import cfg

try:
    from src.constants import (
        AUDIO_TOKEN_INDEX,
        DEFAULT_AUDIO_TOKEN,
        DEFAULT_AUDIO_START_TOKEN,
        DEFAULT_AUDIO_END_TOKEN,
        DEFAULT_TTS_START_TOKEN,
        IGNORE_INDEX,
    )
except Exception:
    # Fallback if OpenS2S is not on sys.path yet.
    AUDIO_TOKEN_INDEX = -200
    DEFAULT_AUDIO_TOKEN = "<|im_audio|>"
    DEFAULT_AUDIO_START_TOKEN = "<|im_audio_start|>"
    DEFAULT_AUDIO_END_TOKEN = "<|im_audio_end|>"
    DEFAULT_TTS_START_TOKEN = "<|im_tts_start|>"
    IGNORE_INDEX = -100


def _ensure_opens2s_on_path(opens2s_root: Path) -> None:
    if str(opens2s_root) not in sys.path:
        sys.path.insert(0, str(opens2s_root))


def _cuda_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        torch.cuda.get_device_properties(0)
    except Exception:
        return False
    return True


class TorchWhisperFeatureExtractor:
    def __init__(
        self,
        feature_size: int,
        sampling_rate: int,
        hop_length: int,
        chunk_length: int,
        n_fft: int,
        dither: float,
    ):
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.n_fft = n_fft
        self.dither = dither
        self.n_samples = self.chunk_length * self.sampling_rate
        self._mel_filters = {}

    def _get_mel_filters(self, device: torch.device) -> torch.Tensor:
        cached = self._mel_filters.get(device)
        if cached is not None:
            return cached
        try:
            from transformers.audio_utils import mel_filter_bank
        except Exception as exc:
            raise RuntimeError("transformers.audio_utils.mel_filter_bank not available.") from exc

        mel = mel_filter_bank(
            num_frequency_bins=1 + self.n_fft // 2,
            num_mel_filters=self.feature_size,
            min_frequency=0.0,
            max_frequency=self.sampling_rate / 2.0,
            sampling_rate=self.sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        mel_t = torch.from_numpy(mel).to(device=device, dtype=torch.float32)
        self._mel_filters[device] = mel_t
        return mel_t

    def __call__(self, waveform: torch.Tensor, sr: int) -> tuple[torch.Tensor, torch.Tensor]:
        if sr != self.sampling_rate:
            raise ValueError(f"Expected sampling rate {self.sampling_rate}, got {sr}.")

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        waveform = waveform.to(dtype=torch.float32)

        batch, length = waveform.shape
        if length >= self.n_samples:
            waveform = waveform[:, : self.n_samples]
            attn = torch.ones(batch, self.n_samples, device=waveform.device, dtype=torch.long)
        else:
            pad_len = self.n_samples - length
            pad = torch.zeros(batch, pad_len, device=waveform.device, dtype=waveform.dtype)
            waveform = torch.cat([waveform, pad], dim=1)
            attn = torch.cat(
                [
                    torch.ones(batch, length, device=waveform.device, dtype=torch.long),
                    torch.zeros(batch, pad_len, device=waveform.device, dtype=torch.long),
                ],
                dim=1,
            )

        if self.dither != 0.0:
            waveform = waveform + self.dither * torch.randn_like(waveform)

        window = torch.hann_window(self.n_fft, device=waveform.device)
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
        )
        magnitudes = stft[..., :-1].abs() ** 2
        mel_filters = self._get_mel_filters(waveform.device)
        mel_spec = mel_filters.T @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        if batch > 1:
            max_val = log_spec.amax(dim=2, keepdim=True).amax(dim=1, keepdim=True)
            log_spec = torch.maximum(log_spec, max_val - 8.0)
        else:
            log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        attn_frames = attn[:, :: self.hop_length]
        if attn.shape[1] % self.hop_length != 0:
            attn_frames = attn_frames[:, :-1]

        return log_spec, attn_frames


def load_opens2s(model_path: Path, device: str, opens2s_root: Path) -> tuple[Any, Any, Any, TorchWhisperFeatureExtractor]:
    """
    Load OpenS2S model + tokenizer.
    Methodology §4.1: reuse OpenS2S input conventions.
    """
    _ensure_opens2s_on_path(opens2s_root)
    try:
        from src.modeling_omnispeech import OmniSpeechModel  # type: ignore
        from transformers import AutoTokenizer, WhisperFeatureExtractor
    except Exception as exc:
        raise RuntimeError("OpenS2S imports failed. Check opens2s_root and dependencies.") from exc

    # Explicitly set CUDA device before model loading
    if device.startswith("cuda") and _cuda_available():
        if ":" in device:
            torch.cuda.set_device(int(device.split(":")[1]))
        else:
            torch.cuda.set_device(0)

        # Clear cache before loading
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
    except TypeError:
         tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = OmniSpeechModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        device_map=None,
    )
    model = model.to(device)
    target_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = model.to(dtype=target_dtype)
    model.eval()

    # Freeze model params: attack only needs grad w.r.t. input delta, not model weights.
    # This avoids storing ~14GB of useless param gradients during backward.
    model.requires_grad_(False)

    
    # Enable gradient checkpointing to save memory
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except ValueError:
            # Fallback for models that don't support it directly
            if hasattr(model, "llm_model") and hasattr(model.llm_model, "gradient_checkpointing_enable"):
                 model.llm_model.gradient_checkpointing_enable()
                 print("Enabled gradient checkpointing on llm_model")
            
            if hasattr(model, "audio_encoder_model") and hasattr(model.audio_encoder_model, "gradient_checkpointing_enable"):
                 model.audio_encoder_model.gradient_checkpointing_enable()
                 print("Enabled gradient checkpointing on audio_encoder_model")

            else:
                 print("Could not enable gradient checkpointing completely")
        
    audio_extractor = WhisperFeatureExtractor.from_pretrained(Path(model_path) / "audio")
    torch_extractor = TorchWhisperFeatureExtractor(
        feature_size=audio_extractor.feature_size,
        sampling_rate=audio_extractor.sampling_rate,
        hop_length=audio_extractor.hop_length,
        chunk_length=audio_extractor.chunk_length,
        n_fft=audio_extractor.n_fft,
        dither=audio_extractor.dither,
    )

    return model, tokenizer, audio_extractor, torch_extractor


def _build_input_ids(tokenizer, prompt: str, system_prompt: str | None) -> torch.LongTensor:
    # Methodology §4.1: OpenS2S chat template + audio token insertion.
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": f"{DEFAULT_AUDIO_START_TOKEN}{DEFAULT_AUDIO_TOKEN}{DEFAULT_AUDIO_END_TOKEN}{prompt}",
        }
    )
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
    except TypeError:
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    prompt_text += DEFAULT_TTS_START_TOKEN

    segments = prompt_text.split(f"{DEFAULT_AUDIO_TOKEN}")
    input_ids = []
    for idx, segment in enumerate(segments):
        if idx != 0:
            input_ids.append(AUDIO_TOKEN_INDEX)
        input_ids.extend(tokenizer.encode(segment))
    return torch.LongTensor(input_ids).unsqueeze(0)


def _extract_official_features(audio_extractor, waveform: torch.Tensor, sr: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    wav = waveform.detach().cpu().numpy().squeeze()
    outputs = audio_extractor(
        wav,
        sampling_rate=sr,
        return_attention_mask=True,
        return_tensors="pt",
    )
    speech_values = outputs.input_features.to(device)
    speech_mask = outputs.attention_mask.to(device)
    return speech_values, speech_mask


def build_inputs(
    waveform: torch.Tensor,
    sr: int,
    prompt: str,
    tokenizer,
    device: str,
    audio_extractor=None,
    torch_extractor: TorchWhisperFeatureExtractor | None = None,
    differentiable: bool = False,
    system_prompt: str | None = None,
    dtype: torch.dtype | None = None,
) -> dict:
    """
    Build inputs for OpenS2S forward.
    Methodology §4.1: (input_ids, speech_values, speech_mask) are required.
    """
    input_ids = _build_input_ids(tokenizer, prompt, system_prompt).to(device)

    if differentiable:
        if torch_extractor is None:
            raise ValueError("torch_extractor is required for differentiable feature extraction.")
        speech_values, speech_mask = torch_extractor(waveform, sr)
    else:
        if audio_extractor is None:
            raise ValueError("audio_extractor is required for inference feature extraction.")
        speech_values, speech_mask = _extract_official_features(audio_extractor, waveform, sr, device)

    speech_values = speech_values.to(device)
    speech_mask = speech_mask.to(device)
    if dtype is not None:
        speech_values = speech_values.to(dtype=dtype)

    attention_mask = torch.ones_like(input_ids, device=device)

    if differentiable and not speech_values.requires_grad:
        speech_values.requires_grad_(True)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "speech_values": speech_values,
        "speech_mask": speech_mask,
    }


def forward_logits(model, inputs: dict) -> Any:
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        speech_values=inputs.get("speech_values"),
        speech_mask=inputs.get("speech_mask"),
        labels=inputs.get("labels"),
        token_types=None,
        speech_units=None,
        speech_units_mask=None,
        spk_embs=None,
        return_dict=True,
    )
    return outputs


def decode_text(
    model,
    tokenizer,
    waveform: torch.Tensor,
    sr: int,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    audio_extractor=None,
    system_prompt: str | None = None,
) -> str:
    # Methodology §8.1: temperature=0, greedy decode.
    inputs = build_inputs(
        waveform,
        sr,
        prompt,
        tokenizer,
        waveform.device,
        audio_extractor=audio_extractor,
        torch_extractor=None,
        differentiable=False,
        system_prompt=system_prompt,
        dtype=None,
    )
    try:
        model_dtype = next(model.parameters()).dtype
        inputs["speech_values"] = inputs["speech_values"].to(dtype=model_dtype)
    except StopIteration:
        pass

    do_sample = temperature > 0.001
    
    from transformers import GenerationConfig
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    generated = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        speech_values=inputs.get("speech_values"),
        speech_mask=inputs.get("speech_mask"),
        spk_emb=None,
        generation_config=gen_config,
    )

    # OpenS2S model.generate() returns ONLY generated tokens, not input+generated
    # So we don't need to slice
    gen_tokens = generated
    return tokenizer.decode(gen_tokens[0], skip_special_tokens=True).strip()
