"""MERaLiON-2-3B model I/O with differentiable audio pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg


# ---------------------------------------------------------------------------
# Differentiable Whisper mel-spectrogram extractor
# ---------------------------------------------------------------------------
class TorchWhisperFeatureExtractor(nn.Module):
    """
    Differentiable reimplementation of WhisperFeatureExtractor.
    Computes log-mel spectrogram from raw waveform using PyTorch ops,
    allowing gradient flow for adversarial attacks.
    """

    def __init__(
        self,
        feature_size: int = 128,
        sampling_rate: int = 16000,
        hop_length: int = 160,
        chunk_length: int = 30,
        n_fft: int = 400,
        dither: float = 0.0,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.n_fft = n_fft
        self.n_samples = chunk_length * sampling_rate
        self.nb_max_frames = self.n_samples // hop_length
        self.dither = dither

        mel_filters = self._build_mel_filters()
        self.register_buffer("mel_filters", mel_filters)

    def _build_mel_filters(self) -> torch.Tensor:
        """Returns mel filter bank of shape (freq_bins, n_mels) = (n_fft//2+1, feature_size)."""
        from transformers.audio_utils import mel_filter_bank

        filters = mel_filter_bank(
            num_frequency_bins=1 + self.n_fft // 2,
            num_mel_filters=self.feature_size,
            min_frequency=0.0,
            max_frequency=self.sampling_rate / 2.0,
            sampling_rate=self.sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        return torch.from_numpy(np.array(filters)).float()

    def _get_mel_filters(self, device: torch.device) -> torch.Tensor:
        return self.mel_filters.to(device)

    def forward(
        self,
        waveform: torch.Tensor,
        sr: int,
        do_normalize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            waveform: (batch, samples) or (samples,)
            sr: sample rate
            do_normalize: zero-mean unit-variance normalization (MERaLiON default)

        Returns:
            log_spec: (batch, n_mels, n_frames)
            attn: (batch, n_samples) sample-level attention mask
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        batch, length = waveform.shape

        # Zero-mean unit-variance normalization (differentiable)
        if do_normalize:
            mean = waveform.mean(dim=-1, keepdim=True)
            std = waveform.std(dim=-1, keepdim=True)
            waveform = (waveform - mean) / (std + 1e-7)

        # Build sample-level attention mask BEFORE padding
        if length < self.n_samples:
            pad_len = self.n_samples - length
            attn = torch.cat(
                [
                    torch.ones(batch, length, device=waveform.device, dtype=torch.long),
                    torch.zeros(batch, pad_len, device=waveform.device, dtype=torch.long),
                ],
                dim=1,
            )
            waveform = F.pad(waveform, (0, pad_len))
        else:
            waveform = waveform[:, : self.n_samples]
            attn = torch.ones(batch, self.n_samples, device=waveform.device, dtype=torch.long)

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

        return log_spec, attn


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_meralion(
    model_path: Path, device: str
) -> tuple[Any, Any, TorchWhisperFeatureExtractor]:
    """Load MERaLiON-2-3B model + processor + differentiable feature extractor."""
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    if device.startswith("cuda"):
        idx = int(device.split(":")[1]) if ":" in device else 0
        torch.cuda.set_device(idx)
        torch.cuda.empty_cache()

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path,
        use_safetensors=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        device_map=None,
        attn_implementation="eager",  # avoid transformers>=4.52 SDPA-init bug with MERaLiON custom code
    )
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)

    # Enable gradient checkpointing for memory efficiency
    for sub_name in ["speech_encoder", "text_decoder"]:
        sub = getattr(model, sub_name, None)
        if sub is not None and hasattr(sub, "gradient_checkpointing_enable"):
            try:
                sub.gradient_checkpointing_enable()
                print(f"Enabled gradient checkpointing on {sub_name}")
            except Exception:
                pass

    # Build differentiable feature extractor from processor params
    fe = processor.feature_extractor
    torch_extractor = TorchWhisperFeatureExtractor(
        feature_size=fe.feature_size,
        sampling_rate=fe.sampling_rate,
        hop_length=fe.hop_length,
        chunk_length=fe.chunk_length,
        n_fft=fe.n_fft,
        dither=getattr(fe, "dither", 0.0),
    )

    return model, processor, torch_extractor


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = "Instruction: {query} \nFollow the text instruction based on the following audio: <SpeechHere>"


def build_input_ids(processor, query: str) -> torch.LongTensor:
    """Build input_ids with speech tokens for a single-chunk audio (<30s)."""
    prompt_text = PROMPT_TEMPLATE.format(query=query)
    conversation = [{"role": "user", "content": prompt_text}]
    chat_text = processor.tokenizer.apply_chat_template(
        conversation=conversation, tokenize=False, add_generation_prompt=True
    )
    # Expand <SpeechHere> -> speech_token * fixed_speech_embeds_length (100)
    speech_token = processor.speech_token
    expanded = chat_text.replace(speech_token, speech_token * processor.fixed_speech_embeds_length)
    tokens = processor.tokenizer(expanded, return_tensors="pt", add_special_tokens=False)
    return tokens.input_ids


def build_inputs(
    waveform: torch.Tensor,
    sr: int,
    prompt: str,
    processor,
    device: str,
    torch_extractor: TorchWhisperFeatureExtractor | None = None,
    differentiable: bool = False,
    dtype: torch.dtype | None = None,
) -> dict:
    """Build model inputs for MERaLiON forward pass."""
    input_ids = build_input_ids(processor, prompt).to(device)

    if differentiable:
        if torch_extractor is None:
            raise ValueError("torch_extractor required for differentiable mode")
        input_features, attn = torch_extractor(waveform, sr, do_normalize=True)
        input_features = input_features.to(device)
        if dtype is not None:
            input_features = input_features.to(dtype=dtype)
        if not input_features.requires_grad:
            input_features.requires_grad_(True)
        feature_attention_mask = attn.to(device)
    else:
        wav_np = waveform.detach().cpu().numpy().squeeze()
        audio_inputs = processor.feature_extractor(
            [wav_np],
            sampling_rate=sr,
            return_tensors="pt",
            return_attention_mask=True,
            padding="max_length",
            do_normalize=getattr(processor, "do_normalize", True),
        )
        input_features = audio_inputs.input_features.to(device)
        feature_attention_mask = audio_inputs.attention_mask.to(device)
        if dtype is not None:
            input_features = input_features.to(dtype=dtype)

    attention_mask = torch.ones_like(input_ids)

    return {
        "input_ids": input_ids,
        "input_features": input_features,
        "feature_attention_mask": feature_attention_mask,
        "attention_mask": attention_mask,
    }


# ---------------------------------------------------------------------------
# Forward / decode
# ---------------------------------------------------------------------------
def forward_logits(model, inputs: dict) -> Any:
    """Forward pass returning model outputs (with loss if labels provided)."""
    return model(
        input_ids=inputs["input_ids"],
        input_features=inputs.get("input_features"),
        feature_attention_mask=inputs.get("feature_attention_mask"),
        attention_mask=inputs.get("attention_mask"),
        labels=inputs.get("labels"),
        return_dict=True,
    )


def decode_text(
    model,
    processor,
    waveform: torch.Tensor,
    sr: int,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Inference-only decode."""
    inputs = build_inputs(
        waveform, sr, prompt, processor, str(waveform.device),
        differentiable=False,
    )

    try:
        model_dtype = next(model.parameters()).dtype
        inputs["input_features"] = inputs["input_features"].to(dtype=model_dtype)
    except StopIteration:
        pass

    from transformers import GenerationConfig

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0.001,
        temperature=temperature,
        use_cache=True,
        pad_token_id=processor.tokenizer.eos_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )

    generated = model.generate(
        input_ids=inputs["input_ids"],
        input_features=inputs["input_features"],
        feature_attention_mask=inputs["feature_attention_mask"],
        attention_mask=inputs["attention_mask"],
        generation_config=gen_config,
    )

    input_len = inputs["input_ids"].shape[1]
    gen_tokens = generated[:, input_len:]
    return processor.tokenizer.decode(gen_tokens[0], skip_special_tokens=True).strip()
