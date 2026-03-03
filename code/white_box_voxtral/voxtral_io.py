from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from config import cfg


class TorchWhisperFeatureExtractor:
    """Differentiable Whisper mel-spectrogram extractor (identical to OpenS2S version)."""

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
        from transformers.audio_utils import mel_filter_bank

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


def load_voxtral(model_path: Path, device: str) -> tuple[Any, Any, TorchWhisperFeatureExtractor]:
    """
    Load Voxtral model + processor.
    Returns: (model, processor, torch_extractor)
    """
    from transformers import VoxtralForConditionalGeneration, AutoProcessor

    if device.startswith("cuda"):
        if ":" in device:
            torch.cuda.set_device(int(device.split(":")[1]))
        else:
            torch.cuda.set_device(0)
        torch.cuda.empty_cache()

    processor = AutoProcessor.from_pretrained(model_path)
    model = VoxtralForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        device_map=None,
    )
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)

    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            for sub in ["language_model", "audio_encoder"]:
                m = getattr(model, sub, None)
                if m is not None and hasattr(m, "gradient_checkpointing_enable"):
                    m.gradient_checkpointing_enable()
                    print(f"Enabled gradient checkpointing on {sub}")

    # Build torch extractor from processor's feature extractor params
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


def _validate_input_ids_once(processor, prompt: str) -> None:
    """One-time validation: compare manual build_input_ids vs processor template."""
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    ref_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    ref_ids = processor.tokenizer.encode(ref_text, add_special_tokens=False)

    manual_ids = build_input_ids(processor.tokenizer, prompt).squeeze(0).tolist()
    # Strip audio tokens from manual for comparison of text structure
    manual_text_ids = [t for t in manual_ids if t not in (cfg.audio_token_id, cfg.begin_audio_id)]

    # The ref should NOT have audio tokens either (text-only template)
    if manual_text_ids != ref_ids:
        print(f"WARNING: input_ids mismatch!\n  manual text ids: {manual_text_ids[:20]}...\n  ref ids: {ref_ids[:20]}...")
    else:
        print("input_ids validation passed.")


_validated = False


def build_input_ids(tokenizer, prompt: str) -> torch.LongTensor:
    """
    Manually construct input_ids for Voxtral.
    Format: [BOS=1][INST=3][BEGIN_AUDIO=25][AUDIO=24]*375 <text_tokens> [INST_END=4]
    """
    text_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    ids = (
        [cfg.bos_id, cfg.inst_id, cfg.begin_audio_id]
        + [cfg.audio_token_id] * cfg.n_audio_tokens
        + text_tokens
        + [cfg.inst_end_id]
    )
    return torch.LongTensor(ids).unsqueeze(0)


def build_inputs(
    waveform: torch.Tensor,
    sr: int,
    prompt: str,
    tokenizer,
    device: str,
    torch_extractor: TorchWhisperFeatureExtractor | None = None,
    differentiable: bool = False,
    dtype: torch.dtype | None = None,
    processor=None,
) -> dict:
    """
    Build inputs for Voxtral forward.
    differentiable=True: use TorchWhisperFeatureExtractor for gradient flow.
    differentiable=False: use processor for inference.
    """
    input_ids = build_input_ids(tokenizer, prompt).to(device)

    if differentiable:
        if torch_extractor is None:
            raise ValueError("torch_extractor required for differentiable mode")
        input_features, _ = torch_extractor(waveform, sr)
        # Voxtral expects input_features shape: (batch, n_mels, time)
        input_features = input_features.to(device)
        if dtype is not None:
            input_features = input_features.to(dtype=dtype)
        if not input_features.requires_grad:
            input_features.requires_grad_(True)
    else:
        if processor is None:
            raise ValueError("processor required for inference mode")
        wav_np = waveform.detach().cpu().numpy().squeeze()
        feat = processor.feature_extractor(
            wav_np, sampling_rate=sr, return_tensors="pt"
        )
        input_features = feat.input_features.to(device)
        if dtype is not None:
            input_features = input_features.to(dtype=dtype)

    attention_mask = torch.ones_like(input_ids, device=device)

    return {
        "input_ids": input_ids,
        "input_features": input_features,
        "attention_mask": attention_mask,
    }


def forward_logits(model, inputs: dict) -> Any:
    outputs = model(
        input_ids=inputs["input_ids"],
        input_features=inputs.get("input_features"),
        attention_mask=inputs.get("attention_mask"),
        labels=inputs.get("labels"),
        return_dict=True,
    )
    return outputs


def decode_text(
    model,
    processor,
    waveform: torch.Tensor,
    sr: int,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Inference-only decode: processor handles audio, model.generate, slice output."""
    global _validated
    if not _validated:
        _validate_input_ids_once(processor, prompt)
        _validated = True

    inputs = build_inputs(
        waveform, sr, prompt, processor.tokenizer, str(waveform.device),
        differentiable=False, processor=processor,
    )

    try:
        model_dtype = next(model.parameters()).dtype
        inputs["input_features"] = inputs["input_features"].to(dtype=model_dtype)
    except StopIteration:
        pass

    do_sample = temperature > 0.001

    from transformers import GenerationConfig
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        use_cache=True,
        pad_token_id=processor.tokenizer.eos_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )

    generated = model.generate(
        input_ids=inputs["input_ids"],
        input_features=inputs["input_features"],
        attention_mask=inputs["attention_mask"],
        generation_config=gen_config,
    )

    # Voxtral generate returns full sequence; slice off input
    input_len = inputs["input_ids"].shape[1]
    gen_tokens = generated[:, input_len:]
    return processor.tokenizer.decode(gen_tokens[0], skip_special_tokens=True).strip()
