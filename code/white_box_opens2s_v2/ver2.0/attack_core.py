from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from config import cfg
from opens2s_io import build_inputs, forward_logits


def compute_target_token_ids(tokenizer, target_str: str) -> list[int]:
    # Methodology §5.1: target emotion token ids for "happy".
    tokens = tokenizer.encode(target_str, add_special_tokens=False)
    if len(tokens) == 0:
        tokens = tokenizer.encode(" " + target_str, add_special_tokens=False)
    return tokens


def _attach_labels(inputs: dict, target_token_ids: list[int], ignore_index: int) -> dict:
    input_ids = inputs["input_ids"]
    device = input_ids.device
    target = torch.tensor(target_token_ids, device=device, dtype=input_ids.dtype).unsqueeze(0)
    new_input_ids = torch.cat([input_ids, target], dim=1)

    labels = torch.full_like(new_input_ids, ignore_index)
    labels[:, -target.shape[1]:] = target

    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(new_input_ids)
    else:
        attention_mask = torch.cat([attention_mask, torch.ones_like(target)], dim=1)

    out = dict(inputs)
    out["input_ids"] = new_input_ids
    out["attention_mask"] = attention_mask
    out["labels"] = labels
    return out


def _masked_ce_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int) -> torch.Tensor:
    # Causal LM shift (standard teacher forcing).
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
        reduction="mean",
    )
    return loss


def loss_emo(
    model,
    tokenizer,
    waveform: torch.Tensor,
    sr: int,
    prompts: list[str],
    target_token_ids: list[int],
    device: str,
    ignore_index: int,
    torch_extractor,
    system_prompt: str | None,
    dtype: torch.dtype | None,
) -> torch.Tensor:
    # Methodology §5.1: token-level CE for target emotion ("happy").
    losses = []
    for prompt in prompts:
        inputs = build_inputs(
            waveform,
            sr,
            prompt,
            tokenizer,
            device,
            audio_extractor=None,
            torch_extractor=torch_extractor,
            differentiable=True,
            system_prompt=system_prompt,
            dtype=dtype,
        )
        inputs = _attach_labels(inputs, target_token_ids, ignore_index)
        outputs = forward_logits(model, inputs)
        losses.append(outputs.loss)
    return torch.stack(losses).mean()


def loss_asr(
    model,
    tokenizer,
    waveform: torch.Tensor,
    sr: int,
    prompt: str,
    target_asr_token_ids: list[int],
    device: str,
    ignore_index: int,
    torch_extractor,
    system_prompt: str | None,
    dtype: torch.dtype | None,
) -> torch.Tensor:
    # Methodology §5.2: OpenS2S self-consistency transcription loss.
    inputs = build_inputs(
        waveform,
        sr,
        prompt,
        tokenizer,
        device,
        audio_extractor=None,
        torch_extractor=torch_extractor,
        differentiable=True,
        system_prompt=system_prompt,
        dtype=dtype,
    )
    inputs = _attach_labels(inputs, target_asr_token_ids, ignore_index)
    outputs = forward_logits(model, inputs)
    return outputs.loss


def _stft_mag(waveform: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
    window = torch.hann_window(win, device=waveform.device)
    spec = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        window=window,
        return_complex=True,
    )
    return spec.abs()


def loss_per(waveform_adv: torch.Tensor, waveform_clean: torch.Tensor) -> torch.Tensor:
    # Methodology §5.3: multi-res STFT magnitude L1.
    losses = []
    for n_fft, hop, win in zip(cfg.per_fft_sizes, cfg.per_hop_sizes, cfg.per_win_lengths):
        mag_adv = _stft_mag(waveform_adv, n_fft, hop, win)
        mag_clean = _stft_mag(waveform_clean, n_fft, hop, win)
        losses.append((mag_adv - mag_clean).abs().mean())
    return torch.stack(losses).mean()


@dataclass
class EoTParams:
    shift: int
    gain: float
    noise_std: float


def sample_eot_params(device: str) -> EoTParams:
    # Methodology §6: EoT sampling.
    shift = int(torch.randint(-cfg.eot_max_shift, cfg.eot_max_shift + 1, (1,), device=device).item())
    gain = float(torch.empty(1, device=device).uniform_(cfg.eot_gain_min, cfg.eot_gain_max).item())
    return EoTParams(shift=shift, gain=gain, noise_std=cfg.eot_noise_std)


def apply_eot(waveform: torch.Tensor, params: EoTParams) -> torch.Tensor:
    # Methodology §6: differentiable time shift + gain (+ optional noise).
    if params.shift != 0:
        waveform = torch.roll(waveform, shifts=params.shift, dims=-1)
    waveform = waveform * params.gain
    if params.noise_std > 0:
        waveform = waveform + params.noise_std * torch.randn_like(waveform)
    if cfg.eot_band_limit:
        # TODO: implement differentiable band-limit transform if needed.
        waveform = waveform
    return waveform


def _stage_weights(step: int) -> tuple[float, float, float]:
    # Methodology §7.2: two-stage weight schedule.
    if step < cfg.stage_a_steps:
        return cfg.lambda_emo, cfg.lambda_asr_stage_a, cfg.lambda_per_stage_a
    return cfg.lambda_emo, cfg.lambda_asr_stage_b, cfg.lambda_per_stage_b


def attack_one_sample(
    model: Any,
    tokenizer: Any,
    waveform: torch.Tensor,
    sr: int,
    target_token_ids: list[int],
    asr_prompt: str,
    asr_target_token_ids: list[int],
    device: str,
    ignore_index: int,
    torch_extractor,
    system_prompt: str | None,
) -> dict:
    """
    Returns: dict with waveform_adv + trace.
    """
    waveform = waveform.detach().to(device)
    waveform.requires_grad_(True)

    delta = torch.zeros_like(waveform, requires_grad=True)
    if cfg.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam([delta], lr=cfg.lr)
    else:
        optimizer = torch.optim.SGD([delta], lr=cfg.lr)

    loss_trace = []
    grad_trace = []
    small_grad_steps = 0
    checked_grad_link = False

    model_dtype = None
    try:
        model_dtype = next(model.parameters()).dtype
    except StopIteration:
        model_dtype = None

    for step in range(cfg.total_steps):
        optimizer.zero_grad(set_to_none=True)

        lambda_emo, lambda_asr, lambda_per = _stage_weights(step)
        loss_emo_val = 0.0
        loss_asr_val = 0.0
        loss_per_val = 0.0

        # Use gradient accumulation to reduce memory peak
        for eot_idx in range(cfg.eot_samples):
            params = sample_eot_params(device)
            waveform_adv = torch.clamp(waveform + delta, -1.0, 1.0)
            waveform_adv = apply_eot(waveform_adv, params)
            waveform_clean = apply_eot(waveform, params)

            l_emo = loss_emo(
                model,
                tokenizer,
                waveform_adv,
                sr,
                cfg.emo_prompts,
                target_token_ids,
                device,
                ignore_index,
                torch_extractor,
                system_prompt,
                model_dtype,
            )
            l_asr = loss_asr(
                model,
                tokenizer,
                waveform_adv,
                sr,
                asr_prompt,
                asr_target_token_ids,
                device,
                ignore_index,
                torch_extractor,
                system_prompt,
                model_dtype,
            )
            l_per = loss_per(waveform_adv, waveform_clean)

            loss_emo_val += float(l_emo.detach().item())
            loss_asr_val += float(l_asr.detach().item())
            loss_per_val += float(l_per.detach().item())

            # Scale loss and backward immediately to reduce memory
            sample_loss = (lambda_emo * l_emo + lambda_asr * l_asr + lambda_per * l_per) / float(cfg.eot_samples)
            sample_loss.backward()

            # Clear intermediate tensors
            del l_emo, l_asr, l_per, sample_loss, waveform_adv, waveform_clean
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        if delta.grad is None:
             raise RuntimeError("delta.grad is None after backward. Gradient chain broken.")

        grad_norm = float(delta.grad.detach().norm(p=2).item())
        grad_trace.append(grad_norm)
        if grad_norm < cfg.grad_norm_min:
            small_grad_steps += 1
            if small_grad_steps >= cfg.grad_norm_patience:
                raise RuntimeError("Grad norm too small; check gradient chain (Methodology §4.2).")
        else:
            small_grad_steps = 0

        optimizer.step()
        delta.data.clamp_(-cfg.epsilon, cfg.epsilon)

        # Calculate average loss for logging
        total_loss_val = (
            lambda_emo * (loss_emo_val / float(cfg.eot_samples))
            + lambda_asr * (loss_asr_val / float(cfg.eot_samples))
            + lambda_per * (loss_per_val / float(cfg.eot_samples))
        )

        loss_trace.append(
            {
                "step": step,
                "total": total_loss_val,
                "emo": loss_emo_val / float(cfg.eot_samples),
                "asr": loss_asr_val / float(cfg.eot_samples),
                "per": loss_per_val / float(cfg.eot_samples),
                "lambda_emo": lambda_emo,
                "lambda_asr": lambda_asr,
                "lambda_per": lambda_per,
            }
        )

    waveform_adv = torch.clamp(waveform + delta, -1.0, 1.0).detach()
    return {
        "waveform_adv": waveform_adv,
        "loss_trace": loss_trace,
        "grad_trace": grad_trace,
    }
