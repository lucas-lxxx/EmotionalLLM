"""
Attack Core for Moshi Model
Adapted from OpenS2S attack framework

Note: This is a simplified version focusing on audio-level attacks.
Moshi's architecture differs from OpenS2S, so we adapt the attack strategy.
"""
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any

from config import cfg
from moshi_io import MoshiModel


def loss_per(waveform_adv: torch.Tensor, waveform_clean: torch.Tensor) -> torch.Tensor:
    """
    Perceptual loss using multi-resolution STFT

    Args:
        waveform_adv: Adversarial waveform
        waveform_clean: Clean waveform

    Returns:
        Perceptual loss value
    """
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

    losses = []
    for n_fft, hop, win in zip(cfg.per_fft_sizes, cfg.per_hop_sizes, cfg.per_win_lengths):
        mag_adv = _stft_mag(waveform_adv, n_fft, hop, win)
        mag_clean = _stft_mag(waveform_clean, n_fft, hop, win)
        losses.append((mag_adv - mag_clean).abs().mean())
    return torch.stack(losses).mean()


@dataclass
class EoTParams:
    """Expectation over Transformation parameters"""
    shift: int
    gain: float
    noise_std: float


def sample_eot_params(device: str) -> EoTParams:
    """Sample EoT transformation parameters"""
    shift = int(torch.randint(-cfg.eot_max_shift, cfg.eot_max_shift + 1, (1,), device=device).item())
    gain = float(torch.empty(1, device=device).uniform_(cfg.eot_gain_min, cfg.eot_gain_max).item())
    return EoTParams(shift=shift, gain=gain, noise_std=cfg.eot_noise_std)


def apply_eot(waveform: torch.Tensor, params: EoTParams) -> torch.Tensor:
    """Apply EoT transformations to waveform"""
    if params.shift != 0:
        waveform = torch.roll(waveform, shifts=params.shift, dims=-1)
    waveform = waveform * params.gain
    if params.noise_std > 0:
        waveform = waveform + params.noise_std * torch.randn_like(waveform)
    return waveform


def _stage_weights(step: int) -> tuple[float, float, float]:
    """Get loss weights for current step (two-stage schedule)"""
    if step < cfg.stage_a_steps:
        return cfg.lambda_emo, cfg.lambda_asr_stage_a, cfg.lambda_per_stage_a
    return cfg.lambda_emo, cfg.lambda_asr_stage_b, cfg.lambda_per_stage_b


def loss_emo_simple(
    model: MoshiModel,
    waveform: torch.Tensor,
    target_emotion: str = "happy"
) -> torch.Tensor:
    """
    Simplified emotion loss for Moshi

    Note: This is a placeholder. Moshi's emotion recognition capability
    needs to be explored. For now, we use a simple approach based on
    audio features or model outputs.

    Args:
        model: Moshi model
        waveform: Input waveform [1, 1, T]
        target_emotion: Target emotion string

    Returns:
        Loss value
    """
    # Encode audio to codes
    codes = model.mimi.encode(waveform)

    # Forward through Moshi LM
    # Note: This is simplified - actual implementation depends on Moshi's API
    outputs = model.moshi_lm(codes)

    # Placeholder loss - needs to be adapted based on Moshi's actual output
    # For now, we use a simple reconstruction-based loss
    loss = torch.tensor(0.0, device=waveform.device, requires_grad=True)

    return loss


def loss_asr_simple(
    model: MoshiModel,
    waveform: torch.Tensor,
    target_transcript: str = ""
) -> torch.Tensor:
    """
    Simplified ASR loss for semantic preservation

    Args:
        model: Moshi model
        waveform: Input waveform
        target_transcript: Target transcript for preservation

    Returns:
        Loss value
    """
    # Placeholder - needs actual implementation
    loss = torch.tensor(0.0, device=waveform.device, requires_grad=True)
    return loss


def attack_one_sample(
    model: MoshiModel,
    waveform: torch.Tensor,
    sr: int,
    target_emotion: str = "happy",
    device: str = "cuda"
) -> dict:
    """
    PGD attack on one audio sample

    Args:
        model: Moshi model
        waveform: Clean waveform [1, 1, T]
        sr: Sample rate
        target_emotion: Target emotion
        device: Device

    Returns:
        dict with waveform_adv, loss_trace, grad_trace
    """
    waveform = waveform.detach().to(device)
    waveform.requires_grad_(False)

    # Initialize perturbation
    delta = torch.zeros_like(waveform, requires_grad=True)

    # Optimizer
    if cfg.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam([delta], lr=cfg.lr)
    else:
        optimizer = torch.optim.SGD([delta], lr=cfg.lr)

    loss_trace = []
    grad_trace = []
    small_grad_steps = 0

    print(f"Starting attack: {cfg.total_steps} steps, epsilon={cfg.epsilon}")

    for step in range(cfg.total_steps):
        optimizer.zero_grad(set_to_none=True)

        lambda_emo, lambda_asr, lambda_per = _stage_weights(step)
        loss_emo_val = 0.0
        loss_asr_val = 0.0
        loss_per_val = 0.0

        # EoT loop
        for eot_idx in range(cfg.eot_samples):
            params = sample_eot_params(device)
            waveform_adv = torch.clamp(waveform + delta, -1.0, 1.0)
            waveform_adv = apply_eot(waveform_adv, params)
            waveform_clean = apply_eot(waveform, params)

            # Compute losses
            l_emo = loss_emo_simple(model, waveform_adv, target_emotion)
            l_asr = loss_asr_simple(model, waveform_adv)
            l_per = loss_per(waveform_adv, waveform_clean)

            loss_emo_val += float(l_emo.detach().item())
            loss_asr_val += float(l_asr.detach().item())
            loss_per_val += float(l_per.detach().item())

            # Backward
            sample_loss = (lambda_emo * l_emo + lambda_asr * l_asr + lambda_per * l_per) / float(cfg.eot_samples)
            sample_loss.backward()

            # Clear memory
            del l_emo, l_asr, l_per, sample_loss, waveform_adv, waveform_clean
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        # Check gradient
        if delta.grad is None:
            raise RuntimeError("delta.grad is None - gradient chain broken")

        grad_norm = float(delta.grad.detach().norm(p=2).item())
        grad_trace.append(grad_norm)

        if grad_norm < cfg.grad_norm_min:
            small_grad_steps += 1
            if small_grad_steps >= cfg.grad_norm_patience:
                print(f"Warning: Gradient norm too small at step {step}")
        else:
            small_grad_steps = 0

        # Update
        optimizer.step()
        delta.data.clamp_(-cfg.epsilon, cfg.epsilon)

        # Log
        total_loss_val = (
            lambda_emo * (loss_emo_val / float(cfg.eot_samples))
            + lambda_asr * (loss_asr_val / float(cfg.eot_samples))
            + lambda_per * (loss_per_val / float(cfg.eot_samples))
        )

        loss_trace.append({
            "step": step,
            "total": total_loss_val,
            "emo": loss_emo_val / float(cfg.eot_samples),
            "asr": loss_asr_val / float(cfg.eot_samples),
            "per": loss_per_val / float(cfg.eot_samples),
            "lambda_emo": lambda_emo,
            "lambda_asr": lambda_asr,
            "lambda_per": lambda_per,
        })

        if (step + 1) % 10 == 0:
            print(f"  Step {step+1}/{cfg.total_steps}: loss={total_loss_val:.4f}, grad_norm={grad_norm:.6f}")

    waveform_adv = torch.clamp(waveform + delta, -1.0, 1.0).detach()

    return {
        "waveform_adv": waveform_adv,
        "loss_trace": loss_trace,
        "grad_trace": grad_trace,
    }
