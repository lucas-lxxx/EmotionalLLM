"""Step 2b: 训练 STAA-Net Generator

冻结 surrogate SER，只训练 Wave-U-Net generator。
Loss = L_adv(C&W) + λ_mag·L_mag + λ_spa·L_spa + λ_qua·L_qua

论文默认超参数，不做调优。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import cfg
from esd_en_dataset import ESDDataset, collate_fn
from surrogate_ser import SurrogateSER
from wave_unet import WaveUNetGenerator


# ── Loss functions ──

def cw_untargeted_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    confidence: float = 0.0,
) -> torch.Tensor:
    """C&W untargeted loss: max(Z_true - max_{j≠true} Z_j, -κ)

    最小化此 loss 使 true class logit 低于最强 competitor。
    """
    batch_size, num_classes = logits.shape

    # Z_true
    z_true = logits.gather(1, labels.unsqueeze(1)).squeeze(1)  # (batch,)

    # max_{j≠true} Z_j
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(1, labels.unsqueeze(1), False)
    z_other_max = logits.masked_fill(~mask, float("-inf")).max(dim=1).values  # (batch,)

    loss = torch.clamp(z_true - z_other_max, min=-confidence)
    return loss.mean()


def magnitude_loss(v: torch.Tensor) -> torch.Tensor:
    """L_mag = ||v||_2"""
    return v.norm(p=2)


def sparsity_loss(m: torch.Tensor) -> torch.Tensor:
    """L_spa = ||m||_1"""
    return m.norm(p=1)


def quantization_loss(m_soft: torch.Tensor) -> torch.Tensor:
    """L_qua = ||m_soft - round(m_soft)||_2

    鼓励 mask 值接近 0 或 1（binary）。
    """
    m_hard = (m_soft >= 0.5).float()
    return (m_soft - m_hard).norm(p=2)


# ── Training ──

def train_one_epoch(
    generator: WaveUNetGenerator,
    ser_model: SurrogateSER,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    eps: float,
    device: str,
) -> dict:
    generator.train()
    ser_model.eval()

    total_loss_sum = 0.0
    adv_loss_sum = 0.0
    attack_success = 0
    total = 0

    for waveforms, labels, _, _ in loader:
        waveforms = waveforms.to(device)  # (batch, time)
        labels = labels.to(device)

        # Generator forward
        x_adv, v, m = generator(waveforms, eps, training=True)

        # x_adv shape: (batch, 1, time) → squeeze for SER
        x_adv_flat = x_adv.squeeze(1)  # (batch, time)

        # SER forward (frozen)
        with torch.no_grad():
            logits_clean = ser_model(waveforms)
        logits_adv = ser_model(x_adv_flat)

        # Losses
        l_adv = cw_untargeted_loss(logits_adv, labels, cfg.cw_confidence)
        l_mag = magnitude_loss(v)
        l_spa = sparsity_loss(m)
        l_qua = quantization_loss(m)

        loss = l_adv + cfg.lambda_mag * l_mag + cfg.lambda_spa * l_spa + cfg.lambda_qua * l_qua

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss_sum += loss.item() * labels.size(0)
        adv_loss_sum += l_adv.item() * labels.size(0)
        pred_adv = logits_adv.argmax(dim=-1)
        attack_success += (pred_adv != labels).sum().item()
        total += labels.size(0)

    return {
        "loss": total_loss_sum / total,
        "adv_loss": adv_loss_sum / total,
        "attack_success_rate": attack_success / total,
    }


@torch.no_grad()
def evaluate_generator(
    generator: WaveUNetGenerator,
    ser_model: SurrogateSER,
    loader: DataLoader,
    eps: float,
    device: str,
) -> dict:
    generator.eval()
    ser_model.eval()

    attack_success = 0
    total = 0

    for waveforms, labels, _, _ in loader:
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        x_adv, v, m = generator(waveforms, eps, training=False)
        x_adv_flat = x_adv.squeeze(1)
        logits_adv = ser_model(x_adv_flat)
        pred_adv = logits_adv.argmax(dim=-1)

        attack_success += (pred_adv != labels).sum().item()
        total += labels.size(0)

    return {"attack_success_rate": attack_success / total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esd_root", type=str, default=str(cfg.esd_root))
    parser.add_argument("--surrogate_ckpt", type=str, default=str(cfg.surrogate_ckpt))
    parser.add_argument("--output", type=str, default=str(cfg.generator_ckpt))
    parser.add_argument("--epochs", type=int, default=cfg.gen_epochs)
    parser.add_argument("--batch_size", type=int, default=cfg.gen_batch_size)
    parser.add_argument("--lr", type=float, default=cfg.gen_lr)
    parser.add_argument("--device", type=str, default=cfg.device)
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    print(f"Device: {device}")

    # 数据集
    train_ds = ESDDataset(
        Path(args.esd_root), cfg.en_speakers, cfg.emotions,
        split="train", sample_rate=cfg.sample_rate, max_len=cfg.max_audio_len,
    )
    test_ds = ESDDataset(
        Path(args.esd_root), cfg.en_speakers, cfg.emotions,
        split="test", sample_rate=cfg.sample_rate, max_len=cfg.max_audio_len,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )

    # 加载冻结的 surrogate SER
    ser_model = SurrogateSER().to(device)
    ser_model.load_state_dict(torch.load(args.surrogate_ckpt, map_location=device))
    ser_model.eval()
    for param in ser_model.parameters():
        param.requires_grad = False
    print(f"Loaded surrogate SER from {args.surrogate_ckpt}")

    # Generator
    generator = WaveUNetGenerator().to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.gen_scheduler_step, gamma=0.5)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    eps = cfg.epsilon
    best_asr = 0.0

    for epoch in range(args.epochs):
        # 论文中 eps 调度（可选）
        # if epoch >= 9: eps = 0.03
        # if epoch >= 19: eps = 0.01

        train_stats = train_one_epoch(generator, ser_model, train_loader, optimizer, eps, device)
        test_stats = evaluate_generator(generator, ser_model, test_loader, eps, device)
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{args.epochs}  "
            f"loss={train_stats['loss']:.4f} adv_loss={train_stats['adv_loss']:.4f} "
            f"train_ASR={train_stats['attack_success_rate']:.4f} "
            f"test_ASR={test_stats['attack_success_rate']:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )

        if test_stats["attack_success_rate"] > best_asr:
            best_asr = test_stats["attack_success_rate"]
            torch.save(generator.state_dict(), output_path)
            print(f"  => 保存最佳 generator (test_ASR={best_asr:.4f})")

    # 最终也保存一份
    torch.save(generator.state_dict(), output_path.parent / "generator_final.pt")

    print(f"\n训练完成。最佳 test ASR (on surrogate SER) = {best_asr:.4f}")

    info = {
        "best_test_asr": best_asr,
        "epochs": args.epochs,
        "lr": args.lr,
        "epsilon": cfg.epsilon,
        "lambda_mag": cfg.lambda_mag,
        "lambda_spa": cfg.lambda_spa,
        "lambda_qua": cfg.lambda_qua,
    }
    info_path = output_path.parent / "generator_info.json"
    info_path.write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
