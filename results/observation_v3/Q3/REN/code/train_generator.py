"""训练 Atrous CNN Generator 对 surrogate SER 做 untargeted 攻击

Loss = α * C&W_loss + (1-α) * MSE_loss

C&W untargeted loss: max(Z_true - max_{j≠true} Z_j, 0)
MSE loss: ||x_adv - x||^2 (保持与原始音频相似)

参考 Ren et al. Eq. (1)(2)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from config import cfg
from esd_en_dataset import ESDDataset, collate_fn
from surrogate_ser import SurrogateSER
from atrous_generator import AtrousCNNGenerator


idx2emotion = {v: k for k, v in cfg.emotion2idx.items()}


def cw_untargeted_loss(logits: torch.Tensor, labels: torch.Tensor,
                       confidence: float = 0.0) -> torch.Tensor:
    """C&W untargeted loss: max(Z_true - max_{j≠true} Z_j, -κ)

    目标：让 true class 的 logit 低于其他最大 class 的 logit
    """
    batch_size = logits.shape[0]
    num_classes = logits.shape[1]

    # 获取 true class 的 logit
    z_true = logits[torch.arange(batch_size), labels]

    # 获取其他 class 的最大 logit
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[torch.arange(batch_size), labels] = False
    z_other_max = logits.masked_fill(~mask, -float('inf')).max(dim=-1).values

    # C&W loss: max(z_true - z_other_max, -κ)
    loss = torch.clamp(z_true - z_other_max, min=-confidence)
    return loss.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esd_root", type=str, default=str(cfg.esd_root))
    parser.add_argument("--surrogate_ckpt", type=str, default=str(cfg.surrogate_ckpt))
    parser.add_argument("--device", type=str, default=cfg.device)
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    # 数据集
    train_ds = ESDDataset(
        Path(args.esd_root), cfg.en_speakers, cfg.emotions,
        split="train", sample_rate=cfg.sample_rate, max_len=cfg.max_audio_len,
    )
    # 限制训练样本数
    if cfg.gen_max_train_samples > 0 and len(train_ds) > cfg.gen_max_train_samples:
        indices = list(range(cfg.gen_max_train_samples))
        train_ds = Subset(train_ds, indices)
        print(f"限制训练集为 {cfg.gen_max_train_samples} 样本")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.gen_batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2,
    )

    # 加载 surrogate SER（冻结）
    ser_model = SurrogateSER().to(device)
    ser_model.load_state_dict(torch.load(args.surrogate_ckpt, map_location=device))
    ser_model.eval()
    ser_model.requires_grad_(False)

    # 创建 Generator
    generator = AtrousCNNGenerator().to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=cfg.gen_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)

    # Checkpoint 目录
    ckpt_dir = Path(cfg.generator_ckpt).parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_log = []

    print(f"\n开始训练 Atrous CNN Generator")
    print(f"  α (C&W weight): {cfg.alpha_loss}")
    print(f"  ε (perturbation): {cfg.epsilon}")
    print(f"  epochs: {cfg.gen_epochs}")
    print(f"  lr: {cfg.gen_lr}")
    print(f"  device: {device}")

    for epoch in range(1, cfg.gen_epochs + 1):
        generator.train()
        epoch_cw = 0.0
        epoch_mse = 0.0
        epoch_total = 0.0
        n_correct_clean = 0
        n_attack_success = 0
        n_samples = 0

        for batch_idx, (waveforms, labels, emotions, paths) in enumerate(train_loader):
            waveforms = waveforms.to(device)
            labels = labels.to(device)

            # Generator forward
            x_adv, eta = generator(waveforms, cfg.epsilon)

            # Surrogate SER forward on adversarial
            logits_adv = ser_model(x_adv)

            # C&W untargeted loss
            loss_cw = cw_untargeted_loss(logits_adv, labels, cfg.cw_confidence)

            # MSE loss（保持与原始音频相似）
            loss_mse = F.mse_loss(x_adv, waveforms)

            # Total loss: α * L_cw + (1-α) * L_mse
            loss = cfg.alpha_loss * loss_cw + (1 - cfg.alpha_loss) * loss_mse

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            with torch.no_grad():
                pred_adv = logits_adv.argmax(dim=-1)
                pred_clean = ser_model(waveforms).argmax(dim=-1)
                n_correct_clean += (pred_clean == labels).sum().item()
                n_attack_success += (pred_adv != labels).sum().item()

            batch_size = waveforms.shape[0]
            n_samples += batch_size
            epoch_cw += loss_cw.item() * batch_size
            epoch_mse += loss_mse.item() * batch_size
            epoch_total += loss.item() * batch_size

        scheduler.step()

        avg_cw = epoch_cw / n_samples
        avg_mse = epoch_mse / n_samples
        avg_total = epoch_total / n_samples
        clean_acc = n_correct_clean / n_samples
        attack_asr = n_attack_success / n_samples

        log_entry = {
            "epoch": epoch,
            "total_loss": avg_total,
            "cw_loss": avg_cw,
            "mse_loss": avg_mse,
            "clean_acc": clean_acc,
            "attack_asr": attack_asr,
            "lr": optimizer.param_groups[0]["lr"],
        }
        train_log.append(log_entry)

        print(
            f"Epoch {epoch:2d}/{cfg.gen_epochs} | "
            f"loss={avg_total:.4f} (cw={avg_cw:.4f} mse={avg_mse:.6f}) | "
            f"ASR={attack_asr:.4f} | clean_acc={clean_acc:.4f}"
        )

    # 保存 checkpoint
    torch.save(generator.state_dict(), str(cfg.generator_ckpt))
    print(f"\nGenerator saved to {cfg.generator_ckpt}")

    # 保存训练日志
    log_path = ckpt_dir / "train_log.json"
    log_path.write_text(json.dumps(train_log, indent=2, ensure_ascii=False))
    print(f"Train log saved to {log_path}")


if __name__ == "__main__":
    main()
