"""Step 2a: 训练 Surrogate SER 模型

在 ESD English 上 fine-tune wav2vec2-base + linear head。
保持简单，不追求高精度。
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


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for waveforms, labels, _, _ in loader:
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        logits = model(waveforms)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for waveforms, labels, _, _ in loader:
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        logits = model(waveforms)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esd_root", type=str, default=str(cfg.esd_root))
    parser.add_argument("--output", type=str, default=str(cfg.surrogate_ckpt))
    parser.add_argument("--epochs", type=int, default=cfg.ser_epochs)
    parser.add_argument("--batch_size", type=int, default=cfg.ser_batch_size)
    parser.add_argument("--lr", type=float, default=cfg.ser_lr)
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

    # 模型
    model = SurrogateSER().to(device)
    for param in model.wav2vec.parameters():
        param.requires_grad = False
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(
            f"Epoch {epoch+1}/{args.epochs}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), output_path)
            print(f"  => 保存最佳模型 (test_acc={best_acc:.4f})")

    print(f"\n训练完成。最佳 test_acc={best_acc:.4f}")
    print(f"Checkpoint: {output_path}")

    # 保存训练信息
    info = {
        "best_test_acc": best_acc,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "model": cfg.wav2vec_model,
        "num_classes": cfg.ser_num_classes,
        "train_samples": len(train_ds),
        "test_samples": len(test_ds),
    }
    info_path = output_path.parent / "surrogate_info.json"
    info_path.write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
