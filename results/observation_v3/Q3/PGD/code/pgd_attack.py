"""PGD 攻击：对 surrogate SER 做 PGD 迭代梯度攻击，生成对抗音频

参考 Facchinetti et al. "A systematic evaluation of adversarial attacks
against speech emotion recognition models" 中的 BIM (Basic Iterative Method)。

在原始波形上做 L∞ 约束的 PGD 攻击（untargeted），保存对抗音频 + 元数据。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import DataLoader

from config import cfg
from esd_en_dataset import ESDDataset, collate_fn
from surrogate_ser import SurrogateSER


idx2emotion = {v: k for k, v in cfg.emotion2idx.items()}


def pgd_attack(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    label: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
    random_start: bool = True,
) -> torch.Tensor:
    """PGD untargeted attack on waveform against SER model.

    Args:
        model: surrogate SER model (frozen, eval mode)
        waveform: (batch, time) clean audio
        label: (batch,) ground-truth label
        epsilon: L-inf perturbation bound
        alpha: step size per iteration
        steps: number of PGD iterations
        random_start: whether to start from random point in eps-ball

    Returns:
        x_adv: (batch, time) adversarial audio
    """
    x_adv = waveform.clone().detach()

    if random_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_adv, -1.0, 1.0)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        # untargeted: maximize CE loss w.r.t. true label
        loss = torch.nn.functional.cross_entropy(logits, label)
        loss.backward()

        with torch.no_grad():
            grad_sign = x_adv.grad.sign()
            x_adv = x_adv.detach() + alpha * grad_sign
            # project back to eps-ball
            delta = torch.clamp(x_adv - waveform, min=-epsilon, max=epsilon)
            x_adv = torch.clamp(waveform + delta, -1.0, 1.0)

    return x_adv.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esd_root", type=str, default=str(cfg.esd_root))
    parser.add_argument("--surrogate_ckpt", type=str, default=str(cfg.surrogate_ckpt))
    parser.add_argument("--output_dir", type=str, default=str(cfg.adv_audio_dir))
    parser.add_argument("--device", type=str, default=cfg.device)
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    # 数据集（测试集）
    test_ds = ESDDataset(
        Path(args.esd_root), cfg.en_speakers, cfg.emotions,
        split="test", sample_rate=cfg.sample_rate, max_len=cfg.max_audio_len,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=2,
    )

    # 加载 surrogate SER
    ser_model = SurrogateSER().to(device)
    ser_model.load_state_dict(torch.load(args.surrogate_ckpt, map_location=device))
    ser_model.eval()
    ser_model.requires_grad_(False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for i, (waveforms, labels, emotions, paths) in enumerate(test_loader):
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        # Clean SER prediction
        with torch.no_grad():
            logits_clean = ser_model(waveforms)
            pred_clean = logits_clean.argmax(dim=-1).item()

        # PGD attack
        x_adv = pgd_attack(
            ser_model, waveforms, labels,
            epsilon=cfg.epsilon,
            alpha=cfg.pgd_alpha,
            steps=cfg.pgd_steps,
            random_start=cfg.pgd_random_start,
        )

        # Adversarial SER prediction
        with torch.no_grad():
            logits_adv = ser_model(x_adv)
            pred_adv = logits_adv.argmax(dim=-1).item()

        # 保存对抗音频
        gt_emo = emotions[0]
        wav_name = f"{i:04d}_{gt_emo}.wav"
        wav_path = output_dir / wav_name
        adv_wav = x_adv.squeeze(0).cpu()
        torchaudio.save(str(wav_path), adv_wav.unsqueeze(0), cfg.sample_rate)

        # 保存 clean 音频
        clean_dir = output_dir / "clean"
        clean_dir.mkdir(parents=True, exist_ok=True)
        clean_wav = waveforms.squeeze(0).cpu()
        torchaudio.save(str(clean_dir / wav_name), clean_wav.unsqueeze(0), cfg.sample_rate)

        # 扰动统计
        delta = (x_adv - waveforms).squeeze(0).cpu()
        linf = delta.abs().max().item()
        l2 = delta.norm(p=2).item()

        entry = {
            "index": i,
            "source_path": paths[0],
            "ground_truth": gt_emo,
            "ser_pred_clean": idx2emotion.get(pred_clean, str(pred_clean)),
            "ser_pred_adv": idx2emotion.get(pred_adv, str(pred_adv)),
            "ser_attack_success": pred_adv != labels.item(),
            "adv_wav_path": str(wav_path),
            "delta_linf": linf,
            "delta_l2": l2,
        }
        results.append(entry)

        if (i + 1) % 50 == 0:
            success = sum(1 for r in results if r["ser_attack_success"])
            print(f"  [{i+1}/{len(test_ds)}] SER ASR so far: {success}/{len(results)} = {success/len(results):.4f}")

    # 汇总
    total = len(results)
    ser_success = sum(1 for r in results if r["ser_attack_success"])
    avg_linf = sum(r["delta_linf"] for r in results) / total
    avg_l2 = sum(r["delta_l2"] for r in results) / total

    summary = {
        "total_samples": total,
        "ser_attack_success": ser_success,
        "ser_attack_success_rate": ser_success / total,
        "avg_delta_linf": avg_linf,
        "avg_delta_l2": avg_l2,
        "epsilon": cfg.epsilon,
        "pgd_steps": cfg.pgd_steps,
        "pgd_alpha": cfg.pgd_alpha,
    }

    print(f"\nPGD 攻击完成：{total} 个对抗样本")
    print(f"SER attack success rate: {ser_success}/{total} = {ser_success/total:.4f}")
    print(f"Avg ΔL∞={avg_linf:.6f}, ΔL2={avg_l2:.4f}")

    # 保存结果
    results_path = output_dir / "generation_results.json"
    results_path.write_text(json.dumps({"summary": summary, "samples": results}, indent=2, ensure_ascii=False))
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
