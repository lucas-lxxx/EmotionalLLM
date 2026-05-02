"""Step 2c: 生成对抗样本

加载训练好的 generator，对 ESD EN 测试集生成对抗音频。
保存 wav 文件 + 元数据（ground truth、SER 预测）。
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
from wave_unet import WaveUNetGenerator


idx2emotion = {v: k for k, v in cfg.emotion2idx.items()}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esd_root", type=str, default=str(cfg.esd_root))
    parser.add_argument("--surrogate_ckpt", type=str, default=str(cfg.surrogate_ckpt))
    parser.add_argument("--generator_ckpt", type=str, default=str(cfg.generator_ckpt))
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

    # 加载模型
    ser_model = SurrogateSER().to(device)
    ser_model.load_state_dict(torch.load(args.surrogate_ckpt, map_location=device))
    ser_model.eval()

    generator = WaveUNetGenerator().to(device)
    generator.load_state_dict(torch.load(args.generator_ckpt, map_location=device))
    generator.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for i, (waveforms, labels, emotions, paths) in enumerate(test_loader):
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        # Clean SER prediction
        logits_clean = ser_model(waveforms)
        pred_clean = logits_clean.argmax(dim=-1).item()

        # Generate adversarial
        x_adv, v, m = generator(waveforms, cfg.epsilon, training=False)
        x_adv_flat = x_adv.squeeze(1)

        # Adversarial SER prediction
        logits_adv = ser_model(x_adv_flat)
        pred_adv = logits_adv.argmax(dim=-1).item()

        # 保存对抗音频
        gt_emo = emotions[0]
        wav_name = f"{i:04d}_{gt_emo}.wav"
        wav_path = output_dir / wav_name
        adv_wav = x_adv_flat.squeeze(0).cpu()
        torchaudio.save(str(wav_path), adv_wav.unsqueeze(0), cfg.sample_rate)

        # 同时保存 clean 音频（用于 Voxtral baseline）
        clean_dir = output_dir / "clean"
        clean_dir.mkdir(parents=True, exist_ok=True)
        clean_wav = waveforms.squeeze(0).cpu()
        torchaudio.save(str(clean_dir / wav_name), clean_wav.unsqueeze(0), cfg.sample_rate)

        # 计算扰动信息
        delta = (x_adv_flat - waveforms).squeeze(0).cpu()
        linf = delta.abs().max().item()
        l2 = delta.norm(p=2).item()
        sparsity = (m.squeeze() > 0.5).float().mean().item()

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
            "mask_sparsity": sparsity,
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
    avg_sparsity = sum(r["mask_sparsity"] for r in results) / total

    summary = {
        "total_samples": total,
        "ser_attack_success": ser_success,
        "ser_attack_success_rate": ser_success / total,
        "avg_delta_linf": avg_linf,
        "avg_delta_l2": avg_l2,
        "avg_mask_sparsity": avg_sparsity,
        "epsilon": cfg.epsilon,
    }

    print(f"\n生成完成：{total} 个对抗样本")
    print(f"SER attack success rate: {ser_success}/{total} = {ser_success/total:.4f}")
    print(f"Avg ΔL∞={avg_linf:.6f}, ΔL2={avg_l2:.4f}, mask sparsity={avg_sparsity:.4f}")

    # 保存结果
    results_path = output_dir / "generation_results.json"
    results_path.write_text(json.dumps({"summary": summary, "samples": results}, indent=2, ensure_ascii=False))
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
