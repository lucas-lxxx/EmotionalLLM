"""对 ESD 测试集应用三种语音畸变（VTLN, McAdams, MSS），保存畸变音频 + 元数据。

无需 surrogate SER——纯黑盒信号处理攻击。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio

from config import cfg
from distortions import vtln, mcadams, mss


# ── ESD 扫描（简化版，不依赖 esd_en_dataset.py） ──

EMO_NORMALIZE = {
    "angry": "angry", "Angry": "angry",
    "happy": "happy", "Happy": "happy",
    "neutral": "neutral", "Neutral": "neutral",
    "sad": "sad", "Sad": "sad",
    "surprise": "surprise", "Surprise": "surprise",
}


def scan_test_samples(esd_root: Path, speakers: list[str],
                      emotions: list[str]) -> list[dict]:
    """扫描 ESD 测试集（自动检测 standard/flat 结构）"""
    entries = []
    # 检测结构
    is_standard = False
    for sp in speakers:
        sp_dir = esd_root / sp
        if not sp_dir.exists():
            continue
        for emo_dir in sp_dir.iterdir():
            if emo_dir.is_dir() and (emo_dir / "test").is_dir():
                is_standard = True
                break
        if is_standard:
            break

    for sp in speakers:
        sp_dir = esd_root / sp
        if not sp_dir.exists():
            continue
        for emo_dir in sp_dir.iterdir():
            if not emo_dir.is_dir():
                continue
            emo = EMO_NORMALIZE.get(emo_dir.name)
            if emo is None or emo not in emotions:
                continue

            if is_standard:
                test_dir = emo_dir / "test"
                if test_dir.exists():
                    for wav in sorted(test_dir.glob("*.wav")):
                        entries.append({"path": str(wav), "speaker": sp, "emotion": emo})
            else:
                for wav in sorted(emo_dir.glob("*.wav")):
                    entries.append({"path": str(wav), "speaker": sp, "emotion": emo})

    if not is_standard:
        # flat 结构：取后 10% 作为测试集
        random.seed(42)
        random.shuffle(entries)
        n_test = max(1, int(len(entries) * 0.1))
        entries = entries[-n_test:]

    print(f"扫描到 {len(entries)} 个测试样本")
    return entries


def load_audio(path: str, target_sr: int) -> tuple[np.ndarray, int]:
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)  # (time,)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr
    # truncate
    max_len = cfg.max_audio_len
    if waveform.shape[0] > max_len:
        waveform = waveform[:max_len]
    return waveform.numpy().astype(np.float32), sr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esd_root", type=str, default=str(cfg.esd_root))
    parser.add_argument("--output_dir", type=str, default=str(cfg.distorted_audio_dir))
    args = parser.parse_args()

    samples = scan_test_samples(Path(args.esd_root), cfg.en_speakers, cfg.emotions)
    output_dir = Path(args.output_dir)

    distortion_configs = {
        "vtln": {"func": vtln, "kwargs": {"alpha": cfg.vtln_alpha}},
        "mcadams": {"func": mcadams, "kwargs": {"alpha": cfg.mcadams_alpha}},
        "mss": {"func": mss, "kwargs": {"alpha": cfg.mss_alpha}},
    }

    all_results = {}

    for dist_name, dist_cfg in distortion_configs.items():
        print(f"\n{'='*50}")
        print(f"应用畸变: {dist_name} (alpha={list(dist_cfg['kwargs'].values())[0]})")
        print(f"{'='*50}")

        dist_dir = output_dir / dist_name
        dist_dir.mkdir(parents=True, exist_ok=True)
        clean_dir = output_dir / "clean"
        clean_dir.mkdir(parents=True, exist_ok=True)

        results = []

        for i, sample in enumerate(samples):
            wav_np, sr = load_audio(sample["path"], cfg.sample_rate)
            gt_emo = sample["emotion"]

            # 应用畸变
            try:
                distorted = dist_cfg["func"](wav_np, sr, **dist_cfg["kwargs"])
            except Exception as e:
                print(f"  Warning: {dist_name} failed on sample {i}: {e}")
                distorted = wav_np.copy()

            # 保存
            wav_name = f"{i:04d}_{gt_emo}.wav"

            # 保存畸变音频
            dist_wav = torch.from_numpy(distorted).unsqueeze(0)
            torchaudio.save(str(dist_dir / wav_name), dist_wav, sr)

            # 保存 clean 音频（只在第一种畸变时保存）
            if dist_name == "vtln":
                clean_wav = torch.from_numpy(wav_np).unsqueeze(0)
                torchaudio.save(str(clean_dir / wav_name), clean_wav, sr)

            # 扰动统计
            delta = distorted - wav_np
            linf = float(np.max(np.abs(delta)))
            l2 = float(np.sqrt(np.sum(delta ** 2)))
            snr = float(10 * np.log10(
                np.sum(wav_np ** 2) / (np.sum(delta ** 2) + 1e-10)
            ))

            entry = {
                "index": i,
                "source_path": sample["path"],
                "ground_truth": gt_emo,
                "distorted_wav_path": str(dist_dir / wav_name),
                "delta_linf": linf,
                "delta_l2": l2,
                "snr_db": snr,
            }
            results.append(entry)

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(samples)}] avg ΔL∞={np.mean([r['delta_linf'] for r in results]):.4f}")

        # 汇总
        total = len(results)
        avg_linf = np.mean([r["delta_linf"] for r in results])
        avg_l2 = np.mean([r["delta_l2"] for r in results])
        avg_snr = np.mean([r["snr_db"] for r in results])

        summary = {
            "distortion": dist_name,
            "alpha": list(dist_cfg["kwargs"].values())[0],
            "total_samples": total,
            "avg_delta_linf": float(avg_linf),
            "avg_delta_l2": float(avg_l2),
            "avg_snr_db": float(avg_snr),
        }

        print(f"\n{dist_name} 完成：{total} 个样本")
        print(f"  Avg ΔL∞={avg_linf:.4f}, ΔL2={avg_l2:.4f}, SNR={avg_snr:.1f} dB")

        # 保存结果
        results_path = dist_dir / "generation_results.json"
        results_path.write_text(
            json.dumps({"summary": summary, "samples": results}, indent=2, ensure_ascii=False)
        )

        all_results[dist_name] = summary

    # 保存总汇总
    overall_path = output_dir / "all_distortions_summary.json"
    overall_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"\n全部畸变完成，总结保存到 {overall_path}")


if __name__ == "__main__":
    main()
