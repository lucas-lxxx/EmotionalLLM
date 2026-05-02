"""Voxtral 评估：将 clean 和三种畸变音频输入 Voxtral，测量情绪翻转率。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import torchaudio

from config import cfg


# ── Voxtral 加载与推理 ──

def load_voxtral(model_path: Path, device: str):
    from transformers import VoxtralForConditionalGeneration, AutoProcessor

    if device.startswith("cuda"):
        if ":" in device:
            torch.cuda.set_device(int(device.split(":")[1]))
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
    return model, processor


def build_input_ids(tokenizer, prompt: str) -> torch.LongTensor:
    text_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    ids = (
        [cfg.bos_id, cfg.inst_id, cfg.begin_audio_id]
        + [cfg.audio_token_id] * cfg.n_audio_tokens
        + text_tokens
        + [cfg.inst_end_id]
    )
    return torch.LongTensor(ids).unsqueeze(0)


def decode_emotion(model, processor, waveform: torch.Tensor, sr: int, device: str) -> str:
    wav_np = waveform.detach().cpu().numpy().squeeze()
    feat = processor.feature_extractor(wav_np, sampling_rate=sr, return_tensors="pt")
    input_features = feat.input_features.to(device)
    model_dtype = next(model.parameters()).dtype
    input_features = input_features.to(dtype=model_dtype)

    input_ids = build_input_ids(processor.tokenizer, cfg.emo_prompt).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)

    from transformers import GenerationConfig
    gen_config = GenerationConfig(
        max_new_tokens=cfg.emo_max_new_tokens,
        do_sample=False,
        temperature=cfg.temperature,
        use_cache=True,
        pad_token_id=processor.tokenizer.eos_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )

    generated = model.generate(
        input_ids=input_ids,
        input_features=input_features,
        attention_mask=attention_mask,
        generation_config=gen_config,
    )

    input_len = input_ids.shape[1]
    gen_tokens = generated[:, input_len:]
    text = processor.tokenizer.decode(gen_tokens[0], skip_special_tokens=True).strip()
    return text


def normalize_emo(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z]+", " ", text).strip()
    for label in cfg.emo_labels:
        if re.search(rf"\b{re.escape(label)}\b", text):
            return label
    return text.split()[0] if text else "unknown"


def load_audio(path: Path, target_sr: int) -> tuple[torch.Tensor, int]:
    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr
    return waveform.float(), sr


# ── 主流程 ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distorted_dir", type=str, default=str(cfg.distorted_audio_dir))
    parser.add_argument("--results_dir", type=str, default=str(cfg.results_dir))
    parser.add_argument("--device", type=str, default=cfg.device)
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    distorted_dir = Path(args.distorted_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 加载 Voxtral
    print(f"加载 Voxtral: {cfg.voxtral_model_path}")
    model, processor = load_voxtral(cfg.voxtral_model_path, device)
    print("Voxtral 加载完成")

    distortion_names = ["vtln", "mcadams", "mss"]
    all_summaries = {}

    # 先评估 clean（只做一次）
    clean_dir = distorted_dir / "clean"
    clean_preds = {}  # index -> (voxtral_raw, voxtral_norm)

    # 获取 clean 文件列表
    clean_files = sorted(clean_dir.glob("*.wav")) if clean_dir.exists() else []
    if args.max_samples > 0:
        clean_files = clean_files[:args.max_samples]

    print(f"\n评估 clean 音频 ({len(clean_files)} 个样本)...")
    for wav_path in clean_files:
        # 从文件名提取 index 和 gt_emo
        stem = wav_path.stem  # e.g. "0000_angry"
        parts = stem.split("_", 1)
        idx = int(parts[0])

        waveform, sr = load_audio(wav_path, cfg.sample_rate)
        waveform = waveform.to(device)
        raw = decode_emotion(model, processor, waveform, sr, device)
        norm = normalize_emo(raw)
        clean_preds[idx] = (raw, norm)

        if (idx + 1) % 10 == 0:
            print(f"  clean [{idx+1}/{len(clean_files)}]")

    # 评估每种畸变
    for dist_name in distortion_names:
        print(f"\n{'='*50}")
        print(f"评估畸变: {dist_name}")
        print(f"{'='*50}")

        dist_dir = distorted_dir / dist_name
        gen_results_path = dist_dir / "generation_results.json"
        if not gen_results_path.exists():
            print(f"  跳过 {dist_name}：找不到 generation_results.json")
            continue

        gen_data = json.loads(gen_results_path.read_text())
        samples = gen_data["samples"]
        if args.max_samples > 0:
            samples = samples[:args.max_samples]

        eval_results = []

        for i, sample in enumerate(samples):
            gt_emo = sample["ground_truth"]
            idx = sample["index"]
            dist_wav_path = Path(sample["distorted_wav_path"])

            # Clean prediction
            if idx in clean_preds:
                voxtral_clean_raw, voxtral_clean = clean_preds[idx]
            else:
                voxtral_clean_raw = ""
                voxtral_clean = "unknown"

            # Distorted prediction
            waveform_dist, sr = load_audio(dist_wav_path, cfg.sample_rate)
            waveform_dist = waveform_dist.to(device)
            voxtral_dist_raw = decode_emotion(model, processor, waveform_dist, sr, device)
            voxtral_dist = normalize_emo(voxtral_dist_raw)

            clean_correct = (voxtral_clean == gt_emo)
            dist_flipped = (voxtral_dist != gt_emo)
            prediction_changed = (voxtral_dist != voxtral_clean)

            entry = {
                "index": idx,
                "ground_truth": gt_emo,
                "voxtral_clean_raw": voxtral_clean_raw,
                "voxtral_clean": voxtral_clean,
                "voxtral_clean_correct": clean_correct,
                "voxtral_dist_raw": voxtral_dist_raw,
                "voxtral_dist": voxtral_dist,
                "voxtral_dist_flipped": dist_flipped,
                "prediction_changed": prediction_changed,
                "delta_linf": sample.get("delta_linf", 0),
                "delta_l2": sample.get("delta_l2", 0),
                "snr_db": sample.get("snr_db", 0),
            }
            eval_results.append(entry)

            if (i + 1) % 10 == 0:
                n_changed = sum(1 for r in eval_results if r["prediction_changed"])
                n = len(eval_results)
                print(f"  [{i+1}/{len(samples)}] prediction_changed={n_changed/n:.4f}")

        # 汇总
        total = len(eval_results)
        n_clean_correct = sum(1 for r in eval_results if r["voxtral_clean_correct"])
        n_dist_flipped = sum(1 for r in eval_results if r["voxtral_dist_flipped"])
        n_changed = sum(1 for r in eval_results if r["prediction_changed"])

        by_emotion = {}
        for r in eval_results:
            emo = r["ground_truth"]
            if emo not in by_emotion:
                by_emotion[emo] = {"total": 0, "clean_correct": 0, "dist_flipped": 0, "changed": 0}
            by_emotion[emo]["total"] += 1
            if r["voxtral_clean_correct"]:
                by_emotion[emo]["clean_correct"] += 1
            if r["voxtral_dist_flipped"]:
                by_emotion[emo]["dist_flipped"] += 1
            if r["prediction_changed"]:
                by_emotion[emo]["changed"] += 1

        for emo, stats in by_emotion.items():
            n = stats["total"]
            stats["clean_acc"] = stats["clean_correct"] / n
            stats["dist_flip_rate"] = stats["dist_flipped"] / n
            stats["change_rate"] = stats["changed"] / n

        summary = {
            "method": f"GAO_{dist_name}",
            "distortion": dist_name,
            "total_samples": total,
            "voxtral_clean_accuracy": n_clean_correct / total,
            "voxtral_dist_flip_rate": n_dist_flipped / total,
            "prediction_change_rate": n_changed / total,
            "by_emotion": by_emotion,
        }

        print(f"\n{dist_name} 评估完成：{total} 个样本")
        print(f"  Clean accuracy:        {n_clean_correct}/{total} = {n_clean_correct/total:.4f}")
        print(f"  Dist flip rate:        {n_dist_flipped}/{total} = {n_dist_flipped/total:.4f}")
        print(f"  Prediction changed:    {n_changed}/{total} = {n_changed/total:.4f}")

        # 保存
        dist_results_dir = results_dir / dist_name
        dist_results_dir.mkdir(parents=True, exist_ok=True)
        (dist_results_dir / "eval_results.json").write_text(
            json.dumps({"summary": summary, "samples": eval_results}, indent=2, ensure_ascii=False)
        )
        (dist_results_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False)
        )

        all_summaries[dist_name] = summary

    # 总汇总
    (results_dir / "all_summaries.json").write_text(
        json.dumps(all_summaries, indent=2, ensure_ascii=False)
    )
    print(f"\n全部评估完成，结果保存到 {results_dir}")


if __name__ == "__main__":
    main()
