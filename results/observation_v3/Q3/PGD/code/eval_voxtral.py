"""Voxtral 评估：将 clean 和 PGD 对抗音频输入 Voxtral，测量情绪翻转率。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import torchaudio

from config import cfg


# ── Voxtral 加载与推理（从 white_box_voxtral 适配） ──

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
    """Voxtral input_ids: [BOS][INST][BEGIN_AUDIO][AUDIO]*375 <text> [INST_END]"""
    text_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    ids = (
        [cfg.bos_id, cfg.inst_id, cfg.begin_audio_id]
        + [cfg.audio_token_id] * cfg.n_audio_tokens
        + text_tokens
        + [cfg.inst_end_id]
    )
    return torch.LongTensor(ids).unsqueeze(0)


def decode_emotion(
    model, processor, waveform: torch.Tensor, sr: int, device: str,
) -> str:
    """对单条音频做情绪识别推理"""
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
    """从 Voxtral 输出中提取情绪标签"""
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
    parser.add_argument("--adv_dir", type=str, default=str(cfg.adv_audio_dir))
    parser.add_argument("--results_dir", type=str, default=str(cfg.results_dir))
    parser.add_argument("--generation_results", type=str, default=None)
    parser.add_argument("--device", type=str, default=cfg.device)
    parser.add_argument("--max_samples", type=int, default=0, help="限制样本数，0=全部")
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    adv_dir = Path(args.adv_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 加载 generation results
    gen_results_path = args.generation_results or str(adv_dir / "generation_results.json")
    gen_data = json.loads(Path(gen_results_path).read_text())
    samples = gen_data["samples"]

    if args.max_samples > 0:
        samples = samples[:args.max_samples]

    print(f"加载 Voxtral: {cfg.voxtral_model_path}")
    model, processor = load_voxtral(cfg.voxtral_model_path, device)
    print("Voxtral 加载完成")

    eval_results = []

    for i, sample in enumerate(samples):
        gt_emo = sample["ground_truth"]
        adv_wav_path = Path(sample["adv_wav_path"])
        clean_wav_path = adv_dir / "clean" / adv_wav_path.name

        # ── Clean 音频评估 ──
        if clean_wav_path.exists():
            waveform_clean, sr = load_audio(clean_wav_path, cfg.sample_rate)
            waveform_clean = waveform_clean.to(device)
            voxtral_clean_raw = decode_emotion(model, processor, waveform_clean, sr, device)
            voxtral_clean = normalize_emo(voxtral_clean_raw)
        else:
            voxtral_clean_raw = ""
            voxtral_clean = "unknown"

        # ── 对抗音频评估 ──
        waveform_adv, sr = load_audio(adv_wav_path, cfg.sample_rate)
        waveform_adv = waveform_adv.to(device)
        voxtral_adv_raw = decode_emotion(model, processor, waveform_adv, sr, device)
        voxtral_adv = normalize_emo(voxtral_adv_raw)

        # 翻转判断
        clean_correct = (voxtral_clean == gt_emo)
        adv_flipped = (voxtral_adv != gt_emo)

        entry = {
            "index": sample["index"],
            "ground_truth": gt_emo,
            "ser_pred_clean": sample["ser_pred_clean"],
            "ser_pred_adv": sample["ser_pred_adv"],
            "ser_attack_success": sample["ser_attack_success"],
            "voxtral_clean_raw": voxtral_clean_raw,
            "voxtral_clean": voxtral_clean,
            "voxtral_clean_correct": clean_correct,
            "voxtral_adv_raw": voxtral_adv_raw,
            "voxtral_adv": voxtral_adv,
            "voxtral_adv_flipped": adv_flipped,
            "delta_linf": sample["delta_linf"],
            "delta_l2": sample["delta_l2"],
        }
        eval_results.append(entry)

        if (i + 1) % 10 == 0:
            n_clean_correct = sum(1 for r in eval_results if r["voxtral_clean_correct"])
            n_adv_flipped = sum(1 for r in eval_results if r["voxtral_adv_flipped"])
            n = len(eval_results)
            print(
                f"  [{i+1}/{len(samples)}] "
                f"clean_acc={n_clean_correct/n:.4f} "
                f"adv_flip_rate={n_adv_flipped/n:.4f}"
            )

    # ── 汇总 ──
    total = len(eval_results)
    n_clean_correct = sum(1 for r in eval_results if r["voxtral_clean_correct"])
    n_adv_flipped = sum(1 for r in eval_results if r["voxtral_adv_flipped"])
    n_ser_success = sum(1 for r in eval_results if r["ser_attack_success"])

    # 按情绪统计
    by_emotion = {}
    for r in eval_results:
        emo = r["ground_truth"]
        if emo not in by_emotion:
            by_emotion[emo] = {"total": 0, "clean_correct": 0, "adv_flipped": 0, "ser_success": 0}
        by_emotion[emo]["total"] += 1
        if r["voxtral_clean_correct"]:
            by_emotion[emo]["clean_correct"] += 1
        if r["voxtral_adv_flipped"]:
            by_emotion[emo]["adv_flipped"] += 1
        if r["ser_attack_success"]:
            by_emotion[emo]["ser_success"] += 1

    for emo, stats in by_emotion.items():
        n = stats["total"]
        stats["clean_acc"] = stats["clean_correct"] / n
        stats["adv_flip_rate"] = stats["adv_flipped"] / n
        stats["ser_asr"] = stats["ser_success"] / n

    summary = {
        "method": "PGD",
        "total_samples": total,
        "voxtral_clean_accuracy": n_clean_correct / total,
        "voxtral_adv_flip_rate": n_adv_flipped / total,
        "ser_attack_success_rate": n_ser_success / total,
        "by_emotion": by_emotion,
    }

    print(f"\n{'='*60}")
    print(f"PGD 评估完成：{total} 个样本")
    print(f"  Voxtral clean accuracy:    {n_clean_correct}/{total} = {n_clean_correct/total:.4f}")
    print(f"  Voxtral adv flip rate:     {n_adv_flipped}/{total} = {n_adv_flipped/total:.4f}")
    print(f"  SER attack success rate:   {n_ser_success}/{total} = {n_ser_success/total:.4f}")
    print(f"\n按情绪统计:")
    for emo, stats in sorted(by_emotion.items()):
        print(
            f"  {emo:10s}: clean_acc={stats['clean_acc']:.4f} "
            f"adv_flip={stats['adv_flip_rate']:.4f} "
            f"ser_asr={stats['ser_asr']:.4f} "
            f"(n={stats['total']})"
        )

    # 保存
    full_results = {"summary": summary, "samples": eval_results}
    (results_dir / "eval_results.json").write_text(
        json.dumps(full_results, indent=2, ensure_ascii=False)
    )
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(f"\n结果保存到 {results_dir}")


if __name__ == "__main__":
    main()
