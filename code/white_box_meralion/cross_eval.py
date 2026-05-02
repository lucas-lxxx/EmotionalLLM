"""
Cross-model transferability evaluation: use MERaLiON-2-3B as the TARGET (evaluator)
on adversarial wav files produced by OTHER source models.

For each wav in source_dir (recursive .wav glob), run MERaLiON inference with
the 3 emotion prompts, majority vote, and record whether the prediction equals
the configured target_emotion ('happy'). Writes per-sample results + summary.

Invocation:
    python cross_eval.py --source_dir <dir> --tag <label> --max_per_dataset 60
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from config import cfg
from meralion_io import decode_text, load_meralion


def normalize_emo(text: str, labels: list[str]) -> str:
    t = text.strip().lower()
    for lab in labels:
        if re.search(rf"\b{re.escape(lab)}\b", t):
            return lab
    return t.split(" ")[0] if t else ""


def load_audio(path: Path, target_sr: int) -> torch.Tensor:
    wav, sr = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav_t = torch.from_numpy(wav).float()
    if sr != target_sr:
        import torchaudio
        wav_t = torchaudio.functional.resample(wav_t, sr, target_sr)
    return wav_t.unsqueeze(0)


def idx_key(p: Path) -> int:
    m = re.match(r"(\d+)_", p.name)
    return int(m.group(1)) if m else 10**9


def eval_one(model, processor, target_sr, device, source_dir: Path, tag: str,
             max_per_dataset: int, out_dir: Path) -> dict:
    all_wavs = list(source_dir.rglob("*.wav"))
    all_wavs.sort(key=idx_key)
    if max_per_dataset > 0:
        wavs = [w for w in all_wavs if idx_key(w) < max_per_dataset]
    else:
        wavs = all_wavs
    print(f"\n=== tag={tag} source={source_dir} total={len(all_wavs)} selected={len(wavs)} ===")

    per_sample = []
    success = 0
    for i, wav_path in enumerate(wavs):
        try:
            wav = load_audio(wav_path, target_sr).to(device)
        except Exception as e:
            print(f"  [skip] {wav_path.name}: {e}")
            continue
        preds = []
        for prompt in cfg.emo_prompts:
            txt = decode_text(model, processor, wav, target_sr, prompt,
                              cfg.emo_max_new_tokens, cfg.temperature)
            preds.append(normalize_emo(txt, cfg.emo_labels))
        vote = Counter(preds).most_common(1)[0][0]
        ok = vote == cfg.target_emotion
        if ok:
            success += 1
        per_sample.append({
            "wav": str(wav_path), "preds": preds,
            "majority": vote, "success": ok,
        })
        if (i + 1) % 10 == 0 or i == len(wavs) - 1:
            print(f"  [{i+1}/{len(wavs)}] running ASR={success/(i+1)*100:.2f}%")

    asr = success / len(per_sample) * 100 if per_sample else 0.0
    summary = {
        "tag": tag, "target_model": "MERaLiON-2-3B",
        "source_dir": str(source_dir), "num_samples": len(per_sample),
        "targeted_asr": asr, "target_emotion": cfg.target_emotion,
    }
    (out_dir / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    (out_dir / f"detail_{tag}.json").write_text(json.dumps(per_sample, indent=2))
    print(f"[target=MERaLiON] tag={tag} ASR={asr:.2f}% ({success}/{len(per_sample)})")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True,
                    help="comma-separated 'source_dir:tag' pairs, e.g. "
                         "'/path/to/voxtral_en:V2M_EN,/path/to/voxtral_cn:V2M_CN'")
    ap.add_argument("--max_per_dataset", type=int, default=60)
    ap.add_argument("--out_dir", default="/data1/lixiang/EmotionalLLM/code/white_box_meralion/result/cross_eval")
    args = ap.parse_args()

    device = cfg.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    print(f"[target=MERaLiON] loading model on {device}...")
    model, processor, _ = load_meralion(cfg.model_path, device)
    target_sr = processor.feature_extractor.sampling_rate

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = [p for p in args.pairs.split(",") if p]
    all_summaries = []
    for p in pairs:
        src, tag = p.rsplit(":", 1)
        s = eval_one(model, processor, target_sr, device,
                     Path(src), tag, args.max_per_dataset, out_dir)
        all_summaries.append(s)

    (out_dir / "all_summaries.json").write_text(json.dumps(all_summaries, indent=2))
    print("\n=== DONE ===")
    for s in all_summaries:
        print(f"  {s['tag']:30s}: ASR={s['targeted_asr']:.2f}% (N={s['num_samples']})")


if __name__ == "__main__":
    main()
