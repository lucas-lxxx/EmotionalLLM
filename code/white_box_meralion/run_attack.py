from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch

from config import cfg
from attack_core import attack_one_sample, compute_target_token_ids
from esd_dataset import AudioSample, create_experiment_samples
from eval_metrics import (
    aggregate_results,
    aggregate_results_by_emotion,
    aggregate_results_by_speaker,
    compute_semantic_similarity,
    compute_wer,
    signal_metrics,
)
from meralion_io import decode_text, load_meralion


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cuda_available() -> bool:
    return torch.cuda.is_available()


def load_audio(path: Path, target_sr: int) -> tuple[torch.Tensor, int]:
    import soundfile as sf

    wav, sr = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = torch.from_numpy(wav).float()
    if sr != target_sr:
        import torchaudio

        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav.unsqueeze(0), sr


def save_audio(path: Path, waveform: torch.Tensor, sr: int) -> None:
    import soundfile as sf

    wav = waveform.detach().cpu().numpy().squeeze()
    sf.write(str(path), wav, sr)


def normalize_emo(text: str, emo_labels: list[str]) -> str:
    text = text.strip().lower()
    # Look for exact label match first
    for label in emo_labels:
        if re.search(rf"\b{re.escape(label)}\b", text):
            return label
    # Fallback: first token
    return text.strip().split(" ")[0] if text else ""


def should_process(
    idx: int,
    start_idx: int | None,
    end_idx: int | None,
    shard_id: int | None,
    num_shards: int,
) -> bool:
    if start_idx is not None and idx < start_idx:
        return False
    if end_idx is not None and idx >= end_idx:
        return False
    if shard_id is not None and (idx % num_shards) != shard_id:
        return False
    return True


def load_esd_samples(args) -> dict[str, list[AudioSample]]:
    all_samples = create_experiment_samples(
        dataset_root=Path(args.esd_root),
        exclude_emotion=cfg.esd_exclude_emotion,
        samples_per_emotion=cfg.esd_samples_per_emotion,
        seed=cfg.seed,
    )
    if args.speaker_id:
        if args.speaker_id not in all_samples:
            raise ValueError(f"Speaker {args.speaker_id} not found in dataset")
        return {args.speaker_id: all_samples[args.speaker_id]}
    return all_samples


def main(args) -> None:
    set_seed(cfg.seed)

    samples_dict = load_esd_samples(args)
    all_samples = []
    for speaker_id, samples in samples_dict.items():
        all_samples.extend(samples)

    print(f"Total samples to process: {len(all_samples)}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    runtime_device = cfg.device
    if runtime_device.startswith("cuda") and not cuda_available():
        runtime_device = "cpu"

    model, processor, torch_extractor = load_meralion(cfg.model_path, runtime_device)
    tokenizer = processor.tokenizer
    target_token_ids = compute_target_token_ids(tokenizer, cfg.target_emotion)
    ignore_index = -100
    target_sr = processor.feature_extractor.sampling_rate

    per_sample_results = []

    for idx, sample in enumerate(all_samples):
        if not should_process(idx, args.start_idx, args.end_idx, args.shard_id, args.num_shards):
            continue

        sample_id = f"{idx:05d}_{sample.speaker_id}_{sample.emotion}_{sample.filename.replace('.wav', '')}"

        if cfg.results_by_speaker and sample.speaker_id != "unknown":
            speaker_dir = results_dir / sample.speaker_id
            speaker_dir.mkdir(parents=True, exist_ok=True)
            out_json = speaker_dir / f"{sample_id}.json"
            out_wav = speaker_dir / f"{sample_id}.wav"
        else:
            out_json = results_dir / f"{sample_id}.json"
            out_wav = results_dir / f"{sample_id}.wav"

        if cfg.skip_existing and out_json.exists():
            continue

        try:
            waveform, sr = load_audio(sample.path, target_sr)
        except Exception as e:
            print(f"[skip] failed to load {sample.path}: {e}")
            continue

        waveform = waveform.to(runtime_device)

        # Decode clean emotion and transcript
        emo_text_clean = [
            decode_text(model, processor, waveform, sr, prompt, cfg.emo_max_new_tokens, cfg.temperature)
            for prompt in cfg.emo_prompts
        ]
        asr_text_clean = decode_text(
            model, processor, waveform, sr, cfg.asr_prompts[0], cfg.asr_max_new_tokens, cfg.temperature,
        )

        # Self-consistency: use model's own transcript as ASR target
        asr_target_token_ids = tokenizer.encode(asr_text_clean, add_special_tokens=False)
        if not asr_target_token_ids:
            asr_target_token_ids = tokenizer.encode(" " + asr_text_clean, add_special_tokens=False)

        try:
            attack_out = attack_one_sample(
                model=model, processor=processor, waveform=waveform, sr=sr,
                target_token_ids=target_token_ids, asr_prompt=cfg.asr_prompts[0],
                asr_target_token_ids=asr_target_token_ids, device=runtime_device,
                ignore_index=ignore_index, torch_extractor=torch_extractor,
            )
        except Exception as e:
            print(f"[skip] attack failed for sample {sample_id}: {e}")
            continue

        waveform_adv = attack_out["waveform_adv"]
        save_audio(out_wav, waveform_adv, sr)

        emo_text_adv = [
            decode_text(model, processor, waveform_adv, sr, prompt, cfg.emo_max_new_tokens, cfg.temperature)
            for prompt in cfg.emo_prompts
        ]
        asr_text_adv = decode_text(
            model, processor, waveform_adv, sr, cfg.asr_prompts[0], cfg.asr_max_new_tokens, cfg.temperature,
        )

        emo_pred_clean = [normalize_emo(t, cfg.emo_labels) for t in emo_text_clean]
        emo_pred_adv = [normalize_emo(t, cfg.emo_labels) for t in emo_text_adv]
        success_emo = all(p == cfg.target_emotion for p in emo_pred_adv)
        wer = compute_wer(asr_text_clean, asr_text_adv)
        semantic_sim = compute_semantic_similarity(asr_text_clean, asr_text_adv, cfg.semantic_sim_model)

        metrics = signal_metrics(waveform_adv, waveform)
        sample_result = {
            "sample_id": sample_id, "path": str(sample.path),
            "speaker_id": sample.speaker_id, "ground_truth_emotion": sample.emotion,
            "target_emotion": cfg.target_emotion,
            "emo_text_clean": emo_text_clean, "emo_text_adv": emo_text_adv,
            "emo_pred_clean": emo_pred_clean, "emo_pred_adv": emo_pred_adv,
            "asr_text_clean": asr_text_clean, "asr_text_adv": asr_text_adv,
            "success_emo": success_emo, "wer": wer,
            "semantic_sim": semantic_sim, "semantic_preserved": semantic_sim >= cfg.semantic_threshold,
            "delta_linf": metrics["delta_linf"], "delta_l2": metrics["delta_l2"], "snr_db": metrics["snr_db"],
            "grad_norm_trace": attack_out["grad_trace"], "loss_trace": attack_out["loss_trace"],
        }

        out_json.write_text(json.dumps(sample_result, ensure_ascii=True, indent=2), encoding="utf-8")
        per_sample_results.append(sample_result)
        print(f"[{idx}] {sample_id} success_emo={success_emo} sem={semantic_sim:.3f} snr={metrics['snr_db']:.2f}dB")

    if args.shard_id is None:
        write_summaries(results_dir, per_sample_results)
    else:
        print(f"Shard {args.shard_id} done ({len(per_sample_results)} samples). "
              f"Run --aggregate_only after all shards finish.")


def collect_results_from_disk(results_dir: Path) -> list[dict]:
    all_results = []
    for json_path in sorted(results_dir.rglob("*.json")):
        if json_path.name.startswith("summary"):
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            if "sample_id" in data:
                all_results.append(data)
        except Exception:
            continue
    return all_results


def write_summaries(results_dir: Path, per_sample_results: list[dict]) -> None:
    summary = aggregate_results(per_sample_results, cfg.wer_thresholds, cfg.semantic_threshold)
    (results_dir / "summary_all.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    summary_by_speaker = aggregate_results_by_speaker(per_sample_results, cfg.wer_thresholds, cfg.semantic_threshold)
    (results_dir / "summary_by_speaker.json").write_text(
        json.dumps(summary_by_speaker, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    summary_by_emotion = aggregate_results_by_emotion(per_sample_results, cfg.wer_thresholds, cfg.semantic_threshold)
    (results_dir / "summary_by_emotion.json").write_text(
        json.dumps(summary_by_emotion, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    print(f"Summaries written to {results_dir} ({summary.get('num_samples', 0)} samples)")


def aggregate_only(args) -> None:
    results_dir = Path(args.results_dir)
    per_sample_results = collect_results_from_disk(results_dir)
    print(f"Collected {len(per_sample_results)} sample results from {results_dir}")
    write_summaries(results_dir, per_sample_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate_only", action="store_true")
    parser.add_argument("--esd_root", type=str, default=str(cfg.esd_dataset_root))
    parser.add_argument("--speaker_id", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default=str(cfg.results_dir))
    parser.add_argument("--start_idx", type=int, default=None)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--shard_id", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    args = parser.parse_args()

    if args.aggregate_only:
        aggregate_only(args)
    else:
        main(args)
