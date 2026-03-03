from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch

from config import cfg
from attack_core import attack_one_sample, compute_target_token_ids
from eval_metrics import (
    aggregate_results, aggregate_results_by_speaker, aggregate_results_by_emotion,
    compute_semantic_similarity, compute_wer, normalize_text, signal_metrics,
)
from voxtral_io import decode_text, load_voxtral
from esd_dataset import AudioSample, create_experiment_samples


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def cuda_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        torch.cuda.get_device_properties(0)
    except Exception:
        return False
    return True


def load_audio(path: Path, target_sr: int) -> tuple[torch.Tensor, int]:
    try:
        import torchaudio
        waveform, sr = torchaudio.load(str(path))
        if waveform.dim() > 1 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
            sr = target_sr
        return waveform.float(), sr
    except Exception:
        try:
            import soundfile as sf
            data, sr = sf.read(str(path), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            waveform = torch.from_numpy(data).unsqueeze(0)
            if sr != target_sr:
                raise RuntimeError("Resample requires torchaudio; install or pre-resample inputs.")
            return waveform, sr
        except Exception:
            import array, subprocess
            cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(target_sr), "-f", "f32le", "-"]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            data = array.array("f")
            data.frombytes(proc.stdout)
            waveform = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
            return waveform, target_sr


def save_audio(path: Path, waveform: torch.Tensor, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = waveform.detach().cpu()
    if data.dim() == 2 and data.size(0) == 1:
        data = data.squeeze(0)
    try:
        import soundfile as sf
        sf.write(str(path), data.numpy(), sr)
    except Exception:
        import torchaudio
        torchaudio.save(str(path), data.unsqueeze(0), sr)


def normalize_emo(text: str, labels: list[str]) -> str:
    text = normalize_text(text)
    text = re.sub(r"[^a-z]+", " ", text)
    if not text.strip():
        return ""
    for label in labels:
        if re.search(rf"\b{re.escape(label)}\b", text):
            return label
    return text.strip().split(" ")[0]


def should_process(idx: int, start_idx: int | None, end_idx: int | None, shard_id: int | None, num_shards: int) -> bool:
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

    model, processor, torch_extractor = load_voxtral(cfg.model_path, runtime_device)
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

        waveform, sr = load_audio(sample.path, target_sr)
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

        attack_out = attack_one_sample(
            model=model, tokenizer=tokenizer, waveform=waveform, sr=sr,
            target_token_ids=target_token_ids, asr_prompt=cfg.asr_prompts[0],
            asr_target_token_ids=asr_target_token_ids, device=runtime_device,
            ignore_index=ignore_index, torch_extractor=torch_extractor,
        )

        waveform_adv = attack_out["waveform_adv"]
        save_audio(out_wav, waveform_adv, sr)

        # Decode adversarial outputs
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
