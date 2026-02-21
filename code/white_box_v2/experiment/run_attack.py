from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch

from config import cfg
from attack_core import attack_one_sample, compute_target_token_ids
from eval_metrics import aggregate_results, aggregate_results_by_speaker, aggregate_results_by_emotion, compute_semantic_similarity, compute_wer, normalize_text, signal_metrics
from opens2s_io import decode_text, load_opens2s
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
                raise RuntimeError("Resample requires torchaudio or ffmpeg; install or pre-resample inputs.")
            return waveform, sr
        except Exception:
            import array
            import subprocess

            cmd = [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-ac",
                "1",
                "-ar",
                str(target_sr),
                "-f",
                "f32le",
                "-",
            ]
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


def parse_sample_list(path: Path) -> list[Path]:
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        samples.append(Path(parts[0]))
    return samples


def normalize_emo(text: str, labels: list[str]) -> str:
    # Methodology §1.2: success if exact target word after normalization.
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
    """从 ESD 数据集加载样本"""
    all_samples = create_experiment_samples(
        dataset_root=Path(args.esd_root),
        exclude_emotion=cfg.esd_exclude_emotion,
        samples_per_emotion=cfg.esd_samples_per_emotion,
        seed=cfg.seed,
    )

    # 如果指定了说话人，只返回该说话人
    if args.speaker_id:
        if args.speaker_id not in all_samples:
            raise ValueError(f"Speaker {args.speaker_id} not found in dataset")
        return {args.speaker_id: all_samples[args.speaker_id]}

    return all_samples


def load_sample_list(args) -> dict[str, list[AudioSample]]:
    """从 sample_list 加载样本（兼容旧模式）"""
    sample_paths = parse_sample_list(Path(args.sample_list))

    # 转换为 AudioSample 格式
    samples = []
    for path in sample_paths:
        samples.append(
            AudioSample(
                path=path,
                speaker_id="unknown",
                emotion="unknown",
                filename=path.name,
            )
        )

    return {"default": samples}


def main() -> None:
    parser = argparse.ArgumentParser()
    # 新增：模式选择
    parser.add_argument("--mode", type=str, default="esd", choices=["esd", "sample_list"],
                       help="数据加载模式：esd 或 sample_list")
    parser.add_argument("--esd_root", type=str, default=str(cfg.esd_dataset_root),
                       help="ESD 数据集根目录")
    parser.add_argument("--speaker_id", type=str, default=None,
                       help="指定单个说话人ID，None表示处理所有说话人")
    # 原有参数
    parser.add_argument("--sample_list", type=str, default=str(cfg.sample_list_path))
    parser.add_argument("--results_dir", type=str, default=str(cfg.results_dir))
    parser.add_argument("--start_idx", type=int, default=None)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--shard_id", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    args = parser.parse_args()

    set_seed(cfg.seed)

    # 根据模式加载数据
    if args.mode == "esd":
        samples_dict = load_esd_samples(args)
    else:
        samples_dict = load_sample_list(args)

    # 展平所有样本为单个列表
    all_samples = []
    for speaker_id, samples in samples_dict.items():
        all_samples.extend(samples)

    print(f"Total samples to process: {len(all_samples)}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    runtime_device = cfg.device
    if runtime_device.startswith("cuda") and not cuda_available():
        runtime_device = "cpu"
    model, tokenizer, audio_extractor, torch_extractor = load_opens2s(cfg.model_path, runtime_device, cfg.opens2s_root)
    target_token_ids = compute_target_token_ids(tokenizer, cfg.target_emotion)
    ignore_index = -100
    target_sr = audio_extractor.sampling_rate

    per_sample_results = []

    for idx, sample in enumerate(all_samples):
        if not should_process(idx, args.start_idx, args.end_idx, args.shard_id, args.num_shards):
            continue

        # 生成样本ID和输出路径
        sample_id = f"{idx:05d}_{sample.speaker_id}_{sample.emotion}_{sample.filename.replace('.wav', '')}"

        # 按说话人分目录（如果启用）
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

        # Methodology §8.1: decode clean emotion and transcript with fixed settings.
        emo_text_clean = [
            decode_text(
                model,
                tokenizer,
                waveform,
                sr,
                prompt,
                cfg.emo_max_new_tokens,
                cfg.temperature,
                audio_extractor=audio_extractor,
                system_prompt=cfg.system_prompt,
            )
            for prompt in cfg.emo_prompts
        ]
        asr_text_clean = decode_text(
            model,
            tokenizer,
            waveform,
            sr,
            cfg.asr_prompts[0],
            cfg.asr_max_new_tokens,
            cfg.temperature,
            audio_extractor=audio_extractor,
            system_prompt=cfg.system_prompt,
        )

        # Methodology §5.2: self-consistency uses OpenS2S transcript as target.
        asr_target_token_ids = tokenizer.encode(asr_text_clean, add_special_tokens=False)
        if not asr_target_token_ids:
            asr_target_token_ids = tokenizer.encode(" " + asr_text_clean, add_special_tokens=False)

        attack_out = attack_one_sample(
            model=model,
            tokenizer=tokenizer,
            waveform=waveform,
            sr=sr,
            target_token_ids=target_token_ids,
            asr_prompt=cfg.asr_prompts[0],
            asr_target_token_ids=asr_target_token_ids,
            device=runtime_device,
            ignore_index=ignore_index,
            torch_extractor=torch_extractor,
            system_prompt=cfg.system_prompt,
        )

        waveform_adv = attack_out["waveform_adv"]
        save_audio(out_wav, waveform_adv, sr)

        # Methodology §8.1: decode adversarial outputs.
        emo_text_adv = [
            decode_text(
                model,
                tokenizer,
                waveform_adv,
                sr,
                prompt,
                cfg.emo_max_new_tokens,
                cfg.temperature,
                audio_extractor=audio_extractor,
                system_prompt=cfg.system_prompt,
            )
            for prompt in cfg.emo_prompts
        ]
        asr_text_adv = decode_text(
            model,
            tokenizer,
            waveform_adv,
            sr,
            cfg.asr_prompts[0],
            cfg.asr_max_new_tokens,
            cfg.temperature,
            audio_extractor=audio_extractor,
            system_prompt=cfg.system_prompt,
        )

        # Methodology §1.2: success criteria.
        emo_pred_clean = [normalize_emo(t, cfg.emo_labels) for t in emo_text_clean]
        emo_pred_adv = [normalize_emo(t, cfg.emo_labels) for t in emo_text_adv]
        success_emo = all(p == cfg.target_emotion for p in emo_pred_adv)
        wer = compute_wer(asr_text_clean, asr_text_adv)
        semantic_sim = compute_semantic_similarity(
            asr_text_clean, asr_text_adv, cfg.semantic_sim_model
        )

        metrics = signal_metrics(waveform_adv, waveform)
        sample_result = {
            "sample_id": sample_id,
            "path": str(sample.path),
            # 新增：元数据
            "speaker_id": sample.speaker_id,
            "ground_truth_emotion": sample.emotion,
            "target_emotion": cfg.target_emotion,
            # 原有字段
            "emo_text_clean": emo_text_clean,
            "emo_text_adv": emo_text_adv,
            "emo_pred_clean": emo_pred_clean,
            "emo_pred_adv": emo_pred_adv,
            "asr_text_clean": asr_text_clean,
            "asr_text_adv": asr_text_adv,
            "success_emo": success_emo,
            "wer": wer,
            "semantic_sim": semantic_sim,
            "semantic_preserved": semantic_sim >= cfg.semantic_threshold,
            "delta_linf": metrics["delta_linf"],
            "delta_l2": metrics["delta_l2"],
            "snr_db": metrics["snr_db"],
            "grad_norm_trace": attack_out["grad_trace"],
            "loss_trace": attack_out["loss_trace"],
        }

        out_json.write_text(json.dumps(sample_result, ensure_ascii=True, indent=2), encoding="utf-8")
        per_sample_results.append(sample_result)

    summary = aggregate_results(per_sample_results, cfg.wer_thresholds, cfg.semantic_threshold)
    summary_path = results_dir / "summary_all.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    summary_by_speaker = aggregate_results_by_speaker(per_sample_results, cfg.wer_thresholds, cfg.semantic_threshold)
    (results_dir / "summary_by_speaker.json").write_text(
        json.dumps(summary_by_speaker, ensure_ascii=True, indent=2), encoding="utf-8"
    )

    summary_by_emotion = aggregate_results_by_emotion(per_sample_results, cfg.wer_thresholds, cfg.semantic_threshold)
    (results_dir / "summary_by_emotion.json").write_text(
        json.dumps(summary_by_emotion, ensure_ascii=True, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
