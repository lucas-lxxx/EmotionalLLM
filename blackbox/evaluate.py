"""Main evaluation pipeline for black-box transfer attack.

Supports all active target APIs × 4 surrogate groups.
Usage:
    python evaluate.py --surrogate opens2s_en --target gemini_flash
    python evaluate.py --surrogate opens2s_en --target gemini_flash --dry_run
    python evaluate.py --surrogate opens2s_en --target gemini_flash --aggregate_only
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import local

from config import cfg
from sample_loader import load_surrogate_group, AdversarialSample


_thread_clients = local()


def _normalize_loaded_record(record: dict) -> dict:
    majority_label = record.get("majority_label", "")
    target_emotion = record.get("target_emotion", "")
    ground_truth = record.get("ground_truth_emotion", "")
    if "matches_target" not in record:
        record["matches_target"] = majority_label == target_emotion or record.get("transfer_success", False)
    if "is_correct" not in record:
        record["is_correct"] = majority_label == ground_truth
    if "transfer_success" not in record:
        record["transfer_success"] = record["matches_target"]
    return record


def get_client(target_key: str):
    """Create API client for the given target."""
    info = cfg.target_list[target_key]
    client_type = info["client"]
    model = cfg.get_target_model(target_key)

    if client_type == "gemini":
        from gemini_client import GeminiClient
        return GeminiClient(model=model)
    elif client_type == "qwen":
        from qwen_client import QwenClient
        return QwenClient(model=model)
    elif client_type == "gpt4o":
        from gpt4o_client import GPT4oClient
        return GPT4oClient(model=model)
    else:
        raise ValueError(f"Unknown client type: {client_type}")


def _resolve_eval_audio_path(sample: AdversarialSample, audio_type: str) -> Path | None:
    if audio_type == "adv":
        return sample.adv_audio_path
    if audio_type == "noise":
        return cfg.noise_dir / sample.surrogate_key / f"{sample.sample_id}.wav"
    if audio_type == "clean":
        return sample.resolve_clean_audio_path()
    raise ValueError(f"Unsupported audio_type: {audio_type}")


def _get_thread_client(target_key: str):
    cache = getattr(_thread_clients, "cache", None)
    if cache is None:
        cache = {}
        _thread_clients.cache = cache
    if target_key not in cache:
        cache[target_key] = get_client(target_key)
    return cache[target_key]


def _evaluate_sample_for_target(target_key: str, sample: AdversarialSample, out_dir: Path, audio_type: str) -> dict | None:
    client = _get_thread_client(target_key)
    return evaluate_sample(client, sample, out_dir, audio_type)


def evaluate_sample(client, sample: AdversarialSample, out_dir: Path, audio_type: str) -> dict | None:
    """Evaluate one sample against a target API."""
    out_json = out_dir / f"{sample.sample_id}.json"

    if cfg.skip_existing and out_json.exists():
        try:
            return _normalize_loaded_record(json.loads(out_json.read_text(encoding="utf-8")))
        except Exception:
            pass

    eval_audio_path = _resolve_eval_audio_path(sample, audio_type)
    if eval_audio_path is None or not eval_audio_path.exists():
        return None

    result_data = client.query_emotion_3prompt(eval_audio_path)

    majority_label = result_data["majority_label"]
    matches_target = majority_label == sample.target_emotion
    is_correct = majority_label == sample.ground_truth_emotion

    record = {
        "sample_id": sample.sample_id,
        "speaker_id": sample.speaker_id,
        "ground_truth_emotion": sample.ground_truth_emotion,
        "target_emotion": sample.target_emotion,
        "whitebox_success": sample.whitebox_success,
        "audio_type": audio_type,
        "adv_audio_path": str(sample.adv_audio_path),
        "eval_audio_path": str(eval_audio_path),
        "per_prompt_results": result_data["per_prompt"],
        "majority_label": majority_label,
        "transfer_success": matches_target,
        "matches_target": matches_target,
        "is_correct": is_correct,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def compute_summary(results: list[dict], audio_type: str) -> dict:
    """Compute summary metrics for the current audio type."""
    total = len(results)
    if total == 0:
        return {"total_samples": 0}

    target_hits = sum(1 for r in results if r.get("matches_target", r.get("transfer_success", False)))
    correct_hits = sum(1 for r in results if r.get("is_correct", r.get("majority_label", "") == r.get("ground_truth_emotion", "")))
    primary_hits = target_hits if audio_type == "adv" else correct_hits

    # Per-emotion
    by_emotion = defaultdict(list)
    for r in results:
        by_emotion[r["ground_truth_emotion"]].append(r)

    per_emotion = {}
    for emo, emo_results in sorted(by_emotion.items()):
        emo_total = len(emo_results)
        emo_target_hits = sum(1 for r in emo_results if r.get("matches_target", r.get("transfer_success", False)))
        emo_correct_hits = sum(1 for r in emo_results if r.get("is_correct", r.get("majority_label", "") == r.get("ground_truth_emotion", "")))
        emo_entry = {
            "total": emo_total,
            "target_hits": emo_target_hits,
            "target_rate": round(emo_target_hits / emo_total, 4) if emo_total > 0 else 0.0,
            "correct_predictions": emo_correct_hits,
            "accuracy": round(emo_correct_hits / emo_total, 4) if emo_total > 0 else 0.0,
        }
        if audio_type == "adv":
            emo_entry["transfer_success"] = emo_target_hits
            emo_entry["transfer_asr"] = emo_entry["target_rate"]
        per_emotion[emo] = emo_entry

    # Per-speaker
    by_speaker = defaultdict(list)
    for r in results:
        by_speaker[r.get("speaker_id", "unknown")].append(r)

    per_speaker = {}
    for spk, spk_results in sorted(by_speaker.items()):
        spk_total = len(spk_results)
        spk_target_hits = sum(1 for r in spk_results if r.get("matches_target", r.get("transfer_success", False)))
        spk_correct_hits = sum(1 for r in spk_results if r.get("is_correct", r.get("majority_label", "") == r.get("ground_truth_emotion", "")))
        spk_entry = {
            "total": spk_total,
            "target_hits": spk_target_hits,
            "target_rate": round(spk_target_hits / spk_total, 4) if spk_total > 0 else 0.0,
            "correct_predictions": spk_correct_hits,
            "accuracy": round(spk_correct_hits / spk_total, 4) if spk_total > 0 else 0.0,
        }
        if audio_type == "adv":
            spk_entry["transfer_success"] = spk_target_hits
            spk_entry["transfer_asr"] = spk_entry["target_rate"]
        per_speaker[spk] = spk_entry

    # Per-prompt
    prompt_labels = defaultdict(list)
    for r in results:
        for pr in r.get("per_prompt_results", []):
            idx = pr["prompt_idx"]
            if audio_type == "adv":
                prompt_labels[idx].append(pr["label"] == r["target_emotion"])
            else:
                prompt_labels[idx].append(pr["label"] == r["ground_truth_emotion"])

    per_prompt_rates = {}
    for idx in sorted(prompt_labels.keys()):
        hits = prompt_labels[idx]
        per_prompt_rates[f"prompt_{idx}"] = round(sum(hits) / len(hits), 4) if hits else 0.0

    # Prediction distribution (what label did the API actually return?)
    pred_dist = Counter(r["majority_label"] for r in results)

    summary = {
        "total_samples": total,
        "metric_name": "transfer_asr" if audio_type == "adv" else "accuracy",
        "metric_count": primary_hits,
        "metric_rate": round(primary_hits / total, 4),
        "target_hits": target_hits,
        "target_rate": round(target_hits / total, 4),
        "correct_predictions": correct_hits,
        "accuracy": round(correct_hits / total, 4),
        "per_emotion": per_emotion,
        "per_speaker": per_speaker,
        "per_prompt_rate": per_prompt_rates,
        "prediction_distribution": dict(pred_dist.most_common()),
    }
    if audio_type == "adv":
        summary["transfer_success"] = target_hits
        summary["transfer_asr"] = summary["target_rate"]
    return summary


def run_evaluation(
    surrogate_key: str,
    target_key: str,
    per_emotion: int = 250,
    dry_run: bool = False,
    audio_type: str = "adv",  # "adv", "clean", "noise"
):
    """Run evaluation for one surrogate × target combination."""
    surrogate_info = cfg.surrogate_groups[surrogate_key]
    target_info = cfg.target_list[target_key]

    print(f"\n{'='*70}")
    print(f"Surrogate: {surrogate_info['name']}  →  Target: {target_info['name']}")
    print(f"Audio type: {audio_type}")
    print(f"{'='*70}")

    # Load samples
    samples = load_surrogate_group(surrogate_key, per_emotion=per_emotion, check_wav=True)

    if not samples:
        print("No samples loaded. Skipping.")
        return None

    for s in samples:
        s.surrogate_key = surrogate_key

    # Filter to samples with the requested audio available
    available = []
    for s in samples:
        eval_path = _resolve_eval_audio_path(s, audio_type)
        if eval_path is not None and eval_path.exists():
            available.append(s)
    print(f"Samples with requested audio: {len(available)}/{len(samples)}")

    if not available:
        print("No WAV files available. Skipping.")
        return None

    if dry_run:
        print(f"\n[DRY RUN] Would evaluate {len(available)} samples")
        for s in available[:10]:
            print(f"  {s.sample_id} | {s.ground_truth_emotion}->{s.target_emotion}")
        if len(available) > 10:
            print(f"  ... ({len(available)} total)")
        return None

    # Output directory
    out_dir = cfg.results_dir / audio_type / surrogate_key / target_key
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    worker_count = min(max(cfg.sample_parallelism, 1), len(available))
    print(f"Client initialized: {target_info['name']} ({cfg.get_target_model(target_key)})")
    print(f"Sample workers: {worker_count}")

    if worker_count == 1:
        client = get_client(target_key)
        for i, sample in enumerate(available):
            print(f"[{i+1}/{len(available)}] {sample.sample_id}", end="")
            record = evaluate_sample(client, sample, out_dir, audio_type)
            if record is not None:
                all_results.append(record)
                is_primary_hit = record["matches_target"] if audio_type == "adv" else record["is_correct"]
                status = "✓" if is_primary_hit else "✗"
                print(f" → {record['majority_label']} ({status})")
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(_evaluate_sample_for_target, target_key, sample, out_dir, audio_type): sample
                for sample in available
            }
            for i, future in enumerate(as_completed(future_map), start=1):
                sample = future_map[future]
                try:
                    record = future.result()
                except Exception as e:
                    print(f"[{i}/{len(available)}] {sample.sample_id} → ERROR {e}")
                    continue

                if record is None:
                    print(f"[{i}/{len(available)}] {sample.sample_id} → missing audio")
                    continue

                all_results.append(record)
                is_primary_hit = record["matches_target"] if audio_type == "adv" else record["is_correct"]
                status = "✓" if is_primary_hit else "✗"
                if i <= 10 or i % 25 == 0 or i == len(available):
                    print(f"[{i}/{len(available)}] {sample.sample_id} → {record['majority_label']} ({status})")

    all_results.sort(key=lambda item: item["sample_id"])

    # Save summary
    summary = compute_summary(all_results, audio_type)
    summary["surrogate"] = surrogate_key
    summary["target"] = target_key
    summary["audio_type"] = audio_type
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    metric_key = summary.get("metric_name", "metric_rate")
    metric_rate = summary.get(metric_key, summary.get("metric_rate", 0))
    print(f"\n{metric_key}: {metric_rate:.2%} ({summary.get('metric_count', 0)}/{summary.get('total_samples', 0)})")
    for emo, stats in summary.get("per_emotion", {}).items():
        emo_rate = stats["target_rate"] if audio_type == "adv" else stats["accuracy"]
        emo_hits = stats["target_hits"] if audio_type == "adv" else stats["correct_predictions"]
        print(f"  {emo}: {emo_rate:.2%} ({emo_hits}/{stats['total']})")

    return summary


def collect_and_summarize(result_dir: Path) -> dict | None:
    """Recompute summary from existing result JSONs."""
    results = []
    for json_path in sorted(result_dir.rglob("*.json")):
        if json_path.name == "summary.json":
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            if "sample_id" in data and "transfer_success" in data:
                results.append(_normalize_loaded_record(data))
        except Exception:
            continue

    if not results:
        print(f"No results found in {result_dir}")
        return None

    audio_type = result_dir.parts[-3] if len(result_dir.parts) >= 3 else "adv"
    summary = compute_summary(results, audio_type)
    summary_path = result_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Recomputed from {len(results)} samples: ASR={summary['transfer_asr']:.2%}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Black-box transfer attack evaluation")
    parser.add_argument("--surrogate", type=str, required=True,
                        choices=list(cfg.surrogate_groups.keys()),
                        help="Surrogate group key")
    parser.add_argument("--target", type=str, required=True,
                        choices=list(cfg.target_list.keys()),
                        help="Target API key")
    parser.add_argument("--per_emotion", type=int, default=250)
    parser.add_argument("--audio_type", type=str, default="adv",
                        choices=["adv", "clean", "noise"])
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")
    args = parser.parse_args()

    if args.aggregate_only:
        rdir = cfg.results_dir / args.audio_type / args.surrogate / args.target
        collect_and_summarize(rdir)
    else:
        run_evaluation(
            surrogate_key=args.surrogate,
            target_key=args.target,
            per_emotion=args.per_emotion,
            dry_run=args.dry_run,
            audio_type=args.audio_type,
        )
