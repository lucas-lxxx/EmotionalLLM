"""Main evaluation pipeline for black-box transfer attack."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from config import cfg
from sample_loader import load_and_select, AdversarialSample


def get_client(target: str):
    if target == "gemini":
        from gemini_client import GeminiClient
        return GeminiClient()
    elif target == "qwen":
        from qwen_client import QwenClient
        return QwenClient()
    else:
        raise ValueError(f"Unknown target: {target}. Use 'gemini' or 'qwen'.")


def evaluate_sample(client, sample: AdversarialSample, out_dir: Path) -> dict | None:
    """Evaluate one adversarial sample against the target API."""
    out_json = out_dir / f"{sample.sample_id}.json"

    if cfg.skip_existing and out_json.exists():
        try:
            return json.loads(out_json.read_text(encoding="utf-8"))
        except Exception:
            pass

    if not sample.adv_audio_path.exists():
        print(f"  SKIP {sample.sample_id}: adv audio not found at {sample.adv_audio_path}")
        return None

    print(f"  Evaluating {sample.sample_id} ({sample.ground_truth_emotion} -> {sample.target_emotion})...")

    result_data = client.query_emotion_3prompt(sample.adv_audio_path)

    is_success = result_data["majority_label"] == sample.target_emotion

    record = {
        "sample_id": sample.sample_id,
        "ground_truth_emotion": sample.ground_truth_emotion,
        "target_emotion": sample.target_emotion,
        "whitebox_success": sample.whitebox_success,
        "adv_audio_path": str(sample.adv_audio_path),
        "per_prompt_results": result_data["per_prompt"],
        "majority_label": result_data["majority_label"],
        "transfer_success": is_success,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def compute_summary(results: list[dict]) -> dict:
    """Compute Transfer ASR and per-emotion stats."""
    total = len(results)
    if total == 0:
        return {}

    successes = sum(1 for r in results if r.get("transfer_success", False))

    by_emotion = defaultdict(list)
    for r in results:
        by_emotion[r["ground_truth_emotion"]].append(r)

    per_emotion = {}
    for emo, emo_results in sorted(by_emotion.items()):
        emo_total = len(emo_results)
        emo_success = sum(1 for r in emo_results if r.get("transfer_success", False))
        per_emotion[emo] = {
            "total": emo_total,
            "transfer_success": emo_success,
            "transfer_asr": emo_success / emo_total if emo_total > 0 else 0.0,
        }

    # Per-prompt breakdown
    prompt_labels = defaultdict(list)
    for r in results:
        for pr in r.get("per_prompt_results", []):
            idx = pr["prompt_idx"]
            prompt_labels[idx].append(pr["label"] == r["target_emotion"])

    per_prompt_asr = {}
    for idx in sorted(prompt_labels.keys()):
        hits = prompt_labels[idx]
        per_prompt_asr[f"prompt_{idx}"] = sum(hits) / len(hits) if hits else 0.0

    return {
        "total_samples": total,
        "transfer_success": successes,
        "transfer_asr": successes / total,
        "per_emotion": per_emotion,
        "per_prompt_asr": per_prompt_asr,
    }


def run_evaluation(
    target: str,
    num_samples: int | None = None,
    per_emotion: int = 125,
    dry_run: bool = False,
    result_dir: Path | None = None,
    whitebox_dir: Path | None = None,
):
    print(f"=== Black-box Transfer Attack Evaluation ===")
    print(f"Target: {target}")

    if per_emotion and num_samples:
        per_emotion = max(1, num_samples // 4)

    samples = load_and_select(
        result_dir=whitebox_dir,
        per_emotion=per_emotion,
        only_whitebox_success=True,
    )

    if num_samples and len(samples) > num_samples:
        samples = samples[:num_samples]

    print(f"\nSamples to evaluate: {len(samples)}")

    if dry_run:
        print("\n[DRY RUN] Would evaluate the following samples:")
        for s in samples[:20]:
            exists = "EXISTS" if s.adv_audio_path.exists() else "MISSING"
            print(f"  {s.sample_id} | {s.ground_truth_emotion}->{s.target_emotion} | wav={exists}")
        print(f"  ... ({len(samples)} total)")
        return

    if result_dir is None:
        result_dir = cfg.results_dir / target
    result_dir.mkdir(parents=True, exist_ok=True)

    client = get_client(target)
    print(f"Client initialized: {target}")

    all_results = []
    for i, sample in enumerate(samples):
        print(f"[{i+1}/{len(samples)}]", end="")
        record = evaluate_sample(client, sample, result_dir)
        if record is not None:
            all_results.append(record)
            status = "SUCCESS" if record["transfer_success"] else "FAIL"
            print(f"    -> {record['majority_label']} ({status})")

    # Summary
    summary = compute_summary(all_results)
    summary_path = result_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== Results ===")
    print(f"Transfer ASR: {summary.get('transfer_asr', 0):.2%} ({summary.get('transfer_success', 0)}/{summary.get('total_samples', 0)})")
    print(f"\nPer-emotion:")
    for emo, stats in summary.get("per_emotion", {}).items():
        print(f"  {emo}: {stats['transfer_asr']:.2%} ({stats['transfer_success']}/{stats['total']})")
    print(f"\nPer-prompt ASR:")
    for prompt_key, asr in summary.get("per_prompt_asr", {}).items():
        print(f"  {prompt_key}: {asr:.2%}")
    print(f"\nSummary saved to: {summary_path}")


def collect_and_summarize(result_dir: Path):
    """Collect existing results from disk and recompute summary."""
    results = []
    for json_path in sorted(result_dir.rglob("*.json")):
        if json_path.name == "summary.json":
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            if "sample_id" in data and "transfer_success" in data:
                results.append(data)
        except Exception:
            continue

    if not results:
        print("No results found.")
        return

    summary = compute_summary(results)
    summary_path = result_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Recomputed summary from {len(results)} samples -> {summary_path}")
    print(f"Transfer ASR: {summary['transfer_asr']:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Black-box transfer attack evaluation")
    parser.add_argument("--target", type=str, required=True, choices=["gemini", "qwen"],
                        help="Target API: gemini or qwen")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Total number of samples to evaluate (default: all)")
    parser.add_argument("--per_emotion", type=int, default=125,
                        help="Samples per emotion for subset selection (default: 125)")
    parser.add_argument("--dry_run", action="store_true",
                        help="List samples without calling API")
    parser.add_argument("--result_dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--whitebox_dir", type=str, default=None,
                        help="White-box result directory (source of adv samples)")
    parser.add_argument("--aggregate_only", action="store_true",
                        help="Only recompute summary from existing results")
    args = parser.parse_args()

    if args.aggregate_only:
        rdir = Path(args.result_dir) if args.result_dir else cfg.results_dir / args.target
        collect_and_summarize(rdir)
    else:
        run_evaluation(
            target=args.target,
            num_samples=args.num_samples,
            per_emotion=args.per_emotion,
            dry_run=args.dry_run,
            result_dir=Path(args.result_dir) if args.result_dir else None,
            whitebox_dir=Path(args.whitebox_dir) if args.whitebox_dir else None,
        )
