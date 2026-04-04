"""Load white-box attack results and select adversarial samples for black-box evaluation."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config import cfg


@dataclass
class AdversarialSample:
    sample_id: str
    original_audio_path: str  # original ESD path on server
    adv_audio_path: Path      # adversarial WAV path (same dir as JSON)
    ground_truth_emotion: str
    target_emotion: str
    whitebox_success: bool
    whitebox_json_path: Path

    def __str__(self) -> str:
        return f"{self.sample_id} ({self.ground_truth_emotion}->{self.target_emotion}, wb={'Y' if self.whitebox_success else 'N'})"


def load_whitebox_results(result_dir: Path) -> list[AdversarialSample]:
    """Scan all per-sample JSON files from white-box result directory."""
    samples = []
    for json_path in sorted(result_dir.rglob("*.json")):
        if json_path.name.startswith("summary") or json_path.name.startswith("report"):
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            if "sample_id" not in data:
                continue

            wav_path = json_path.with_suffix(".wav")

            samples.append(AdversarialSample(
                sample_id=data["sample_id"],
                original_audio_path=data.get("path", ""),
                adv_audio_path=wav_path,
                ground_truth_emotion=data.get("ground_truth_emotion", ""),
                target_emotion=data.get("target_emotion", ""),
                whitebox_success=data.get("success_emo", False),
                whitebox_json_path=json_path,
            ))
        except Exception as e:
            print(f"Warning: skipping {json_path}: {e}")
            continue

    return samples


def select_subset(
    samples: list[AdversarialSample],
    per_emotion: int = 125,
    only_whitebox_success: bool = True,
    seed: int = 42,
) -> list[AdversarialSample]:
    """Select a balanced subset: per_emotion samples per source emotion."""
    if only_whitebox_success:
        samples = [s for s in samples if s.whitebox_success]

    by_emotion: dict[str, list[AdversarialSample]] = {}
    for s in samples:
        by_emotion.setdefault(s.ground_truth_emotion, []).append(s)

    rng = random.Random(seed)
    selected = []
    for emotion in sorted(by_emotion.keys()):
        pool = by_emotion[emotion]
        if len(pool) <= per_emotion:
            selected.extend(pool)
            print(f"  {emotion}: {len(pool)} samples (all available)")
        else:
            chosen = rng.sample(pool, per_emotion)
            selected.extend(chosen)
            print(f"  {emotion}: {per_emotion}/{len(pool)} samples selected")

    return selected


def load_and_select(
    result_dir: Optional[Path] = None,
    per_emotion: int = 125,
    only_whitebox_success: bool = True,
    seed: int = 42,
) -> list[AdversarialSample]:
    """Convenience: load + select."""
    if result_dir is None:
        result_dir = cfg.whitebox_result_dir

    print(f"Loading white-box results from: {result_dir}")
    all_samples = load_whitebox_results(result_dir)
    print(f"Total samples loaded: {len(all_samples)}")

    success_count = sum(1 for s in all_samples if s.whitebox_success)
    print(f"White-box successes: {success_count}/{len(all_samples)} ({100*success_count/len(all_samples):.1f}%)")

    print(f"Selecting subset (per_emotion={per_emotion}, only_success={only_whitebox_success}):")
    subset = select_subset(all_samples, per_emotion, only_whitebox_success, seed)
    print(f"Selected {len(subset)} samples total")
    return subset


if __name__ == "__main__":
    samples = load_and_select(per_emotion=5)
    for s in samples[:10]:
        print(f"  {s}")
