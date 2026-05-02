"""Load white-box attack results and select adversarial samples for black-box evaluation."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config import cfg


@dataclass
class AdversarialSample:
    sample_id: str
    speaker_id: str
    original_audio_path: str  # original ESD path (may be remote)
    adv_audio_path: Path      # adversarial WAV path
    ground_truth_emotion: str
    target_emotion: str
    whitebox_success: bool
    whitebox_json_path: Path
    snr_db: float = 0.0
    surrogate_key: str = ""

    def __str__(self) -> str:
        return f"{self.sample_id} ({self.ground_truth_emotion}->{self.target_emotion}, wb={'Y' if self.whitebox_success else 'N'})"

    def resolve_clean_audio_path(self) -> Path | None:
        return cfg.resolve_clean_audio_path(self.original_audio_path)


def load_whitebox_results(result_dir: Path) -> list[AdversarialSample]:
    """Scan all per-sample JSON files from white-box result directory."""
    samples = []
    for json_path in sorted(result_dir.rglob("*.json")):
        if json_path.name.startswith(("summary", "report", "cleaned", "judge")):
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            if "sample_id" not in data:
                continue

            wav_path = json_path.with_suffix(".wav")

            samples.append(AdversarialSample(
                sample_id=data["sample_id"],
                speaker_id=data.get("speaker_id", ""),
                original_audio_path=data.get("path", ""),
                adv_audio_path=wav_path,
                ground_truth_emotion=data.get("ground_truth_emotion", ""),
                target_emotion=data.get("target_emotion", cfg.target_emotion),
                whitebox_success=data.get("success_emo", False),
                whitebox_json_path=json_path,
                snr_db=data.get("snr_db", 0.0),
            ))
        except Exception as e:
            continue

    return samples


def select_subset(
    samples: list[AdversarialSample],
    per_emotion: int = 250,
    only_whitebox_success: bool = True,
    seed: int = 42,
) -> list[AdversarialSample]:
    """Select a balanced subset: per_emotion samples per source emotion, speaker-balanced."""
    if only_whitebox_success:
        samples = [s for s in samples if s.whitebox_success]

    by_emotion: dict[str, list[AdversarialSample]] = defaultdict(list)
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
            # Speaker-balanced selection
            by_speaker: dict[str, list] = defaultdict(list)
            for s in pool:
                by_speaker[s.speaker_id].append(s)

            n_speakers = len(by_speaker)
            per_speaker = per_emotion // n_speakers if n_speakers > 0 else per_emotion
            remainder = per_emotion - per_speaker * n_speakers

            chosen = []
            for i, (spk, spk_samples) in enumerate(sorted(by_speaker.items())):
                n = per_speaker + (1 if i < remainder else 0)
                if len(spk_samples) <= n:
                    chosen.extend(spk_samples)
                else:
                    chosen.extend(rng.sample(spk_samples, n))

            selected.extend(chosen)
            print(f"  {emotion}: {len(chosen)}/{len(pool)} samples selected ({n_speakers} speakers)")

    return selected


def load_and_select(
    result_dir: Optional[Path] = None,
    per_emotion: int = 250,
    only_whitebox_success: bool = True,
    seed: int = 42,
    check_wav: bool = False,
) -> list[AdversarialSample]:
    """Load white-box results and select balanced subset."""
    if result_dir is None:
        result_dir = cfg.opens2s_en_dir

    print(f"Loading white-box results from: {result_dir}")
    all_samples = load_whitebox_results(result_dir)
    print(f"Total samples loaded: {len(all_samples)}")

    if not all_samples:
        print("WARNING: No samples found!")
        return []

    success_count = sum(1 for s in all_samples if s.whitebox_success)
    print(f"White-box successes: {success_count}/{len(all_samples)} ({100*success_count/len(all_samples):.1f}%)")

    if check_wav:
        wav_exists = sum(1 for s in all_samples if s.adv_audio_path.exists())
        print(f"WAV files present: {wav_exists}/{len(all_samples)}")
        if wav_exists == 0:
            print("WARNING: No WAV files found! Voxtral WAVs may be on remote server only.")

    print(f"Selecting subset (per_emotion={per_emotion}, only_success={only_whitebox_success}):")
    subset = select_subset(all_samples, per_emotion, only_whitebox_success, seed)
    print(f"Selected {len(subset)} samples total")
    return subset


def load_surrogate_group(surrogate_key: str, per_emotion: int = 250, check_wav: bool = True) -> list[AdversarialSample]:
    """Load samples for a specific surrogate group."""
    result_dir = cfg.get_surrogate_dir(surrogate_key)
    info = cfg.surrogate_groups[surrogate_key]
    print(f"\n{'='*60}")
    print(f"Loading surrogate: {info['name']} ({surrogate_key})")
    print(f"{'='*60}")
    return load_and_select(
        result_dir=result_dir,
        per_emotion=per_emotion,
        check_wav=check_wav,
    )


if __name__ == "__main__":
    for key in cfg.surrogate_groups:
        samples = load_surrogate_group(key, per_emotion=5, check_wav=True)
        for s in samples[:3]:
            wav_status = "EXISTS" if s.adv_audio_path.exists() else "MISSING"
            print(f"  {s} | wav={wav_status}")
        print()
