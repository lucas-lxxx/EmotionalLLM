"""Pair construction for activation patching."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Tuple, Iterable
import random


@dataclass
class PairSpec:
    pair_id: str
    pair_type: str  # "prosody" or "semantic"
    sample_a: any
    sample_b: any


def _sample_pairs(pairs: List[PairSpec], max_pairs: int | None, seed: int) -> List[PairSpec]:
    if max_pairs is None or max_pairs <= 0 or len(pairs) <= max_pairs:
        return pairs
    rng = random.Random(seed)
    pairs_copy = pairs[:]
    rng.shuffle(pairs_copy)
    return pairs_copy[:max_pairs]


def _sample_info(sample) -> Dict:
    return {
        "sample_id": sample.sample_id,
        "text_id": sample.text_id,
        "semantic_emotion": sample.semantic_emotion,
        "prosody_emotion": sample.prosody_emotion,
        "audio_path": sample.audio_path,
    }


def build_pair_report(
    pair_type: str,
    total_candidates: int,
    selected_pairs: int,
    per_group_counts: Dict[str, int],
    skipped_groups: Dict[str, int],
) -> Dict:
    return {
        "pair_type": pair_type,
        "total_candidates": total_candidates,
        "selected_pairs": selected_pairs,
        "per_group_counts": per_group_counts,
        "skipped_groups": skipped_groups,
    }


def construct_prosody_pairs(
    samples: List[any],
    max_pairs: int | None,
    seed: int,
) -> Tuple[List[PairSpec], Dict]:
    """Same text_id, different prosody emotion."""
    groups: Dict[str, List] = {}
    for s in samples:
        groups.setdefault(s.text_id, []).append(s)

    pairs: List[PairSpec] = []
    per_group_counts: Dict[str, int] = {}
    skipped_groups: Dict[str, int] = {}

    for text_id, group in groups.items():
        if len(group) < 2:
            skipped_groups[text_id] = len(group)
            continue
        # Build combinations within the group
        group_pairs = []
        for a, b in combinations(group, 2):
            if a.prosody_emotion == b.prosody_emotion:
                continue
            if a.semantic_emotion != b.semantic_emotion:
                # Should not happen for same text_id; skip to be safe.
                continue
            pair_id = f"prosody_{text_id}_{a.prosody_emotion}_to_{b.prosody_emotion}"
            group_pairs.append(PairSpec(pair_id, "prosody", a, b))
        per_group_counts[text_id] = len(group_pairs)
        pairs.extend(group_pairs)

    total_candidates = len(pairs)
    pairs = _sample_pairs(pairs, max_pairs, seed)
    report = build_pair_report(
        "prosody",
        total_candidates=total_candidates,
        selected_pairs=len(pairs),
        per_group_counts=per_group_counts,
        skipped_groups=skipped_groups,
    )
    return pairs, report


def construct_semantic_pairs(
    samples: List[any],
    max_pairs: int | None,
    seed: int,
) -> Tuple[List[PairSpec], Dict]:
    """Same prosody emotion, different semantic emotion and text_id."""
    groups: Dict[str, List] = {}
    for s in samples:
        groups.setdefault(s.prosody_emotion, []).append(s)

    pairs: List[PairSpec] = []
    per_group_counts: Dict[str, int] = {}
    skipped_groups: Dict[str, int] = {}

    for prosody, group in groups.items():
        if len(group) < 2:
            skipped_groups[prosody] = len(group)
            continue
        group_pairs = []
        for a, b in combinations(group, 2):
            if a.text_id == b.text_id:
                continue
            if a.semantic_emotion == b.semantic_emotion:
                continue
            pair_id = f"semantic_{prosody}_{a.text_id}_to_{b.text_id}"
            group_pairs.append(PairSpec(pair_id, "semantic", a, b))
        per_group_counts[prosody] = len(group_pairs)
        pairs.extend(group_pairs)

    total_candidates = len(pairs)
    pairs = _sample_pairs(pairs, max_pairs, seed)
    report = build_pair_report(
        "semantic",
        total_candidates=total_candidates,
        selected_pairs=len(pairs),
        per_group_counts=per_group_counts,
        skipped_groups=skipped_groups,
    )
    return pairs, report


def pair_to_dict(pair: PairSpec) -> Dict:
    return {
        "pair_id": pair.pair_id,
        "pair_type": pair.pair_type,
        "sample_a": _sample_info(pair.sample_a),
        "sample_b": _sample_info(pair.sample_b),
    }
