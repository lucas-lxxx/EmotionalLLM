from __future__ import annotations

import math
from typing import Any

import torch


_semantic_model = None


def _get_semantic_model(model_name: str):
    global _semantic_model
    if _semantic_model is None:
        from sentence_transformers import SentenceTransformer
        _semantic_model = SentenceTransformer(model_name, device="cpu")
    return _semantic_model


def compute_semantic_similarity(
    text_a: str, text_b: str, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
) -> float:
    """Compute cosine similarity between two texts."""
    if not text_a.strip() or not text_b.strip():
        return 0.0
    model = _get_semantic_model(model_name)
    embeddings = model.encode([text_a, text_b], normalize_embeddings=True)
    return float(embeddings[0] @ embeddings[1])


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def compute_wer(ref: str, hyp: str) -> float:
    # Methodology §8.2: WER for semantic consistency.
    try:
        import jiwer

        return float(jiwer.wer(ref, hyp))
    except Exception:
        ref_words = normalize_text(ref).split()
        hyp_words = normalize_text(hyp).split()
        if not ref_words:
            return 0.0 if not hyp_words else 1.0
        return _edit_distance(ref_words, hyp_words) / float(len(ref_words))


def _edit_distance(a: list[str], b: list[str]) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1]


def signal_metrics(waveform_adv: torch.Tensor, waveform_clean: torch.Tensor) -> dict[str, float]:
    # Methodology §8.2: L_inf / L2 / SNR.
    delta = waveform_adv - waveform_clean
    linf = float(delta.abs().max().item())
    l2 = float(delta.norm(p=2).item())
    denom = float(delta.norm(p=2).item()) + 1e-12
    snr = 20.0 * math.log10(float(waveform_clean.norm(p=2).item()) / denom)
    return {"delta_linf": linf, "delta_l2": l2, "snr_db": snr}


def aggregate_results(
    per_sample: list[dict[str, Any]],
    wer_thresholds: tuple[float, ...],
    semantic_threshold: float = 0.8,
) -> dict[str, Any]:
    # Methodology §8.3: aggregate success and perturbation stats.
    if not per_sample:
        return {}
    total = len(per_sample)
    success_emo = sum(1 for s in per_sample if s.get("success_emo", False))
    stats = {
        "num_samples": total,
        "emo_success_rate": success_emo / float(total),
    }
    for thr in wer_thresholds:
        key = f"wer_le_{thr}"
        stats[key] = sum(1 for s in per_sample if s.get("wer", 1.0) <= thr) / float(total)
        key_joint = f"joint_success_le_{thr}"
        stats[key_joint] = sum(
            1 for s in per_sample if s.get("success_emo", False) and s.get("wer", 1.0) <= thr
        ) / float(total)

    # Semantic similarity metrics
    semantic_preserved = sum(
        1 for s in per_sample if s.get("semantic_sim", 0.0) >= semantic_threshold
    )
    stats["semantic_preserve_rate"] = semantic_preserved / float(total)
    stats["joint_success_semantic"] = sum(
        1 for s in per_sample
        if s.get("success_emo", False) and s.get("semantic_sim", 0.0) >= semantic_threshold
    ) / float(total)

    return stats
