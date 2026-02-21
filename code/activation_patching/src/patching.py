"""Core activation patching logic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import importlib.util
import json

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from label_tokenizer import LabelTokenizer
from hooks import HookManager

AUDIO_TOKEN_INDEX = -200
IGNORE_INDEX = -100


@dataclass
class PatchMetrics:
    layer_indices: List[int]
    flip_to_target: np.ndarray
    flip_from_base: np.ndarray
    delta_logit_target: np.ndarray
    eligible_count: np.ndarray
    n_pairs: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_indices": self.layer_indices,
            "flip_to_target_rate": (self.flip_to_target / max(self.n_pairs, 1)).tolist(),
            "flip_from_base_rate": (self.flip_from_base / max(self.n_pairs, 1)).tolist(),
            "delta_logit_target_mean": (self.delta_logit_target / max(self.n_pairs, 1)).tolist(),
            "eligible_count": self.eligible_count.tolist(),
            "n_pairs": self.n_pairs,
        }


def load_opens2s_io(opens2s_io_path: Path):
    opens2s_io_path = Path(opens2s_io_path)
    spec = importlib.util.spec_from_file_location("opens2s_io", opens2s_io_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load opens2s_io from {opens2s_io_path}")
    module = importlib.util.module_from_spec(spec)
    # Ensure its directory is on sys.path so 'config' import works
    import sys

    if str(opens2s_io_path.parent) not in sys.path:
        sys.path.insert(0, str(opens2s_io_path.parent))
    spec.loader.exec_module(module)
    return module


def load_audio(audio_path: str, device: str) -> Tuple[torch.Tensor, int]:
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000
    return waveform.to(device), sr


def add_labels(inputs: Dict[str, torch.Tensor]) -> None:
    labels = inputs["input_ids"].clone()
    labels[labels == AUDIO_TOKEN_INDEX] = IGNORE_INDEX
    inputs["labels"] = labels


def find_audio_token_index(input_ids: torch.Tensor) -> int:
    positions = (input_ids[0] == AUDIO_TOKEN_INDEX).nonzero(as_tuple=True)[0]
    if len(positions) != 1:
        raise ValueError(f"Expected exactly one AUDIO_TOKEN_INDEX, found {len(positions)}")
    return int(positions[0].item())


def compute_audio_span(input_len: int, audio_token_idx: int, hidden_len: int) -> Tuple[int, int, int]:
    # hidden_len = (input_len - 1) + speech_len
    speech_len = hidden_len - (input_len - 1)
    if speech_len <= 0:
        raise ValueError("Invalid speech_len computed for audio span.")
    start = audio_token_idx
    end = start + speech_len - 1
    if end >= hidden_len:
        raise ValueError("Audio span exceeds hidden length.")
    return start, end, speech_len


def compute_readout_pos(input_len: int, logits_len: int) -> int:
    # Adjust for speech token expansion
    delta = logits_len - input_len
    pos = (input_len - 1) + delta
    if pos < 0 or pos >= logits_len:
        raise ValueError(f"readout_pos out of range: {pos} vs {logits_len}")
    return pos


def restricted_pred(logits: torch.Tensor, label_token_ids: List[int], emotions: List[str]) -> str:
    label_logits = logits[label_token_ids]
    pred_idx = int(torch.argmax(label_logits).item())
    return emotions[pred_idx]


def align_and_mix(
    a_slice: torch.Tensor,
    b_slice: torch.Tensor,
    patch_alpha: float,
    align_strategy: str,
) -> torch.Tensor:
    len_a = a_slice.shape[1]
    len_b = b_slice.shape[1]

    if align_strategy == "truncate_to_min":
        l = min(len_a, len_b)
        out = a_slice.clone()
        if l > 0:
            mix_b = b_slice[:, :l, :]
            if patch_alpha < 1.0:
                mix_b = (1 - patch_alpha) * out[:, :l, :] + patch_alpha * mix_b
            out[:, :l, :] = mix_b
        return out

    if align_strategy == "resample_linear":
        if len_b == 0 or len_a == 0:
            return a_slice.clone()
        b_t = b_slice.permute(0, 2, 1)  # [B, C, L]
        b_res = F.interpolate(b_t, size=len_a, mode="linear", align_corners=False)
        b_res = b_res.permute(0, 2, 1)
        if patch_alpha < 1.0:
            b_res = (1 - patch_alpha) * a_slice + patch_alpha * b_res
        return b_res

    raise ValueError(f"Unknown align_strategy: {align_strategy}")


def cache_layer_activations(
    model,
    layers: List,
    inputs: Dict[str, torch.Tensor],
    input_len: int,
    audio_token_idx: int,
    cache_device: str,
) -> Tuple[Dict[int, torch.Tensor], Tuple[int, int, int]]:
    cache: Dict[int, torch.Tensor] = {}
    audio_span = {"start": None, "end": None, "speech_len": None}

    def hook_factory(layer_idx: int):
        def hook(module, _inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden_len = hidden.shape[1]
            if audio_span["start"] is None:
                start, end, speech_len = compute_audio_span(input_len, audio_token_idx, hidden_len)
                audio_span["start"] = start
                audio_span["end"] = end
                audio_span["speech_len"] = speech_len
            start = audio_span["start"]
            end = audio_span["end"]
            act = hidden[:, start : end + 1, :].detach()
            if cache_device == "cpu":
                act = act.to("cpu")
            cache[layer_idx] = act
            return output
        return hook

    manager = HookManager()
    for idx, layer in enumerate(layers):
        handle = layer.register_forward_hook(hook_factory(idx))
        manager.add(handle)

    with torch.inference_mode():
        _ = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            speech_values=inputs.get("speech_values"),
            speech_mask=inputs.get("speech_mask"),
            labels=inputs.get("labels"),
            token_types=None,
            speech_units=None,
            speech_units_mask=None,
            spk_embs=None,
            return_dict=True,
        )

    manager.remove_all()
    if audio_span["start"] is None:
        raise RuntimeError("Failed to determine audio span during caching.")
    return cache, (audio_span["start"], audio_span["end"], audio_span["speech_len"])


def run_activation_patching(
    pairs: List,
    pair_type: str,
    model,
    tokenizer,
    audio_extractor,
    opens2s_io,
    label_tokenizer: LabelTokenizer,
    prompt: str,
    system_prompt: str | None,
    layers_to_patch: List[int],
    align_strategy: str,
    patch_alpha: float,
    cache_device: str,
    save_records: bool = False,
) -> Tuple[PatchMetrics, List[Dict[str, Any]]]:
    if not pairs:
        layer_count = len(layers_to_patch)
        metrics = PatchMetrics(
            layers_to_patch,
            np.zeros(layer_count),
            np.zeros(layer_count),
            np.zeros(layer_count),
            np.zeros(layer_count, dtype=np.int64),
            0,
        )
        return metrics, []

    layer_count = len(layers_to_patch)
    flip_to_target = np.zeros(layer_count, dtype=np.float64)
    flip_from_base = np.zeros(layer_count, dtype=np.float64)
    delta_logit_target = np.zeros(layer_count, dtype=np.float64)
    eligible_count = np.zeros(layer_count, dtype=np.int64)

    label_token_ids = label_tokenizer.get_label_token_id_list()
    emotions = label_tokenizer.emotions

    layers = model.llm_model.model.layers

    records: List[Dict[str, Any]] = []

    for pair in pairs:
        wave_a, sr_a = load_audio(pair.sample_a.audio_path, device="cpu")
        wave_b, sr_b = load_audio(pair.sample_b.audio_path, device="cpu")

        inputs_a = opens2s_io.build_inputs(
            wave_a,
            sr_a,
            prompt,
            tokenizer,
            device=str(next(model.parameters()).device),
            audio_extractor=audio_extractor,
            torch_extractor=None,
            differentiable=False,
            system_prompt=system_prompt,
            dtype=next(model.parameters()).dtype,
        )
        inputs_b = opens2s_io.build_inputs(
            wave_b,
            sr_b,
            prompt,
            tokenizer,
            device=str(next(model.parameters()).device),
            audio_extractor=audio_extractor,
            torch_extractor=None,
            differentiable=False,
            system_prompt=system_prompt,
            dtype=next(model.parameters()).dtype,
        )

        add_labels(inputs_a)
        add_labels(inputs_b)

        input_len_a = inputs_a["input_ids"].shape[1]
        input_len_b = inputs_b["input_ids"].shape[1]
        audio_token_idx_a = find_audio_token_index(inputs_a["input_ids"])
        audio_token_idx_b = find_audio_token_index(inputs_b["input_ids"])

        # Baseline forward on A
        with torch.inference_mode():
            outputs_a = model(
                input_ids=inputs_a["input_ids"],
                attention_mask=inputs_a.get("attention_mask"),
                speech_values=inputs_a.get("speech_values"),
                speech_mask=inputs_a.get("speech_mask"),
                labels=inputs_a.get("labels"),
                token_types=None,
                speech_units=None,
                speech_units_mask=None,
                spk_embs=None,
                return_dict=True,
            )

        logits_base = outputs_a.logits[0]
        readout_pos = compute_readout_pos(input_len_a, logits_base.shape[0])
        logits_base_pos = logits_base[readout_pos]
        pred_base = restricted_pred(logits_base_pos, label_token_ids, emotions)

        if pair_type == "prosody":
            target_label = pair.sample_b.prosody_emotion
        else:
            target_label = pair.sample_b.semantic_emotion
        target_token_id = label_tokenizer.get_label_token_ids()[target_label]
        base_logit_target = float(logits_base_pos[target_token_id].item())

        # Cache B activations
        cache_b, _audio_span_b = cache_layer_activations(
            model,
            [layers[i] for i in layers_to_patch],
            inputs_b,
            input_len_b,
            audio_token_idx_b,
            cache_device=cache_device,
        )

        for layer_offset, layer_idx in enumerate(layers_to_patch):
            cached = cache_b[layer_offset]

            def patch_hook(module, _inp, output):
                hidden = output[0] if isinstance(output, tuple) else output
                hidden_len = hidden.shape[1]
                start_a, end_a, _ = compute_audio_span(input_len_a, audio_token_idx_a, hidden_len)
                a_slice = hidden[:, start_a : end_a + 1, :]
                b_slice = cached
                if b_slice.dim() == 2:
                    b_slice = b_slice.unsqueeze(0)
                if b_slice.device != hidden.device:
                    b_slice = b_slice.to(hidden.device)

                patched_slice = align_and_mix(a_slice, b_slice, patch_alpha, align_strategy)
                patched = hidden.clone()
                patched[:, start_a : end_a + 1, :] = patched_slice

                if isinstance(output, tuple):
                    return (patched,) + output[1:]
                return patched

            handle = layers[layer_idx].register_forward_hook(patch_hook)

            with torch.inference_mode():
                outputs_patch = model(
                    input_ids=inputs_a["input_ids"],
                    attention_mask=inputs_a.get("attention_mask"),
                    speech_values=inputs_a.get("speech_values"),
                    speech_mask=inputs_a.get("speech_mask"),
                    labels=inputs_a.get("labels"),
                    token_types=None,
                    speech_units=None,
                    speech_units_mask=None,
                    spk_embs=None,
                    return_dict=True,
                )

            handle.remove()

            logits_patch = outputs_patch.logits[0]
            readout_pos_patch = compute_readout_pos(input_len_a, logits_patch.shape[0])
            logits_patch_pos = logits_patch[readout_pos_patch]
            pred_patch = restricted_pred(logits_patch_pos, label_token_ids, emotions)
            logit_target_patch = float(logits_patch_pos[target_token_id].item())

            if pred_patch == target_label:
                flip_to_target[layer_offset] += 1
            if pred_patch != pred_base:
                flip_from_base[layer_offset] += 1
            if pred_base != target_label:
                eligible_count[layer_offset] += 1
            delta_logit_target[layer_offset] += (logit_target_patch - base_logit_target)

            if save_records:
                records.append(
                    {
                        "pair_id": pair.pair_id,
                        "pair_type": pair_type,
                        "layer": layer_idx,
                        "pred_base": pred_base,
                        "pred_patch": pred_patch,
                        "target_label": target_label,
                        "flip_to_target": int(pred_patch == target_label),
                        "flip_from_base": int(pred_patch != pred_base),
                        "delta_logit_target": logit_target_patch - base_logit_target,
                        "eligible": int(pred_base != target_label),
                    }
                )

    metrics = PatchMetrics(
        layer_indices=layers_to_patch,
        flip_to_target=flip_to_target,
        flip_from_base=flip_from_base,
        delta_logit_target=delta_logit_target,
        eligible_count=eligible_count,
        n_pairs=len(pairs),
    )
    return metrics, records
