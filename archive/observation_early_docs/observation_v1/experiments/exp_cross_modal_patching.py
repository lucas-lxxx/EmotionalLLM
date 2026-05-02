#!/usr/bin/env python3
"""
EXP-3: 跨模态 Activation Patching 扩大规模

扩大 Section 2.2 的跨模态 Patching 实验样本量（从 N=3 → N≥20）。
复用已有 activation_patching 框架，适配跨模态条件（PatchText vs PatchAudio）。

需要在服务器上运行（需要模型 + GPU）。

用法:
    CUDA_VISIBLE_DEVICES=X python exp_cross_modal_patching.py --config config.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
import yaml

AUDIO_TOKEN_INDEX = -200
IGNORE_INDEX = -100


@dataclass
class CrossModalSample:
    """跨模态实验样本"""
    sample_id: str
    audio_path: str
    audio_emotion: str       # 音频韵律情绪
    text_emotion: str        # 文本指令要求的情绪 (None = audio-only)
    condition: str            # "audio_only" / "conflict" / "consistent"
    prompt: str               # 完整 prompt 文本


@dataclass
class CrossModalPair:
    """跨模态 patching pair"""
    pair_id: str
    sample_base: CrossModalSample    # A: 被 patch 的样本
    sample_source: CrossModalSample  # B: 提供 activation 的样本
    patch_region: str                # "text" or "audio"
    target_label: str                # 期望 patch 后翻转到的标签


def build_cross_modal_samples(
    audio_root: str,
    emotions: List[str],
    base_prompt: str,
    system_prompt: str,
    max_per_condition: int = 20,
) -> List[CrossModalSample]:
    """
    构建跨模态实验样本。

    假设数据目录结构:
      audio_root/
        <emotion>/
          *.wav

    三种条件:
      1. audio_only: 中性 prompt + 情绪音频
      2. conflict:   要求情绪 T 的 prompt + 情绪 A 的音频 (T ≠ A)
      3. consistent: 要求情绪 T 的 prompt + 情绪 T 的音频

    注意：此函数需要根据实际数据集结构调整。
    """
    audio_root = Path(audio_root)
    samples = []

    # 收集所有可用音频
    audio_files = {}
    for emo in emotions:
        emo_dir = audio_root / emo
        if emo_dir.exists():
            wavs = sorted(emo_dir.glob("*.wav"))
            audio_files[emo] = wavs[:max_per_condition]

    if not audio_files:
        # 尝试 flat 结构 (text_id_emotion.wav)
        all_wavs = sorted(audio_root.glob("**/*.wav"))
        for wav in all_wavs:
            for emo in emotions:
                if emo in wav.stem:
                    if emo not in audio_files:
                        audio_files[emo] = []
                    if len(audio_files[emo]) < max_per_condition:
                        audio_files[emo].append(wav)

    sample_idx = 0

    for audio_emo, wavs in audio_files.items():
        for wav in wavs:
            # Condition 1: audio_only
            samples.append(CrossModalSample(
                sample_id=f"cm_{sample_idx:04d}",
                audio_path=str(wav),
                audio_emotion=audio_emo,
                text_emotion=None,
                condition="audio_only",
                prompt=base_prompt,
            ))
            sample_idx += 1

            # Condition 2: conflict (pick a different emotion)
            other_emos = [e for e in emotions if e != audio_emo and e != "neutral"]
            if other_emos:
                text_emo = other_emos[sample_idx % len(other_emos)]
                conflict_prompt = base_prompt.replace(
                    "What is the emotion",
                    f"The emotion is {text_emo}. What is the emotion"
                )
                samples.append(CrossModalSample(
                    sample_id=f"cm_{sample_idx:04d}",
                    audio_path=str(wav),
                    audio_emotion=audio_emo,
                    text_emotion=text_emo,
                    condition="conflict",
                    prompt=conflict_prompt,
                ))
                sample_idx += 1

            # Condition 3: consistent
            samples.append(CrossModalSample(
                sample_id=f"cm_{sample_idx:04d}",
                audio_path=str(wav),
                audio_emotion=audio_emo,
                text_emotion=audio_emo,
                condition="consistent",
                prompt=base_prompt.replace(
                    "What is the emotion",
                    f"The emotion is {audio_emo}. What is the emotion"
                ),
            ))
            sample_idx += 1

    return samples


def construct_cross_modal_pairs(
    samples: List[CrossModalSample],
    max_pairs: int = 20,
    seed: int = 42,
) -> Tuple[List[CrossModalPair], List[CrossModalPair]]:
    """
    构建 PatchText 和 PatchAudio pairs。

    PatchText: base=conflict, source=audio_only (同音频)
      → 替换 text region 后，预期从 text_emotion 翻转回 audio_emotion

    PatchAudio: base=conflict, source=另一个 audio_only (不同音频)
      → 替换 audio region 后，预期从 conflict 状态变化
    """
    rng = np.random.RandomState(seed)

    conflict_samples = [s for s in samples if s.condition == "conflict"]
    audio_only_samples = [s for s in samples if s.condition == "audio_only"]

    # Build audio_only lookup by audio_path
    ao_by_path = {}
    for s in audio_only_samples:
        ao_by_path[s.audio_path] = s

    # Build audio_only by emotion
    ao_by_emo = {}
    for s in audio_only_samples:
        if s.audio_emotion not in ao_by_emo:
            ao_by_emo[s.audio_emotion] = []
        ao_by_emo[s.audio_emotion].append(s)

    patch_text_pairs = []
    patch_audio_pairs = []

    for cs in conflict_samples:
        if len(patch_text_pairs) >= max_pairs and len(patch_audio_pairs) >= max_pairs:
            break

        # PatchText: same audio, no text emotion instruction
        if cs.audio_path in ao_by_path and len(patch_text_pairs) < max_pairs:
            ao = ao_by_path[cs.audio_path]
            patch_text_pairs.append(CrossModalPair(
                pair_id=f"pt_{len(patch_text_pairs):04d}",
                sample_base=cs,
                sample_source=ao,
                patch_region="text",
                target_label=cs.audio_emotion,  # expect revert to audio emotion
            ))

        # PatchAudio: same prompt, different audio
        if cs.text_emotion and len(patch_audio_pairs) < max_pairs:
            # Find an audio_only with different emotion
            other_ao = []
            for emo, ao_list in ao_by_emo.items():
                if emo != cs.audio_emotion:
                    other_ao.extend(ao_list)
            if other_ao:
                source = other_ao[rng.randint(0, len(other_ao))]
                patch_audio_pairs.append(CrossModalPair(
                    pair_id=f"pa_{len(patch_audio_pairs):04d}",
                    sample_base=cs,
                    sample_source=source,
                    patch_region="audio",
                    target_label=source.audio_emotion,
                ))

    return patch_text_pairs, patch_audio_pairs


def find_text_and_audio_spans(
    input_ids: torch.Tensor,
    hidden_len: int,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    定位 text 和 audio 的 token span。

    input_ids 中 AUDIO_TOKEN_INDEX 会被展开为多个 speech tokens。
    text region = 所有非 audio 的 token。
    audio region = 展开后的 speech token span。
    """
    audio_pos = (input_ids[0] == AUDIO_TOKEN_INDEX).nonzero(as_tuple=True)[0]
    if len(audio_pos) != 1:
        raise ValueError(f"Expected 1 audio token, got {len(audio_pos)}")

    audio_idx = int(audio_pos[0].item())
    input_len = input_ids.shape[1]
    speech_len = hidden_len - (input_len - 1)

    if speech_len <= 0:
        raise ValueError(f"Invalid speech_len: {speech_len}")

    audio_start = audio_idx
    audio_end = audio_idx + speech_len - 1

    # Text region: everything before audio_start and after audio_end
    # For simplicity, we define text_span as the post-audio tokens
    # (which include the prompt after audio)
    text_start = audio_end + 1
    text_end = hidden_len - 1

    return (audio_start, audio_end), (text_start, text_end)


def run_cross_modal_patching(
    pairs: List[CrossModalPair],
    patch_region: str,
    model,
    tokenizer,
    audio_extractor,
    opens2s_io,
    label_tokenizer,
    layers_to_patch: List[int],
    align_strategy: str = "truncate_to_min",
    patch_alpha: float = 1.0,
    cache_device: str = "cpu",
) -> Tuple[Dict[str, Any], List[Dict]]:
    """
    执行跨模态 Activation Patching。

    与 audio-internal patching 的关键区别:
    - patch_region="text": 替换 text token 区域的 hidden states
    - patch_region="audio": 替换 audio token 区域的 hidden states
    """
    if not pairs:
        return {"layer_indices": layers_to_patch, "n_pairs": 0}, []

    layer_count = len(layers_to_patch)
    flip_to_target = np.zeros(layer_count, dtype=np.float64)
    flip_from_base = np.zeros(layer_count, dtype=np.float64)
    delta_logit_target = np.zeros(layer_count, dtype=np.float64)

    label_token_ids = label_tokenizer.get_label_token_id_list()
    emotions = label_tokenizer.emotions
    layers = model.llm_model.model.layers
    device = str(next(model.parameters()).device)
    dtype = next(model.parameters()).dtype

    records = []

    for pair_idx, pair in enumerate(pairs):
        print(f"  Pair {pair_idx+1}/{len(pairs)}: {pair.pair_id}")

        # Build inputs
        wave_a, sr_a = torchaudio.load(pair.sample_base.audio_path)
        if sr_a != 16000:
            wave_a = torchaudio.functional.resample(wave_a, sr_a, 16000)
            sr_a = 16000

        wave_b, sr_b = torchaudio.load(pair.sample_source.audio_path)
        if sr_b != 16000:
            wave_b = torchaudio.functional.resample(wave_b, sr_b, 16000)
            sr_b = 16000

        inputs_a = opens2s_io.build_inputs(
            wave_a, sr_a, pair.sample_base.prompt, tokenizer,
            device=device, audio_extractor=audio_extractor,
            torch_extractor=None, differentiable=False,
            system_prompt=None, dtype=dtype,
        )
        inputs_b = opens2s_io.build_inputs(
            wave_b, sr_b, pair.sample_source.prompt, tokenizer,
            device=device, audio_extractor=audio_extractor,
            torch_extractor=None, differentiable=False,
            system_prompt=None, dtype=dtype,
        )

        # Add labels
        labels_a = inputs_a["input_ids"].clone()
        labels_a[labels_a == AUDIO_TOKEN_INDEX] = IGNORE_INDEX
        inputs_a["labels"] = labels_a

        labels_b = inputs_b["input_ids"].clone()
        labels_b[labels_b == AUDIO_TOKEN_INDEX] = IGNORE_INDEX
        inputs_b["labels"] = labels_b

        # Baseline forward A
        with torch.inference_mode():
            outputs_a = model(
                input_ids=inputs_a["input_ids"],
                attention_mask=inputs_a.get("attention_mask"),
                speech_values=inputs_a.get("speech_values"),
                speech_mask=inputs_a.get("speech_mask"),
                labels=inputs_a.get("labels"),
                token_types=None, speech_units=None,
                speech_units_mask=None, spk_embs=None,
                return_dict=True,
            )

        logits_base = outputs_a.logits[0]
        input_len_a = inputs_a["input_ids"].shape[1]
        logits_len = logits_base.shape[0]
        delta = logits_len - input_len_a
        readout_pos = (input_len_a - 1) + delta

        logits_base_pos = logits_base[readout_pos]
        label_logits_base = logits_base_pos[label_token_ids]
        pred_base = emotions[int(torch.argmax(label_logits_base).item())]

        target_idx = emotions.index(pair.target_label)
        target_token_id = label_token_ids[target_idx]
        base_logit_target = float(logits_base_pos[target_token_id].item())

        # Determine spans from A's hidden dimension
        hidden_len_a = logits_len
        audio_span_a, text_span_a = find_text_and_audio_spans(
            inputs_a["input_ids"], hidden_len_a
        )

        # Cache B's activations for the target region
        # We need a full forward of B to get layer activations
        cache_b = {}

        def make_cache_hook(layer_offset, region_start, region_end):
            def hook(module, _inp, output):
                hidden = output[0] if isinstance(output, tuple) else output
                act = hidden[:, region_start:region_end+1, :].detach()
                if cache_device == "cpu":
                    act = act.to("cpu")
                cache_b[layer_offset] = act
                return output
            return hook

        # Determine B's spans
        with torch.inference_mode():
            outputs_b = model(
                input_ids=inputs_b["input_ids"],
                attention_mask=inputs_b.get("attention_mask"),
                speech_values=inputs_b.get("speech_values"),
                speech_mask=inputs_b.get("speech_mask"),
                labels=inputs_b.get("labels"),
                token_types=None, speech_units=None,
                speech_units_mask=None, spk_embs=None,
                return_dict=True,
                output_hidden_states=True,
            )

        hidden_len_b = outputs_b.hidden_states[0].shape[1]
        audio_span_b, text_span_b = find_text_and_audio_spans(
            inputs_b["input_ids"], hidden_len_b
        )

        if patch_region == "text":
            region_b_start, region_b_end = text_span_b
        else:
            region_b_start, region_b_end = audio_span_b

        # Cache B activations per layer
        for layer_offset, layer_idx in enumerate(layers_to_patch):
            hs = outputs_b.hidden_states[layer_idx + 1]  # +1 for embedding offset
            act = hs[:, region_b_start:region_b_end+1, :].detach()
            if cache_device == "cpu":
                act = act.to("cpu")
            cache_b[layer_offset] = act

        # Clean up B outputs
        del outputs_b
        torch.cuda.empty_cache()

        # Patching loop
        if patch_region == "text":
            region_a_start, region_a_end = text_span_a
        else:
            region_a_start, region_a_end = audio_span_a

        for layer_offset, layer_idx in enumerate(layers_to_patch):
            cached = cache_b[layer_offset]

            def patch_hook(module, _inp, output,
                           _cached=cached, _start=region_a_start, _end=region_a_end):
                hidden = output[0] if isinstance(output, tuple) else output
                a_slice = hidden[:, _start:_end+1, :]

                b_slice = _cached
                if b_slice.device != hidden.device:
                    b_slice = b_slice.to(hidden.device)

                # Align lengths
                len_a = a_slice.shape[1]
                len_b = b_slice.shape[1]
                l = min(len_a, len_b)

                patched = hidden.clone()
                if l > 0:
                    patched[:, _start:_start+l, :] = b_slice[:, :l, :]

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
                    token_types=None, speech_units=None,
                    speech_units_mask=None, spk_embs=None,
                    return_dict=True,
                )

            handle.remove()

            logits_patch = outputs_patch.logits[0]
            logits_patch_pos = logits_patch[readout_pos]
            label_logits_patch = logits_patch_pos[label_token_ids]
            pred_patch = emotions[int(torch.argmax(label_logits_patch).item())]
            logit_target_patch = float(logits_patch_pos[target_token_id].item())

            if pred_patch == pair.target_label:
                flip_to_target[layer_offset] += 1
            if pred_patch != pred_base:
                flip_from_base[layer_offset] += 1
            delta_logit_target[layer_offset] += (logit_target_patch - base_logit_target)

            records.append({
                "pair_id": pair.pair_id,
                "patch_region": patch_region,
                "layer": layer_idx,
                "pred_base": pred_base,
                "pred_patch": pred_patch,
                "target_label": pair.target_label,
                "flip_to_target": int(pred_patch == pair.target_label),
                "flip_from_base": int(pred_patch != pred_base),
                "delta_logit_target": logit_target_patch - base_logit_target,
                "base_audio_emotion": pair.sample_base.audio_emotion,
                "base_text_emotion": pair.sample_base.text_emotion,
            })

            del outputs_patch
            torch.cuda.empty_cache()

        # Clean up caches
        del cache_b
        torch.cuda.empty_cache()

    n_pairs = len(pairs)
    metrics = {
        "layer_indices": layers_to_patch,
        "flip_to_target_rate": (flip_to_target / max(n_pairs, 1)).tolist(),
        "flip_from_base_rate": (flip_from_base / max(n_pairs, 1)).tolist(),
        "delta_logit_target_mean": (delta_logit_target / max(n_pairs, 1)).tolist(),
        "n_pairs": n_pairs,
        "patch_region": patch_region,
    }

    return metrics, records


def main():
    parser = argparse.ArgumentParser(description="EXP-3: Cross-Modal Patching Scale-up")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.device:
        config["model"]["device"] = args.device
    if args.max_pairs:
        config["cross_modal_patching"]["max_pairs"] = args.max_pairs

    output_dir = Path(config["paths"]["opus_results"]) / "cross_modal_patching"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EXP-3: 跨模态 Activation Patching 扩大规模")
    print("=" * 60)

    # Load model
    print("\n[1] 加载模型...")
    opens2s_io_path = Path(config["paths"]["opens2s_io_path"])
    import importlib.util
    spec = importlib.util.spec_from_file_location("opens2s_io", opens2s_io_path)
    opens2s_io = importlib.util.module_from_spec(spec)
    if str(opens2s_io_path.parent) not in sys.path:
        sys.path.insert(0, str(opens2s_io_path.parent))
    spec.loader.exec_module(opens2s_io)

    model, tokenizer, audio_extractor, _ = opens2s_io.load_opens2s(
        Path(config["model"]["model_path"]),
        config["model"]["device"],
        Path(config["paths"]["opens2s_root"]),
    )
    print(f"    模型加载完成, 设备: {config['model']['device']}")

    # Initialize label tokenizer
    dataset_module = Path(config["data"]["dataset_module"])
    dataset_src = dataset_module.parent.parent
    if str(dataset_src) not in sys.path:
        sys.path.insert(0, str(dataset_src))

    # Use label tokenizer from existing code
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "code" / "activation_patching" / "src"))
    from label_tokenizer import LabelTokenizer
    label_tokenizer = LabelTokenizer(tokenizer, emotions=config["data"]["emotions"])
    label_tokenizer.print_summary()

    # Build samples
    print("\n[2] 构建跨模态实验样本...")
    cm_config = config["cross_modal_patching"]
    samples = build_cross_modal_samples(
        audio_root=config["data"]["audio_root"],
        emotions=config["data"]["emotions"],
        base_prompt=config["prompt"],
        system_prompt=config.get("system_prompt"),
        max_per_condition=cm_config["max_pairs"],
    )
    print(f"    总样本数: {len(samples)}")
    for cond in ["audio_only", "conflict", "consistent"]:
        n = sum(1 for s in samples if s.condition == cond)
        print(f"    {cond}: {n}")

    # Build pairs
    print("\n[3] 构建 patching pairs...")
    patch_text_pairs, patch_audio_pairs = construct_cross_modal_pairs(
        samples,
        max_pairs=cm_config["max_pairs"],
        seed=cm_config["pairing_seed"],
    )
    print(f"    PatchText pairs: {len(patch_text_pairs)}")
    print(f"    PatchAudio pairs: {len(patch_audio_pairs)}")

    # Determine layers
    layers_to_patch = cm_config.get("layers_to_patch", [])
    if not layers_to_patch:
        n_layers = model.llm_model.config.num_hidden_layers
        layers_to_patch = list(range(n_layers))

    # Save config
    with open(output_dir / "cross_modal_config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    # Run PatchText
    print("\n[4] 运行 PatchText...")
    metrics_text, records_text = run_cross_modal_patching(
        pairs=patch_text_pairs,
        patch_region="text",
        model=model,
        tokenizer=tokenizer,
        audio_extractor=audio_extractor,
        opens2s_io=opens2s_io,
        label_tokenizer=label_tokenizer,
        layers_to_patch=layers_to_patch,
        align_strategy=cm_config["align_strategy"],
        patch_alpha=cm_config["patch_alpha"],
        cache_device=cm_config["cache_device"],
    )

    # Save PatchText results
    with open(output_dir / "cross_modal_metrics_text.json", "w") as f:
        json.dump(metrics_text, f, indent=2)

    if records_text:
        with open(output_dir / "cross_modal_records_text.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=records_text[0].keys())
            writer.writeheader()
            writer.writerows(records_text)

    # Run PatchAudio
    print("\n[5] 运行 PatchAudio...")
    metrics_audio, records_audio = run_cross_modal_patching(
        pairs=patch_audio_pairs,
        patch_region="audio",
        model=model,
        tokenizer=tokenizer,
        audio_extractor=audio_extractor,
        opens2s_io=opens2s_io,
        label_tokenizer=label_tokenizer,
        layers_to_patch=layers_to_patch,
        align_strategy=cm_config["align_strategy"],
        patch_alpha=cm_config["patch_alpha"],
        cache_device=cm_config["cache_device"],
    )

    # Save PatchAudio results
    with open(output_dir / "cross_modal_metrics_audio.json", "w") as f:
        json.dump(metrics_audio, f, indent=2)

    if records_audio:
        with open(output_dir / "cross_modal_records_audio.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=records_audio[0].keys())
            writer.writeheader()
            writer.writerows(records_audio)

    print("\n" + "=" * 60)
    print("EXP-3 完成")
    print(f"PatchText: {metrics_text['n_pairs']} pairs")
    print(f"PatchAudio: {metrics_audio['n_pairs']} pairs")
    print(f"结果保存至: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
