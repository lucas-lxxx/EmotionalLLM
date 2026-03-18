"""
Observation 3: Emotion Misperception → Downstream Reasoning Impact

对同一段音频，用 Aligned（正确情绪）vs Conflict（错误情绪）两种 Prompt 做推理，
对比 Voxtral 模型回复差异。

Usage:
    # Dry-run（仅打印 Prompt，不推理）
    python run_inference.py --config config_demo.json --dry-run

    # 实际推理
    python run_inference.py --config config_demo.json --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, VoxtralForConditionalGeneration

# ── Prompt Templates ──────────────────────────────────────────────────────────

EMOTION_DESCRIPTORS: dict[str, str] = {
    "happy": "happy and cheerful",
    "angry": "angry and frustrated",
    "sad": "sad and distressed",
    "neutral": "calm and composed",
    "surprise": "surprised and taken aback",
}

PROMPT_TEMPLATE = (
    "The speaker seems to be feeling {emotion_desc}. "
    "Listen to what they say and respond appropriately."
)


def build_prompt(emotion: str) -> str:
    """根据情绪标签生成 Prompt 文本。"""
    desc = EMOTION_DESCRIPTORS.get(emotion)
    if desc is None:
        raise ValueError(
            f"Unknown emotion '{emotion}'. "
            f"Supported: {list(EMOTION_DESCRIPTORS.keys())}"
        )
    return PROMPT_TEMPLATE.format(emotion_desc=desc)


def build_conversation(audio_path: str, emotion: str) -> list[dict]:
    """构建 Voxtral chat conversation 格式。"""
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "path": audio_path},
                {"type": "text", "text": build_prompt(emotion)},
            ],
        }
    ]


# ── Model Loading ─────────────────────────────────────────────────────────────


def load_model(
    model_path: str, device: str = "cuda:0"
) -> tuple:
    """加载 Voxtral 模型和 processor。"""
    logging.info(f"Loading model from {model_path} ...")
    t0 = time.time()

    processor = AutoProcessor.from_pretrained(model_path)
    model = VoxtralForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        device_map=device,
    )
    model.eval()

    elapsed = time.time() - t0
    logging.info(f"Model loaded in {elapsed:.1f}s.")
    return model, processor


# ── Inference ─────────────────────────────────────────────────────────────────


def run_single(
    model,
    processor,
    conversation: list[dict],
    device: str,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    """对一条 conversation 做推理，返回解码后的文本回复。"""
    inputs = processor.apply_chat_template(conversation)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    inputs = inputs.to(device, dtype=dtype)

    do_sample = temperature > 1e-4
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        use_cache=True,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.95

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    new_tokens = outputs[:, inputs.input_ids.shape[1] :]
    decoded = processor.batch_decode(new_tokens, skip_special_tokens=True)
    return decoded[0].strip()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Observation 3: Aligned vs Conflict emotion inference"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config JSON"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print prompts and audio paths, skip actual inference",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device (default: cuda:0)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # ── Load config ───────────────────────────────────────────────────────
    with open(args.config, "r", encoding="utf-8-sig") as f:
        config = json.load(f)

    model_path = config["model_path"]
    audio_dir = Path(config["audio_dir"])
    output_path = Path(config["output_path"])
    samples = config["samples"]
    temperature = config.get("temperature", 0.2)
    max_new_tokens = config.get("max_new_tokens", 512)

    logging.info(
        f"Config loaded: {len(samples)} samples, audio_dir={audio_dir}, "
        f"temperature={temperature}, max_new_tokens={max_new_tokens}"
    )

    # ── Build records ─────────────────────────────────────────────────────
    results: list[dict] = []
    for sample in samples:
        audio_path = str(audio_dir / sample["audio_file"])
        true_emo = sample["true_emotion"]
        misled_emo = sample["misled_emotion"]

        results.append(
            {
                "audio_file": sample["audio_file"],
                "true_emotion": true_emo,
                "misled_emotion": misled_emo,
                "aligned_prompt": build_prompt(true_emo),
                "conflict_prompt": build_prompt(misled_emo),
                "aligned_response": None,
                "conflict_response": None,
            }
        )

    # ── Dry-run mode ──────────────────────────────────────────────────────
    if args.dry_run:
        for i, rec in enumerate(results):
            audio_path = str(audio_dir / rec["audio_file"])
            logging.info(f"[{i + 1}/{len(results)}] DRY RUN")
            logging.info(f"  Audio:    {audio_path}")
            logging.info(f"  True emo: {rec['true_emotion']}")
            logging.info(f"  Misled:   {rec['misled_emotion']}")
            logging.info(f"  Aligned prompt:  {rec['aligned_prompt']}")
            logging.info(f"  Conflict prompt: {rec['conflict_prompt']}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logging.info(f"Dry-run results saved to {output_path}")
        return

    # ── Load model ────────────────────────────────────────────────────────
    model, processor = load_model(model_path, args.device)

    # ── Run inference ─────────────────────────────────────────────────────
    for i, rec in enumerate(results):
        audio_path = str(audio_dir / rec["audio_file"])
        true_emo = rec["true_emotion"]
        misled_emo = rec["misled_emotion"]

        logging.info(
            f"[{i + 1}/{len(results)}] {rec['audio_file']}  "
            f"(true={true_emo}, misled={misled_emo})"
        )

        conv_aligned = build_conversation(audio_path, true_emo)
        conv_conflict = build_conversation(audio_path, misled_emo)

        t0 = time.time()
        aligned_resp = run_single(
            model, processor, conv_aligned, args.device,
            max_new_tokens=max_new_tokens, temperature=temperature,
        )
        conflict_resp = run_single(
            model, processor, conv_conflict, args.device,
            max_new_tokens=max_new_tokens, temperature=temperature,
        )
        elapsed = time.time() - t0

        rec["aligned_response"] = aligned_resp
        rec["conflict_response"] = conflict_resp

        logging.info(f"  Aligned  ({elapsed / 2:.1f}s): {aligned_resp[:100]}...")
        logging.info(f"  Conflict ({elapsed / 2:.1f}s): {conflict_resp[:100]}...")

    # ── Save results ──────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logging.info(f"Results saved to {output_path} ({len(results)} samples)")


if __name__ == "__main__":
    main()
