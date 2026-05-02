"""Quick smoke test: load MERaLiON model, run a single forward pass, and decode."""
import sys
import torch

from meralion_io import load_meralion, decode_text, build_inputs, forward_logits
from config import cfg


def main():
    device = "cuda:0"
    print("Loading MERaLiON-2-3B...")
    model, processor, torch_extractor = load_meralion(cfg.model_path, device)
    print(f"Model loaded. dtype={next(model.parameters()).dtype}")
    print(f"speech_token: {processor.speech_token!r} (id={processor.speech_token_index})")
    print(f"fixed_speech_embeds_length: {processor.fixed_speech_embeds_length}")

    # Generate a dummy 3-second 16kHz sinusoidal waveform
    sr = 16000
    t = torch.arange(3 * sr, dtype=torch.float32) / sr
    wave = 0.1 * torch.sin(2 * torch.pi * 440 * t).unsqueeze(0).to(device)

    print("\n=== Inference-mode decode ===")
    for prompt in cfg.emo_prompts[:1]:
        out = decode_text(model, processor, wave, sr, prompt, 16, 0.0)
        print(f"Prompt: {prompt}")
        print(f"Output: {out!r}")

    print("\n=== Differentiable forward pass ===")
    wave.requires_grad_(True)
    inputs = build_inputs(
        wave, sr, cfg.emo_prompts[0], processor, device,
        torch_extractor=torch_extractor, differentiable=True,
        dtype=torch.bfloat16,
    )
    print(f"input_ids shape: {inputs['input_ids'].shape}")
    print(f"input_features shape: {inputs['input_features'].shape}")
    print(f"feature_attention_mask shape: {inputs['feature_attention_mask'].shape}")
    print(f"input_features requires_grad: {inputs['input_features'].requires_grad}")

    # Attach target label
    target_ids = processor.tokenizer.encode("happy", add_special_tokens=False)
    target = torch.tensor(target_ids, device=device, dtype=inputs["input_ids"].dtype).unsqueeze(0)
    new_input_ids = torch.cat([inputs["input_ids"], target], dim=1)
    labels = torch.full_like(new_input_ids, -100)
    labels[:, -target.shape[1]:] = target
    new_attn = torch.cat([inputs["attention_mask"], torch.ones_like(target)], dim=1)

    out = model(
        input_ids=new_input_ids,
        input_features=inputs["input_features"],
        feature_attention_mask=inputs["feature_attention_mask"],
        attention_mask=new_attn,
        labels=labels,
        return_dict=True,
    )
    print(f"Loss: {out.loss.item():.4f}")

    print("\n=== Backward pass ===")
    out.loss.backward()
    grad_norm = wave.grad.norm().item() if wave.grad is not None else float("nan")
    print(f"Gradient norm on waveform: {grad_norm:.6f}")
    if wave.grad is None or grad_norm == 0.0:
        print("ERROR: no gradient on waveform!")
        sys.exit(1)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
