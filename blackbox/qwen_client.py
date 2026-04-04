"""Qwen3.5-Omni API client for audio emotion detection via OpenAI-compatible API."""
from __future__ import annotations

import base64
import re
import time
from pathlib import Path

from config import cfg


class QwenClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or cfg.dashscope_api_key
        self.model = model or cfg.qwen_model
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY not set")

        from openai import OpenAI
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=cfg.qwen_base_url,
        )

    def _encode_audio(self, audio_path: Path) -> str:
        with open(audio_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _detect_format(self, audio_path: Path) -> str:
        suffix = audio_path.suffix.lower()
        fmt_map = {".wav": "wav", ".mp3": "mp3", ".flac": "flac", ".aac": "aac"}
        return fmt_map.get(suffix, "wav")

    def query_emotion(self, audio_path: Path, prompt: str) -> str:
        """Send audio + prompt to Qwen3.5-Omni, return raw text response."""
        audio_b64 = self._encode_audio(audio_path)
        audio_fmt = self._detect_format(audio_path)

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": f"data:;base64,{audio_b64}",
                        "format": audio_fmt,
                    },
                },
                {
                    "type": "text",
                    "text": prompt,
                }
            ]
        }]

        last_error = None
        for attempt in range(cfg.max_retries):
            try:
                # Qwen-Omni requires stream=True, text-only output
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    modalities=["text"],
                    stream=True,
                    stream_options={"include_usage": True},
                    temperature=0.0,
                    max_tokens=32,
                )

                text_parts = []
                for chunk in completion:
                    if chunk.choices and chunk.choices[0].delta.content:
                        text_parts.append(chunk.choices[0].delta.content)

                return "".join(text_parts).strip()

            except Exception as e:
                last_error = str(e)
                if "rate" in last_error.lower() or "429" in last_error:
                    wait = cfg.retry_delay * (attempt + 1)
                    print(f"  Qwen rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                if "500" in last_error or "502" in last_error or "503" in last_error:
                    time.sleep(cfg.retry_delay)
                    continue
                break

        print(f"  Qwen error after {cfg.max_retries} attempts: {last_error}")
        return ""

    def query_emotion_3prompt(self, audio_path: Path) -> dict:
        """Query with 3 prompts, return per-prompt and majority vote results."""
        results = []
        for i, prompt in enumerate(cfg.emo_prompts):
            raw = self.query_emotion(audio_path, prompt)
            label = normalize_emotion(raw)
            results.append({"prompt_idx": i, "raw": raw, "label": label})
            time.sleep(cfg.request_delay)

        labels = [r["label"] for r in results]
        majority = _majority_vote(labels)
        return {
            "per_prompt": results,
            "majority_label": majority,
        }


def normalize_emotion(text: str) -> str:
    """Extract and normalize emotion label from free-form response."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()

    for token in tokens:
        if token in cfg.label_map:
            return cfg.label_map[token]

    for label in cfg.emo_labels:
        if label in text:
            return label

    return tokens[0] if tokens else ""


def _majority_vote(labels: list[str]) -> str:
    from collections import Counter
    counts = Counter(labels)
    if not counts:
        return ""
    winner, count = counts.most_common(1)[0]
    return winner if count >= 2 else ""


if __name__ == "__main__":
    client = QwenClient()
    print(f"Qwen client initialized: model={client.model}")
    print("Ready for evaluation. Use query_emotion_3prompt(audio_path) to test.")
