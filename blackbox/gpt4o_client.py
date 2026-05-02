"""OpenAI audio API client for audio emotion detection."""
from __future__ import annotations

import base64
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from config import cfg


class GPT4oClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or cfg.openai_api_key
        self.model = model or cfg.openai_audio_model
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")

        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)

    def _encode_audio(self, audio_path: Path) -> str:
        with open(audio_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _detect_format(self, audio_path: Path) -> str:
        suffix = audio_path.suffix.lower()
        fmt_map = {".wav": "wav", ".mp3": "mp3", ".flac": "flac"}
        return fmt_map.get(suffix, "wav")

    def query_emotion(self, audio_path: Path, prompt: str) -> str:
        """Send audio + prompt to the OpenAI audio model."""
        audio_b64 = self._encode_audio(audio_path)
        audio_fmt = self._detect_format(audio_path)

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_b64,
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
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    modalities=["text"],
                    temperature=0.0,
                    max_tokens=32,
                )
                return completion.choices[0].message.content.strip()

            except Exception as e:
                last_error = str(e)
                if "rate" in last_error.lower() or "429" in last_error:
                    wait = cfg.retry_delay * (attempt + 1)
                    print(f"  OpenAI audio rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                if any(code in last_error for code in ["500", "502", "503"]):
                    time.sleep(cfg.retry_delay)
                    continue
                break

        print(f"  OpenAI audio error after {cfg.max_retries} attempts: {last_error}")
        return ""

    def query_emotion_3prompt(self, audio_path: Path) -> dict:
        """Query with 3 prompts, return per-prompt and majority vote results."""
        max_workers = min(max(cfg.prompt_parallelism, 1), len(cfg.emo_prompts))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                i: executor.submit(self.query_emotion, audio_path, prompt)
                for i, prompt in enumerate(cfg.emo_prompts)
            }
            results = []
            for i in range(len(cfg.emo_prompts)):
                raw = futures[i].result()
                results.append({"prompt_idx": i, "raw": raw, "label": normalize_emotion(raw)})

        labels = [r["label"] for r in results]
        majority = _majority_vote(labels)
        return {"per_prompt": results, "majority_label": majority}


def normalize_emotion(text: str) -> str:
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
