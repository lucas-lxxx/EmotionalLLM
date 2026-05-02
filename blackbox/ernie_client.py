"""Baidu ERNIE API client for audio emotion detection.

ERNIE uses Qianfan platform. Audio input via base64 in multimodal message format.
Docs: https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Fm2vrveyu
"""
from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path

import requests

from config import cfg


class ERNIEClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or cfg.ernie_api_key
        self.model = model or cfg.ernie_model
        if not self.api_key:
            raise ValueError("ERNIE_API_KEY not set. Set ERNIE_API_KEY env var with Qianfan access_token.")
        # Qianfan uses access_token as query parameter
        self.base_url = "https://qianfan.baidubce.com/v2/chat/completions"

    def _encode_audio(self, audio_path: Path) -> str:
        with open(audio_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def query_emotion(self, audio_path: Path, prompt: str) -> str:
        """Send audio + prompt to ERNIE."""
        audio_b64 = self._encode_audio(audio_path)

        # Qianfan v2 multimodal format
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio": f"data:audio/wav;base64,{audio_b64}",
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ]
            }],
            "temperature": 0.01,  # ERNIE min is 0.01
            "max_tokens": 64,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        last_error = None
        for attempt in range(cfg.max_retries):
            try:
                resp = requests.post(
                    self.base_url,
                    json=payload,
                    headers=headers,
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    # Qianfan v2 response format
                    choices = data.get("choices", [])
                    if choices:
                        content = choices[0].get("message", {}).get("content", "")
                        return content.strip()
                    # Legacy format fallback
                    return data.get("result", "").strip()

                if resp.status_code == 429:
                    wait = cfg.retry_delay * (attempt + 1)
                    print(f"  ERNIE rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue

                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if resp.status_code >= 500:
                    time.sleep(cfg.retry_delay)
                    continue
                break

            except requests.exceptions.Timeout:
                last_error = "Request timeout"
                time.sleep(cfg.retry_delay)
                continue
            except Exception as e:
                last_error = str(e)
                break

        print(f"  ERNIE error after {cfg.max_retries} attempts: {last_error}")
        return ""

    def query_emotion_3prompt(self, audio_path: Path) -> dict:
        results = []
        for i, prompt in enumerate(cfg.emo_prompts):
            raw = self.query_emotion(audio_path, prompt)
            label = normalize_emotion(raw)
            results.append({"prompt_idx": i, "raw": raw, "label": label})
            time.sleep(cfg.request_delay)

        labels = [r["label"] for r in results]
        majority = _majority_vote(labels)
        return {"per_prompt": results, "majority_label": majority}


def normalize_emotion(text: str) -> str:
    text_lower = text.lower().strip()
    # Try Chinese labels first
    for key, val in cfg.label_map.items():
        if key in text_lower:
            return val
    # Then English
    clean = re.sub(r"[^a-z\s]", "", text_lower)
    tokens = clean.split()
    for token in tokens:
        if token in cfg.label_map:
            return cfg.label_map[token]
    for label in cfg.emo_labels:
        if label in clean:
            return label
    return tokens[0] if tokens else ""


def _majority_vote(labels: list[str]) -> str:
    from collections import Counter
    counts = Counter(labels)
    if not counts:
        return ""
    winner, count = counts.most_common(1)[0]
    return winner if count >= 2 else ""
