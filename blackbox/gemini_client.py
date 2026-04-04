"""Gemini 2.5 Flash API client for audio emotion detection via REST API."""
from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path

import requests

from config import cfg


class GeminiClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or cfg.gemini_api_key
        self.model = model or cfg.gemini_model
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set")

    def _build_url(self) -> str:
        return f"{cfg.gemini_endpoint}/{self.model}:generateContent?key={self.api_key}"

    def _encode_audio(self, audio_path: Path) -> str:
        with open(audio_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _detect_mime(self, audio_path: Path) -> str:
        suffix = audio_path.suffix.lower()
        mime_map = {
            ".wav": "audio/wav",
            ".mp3": "audio/mp3",
            ".flac": "audio/flac",
            ".aac": "audio/aac",
            ".ogg": "audio/ogg",
        }
        return mime_map.get(suffix, "audio/wav")

    def query_emotion(self, audio_path: Path, prompt: str) -> str:
        """Send audio + prompt to Gemini, return raw text response."""
        audio_b64 = self._encode_audio(audio_path)
        mime_type = self._detect_mime(audio_path)

        payload = {
            "contents": [{
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": audio_b64,
                        }
                    },
                    {
                        "text": prompt,
                    }
                ]
            }],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 256,
                "thinkingConfig": {"thinkingBudget": 0},
            }
        }

        url = self._build_url()
        last_error = None

        for attempt in range(cfg.max_retries):
            try:
                resp = requests.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            return parts[0].get("text", "").strip()
                    return ""

                if resp.status_code == 429:
                    wait = cfg.retry_delay * (attempt + 1)
                    print(f"  Gemini rate limited, waiting {wait}s...")
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

        print(f"  Gemini error after {cfg.max_retries} attempts: {last_error}")
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
    """Extract and normalize emotion label from API response text."""
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
    """Return the label that appears most often (>=2/3)."""
    from collections import Counter
    counts = Counter(labels)
    if not counts:
        return ""
    winner, count = counts.most_common(1)[0]
    return winner if count >= 2 else ""


if __name__ == "__main__":
    client = GeminiClient()
    print(f"Gemini client initialized: model={client.model}")
    print("Ready for evaluation. Use query_emotion_3prompt(audio_path) to test.")
