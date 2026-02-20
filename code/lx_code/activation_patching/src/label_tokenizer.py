"""Label tokenization utilities for activation patching."""

from __future__ import annotations

from typing import Dict, List
import json

EMOTIONS = ["neutral", "happy", "sad", "angry", "surprised"]
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}
IDX_TO_EMOTION = {i: e for i, e in enumerate(EMOTIONS)}


class LabelTokenizer:
    """Wrap a tokenizer and prepare label token ids."""

    def __init__(self, tokenizer, emotions: List[str] | None = None) -> None:
        self.tokenizer = tokenizer
        self.emotions = emotions or EMOTIONS
        self.label_info: Dict[str, Dict] = {}
        self._tokenize_labels()

    def _tokenize_labels(self) -> None:
        for emotion in self.emotions:
            token_ids = self.tokenizer.encode(emotion, add_special_tokens=False)
            token_strs = self.tokenizer.convert_ids_to_tokens(token_ids)
            self.label_info[emotion] = {
                "token_ids": token_ids,
                "token_strs": token_strs,
                "is_single_token": len(token_ids) == 1,
                "first_token_id": token_ids[0],
                "first_token_str": token_strs[0],
            }

    def get_label_token_ids(self) -> Dict[str, int]:
        return {e: info["first_token_id"] for e, info in self.label_info.items()}

    def get_label_token_id_list(self) -> List[int]:
        return [self.label_info[e]["first_token_id"] for e in self.emotions]

    def has_multi_token_labels(self) -> bool:
        return any(not info["is_single_token"] for info in self.label_info.values())

    def get_multi_token_warnings(self) -> List[str]:
        warnings = []
        for emotion, info in self.label_info.items():
            if not info["is_single_token"]:
                warnings.append(
                    f"Warning: '{emotion}' tokenizes to multiple tokens: "
                    f"{info['token_strs']}. Using first token: {info['first_token_str']}"
                )
        return warnings

    def report(self) -> Dict:
        return {
            "emotions": self.emotions,
            "label_info": self.label_info,
            "has_multi_token": self.has_multi_token_labels(),
            "warnings": self.get_multi_token_warnings(),
        }

    def save_report(self, path: str) -> None:
        report = self.report()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    def print_summary(self) -> None:
        print("=" * 50)
        print("Label Tokenization Summary")
        print("=" * 50)
        for emotion in self.emotions:
            info = self.label_info[emotion]
            status = "single" if info["is_single_token"] else "multi"
            print(f"  {emotion}: {info['token_ids']} ({status})")
        warnings = self.get_multi_token_warnings()
        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"  {w}")
