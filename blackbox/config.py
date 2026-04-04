from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class BlackboxConfig:
    # API Keys (from environment variables)
    gemini_api_key: str = field(default_factory=lambda: os.environ.get("GEMINI_API_KEY", ""))
    dashscope_api_key: str = field(default_factory=lambda: os.environ.get("DASHSCOPE_API_KEY", ""))

    # Gemini API
    gemini_model: str = "gemini-2.5-flash"
    gemini_endpoint: str = "https://generativelanguage.googleapis.com/v1beta/models"

    # Qwen API (OpenAI-compatible)
    # qwen3-omni-flash: hybrid thinking omni model (free preview, supports audio input)
    # qwen3.5-omni-plus/flash need separate activation in DashScope console
    qwen_model: str = "qwen3-omni-flash"
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # Paths
    blackbox_root: Path = Path(__file__).resolve().parent
    # Local adversarial samples (WAV + JSON copied from server)
    sample_dir: Path = Path(__file__).resolve().parent / "sample"
    # White-box Voxtral EN results (original nested structure, on server)
    whitebox_result_dir: Path = Path(__file__).resolve().parents[1] / "code" / "white_box_voxtral" / "result" / "Voxtral_EN"
    results_dir: Path = Path(__file__).resolve().parent / "results"

    # Emotion labels
    emo_labels: list[str] = field(default_factory=lambda: ["happy", "sad", "angry", "neutral", "surprise"])
    target_emotion: str = "happy"

    # 3 emotion prompts (adapted from white-box config for API use)
    emo_prompts: list[str] = field(default_factory=lambda: [
        "What is the emotion of the speaker in this audio? Answer with exactly one word from: happy, sad, angry, neutral, surprise.",
        "Classify the speaker's emotion. Output exactly one word: happy/sad/angry/neutral/surprise.",
        "Emotion label only (one word): happy, sad, angry, neutral, or surprise.",
    ])

    # Label normalization mapping (various API returns -> standard labels)
    label_map: dict[str, str] = field(default_factory=lambda: {
        "happy": "happy", "happiness": "happy", "joy": "happy", "joyful": "happy",
        "sad": "sad", "sadness": "sad", "sorrow": "sad",
        "angry": "angry", "anger": "angry", "angr": "angry",
        "neutral": "neutral", "calm": "neutral",
        "surprise": "surprise", "surprised": "surprise", "surprising": "surprise",
    })

    # Rate limiting
    request_delay: float = 1.0  # seconds between requests
    max_retries: int = 3
    retry_delay: float = 5.0  # seconds between retries

    # Evaluation
    skip_existing: bool = True


cfg = BlackboxConfig()
