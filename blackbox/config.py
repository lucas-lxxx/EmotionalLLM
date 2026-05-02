"""Configuration for black-box transfer attack experiments."""
from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class BlackboxConfig:
    # ── API Keys (from environment variables) ──
    gemini_api_key: str = field(default_factory=lambda: os.environ.get("GEMINI_API_KEY", ""))
    dashscope_api_key: str = field(default_factory=lambda: os.environ.get("DASHSCOPE_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))

    # ── Target Model Configs ──
    # Gemini 2.5 Flash
    gemini_flash_model: str = "gemini-2.5-flash"
    # Gemini 2.5 Pro
    gemini_pro_model: str = "gemini-2.5-pro"
    gemini_endpoint: str = "https://generativelanguage.googleapis.com/v1beta/models"

    # Qwen3-Omni-Flash (DashScope)
    qwen3_model: str = "qwen3-omni-flash"
    # Qwen-Omni-Turbo (older, also supports audio)
    qwen_turbo_model: str = "qwen-omni-turbo"
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # OpenAI audio model
    openai_audio_model: str = "gpt-audio"
    openai_base_url: str = "https://api.openai.com/v1"

    # ── Paths ──
    blackbox_root: Path = Path(__file__).resolve().parent
    results_dir: Path = Path(__file__).resolve().parent / "results"

    # White-box result directories (source of adversarial samples)
    voxtral_en_dir: Path = Path(__file__).resolve().parents[1] / "code" / "white_box_voxtral" / "result" / "Voxtral_EN"
    voxtral_cn_dir: Path = Path(__file__).resolve().parents[1] / "code" / "white_box_voxtral" / "result" / "Voxtral_CN"
    opens2s_en_dir: Path = Path(__file__).resolve().parents[1] / "code" / "white_box_opens2s_v2" / "result" / "blackbox" / "EN"
    opens2s_cn_dir: Path = Path(__file__).resolve().parents[1] / "code" / "white_box_opens2s_v2" / "result" / "blackbox" / "CN"

    # Clean audio base (ESD dataset) - remote server path prefix
    esd_remote_base: str = "/data1/lixiang/OpenS2S_dataset/ESD"
    esd_local_bases: list[Path] = field(default_factory=lambda: [
        Path(p) for p in [
            os.environ.get("ESD_LOCAL_BASE", ""),
            str(Path(__file__).resolve().parents[1] / "OpenS2S_dataset" / "ESD"),
            str(Path(__file__).resolve().parents[1] / "dataset" / "ESD"),
            str(Path(__file__).resolve().parents[1] / "data" / "ESD"),
            str(Path(__file__).resolve().parents[1] / "ESD"),
        ] if p
    ])

    # Random noise output
    noise_dir: Path = Path(__file__).resolve().parent / "noise_samples"

    # ── Surrogate Groups ──
    # Maps surrogate_key -> (display_name, result_dir_attr, language, speakers)
    # result_dir_attr is used to look up the actual Path on this config object
    surrogate_groups: dict = field(default_factory=lambda: {
        "voxtral_en": {"name": "Voxtral EN", "dir_attr": "voxtral_en_dir", "lang": "EN", "speakers": [f"{i:04d}" for i in range(11, 21)]},
        "voxtral_cn": {"name": "Voxtral CN", "dir_attr": "voxtral_cn_dir", "lang": "CN", "speakers": [f"{i:04d}" for i in range(1, 11)]},
        "opens2s_en": {"name": "OpenS2S EN", "dir_attr": "opens2s_en_dir", "lang": "EN", "speakers": [f"{i:04d}" for i in range(11, 21)]},
        "opens2s_cn": {"name": "OpenS2S CN", "dir_attr": "opens2s_cn_dir", "lang": "CN", "speakers": [f"{i:04d}" for i in range(1, 11)]},
    })

    # ── Target List ──
    # target_key -> (display_name, client_class, model_field)
    target_list: dict = field(default_factory=lambda: {
        "gemini_flash": {"name": "Gemini 2.5 Flash", "client": "gemini", "model_attr": "gemini_flash_model"},
        "gemini_pro":   {"name": "Gemini 2.5 Pro",   "client": "gemini", "model_attr": "gemini_pro_model"},
        "gpt4o_audio":  {"name": "OpenAI gpt-audio", "client": "gpt4o",  "model_attr": "openai_audio_model"},
        "qwen3_omni":   {"name": "Qwen3-Omni-Flash", "client": "qwen",   "model_attr": "qwen3_model"},
        "qwen_turbo":   {"name": "Qwen-Omni-Turbo",  "client": "qwen",   "model_attr": "qwen_turbo_model"},
    })

    # ── Emotion Labels ──
    emo_labels: list[str] = field(default_factory=lambda: ["happy", "sad", "angry", "neutral", "surprise"])
    target_emotion: str = "happy"

    # 3 emotion prompts (same as white-box)
    emo_prompts: list[str] = field(default_factory=lambda: [
        "What is the emotion of the speaker in this audio? Answer with exactly one word from: happy, sad, angry, neutral, surprise.",
        "Classify the speaker's emotion. Output exactly one word: happy/sad/angry/neutral/surprise.",
        "Emotion label only (one word): happy, sad, angry, neutral, or surprise.",
    ])

    # Label normalization mapping
    label_map: dict[str, str] = field(default_factory=lambda: {
        "happy": "happy", "happiness": "happy", "joy": "happy", "joyful": "happy",
        "cheerful": "happy", "excited": "happy", "delighted": "happy",
        "sad": "sad", "sadness": "sad", "sorrow": "sad", "sorrowful": "sad",
        "unhappy": "sad", "depressed": "sad", "melancholy": "sad",
        "angry": "angry", "anger": "angry", "angr": "angry", "furious": "angry",
        "irritated": "angry", "annoyed": "angry", "mad": "angry", "rage": "angry",
        "neutral": "neutral", "calm": "neutral", "flat": "neutral", "normal": "neutral",
        "surprise": "surprise", "surprised": "surprise", "surprising": "surprise",
        "shock": "surprise", "shocked": "surprise", "astonished": "surprise",
        # Chinese label mappings
        "高兴": "happy", "开心": "happy", "快乐": "happy", "喜悦": "happy",
        "悲伤": "sad", "伤心": "sad", "难过": "sad",
        "愤怒": "angry", "生气": "angry", "恼怒": "angry",
        "中性": "neutral", "平静": "neutral", "平淡": "neutral",
        "惊讶": "surprise", "吃惊": "surprise", "震惊": "surprise",
    })

    # ── Rate Limiting ──
    request_delay: float = 1.0
    max_retries: int = 3
    retry_delay: float = 5.0
    prompt_parallelism: int = 3
    sample_parallelism: int = 2
    run_all_max_workers: int = 4

    # ── Evaluation ──
    skip_existing: bool = True
    per_emotion: int = 250  # samples per emotion for selection

    # ── White-box reference SNR for noise baseline ──
    voxtral_avg_snr: float = 20.6  # dB
    opens2s_avg_snr: float = 16.4  # dB

    def get_surrogate_dir(self, surrogate_key: str) -> Path:
        info = self.surrogate_groups[surrogate_key]
        return getattr(self, info["dir_attr"])

    def get_target_model(self, target_key: str) -> str:
        info = self.target_list[target_key]
        return getattr(self, info["model_attr"])

    def resolve_clean_audio_path(self, original_audio_path: str) -> Path | None:
        if not original_audio_path:
            return None

        original = Path(original_audio_path)
        if original.exists():
            return original

        normalized = original_audio_path.replace("\\", "/")
        remote_prefix = self.esd_remote_base.rstrip("/")
        if not normalized.startswith(remote_prefix + "/"):
            return None

        relative = normalized[len(remote_prefix) + 1:]
        for base in self.esd_local_bases:
            candidate = base / Path(relative)
            if candidate.exists():
                return candidate
        return None


cfg = BlackboxConfig()
