from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass
class Config:
    # Paths (edit these in one place; Methodology §0 + §9)
    repo_root: Path = Path(__file__).resolve().parents[3]
    opens2s_root: Path = Path("/data1/lixiang/Opens2s/OpenS2S")
    model_path: Path = Path("/data1/lixiang/Opens2s/OpenS2S/models/OpenS2S")
    sample_list_path: Path = Path(__file__).resolve().parent / "testN10" / "sample_list.txt"
    results_dir: Path = Path(__file__).resolve().parent / "testN10"

    # ESD 数据集配置
    esd_dataset_root: Path = Path("/data1/lixiang/ESD/CN")
    esd_samples_per_emotion: int = 100  # 恢复为正式实验值
    esd_exclude_emotion: str = "happy"
    results_by_speaker: bool = True
    speaker_results_dir: Path = Path(__file__).resolve().parent / "results_esd"

    # Runtime
    device: str = "cuda:6"  # 使用 H100 (80GB)
    seed: int = 1234

    # Prompts (Methodology §3)
    system_prompt: str = "You are a helpful assistant."
    emo_labels: list[str] = field(default_factory=lambda: ["happy", "sad", "angry", "neutral", "surprise"])
    emo_prompts: list[str] = field(
        default_factory=lambda: [
            "What is the emotion of this audio? Answer with exactly one word from: happy, sad, angry, neutral, surprise.",
            "Classify the emotion. Output exactly one word: happy/sad/angry/neutral/surprise.",
            "Emotion label only (one word): happy, sad, angry, neutral, or surprise.",
        ]
    )
    asr_prompts: list[str] = field(
        default_factory=lambda: [
            "Transcribe the speech exactly. Output only the transcript.",
        ]
    )
    target_emotion: str = "happy"

    # Decode (Methodology §8.1)
    temperature: float = 0.0
    emo_max_new_tokens: int = 16
    asr_max_new_tokens: int = 256

    # Attack (Methodology §5-§7)
    epsilon: float = 0.008
    total_steps: int = 60  # 恢复为正式实验值
    stage_a_steps: int = 20  # 恢复为正式实验值
    lr: float = 0.003
    optimizer: str = "adam"

    lambda_emo: float = 1.0
    lambda_asr_stage_a: float = 1e-4
    lambda_asr_stage_b: float = 1e-2
    lambda_per_stage_a: float = 0.0
    lambda_per_stage_b: float = 1e-5

    # EoT (Methodology §6)
    eot_samples: int = 1
    eot_max_shift: int = 160  # samples
    eot_gain_min: float = 0.8
    eot_gain_max: float = 1.2
    eot_noise_std: float = 0.0
    eot_band_limit: bool = False

    # Audio / feature extraction
    sample_rate: int = 16000  # OpenS2S Whisper front-end expects 16kHz.
    n_mels: int = 128
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400

    # Perceptual loss (Methodology §5.3)
    per_fft_sizes: tuple[int, ...] = (256, 512, 1024)
    per_hop_sizes: tuple[int, ...] = (64, 128, 256)
    per_win_lengths: tuple[int, ...] = (256, 512, 1024)

    # Metrics (Methodology §8.2)
    wer_thresholds: tuple[float, ...] = (0.0, 0.05)

    # Gradient chain checks (Methodology §4.2)
    grad_norm_min: float = 1e-8
    grad_norm_patience: int = 3

    # Batch controls
    skip_existing: bool = True


cfg = Config()
