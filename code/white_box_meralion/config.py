from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass
class Config:
    # Paths
    model_path: Path = Path("/data1/lixiang/MERaLiON-2-3B")
    results_dir: Path = Path(__file__).resolve().parent / "result" / "MERaLiON"

    # ESD dataset config
    esd_dataset_root: Path = Path("/data1/lixiang/OpenS2S_dataset/ESD/CN")
    esd_samples_per_emotion: int = 25
    esd_exclude_emotion: str = "happy"
    results_by_speaker: bool = True
    speaker_results_dir: Path = Path(__file__).resolve().parent / "result" / "MERaLiON"

    # Runtime
    device: str = "cuda:0"
    seed: int = 1234

    # Prompts
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
            "Please transcribe this speech.",
        ]
    )
    target_emotion: str = "happy"

    # Decode
    temperature: float = 0.0
    emo_max_new_tokens: int = 16
    asr_max_new_tokens: int = 256

    # Attack (same budget as Voxtral / OpenS2S)
    epsilon: float = 0.008
    total_steps: int = 60
    stage_a_steps: int = 20
    lr: float = 0.003
    optimizer: str = "sgd"

    lambda_emo: float = 1.0
    lambda_asr_stage_a: float = 1e-4
    lambda_asr_stage_b: float = 1e-2
    lambda_per_stage_a: float = 0.0
    lambda_per_stage_b: float = 1e-5

    # EoT
    eot_samples: int = 1
    eot_max_shift: int = 160
    eot_gain_min: float = 0.8
    eot_gain_max: float = 1.2
    eot_noise_std: float = 0.0
    eot_band_limit: bool = False

    # Audio / feature extraction (Whisper front-end)
    sample_rate: int = 16000
    n_mels: int = 128
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400

    # Perceptual loss
    per_fft_sizes: tuple[int, ...] = (256, 512, 1024)
    per_hop_sizes: tuple[int, ...] = (64, 128, 256)
    per_win_lengths: tuple[int, ...] = (256, 512, 1024)

    # Metrics
    wer_thresholds: tuple[float, ...] = (0.0, 0.05)

    # Semantic evaluation
    semantic_sim_model: str = "/data1/lixiang/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
    semantic_threshold: float = 0.8

    # Gradient chain checks
    grad_norm_min: float = 1e-8
    grad_norm_patience: int = 3

    # Batch controls
    skip_existing: bool = True


cfg = Config()
