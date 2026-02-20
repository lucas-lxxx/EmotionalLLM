"""
Configuration for Moshi white-box attack
Adapted from OpenS2S attack framework
"""
from dataclasses import dataclass


@dataclass
class Config:
    # ========== Model Paths ==========
    moshi_model_path: str = "/data1/lixiang/Moshi/moshiko-pytorch-bf16/kyutai/moshiko-pytorch-bf16"

    # ========== Dataset Paths ==========
    ravdess_base: str = "/data1/lixiang/OpenS2S_dataset/RAVDESS"
    pairs_json: str = "/data1/lixiang/OpenS2S_dataset/RAVDESS/pairs.json"

    # ========== Attack Parameters ==========
    epsilon: float = 0.008  # L_inf constraint
    total_steps: int = 60
    stage_a_steps: int = 20  # Focus on emotion manipulation
    lr: float = 0.003
    optimizer: str = "adam"

    # ========== Loss Weights ==========
    # Stage A: Focus on emotion
    lambda_emo: float = 1.0
    lambda_asr_stage_a: float = 1e-4
    lambda_per_stage_a: float = 0.0

    # Stage B: Balance emotion + semantic preservation
    lambda_asr_stage_b: float = 1e-2
    lambda_per_stage_b: float = 1e-5

    # ========== EoT Parameters ==========
    eot_samples: int = 1  # Number of EoT samples per step
    eot_max_shift: int = 160  # Time shift in samples (±160)
    eot_gain_min: float = 0.8
    eot_gain_max: float = 1.2
    eot_noise_std: float = 0.0  # Gaussian noise std
    eot_band_limit: bool = False

    # ========== Prompts ==========
    emo_prompts: list = None  # Will be set in __post_init__
    asr_prompt: str = "Transcribe the speech exactly. Output only the transcript."
    system_prompt: str = None  # Moshi may not use system prompts

    # ========== Perceptual Loss ==========
    per_fft_sizes: list = None
    per_hop_sizes: list = None
    per_win_lengths: list = None

    # ========== Gradient Monitoring ==========
    grad_norm_min: float = 1e-8
    grad_norm_patience: int = 5

    # ========== Device ==========
    device: str = "cuda"

    # ========== Output ==========
    output_dir: str = "./results"
    save_audio: bool = True

    # ========== Experiment ==========
    num_samples: int = 10  # Number of samples to attack
    target_emotion: str = "happy"
    source_emotion: str = "sad"

    def __post_init__(self):
        if self.emo_prompts is None:
            self.emo_prompts = [
                "What is the emotion of this audio? Answer with exactly one word from: happy, sad, angry, neutral.",
                "Classify the emotion. Output exactly one word: happy/sad/angry/neutral.",
            ]

        if self.per_fft_sizes is None:
            self.per_fft_sizes = [512, 1024, 2048]
            self.per_hop_sizes = [128, 256, 512]
            self.per_win_lengths = [512, 1024, 2048]


# Global config instance
cfg = Config()
