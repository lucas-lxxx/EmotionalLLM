"""Q3 实验配置：REN Atrous CNN 生成器攻击迁移到 Voxtral

参考 Ren et al. "Enhancing transferability of black-box adversarial attacks
via lifelong learning for speech emotion recognition models"

简化：去掉 lifelong learning 多模型部分，保留 atrous CNN generator + C&W/MSE loss
的单任务训练。Generator 适配为 1D 波形版本。
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ── 路径 ──
    esd_root: Path = Path("/data1/lixiang/OpenS2S_dataset/ESD/EN")
    en_speakers: list[str] = field(
        default_factory=lambda: ["0011", "0012", "0013"]
    )

    voxtral_model_path: Path = Path("/data1/lixiang/Voxtral")

    # 工作目录（服务器）
    work_dir: Path = Path("/data1/lixiang/EmotionalLLM/observation_v3/Q3/REN")
    surrogate_ckpt: Path = Path("/data1/lixiang/EmotionalLLM/observation_v3/Q3/STAA/checkpoints/surrogate_ser.pt")
    generator_ckpt: Path = Path("/data1/lixiang/EmotionalLLM/observation_v3/Q3/REN/checkpoints/atrous_generator.pt")
    adv_audio_dir: Path = Path("/data1/lixiang/EmotionalLLM/observation_v3/Q3/REN/adv_audio")
    results_dir: Path = Path("/data1/lixiang/EmotionalLLM/observation_v3/Q3/REN/result")

    # ── 数据集 ──
    emotions: list[str] = field(
        default_factory=lambda: ["angry", "happy", "neutral", "sad", "surprise"]
    )
    emotion2idx: dict[str, int] = field(default_factory=lambda: {
        "angry": 0, "happy": 1, "neutral": 2, "sad": 3, "surprise": 4,
    })
    sample_rate: int = 16000
    max_audio_sec: float = 6.0

    # ── Surrogate SER ──
    wav2vec_model: str = "/data1/lixiang/EmotionalLLM/observation_v3/Q3/STAA/checkpoints/wav2vec2-base"
    ser_num_classes: int = 5

    # ── Atrous CNN Generator（论文参数适配为 1D） ──
    # 原文：4 conv layers, channels {64, 128, 64, 1}, dilation {1, 2, 4, 8}, kernel (5,5)
    gen_channels: list[int] = field(
        default_factory=lambda: [64, 128, 64, 1]
    )
    gen_dilations: list[int] = field(
        default_factory=lambda: [1, 2, 4, 8]
    )
    gen_kernel_size: int = 5

    # 训练超参
    gen_lr: float = 1e-4
    gen_epochs: int = 10
    gen_batch_size: int = 1  # 内存限制
    gen_max_train_samples: int = 200
    alpha_loss: float = 0.02   # 论文中 α=0.02，平衡 C&W loss 和 MSE loss
    epsilon: float = 0.03      # L∞ perturbation bound
    cw_confidence: float = 0.0

    # ── Voxtral 评估 ──
    device: str = "cuda:0"
    emo_labels: list[str] = field(
        default_factory=lambda: ["happy", "sad", "angry", "neutral", "surprise"]
    )
    emo_prompt: str = (
        "What is the emotion of this audio? "
        "Answer with exactly one word from: happy, sad, angry, neutral, surprise."
    )
    emo_max_new_tokens: int = 16
    temperature: float = 0.0

    # Voxtral token IDs
    audio_token_id: int = 24
    begin_audio_id: int = 25
    n_audio_tokens: int = 375
    bos_id: int = 1
    eos_id: int = 2
    inst_id: int = 3
    inst_end_id: int = 4

    @property
    def max_audio_len(self) -> int:
        return int(self.sample_rate * self.max_audio_sec)


cfg = Config()
