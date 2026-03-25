"""Q3 实验配置：PGD SER 攻击迁移到 Voxtral"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ── 路径 ──
    esd_root: Path = Path("/data1/lixiang/OpenS2S_dataset/ESD/EN")
    # ESD EN speakers: 0011-0020; 结构: speaker/Emotion/split/*.wav
    en_speakers: list[str] = field(
        default_factory=lambda: ["0011", "0012", "0013"]  # 3 speakers 足够
    )

    voxtral_model_path: Path = Path("/data1/lixiang/Voxtral")

    # 工作目录（服务器）- PGD 子目录
    work_dir: Path = Path("/data1/lixiang/OPUS/Q3_PGD")
    surrogate_ckpt: Path = Path("/data1/lixiang/OPUS/Q3/checkpoints/surrogate_ser.pt")  # 复用 STAA 的 surrogate
    adv_audio_dir: Path = Path("/data1/lixiang/OPUS/Q3_PGD/adv_audio")
    results_dir: Path = Path("/data1/lixiang/OPUS/Q3_PGD/results")

    # ── 数据集 ──
    emotions: list[str] = field(
        default_factory=lambda: ["angry", "happy", "neutral", "sad", "surprise"]
    )
    emotion2idx: dict[str, int] = field(default_factory=lambda: {
        "angry": 0, "happy": 1, "neutral": 2, "sad": 3, "surprise": 4,
    })
    sample_rate: int = 16000
    max_audio_sec: float = 6.0  # 截断/padding 长度

    # ── Surrogate SER ──
    wav2vec_model: str = "facebook/wav2vec2-base"
    ser_num_classes: int = 5
    ser_lr: float = 1e-4
    ser_epochs: int = 5
    ser_batch_size: int = 16

    # ── PGD 攻击参数 ──
    epsilon: float = 0.03       # L-inf 扰动上界（与 STAA 一致）
    pgd_steps: int = 50         # 迭代步数
    pgd_alpha: float = 0.001    # 步长
    pgd_random_start: bool = True

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

    # Voxtral token IDs（与 white_box_voxtral 保持一致）
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
