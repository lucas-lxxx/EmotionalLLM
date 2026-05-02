"""Q3 实验配置：GAO 黑盒语音畸变攻击迁移到 Voxtral

参考 Gao et al. "Black-box adversarial attacks through speech distortion
for speech emotion recognition" 中的三种语音畸变方法：
  - VTLN (Vocal Tract Length Normalization)
  - McAdams transformation
  - MSS (Modulation Spectrum Smoothing)
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
    work_dir: Path = Path("/data1/lixiang/EmotionalLLM/observation_v3/Q3/GAO")
    distorted_audio_dir: Path = Path("/data1/lixiang/EmotionalLLM/observation_v3/Q3/GAO/distorted_audio")
    results_dir: Path = Path("/data1/lixiang/EmotionalLLM/observation_v3/Q3/GAO/result")

    # ── 数据集 ──
    emotions: list[str] = field(
        default_factory=lambda: ["angry", "happy", "neutral", "sad", "surprise"]
    )
    emotion2idx: dict[str, int] = field(default_factory=lambda: {
        "angry": 0, "happy": 1, "neutral": 2, "sad": 3, "surprise": 4,
    })
    sample_rate: int = 16000
    max_audio_sec: float = 6.0

    # ── 畸变超参数（论文最佳攻击效果） ──
    vtln_alpha: float = 0.15       # VTLN warping factor (论文 Table 2 最佳)
    mcadams_alpha: float = 0.80    # McAdams coefficient (论文 Table 3 最佳)
    mss_alpha: float = 0.25        # MSS cutoff (论文 Table 4 最佳)

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
