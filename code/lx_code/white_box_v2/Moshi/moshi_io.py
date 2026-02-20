"""
Moshi Model I/O Module
Adapted from OpenS2S I/O for Moshi model
"""
import torch
import torchaudio
import soundfile as sf
from moshi.models import loaders
from typing import Optional, Any


class MoshiModel:
    """Wrapper for Moshi model (Mimi + Moshi LM)"""

    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Load Moshi model components

        Args:
            model_path: Path to Moshi model directory
            device: Device to load model on
        """
        self.device = device
        self.model_path = model_path

        print(f"Loading Moshi model from {model_path}...")

        # Load Mimi audio encoder
        mimi_path = f"{model_path}/tokenizer-e351c8d8-checkpoint125.safetensors"
        self.mimi = loaders.get_mimi(mimi_path, device=device)
        self.mimi.eval()
        print("✓ Mimi encoder loaded")

        # Load Moshi language model
        moshi_lm_path = f"{model_path}/model.safetensors"
        self.moshi_lm = loaders.get_moshi_lm(moshi_lm_path, device=device)
        self.moshi_lm.eval()
        print("✓ Moshi LM loaded")

        # Get model dtype
        try:
            self.dtype = next(self.moshi_lm.parameters()).dtype
        except StopIteration:
            self.dtype = torch.float32

        print(f"Model dtype: {self.dtype}")

    def encode_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Encode audio waveform to codes

        Args:
            waveform: Audio tensor [B, C, T] at 24kHz

        Returns:
            codes: Encoded audio codes
        """
        with torch.no_grad():
            codes = self.mimi.encode(waveform)
        return codes

    def forward(self, codes: torch.Tensor, **kwargs) -> Any:
        """
        Forward pass through Moshi LM

        Args:
            codes: Encoded audio codes
            **kwargs: Additional arguments

        Returns:
            Model outputs
        """
        return self.moshi_lm(codes, **kwargs)


def load_moshi_model(model_path: str, device: str = "cuda") -> MoshiModel:
    """
    Load Moshi model

    Args:
        model_path: Path to model directory
        device: Device to load on

    Returns:
        MoshiModel instance
    """
    return MoshiModel(model_path, device)


def load_audio(audio_path: str, target_sr: int = 24000, device: str = "cuda") -> tuple[torch.Tensor, int]:
    """
    Load and preprocess audio file

    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (Moshi uses 24kHz)
        device: Device to load on

    Returns:
        waveform: Audio tensor [1, 1, T]
        sr: Sample rate
    """
    # Load audio using soundfile
    wav, sr = sf.read(audio_path, dtype='float32')

    # Convert to torch tensor
    wav = torch.from_numpy(wav)

    # Handle mono/stereo
    if wav.dim() == 1:
        # Mono: add channel dimension
        wav = wav.unsqueeze(0)
    else:
        # Stereo: transpose and convert to mono
        wav = wav.T
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        wav = resampler(wav)
        sr = target_sr

    # Add batch dimension: [1, 1, T]
    wav = wav.unsqueeze(0).to(device)

    return wav, sr


def save_audio(waveform: torch.Tensor, output_path: str, sr: int = 24000):
    """
    Save audio waveform to file

    Args:
        waveform: Audio tensor [1, 1, T] or [1, T]
        output_path: Output file path
        sr: Sample rate
    """
    # Remove batch dimension
    if waveform.dim() == 3:
        waveform = waveform.squeeze(0)

    # Move to CPU
    waveform = waveform.cpu()

    # Save
    torchaudio.save(output_path, waveform, sr)
