"""Utility helpers for OpenS2S white-box attack experiments."""

import os
from typing import Dict, List, Optional, Tuple

import torch
import soundfile as sf
from transformers import AutoTokenizer

# Note: These imports need to be available in the OpenS2S environment
# Users should ensure OpenS2S is properly installed
try:
    from src.modeling_omnispeech import OmniSpeechModel
    from src.feature_extraction_audio import WhisperFeatureExtractor
except ImportError:
    print("Warning: OpenS2S model classes not found. Please ensure OpenS2S is properly installed.")
    OmniSpeechModel = None
    WhisperFeatureExtractor = None


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_tokenizer(tokenizer_path: Optional[str], omnipath: str):
    if tokenizer_path is None:
        tokenizer_path = omnipath
    tokenizer_json = os.path.join(tokenizer_path, "tokenizer.json") if tokenizer_path else None
    if tokenizer_json and os.path.exists(tokenizer_json):
        try:
            return AutoTokenizer.from_pretrained(
                tokenizer_path, 
                trust_remote_code=True,
                local_files_only=True,
                fix_mistral_regex=True
            )
        except Exception as e:
            print(f"Warning: Failed to load tokenizer with local_files_only=True: {e}")
            print(f"Trying without local_files_only...")
            return AutoTokenizer.from_pretrained(
                tokenizer_path, 
                trust_remote_code=True,
                fix_mistral_regex=True
            )
    print(f"Warning: Tokenizer not found at {tokenizer_path}, trying Qwen/Qwen3-8B-Instruct")
    try:
        return AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-8B-Instruct", 
            trust_remote_code=True,
            local_files_only=True,
            fix_mistral_regex=True
        )
    except Exception as e:
        print(f"Warning: Failed to load with local_files_only=True: {e}")
        return AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-8B-Instruct", 
            trust_remote_code=True,
            fix_mistral_regex=True
        )


def load_model(
    omnipath: str,
    tokenizer_path: Optional[str] = None,
    device: str = "cuda:0",
) -> Tuple[OmniSpeechModel, AutoTokenizer]:
    if OmniSpeechModel is None:
        raise ImportError("OmniSpeechModel not available. Please install OpenS2S.")
    tokenizer = load_tokenizer(tokenizer_path, omnipath)
    model = OmniSpeechModel.from_pretrained(
        omnipath,
        dtype=torch.bfloat16,
        device_map=device,
    )
    model.tokenizer = tokenizer
    model.eval()
    return model, tokenizer


def load_audio_extractor(omnipath: str) -> WhisperFeatureExtractor:
    if WhisperFeatureExtractor is None:
        raise ImportError("WhisperFeatureExtractor not available. Please install OpenS2S.")
    audio_dir = os.path.join(omnipath, "audio")
    if not os.path.isdir(audio_dir):
        raise FileNotFoundError(f"Audio extractor directory not found: {audio_dir}")
    return WhisperFeatureExtractor.from_pretrained(audio_dir)


def load_waveform(audio_path: str) -> Tuple[torch.Tensor, int]:
    data, sr = sf.read(audio_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    waveform = torch.from_numpy(data.astype("float32"))
    return waveform, sr


def get_audio_encoder_layers(model: OmniSpeechModel) -> Dict[str, torch.nn.Module]:
    encoder = model.audio_encoder_model
    layer_iterables: List[torch.nn.Module] = []

    candidate_attrs = ["encoder", "model", "audio_encoder"]
    for attr in candidate_attrs:
        if hasattr(encoder, attr):
            layer_iterables.append(getattr(encoder, attr))
    layer_iterables.append(encoder)

    layers = None
    for root in layer_iterables:
        if root is None:
            continue
        for attr in ["layers", "layer", "blocks", "h"]:
            if hasattr(root, attr):
                maybe_layers = getattr(root, attr)
                if isinstance(maybe_layers, (list, tuple, torch.nn.ModuleList)) and len(maybe_layers) > 0:
                    layers = maybe_layers
                    break
        if layers is not None:
            break

    if layers is None:
        raise RuntimeError("Unable to find audio encoder layers for hooking.")

    layer_dict = {}
    for idx, module in enumerate(layers):
        layer_dict[f"layer_{idx:02d}"] = module
    return layer_dict

