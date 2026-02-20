"""
Hidden States Extractor Module

从OmniSpeechModel提取各层hidden states，定位audio span并做mean pooling。
"""

import os
import sys
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm

# 保存当前的src模块引用（如果存在）
_original_src_module = sys.modules.get('src', None)

# 临时移除我们项目的src，添加OpenS2S的路径
OPENS2S_ROOT = "/data1/lixiang/Opens2s/OpenS2S/"
if OPENS2S_ROOT not in sys.path:
    sys.path.insert(0, OPENS2S_ROOT)

# 如果sys.modules中有src，临时移除
if 'src' in sys.modules:
    del sys.modules['src']

try:
    # 现在导入OpenS2S的模块
    from src.modeling_omnispeech import OmniSpeechModel
    from src.configuration_omnispeech import OmniSpeechConfig
    from src.feature_extraction_audio import WhisperFeatureExtractor
    from src.constants import (
        AUDIO_TOKEN_INDEX,
        DEFAULT_AUDIO_TOKEN,
        DEFAULT_AUDIO_START_TOKEN,
        DEFAULT_AUDIO_END_TOKEN
    )
finally:
    # 恢复我们项目的src模块
    if _original_src_module is not None:
        sys.modules['src'] = _original_src_module
    # 移除OpenS2S路径
    if OPENS2S_ROOT in sys.path:
        sys.path.remove(OPENS2S_ROOT)

from ..data.dataset import ModalConflictSample


class HiddenStatesExtractor:
    """从OmniSpeechModel提取hidden states"""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        cache_dir: Optional[str] = None,
        use_cache: bool = True
    ):
        """
        Args:
            model_path: 模型权重路径
            device: 计算设备
            dtype: 数据类型 (bfloat16, float16, float32)
            cache_dir: hidden states缓存目录
            use_cache: 是否使用缓存
        """
        self.model_path = model_path
        self.device = device
        self.dtype = getattr(torch, dtype)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = use_cache

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.feature_extractor = None
        self.tokenizer = None

    def load_model(self):
        """加载模型"""
        if self.model is not None:
            return

        print(f"Loading model from {self.model_path}...")

        # 加载模型
        self.model = OmniSpeechModel.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=True
        )
        self.model.eval()

        # 加载feature extractor
        self.feature_extractor = WhisperFeatureExtractor(
            feature_size=128,
            sampling_rate=16000,
            hop_length=160,
            chunk_length=30,
            n_fft=400,
            return_attention_mask=True
        )

        # 加载tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        # 添加特殊token（如果不存在）
        special_tokens = [DEFAULT_AUDIO_TOKEN, DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN]
        special_tokens_dict = {'additional_special_tokens': special_tokens}
        num_added = self.tokenizer.add_special_tokens(special_tokens_dict)
        if num_added > 0:
            print(f"Added {num_added} special tokens to tokenizer")
            # 调整模型的token embeddings大小
            self.model.llm_model.resize_token_embeddings(len(self.tokenizer))

        print("Model loaded successfully.")

    def _get_cache_path(self, sample: ModalConflictSample) -> Path:
        """获取缓存文件路径"""
        # 使用sample_id和audio_path的hash作为缓存key
        cache_key = hashlib.md5(
            f"{sample.sample_id}_{sample.audio_path}".encode()
        ).hexdigest()
        return self.cache_dir / f"{cache_key}.pt"

    def _load_from_cache(self, sample: ModalConflictSample) -> Optional[torch.Tensor]:
        """从缓存加载hidden states"""
        if not self.use_cache or self.cache_dir is None:
            return None

        cache_path = self._get_cache_path(sample)
        if cache_path.exists():
            return torch.load(cache_path, map_location="cpu")
        return None

    def _save_to_cache(self, sample: ModalConflictSample, hidden_states: torch.Tensor):
        """保存hidden states到缓存"""
        if not self.use_cache or self.cache_dir is None:
            return

        cache_path = self._get_cache_path(sample)
        torch.save(hidden_states, cache_path)

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """加载音频文件"""
        audio, sr = sf.read(audio_path)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        return audio, sr

    def prepare_inputs(
        self,
        text: str,
        audio_path: str
    ) -> Dict[str, torch.Tensor]:
        """准备模型输入"""
        # 加载音频
        audio, sr = self.load_audio(audio_path)

        # 提取音频特征
        audio_features = self.feature_extractor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            return_attention_mask=True
        )

        # 处理input_features（可能是列表或tensor）
        input_features = audio_features["input_features"]
        if isinstance(input_features, list):
            # 如果是列表，转换为tensor并stack
            input_features = torch.stack([torch.from_numpy(f) if isinstance(f, np.ndarray) else f for f in input_features])
        elif not isinstance(input_features, torch.Tensor):
            input_features = torch.from_numpy(input_features)

        # 确保是3D tensor: [batch, time, features]
        if input_features.dim() == 2:
            input_features = input_features.unsqueeze(0)

        # 构建输入文本
        # 不使用<|im_audio|>，而是使用一个占位符，然后手动插入AUDIO_TOKEN_INDEX
        # 使用<|im_audio_start|>作为占位符
        input_text = f"{DEFAULT_AUDIO_START_TOKEN}{text}"

        # Tokenize
        tokens = self.tokenizer(
            input_text,
            return_tensors="pt",
            add_special_tokens=True
        )

        # 将<|im_audio_start|>替换为AUDIO_TOKEN_INDEX
        input_ids = tokens["input_ids"]
        audio_start_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_AUDIO_START_TOKEN)

        if audio_start_token_id is not None:
            input_ids[input_ids == audio_start_token_id] = AUDIO_TOKEN_INDEX
        else:
            print(f"Warning: Could not find audio start token in tokenizer")

        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": tokens["attention_mask"].to(self.device),
            "speech_values": input_features.to(self.device, self.dtype),
            "speech_mask": audio_features["attention_mask"].to(self.device),
        }

    @torch.no_grad()
    def extract_hidden_states(
        self,
        sample: ModalConflictSample
    ) -> torch.Tensor:
        """
        提取单个样本的hidden states

        Args:
            sample: 模态冲突样本

        Returns:
            hidden_states: [n_layers, hidden_dim] 各层audio span的mean pooling特征
        """
        # 尝试从缓存加载
        cached = self._load_from_cache(sample)
        if cached is not None:
            return cached

        # 确保模型已加载
        self.load_model()

        # 准备输入
        inputs = self.prepare_inputs(sample.text, sample.audio_path)

        # 调试信息
        num_audio_tokens = (inputs["input_ids"] == AUDIO_TOKEN_INDEX).sum().item()
        print(f"Debug - Sample: {sample.sample_id}")
        print(f"  input_ids shape: {inputs['input_ids'].shape}")
        print(f"  num_audio_tokens: {num_audio_tokens}")
        print(f"  speech_values shape: {inputs['speech_values'].shape}")
        print(f"  speech_mask shape: {inputs['speech_mask'].shape}")

        # Forward pass with hidden states output
        # 使用prepare_inputs_labels_for_llm获取处理后的inputs_embeds
        (
            inputs_embeds,
            attention_mask,
            _,
            _
        ) = self.model.prepare_inputs_labels_for_llm(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=None,
            speech_values=inputs["speech_values"],
            speech_mask=inputs["speech_mask"],
            token_types=None,
            inference=True
        )

        # 计算audio span的位置
        # 找到原始input_ids中AUDIO_TOKEN_INDEX的位置
        audio_token_pos = (inputs["input_ids"] == AUDIO_TOKEN_INDEX).nonzero()
        if len(audio_token_pos) == 0:
            raise ValueError(f"No audio token found in input_ids for sample {sample.sample_id}")

        audio_token_idx = audio_token_pos[0, 1].item()

        # 获取speech features的长度
        speech_features, speech_attention_mask = self.model.get_speech_features(
            inputs["speech_values"],
            inputs["speech_mask"]
        )
        speech_length = speech_attention_mask.sum().item()

        # audio span: 从audio_token_idx开始，长度为speech_length
        # 注意: 在inputs_embeds中，audio token被替换为speech_features
        audio_start = audio_token_idx
        audio_end = audio_token_idx + speech_length

        # LLM forward获取所有层hidden states
        llm_output = self.model.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True
        )

        # 提取各层hidden states并对audio span做mean pooling
        all_hidden_states = llm_output.hidden_states  # tuple of (batch, seq_len, hidden_dim)
        n_layers = len(all_hidden_states) - 1  # 第0层是embedding层

        layer_features = []
        for layer_idx in range(1, len(all_hidden_states)):  # 跳过embedding层
            layer_hidden = all_hidden_states[layer_idx]  # [1, seq_len, hidden_dim]
            audio_hidden = layer_hidden[0, audio_start:audio_end, :]  # [audio_len, hidden_dim]
            pooled = audio_hidden.mean(dim=0)  # [hidden_dim]
            layer_features.append(pooled.cpu())

        hidden_states = torch.stack(layer_features, dim=0)  # [n_layers, hidden_dim]

        # 保存到缓存
        self._save_to_cache(sample, hidden_states)

        return hidden_states

    def extract_batch(
        self,
        samples: List[ModalConflictSample],
        show_progress: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        批量提取hidden states

        Args:
            samples: 样本列表
            show_progress: 是否显示进度条

        Returns:
            Dict with keys:
                - hidden_states: [n_samples, n_layers, hidden_dim]
                - semantic_labels: [n_samples]
                - prosody_labels: [n_samples]
                - text_ids: [n_samples]
                - is_conflict: [n_samples]
        """
        all_hidden_states = []
        semantic_labels = []
        prosody_labels = []
        text_ids = []
        is_conflict = []

        iterator = tqdm(samples, desc="Extracting hidden states") if show_progress else samples

        for sample in iterator:
            try:
                hs = self.extract_hidden_states(sample)
                all_hidden_states.append(hs)
                semantic_labels.append(sample.semantic_label)
                prosody_labels.append(sample.prosody_label)
                text_ids.append(sample.text_id)
                is_conflict.append(sample.is_conflict)
            except Exception as e:
                import traceback
                print(f"Error processing sample {sample.sample_id}:")
                print(f"Exception type: {type(e).__name__}")
                print(f"Exception message: {str(e)}")
                traceback.print_exc()
                continue

        return {
            "hidden_states": torch.stack(all_hidden_states, dim=0),  # [N, L, H]
            "semantic_labels": torch.tensor(semantic_labels, dtype=torch.long),
            "prosody_labels": torch.tensor(prosody_labels, dtype=torch.long),
            "text_ids": text_ids,  # 保持为列表，因为是字符串ID
            "is_conflict": torch.tensor(is_conflict, dtype=torch.bool)
        }

    def clear_cache(self):
        """清除缓存"""
        if self.cache_dir and self.cache_dir.exists():
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)


def load_extractor(config: Dict[str, Any]) -> HiddenStatesExtractor:
    """从配置加载提取器"""
    return HiddenStatesExtractor(
        model_path=config["model"]["model_path"],
        device=config["model"]["device"],
        dtype=config["model"]["dtype"],
        cache_dir=config["extraction"]["cache_dir"],
        use_cache=config["extraction"]["use_cache"]
    )
