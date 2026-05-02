"""训练 Sad/Happy 情绪分类器（使用 ESD 数据集，保证泛化性）"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils_audio import load_model, load_audio_extractor, load_waveform
from utils.emotion_classifier import train_emotion_classifier, FrozenEmotionClassifier
from utils.esd_data_loader import (
    get_esd_speakers, get_audio_files, get_sample_with_text, ESD_BASE_PATH
)
from utils_audio import get_audio_encoder_layers
from constants import (
    DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_TOKEN,
    DEFAULT_AUDIO_END_TOKEN, DEFAULT_TTS_START_TOKEN, AUDIO_TOKEN_INDEX
)


def extract_hidden_states_from_audio(
    model, tokenizer, audio_extractor,
    audio_path: str,
    text: str = "",
    target_layers: list = ['layer_06', 'layer_16', 'layer_25'],
    device: str = "cuda:0"
) -> np.ndarray:
    """
    从音频中提取 hidden states
    
    Returns:
        z: [D] 或 [R] 特征向量
    """
    # 注册 hooks 来提取 hidden states
    audio_layers = get_audio_encoder_layers(model)
    target_layer_modules = {
        name: audio_layers[name] 
        for name in target_layers 
        if name in audio_layers
    }
    
    hidden_states_cache = {}
    hooks = []
    
    def make_hook(layer_name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            hidden_states_cache[layer_name] = hidden
        return hook
    
    for layer_name, layer_module in target_layer_modules.items():
        hook = make_hook(layer_name)
        handle = layer_module.register_forward_hook(hook)
        hooks.append(handle)
    
    try:
        # 加载音频
        waveform, sample_rate = load_waveform(audio_path)
        wave_np = waveform.detach().cpu().numpy() if isinstance(waveform, torch.Tensor) else waveform
        inputs = audio_extractor(
            [wave_np],
            sampling_rate=audio_extractor.sampling_rate,
            return_attention_mask=True,
            return_tensors="pt",
        )
        speech_values = inputs.input_features.to(device)
        speech_mask = inputs.attention_mask.to(device)
        
        # 准备文本输入
        prompt_with_audio = DEFAULT_AUDIO_START_TOKEN + DEFAULT_AUDIO_TOKEN
        if text:
            prompt_with_audio += text
        prompt_with_audio += DEFAULT_AUDIO_END_TOKEN + DEFAULT_TTS_START_TOKEN
        
        segments = prompt_with_audio.split(DEFAULT_AUDIO_TOKEN)
        ids = []
        for idx, seg in enumerate(segments):
            if idx != 0:
                ids.append(AUDIO_TOKEN_INDEX)
            ids.extend(tokenizer.encode(seg))
        input_ids = torch.LongTensor(ids).unsqueeze(0).to(device)
        
        # 前向传播
        with torch.no_grad():
            (
                inputs_embeds,
                attention_mask_llm,
                _,
                _
            ) = model.prepare_inputs_labels_for_llm(
                input_ids,
                torch.ones_like(input_ids),
                None,
                speech_values,
                speech_mask,
                None,
                left_pad=True,
                inference=True
            )
            
            outputs = model.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask_llm,
                return_dict=True,
                output_hidden_states=True,
            )
        
        # 提取 hidden states
        all_hiddens = []
        for layer_name in target_layers:
            if layer_name in hidden_states_cache:
                hidden = hidden_states_cache[layer_name]
                if hidden.dim() == 3:
                    hidden = hidden.mean(dim=1)  # [B, T, D] -> [B, D]
                elif hidden.dim() == 2:
                    pass  # [B, D] 已经是正确的
                else:
                    continue
                all_hiddens.append(hidden)
        
        if not all_hiddens:
            # 回退：使用 LLM 的 hidden states
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                z = outputs.hidden_states[-1][0, -1, :].cpu().float().numpy()
            else:
                return None
        else:
            # 平均所有层的 hidden states
            z = torch.stack(all_hiddens, dim=0).mean(dim=0)  # [B, D]
            # 转换为 float32 再转为 numpy（避免 bfloat16 问题）
            z = z[0].cpu().float().numpy()  # [D]
        
        return z
    
    finally:
        # 移除 hooks
        for handle in hooks:
            handle.remove()


def collect_esd_data(
    omnispeech_path: str,
    emotions: list = ["Sad", "Happy"],
    split: str = "train",
    max_samples_per_emotion: int = None,
    max_speakers: int = None,
    device: str = "cuda:0"
):
    """
    从 ESD 数据集中收集数据并提取特征
    
    Returns:
        Z: [N, D] 特征矩阵
        y: [N] 情绪标签
        label_to_idx: {emotion: index}
    """
    print("=" * 80)
    print("Collecting ESD data and extracting features...")
    print("=" * 80)
    
    # 加载模型
    print("Loading OpenS2S model...")
    model, tokenizer = load_model(omnispeech_path, device=device)
    model.eval()
    audio_extractor = load_audio_extractor(omnispeech_path)
    
    # 获取说话人列表
    speakers = get_esd_speakers()
    if max_speakers is not None:
        speakers = speakers[:max_speakers]
    print(f"Using {len(speakers)} speakers: {speakers}")
    
    # 创建标签映射
    label_to_idx = {emotion: idx for idx, emotion in enumerate(emotions)}
    print(f"Label mapping: {label_to_idx}")
    
    # 收集数据
    all_features = []
    all_labels = []
    
    for emotion in emotions:
        print(f"\nProcessing emotion: {emotion}")
        emotion_samples = []
        
        # 收集所有说话人的样本
        for speaker_id in tqdm(speakers, desc=f"  Collecting {emotion} samples"):
            samples = get_sample_with_text(speaker_id, emotion, split, max_samples=None)
            emotion_samples.extend(samples)
        
        if max_samples_per_emotion is not None and len(emotion_samples) > max_samples_per_emotion:
            # 随机采样
            import random
            random.shuffle(emotion_samples)
            emotion_samples = emotion_samples[:max_samples_per_emotion]
        
        print(f"  Total {emotion} samples: {len(emotion_samples)}")
        
        # 提取特征
        emotion_features = []
        for audio_path, text in tqdm(emotion_samples, desc=f"  Extracting {emotion} features"):
            try:
                z = extract_hidden_states_from_audio(
                    model, tokenizer, audio_extractor,
                    audio_path, text,
                    device=device
                )
                if z is not None:
                    emotion_features.append(z)
                    all_labels.append(label_to_idx[emotion])
            except Exception as e:
                print(f"    Error processing {audio_path}: {e}")
                continue
        
        all_features.extend(emotion_features)
        print(f"  Successfully extracted {len(emotion_features)} {emotion} features")
    
    # 转换为 numpy 数组
    if len(all_features) == 0:
        raise ValueError("No features extracted! All samples failed. Check the error messages above.")
    
    Z = np.array(all_features)  # [N, D]
    y = np.array(all_labels)  # [N]
    
    print("\n" + "=" * 80)
    print("Data collection completed!")
    print(f"Total samples: {Z.shape[0]}")
    if Z.shape[0] > 0:
        print(f"Feature dimension: {Z.shape[1]}")
    print(f"Label distribution:")
    for emotion, idx in label_to_idx.items():
        count = np.sum(y == idx)
        print(f"  {emotion}: {count} samples")
    print("=" * 80)
    
    return Z, y, label_to_idx


def train_classifier(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    label_to_idx: dict,
    output_path: str,
    svd_rank: int = 20,
    epochs: int = 100,
    lr: float = 0.01,
    device: str = "cuda:0"
):
    """
    训练分类器并保存 checkpoint
    """
    print("=" * 80)
    print("Training emotion classifier...")
    print("=" * 80)
    
    # 可选：SVD 降维
    if svd_rank is not None and svd_rank < Z_train.shape[1]:
        print(f"Applying SVD: {Z_train.shape[1]} -> {svd_rank}")
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=svd_rank, random_state=42)
        Z_train = svd.fit_transform(Z_train)
        print(f"  Explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}")
        R_global = svd_rank
    else:
        svd = None
        R_global = Z_train.shape[1]
    
    print(f"Training on {Z_train.shape[0]} samples, feature dim: {Z_train.shape[1]}")
    print(f"Number of emotions: {len(label_to_idx)}")
    
    # 训练分类器
    classifier = train_emotion_classifier(
        Z_train, y_train,
        device=device,
        epochs=epochs,
        lr=lr
    )
    
    # 评估
    classifier.eval()
    with torch.no_grad():
        Z_tensor = torch.from_numpy(Z_train).float().to(device)
        y_tensor = torch.from_numpy(y_train).long().to(device)
        logits = classifier(Z_tensor)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y_tensor).float().mean()
        print(f"\nTraining accuracy: {acc.item():.4f}")
    
    # 保存 checkpoint
    checkpoint = {
        'classifier': classifier.state_dict(),
        'label_to_idx': label_to_idx,
        'svd_rank': R_global,
        'feature_dim': Z_train.shape[1],
        'num_emotions': len(label_to_idx),
    }
    
    # 如果使用了 SVD，保存 SVD 参数（但不保存 sklearn 对象，因为加载时可能有问题）
    if svd is not None:
        checkpoint['use_svd'] = True
        checkpoint['svd_components'] = svd.components_  # [R, D]
        checkpoint['svd_mean'] = svd.mean_ if hasattr(svd, 'mean_') else None
    else:
        checkpoint['use_svd'] = False
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_file)
    
    print(f"\n✅ Checkpoint saved to: {output_file}")
    
    # 保存配置信息
    config_file = output_file.with_suffix('.json')
    config = {
        'label_to_idx': label_to_idx,
        'svd_rank': R_global,
        'feature_dim': Z_train.shape[1],
        'num_emotions': len(label_to_idx),
        'num_samples': Z_train.shape[0],
        'training_accuracy': acc.item(),
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Config saved to: {config_file}")
    
    return classifier, checkpoint


def main():
    parser = argparse.ArgumentParser(description="Train Sad/Happy emotion classifier using ESD dataset")
    parser.add_argument("--omnispeech-path", required=True, help="Path to OpenS2S model")
    parser.add_argument("--output", default="emotion_editing_v6/checkpoints/sad_happy_classifier.pt",
                        help="Output checkpoint path")
    parser.add_argument("--emotions", nargs="+", default=["Sad", "Happy"],
                        help="Emotions to classify")
    parser.add_argument("--split", default="train", choices=["train", "evaluation", "test"],
                        help="ESD data split to use")
    parser.add_argument("--max-samples-per-emotion", type=int, default=None,
                        help="Maximum samples per emotion (None for all)")
    parser.add_argument("--max-speakers", type=int, default=None,
                        help="Maximum number of speakers to use (None for all)")
    parser.add_argument("--svd-rank", type=int, default=20,
                        help="SVD rank for dimensionality reduction (None to disable)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--device", default="cuda:0", help="Device")
    
    args = parser.parse_args()
    
    # Step 1: 收集数据并提取特征
    Z, y, label_to_idx = collect_esd_data(
        omnispeech_path=args.omnispeech_path,
        emotions=args.emotions,
        split=args.split,
        max_samples_per_emotion=args.max_samples_per_emotion,
        max_speakers=args.max_speakers,
        device=args.device
    )
    
    # Step 2: 训练分类器
    classifier, checkpoint = train_classifier(
        Z_train=Z,
        y_train=y,
        label_to_idx=label_to_idx,
        output_path=args.output,
        svd_rank=args.svd_rank if args.svd_rank > 0 else None,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device
    )
    
    print("\n" + "=" * 80)
    print("✅ Training completed!")
    print("=" * 80)
    print(f"Checkpoint: {args.output}")
    print(f"Label mapping: {label_to_idx}")
    print("=" * 80)


if __name__ == "__main__":
    main()

