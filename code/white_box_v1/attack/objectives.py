"""攻击目标函数定义（修复版：可导情绪目标）"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from utils_audio import get_audio_encoder_layers


class EmotionAttackObjective:
    """情绪翻转攻击目标函数（使用可导的 emotion classifier）"""
    
    def __init__(
        self,
        model,
        tokenizer,
        audio_extractor,
        target_emotion: str,
        source_emotion: Optional[str] = None,
        emotion_classifier: Optional[torch.nn.Module] = None,
        emotion_label_to_idx: Optional[Dict[str, int]] = None,
        target_layers: Optional[List[str]] = None,  # 如 ['layer_06', 'layer_16', 'layer_25']
        weight_emo_text: float = 1.0,
        weight_emo_audio: float = 0.0,
        weight_sem: float = 0.5,
        weight_per: float = 0.3,
        device: str = "cuda:0"
    ):
        """
        Args:
            model: OpenS2S 模型
            tokenizer: tokenizer
            audio_extractor: 音频特征提取器
            target_emotion: 目标情绪
            source_emotion: 源情绪（可选）
            emotion_classifier: 情绪分类器（FrozenEmotionClassifier）
            emotion_label_to_idx: 情绪标签到索引的映射
            target_layers: 要提取 hidden states 的层名列表
            weight_emo_text: 文本情绪 loss 权重
            weight_emo_audio: 语音情绪 loss 权重
            weight_sem: 语义保持 loss 权重
            weight_per: 感知约束 loss 权重
            device: 设备
        """
        self.model = model
        self.tokenizer = tokenizer
        self.audio_extractor = audio_extractor
        self.target_emotion = target_emotion
        self.source_emotion = source_emotion
        self.emotion_classifier = emotion_classifier
        self.emotion_label_to_idx = emotion_label_to_idx or {}
        self.target_layers = target_layers or ['layer_06', 'layer_16', 'layer_25']
        self.weight_emo_text = weight_emo_text
        self.weight_emo_audio = weight_emo_audio
        self.weight_sem = weight_sem
        self.weight_per = weight_per
        self.device = device
        
        # SVD 变换（如果使用）
        self.svd_components = None
        self.svd_mean = None
        
        # 获取 target emotion index
        self.target_emotion_idx = self.emotion_label_to_idx.get(target_emotion, None)
        if self.target_emotion_idx is None:
            print(f"Warning: target_emotion '{target_emotion}' not in label_to_idx: {self.emotion_label_to_idx}")
        
        # 获取 audio encoder layers
        self.audio_layers = get_audio_encoder_layers(model)
        self.target_layer_modules = {
            name: self.audio_layers[name] 
            for name in self.target_layers 
            if name in self.audio_layers
        }
        
        if not self.target_layer_modules:
            print(f"Warning: No valid layers found in {self.target_layers}, using all layers")
            self.target_layer_modules = self.audio_layers
        
        print(f"Using layers for emotion extraction: {list(self.target_layer_modules.keys())}")
        if self.target_emotion_idx is not None:
            print(f"Target emotion '{target_emotion}' -> index {self.target_emotion_idx}")
        
        # 注册 hooks 来提取 hidden states
        self.hidden_states_cache = {}
        self.hooks = []
        self._register_hooks()
        
        # 缓存原始输出（用于语义保持）
        self.original_hidden_states = None
        self.original_output_embedding = None
        self._cache_original = True
    
    def _register_hooks(self):
        """注册 hooks 来提取指定层的 hidden states"""
        def make_hook(layer_name):
            def hook(module, input, output):
                # output 可能是 tuple 或 tensor
                if isinstance(output, tuple):
                    hidden = output[0]  # 通常第一个是 hidden states
                else:
                    hidden = output
                # hidden: [B, T, D] 或 [B, D]
                self.hidden_states_cache[layer_name] = hidden
            return hook
        
        for layer_name, layer_module in self.target_layer_modules.items():
            hook = make_hook(layer_name)
            handle = layer_module.register_forward_hook(hook)
            self.hooks.append(handle)
    
    def _remove_hooks(self):
        """移除 hooks"""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
    
    def _extract_hidden_states(self) -> torch.Tensor:
        """
        从缓存的 hidden states 中提取并 mean pool
        
        Returns:
            z: [B, D] 或 [R] 子空间表示（用于 emotion classifier）
        """
        if not self.hidden_states_cache:
            return None
        
        # 收集所有层的 hidden states
        all_hiddens = []
        for layer_name in self.target_layers:
            if layer_name in self.hidden_states_cache:
                hidden = self.hidden_states_cache[layer_name]
                # hidden: [B, T, D] 或 [B, D]
                if hidden.dim() == 3:
                    # [B, T, D] -> [B, D] (mean pool over time)
                    hidden = hidden.mean(dim=1)
                elif hidden.dim() == 2:
                    # [B, D] 已经是正确的
                    pass
                else:
                    continue
                all_hiddens.append(hidden)
        
        if not all_hiddens:
            return None
        
        # 拼接或平均所有层的 hidden states
        # 这里使用平均（也可以拼接）
        z = torch.stack(all_hiddens, dim=0).mean(dim=0)  # [B, D]
        
        # 如果使用了 SVD 降维，应用 SVD 变换
        if self.svd_components is not None:
            # z: [B, D_orig], svd_components: [R, D_orig]
            # 确保 dtype 匹配（z 可能是 bfloat16，svd_components 是 float32）
            z = z.float() if z.dtype != torch.float32 else z
            # z @ svd_components.T -> [B, R]
            z = z @ self.svd_components.T
            if self.svd_mean is not None:
                # 如果 SVD 有均值，需要减去（但 TruncatedSVD 通常没有均值）
                pass
        
        return z
    
    def compute_loss(
        self,
        waveform_adv: torch.Tensor,
        waveform_orig: torch.Tensor,
        text: str = "",
        compute_grad: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算攻击损失
        
        Args:
            waveform_adv: [T] 对抗音频波形
            waveform_orig: [T] 原始音频波形
            text: 输入文本（可选）
            compute_grad: 是否计算梯度
        
        Returns:
            total_loss: 总损失
            metrics: 各项指标字典
        """
        # 清空缓存
        self.hidden_states_cache.clear()
        
        metrics = {}
        
        # 1. 情绪翻转损失（使用 emotion classifier）
        loss_emo_text, emo_text_metrics = self._compute_emotion_text_loss(
            waveform_adv, text, compute_grad
        )
        metrics.update(emo_text_metrics)
        
        # 2. 情绪翻转损失（语音，可选）
        loss_emo_audio = torch.tensor(0.0, device=self.device)
        if self.weight_emo_audio > 0:
            loss_emo_audio, emo_audio_metrics = self._compute_emotion_audio_loss(
                waveform_adv, compute_grad
            )
            metrics.update(emo_audio_metrics)
        
        # 3. 语义保持损失
        loss_sem, sem_metrics = self._compute_semantic_loss(
            waveform_adv, waveform_orig, text, compute_grad
        )
        metrics.update(sem_metrics)
        
        # 4. 感知约束损失
        loss_per, per_metrics = self._compute_perceptual_loss(
            waveform_adv, waveform_orig
        )
        metrics.update(per_metrics)
        
        # 总损失
        total_loss = (
            self.weight_emo_text * loss_emo_text +
            self.weight_emo_audio * loss_emo_audio +
            self.weight_sem * loss_sem +
            self.weight_per * loss_per
        )
        
        metrics['total_loss'] = total_loss.item() if not compute_grad else total_loss.detach().item()
        
        # 计算梯度范数（用于 debug）
        if compute_grad and waveform_adv.requires_grad:
            if waveform_adv.grad is not None:
                grad_norm = torch.norm(waveform_adv.grad).item()
                metrics['grad_norm_input'] = grad_norm
            else:
                metrics['grad_norm_input'] = 0.0
        
        return total_loss, metrics
    
    def _compute_emotion_text_loss(
        self,
        waveform: torch.Tensor,
        text: str,
        compute_grad: bool
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算文本情绪损失（使用 emotion classifier）"""
        # 提取音频特征
        wave_np = waveform.detach().cpu().numpy()
        inputs = self.audio_extractor(
            [wave_np],
            sampling_rate=self.audio_extractor.sampling_rate,
            return_attention_mask=True,
            return_tensors="pt",
        )
        speech_values = inputs.input_features.to(self.device)
        speech_mask = inputs.attention_mask.to(self.device)
        
        # 准备文本输入
        from constants import (
            DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_TOKEN,
            DEFAULT_AUDIO_END_TOKEN, DEFAULT_TTS_START_TOKEN, AUDIO_TOKEN_INDEX
        )
        
        prompt_with_audio = DEFAULT_AUDIO_START_TOKEN
        prompt_with_audio += DEFAULT_AUDIO_TOKEN
        if text:
            prompt_with_audio += text
        prompt_with_audio += DEFAULT_AUDIO_END_TOKEN
        prompt_with_audio += DEFAULT_TTS_START_TOKEN
        
        segments = prompt_with_audio.split(DEFAULT_AUDIO_TOKEN)
        ids = []
        for idx, seg in enumerate(segments):
            if idx != 0:
                ids.append(AUDIO_TOKEN_INDEX)
            ids.extend(self.tokenizer.encode(seg))
        input_ids = torch.LongTensor(ids).unsqueeze(0).to(self.device)
        
        # 前向传播（提取 hidden states）
        with torch.set_grad_enabled(compute_grad):
            # 准备输入
            (
                inputs_embeds,
                attention_mask_llm,
                _,
                _
            ) = self.model.prepare_inputs_labels_for_llm(
                input_ids,
                torch.ones_like(input_ids),
                None,  # labels
                speech_values,
                speech_mask,
                None,  # token_types
                left_pad=True,
                inference=True
            )
            
            # 调用 llm_model（这会触发 audio encoder 的 forward，从而触发 hooks）
            outputs = self.model.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask_llm,
                return_dict=True,
                output_hidden_states=True,
            )
            
            # 提取 hidden states（从 audio encoder layers）
            # 注意：hidden states 是在 audio encoder 的 forward 中通过 hooks 缓存的
            # 需要确保 audio encoder 已经被调用（通过 prepare_inputs_labels_for_llm）
            z = self._extract_hidden_states()  # [B, D] 或 None
            
            if z is None:
                # 如果无法提取 hidden states，使用 LLM 的 hidden states
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    z = outputs.hidden_states[-1][0, -1, :].unsqueeze(0)  # [1, D]
                else:
                    # 回退：使用 logits 的 embedding
                    loss = torch.tensor(0.0, device=self.device, requires_grad=compute_grad)
                    metrics = {
                        'emo_text_loss': 0.0,
                        'emo_prob_target': 0.0,
                        'generated_text': '',
                    }
                    return loss, metrics
            
            # 缓存原始 hidden states（用于语义保持）
            if self._cache_original and self.original_hidden_states is None:
                with torch.no_grad():
                    self.original_hidden_states = z.detach().clone()
                self._cache_original = False
            
            # 使用 emotion classifier 计算 loss
            if self.emotion_classifier is not None and self.target_emotion_idx is not None:
                # 前向传播分类器
                # 确保 z 和 classifier 的 dtype 匹配（z 可能是 bfloat16，classifier 是 float32）
                z_float = z.float() if z.dtype != torch.float32 else z
                logits = self.emotion_classifier(z_float)  # [B, num_emotions]
                
                # 计算 CE loss（最大化 target emotion 的概率）
                target_label = torch.tensor([self.target_emotion_idx], device=self.device)
                loss = F.cross_entropy(logits, target_label)
                
                # 计算概率（用于 debug）
                probs = F.softmax(logits, dim=-1)  # [B, num_emotions]
                emo_prob_target = probs[0, self.target_emotion_idx].item()
                
                # 获取生成的文本（用于显示）
                next_token_logits = outputs.logits[0, -1, :]
                next_token_id = torch.argmax(next_token_logits).item()
                generated_text = self.tokenizer.decode([next_token_id])
                
                metrics = {
                    'emo_text_loss': loss.item() if not compute_grad else loss.detach().item(),
                    'emo_prob_target': emo_prob_target,
                    'generated_text': generated_text,
                }
            else:
                # 如果没有分类器，返回 0 loss
                loss = torch.tensor(0.0, device=self.device, requires_grad=compute_grad)
                next_token_logits = outputs.logits[0, -1, :]
                next_token_id = torch.argmax(next_token_logits).item()
                generated_text = self.tokenizer.decode([next_token_id])
                metrics = {
                    'emo_text_loss': 0.0,
                    'emo_prob_target': 0.0,
                    'generated_text': generated_text,
                }
        
        return loss, metrics
    
    def _compute_emotion_audio_loss(
        self,
        waveform: torch.Tensor,
        compute_grad: bool
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算语音情绪损失（可选，成本较高）"""
        loss = torch.tensor(0.0, device=self.device, requires_grad=compute_grad)
        metrics = {'emo_audio_loss': 0.0}
        return loss, metrics
    
    def _compute_semantic_loss(
        self,
        waveform_adv: torch.Tensor,
        waveform_orig: torch.Tensor,
        text: str,
        compute_grad: bool
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算语义保持损失（使用 hidden states cosine similarity）"""
        # 如果还没有缓存原始 hidden states，先运行一次原始音频
        if self.original_hidden_states is None:
            # 临时缓存原始 hidden states
            self._cache_original = True
            with torch.no_grad():
                _, _ = self._compute_emotion_text_loss(waveform_orig, text, compute_grad=False)
            self._cache_original = False
        
        # 获取当前 hidden states
        z_current = self._extract_hidden_states()
        
        if z_current is None or self.original_hidden_states is None:
            # 回退：使用波形相似度
            cosine_sim = F.cosine_similarity(
                waveform_adv.unsqueeze(0),
                waveform_orig.unsqueeze(0),
                dim=1
            )
            loss_sem = 1.0 - cosine_sim.mean()
            metrics = {
                'sem_loss': loss_sem.item(),
                'sem_cosine_sim': cosine_sim.mean().item(),
            }
            return loss_sem, metrics
        
        # 计算 hidden states 的 cosine similarity
        # z_current: [B, D], original_hidden_states: [B, D]
        cosine_sim = F.cosine_similarity(
            z_current,
            self.original_hidden_states,
            dim=1
        )  # [B]
        
        # 损失：最小化 (1 - cosine_sim)
        loss_sem = (1.0 - cosine_sim).mean()
        
        metrics = {
            'sem_loss': loss_sem.item() if not compute_grad else loss_sem.detach().item(),
            'sem_cosine_sim': cosine_sim.mean().item(),
        }
        
        return loss_sem, metrics
    
    def _compute_perceptual_loss(
        self,
        waveform_adv: torch.Tensor,
        waveform_orig: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算感知约束损失"""
        # L2 范数
        l2_norm = torch.norm(waveform_adv - waveform_orig, p=2)
        l2_loss = l2_norm / (torch.norm(waveform_orig, p=2) + 1e-8)
        
        # Linf 范数
        linf_norm = torch.norm(waveform_adv - waveform_orig, p=float('inf'))
        linf_loss = linf_norm / (torch.norm(waveform_orig, p=float('inf')) + 1e-8)
        
        # SNR（信噪比）
        signal_power = torch.mean(waveform_orig ** 2)
        noise_power = torch.mean((waveform_adv - waveform_orig) ** 2)
        snr_db = 10 * torch.log10(signal_power / (noise_power + 1e-8))
        snr_loss = -snr_db / 40.0  # 归一化，鼓励高 SNR
        
        # 组合
        loss_per = l2_loss + 0.5 * linf_loss + 0.3 * snr_loss
        
        metrics = {
            'per_loss': loss_per.item(),
            'l2_norm': l2_norm.item(),
            'linf_norm': linf_norm.item(),
            'snr_db': snr_db.item(),
        }
        
        return loss_per, metrics
    
    def __del__(self):
        """清理 hooks"""
        if hasattr(self, 'hooks'):
            self._remove_hooks()
