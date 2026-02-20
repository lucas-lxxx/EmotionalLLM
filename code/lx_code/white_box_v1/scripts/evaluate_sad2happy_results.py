"""评估 sad2happy 批量实验结果（根据 README.md 要求）"""

import argparse
import sys
import os
from pathlib import Path
import csv
import json
import numpy as np
import re
from tqdm import tqdm
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import soundfile as sf
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 尝试导入 sentence_transformers，如果失败则使用 transformers 的替代方案
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("⚠️  Warning: sentence_transformers not found, will use transformers-based semantic similarity")


def extract_emotion_from_text(text: str) -> Tuple[str, Dict[str, float]]:
    """
    从 OpenS2S 的输出文本中提取情绪标签
    假设 prompt 要求模型输出音频的情绪，我们从文本中查找情绪关键词
    
    Returns:
        predicted_emotion: 提取的情绪标签
        emotion_scores: 各情绪关键词的匹配分数（基于关键词出现）
    """
    text_lower = text.lower().strip()
    
    # 定义情绪关键词及其权重
    emotion_keywords = {
        'happy': ['happy', 'happiness', 'joy', 'joyful', 'cheerful', 'glad', 'pleased', 'delighted'],
        'sad': ['sad', 'sadness', 'sorrow', 'unhappy', 'depressed', 'melancholy', 'gloomy'],
        'neutral': ['neutral', 'calm', 'normal', 'flat', 'emotionless'],
        'angry': ['angry', 'anger', 'mad', 'furious', 'irritated'],
        'fear': ['fear', 'afraid', 'scared', 'frightened', 'anxious'],
        'surprise': ['surprise', 'surprised', 'shocked', 'amazed'],
        'disgust': ['disgust', 'disgusted', 'revolted']
    }
    
    # 计算每个情绪的匹配分数
    emotion_scores = {}
    for emotion, keywords in emotion_keywords.items():
        score = 0.0
        for keyword in keywords:
            # 计算关键词出现次数（考虑词边界）
            pattern = r'\b' + re.escape(keyword) + r'\b'
            matches = len(re.findall(pattern, text_lower))
            score += matches * (1.0 / len(keywords))  # 归一化权重
        
        # 如果情绪词出现在文本开头或单独出现，给予更高权重
        if any(text_lower.startswith(kw) for kw in keywords):
            score += 0.5
        if any(text_lower == kw for kw in keywords):
            score += 1.0
        
        emotion_scores[emotion] = score
    
    # 找到得分最高的情绪
    if max(emotion_scores.values()) > 0:
        predicted_emotion = max(emotion_scores, key=emotion_scores.get)
    else:
        # 如果没有找到明确的情绪，返回 neutral
        predicted_emotion = 'neutral'
        emotion_scores['neutral'] = 0.1
    
    # 归一化分数为概率（用于兼容性）
    total_score = sum(emotion_scores.values())
    if total_score > 0:
        emotion_probs = {k: v / total_score for k, v in emotion_scores.items()}
    else:
        emotion_probs = {k: 0.0 for k in emotion_scores.keys()}
        emotion_probs[predicted_emotion] = 1.0
    
    return predicted_emotion, emotion_probs


def load_semantic_model(device: str = "cuda:0"):
    """加载语义相似度模型（Sentence-BERT 或 transformers 替代方案）"""
    print("Loading semantic similarity model...")
    
    if HAS_SENTENCE_TRANSFORMERS:
        try:
            # 使用 all-MiniLM-L6-v2 或 all-mpnet-base-v2
            model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
            print("✅ Semantic model loaded: all-MiniLM-L6-v2 (Sentence-BERT)")
            return model
        except Exception as e:
            print(f"⚠️  Warning: Failed to load Sentence-BERT: {e}")
            print("   Trying alternative model...")
            try:
                model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)
                print("✅ Semantic model loaded: paraphrase-MiniLM-L6-v2 (Sentence-BERT)")
                return model
            except Exception as e2:
                print(f"⚠️  Warning: Failed to load Sentence-BERT models: {e2}")
                print("   Falling back to transformers-based approach...")
    
    # 使用 transformers 的替代方案
    try:
        from transformers import AutoModel
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()
        print(f"✅ Semantic model loaded: {model_name} (transformers)")
        return {'tokenizer': tokenizer, 'model': model, 'device': device, 'use_transformers': True}
    except Exception as e:
        print(f"❌ Error: Failed to load semantic model: {e}")
        print("   Please install sentence_transformers: pip install sentence-transformers")
        raise


def compute_semantic_similarity(text1: str, text2: str, model) -> float:
    """计算两个文本的语义相似度（cosine similarity）"""
    if HAS_SENTENCE_TRANSFORMERS and isinstance(model, SentenceTransformer):
        # 使用 SentenceTransformer
        embeddings = model.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(similarity)
    elif isinstance(model, dict) and model.get('use_transformers', False):
        # 使用 transformers 的替代方案
        tokenizer = model['tokenizer']
        encoder = model['model']
        device = model['device']
        
        # 编码两个文本
        def encode_text(text):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
            with torch.no_grad():
                outputs = encoder(**inputs)
                # 使用 mean pooling
                embeddings = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']
                # 应用 attention mask 并平均
                masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
                sum_embeddings = masked_embeddings.sum(dim=1)
                sum_mask = attention_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
                mean_embeddings = sum_embeddings / sum_mask
                return mean_embeddings[0].cpu().numpy()
        
        emb1 = encode_text(text1)
        emb2 = encode_text(text2)
        
        similarity = np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8
        )
        return float(similarity)
    else:
        raise ValueError("Unknown model type for semantic similarity")


def compute_audio_metrics(audio_clean_path: str, audio_adv_path: str) -> Tuple[float, float]:
    """
    计算音频扰动指标
    
    Returns:
        l2: L2范数
        snr: 信噪比（dB）
    """
    # 加载音频
    audio_clean, sr_clean = sf.read(audio_clean_path)
    audio_adv, sr_adv = sf.read(audio_adv_path)
    
    # 确保采样率一致
    if sr_clean != sr_adv:
        raise ValueError(f"Sample rate mismatch: {sr_clean} vs {sr_adv}")
    
    # 确保长度一致
    min_len = min(len(audio_clean), len(audio_adv))
    audio_clean = audio_clean[:min_len]
    audio_adv = audio_adv[:min_len]
    
    # 转换为 tensor
    audio_clean_t = torch.from_numpy(audio_clean).float()
    audio_adv_t = torch.from_numpy(audio_adv).float()
    
    # 计算扰动
    perturbation = audio_adv_t - audio_clean_t
    
    # L2范数
    l2 = torch.norm(perturbation, p=2).item()
    
    # SNR (dB)
    signal_power = torch.mean(audio_clean_t ** 2).item()
    noise_power = torch.mean(perturbation ** 2).item()
    if noise_power > 0:
        snr = 10 * np.log10(signal_power / noise_power)
    else:
        snr = float('inf')
    
    return l2, snr


def evaluate_batch_results(
    results_csv_path: str,
    output_dir: str,
    device: str = "cuda:0",
    recompute_audio_metrics: bool = False
):
    """
    评估批量实验结果
    
    Args:
        results_csv_path: results.csv 路径
        output_dir: 输出目录
        device: 设备
        recompute_audio_metrics: 是否重新计算音频指标（如果results.csv已有则跳过）
    """
    print("=" * 80)
    print("Evaluating sad2happy Batch Experiment Results")
    print("=" * 80)
    
    # 读取 results.csv
    print(f"\n[1/5] Reading results.csv...")
    results = []
    with open(results_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    
    print(f"  Total samples: {len(results)}")
    
    # 加载模型（不再需要情绪分类器，直接从文本提取）
    print(f"\n[2/5] Loading evaluation models...")
    print("  Note: Using emotion extraction from OpenS2S output text (not external classifier)")
    sem_model = load_semantic_model(device)
    
    # 评估每条样本
    print(f"\n[3/5] Evaluating samples...")
    eval_results = []
    
    for result in tqdm(results, desc="Evaluating"):
        sample_id = result['sample_id']
        eval_result = {
            'sample_id': sample_id,
        }
        
        # 读取文本
        t_clean_path = result.get('t_clean_path', '')
        t_adv_path = result.get('t_adv_path', '')
        
        if not t_clean_path or not t_adv_path:
            print(f"  ⚠️  Warning: Missing text paths for sample {sample_id}")
            continue
        
        try:
            # 读取文本内容
            with open(t_clean_path, 'r', encoding='utf-8') as f:
                t_clean = f.read().strip()
            with open(t_adv_path, 'r', encoding='utf-8') as f:
                t_adv = f.read().strip()
            
            # Part I: 情绪评估（从 OpenS2S 输出文本中提取情绪）
            emo_clean, probs_clean = extract_emotion_from_text(t_clean)
            emo_adv, probs_adv = extract_emotion_from_text(t_adv)
            
            # 提取情绪概率
            # 获取所有可能的情绪标签
            all_emotions = set(list(probs_clean.keys()) + list(probs_adv.keys()))
            
            # 提取 Sad 和 Happy 相关的概率（从提取的情绪概率中获取）
            p_sad_clean = probs_clean.get('sad', 0.0)
            p_happy_clean = probs_clean.get('happy', 0.0)
            p_sad_adv = probs_adv.get('sad', 0.0)
            p_happy_adv = probs_adv.get('happy', 0.0)
            
            delta_happy = p_happy_adv - p_happy_clean
            delta_sad = p_sad_adv - p_sad_clean  # 应该是负数（sad减少）
            
            # 判断情绪翻转：基于 OpenS2S 输出的音频情绪
            # 成功翻转：clean 输出 sad，adv 输出 happy
            # 或者：adv 的 happy 概率明显提升，sad 概率明显下降
            emotion_flip = 0
            if emo_clean.lower() in ['sad', 'sadness'] and emo_adv.lower() in ['happy', 'joy']:
                # 明确的翻转：sad -> happy
                emotion_flip = 1
            elif p_happy_adv > p_sad_adv and delta_happy > 0 and delta_sad < 0:
                # 概率提升：happy 增加，sad 减少
                emotion_flip = 1
            elif emo_adv.lower() in ['happy', 'joy'] and delta_happy > 0.1:
                # adv 输出 happy 且概率有明显提升
                emotion_flip = 1
            
            eval_result.update({
                'emo_clean': emo_clean,
                'emo_adv': emo_adv,
                'p_happy_clean': p_happy_clean,
                'p_happy_adv': p_happy_adv,
                'delta_happy': delta_happy,
                'emotion_flip': emotion_flip,
            })
            
            # Part II: 语义相似度
            semantic_sim = compute_semantic_similarity(t_clean, t_adv, sem_model)
            eval_result['semantic_sim'] = semantic_sim
            
            # Part III: 音频扰动指标
            if recompute_audio_metrics or not result.get('l2') or not result.get('snr'):
                audio_clean_path = result.get('audio_clean_path', '')
                audio_adv_path = result.get('audio_adv_path', '')
                
                if audio_clean_path and audio_adv_path and os.path.exists(audio_clean_path) and os.path.exists(audio_adv_path):
                    try:
                        l2, snr = compute_audio_metrics(audio_clean_path, audio_adv_path)
                        eval_result['l2'] = l2
                        eval_result['snr'] = snr
                    except Exception as e:
                        print(f"  ⚠️  Warning: Failed to compute audio metrics for {sample_id}: {e}")
                        eval_result['l2'] = result.get('l2', 0.0)
                        eval_result['snr'] = result.get('snr', 0.0)
                else:
                    eval_result['l2'] = result.get('l2', 0.0)
                    eval_result['snr'] = result.get('snr', 0.0)
            else:
                eval_result['l2'] = float(result.get('l2', 0.0))
                eval_result['snr'] = float(result.get('snr', 0.0))
            
            eval_results.append(eval_result)
            
        except Exception as e:
            print(f"  ⚠️  Error processing sample {sample_id}: {e}")
            continue
    
    # 保存评估结果
    print(f"\n[4/5] Saving evaluation results...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    eval_csv_path = output_path / "results_eval.csv"
    if eval_results:
        fieldnames = [
            'sample_id', 'emo_clean', 'emo_adv',
            'p_happy_clean', 'p_happy_adv', 'delta_happy', 'emotion_flip',
            'semantic_sim', 'l2', 'snr'
        ]
        with open(eval_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(eval_results)
        print(f"  ✅ Saved: {eval_csv_path}")
    
    # 计算统计信息
    print(f"\n[5/5] Computing statistics...")
    stats = compute_statistics(eval_results)
    
    # 保存统计摘要
    stats_path = output_path / "stats_summary.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Saved: {stats_path}")
    
    # 打印统计摘要
    print("\n" + "=" * 80)
    print("Evaluation Statistics Summary")
    print("=" * 80)
    print_statistics(stats)
    print("=" * 80)
    
    return eval_results, stats


def compute_statistics(eval_results: List[Dict]) -> Dict:
    """计算统计信息"""
    if not eval_results:
        return {}
    
    # 提取所有指标
    emotion_flips = [r['emotion_flip'] for r in eval_results]
    delta_happies = [r['delta_happy'] for r in eval_results]
    semantic_sims = [r['semantic_sim'] for r in eval_results]
    snrs = [r['snr'] for r in eval_results if not np.isinf(r['snr'])]
    l2s = [r['l2'] for r in eval_results]
    
    # 整体统计
    stats = {
        'overall': {
            'total_samples': len(eval_results),
            'emotion_flip_rate': np.mean(emotion_flips),
            'delta_happy_mean': np.mean(delta_happies),
            'delta_happy_median': np.median(delta_happies),
            'semantic_sim_mean': np.mean(semantic_sims),
            'semantic_sim_median': np.median(semantic_sims),
            'snr_mean': np.mean(snrs) if snrs else None,
            'snr_median': np.median(snrs) if snrs else None,
            'l2_mean': np.mean(l2s),
            'l2_median': np.median(l2s),
        }
    }
    
    # 条件统计：semantic_sim >= 0.85
    high_sem_samples = [r for r in eval_results if r['semantic_sim'] >= 0.85]
    if high_sem_samples:
        high_sem_flips = [r['emotion_flip'] for r in high_sem_samples]
        stats['conditional_semantic_high'] = {
            'count': len(high_sem_samples),
            'emotion_flip_rate': np.mean(high_sem_flips),
            'delta_happy_mean': np.mean([r['delta_happy'] for r in high_sem_samples]),
        }
    
    # 条件统计：snr >= 30 dB
    high_snr_samples = [r for r in eval_results if not np.isinf(r['snr']) and r['snr'] >= 30.0]
    if high_snr_samples:
        high_snr_flips = [r['emotion_flip'] for r in high_snr_samples]
        stats['conditional_snr_high'] = {
            'count': len(high_snr_samples),
            'emotion_flip_rate': np.mean(high_snr_flips),
            'delta_happy_mean': np.mean([r['delta_happy'] for r in high_snr_samples]),
        }
    
    # 双重条件：semantic_sim >= 0.85 AND snr >= 30 dB
    high_quality_samples = [
        r for r in eval_results
        if r['semantic_sim'] >= 0.85 and not np.isinf(r['snr']) and r['snr'] >= 30.0
    ]
    if high_quality_samples:
        high_quality_flips = [r['emotion_flip'] for r in high_quality_samples]
        stats['conditional_high_quality'] = {
            'count': len(high_quality_samples),
            'emotion_flip_rate': np.mean(high_quality_flips),
            'delta_happy_mean': np.mean([r['delta_happy'] for r in high_quality_samples]),
        }
    
    return stats


def print_statistics(stats: Dict):
    """打印统计信息"""
    overall = stats.get('overall', {})
    print(f"\n📊 Overall Statistics:")
    print(f"  Total samples: {overall.get('total_samples', 0)}")
    print(f"  Emotion Flip Rate (EFR): {overall.get('emotion_flip_rate', 0):.2%}")
    print(f"  Mean delta_happy: {overall.get('delta_happy_mean', 0):.4f}")
    print(f"  Median delta_happy: {overall.get('delta_happy_median', 0):.4f}")
    print(f"  Mean semantic_sim: {overall.get('semantic_sim_mean', 0):.4f}")
    print(f"  Median semantic_sim: {overall.get('semantic_sim_median', 0):.4f}")
    print(f"  Mean SNR: {overall.get('snr_mean', 0):.2f} dB")
    print(f"  Median SNR: {overall.get('snr_median', 0):.2f} dB")
    
    if 'conditional_semantic_high' in stats:
        cond = stats['conditional_semantic_high']
        print(f"\n📊 Conditional Statistics (semantic_sim >= 0.85):")
        print(f"  Count: {cond['count']}")
        print(f"  EFR: {cond['emotion_flip_rate']:.2%}")
        print(f"  Mean delta_happy: {cond['delta_happy_mean']:.4f}")
    
    if 'conditional_snr_high' in stats:
        cond = stats['conditional_snr_high']
        print(f"\n📊 Conditional Statistics (SNR >= 30 dB):")
        print(f"  Count: {cond['count']}")
        print(f"  EFR: {cond['emotion_flip_rate']:.2%}")
        print(f"  Mean delta_happy: {cond['delta_happy_mean']:.4f}")
    
    if 'conditional_high_quality' in stats:
        cond = stats['conditional_high_quality']
        print(f"\n📊 Conditional Statistics (semantic_sim >= 0.85 AND SNR >= 30 dB):")
        print(f"  Count: {cond['count']}")
        print(f"  EFR: {cond['emotion_flip_rate']:.2%}")
        print(f"  Mean delta_happy: {cond['delta_happy_mean']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate sad2happy batch experiment results")
    parser.add_argument("--results-csv", required=True,
                        help="Path to results.csv from batch experiment")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for evaluation results")
    parser.add_argument("--device", default="cuda:0", help="Device")
    parser.add_argument("--recompute-audio", action="store_true",
                        help="Recompute audio metrics even if already in results.csv")
    
    args = parser.parse_args()
    
    evaluate_batch_results(
        results_csv_path=args.results_csv,
        output_dir=args.output_dir,
        device=args.device,
        recompute_audio_metrics=args.recompute_audio
    )


if __name__ == "__main__":
    main()

