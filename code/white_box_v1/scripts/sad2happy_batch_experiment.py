"""sad2happy批量实验：完整pipeline（数据准备 + Clean推理 + Attack + Attack推理）"""

import argparse
import sys
import os
from pathlib import Path
import json
import csv
import time
import subprocess
from typing import Dict, List, Optional
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import soundfile as sf
from transformers import GenerationConfig

from utils_audio import load_model, load_audio_extractor, load_waveform
from attack.transforms import create_default_eot_transform
from attack.objectives import EmotionAttackObjective
from attack.optimizers.pgd import PGD
from utils.emotion_classifier import FrozenEmotionClassifier
from constants import (
    DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_TOKEN,
    DEFAULT_AUDIO_END_TOKEN, DEFAULT_TTS_START_TOKEN, AUDIO_TOKEN_INDEX
)


def prepare_data(
    data_root: str = "/data3/xuzhenyu/OpenS2S/data/en_query_wav/",
    output_dir: str = "/data3/xuzhenyu/OpenS2S/exp/sad2happy_batch_v1/",
    seed: int = 2025
):
    """准备数据：生成sad_list.txt并进行80/20划分"""
    print("=" * 80)
    print("Step 1: Preparing data...")
    print("=" * 80)
    
    # 调用数据准备脚本
    script_path = Path(__file__).parent.parent / "scripts" / "sad2happy_batch_data_prep.py"
    cmd = [
        "python", str(script_path),
        "--data-root", data_root,
        "--output-dir", output_dir,
        "--seed", str(seed)
    ]
    subprocess.run(cmd, check=True)
    
    output_path = Path(output_dir)
    return {
        'sad_test': output_path / "sad_test.txt"
    }


def clean_inference(
    model, tokenizer, audio_extractor,
    audio_path: str,
    prompt: str,
    device: str = "cuda:0"
) -> str:
    """Clean推理：输入原始音频，得到文本回复"""
    # 准备输入
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
    
    # 构建prompt
    prompt_with_audio = DEFAULT_AUDIO_START_TOKEN + DEFAULT_AUDIO_TOKEN
    if prompt:
        prompt_with_audio += prompt
    prompt_with_audio += DEFAULT_AUDIO_END_TOKEN + DEFAULT_TTS_START_TOKEN
    
    segments = prompt_with_audio.split(DEFAULT_AUDIO_TOKEN)
    ids = []
    for idx, seg in enumerate(segments):
        if idx != 0:
            ids.append(AUDIO_TOKEN_INDEX)
        ids.extend(tokenizer.encode(seg))
    input_ids = torch.LongTensor(ids).unsqueeze(0).to(device)
    
    # 生成配置（使用默认配置）
    generation_config = GenerationConfig(
        max_new_tokens=512,
        do_sample=True,
        temperature=0.85,
        top_p=0.92,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    
    # 生成
    with torch.no_grad():
        out_ids = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            speech_values=speech_values,
            speech_mask=speech_mask,
            spk_emb=None,
            generation_config=generation_config,
        )
    
    in_len = input_ids.shape[1]
    gen_ids = out_ids[0, in_len:] if out_ids.shape[1] > in_len else out_ids[0]
    generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    
    return generated_text


def run_attack(
    model, tokenizer, audio_extractor,
    audio_path: str,
    prompt: str,
    target_emotion: str,
    source_emotion: str,
    emotion_classifier: Optional[torch.nn.Module],
    emotion_label_to_idx: Optional[Dict[str, int]],
    svd_components: Optional[torch.Tensor] = None,
    svd_mean: Optional[torch.Tensor] = None,
    epsilon: float = 0.002,
    steps: int = 30,
    alpha: float = None,
    lambda_emo: float = 1.0,
    lambda_sem: float = 1e-2,
    lambda_per: float = 1e-4,
    device: str = "cuda:0"
) -> tuple:
    """
    运行攻击，返回对抗音频和指标
    
    Returns:
        waveform_adv: 对抗音频
        metrics: 攻击指标（包含linf, l2, snr等）
        attack_time: 攻击时间
    """
    if alpha is None:
        alpha = epsilon / 10.0
    
    # 加载音频
    waveform_orig, sample_rate = load_waveform(audio_path)
    waveform_orig = waveform_orig.to(device)
    
    # 创建目标函数
    objective = EmotionAttackObjective(
        model=model,
        tokenizer=tokenizer,
        audio_extractor=audio_extractor,
        target_emotion=target_emotion,
        source_emotion=source_emotion,
        emotion_classifier=emotion_classifier,
        emotion_label_to_idx=emotion_label_to_idx,
        target_layers=['layer_06', 'layer_16', 'layer_25'],
        weight_emo_text=lambda_emo,
        weight_sem=lambda_sem,
        weight_per=lambda_per,
        device=device
    )
    
    # 设置 SVD 变换（如果存在）
    if svd_components is not None:
        objective.svd_components = svd_components
        objective.svd_mean = svd_mean
    
    # 创建优化器
    attacker = PGD(
        objective_fn=lambda w, w_orig, compute_grad=True: objective.compute_loss(w, w_orig, prompt, compute_grad),
        eps=epsilon,
        alpha=alpha,
        steps=steps,
        eot_k=1,  # 不使用EOT
        norm="Linf",
        device=device
    )
    
    # 执行攻击
    start_time = time.time()
    waveform_adv, metrics_history = attacker.attack(
        waveform_orig=waveform_orig,
        eot_transform=None
    )
    attack_time = time.time() - start_time
    
    # 计算扰动指标
    perturbation = waveform_adv - waveform_orig
    linf = torch.max(torch.abs(perturbation)).item()
    l2 = torch.norm(perturbation, p=2).item()
    
    # 计算SNR
    signal_power = torch.mean(waveform_orig ** 2).item()
    noise_power = torch.mean(perturbation ** 2).item()
    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-8))
    
    final_metrics = metrics_history[-1] if metrics_history else {}
    final_metrics.update({
        'linf': linf,
        'l2': l2,
        'snr': snr_db,
        'attack_time': attack_time
    })
    
    return waveform_adv, final_metrics, attack_time, sample_rate


def compute_sample_id(audio_path: str) -> str:
    """从音频路径生成sample_id（使用文件名，不含扩展名）"""
    return Path(audio_path).stem


def run_batch_experiment(
    omnispeech_path: str,
    checkpoint_path: Optional[str] = None,
    data_root: str = "/data3/xuzhenyu/OpenS2S/data/en_query_wav/",
    output_dir: str = "/data3/xuzhenyu/OpenS2S/exp/sad2happy_batch_v1/",
    prompt: str = "What is the emotion of this audio? Please answer with only the emotion label (e.g., happy, sad, neutral).",
    epsilon: float = 0.002,
    steps: int = 30,
    alpha: float = None,
    lambda_emo: float = 1.0,
    lambda_sem: float = 1e-2,
    lambda_per: float = 1e-4,
    device: str = "cuda:0",
    seed: int = 2025
):
    """
    运行完整的批量实验
    
    Args:
        omnispeech_path: OpenS2S模型路径
        checkpoint_path: emotion classifier checkpoint路径（可选）
        data_root: 数据根目录
        output_dir: 输出目录
        prompt: 固定prompt
        epsilon: 扰动上界
        steps: 攻击步数
        alpha: 步长（如果为None，则使用epsilon/10）
        lambda_emo: 情绪loss权重
        lambda_sem: 语义loss权重
        lambda_per: 感知loss权重
        device: 设备
        seed: 随机种子
    """
    if alpha is None:
        alpha = epsilon / 10.0
    
    print("=" * 80)
    print("sad2happy Batch Experiment")
    print("=" * 80)
    print(f"Data root: {data_root}")
    print(f"Output dir: {output_dir}")
    print(f"Prompt: {prompt}")
    print(f"Attack params: epsilon={epsilon}, steps={steps}, alpha={alpha}")
    print(f"Loss weights: lambda_emo={lambda_emo}, lambda_sem={lambda_sem}, lambda_per={lambda_per}")
    print("=" * 80)
    
    # Step 1: 准备数据
    data_files = prepare_data(data_root, output_dir, seed)
    sad_test_file = data_files['sad_test']
    
    # 读取测试样本列表
    with open(sad_test_file, 'r', encoding='utf-8') as f:
        audio_paths = [line.strip() for line in f if line.strip()]
    
    print(f"\nTotal test samples: {len(audio_paths)}")
    
    # Step 2: 加载模型
    print("\n" + "=" * 80)
    print("Step 2: Loading models...")
    print("=" * 80)
    model, tokenizer = load_model(omnispeech_path, device=device)
    model.eval()
    audio_extractor = load_audio_extractor(omnispeech_path)
    
    # 加载emotion classifier（如果提供）
    emotion_classifier = None
    emotion_label_to_idx = None
    svd_components = None
    svd_mean = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading emotion classifier from: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            classifier_state = checkpoint.get('classifier', None)
            emotion_label_to_idx = checkpoint.get('label_to_idx', None)
            
            if classifier_state is not None and emotion_label_to_idx is not None:
                R_global = checkpoint.get('svd_rank', 20)
                if isinstance(classifier_state, dict) and 'linear.weight' in classifier_state:
                    saved_R = classifier_state['linear.weight'].shape[1]
                    R_global = saved_R
                
                num_emotions = len(emotion_label_to_idx)
                emotion_classifier = FrozenEmotionClassifier(R_global, num_emotions).to(device)
                emotion_classifier.load_state_dict(classifier_state)
                emotion_classifier.freeze()
                emotion_classifier.eval()
                
                # 加载 SVD 变换（如果存在）
                svd_components_np = checkpoint.get('svd_components', None)
                svd_mean_np = checkpoint.get('svd_mean', None)
                
                print(f"✅ Emotion classifier loaded: R={R_global}, num_emotions={num_emotions}")
                print(f"   Label mapping: {emotion_label_to_idx}")
                if svd_components_np is not None:
                    print(f"   SVD components loaded: {svd_components_np.shape}")
                    # 转换为 torch tensor 并移到 device
                    svd_components = torch.from_numpy(svd_components_np).float().to(device)
                    if svd_mean_np is not None:
                        svd_mean = torch.from_numpy(svd_mean_np).float().to(device)
                else:
                    print("   No SVD components found, using raw features")
        except Exception as e:
            print(f"⚠️  Warning: Failed to load emotion classifier: {e}")
    else:
        print("⚠️  Warning: No checkpoint path provided, emotion classifier will be None")
    
    # Step 3: 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    audio_clean_dir = output_path / "audio" / "clean"
    audio_adv_dir = output_path / "audio" / "adv"
    text_clean_dir = output_path / "text" / "clean"
    text_adv_dir = output_path / "text" / "adv"
    
    for d in [audio_clean_dir, audio_adv_dir, text_clean_dir, text_adv_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Step 4: 批量处理
    print("\n" + "=" * 80)
    print("Step 3: Running batch experiment...")
    print("=" * 80)
    
    results = []
    target_emotion = "Happy"
    source_emotion = "Sad"
    
    for idx, audio_path in enumerate(tqdm(audio_paths, desc="Processing samples")):
        sample_id = compute_sample_id(audio_path)
        result = {
            'sample_id': sample_id,
            'src_path': audio_path,
            'split': 'test',
            'prompt': prompt,
            'epsilon': epsilon,
            'steps': steps,
            'alpha': alpha,
            'lambda_emo': lambda_emo,
            'lambda_sem': lambda_sem,
            'lambda_per': lambda_per,
            'status': 'ok',
            'error_msg': '',
            'runtime_sec': 0.0,
        }
        
        try:
            start_time = time.time()
            
            # Group 0: Clean推理
            t_clean = clean_inference(
                model, tokenizer, audio_extractor,
                audio_path, prompt, device
            )
            
            t_clean_path = text_clean_dir / f"{sample_id}.txt"
            with open(t_clean_path, 'w', encoding='utf-8') as f:
                f.write(t_clean)
            
            # 保存原始音频（软链接或复制）
            audio_clean_path = audio_clean_dir / f"{sample_id}.wav"
            if not audio_clean_path.exists():
                try:
                    os.symlink(os.path.abspath(audio_path), audio_clean_path)
                except OSError:
                    # 如果软链接失败（例如Windows），则复制文件
                    import shutil
                    shutil.copy2(audio_path, audio_clean_path)
            
            # Group 1: Attack
            waveform_adv, attack_metrics, attack_time, sample_rate = run_attack(
                model, tokenizer, audio_extractor,
                audio_path, prompt,
                target_emotion, source_emotion,
                emotion_classifier, emotion_label_to_idx,
                svd_components=svd_components, svd_mean=svd_mean,
                epsilon=epsilon, steps=steps, alpha=alpha,
                lambda_emo=lambda_emo, lambda_sem=lambda_sem, lambda_per=lambda_per,
                device=device
            )
            
            # 保存对抗音频
            audio_adv_path = audio_adv_dir / f"{sample_id}.wav"
            waveform_adv_np = waveform_adv.detach().cpu().numpy()
            sf.write(str(audio_adv_path), waveform_adv_np, sample_rate)
            
            # Group 1: Attack推理
            t_adv = clean_inference(
                model, tokenizer, audio_extractor,
                str(audio_adv_path), prompt, device
            )
            
            t_adv_path = text_adv_dir / f"{sample_id}.txt"
            with open(t_adv_path, 'w', encoding='utf-8') as f:
                f.write(t_adv)
            
            runtime = time.time() - start_time
            
            # 记录结果
            result.update({
                'linf': attack_metrics.get('linf', 0.0),
                'l2': attack_metrics.get('l2', 0.0),
                'snr': attack_metrics.get('snr', 0.0),
                't_clean_path': str(t_clean_path),
                't_adv_path': str(t_adv_path),
                'audio_clean_path': str(audio_clean_path),
                'audio_adv_path': str(audio_adv_path),
                'runtime_sec': runtime,
            })
            
        except Exception as e:
            result['status'] = 'failed'
            result['error_msg'] = str(e)
            print(f"\nError processing sample {sample_id}: {e}")
        
        results.append(result)
    
    # Step 5: 保存results.csv
    print("\n" + "=" * 80)
    print("Step 4: Saving results...")
    print("=" * 80)
    
    csv_file = output_path / "results.csv"
    if results:
        fieldnames = [
            'sample_id', 'src_path', 'split', 'prompt',
            'epsilon', 'steps', 'alpha', 'lambda_emo', 'lambda_sem', 'lambda_per',
            'linf', 'l2', 'snr',
            't_clean_path', 't_adv_path', 'audio_clean_path', 'audio_adv_path',
            'runtime_sec', 'status', 'error_msg'
        ]
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"✅ Results CSV saved: {csv_file}")
    
    # Step 6: 保存config.json
    import platform
    
    config = {
        'experiment_name': 'sad2happy_batch_v1',
        'data_root': data_root,
        'output_dir': str(output_path),
        'prompt': prompt,
        'attack_params': {
            'epsilon': epsilon,
            'steps': steps,
            'alpha': alpha,
            'lambda_emo': lambda_emo,
            'lambda_sem': lambda_sem,
            'lambda_per': lambda_per,
        },
        'model': {
            'omnispeech_path': omnispeech_path,
            'checkpoint_path': checkpoint_path,
        },
        'device': device,
        'seed': seed,
        'environment': {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
        }
    }
    
    # 尝试获取git信息
    try:
        import git
        repo = git.Repo(search_parent_directories=True)
        config['git'] = {
            'commit': repo.head.object.hexsha,
            'branch': repo.active_branch.name,
            'is_dirty': repo.is_dirty(),
        }
    except:
        config['git'] = {'commit': 'unknown', 'branch': 'unknown', 'is_dirty': False}
    
    config_file = output_path / "config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"✅ Config JSON saved: {config_file}")
    
    # 统计信息
    successful = [r for r in results if r['status'] == 'ok']
    failed = [r for r in results if r['status'] == 'failed']
    
    print("\n" + "=" * 80)
    print("Experiment Summary")
    print("=" * 80)
    print(f"Total samples: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    if successful:
        avg_linf = np.mean([r['linf'] for r in successful])
        avg_l2 = np.mean([r['l2'] for r in successful])
        avg_snr = np.mean([r['snr'] for r in successful])
        avg_runtime = np.mean([r['runtime_sec'] for r in successful])
        print(f"\nAverage metrics:")
        print(f"  L-inf: {avg_linf:.6f}")
        print(f"  L2: {avg_l2:.6f}")
        print(f"  SNR: {avg_snr:.2f} dB")
        print(f"  Runtime: {avg_runtime:.2f} sec")
    print("=" * 80)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="sad2happy Batch Experiment")
    parser.add_argument("--omnispeech-path", required=True, help="Path to OpenS2S model")
    parser.add_argument("--checkpoint", default=None, help="Path to emotion classifier checkpoint (optional)")
    parser.add_argument("--data-root", default="/data3/xuzhenyu/OpenS2S/data/en_query_wav/",
                        help="Root directory of audio data")
    parser.add_argument("--output-dir", default="/data3/xuzhenyu/OpenS2S/exp/sad2happy_batch_v1/",
                        help="Output directory")
    parser.add_argument("--prompt", default="What is the emotion of this audio? Please answer with only the emotion label (e.g., happy, sad, neutral).",
                        help="Fixed prompt for inference (should ask for audio emotion)")
    parser.add_argument("--epsilon", type=float, default=0.002, help="Perturbation bound")
    parser.add_argument("--steps", type=int, default=30, help="Number of attack steps")
    parser.add_argument("--alpha", type=float, default=None, help="Step size (default: epsilon/10)")
    parser.add_argument("--lambda-emo", type=float, default=1.0, help="Emotion loss weight")
    parser.add_argument("--lambda-sem", type=float, default=1e-2, help="Semantic loss weight")
    parser.add_argument("--lambda-per", type=float, default=1e-4, help="Perceptual loss weight")
    parser.add_argument("--device", default="cuda:0", help="Device")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for data splitting")
    
    args = parser.parse_args()
    
    run_batch_experiment(
        omnispeech_path=args.omnispeech_path,
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        output_dir=args.output_dir,
        prompt=args.prompt,
        epsilon=args.epsilon,
        steps=args.steps,
        alpha=args.alpha,
        lambda_emo=args.lambda_emo,
        lambda_sem=args.lambda_sem,
        lambda_per=args.lambda_per,
        device=args.device,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

