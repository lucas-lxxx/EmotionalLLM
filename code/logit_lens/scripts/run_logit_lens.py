#!/usr/bin/env python3
"""
Logit Lens 实验主入口脚本
"""

import sys
import argparse
import json
from pathlib import Path

import torch
import yaml
import pandas as pd
from tqdm import tqdm


def setup_paths(config: dict):
    """设置必要的路径"""
    src_path = Path(config['model']['src_path'])
    if str(src_path.parent) not in sys.path:
        sys.path.insert(0, str(src_path.parent))

    # 添加 modal_conflict 模块路径（使用 data 包，避免与 OpenS2S 的 src 冲突）
    dataset_module = Path(config['data']['dataset_module'])
    dataset_src = dataset_module.parent.parent
    if str(dataset_src) not in sys.path:
        sys.path.insert(0, str(dataset_src))


def load_model_and_tokenizer(config: dict):
    """加载模型和 tokenizer"""
    from src.modeling_omnispeech import OmniSpeechModel
    from transformers import AutoTokenizer, WhisperFeatureExtractor

    model_path = Path(config['model']['model_path'])
    device = config['model']['device']

    print(f"加载模型: {model_path}")

    # 加载 tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 加载模型
    model = OmniSpeechModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        device_map=None,
    )
    model = model.to(device)
    model.eval()

    # 加载音频特征提取器
    audio_extractor = WhisperFeatureExtractor.from_pretrained(model_path / "audio")

    return model, tokenizer, audio_extractor


def main():
    parser = argparse.ArgumentParser(description="Logit Lens 实验")
    parser.add_argument("--config", type=str,
                        default="/data1/lixiang/lx_code/logit_lens/configs/logit_lens_config.yaml",
                        help="配置文件路径")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大样本数（用于调试）")
    parser.add_argument("--no_sanity_check", action="store_true",
                        help="跳过 sanity check")
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("Logit Lens 实验")
    print("=" * 60)

    # 设置路径
    setup_paths(config)

    # 创建输出目录
    output_dir = Path(config['output']['results_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    print("\n[1] 加载模型...")
    model, tokenizer, audio_extractor = load_model_and_tokenizer(config)
    device = config['model']['device']
    print(f"    模型加载完成，设备: {device}")

    # 加载数据集
    print("\n[2] 加载数据集...")
    from data.dataset import ModalConflictDataset

    dataset = ModalConflictDataset(
        text_jsonl=config['data']['text_jsonl'],
        audio_root=config['data']['audio_root'],
        emotions=config['data'].get('emotions'),
    )
    print(f"    数据集加载完成，共 {len(dataset)} 个样本")

    # 获取冲突样本
    conflict_samples = dataset.get_conflict_samples()
    print(f"    冲突样本数: {len(conflict_samples)}")

    # 限制样本数（调试用）
    if args.max_samples:
        conflict_samples = conflict_samples[:args.max_samples]
        print(f"    限制为 {len(conflict_samples)} 个样本")

    # 初始化 LabelTokenizer
    print("\n[3] 初始化 LabelTokenizer...")
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from label_tokenizer import LabelTokenizer
    from logit_lens import LogitLensExtractor

    label_tokenizer = LabelTokenizer(tokenizer, emotions=config['data'].get('emotions'))
    label_tokenizer.print_summary()

    # 保存 tokenization 报告
    tokenization_report_path = output_dir / "tokenization_report.json"
    label_tokenizer.save_report(str(tokenization_report_path))
    print(f"    Tokenization 报告已保存: {tokenization_report_path}")

    # 初始化 LogitLensExtractor
    print("\n[4] 初始化 LogitLensExtractor...")
    extractor = LogitLensExtractor(
        model=model,
        tokenizer=tokenizer,
        audio_extractor=audio_extractor,
        label_tokenizer=label_tokenizer,
        device=device,
        expected_n_layers=config['model'].get('n_layers'),
    )

    # 提取 Logit Lens 结果
    print("\n[5] 提取 Logit Lens 结果...")
    prompt = config['prompt']
    system_prompt = config.get('system_prompt')
    do_sanity_check = not args.no_sanity_check

    results = []
    for sample in tqdm(conflict_samples, desc="Processing"):
        try:
            result = extractor.extract_single_sample(
                sample, prompt, system_prompt, do_sanity_check
            )
            results.append(result)
        except Exception as e:
            if isinstance(e, ValueError) and "hidden_states length" in str(e):
                raise
            print(f"\n    Error processing {sample.sample_id}: {e}")
            continue

    print(f"\n    成功处理 {len(results)} 个样本")
    if not results:
        print("    未获得有效结果，终止后续步骤。")
        return

    # 计算聚合指标
    print("\n[6] 计算聚合指标...")
    metrics = LogitLensExtractor.compute_metrics(results)
    if not metrics:
        print("    聚合指标为空，终止后续步骤。")
        return
    print(f"    样本数: {metrics['n_samples']}")
    print(f"    层数: {metrics['n_layers']}")

    # 生成可视化
    print("\n[7] 生成可视化...")
    from visualization import plot_margin_curve, plot_winrate_curve

    margin_path = output_dir / "margin_curve_conflict.png"
    winrate_path = output_dir / "winrate_curve_conflict.png"

    plot_margin_curve(metrics, str(margin_path))
    plot_winrate_curve(metrics, str(winrate_path))

    # 保存 sample-level 逐层结果
    print("\n[8] 保存结果...")
    records = []
    emotions = label_tokenizer.emotions
    for r in results:
        semantic_idx = emotions.index(r.semantic_emotion)
        prosody_idx = emotions.index(r.prosody_emotion)
        for layer_idx in range(r.n_layers):
            layer_logits = r.layer_logits[layer_idx]
            pred = r.layer_predictions[layer_idx]
            record = {
                'sample_id': r.sample_id,
                'text_id': r.text_id,
                'layer': layer_idx,
                'semantic_emotion': r.semantic_emotion,
                'prosody_emotion': r.prosody_emotion,
                'pred_emotion': pred,
                'margin': r.layer_margins[layer_idx],
                'logit_semantic': layer_logits[semantic_idx],
                'logit_prosody': layer_logits[prosody_idx],
                'win_semantic': int(pred == r.semantic_emotion),
                'win_prosody': int(pred == r.prosody_emotion),
                'win_other': int(pred not in (r.semantic_emotion, r.prosody_emotion)),
                'generated_token': r.generated_token,
                'predicted_token': r.predicted_token,
                'sanity_match': r.sanity_match,
            }
            for emo_idx, emo in enumerate(emotions):
                record[f'logit_{emo}'] = layer_logits[emo_idx]
            records.append(record)

    df = pd.DataFrame(records)
    sample_csv_path = output_dir / "logit_lens_metrics_sample.csv"
    df.to_csv(sample_csv_path, index=False)
    print(f"    Sample-level 结果已保存: {sample_csv_path}")

    # 保存 group-by-text 逐层结果
    group_cols = ['text_id', 'layer']
    agg_map = {
        'semantic_emotion': 'first',
        'margin': 'mean',
        'logit_semantic': 'mean',
        'logit_prosody': 'mean',
        'win_semantic': 'mean',
        'win_prosody': 'mean',
        'win_other': 'mean',
    }
    for emo in emotions:
        agg_map[f'logit_{emo}'] = 'mean'
    group_df = df.groupby(group_cols, as_index=False).agg(agg_map)
    group_counts = df.groupby(group_cols).size().reset_index(name='n_samples')
    group_df = group_df.merge(group_counts, on=group_cols, how='left')

    group_csv_path = output_dir / "logit_lens_metrics_group.csv"
    group_df.to_csv(group_csv_path, index=False)
    print(f"    Group-level 结果已保存: {group_csv_path}")

    # 保存聚合指标
    metrics_to_save = {
        'n_samples': metrics['n_samples'],
        'n_layers': metrics['n_layers'],
        'mean_margins': metrics['mean_margins'].tolist(),
        'std_margins': metrics['std_margins'].tolist(),
        'win_semantic': metrics['win_semantic'].tolist(),
        'win_prosody': metrics['win_prosody'].tolist(),
        'win_other': metrics['win_other'].tolist(),
    }
    metrics_path = output_dir / "logit_lens_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"    聚合指标已保存: {metrics_path}")

    print("\n" + "=" * 60)
    print("实验完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
