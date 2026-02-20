"""sad2happy批量实验数据准备：生成sad_list.txt和happy_list.txt，并进行80/20划分"""

import argparse
import sys
import os
from pathlib import Path
import random

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def prepare_sad2happy_data(
    data_root: str = "/data3/xuzhenyu/OpenS2S/data/en_query_wav/",
    output_dir: str = "/data3/xuzhenyu/OpenS2S/exp/sad2happy_batch_v1/",
    seed: int = 2025
):
    """
    准备sad2happy批量实验数据
    
    Args:
        data_root: 数据根目录
        output_dir: 输出目录
        seed: 随机种子
    """
    random.seed(seed)
    
    data_path = Path(data_root)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找所有sad和happy的wav文件
    print(f"Scanning data directory: {data_path}")
    
    sad_files = []
    happy_files = []
    
    # 遍历所有wav文件
    # 目录结构：{emotion}/{age}/{gender}/*.wav
    # 例如：Sad/adult/female/xxx.wav 或 Happy/child/male/xxx.wav
    for wav_file in data_path.rglob("*.wav"):
        abs_path = str(wav_file.resolve())
        # 获取完整路径的各个部分
        parts = abs_path.lower().split(os.sep)
        
        # 检查路径中是否包含情绪标签
        # 路径格式：.../Sad/... 或 .../Happy/...
        if 'sad' in parts:
            sad_files.append(abs_path)
        elif 'happy' in parts:
            happy_files.append(abs_path)
    
    print(f"Found {len(sad_files)} sad files")
    print(f"Found {len(happy_files)} happy files")
    
    # 保存完整列表
    sad_list_file = output_path / "sad_list.txt"
    happy_list_file = output_path / "happy_list.txt"
    
    with open(sad_list_file, 'w', encoding='utf-8') as f:
        for path in sorted(sad_files):
            f.write(f"{path}\n")
    
    with open(happy_list_file, 'w', encoding='utf-8') as f:
        for path in sorted(happy_files):
            f.write(f"{path}\n")
    
    print(f"✅ Saved {sad_list_file}")
    print(f"✅ Saved {happy_list_file}")
    
    # 对sad文件进行80/20划分
    random.shuffle(sad_files)
    split_idx = int(len(sad_files) * 0.8)
    
    sad_train = sorted(sad_files[:split_idx])
    sad_test = sorted(sad_files[split_idx:])
    
    sad_train_file = output_path / "sad_train.txt"
    sad_test_file = output_path / "sad_test.txt"
    
    with open(sad_train_file, 'w', encoding='utf-8') as f:
        for path in sad_train:
            f.write(f"{path}\n")
    
    with open(sad_test_file, 'w', encoding='utf-8') as f:
        for path in sad_test:
            f.write(f"{path}\n")
    
    print(f"✅ Saved {sad_train_file} ({len(sad_train)} samples)")
    print(f"✅ Saved {sad_test_file} ({len(sad_test)} samples)")
    
    # 可选：对happy文件也进行划分（用于构造方向v或probe）
    if len(happy_files) > 0:
        random.shuffle(happy_files)
        split_idx_happy = int(len(happy_files) * 0.8)
        
        happy_train = sorted(happy_files[:split_idx_happy])
        happy_test = sorted(happy_files[split_idx_happy:])
        
        happy_train_file = output_path / "happy_train.txt"
        happy_test_file = output_path / "happy_test.txt"
        
        with open(happy_train_file, 'w', encoding='utf-8') as f:
            for path in happy_train:
                f.write(f"{path}\n")
        
        with open(happy_test_file, 'w', encoding='utf-8') as f:
            for path in happy_test:
                f.write(f"{path}\n")
        
        print(f"✅ Saved {happy_train_file} ({len(happy_train)} samples)")
        print(f"✅ Saved {happy_test_file} ({len(happy_test)} samples)")
    
    print("\n" + "=" * 80)
    print("✅ Data preparation completed!")
    print("=" * 80)
    print(f"Output directory: {output_path}")
    print(f"  - sad_list.txt: {len(sad_files)} samples")
    print(f"  - happy_list.txt: {len(happy_files)} samples")
    print(f"  - sad_train.txt: {len(sad_train)} samples")
    print(f"  - sad_test.txt: {len(sad_test)} samples")
    print("=" * 80)
    
    return {
        'sad_list': sad_list_file,
        'happy_list': happy_list_file,
        'sad_train': sad_train_file,
        'sad_test': sad_test_file,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare sad2happy batch experiment data")
    parser.add_argument("--data-root", default="/data3/xuzhenyu/OpenS2S/data/en_query_wav/",
                        help="Root directory of audio data")
    parser.add_argument("--output-dir", default="/data3/xuzhenyu/OpenS2S/exp/sad2happy_batch_v1/",
                        help="Output directory for lists")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for splitting")
    
    args = parser.parse_args()
    
    prepare_sad2happy_data(
        data_root=args.data_root,
        output_dir=args.output_dir,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

