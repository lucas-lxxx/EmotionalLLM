"""
标签 Tokenization 辅助模块

处理情绪标签的 tokenization，支持单 token 和多 token 情况。
"""

from typing import Dict, List, Tuple
import json


# 情绪标签
EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'surprised']
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}
IDX_TO_EMOTION = {i: e for i, e in enumerate(EMOTIONS)}


class LabelTokenizer:
    """标签 Tokenizer 封装类"""

    def __init__(self, tokenizer, emotions: List[str] = None):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            emotions: 情绪标签列表
        """
        self.tokenizer = tokenizer
        self.emotions = emotions or EMOTIONS
        self.label_info = {}
        self._tokenize_labels()

    def _tokenize_labels(self):
        """对所有标签进行 tokenization"""
        for emotion in self.emotions:
            token_ids = self.tokenizer.encode(emotion, add_special_tokens=False)
            token_strs = self.tokenizer.convert_ids_to_tokens(token_ids)

            self.label_info[emotion] = {
                'token_ids': token_ids,
                'token_strs': token_strs,
                'is_single_token': len(token_ids) == 1,
                'first_token_id': token_ids[0],
                'first_token_str': token_strs[0],
            }

    def get_label_token_ids(self) -> Dict[str, int]:
        """获取每个标签的 first token id"""
        return {e: info['first_token_id'] for e, info in self.label_info.items()}

    def get_label_token_id_list(self) -> List[int]:
        """按 EMOTIONS 顺序返回 token id 列表"""
        return [self.label_info[e]['first_token_id'] for e in self.emotions]

    def has_multi_token_labels(self) -> bool:
        """检查是否有多 token 标签"""
        return any(not info['is_single_token'] for info in self.label_info.values())

    def get_multi_token_warnings(self) -> List[str]:
        """获取多 token 标签的警告信息"""
        warnings = []
        for emotion, info in self.label_info.items():
            if not info['is_single_token']:
                warnings.append(
                    f"Warning: '{emotion}' tokenizes to multiple tokens: "
                    f"{info['token_strs']}. Using first token: {info['first_token_str']}"
                )
        return warnings

    def report(self) -> Dict:
        """生成 tokenization 报告"""
        return {
            'emotions': self.emotions,
            'label_info': self.label_info,
            'has_multi_token': self.has_multi_token_labels(),
            'warnings': self.get_multi_token_warnings(),
        }

    def save_report(self, path: str):
        """保存 tokenization 报告到 JSON 文件"""
        report = self.report()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    def print_summary(self):
        """打印 tokenization 摘要"""
        print("=" * 50)
        print("标签 Tokenization 摘要")
        print("=" * 50)
        for emotion in self.emotions:
            info = self.label_info[emotion]
            status = "单token" if info['is_single_token'] else "多token"
            print(f"  {emotion}: {info['token_ids']} ({status})")

        warnings = self.get_multi_token_warnings()
        if warnings:
            print("\n警告:")
            for w in warnings:
                print(f"  {w}")
