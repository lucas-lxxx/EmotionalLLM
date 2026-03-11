#!/usr/bin/env python3
"""
DeepSeek LLM Judge — 语义保留评估脚本
======================================
使用 DeepSeek API 判断对抗样本 ASR 文本是否保留了原始语义。
判断标准：ASR 结果只要不影响 LLM 的理解判断即可算 PASS(1)，否则 FAIL(0)。

用法:
    python deepseek_judge.py                       # 处理所有样本
    python deepseek_judge.py --dry-run             # 只统计，不调用 API
    python deepseek_judge.py --limit 50            # 只处理前 50 个需要 API 调用的样本
    python deepseek_judge.py --force               # 强制重新评估所有样本（忽略已有结果）

输出:
    - 每个 JSON 文件的 llm_judge_result 字段被更新为 "1" / "0" / "skip_high_sim" / "skip_identical" / "error"
    - judge_summary.csv — 汇总所有样本的判断结果
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
DEEPSEEK_API_KEY = "sk-6312f1a72bd04a1b9dc8b31d2a8fc271"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

RESULT_DIR = Path(__file__).resolve().parent
SIM_SKIP_THRESHOLD = 0.99  # semantic_sim >= 此值直接判定为 PASS

SYSTEM_PROMPT = (
    "你是语义保留评估员。判断语音识别文本B是否保留了原始文本A的核心语义"
    "（说话人意图、情感态度、关键事实）。\n"
    "允许：措辞差异、同义替换、ASR常见错别字、标点差异。\n"
    "不允许：核心含义改变、关键信息丢失、情感态度反转、产生无关内容。\n"
    "只回复1（语义保留）或0（语义丧失），不要输出任何其他内容。"
)

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def load_json(fpath: Path) -> dict[str, Any] | None:
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def save_json(fpath: Path, data: dict[str, Any]):
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_judge_response(text: str) -> str:
    """从 LLM 回复中提取 1 或 0。"""
    text = text.strip()
    if text in ("1", "0"):
        return text
    # 容错：提取第一个出现的 0 或 1
    for ch in text:
        if ch in ("1", "0"):
            return ch
    return "error"


# ---------------------------------------------------------------------------
# 核心逻辑
# ---------------------------------------------------------------------------

def judge_single(client, text_a: str, text_b: str) -> str:
    """调用 DeepSeek API 判断单个样本，返回 "1" / "0" / "error"。"""
    user_msg = f"文本A: {text_a}\n文本B: {text_b}"
    try:
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=5,
            temperature=0,
        )
        answer = resp.choices[0].message.content
        return parse_judge_response(answer)
    except Exception as e:
        print(f"  [API ERROR] {e}")
        return "error"


def run_judge(
    result_dir: Path,
    dry_run: bool = False,
    limit: int | None = None,
    force: bool = False,
    rate_limit_pause: float = 0.15,
):
    """主流程：遍历所有 JSON，判断语义保留，写回结果。"""
    # 收集所有 JSON 文件
    json_files = sorted(result_dir.rglob("*.json"))
    # 排除非样本 JSON（如可能存在的配置文件）
    json_files = [f for f in json_files if f.parent != result_dir or f.stem not in ("judge_summary",)]

    print(f"[INFO] 找到 {len(json_files)} 个 JSON 文件")

    # 分类统计
    skip_identical = 0    # asr_text_clean == asr_text_adv
    skip_high_sim = 0     # semantic_sim >= 阈值
    skip_already_judged = 0
    need_api = 0
    skip_no_text = 0
    errors = 0

    # 需要调用 API 的样本
    api_tasks: list[tuple[Path, dict]] = []

    for fpath in json_files:
        data = load_json(fpath)
        if data is None:
            errors += 1
            continue

        text_a = data.get("asr_text_clean", "").strip()
        text_b = data.get("asr_text_adv", "").strip()

        # 无文本则跳过
        if not text_a or not text_b:
            skip_no_text += 1
            data["llm_judge_result"] = "error_no_text"
            if not dry_run:
                save_json(fpath, data)
            continue

        # 文本完全一致 → PASS
        if text_a == text_b:
            skip_identical += 1
            data["llm_judge_result"] = "skip_identical"
            if not dry_run:
                save_json(fpath, data)
            continue

        # 高余弦相似度 → PASS
        sim = data.get("semantic_sim", 0)
        if sim >= SIM_SKIP_THRESHOLD:
            skip_high_sim += 1
            data["llm_judge_result"] = "skip_high_sim"
            if not dry_run:
                save_json(fpath, data)
            continue

        # 已有判断结果且不强制重跑
        existing = data.get("llm_judge_result", "")
        if existing in ("1", "0") and not force:
            skip_already_judged += 1
            continue

        api_tasks.append((fpath, data))

    need_api = len(api_tasks)
    if limit is not None:
        api_tasks = api_tasks[:limit]

    print(f"\n[统计摘要]")
    print(f"  文本完全一致（跳过）:     {skip_identical}")
    print(f"  高相似度跳过（>={SIM_SKIP_THRESHOLD}）: {skip_high_sim}")
    print(f"  已有判断结果（跳过）:     {skip_already_judged}")
    print(f"  无文本（跳过）:           {skip_no_text}")
    print(f"  JSON 解析失败:            {errors}")
    print(f"  需要 API 调用:            {need_api}")
    if limit is not None and need_api > limit:
        print(f"  本次限制处理:             {limit}")
    print()

    if dry_run:
        print("[DRY RUN] 不调用 API，退出。")
        return

    if not api_tasks:
        print("[INFO] 没有需要 API 调用的样本。")
    else:
        # 初始化 OpenAI 客户端（DeepSeek 兼容格式）
        try:
            from openai import OpenAI
        except ImportError:
            print("[ERROR] 请先安装 openai 库: pip install openai")
            sys.exit(1)

        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

        total = len(api_tasks)
        pass_count = 0
        fail_count = 0
        err_count = 0

        print(f"[开始 API 调用] 共 {total} 个样本\n")
        for idx, (fpath, data) in enumerate(api_tasks, 1):
            text_a = data["asr_text_clean"].strip()
            text_b = data["asr_text_adv"].strip()

            result = judge_single(client, text_a, text_b)
            data["llm_judge_result"] = result
            save_json(fpath, data)

            if result == "1":
                pass_count += 1
            elif result == "0":
                fail_count += 1
            else:
                err_count += 1

            if idx % 100 == 0 or idx == total:
                print(f"  进度: {idx}/{total} | PASS={pass_count} FAIL={fail_count} ERR={err_count}")

            # 限速
            time.sleep(rate_limit_pause)

        print(f"\n[API 调用完成] PASS={pass_count}, FAIL={fail_count}, ERROR={err_count}")

    # 生成汇总 CSV
    print("\n[生成汇总 CSV]")
    export_summary(result_dir)


def export_summary(result_dir: Path):
    """读取所有 JSON，输出 judge_summary.csv。"""
    json_files = sorted(result_dir.rglob("*.json"))
    rows = []
    stats = {"skip_identical": 0, "skip_high_sim": 0, "1": 0, "0": 0, "error": 0, "other": 0}

    for fpath in json_files:
        data = load_json(fpath)
        if data is None:
            continue
        if "sample_id" not in data:
            continue

        judge = data.get("llm_judge_result", "")
        # 统一映射为成功/失败
        if judge in ("skip_identical", "skip_high_sim", "1"):
            semantic_pass = 1
        elif judge == "0":
            semantic_pass = 0
        else:
            semantic_pass = -1  # 未知/错误

        # 统计
        if judge in stats:
            stats[judge] += 1
        elif judge.startswith("error"):
            stats["error"] += 1
        else:
            stats["other"] += 1

        rows.append({
            "sample_id": data.get("sample_id", ""),
            "speaker_id": data.get("speaker_id", ""),
            "ground_truth_emotion": data.get("ground_truth_emotion", ""),
            "target_emotion": data.get("target_emotion", ""),
            "success_emo": data.get("success_emo", ""),
            "wer": data.get("wer", ""),
            "semantic_sim": data.get("semantic_sim", ""),
            "semantic_preserved": data.get("semantic_preserved", ""),
            "llm_judge_raw": judge,
            "semantic_pass": semantic_pass,
            "asr_text_clean": data.get("asr_text_clean", ""),
            "asr_text_adv": data.get("asr_text_adv", ""),
        })

    csv_path = result_dir / "judge_summary.csv"
    if rows:
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    total = sum(stats.values())
    pass_total = stats["skip_identical"] + stats["skip_high_sim"] + stats["1"]
    fail_total = stats["0"]

    print(f"  总样本: {total}")
    print(f"  PASS（语义保留）: {pass_total}  ({pass_total/total*100:.2f}%)" if total > 0 else "")
    print(f"    - 文本一致:   {stats['skip_identical']}")
    print(f"    - 高相似度:   {stats['skip_high_sim']}")
    print(f"    - API判定通过: {stats['1']}")
    print(f"  FAIL（语义丧失）: {fail_total}  ({fail_total/total*100:.2f}%)" if total > 0 else "")
    print(f"  错误/未判定:     {stats['error'] + stats['other']}")
    print(f"\n  CSV 已导出: {csv_path}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DeepSeek LLM Judge 语义保留评估")
    parser.add_argument("--result-dir", type=str, default=str(RESULT_DIR),
                        help="结果目录路径")
    parser.add_argument("--dry-run", action="store_true",
                        help="只统计不调用 API")
    parser.add_argument("--limit", type=int, default=None,
                        help="限制 API 调用数量")
    parser.add_argument("--force", action="store_true",
                        help="强制重新评估所有样本")
    parser.add_argument("--rate-limit", type=float, default=0.15,
                        help="API 调用间隔（秒），默认 0.15")
    args = parser.parse_args()

    run_judge(
        result_dir=Path(args.result_dir),
        dry_run=args.dry_run,
        limit=args.limit,
        force=args.force,
        rate_limit_pause=args.rate_limit,
    )


if __name__ == "__main__":
    main()
