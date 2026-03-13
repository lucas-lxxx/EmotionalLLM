#!/usr/bin/env python3
"""
DeepSeek LLM Judge — Voxtral 语义保留评估脚本
==============================================
使用 DeepSeek API 判断对抗样本 ASR 文本是否保留了原始语义。
判断标准：ASR 结果只要不影响 LLM 的理解判断即可算 PASS(1)，否则 FAIL(0)。
自动处理 Voxtral_EN / Voxtral_CN 两个子集。

用法:
    python deepseek_judge.py                       # 处理所有样本
    python deepseek_judge.py --dry-run             # 只统计，不调用 API
    python deepseek_judge.py --limit 50            # 只处理前 50 个需要 API 调用的样本
    python deepseek_judge.py --force               # 强制重新评估所有样本（忽略已有结果）

输出:
    - 每个 JSON 文件的 llm_judge_result 字段被更新为 "1" / "0" / "skip_high_sim" / "skip_identical" / "error"
    - 每个子集目录下: judge_summary.csv — 汇总所有样本的判断结果
    - 根目录下: judge_summary_all.csv — EN+CN 合并汇总
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


def run_judge_on_dir(
    subset_dir: Path,
    subset_name: str,
    client,
    dry_run: bool = False,
    limit: int | None = None,
    force: bool = False,
    rate_limit_pause: float = 0.15,
) -> list[tuple[Path, dict]]:
    """对单个子集目录执行 judge 流程。返回所有已处理文件的 (fpath, data) 列表。"""
    print(f"\n{'='*60}")
    print(f"  处理子集: {subset_name}")
    print(f"  数据目录: {subset_dir}")
    print(f"{'='*60}")

    # 收集所有 JSON 文件（排除 summary_*.json 和 judge_summary.json 等非样本文件）
    json_files = sorted(subset_dir.rglob("*.json"))
    json_files = [
        f for f in json_files
        if not f.name.startswith("summary_")
        and not f.name.startswith("judge_summary")
        and not f.name.startswith("cleaned_data")
    ]

    print(f"[INFO] 找到 {len(json_files)} 个 JSON 文件")

    # 分类统计
    skip_identical = 0
    skip_high_sim = 0
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

    print(f"\n[统计摘要 — {subset_name}]")
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
        print(f"[DRY RUN] {subset_name} 不调用 API。")
        return []

    if not api_tasks:
        print(f"[INFO] {subset_name} 没有需要 API 调用的样本。")
    elif client is not None:
        total = len(api_tasks)
        pass_count = 0
        fail_count = 0
        err_count = 0

        print(f"[开始 API 调用 — {subset_name}] 共 {total} 个样本\n")
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

        print(f"\n[API 调用完成 — {subset_name}] PASS={pass_count}, FAIL={fail_count}, ERROR={err_count}")

    # 生成子集汇总 CSV
    print(f"\n[生成汇总 CSV — {subset_name}]")
    export_summary(subset_dir)

    return api_tasks


def export_summary(result_dir: Path, csv_filename: str = "judge_summary.csv"):
    """读取所有 JSON，输出 judge_summary.csv。"""
    json_files = sorted(result_dir.rglob("*.json"))
    rows = []
    stats = {"skip_identical": 0, "skip_high_sim": 0, "1": 0, "0": 0, "error": 0, "other": 0}

    for fpath in json_files:
        # 跳过非样本文件
        if fpath.name.startswith("summary_") or fpath.name.startswith("judge_summary") or fpath.name.startswith("cleaned_data"):
            continue
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

    csv_path = result_dir / csv_filename
    if rows:
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    total = sum(stats.values())
    pass_total = stats["skip_identical"] + stats["skip_high_sim"] + stats["1"]
    fail_total = stats["0"]

    print(f"  总样本: {total}")
    if total > 0:
        print(f"  PASS（语义保留）: {pass_total}  ({pass_total/total*100:.2f}%)")
        print(f"    - 文本一致:   {stats['skip_identical']}")
        print(f"    - 高相似度:   {stats['skip_high_sim']}")
        print(f"    - API判定通过: {stats['1']}")
        print(f"  FAIL（语义丧失）: {fail_total}  ({fail_total/total*100:.2f}%)")
        print(f"  错误/未判定:     {stats['error'] + stats['other']}")
    print(f"\n  CSV 已导出: {csv_path}")

    return rows


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DeepSeek LLM Judge — Voxtral 语义保留评估")
    parser.add_argument("--result-dir", type=str, default=str(RESULT_DIR),
                        help="结果目录路径")
    parser.add_argument("--dry-run", action="store_true",
                        help="只统计不调用 API")
    parser.add_argument("--limit", type=int, default=None,
                        help="限制每个子集的 API 调用数量")
    parser.add_argument("--force", action="store_true",
                        help="强制重新评估所有样本")
    parser.add_argument("--rate-limit", type=float, default=0.15,
                        help="API 调用间隔（秒），默认 0.15")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    print(f"[INFO] 根数据目录: {result_dir}")

    # 自动发现子集目录
    subset_dirs = sorted([
        d for d in result_dir.iterdir()
        if d.is_dir() and d.name.startswith("Voxtral_")
    ])

    if not subset_dirs:
        print(f"[ERROR] 在 {result_dir} 下未找到 Voxtral_* 子目录")
        sys.exit(1)

    print(f"[INFO] 发现 {len(subset_dirs)} 个子集: {[d.name for d in subset_dirs]}")

    # 初始化 API 客户端（非 dry-run 时）
    client = None
    if not args.dry_run:
        try:
            from openai import OpenAI
        except ImportError:
            print("[ERROR] 请先安装 openai 库: pip install openai")
            sys.exit(1)
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    # 逐子集处理
    for subset_dir in subset_dirs:
        run_judge_on_dir(
            subset_dir=subset_dir,
            subset_name=subset_dir.name,
            client=client,
            dry_run=args.dry_run,
            limit=args.limit,
            force=args.force,
            rate_limit_pause=args.rate_limit,
        )

    # 生成合并汇总 CSV
    if len(subset_dirs) > 1:
        print(f"\n{'='*60}")
        print(f"  生成合并汇总 CSV（EN + CN）")
        print(f"{'='*60}")
        all_rows = []
        for subset_dir in subset_dirs:
            csv_path = subset_dir / "judge_summary.csv"
            if csv_path.exists():
                with open(csv_path, "r", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        row["subset"] = subset_dir.name
                        all_rows.append(row)

        if all_rows:
            combined_csv = result_dir / "judge_summary_all.csv"
            fieldnames = list(all_rows[0].keys())
            with open(combined_csv, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_rows)

            total = len(all_rows)
            pass_total = sum(1 for r in all_rows if r.get("semantic_pass") in ("1", 1))
            fail_total = sum(1 for r in all_rows if r.get("semantic_pass") in ("0", 0))
            err_total = total - pass_total - fail_total

            print(f"  总样本: {total}")
            if total > 0:
                print(f"  PASS（语义保留）: {pass_total}  ({pass_total/total*100:.2f}%)")
                print(f"  FAIL（语义丧失）: {fail_total}  ({fail_total/total*100:.2f}%)")
                print(f"  错误/未判定:     {err_total}")
            print(f"\n  合并 CSV 已导出: {combined_csv}")

    print(f"\n[DONE] 全部评估完成！")


if __name__ == "__main__":
    main()
