#!/usr/bin/env python3
"""
ESDfinal 实验结果批量清洗、统计与报告生成脚本
==============================================
用法:
    python analyze_results.py                  # 基础分析（不含 LLM Judge）
    python analyze_results.py --llm-judge      # 启用 LLM Judge 语义一致性评估
    python analyze_results.py --llm-judge --api-key YOUR_KEY --model gpt-4o-mini

输出（均在当前目录）:
    cleaned_data.csv   — 扁平化清洗后数据
    report.md          — 完整统计报告
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
RESULT_DIR = Path(__file__).resolve().parent
EMOTION_LABELS = {"angry", "happy", "sad", "surprise", "neutral"}
PROMPT_NAMES = ["Prompt-1", "Prompt-2", "Prompt-3"]
SNR_BINS = [("<15 dB", 0, 15), ("15-20 dB", 15, 20), ("20-25 dB", 20, 25), (">25 dB", 25, 999)]

# ---------------------------------------------------------------------------
# 1. 数据加载与清洗
# ---------------------------------------------------------------------------

def normalize_emotion(raw: str) -> str:
    """将模型回复中的情绪标签标准化为小写英文标签。"""
    if not raw or not raw.strip():
        return "unparseable"
    text = raw.strip().lower()
    # 直接匹配
    if text in EMOTION_LABELS:
        return text
    # 中文映射
    zh_map = {
        "愤怒": "angry", "生气": "angry", "怒": "angry",
        "高兴": "happy", "快乐": "happy", "开心": "happy", "喜悦": "happy",
        "恭喜": "happy", "幸福": "happy",
        "悲伤": "sad", "伤心": "sad", "难过": "sad",
        "惊讶": "surprise", "吃惊": "surprise", "惊喜": "surprise",
        "中性": "neutral", "平静": "neutral", "中立": "neutral",
    }
    for k, v in zh_map.items():
        if k in text:
            return v
    # 英文子串匹配
    for label in EMOTION_LABELS:
        if label in text:
            return label
    return "unparseable"


def load_and_clean(result_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """加载所有 JSON 并执行清洗，返回 (records, warnings)。"""
    records: list[dict[str, Any]] = []
    warnings: list[str] = []

    json_files = sorted(result_dir.rglob("*.json"))
    if not json_files:
        print(f"[ERROR] 在 {result_dir} 下未找到任何 JSON 文件")
        sys.exit(1)

    for fpath in json_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            warnings.append(f"JSON 解析失败: {fpath.name} — {e}")
            continue

        # 必要字段检查
        required = [
            "sample_id", "speaker_id", "ground_truth_emotion", "target_emotion",
            "emo_pred_clean", "emo_pred_adv", "success_emo",
            "wer", "semantic_sim", "semantic_preserved",
            "delta_linf", "delta_l2", "snr_db",
        ]
        missing = [k for k in required if k not in data]
        if missing:
            warnings.append(f"字段缺失 {fpath.name}: {missing}")
            continue

        # 异常值检测
        wer = data["wer"]
        sem = data["semantic_sim"]
        snr = data["snr_db"]
        linf = data["delta_linf"]
        l2 = data["delta_l2"]

        if not (0 <= wer <= 2):
            warnings.append(f"WER 异常 ({wer:.4f}): {fpath.name}")
        if not (0 <= sem <= 1.01):
            warnings.append(f"Semantic Sim 异常 ({sem:.4f}): {fpath.name}")
        if snr < 0 or snr > 60:
            warnings.append(f"SNR 异常 ({snr:.2f} dB): {fpath.name}")

        # 标准化情绪预测
        pred_clean_raw = data.get("emo_pred_clean", ["", "", ""])
        pred_adv_raw = data.get("emo_pred_adv", ["", "", ""])
        pred_clean_norm = [normalize_emotion(p) for p in pred_clean_raw]
        pred_adv_norm = [normalize_emotion(p) for p in pred_adv_raw]

        # 提取 loss trace 摘要
        loss_trace = data.get("loss_trace", [])
        if loss_trace:
            first_loss = loss_trace[0]
            last_loss = loss_trace[-1]
            initial_emo = first_loss.get("emo", float("nan"))
            final_emo = last_loss.get("emo", float("nan"))
            final_asr_loss = last_loss.get("asr", float("nan"))
            final_per = last_loss.get("per", float("nan"))
            final_total = last_loss.get("total", float("nan"))
            loss_reduction = (initial_emo - final_emo) / initial_emo if initial_emo > 0 else float("nan")
            # 收敛步数：首次 emo < 1.0
            converge_step = len(loss_trace)  # 默认未收敛
            for entry in loss_trace:
                if entry.get("emo", 999) < 1.0:
                    converge_step = entry["step"]
                    break
        else:
            initial_emo = final_emo = final_asr_loss = final_per = final_total = float("nan")
            loss_reduction = float("nan")
            converge_step = -1

        rec = {
            "sample_id": data["sample_id"],
            "speaker_id": data["speaker_id"],
            "ground_truth_emotion": data["ground_truth_emotion"],
            "target_emotion": data["target_emotion"],
            # Per-prompt clean predictions (normalized)
            "pred_clean_1": pred_clean_norm[0],
            "pred_clean_2": pred_clean_norm[1],
            "pred_clean_3": pred_clean_norm[2],
            # Per-prompt adv predictions (normalized)
            "pred_adv_1": pred_adv_norm[0],
            "pred_adv_2": pred_adv_norm[1],
            "pred_adv_3": pred_adv_norm[2],
            # Attack success
            "success_emo": data["success_emo"],
            "attack_success_p1": pred_adv_norm[0] == data["target_emotion"],
            "attack_success_p2": pred_adv_norm[1] == data["target_emotion"],
            "attack_success_p3": pred_adv_norm[2] == data["target_emotion"],
            "attack_success_joint": all(p == data["target_emotion"] for p in pred_adv_norm),
            # Semantic
            "asr_text_clean": data.get("asr_text_clean", ""),
            "asr_text_adv": data.get("asr_text_adv", ""),
            "wer": wer,
            "semantic_sim": sem,
            "semantic_preserved": data["semantic_preserved"],
            # Perturbation
            "delta_linf": linf,
            "delta_l2": l2,
            "snr_db": snr,
            # Optimization
            "initial_emo_loss": initial_emo,
            "final_emo_loss": final_emo,
            "final_asr_loss": final_asr_loss,
            "final_per_loss": final_per,
            "final_total_loss": final_total,
            "loss_reduction_rate": loss_reduction,
            "converge_step": converge_step,
            "total_steps": len(loss_trace),
            # LLM Judge placeholder
            "llm_judge_result": "",
        }
        records.append(rec)

    return records, warnings


# ---------------------------------------------------------------------------
# 2. LLM Judge 语义一致性评估
# ---------------------------------------------------------------------------

def llm_judge_batch(
    records: list[dict[str, Any]],
    api_key: str,
    model: str = "gpt-4o-mini",
    base_url: str | None = None,
    batch_size: int = 20,
) -> list[dict[str, Any]]:
    """
    使用第三方 LLM 作为裁判员，判断 asr_text_clean 与 asr_text_adv 是否语义基本一致。
    结果写入 records[i]["llm_judge_result"]，取值: "consistent" / "inconsistent" / "error"
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("[WARN] openai 库未安装，跳过 LLM Judge。请 pip install openai")
        return records

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    system_prompt = (
        "你是一个语义一致性裁判员。给定两段中文文本（原始文本 A 和对抗文本 B），"
        "判断它们的语义内容是否基本一致（允许措辞差异，但核心含义应相同）。\n"
        "只回答 'consistent' 或 'inconsistent'，不要解释。"
    )

    total = len(records)
    judged = 0
    for i, rec in enumerate(records):
        text_a = rec.get("asr_text_clean", "").strip()
        text_b = rec.get("asr_text_adv", "").strip()

        if not text_a or not text_b:
            rec["llm_judge_result"] = "error"
            continue

        # 如果余弦相似度 >= 0.99，直接标记为 consistent 以节省 API 调用
        if rec.get("semantic_sim", 0) >= 0.99:
            rec["llm_judge_result"] = "consistent"
            judged += 1
            continue

        user_msg = f"文本 A: {text_a}\n文本 B: {text_b}"
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=10,
                temperature=0,
            )
            answer = resp.choices[0].message.content.strip().lower()
            if "consistent" in answer and "inconsistent" not in answer:
                rec["llm_judge_result"] = "consistent"
            elif "inconsistent" in answer:
                rec["llm_judge_result"] = "inconsistent"
            else:
                rec["llm_judge_result"] = answer[:30]
        except Exception as e:
            rec["llm_judge_result"] = "error"
            if judged == 0:
                print(f"[WARN] LLM Judge API 调用失败: {e}")

        judged += 1
        if judged % 100 == 0:
            print(f"  LLM Judge 进度: {judged}/{total}")
        # 简单限速
        if judged % batch_size == 0:
            time.sleep(1)

    return records


# ---------------------------------------------------------------------------
# 3. 统计计算
# ---------------------------------------------------------------------------

def safe_mean(vals: list[float]) -> float:
    return statistics.mean(vals) if vals else float("nan")

def safe_stdev(vals: list[float]) -> float:
    return statistics.stdev(vals) if len(vals) > 1 else 0.0

def safe_median(vals: list[float]) -> float:
    return statistics.median(vals) if vals else float("nan")

def pct(num: int, denom: int) -> str:
    return f"{num/denom*100:.2f}%" if denom > 0 else "N/A"

def pct_val(num: int, denom: int) -> float:
    return num / denom * 100 if denom > 0 else 0.0


def compute_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    """计算所有统计指标。"""
    N = len(records)
    stats: dict[str, Any] = {"total_samples": N}

    # --- A. 攻击成功率 ---
    success_emo_count = sum(1 for r in records if r["success_emo"])
    stats["overall_asr"] = pct_val(success_emo_count, N)

    for pi in range(1, 4):
        key = f"attack_success_p{pi}"
        cnt = sum(1 for r in records if r[key])
        stats[f"asr_prompt_{pi}"] = pct_val(cnt, N)

    joint_cnt = sum(1 for r in records if r["attack_success_joint"])
    stats["asr_joint"] = pct_val(joint_cnt, N)

    # By source emotion
    by_emo: dict[str, list] = defaultdict(list)
    for r in records:
        by_emo[r["ground_truth_emotion"]].append(r)
    stats["asr_by_emotion"] = {}
    for emo, recs in sorted(by_emo.items()):
        n = len(recs)
        s = sum(1 for r in recs if r["success_emo"])
        stats["asr_by_emotion"][emo] = {"n": n, "success": s, "rate": pct_val(s, n)}

    # By speaker
    by_spk: dict[str, list] = defaultdict(list)
    for r in records:
        by_spk[r["speaker_id"]].append(r)
    stats["asr_by_speaker"] = {}
    for spk, recs in sorted(by_spk.items()):
        n = len(recs)
        s = sum(1 for r in recs if r["success_emo"])
        stats["asr_by_speaker"][spk] = {"n": n, "success": s, "rate": pct_val(s, n)}

    # By source × speaker
    by_emo_spk: dict[tuple, list] = defaultdict(list)
    for r in records:
        by_emo_spk[(r["ground_truth_emotion"], r["speaker_id"])].append(r)
    stats["asr_by_emotion_speaker"] = {}
    for (emo, spk), recs in sorted(by_emo_spk.items()):
        n = len(recs)
        s = sum(1 for r in recs if r["success_emo"])
        stats["asr_by_emotion_speaker"][f"{emo}_{spk}"] = {"n": n, "success": s, "rate": pct_val(s, n)}

    # --- B. 语义保持 ---
    sem_preserved_cnt = sum(1 for r in records if r["semantic_preserved"])
    stats["semantic_preserve_rate"] = pct_val(sem_preserved_cnt, N)

    sims = [r["semantic_sim"] for r in records]
    stats["semantic_sim_mean"] = safe_mean(sims)
    stats["semantic_sim_std"] = safe_stdev(sims)
    stats["semantic_sim_median"] = safe_median(sims)

    wers = [r["wer"] for r in records]
    stats["wer_mean"] = safe_mean(wers)
    stats["wer_std"] = safe_stdev(wers)
    stats["wer_zero_rate"] = pct_val(sum(1 for w in wers if w == 0.0), N)

    # 联合成功率
    joint_emo_sem = sum(1 for r in records if r["success_emo"] and r["semantic_preserved"])
    stats["joint_success_emo_semantic"] = pct_val(joint_emo_sem, N)

    # 语义保持按情绪分组
    stats["semantic_by_emotion"] = {}
    for emo, recs in sorted(by_emo.items()):
        n = len(recs)
        sp = sum(1 for r in recs if r["semantic_preserved"])
        sm = safe_mean([r["semantic_sim"] for r in recs])
        wm = safe_mean([r["wer"] for r in recs])
        stats["semantic_by_emotion"][emo] = {
            "n": n, "preserve_rate": pct_val(sp, n),
            "sim_mean": sm, "wer_mean": wm,
        }

    # --- C. 扰动幅度 ---
    snrs = [r["snr_db"] for r in records]
    linfs = [r["delta_linf"] for r in records]
    l2s = [r["delta_l2"] for r in records]

    stats["snr_mean"] = safe_mean(snrs)
    stats["snr_std"] = safe_stdev(snrs)
    stats["snr_median"] = safe_median(snrs)
    stats["linf_mean"] = safe_mean(linfs)
    stats["linf_std"] = safe_stdev(linfs)
    stats["l2_mean"] = safe_mean(l2s)
    stats["l2_std"] = safe_stdev(l2s)

    # SNR 分段
    stats["snr_bins"] = {}
    for label, lo, hi in SNR_BINS:
        cnt = sum(1 for s in snrs if lo <= s < hi)
        stats["snr_bins"][label] = {"count": cnt, "pct": pct_val(cnt, N)}

    # 扰动按情绪分组
    stats["perturbation_by_emotion"] = {}
    for emo, recs in sorted(by_emo.items()):
        stats["perturbation_by_emotion"][emo] = {
            "snr_mean": safe_mean([r["snr_db"] for r in recs]),
            "linf_mean": safe_mean([r["delta_linf"] for r in recs]),
            "l2_mean": safe_mean([r["delta_l2"] for r in recs]),
        }

    # --- D. 优化收敛 ---
    final_emos = [r["final_emo_loss"] for r in records if not math.isnan(r["final_emo_loss"])]
    final_asrs = [r["final_asr_loss"] for r in records if not math.isnan(r["final_asr_loss"])]
    final_pers = [r["final_per_loss"] for r in records if not math.isnan(r["final_per_loss"])]
    loss_reds = [r["loss_reduction_rate"] for r in records if not math.isnan(r["loss_reduction_rate"])]
    conv_steps = [r["converge_step"] for r in records if r["converge_step"] >= 0]

    stats["final_emo_loss_mean"] = safe_mean(final_emos)
    stats["final_emo_loss_std"] = safe_stdev(final_emos)
    stats["final_asr_loss_mean"] = safe_mean(final_asrs)
    stats["final_per_loss_mean"] = safe_mean(final_pers)
    stats["loss_reduction_mean"] = safe_mean(loss_reds)
    stats["loss_reduction_std"] = safe_stdev(loss_reds)

    # 收敛统计
    total_steps_list = [r["total_steps"] for r in records if r["total_steps"] > 0]
    converged = [r for r in records if r["converge_step"] < r["total_steps"] and r["converge_step"] >= 0]
    stats["converge_rate"] = pct_val(len(converged), N)
    if converged:
        stats["converge_step_mean"] = safe_mean([r["converge_step"] for r in converged])
        stats["converge_step_median"] = safe_median([r["converge_step"] for r in converged])
    else:
        stats["converge_step_mean"] = float("nan")
        stats["converge_step_median"] = float("nan")

    # --- E. Clean Baseline ---
    stats["clean_accuracy"] = {}
    for pi in range(1, 4):
        key = f"pred_clean_{pi}"
        correct = sum(1 for r in records if r[key] == r["ground_truth_emotion"])
        stats["clean_accuracy"][f"Prompt-{pi}"] = pct_val(correct, N)

    # Clean → Target (误判为目标情绪)
    stats["clean_to_target"] = {}
    non_target = [r for r in records if r["ground_truth_emotion"] != r["target_emotion"]]
    for pi in range(1, 4):
        key = f"pred_clean_{pi}"
        misclass = sum(1 for r in non_target if r[key] == r["target_emotion"])
        stats["clean_to_target"][f"Prompt-{pi}"] = pct_val(misclass, len(non_target))

    # Clean accuracy by emotion
    stats["clean_accuracy_by_emotion"] = {}
    for emo, recs in sorted(by_emo.items()):
        n = len(recs)
        per_prompt = {}
        for pi in range(1, 4):
            key = f"pred_clean_{pi}"
            correct = sum(1 for r in recs if r[key] == r["ground_truth_emotion"])
            per_prompt[f"Prompt-{pi}"] = pct_val(correct, n)
        stats["clean_accuracy_by_emotion"][emo] = per_prompt

    # --- F. LLM Judge ---
    judged = [r for r in records if r["llm_judge_result"] in ("consistent", "inconsistent")]
    if judged:
        consistent_cnt = sum(1 for r in judged if r["llm_judge_result"] == "consistent")
        stats["llm_judge_total"] = len(judged)
        stats["llm_judge_consistent_rate"] = pct_val(consistent_cnt, len(judged))
        # 与 cosine sim 的对比
        cos_agree = sum(
            1 for r in judged
            if (r["llm_judge_result"] == "consistent") == r["semantic_preserved"]
        )
        stats["llm_judge_cosine_agreement"] = pct_val(cos_agree, len(judged))
    else:
        stats["llm_judge_total"] = 0

    return stats


# ---------------------------------------------------------------------------
# 4. CSV 导出
# ---------------------------------------------------------------------------

def export_csv(records: list[dict[str, Any]], output_path: Path):
    """导出清洗后的数据为 CSV。"""
    if not records:
        return
    fieldnames = [k for k in records[0].keys() if k not in ("asr_text_clean", "asr_text_adv")]
    # 把文本字段放最后
    fieldnames += ["asr_text_clean", "asr_text_adv"]

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"[OK] CSV 已导出: {output_path} ({len(records)} 行)")


# ---------------------------------------------------------------------------
# 5. Markdown 报告生成
# ---------------------------------------------------------------------------

def fmt(val: float, decimals: int = 2) -> str:
    if isinstance(val, float) and math.isnan(val):
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def generate_report(
    stats: dict[str, Any],
    warnings: list[str],
    records: list[dict[str, Any]],
    output_path: Path,
):
    """生成 Markdown 格式的统计报告。"""
    N = stats["total_samples"]
    lines: list[str] = []
    w = lines.append

    w("# ESDfinal 白盒对抗攻击实验统计报告")
    w("")
    w(f"> 自动生成 | 样本总数: **{N}** | 说话人: **{len(stats['asr_by_speaker'])}** | "
      f"源情绪: **{len(stats['asr_by_emotion'])}** 类")
    w("")

    # ===== A. 攻击成功率 =====
    w("## A. 攻击成功率 (Attack Success Rate)")
    w("")
    w("### A.1 总体成功率")
    w("")
    w("| 指标 | 值 |")
    w("|------|-----|")
    w(f"| Overall ASR (多数投票) | **{fmt(stats['overall_asr'])}%** |")
    w(f"| Prompt-1 ASR | {fmt(stats['asr_prompt_1'])}% |")
    w(f"| Prompt-2 ASR | {fmt(stats['asr_prompt_2'])}% |")
    w(f"| Prompt-3 ASR | {fmt(stats['asr_prompt_3'])}% |")
    w(f"| Joint ASR (3-prompt 全部成功) | {fmt(stats['asr_joint'])}% |")
    w("")

    w("### A.2 按源情绪分组")
    w("")
    w("| 源情绪 | 样本数 | 成功数 | ASR |")
    w("|--------|--------|--------|-----|")
    for emo, d in sorted(stats["asr_by_emotion"].items()):
        w(f"| {emo} | {d['n']} | {d['success']} | {fmt(d['rate'])}% |")
    w("")

    w("### A.3 按说话人分组")
    w("")
    w("| 说话人 | 样本数 | 成功数 | ASR |")
    w("|--------|--------|--------|-----|")
    for spk, d in sorted(stats["asr_by_speaker"].items()):
        w(f"| {spk} | {d['n']} | {d['success']} | {fmt(d['rate'])}% |")
    w("")

    w("### A.4 源情绪 × 说话人交叉表")
    w("")
    emotions = sorted(stats["asr_by_emotion"].keys())
    speakers = sorted(stats["asr_by_speaker"].keys())
    header = "| 源情绪 | " + " | ".join(speakers) + " |"
    sep = "|--------|" + "|".join(["------"] * len(speakers)) + "|"
    w(header)
    w(sep)
    for emo in emotions:
        cells = []
        for spk in speakers:
            key = f"{emo}_{spk}"
            d = stats["asr_by_emotion_speaker"].get(key)
            if d:
                cells.append(f"{fmt(d['rate'])}%")
            else:
                cells.append("—")
        w(f"| {emo} | " + " | ".join(cells) + " |")
    w("")

    # ===== B. 语义保持 =====
    w("## B. 语义保持指标")
    w("")
    w("### B.1 总体语义保持")
    w("")
    w("| 指标 | 值 |")
    w("|------|-----|")
    w(f"| Semantic Preservation Rate | **{fmt(stats['semantic_preserve_rate'])}%** |")
    w(f"| Cosine Similarity 均值 | {fmt(stats['semantic_sim_mean'], 4)} |")
    w(f"| Cosine Similarity 标准差 | {fmt(stats['semantic_sim_std'], 4)} |")
    w(f"| Cosine Similarity 中位数 | {fmt(stats['semantic_sim_median'], 4)} |")
    w(f"| WER 均值 | {fmt(stats['wer_mean'], 4)} |")
    w(f"| WER 标准差 | {fmt(stats['wer_std'], 4)} |")
    w(f"| WER = 0 比例 | {fmt(stats['wer_zero_rate'])}% |")
    w(f"| 联合成功率 (ASR ∧ Semantic) | **{fmt(stats['joint_success_emo_semantic'])}%** |")
    w("")

    w("### B.2 语义保持按源情绪分组")
    w("")
    w("| 源情绪 | 样本数 | Preserve Rate | Sim 均值 | WER 均值 |")
    w("|--------|--------|---------------|----------|----------|")
    for emo, d in sorted(stats["semantic_by_emotion"].items()):
        w(f"| {emo} | {d['n']} | {fmt(d['preserve_rate'])}% | {fmt(d['sim_mean'], 4)} | {fmt(d['wer_mean'], 4)} |")
    w("")

    # ===== C. 扰动幅度 =====
    w("## C. 扰动幅度指标")
    w("")
    w("### C.1 总体扰动统计")
    w("")
    w("| 指标 | 均值 | 标准差 | 中位数 |")
    w("|------|------|--------|--------|")
    w(f"| SNR (dB) | {fmt(stats['snr_mean'])} | {fmt(stats['snr_std'])} | {fmt(stats['snr_median'])} |")
    w(f"| L∞ | {fmt(stats['linf_mean'], 6)} | {fmt(stats['linf_std'], 6)} | — |")
    w(f"| L2 | {fmt(stats['l2_mean'], 4)} | {fmt(stats['l2_std'], 4)} | — |")
    w("")

    w("### C.2 SNR 分段统计")
    w("")
    w("| SNR 区间 | 样本数 | 比例 |")
    w("|----------|--------|------|")
    for label, lo, hi in SNR_BINS:
        d = stats["snr_bins"].get(label, {})
        w(f"| {label} | {d.get('count', 0)} | {fmt(d.get('pct', 0))}% |")
    w("")

    w("### C.3 扰动按源情绪分组")
    w("")
    w("| 源情绪 | SNR 均值 (dB) | L∞ 均值 | L2 均值 |")
    w("|--------|---------------|---------|---------|")
    for emo, d in sorted(stats["perturbation_by_emotion"].items()):
        w(f"| {emo} | {fmt(d['snr_mean'])} | {fmt(d['linf_mean'], 6)} | {fmt(d['l2_mean'], 4)} |")
    w("")

    # ===== D. 优化收敛 =====
    w("## D. 优化收敛指标")
    w("")
    w("| 指标 | 值 |")
    w("|------|-----|")
    w(f"| Final Emo Loss 均值 | {fmt(stats['final_emo_loss_mean'], 4)} |")
    w(f"| Final Emo Loss 标准差 | {fmt(stats['final_emo_loss_std'], 4)} |")
    w(f"| Final ASR Loss 均值 | {fmt(stats['final_asr_loss_mean'], 4)} |")
    w(f"| Final Perceptual Loss 均值 | {fmt(stats['final_per_loss_mean'], 4)} |")
    w(f"| Loss 下降率 均值 | {fmt(stats['loss_reduction_mean'], 4)} |")
    w(f"| Loss 下降率 标准差 | {fmt(stats['loss_reduction_std'], 4)} |")
    w(f"| 收敛率 (emo_loss < 1.0) | {fmt(stats['converge_rate'])}% |")
    w(f"| 收敛步数 均值 | {fmt(stats['converge_step_mean'], 1)} |")
    w(f"| 收敛步数 中位数 | {fmt(stats['converge_step_median'], 1)} |")
    w("")

    # ===== E. Clean Baseline =====
    w("## E. Clean Baseline 指标")
    w("")
    w("### E.1 干净音频情绪识别准确率")
    w("")
    w("| Prompt | Clean Accuracy |")
    w("|--------|----------------|")
    for p, v in stats["clean_accuracy"].items():
        w(f"| {p} | {fmt(v)}% |")
    w("")

    w("### E.2 干净音频误判为目标情绪的比例")
    w("")
    w("| Prompt | Clean → Target Rate |")
    w("|--------|---------------------|")
    for p, v in stats["clean_to_target"].items():
        w(f"| {p} | {fmt(v)}% |")
    w("")

    w("### E.3 Clean Accuracy 按源情绪分组")
    w("")
    header_e3 = "| 源情绪 | Prompt-1 | Prompt-2 | Prompt-3 |"
    w(header_e3)
    w("|--------|----------|----------|----------|")
    for emo, d in sorted(stats["clean_accuracy_by_emotion"].items()):
        w(f"| {emo} | {fmt(d.get('Prompt-1', 0))}% | {fmt(d.get('Prompt-2', 0))}% | {fmt(d.get('Prompt-3', 0))}% |")
    w("")

    # ===== F. LLM Judge =====
    if stats.get("llm_judge_total", 0) > 0:
        w("## F. LLM Judge 语义一致性评估")
        w("")
        w("| 指标 | 值 |")
        w("|------|-----|")
        w(f"| 评估样本数 | {stats['llm_judge_total']} |")
        w(f"| LLM 判定语义一致比例 | {fmt(stats['llm_judge_consistent_rate'])}% |")
        w(f"| LLM Judge 与 Cosine Sim 一致率 | {fmt(stats['llm_judge_cosine_agreement'])}% |")
        w("")
    else:
        w("## F. LLM Judge 语义一致性评估")
        w("")
        w("> 未启用 LLM Judge。如需启用，请运行: `python analyze_results.py --llm-judge --api-key YOUR_KEY`")
        w("")

    # ===== 数据质量 =====
    w("## G. 数据质量报告")
    w("")
    w(f"| 项目 | 值 |")
    w("|------|-----|")
    w(f"| 成功加载样本数 | {N} |")
    w(f"| 清洗警告数 | {len(warnings)} |")
    w("")

    if warnings:
        w("### 清洗警告详情（前 50 条）")
        w("")
        for wn in warnings[:50]:
            w(f"- {wn}")
        if len(warnings) > 50:
            w(f"- ... 共 {len(warnings)} 条警告")
        w("")

    # 写入文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] 报告已生成: {output_path}")


# ---------------------------------------------------------------------------
# 6. 主函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ESDfinal 实验结果分析")
    parser.add_argument("--result-dir", type=str, default=str(RESULT_DIR),
                        help="结果目录路径")
    parser.add_argument("--llm-judge", action="store_true",
                        help="启用 LLM Judge 语义一致性评估")
    parser.add_argument("--api-key", type=str, default=os.environ.get("OPENAI_API_KEY", ""),
                        help="OpenAI API Key")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="LLM Judge 使用的模型")
    parser.add_argument("--base-url", type=str, default=None,
                        help="自定义 API base URL（兼容 OpenAI 格式的第三方服务）")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    print(f"[INFO] 数据目录: {result_dir}")

    # 加载与清洗
    print("[1/5] 加载与清洗数据...")
    records, warnings = load_and_clean(result_dir)
    print(f"  加载 {len(records)} 条记录，{len(warnings)} 条警告")

    # LLM Judge
    if args.llm_judge:
        if not args.api_key:
            print("[WARN] 未提供 API Key，跳过 LLM Judge。请通过 --api-key 或 OPENAI_API_KEY 环境变量提供。")
        else:
            print("[2/5] 运行 LLM Judge 语义一致性评估...")
            records = llm_judge_batch(
                records, args.api_key, args.model, args.base_url
            )
    else:
        print("[2/5] LLM Judge 未启用（使用 --llm-judge 启用）")

    # 统计
    print("[3/5] 计算统计指标...")
    stats = compute_stats(records)

    # 导出 CSV
    csv_path = result_dir / "cleaned_data.csv"
    print(f"[4/5] 导出 CSV: {csv_path}")
    export_csv(records, csv_path)

    # 生成报告
    report_path = result_dir / "report.md"
    print(f"[5/5] 生成报告: {report_path}")
    generate_report(stats, warnings, records, report_path)

    print("\n[DONE] 分析完成！")
    print(f"  - 清洗数据: {csv_path}")
    print(f"  - 统计报告: {report_path}")


if __name__ == "__main__":
    main()
