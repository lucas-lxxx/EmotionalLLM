"""Generate the final black-box experiment report in Markdown."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from analyze import (
    analyze_clean_vs_adv_vs_noise,
    analyze_language_comparison,
    build_asr_matrix,
    build_per_emotion_matrix,
    collect_all_summaries,
)
from config import cfg
from sample_loader import load_whitebox_results, select_subset


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return ""
    return f"{value * 100:.{digits}f}%"


def _fmt_count(value: int | None) -> str:
    if value is None:
        return ""
    return str(value)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _selected_sample_counts() -> dict[str, int]:
    counts = {}
    for surrogate_key in cfg.surrogate_groups:
        result_dir = cfg.get_surrogate_dir(surrogate_key)
        samples = load_whitebox_results(result_dir)
        subset = select_subset(samples, per_emotion=cfg.per_emotion, only_whitebox_success=True, seed=42)
        counts[surrogate_key] = len(subset)
    return counts


def _filter_completed_summaries(summaries: dict, sample_counts: dict[str, int]) -> dict:
    completed = {}
    for key, summary in summaries.items():
        parts = key.split("/")
        if len(parts) != 3:
            continue
        _, surrogate_key, _ = parts
        if summary.get("total_samples") == sample_counts.get(surrogate_key):
            completed[key] = summary
    return completed


def _average_from_matrix(matrix: dict[str, dict[str, float | None]], target_key: str) -> float | None:
    values = [matrix[s][target_key] for s in matrix if matrix[s].get(target_key) is not None]
    if not values:
        return None
    return float(np.mean(values))


def _demo_rows(summaries: dict) -> list[list[str]]:
    demo_specs = [
        ("Gemini 2.5 Flash", Path("results/gemini/summary.json"), "adv/voxtral_en/gemini_flash"),
        ("Qwen3-Omni-Flash", Path("results/qwen/summary.json"), "adv/voxtral_en/qwen3_omni"),
    ]
    rows = []
    for model_name, rel_path, final_key in demo_specs:
        demo = _load_json(cfg.blackbox_root / rel_path)
        final_summary = summaries.get(final_key)
        demo_asr = demo.get("transfer_asr") if demo else None
        demo_n = demo.get("total_samples") if demo else None
        final_asr = final_summary.get("transfer_asr") if final_summary else None
        final_n = final_summary.get("total_samples") if final_summary else None
        delta = None
        if demo_asr is not None and final_asr is not None:
            delta = final_asr - demo_asr
        rows.append([
            model_name,
            _fmt_count(demo_n),
            _pct(demo_asr),
            _fmt_count(final_n),
            _pct(final_asr),
            _pct(delta),
        ])
    rows.append([
        "OpenAI gpt-audio",
        "",
        "",
        "",
        "",
        "待 OpenAI key",
    ])
    return rows


def _main_matrix_rows(matrix: dict[str, dict[str, float | None]]) -> list[list[str]]:
    rows = []
    for surrogate_key in cfg.surrogate_groups:
        rows.append(
            [cfg.surrogate_groups[surrogate_key]["name"]]
            + [_pct(matrix[surrogate_key].get(target_key)) for target_key in cfg.target_list]
        )

    avg_row = ["Average"] + [_pct(_average_from_matrix(matrix, target_key)) for target_key in cfg.target_list]
    rows.append(avg_row)
    return rows


def _coverage_rows(summaries: dict, sample_counts: dict[str, int]) -> list[list[str]]:
    rows = []
    for surrogate_key in cfg.surrogate_groups:
        for target_key in cfg.target_list:
            adv = summaries.get(f"adv/{surrogate_key}/{target_key}")
            clean = summaries.get(f"clean/{surrogate_key}/{target_key}")
            noise = summaries.get(f"noise/{surrogate_key}/{target_key}")
            rows.append([
                cfg.surrogate_groups[surrogate_key]["name"],
                cfg.target_list[target_key]["name"],
                str(sample_counts[surrogate_key]),
                _fmt_count(adv.get("total_samples") if adv else None),
                _fmt_count(clean.get("total_samples") if clean else None),
                _fmt_count(noise.get("total_samples") if noise else None),
            ])
    return rows


def _baseline_rows(three_way: dict[str, dict]) -> list[list[str]]:
    rows = []
    for combo_key in sorted(three_way):
        surrogate_key, target_key = combo_key.split("/")
        stats = three_way[combo_key]
        rows.append([
            cfg.surrogate_groups[surrogate_key]["name"],
            cfg.target_list[target_key]["name"],
            _pct(stats.get("clean_accuracy")),
            _pct(stats.get("noise_accuracy")),
            _pct(stats.get("adv_asr")),
            _pct(stats.get("clean_target_rate")),
            _pct(stats.get("noise_target_rate")),
        ])
    return rows


def _per_emotion_rows(per_emotion: dict[str, dict[str, float]]) -> list[list[str]]:
    emotions = ["angry", "sad", "neutral", "surprise"]
    rows = []
    for combo_key in sorted(per_emotion):
        surrogate_key, target_key = combo_key.split("/")
        rows.append(
            [cfg.surrogate_groups[surrogate_key]["name"], cfg.target_list[target_key]["name"]]
            + [_pct(per_emotion[combo_key].get(emotion)) for emotion in emotions]
        )
    return rows


def _language_rows(comparisons: dict) -> list[list[str]]:
    rows = []
    for surrogate_base, data in comparisons.items():
        for target_key in cfg.target_list:
            rows.append([
                surrogate_base,
                cfg.target_list[target_key]["name"],
                _pct(data["EN"].get(target_key)),
                _pct(data["CN"].get(target_key)),
                _pct(
                    None
                    if data["EN"].get(target_key) is None or data["CN"].get(target_key) is None
                    else data["CN"][target_key] - data["EN"][target_key]
                ),
            ])
    return rows


def _key_findings(matrix: dict[str, dict[str, float | None]], three_way: dict[str, dict]) -> list[str]:
    findings = []
    available = [
        (surrogate_key, target_key, value)
        for surrogate_key, row in matrix.items()
        for target_key, value in row.items()
        if value is not None
    ]
    if available:
        best_surrogate, best_target, best_value = max(available, key=lambda item: item[2])
        findings.append(
            f"- 最高迁移 ASR 目前来自 {cfg.surrogate_groups[best_surrogate]['name']} → {cfg.target_list[best_target]['name']}，为 {_pct(best_value)}。"
        )

    avg_by_target = []
    for target_key in cfg.target_list:
        avg_value = _average_from_matrix(matrix, target_key)
        if avg_value is not None:
            avg_by_target.append((target_key, avg_value))
    if avg_by_target:
        best_target, best_avg = max(avg_by_target, key=lambda item: item[1])
        findings.append(f"- 按 surrogate 平均后，最脆弱的目标模型是 {cfg.target_list[best_target]['name']}，平均 ASR 为 {_pct(best_avg)}。")

    comparable = [(combo, stats) for combo, stats in three_way.items() if stats.get("clean_accuracy") is not None and stats.get("adv_asr") is not None]
    if comparable:
        largest_gap_combo, largest_gap_stats = max(
            comparable,
            key=lambda item: (item[1]["clean_accuracy"] - item[1]["adv_asr"]),
        )
        surrogate_key, target_key = largest_gap_combo.split("/")
        gap = largest_gap_stats["clean_accuracy"] - largest_gap_stats["adv_asr"]
        findings.append(
            f"- clean 与 adversarial 的最大落差出现在 {cfg.surrogate_groups[surrogate_key]['name']} → {cfg.target_list[target_key]['name']}，下降 {_pct(gap)}。"
        )

    findings.append("- OpenAI gpt-audio 列保留为空，等待后续取得 API key 后补实验。")
    return findings


def generate_report() -> Path:
    sample_counts = _selected_sample_counts()
    summaries = collect_all_summaries()
    completed_summaries = _filter_completed_summaries(summaries, sample_counts)
    matrix = build_asr_matrix(completed_summaries)
    per_emotion = build_per_emotion_matrix(completed_summaries)
    comparisons = analyze_language_comparison(completed_summaries)
    three_way = analyze_clean_vs_adv_vs_noise(completed_summaries)

    completed_summary_count = len(summaries)
    expected_summary_count = len(cfg.surrogate_groups) * len(cfg.target_list) * 3

    report_lines = [
        "# 黑盒迁移攻击最终实验报告",
        "",
        f"> 更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"> 结构化 summary 完成度：{completed_summary_count}/{expected_summary_count}",
        f"> 主表目标模型：{', '.join(cfg.target_list[target]['name'] for target in cfg.target_list)}",
        "",
        "## 1. 实验范围",
        "",
        _markdown_table(
            ["Surrogate", "Language", "Selected samples", "Source directory"],
            [
                [
                    cfg.surrogate_groups[surrogate_key]["name"],
                    cfg.surrogate_groups[surrogate_key]["lang"],
                    str(sample_counts[surrogate_key]),
                    str(cfg.get_surrogate_dir(surrogate_key)),
                ]
                for surrogate_key in cfg.surrogate_groups
            ],
        ),
        "",
        _markdown_table(
            ["Target", "Client", "Status"],
            [
                [cfg.target_list[target_key]["name"], cfg.target_list[target_key]["client"], "ready" if target_key != "gpt4o_audio" else "pending key"]
                for target_key in cfg.target_list
            ],
        ),
        "",
        "## 2. Demo 与最终批量结果对齐",
        "",
        _markdown_table(
            ["Model", "Demo n", "Demo ASR", "Final n", "Final ASR", "Delta"],
            _demo_rows(completed_summaries),
        ),
        "",
        "## 3. 最终批量主结果",
        "",
        _markdown_table(
            ["Surrogate"] + [cfg.target_list[target_key]["name"] for target_key in cfg.target_list],
            _main_matrix_rows(matrix),
        ),
        "",
        "## 4. 运行覆盖率",
        "",
        _markdown_table(
            ["Surrogate", "Target", "Planned n", "Adv n", "Clean n", "Noise n"],
            _coverage_rows(summaries, sample_counts),
        ),
        "",
        "## 5. Baseline 对比",
        "",
        _markdown_table(
            ["Surrogate", "Target", "Clean acc", "Noise acc", "Adv ASR", "Clean target rate", "Noise target rate"],
            _baseline_rows(three_way),
        ),
        "",
        "## 6. Per-Emotion Transfer ASR",
        "",
        _markdown_table(
            ["Surrogate", "Target", "Angry", "Sad", "Neutral", "Surprise"],
            _per_emotion_rows(per_emotion),
        ),
        "",
        "## 7. 语言对比",
        "",
        _markdown_table(
            ["Surrogate family", "Target", "EN ASR", "CN ASR", "CN-EN"],
            _language_rows(comparisons),
        ),
        "",
        "## 8. 关键结论",
        "",
        *_key_findings(matrix, three_way),
        "",
        "## 9. 交付物",
        "",
        "- 主结果 summary 根目录在 `blackbox/results/{adv,clean,noise}/.../summary.json`。",
        "- Demo 原始 summary 保留在 `blackbox/results/gemini/summary.json` 和 `blackbox/results/qwen/summary.json`。",
        "- 图表输出目录为 `blackbox/figures/`，同时会复制到 `finalpaper/figure/`。",
        "- 本报告由 `blackbox/generate_report.py` 生成，便于后续补跑 OpenAI 列后直接刷新。",
        "",
    ]

    report_path = cfg.blackbox_root / "report.md"
    report_path.write_text("\n".join(report_lines).strip() + "\n", encoding="utf-8")
    return report_path


if __name__ == "__main__":
    path = generate_report()
    print(f"Report written to {path}")
