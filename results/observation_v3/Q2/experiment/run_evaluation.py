"""
Observation 3: LLM Judge Evaluation (DeepSeek V3.2)

Uses DeepSeek as LLM Judge to evaluate Aligned vs Conflict responses
across three dimensions: Faithfulness, Empathy, Relevance.

Usage:
    # Dry-run (print prompts, no API calls)
    python run_evaluation.py --config eval_config.json --dry-run

    # Run evaluation
    python run_evaluation.py --config eval_config.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import requests

# ── Judge Prompt Templates ────────────────────────────────────────────────────
# Faithfulness & Relevance: only source_text + response (NO emotion labels)
# Empathy: only true_emotion + response (NO misled_emotion, NO source_text)

FAITHFULNESS_PROMPT = """\
You are an expert evaluator assessing the **faithfulness** of a conversational response.

**Definition:** Faithfulness measures whether the response is faithful to the speaker's actual spoken content. A faithful response only makes claims that are supported by or can be reasonably inferred from the source text. Introducing assumptions, scenarios, concerns, or facts not present in the source text is considered unfaithful.

**Scoring rubric (1-5):**
- 5: Completely faithful. All claims in the response can be derived from the source text.
- 4: Mostly faithful. Minor reasonable inferences but no fabrications.
- 3: Partially faithful. A few assumptions or claims not present in the source text are introduced.
- 2: Mostly unfaithful. Multiple unsupported claims or fabricated details.
- 1: Completely unfaithful. The response is largely unrelated to the source text or contains extensive fabrications.

**Source text (what the speaker actually said):**
{source_text}

**Response to evaluate:**
{response}

Evaluate the faithfulness of the response. Output ONLY a JSON object in this exact format, with no other text:
{{"score": <integer 1-5>, "reason": "<brief explanation>"}}"""

EMPATHY_PROMPT = """\
You are an expert evaluator assessing the **empathic appropriateness** of a conversational response.

**Definition:** Empathy measures whether the emotional tone and attitude expressed in the response appropriately matches the speaker's true emotional state. An empathically appropriate response correctly identifies and responds to the speaker's emotions with suitable concern, tone, and strategy.

**Scoring rubric (1-5):**
- 5: Perfectly empathic. The response accurately captures the speaker's true emotion and provides a fitting emotional response.
- 4: Mostly empathic. The emotional direction is correct but with minor tonal mismatches.
- 3: Partially empathic. Some recognition of the speaker's emotion but the response strategy is somewhat misaligned.
- 2: Mostly inappropriate. The response misreads the speaker's emotional state.
- 1: Completely inappropriate. The emotional response is opposite to or incompatible with the speaker's true emotion.

**Speaker's true emotional state:** {true_emotion}

**Response to evaluate:**
{response}

Evaluate the empathic appropriateness of the response. Output ONLY a JSON object in this exact format, with no other text:
{{"score": <integer 1-5>, "reason": "<brief explanation>"}}"""

RELEVANCE_PROMPT = """\
You are an expert evaluator assessing the **relevance** of a conversational response.

**Definition:** Relevance measures whether the response addresses the topic and content that the speaker actually talked about. A relevant response stays on-topic and directly engages with what the speaker expressed. Drifting to unrelated topics or focusing on issues the speaker did not raise is considered irrelevant.

**Scoring rubric (1-5):**
- 5: Completely relevant. The response directly addresses what the speaker said.
- 4: Mostly relevant. The response primarily addresses the speaker's content with minor tangents.
- 3: Partially relevant. The response touches on the speaker's content but the focus drifts elsewhere.
- 2: Mostly irrelevant. The response primarily discusses topics the speaker did not raise.
- 1: Completely irrelevant. The response has no connection to what the speaker actually said.

**Source text (what the speaker actually said):**
{source_text}

**Response to evaluate:**
{response}

Evaluate the relevance of the response. Output ONLY a JSON object in this exact format, with no other text:
{{"score": <integer 1-5>, "reason": "<brief explanation>"}}"""

PROMPT_TEMPLATES = {
    "faithfulness": FAITHFULNESS_PROMPT,
    "empathy": EMPATHY_PROMPT,
    "relevance": RELEVANCE_PROMPT,
}


# ── API Call ──────────────────────────────────────────────────────────────────


def call_deepseek(
    api_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_retries: int = 3,
    call_interval: float = 1.0,
) -> dict[str, Any]:
    """Call DeepSeek API with retry, return parsed {"score": int, "reason": str}."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 256,
    }

    for attempt in range(1, max_retries + 1):
        try:
            time.sleep(call_interval)
            resp = requests.post(
                f"{api_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()

            # Parse JSON from response (tolerant of surrounding text)
            json_match = re.search(r"\{[^}]+\}", content)
            if json_match:
                result = json.loads(json_match.group())
                if "score" in result and "reason" in result:
                    score = int(result["score"])
                    if 1 <= score <= 5:
                        return {"score": score, "reason": str(result["reason"])}
                    logging.warning(
                        f"  Attempt {attempt}: score {score} out of range [1,5]"
                    )
                else:
                    logging.warning(
                        f"  Attempt {attempt}: missing keys in: {content[:120]}"
                    )
            else:
                logging.warning(
                    f"  Attempt {attempt}: no JSON found in: {content[:120]}"
                )
        except requests.exceptions.RequestException as e:
            logging.warning(f"  Attempt {attempt}: API error: {e}")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logging.warning(f"  Attempt {attempt}: parse error: {e}")

    logging.error("  All retries exhausted, returning score=-1")
    return {"score": -1, "reason": "API call failed after all retries"}


# ── Build Judge Prompt ────────────────────────────────────────────────────────


def build_judge_prompt(
    dimension: str,
    response: str,
    source_text: str,
    true_emotion: str,
) -> str:
    """Build the judge prompt for a given dimension."""
    template = PROMPT_TEMPLATES[dimension]
    if dimension == "empathy":
        return template.format(true_emotion=true_emotion, response=response)
    else:
        return template.format(source_text=source_text, response=response)


# ── Main ──────────────────────────────────────────────────────────────────────

DIMENSIONS = ["faithfulness", "empathy", "relevance"]
CONDITIONS = ["aligned", "conflict"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Observation 3: LLM Judge Evaluation"
    )
    parser.add_argument("--config", required=True, help="Path to eval config JSON")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print prompts without calling API"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # ── Load config ───────────────────────────────────────────────────────
    with open(args.config, "r", encoding="utf-8-sig") as f:
        config = json.load(f)

    api_url = config["api_url"]
    api_key = config["api_key"]
    model = config["model"]
    results_path = Path(config["results_path"])
    text_jsonl_path = Path(config["text_jsonl_path"])
    output_path = Path(config["output_path"])
    max_retries = config.get("retry_max", 3)
    call_interval = config.get("call_interval", 1.0)

    # ── Load inference results ────────────────────────────────────────────
    with open(results_path, "r", encoding="utf-8-sig") as f:
        results = json.load(f)

    # ── Load source text mapping: id -> text ──────────────────────────────
    text_map: dict[str, str] = {}
    with open(text_jsonl_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            text_map[entry["id"]] = entry["text"]

    logging.info(f"Loaded {len(results)} results, {len(text_map)} text entries")

    # ── Evaluate ──────────────────────────────────────────────────────────
    total_calls = len(results) * len(DIMENSIONS) * len(CONDITIONS)
    call_count = 0

    for i, rec in enumerate(results):
        # "angry/t030.wav" -> "t030"
        sample_id = Path(rec["audio_file"]).stem
        source_text = text_map.get(sample_id)

        if source_text is None:
            logging.error(
                f"Cannot find source_text for {rec['audio_file']} (id={sample_id}), skipping"
            )
            continue

        true_emotion = rec["true_emotion"]
        logging.info(
            f"[{i + 1}/{len(results)}] {rec['audio_file']}  "
            f"(true={true_emotion}, source={source_text[:50]}...)"
        )

        for condition in CONDITIONS:
            response = rec[f"{condition}_response"]

            for dim in DIMENSIONS:
                call_count += 1
                prompt = build_judge_prompt(
                    dimension=dim,
                    response=response,
                    source_text=source_text,
                    true_emotion=true_emotion,
                )

                if args.dry_run:
                    logging.info(
                        f"  [{call_count}/{total_calls}] {condition}_{dim} DRY RUN"
                    )
                    logging.info(f"    Prompt: {prompt[:150]}...")
                    rec[f"{condition}_{dim}_score"] = None
                    rec[f"{condition}_{dim}_reason"] = None
                    continue

                logging.info(f"  [{call_count}/{total_calls}] {condition}_{dim}")
                result = call_deepseek(
                    api_url=api_url,
                    api_key=api_key,
                    model=model,
                    prompt=prompt,
                    max_retries=max_retries,
                    call_interval=call_interval,
                )
                rec[f"{condition}_{dim}_score"] = result["score"]
                rec[f"{condition}_{dim}_reason"] = result["reason"]
                logging.info(
                    f"    Score: {result['score']} | {result['reason'][:80]}"
                )

    # ── Save results ──────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logging.info(f"Evaluation results saved to {output_path}")

    # ── Print summary statistics ──────────────────────────────────────────
    if not args.dry_run:
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)

        for dim in DIMENSIONS:
            aligned_scores = [
                r[f"aligned_{dim}_score"]
                for r in results
                if r.get(f"aligned_{dim}_score", -1) is not None
                and r.get(f"aligned_{dim}_score", -1) > 0
            ]
            conflict_scores = [
                r[f"conflict_{dim}_score"]
                for r in results
                if r.get(f"conflict_{dim}_score", -1) is not None
                and r.get(f"conflict_{dim}_score", -1) > 0
            ]

            if aligned_scores and conflict_scores:
                avg_a = sum(aligned_scores) / len(aligned_scores)
                avg_c = sum(conflict_scores) / len(conflict_scores)
                diff = avg_a - avg_c
                print(f"\n  {dim.upper()}:")
                print(f"    Aligned avg:  {avg_a:.2f}  (n={len(aligned_scores)})")
                print(f"    Conflict avg: {avg_c:.2f}  (n={len(conflict_scores)})")
                print(f"    Difference:   {diff:+.2f}  (Aligned - Conflict)")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
