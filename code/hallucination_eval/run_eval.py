#!/usr/bin/env python3
"""
Hallucination evaluation for adversarial audio.
Generates binary QA probes, runs model inference, saves per-sample responses.

Usage:
  python run_eval.py --model voxtral --dataset iemocap --gpu 2
  python run_eval.py --model meralion --dataset ravdess --gpu 2
  python run_eval.py --model opens2s --dataset iemocap --gpu 2
"""
import argparse
import json
import os
import sys
import glob
import re
from pathlib import Path

# ============================================================
# Configuration
# ============================================================
BASE_DIR = Path("/data1/lixiang/EmotionalLLM")
CODE_DIR = BASE_DIR / "code"

MODEL_CONFIGS = {
    "voxtral": {
        "model_path": "/data1/lixiang/Voxtral",
        "code_dir": CODE_DIR / "white_box_voxtral",
        "result_dirs": {
            "iemocap": CODE_DIR / "white_box_voxtral/result/Voxtral_IEMOCAP",
            "ravdess": CODE_DIR / "white_box_voxtral/result/Voxtral_RAVDESS",
            "esd_en": CODE_DIR / "white_box_voxtral/result/Voxtral_EN",
            "esd_cn": CODE_DIR / "white_box_voxtral/result/Voxtral_CN",
        },
        "sample_rate": 16000,
    },
    "meralion": {
        "model_path": "/data1/lixiang/MERaLiON-2-3B",
        "code_dir": CODE_DIR / "white_box_meralion",
        "result_dirs": {
            "iemocap": CODE_DIR / "white_box_meralion/result/MERaLiON_IEMOCAP",
            "ravdess": CODE_DIR / "white_box_meralion/result/MERaLiON_RAVDESS",
            "esd_en": CODE_DIR / "white_box_meralion/result/MERaLiON_EN",
            "esd_cn": CODE_DIR / "white_box_meralion/result/MERaLiON_CN",
        },
        "sample_rate": 16000,
    },
    "opens2s": {
        "model_path": "/data1/lixiang/Opens2s/OpenS2S/models/OpenS2S",
        "opens2s_root": "/data1/lixiang/Opens2s/OpenS2S",
        "code_dir": CODE_DIR / "white_box_opens2s_v2/ver2.0",
        "result_dirs": {
            "iemocap": CODE_DIR / "white_box_opens2s_v2/result/IEMOCAP",
            "ravdess": CODE_DIR / "white_box_opens2s_v2/result/RAVDESS",
        },
        "sample_rate": 16000,
    },
}

DATASET_LANGUAGE = {
    "iemocap": "English",
    "ravdess": "English",
    "esd_en": "English",
    "esd_cn": "Chinese",
}

WRONG_LANGUAGE = {"English": "Chinese", "Chinese": "English"}

EMOTION_ADJECTIVES = {
    "happy": "happy and cheerful",
    "sad": "sad and melancholic",
    "angry": "angry and aggressive",
    "neutral": "calm and neutral",
    "surprise": "surprised and astonished",
}

MAX_NEW_TOKENS = 32
TEMPERATURE = 0.0
MAX_SAMPLES = 60  # Use at most 60 samples per dataset for consistency


# ============================================================
# QA Probe Generation
# ============================================================
def generate_qa_probes(sample_meta: dict, language: str) -> list[dict]:
    """Generate 6 binary QA probes for a sample."""
    gt_emo = sample_meta["ground_truth_emotion"]
    tgt_emo = sample_meta["target_emotion"]
    correct_lang = language
    wrong_lang = WRONG_LANGUAGE[language]
    gt_adj = EMOTION_ADJECTIVES.get(gt_emo, gt_emo)
    tgt_adj = EMOTION_ADJECTIVES.get(tgt_emo, tgt_emo)

    probes = [
        # Acoustic: emotion perception
        {
            "id": "aco_pos",
            "question": f"Is the speaker expressing {gt_emo} in this audio? Answer only Yes or No.",
            "gt": "Yes",
            "type": "acoustic",
        },
        {
            "id": "aco_neg",
            "question": f"Is the speaker expressing {tgt_emo} in this audio? Answer only Yes or No.",
            "gt": "No",
            "type": "acoustic",
        },
        # Semantic: language perception
        {
            "id": "sem_pos",
            "question": f"Is the speaker speaking in {correct_lang} in this audio? Answer only Yes or No.",
            "gt": "Yes",
            "type": "semantic",
        },
        {
            "id": "sem_neg",
            "question": f"Is the speaker speaking in {wrong_lang} in this audio? Answer only Yes or No.",
            "gt": "No",
            "type": "semantic",
        },
        # SA-Confusion: cross-modal consistency
        {
            "id": "sac_pos",
            "question": f"Does the speaker's tone of voice sound {gt_adj}? Answer only Yes or No.",
            "gt": "Yes",
            "type": "sa_confusion",
        },
        {
            "id": "sac_neg",
            "question": f"Does the speaker's tone of voice sound {tgt_adj}? Answer only Yes or No.",
            "gt": "No",
            "type": "sa_confusion",
        },
    ]
    return probes


# ============================================================
# Response Classification
# ============================================================
def parse_yes_no(response: str) -> str:
    """Classify a free-text response as Yes/No/Unknown."""
    text = response.strip().lower()
    # Direct start patterns
    if re.match(r"^(yes|yeah|yep|correct|right|true)\b", text):
        return "Yes"
    if re.match(r"^(no|nope|incorrect|wrong|false|not)\b", text):
        return "No"
    # Fallback: check if Yes/No appears anywhere
    if "yes" in text and "no" not in text:
        return "Yes"
    if "no" in text and "yes" not in text:
        return "No"
    # Chinese fallback (built from code points to survive encoding issues)
    YES_CHARS = [chr(0x662f), chr(0x5bf9)]  # shi, dui
    NO_CHARS = [chr(0x4e0d), chr(0x5426)]   # bu, fou
    for c in YES_CHARS:
        if text.startswith(c):
            return "Yes"
    for c in NO_CHARS:
        if text.startswith(c):
            return "No"
    return "Unknown"


# ============================================================
# Model Loading (unified interface)
# ============================================================
class ModelWrapper:
    """Unified interface for all three models."""

    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device
        self._load_model()

    def _load_model(self):
        cfg = MODEL_CONFIGS[self.model_name]
        code_dir = str(cfg["code_dir"])
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)

        if self.model_name == "voxtral":
            from voxtral_io import load_voxtral, decode_text as _decode

            self.model, self.processor, _ = load_voxtral(
                Path(cfg["model_path"]), self.device
            )
            self._decode_fn = _decode
            self.sr = cfg["sample_rate"]

        elif self.model_name == "meralion":
            from meralion_io import load_meralion, decode_text as _decode

            self.model, self.processor, _ = load_meralion(
                Path(cfg["model_path"]), self.device
            )
            self._decode_fn = _decode
            self.sr = cfg["sample_rate"]

        elif self.model_name == "opens2s":
            from opens2s_io import load_opens2s, decode_text as _decode

            self.model, self.tokenizer, self.audio_extractor, _ = load_opens2s(
                Path(cfg["model_path"]),
                self.device,
                Path(cfg["opens2s_root"]),
            )
            self._decode_fn = _decode
            self.sr = cfg["sample_rate"]

        print(f"[OK] Loaded {self.model_name} on {self.device}")

    def decode(self, waveform, sr: int, prompt: str) -> str:
        """Run inference and return text response."""
        import torch
        import torchaudio
        # Resample if needed
        if sr != self.sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.sr)
            sr = self.sr

        waveform = waveform.to(self.device)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        import torch
        with torch.no_grad():
            if self.model_name == "voxtral":
                return self._decode_fn(
                    self.model, self.processor, waveform, sr, prompt,
                    MAX_NEW_TOKENS, TEMPERATURE,
                )
            elif self.model_name == "meralion":
                return self._decode_fn(
                    self.model, self.processor, waveform, sr, prompt,
                    MAX_NEW_TOKENS, TEMPERATURE,
                )
            elif self.model_name == "opens2s":
                return self._decode_fn(
                    self.model, self.tokenizer, waveform, sr, prompt,
                    MAX_NEW_TOKENS, TEMPERATURE,
                    audio_extractor=self.audio_extractor,
                    system_prompt="You are a helpful assistant.",
                )


# ============================================================
# Sample Discovery
# ============================================================
def discover_samples(result_dir: Path, max_samples: int = MAX_SAMPLES) -> list[dict]:
    """Find all per-sample JSON+WAV pairs in a result directory."""
    json_files = sorted(glob.glob(str(result_dir / "**/*.json"), recursive=True))
    json_files = [f for f in json_files if "summary" not in os.path.basename(f)]

    samples = []
    for jf in json_files:
        wav_f = jf.replace(".json", ".wav")
        if not os.path.exists(wav_f):
            continue
        with open(jf) as f:
            meta = json.load(f)
        # Remove large trace data
        meta.pop("loss_trace", None)
        meta.pop("grad_norm_trace", None)
        meta.pop("grad_norm_trace", None)
        samples.append({
            "json_path": jf,
            "adv_wav_path": wav_f,
            "clean_wav_path": meta.get("path", ""),
            "meta": meta,
        })
        if len(samples) >= max_samples:
            break

    return samples


# ============================================================
# Main Evaluation
# ============================================================
def evaluate(model_name: str, dataset: str):
    import torch
    import torchaudio
    device = "cuda:0"

    cfg = MODEL_CONFIGS[model_name]
    if dataset not in cfg["result_dirs"]:
        print(f"[SKIP] {model_name}/{dataset}: no result directory configured")
        return

    result_dir = cfg["result_dirs"][dataset]
    if not result_dir.exists():
        print(f"[SKIP] {model_name}/{dataset}: {result_dir} not found")
        return

    language = DATASET_LANGUAGE[dataset]
    samples = discover_samples(result_dir)
    print(f"[INFO] {model_name}/{dataset}: found {len(samples)} samples, lang={language}")
    if not samples:
        return

    # Output directory
    out_dir = Path(__file__).parent / "results" / f"{model_name}_{dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done
    out_file = out_dir / "hallucination_eval.json"
    if out_file.exists():
        print(f"[SKIP] {out_file} already exists. Delete to re-run.")
        return

    # Load model
    wrapper = ModelWrapper(model_name, device)

    results = []
    for idx, sample in enumerate(samples):
        meta = sample["meta"]
        sample_id = meta.get("sample_id", f"sample_{idx:05d}")
        probes = generate_qa_probes(meta, language)

        print(f"  [{idx+1}/{len(samples)}] {sample_id}")

        sample_result = {
            "sample_id": sample_id,
            "ground_truth_emotion": meta["ground_truth_emotion"],
            "target_emotion": meta["target_emotion"],
            "clean_responses": [],
            "adv_responses": [],
        }

        # Load audio files
        clean_path = sample["clean_wav_path"]
        adv_path = sample["adv_wav_path"]

        try:
            clean_wav, clean_sr = torchaudio.load(clean_path)
        except Exception as e:
            print(f"    [WARN] Cannot load clean WAV: {clean_path}: {e}")
            continue

        try:
            adv_wav, adv_sr = torchaudio.load(adv_path)
        except Exception as e:
            print(f"    [WARN] Cannot load adv WAV: {adv_path}: {e}")
            continue

        # Run probes on clean audio
        for probe in probes:
            try:
                resp = wrapper.decode(clean_wav.clone(), clean_sr, probe["question"])
            except Exception as e:
                resp = f"[ERROR] {e}"
            parsed = parse_yes_no(resp)
            sample_result["clean_responses"].append({
                "id": probe["id"],
                "question": probe["question"],
                "gt": probe["gt"],
                "type": probe["type"],
                "response": resp,
                "parsed": parsed,
            })

        # Run probes on adversarial audio
        for probe in probes:
            try:
                resp = wrapper.decode(adv_wav.clone(), adv_sr, probe["question"])
            except Exception as e:
                resp = f"[ERROR] {e}"
            parsed = parse_yes_no(resp)
            sample_result["adv_responses"].append({
                "id": probe["id"],
                "question": probe["question"],
                "gt": probe["gt"],
                "type": probe["type"],
                "response": resp,
                "parsed": parsed,
            })

        results.append(sample_result)

        # Save intermediate results every 10 samples
        if (idx + 1) % 10 == 0:
            _save_results(out_file, model_name, dataset, language, results)

    # Final save
    _save_results(out_file, model_name, dataset, language, results)
    print(f"[DONE] {model_name}/{dataset}: {len(results)} samples → {out_file}")


def _save_results(out_file, model_name, dataset, language, results):
    output = {
        "model": model_name,
        "dataset": dataset,
        "language": language,
        "num_samples": len(results),
        "probes_per_sample": 6,
        "samples": results,
    }
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hallucination evaluation")
    parser.add_argument("--model", required=True, choices=["voxtral", "meralion", "opens2s"])
    parser.add_argument("--dataset", required=True, choices=["iemocap", "ravdess", "esd_en", "esd_cn"])
    parser.add_argument("--gpu", type=int, default=2)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    evaluate(args.model, args.dataset)
