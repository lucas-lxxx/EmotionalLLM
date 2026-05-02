"""Collect per-dataset metrics from MERaLiON experiment results."""
import json
import glob
import os
import numpy as np

os.chdir("/data1/lixiang/EmotionalLLM/code/white_box_meralion")

datasets = {
    "ESD-EN": "result/MERaLiON_EN",
    "ESD-CN": "result/MERaLiON_CN",
    "IEMOCAP": "result/MERaLiON_IEMOCAP",
    "RAVDESS": "result/MERaLiON_RAVDESS",
}

all_asr = []
all_sem = []
all_joint = []
all_snr = []
all_conv = []

for name, path in datasets.items():
    files = sorted(glob.glob(f"{path}/**/*.json", recursive=True))
    files = [f for f in files if "summary" not in os.path.basename(f)]

    snrs = []
    total = 0
    converged = 0
    sem_preserved = 0
    joint = 0

    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        if "sample_id" not in data:
            continue
        total += 1
        emo_ok = data.get("success_emo", False)
        sem_ok = data.get("semantic_preserved", False)
        if emo_ok:
            converged += 1
        if sem_ok:
            sem_preserved += 1
        if emo_ok and sem_ok:
            joint += 1
        if "snr_db" in data:
            snrs.append(data["snr_db"])

    asr_rate = converged / total * 100 if total > 0 else 0
    sem_rate = sem_preserved / total * 100 if total > 0 else 0
    joint_rate = joint / total * 100 if total > 0 else 0
    avg_snr = float(np.mean(snrs)) if snrs else 0
    conv_rate = converged / total * 100 if total > 0 else 0

    all_asr.append(asr_rate)
    all_sem.append(sem_rate)
    all_joint.append(joint_rate)
    all_snr.append(avg_snr)
    all_conv.append(conv_rate)

    print(f"{name}: N={total}, ASR={asr_rate:.2f}%, Sem={sem_rate:.2f}%, Joint={joint_rate:.2f}%, SNR={avg_snr:.2f}dB, Conv={conv_rate:.2f}%")

print()
print(f"Avg: ASR={np.mean(all_asr):.2f}%, Sem={np.mean(all_sem):.2f}%, Joint={np.mean(all_joint):.2f}%, SNR={np.mean(all_snr):.2f}dB, Conv={np.mean(all_conv):.2f}%")
