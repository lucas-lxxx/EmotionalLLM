#!/usr/bin/env python3
"""Check sample rates of clean and adversarial WAV files."""
import torchaudio
files = [
    ("/data1/lixiang/OpenS2S_dataset/data/IEMOCAP_esd/Session2/angry/Ses02F_script03_2_F038.wav", "Voxtral clean IEMOCAP"),
    ("/data1/lixiang/EmotionalLLM/code/white_box_voxtral/result/Voxtral_IEMOCAP/Session2/00000_Session2_angry_Ses02F_script03_2_F038.wav", "Voxtral adv IEMOCAP"),
    ("/data1/lixiang/EmotionalLLM/code/white_box_meralion/result/MERaLiON_IEMOCAP/Session2/00000_Session2_angry_Ses02F_script03_2_F038.wav", "MERaLiON adv IEMOCAP"),
    ("/data1/lixiang/EmotionalLLM/code/white_box_opens2s_v2/result/IEMOCAP/Session2/00000_Session2_angry_Ses02F_script03_2_F038.wav", "OpenS2S adv IEMOCAP"),
]
for path, label in files:
    try:
        info = torchaudio.info(path)
        print(f"{label:30s}: sr={info.sample_rate}, frames={info.num_frames}, ch={info.num_channels}")
    except Exception as e:
        print(f"{label:30s}: ERROR {e}")
