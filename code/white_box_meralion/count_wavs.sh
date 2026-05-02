#!/bin/bash
for d in \
  /data1/lixiang/EmotionalLLM/code/white_box_voxtral/result/Voxtral_IEMOCAP \
  /data1/lixiang/EmotionalLLM/code/white_box_voxtral/result/Voxtral_RAVDESS \
  /data1/lixiang/EmotionalLLM/code/white_box_voxtral/result/Voxtral_EN \
  /data1/lixiang/EmotionalLLM/code/white_box_voxtral/result/Voxtral_CN \
  /data1/lixiang/EmotionalLLM/code/white_box_opens2s_v2/result/IEMOCAP \
  /data1/lixiang/EmotionalLLM/code/white_box_opens2s_v2/result/RAVDESS \
  /data1/lixiang/EmotionalLLM/code/white_box_opens2s_v2/result/blackbox/EN \
  /data1/lixiang/EmotionalLLM/code/white_box_opens2s_v2/result/blackbox/CN \
  /data1/lixiang/EmotionalLLM/code/white_box_meralion/result/MERaLiON_IEMOCAP \
  /data1/lixiang/EmotionalLLM/code/white_box_meralion/result/MERaLiON_RAVDESS \
  /data1/lixiang/EmotionalLLM/code/white_box_meralion/result/MERaLiON_EN \
  /data1/lixiang/EmotionalLLM/code/white_box_meralion/result/MERaLiON_CN; do
  count=$(find $d -name '*.wav' 2>/dev/null | wc -l)
  echo "$d: $count"
done
