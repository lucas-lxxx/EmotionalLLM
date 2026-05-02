# Hallucination Evaluation Recovered Sample Index

This index is reconstructed from the existing white-box per-sample JSON files. The dedicated `code/hallucination_eval/results/*/hallucination_eval.json` files are not present in this local workspace.

- CSV index: `reports/hallucination_eval_recovered_index.csv`
- Selection rule: first 60 JSON+WAV samples per model/dataset, matching `code/hallucination_eval/run_eval.py` and `compute_final_table.py`.
- Response columns use existing fields: `emo_text_clean`, `emo_text_adv`, `asr_text_clean`, `asr_text_adv`.

| Model | Dataset | Rows | Clean audio exists | Adv audio exists | Source directory |
|---|---:|---:|---:|---:|---|
| Voxtral | IEMOCAP | 60 | 0 | 0 | `C:\Users\potte\Desktop\research\emotional LLM\code\white_box_voxtral\result\Voxtral_IEMOCAP` |
| Voxtral | RAVDESS | 60 | 0 | 0 | `C:\Users\potte\Desktop\research\emotional LLM\code\white_box_voxtral\result\Voxtral_RAVDESS` |
| Voxtral | ESD-EN | 60 | 60 | 60 | `C:\Users\potte\Desktop\research\emotional LLM\code\white_box_voxtral\result\Voxtral_EN` |
| Voxtral | ESD-CN | 60 | 60 | 60 | `C:\Users\potte\Desktop\research\emotional LLM\code\white_box_voxtral\result\Voxtral_CN` |
| MERaLiON | IEMOCAP | 0 | 0 | 0 | `C:\Users\potte\Desktop\research\emotional LLM\code\white_box_meralion\result\MERaLiON_IEMOCAP` |
| MERaLiON | RAVDESS | 0 | 0 | 0 | `C:\Users\potte\Desktop\research\emotional LLM\code\white_box_meralion\result\MERaLiON_RAVDESS` |
| MERaLiON | ESD-EN | 0 | 0 | 0 | `C:\Users\potte\Desktop\research\emotional LLM\code\white_box_meralion\result\MERaLiON_EN` |
| MERaLiON | ESD-CN | 0 | 0 | 0 | `C:\Users\potte\Desktop\research\emotional LLM\code\white_box_meralion\result\MERaLiON_CN` |
| OpenS2S | IEMOCAP | 60 | 0 | 0 | `C:\Users\potte\Desktop\research\emotional LLM\code\white_box_opens2s_v2\result\IEMOCAP` |
| OpenS2S | RAVDESS | 60 | 0 | 0 | `C:\Users\potte\Desktop\research\emotional LLM\code\white_box_opens2s_v2\result\RAVDESS` |
| OpenS2S | ESD-EN | 0 | 0 | 0 | `C:\Users\potte\Desktop\research\emotional LLM\code\white_box_opens2s_v2\result\ESDfinal` |
| OpenS2S | ESD-CN | 60 | 60 | 0 | `C:\Users\potte\Desktop\research\emotional LLM\code\white_box_opens2s_v2\result\ESDfinal` |

## Missing Dedicated QA Outputs

The script-level binary QA output was expected at `/data1/lixiang/EmotionalLLM/code/hallucination_eval/results/<model>_<dataset>/hallucination_eval.json`. No local `hallucination_eval.json` files were found under this workspace or `C:/Users/potte/Desktop/research`.
