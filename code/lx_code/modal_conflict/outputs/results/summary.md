# Results Summary (run_20260122_153920)

## Run Metadata
- run_dir: outputs/results/run_20260122_153920
- model_path: /data1/lixiang/Opens2s/OpenS2S/models/OpenS2S/
- prompt: "What is the emotion of this audio? Answer with exactly one word: neutral, happy, sad, angry, surprised."
- system_prompt: "You are a helpful assistant."
- probe: linear (LogisticRegression), C=1.0, max_iter=1000, solver=lbfgs
- evaluation: GroupKFold n_splits=5, random_seed=42

## Dataset Summary (dataset_info.json)
- total_samples: 247
- conflict_samples: 197
- consistent_samples: 50
- unique_texts: 50
- semantic_distribution:
  - neutral: 50
  - happy: 49
  - sad: 50
  - angry: 49
  - surprised: 49
- prosody_distribution:
  - neutral: 49
  - happy: 50
  - sad: 48
  - angry: 50
  - surprised: 50

## Core Metrics (summary.json)
- overall_dominant_modality: prosody
- average_dominance: 0.0526
- max_semantic_acc: layer 27, acc 0.8304
- max_prosody_acc: layer 0, acc 0.8420
- max_prosody_dominance: layer 5, D 0.2146
- max_semantic_dominance: layer 26, D -0.0414
- layer_trends (avg D):
  - early (0-11): 0.1459
  - middle (12-23): 0.0048
  - late (24-35): 0.0071
- conflict_subset (avg over conflict samples):
  - semantic_acc: 0.7299
  - prosody_acc: 0.7895
  - dominance: 0.0596

## Derived From metrics_per_layer.csv
- layers_evaluated: 0-35 (36 layers)
- dominance_crossing: between layers 14 and 15 (sign change)
- dominance_positive_ranges (D > 0):
  - 0-14
  - 20-22
  - 24-24
  - 29-34
- peak_dominance: layer 5, D 0.2146
- peak_conflict_dominance: layer 5, D_conf 0.2182

## File Meanings
- metrics_per_layer.csv
  - per-layer metrics: semantic_acc, semantic_acc_std, semantic_f1, prosody_acc, prosody_acc_std, prosody_f1
  - dominance: D = acc_prosody - acc_semantic
  - conflict metrics: semantic_acc_conflict, prosody_acc_conflict, dominance_conflict
- summary.json
  - run-level summary stats (peaks, averages, modality dominance)
- dataset_info.json
  - sample counts and label distributions after skipping missing audio files
- config.yaml
  - full configuration snapshot for this run

## Plot Meanings
- dominance_curve.png
  - D(layer) for all samples and conflict-only samples; above 0 => prosody-dominant, below 0 => semantic-dominant
- accuracy_curves.png
  - semantic vs prosody accuracy per layer, with std bands
- conflict_curves.png
  - semantic vs prosody accuracy on conflict-only samples; dashed lines show all-sample baselines
- f1_curves.png
  - semantic vs prosody macro-F1 per layer
- layer_heatmap.png
  - heatmap of key metrics across layers (includes conflict subset metrics)
