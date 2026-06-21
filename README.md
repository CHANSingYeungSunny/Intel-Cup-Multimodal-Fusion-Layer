# Intel Cup — Multimodal Fusion Layer (Monorepo)

Monorepo containing three monomodal health-classification layers and one fusion layer that combines them via a Multimodal Transformer Encoder with cross-modal attention.

## Repository Structure

```
intel multimodal (fusion layer)/
│
├── intel multimodal (vision layer)/       # Vision Layer — Swin-Tiny on UBFC rPPG
│   └── vision_layer/
│       ├── model.py                       # SwinHealthClassifier (768-dim features)
│       ├── train.py                       # Training loop
│       ├── run_experiments.py             # 5-experiment ablation runner
│       ├── preprocessing.py               # PPG-based pseudo-label generation
│       ├── dataset.py                     # UBFCWindowDataset
│       ├── config.py                      # Hyperparameters
│       └── output/
│           └── predictions/
│               ├── predictions.csv        # 607 windows, 768-dim feature vectors
│               └── experiment_results_with_accuracy.csv  # 33-column schema
│
├── intel multimodal (audio layer)/        # Audio Layer — AST on COUGHVID
│   └── audio_layer/
│       ├── model.py                       # AudioSpectrogramTransformer (128-dim)
│       ├── train.py                       # Training with early stopping
│       ├── main.py                        # End-to-end pipeline
│       ├── run_ablation.py                # 5-experiment runner
│       ├── evaluate.py                    # Metrics + CSV export
│       ├── config.py                      # Hyperparameters
│       └── outputs/
│           ├── predictions.csv            # ~3100 samples, 128-dim feature vectors
│           └── experiment_results_with_accuracy.csv  # 33-column schema
│
├── intel multimodal (physiological layer)/ # Physiological Layer — iTransformer on BIDMC
│   ├── model.py                           # iTransformerClassifier (128-dim)
│   ├── train.py                           # Training with AMP, grad checkpointing
│   ├── data_loader.py                     # BIDMCDataset + window caching
│   ├── preprocess.py                      # Label generation, normalization
│   ├── utils.py                           # Metrics, CSV, plotting
│   └── outputs/
│       ├── predictions.csv                # ~855 windows, 128-dim feature vectors
│       └── experiment_results_with_accuracy.csv  # 33-column schema
│
├── Fusion-Layer/                          # Fusion Layer — Multimodal Transformer
│   ├── fusion_loader.py                   # Label-matched pairing across 3 modalities
│   ├── fusion_model.py                    # MultimodalFusionEncoder (2.4M params)
│   ├── fusion_train.py                    # Training loop (AMP, early stop, 5 exps)
│   ├── fusion_utils.py                    # Metrics, CSV serialization, plotting
│   ├── fusion_preprocess.py               # Normalization, split, label handling
│   ├── README.md                          # Fusion Layer documentation
│   ├── summary.md                         # Detailed design explanation
│   └── outputs/
│       ├── predictions.csv                # 256-dim fusion embeddings
│       ├── experiment_results_with_accuracy.csv  # 33-column schema (5 rows)
│       ├── best_fusion.pt                 # Best checkpoint (Exp 2, binary)
│       ├── fusion_training_curves.png
│       └── fusion_confusion_matrix.png
│
└── README.md                              # This file
```

## Architecture Overview

```
  ┌─────────────────────────┐
  │   Vision Layer          │
  │   Swin-Tiny (768-dim)   │──┐
  │   UBFC rPPG dataset     │  │
  └─────────────────────────┘  │
                               │
  ┌─────────────────────────┐  │    ┌──────────────────────────────┐
  │   Audio Layer           │  │    │   Fusion Layer               │
  │   AST (128-dim)         │──┼───▶│   Multimodal Transformer     │
  │   COUGHVID dataset      │  │    │   Encoder (4L, d=256, h=8)   │
  └─────────────────────────┘  │    │                              │
                               │    │   [V|A|P] + CLS → 4 tokens   │
  ┌─────────────────────────┐  │    │   Cross-attention → CLS →    │
  │   Physiological Layer   │  │    │   Classifier                 │
  │   iTransformer (128-dim)│──┘    │                              │
  │   BIDMC PPG dataset     │       │   1024-dim → 256-dim → logits│
  └─────────────────────────┘       └──────────────────────────────┘
```

## Datasets

| Layer | Dataset | Subjects | Modality |
|-------|---------|----------|----------|
| Vision | [UBFC rPPG](https://sites.google.com/view/ubfc-rppg) | 50 | Facial video (30fps) |
| Audio | [COUGHVID](https://zenodo.org/record/4498364) | ~2,900 | Cough sounds |
| Physiological | [BIDMC PPG](https://physionet.org/content/bidmc/) | 53 | PPG + ECG waveforms |

Since the three modalities use **different subjects with no overlap**, the Fusion Layer uses **label-matched pairing** — each Vision sample (anchor, 607 windows) is paired with randomly selected Audio and Physiological samples sharing the same health label.

## Quick Start

Each layer can be run independently:

```bash
# Vision Layer
cd "intel multimodal (vision layer)/vision_layer"
python run_experiments.py

# Audio Layer
cd "intel multimodal (audio layer)"
python main.py

# Physiological Layer
cd "intel multimodal (physiological layer)"
python train.py --exp_id all

# Fusion Layer (requires all three layers' predictions.csv)
cd Fusion-Layer
python fusion_train.py --exp_id all
```

## Reproducibility

All layers use seed=42 for deterministic splits, sampling, and initialization. The Fusion Layer caches aligned features to `Fusion-Layer/cache/fused_features.pt`.

## Output Format

All four layers produce the same CSV schemas:

- **predictions.csv**: `filename, prediction, label, feature_vector` (feature_vector is a JSON array)
- **experiment_results_with_accuracy.csv**: 33 columns — 9 config, 10 performance, 9 confusion matrix, 5 training curves
