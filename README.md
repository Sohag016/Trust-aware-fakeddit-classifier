# Multimodal Consistency Reasoning for Fake News Detection

A multimodal fake-news detection and **trust-calibration** framework built on the **Fakeddit** dataset. The pipeline combines a fine-tuned language–vision classifier, CLIP-based text–image alignment, OCR-based text consistency, and metadata-driven modeling — then layers an **uncertainty-aware trust score** on top so the system knows *when to trust its own predictions*.

## ✨ Key Features

- **RoBERTa-base + ResNet-50** fine-tuned multimodal classifier (6-way Fakeddit labels)
- **CLIP ViT-B/32** for semantic text–image alignment scoring
- **EasyOCR + Sentence-Transformers (MiniLM)** for embedded-text vs. headline consistency
- **XGBoost** metadata model on image-level statistical features
- **Learned MLP fusion** layer combining model outputs (replacing fixed-alpha blending)
- **Consistency Score** — a novel metric combining text-image alignment, OCR similarity, and metadata confidence
- **Uncertainty-aware trust pipeline**:
  - MC-Dropout epistemic uncertainty estimation
  - Temperature scaling + Expected Calibration Error (ECE)
  - Cross-model (Jensen–Shannon) disagreement signal
  - Held-out weight validation to avoid overfitting trust weights to the eval set
  - Risk–coverage (AURC) analysis and percentile-based Accept/Abstain/Reject decision pipeline
- Ablation studies, confusion matrices, multi-seed robustness checks, and baseline comparisons (max-softmax vs. deep ensemble vs. trust score)

## 🗂️ Dataset

[Fakeddit](https://github.com/entitize/Fakeddit) (multimodal subset), downloaded via `kagglehub` (`vanshikavmittal/fakeddit-dataset`). Labels follow the **6-way** classification scheme:

| Label | Class |
|---|---|
| 0 | True |
| 1 | Satire |
| 2 | False Connection |
| 3 | Imposter |
| 4 | Manipulated |
| 5 | Misleading |

## 🏗️ Pipeline Overview

```
Text + Image
     │
     ├── RoBERTa + ResNet-50 ──► 6-way classification
     ├── CLIP ViT-B/32 ────────► text–image semantic alignment
     ├── EasyOCR + MiniLM ─────► OCR text vs. headline similarity
     └── XGBoost (image stats) ► metadata-based signal
                     │
             Learned MLP Fusion
                     │
          Consistency Score (core novelty)
                     │
   Uncertainty & Trust Calibration Layer
   (MC-Dropout + Temp Scaling + Disagreement)
                     │
        Accept / Abstain / Reject Decision
```

## 📦 Requirements

```bash
pip install transformers datasets timm pillow scikit-learn matplotlib pandas tqdm \
            xgboost sentence-transformers easyocr kagglehub torch torchvision
```

- Python 3.9+
- GPU strongly recommended (CUDA) for training RoBERTa/ResNet-50/CLIP

## 🚀 Usage

1. Open `2_Multimodal_Consistency_Reasoning_FINAL.ipynb` in Google Colab or Jupyter.
2. Run the setup cells to install dependencies and download the Fakeddit dataset.
3. Run through the notebook sequentially — it is organized into numbered steps (STEP 1 → STEP 46), covering:
   - Data loading & balanced sampling (400/class)
   - Image caching
   - Model training (RoBERTa+ResNet-50, CLIP)
   - Metadata modeling (XGBoost)
   - Fusion, consistency scoring, and the full trust-aware reasoning pipeline
4. Outputs (tables, figures, CSV results) are saved under `outputs/`.

## 📊 Outputs

The notebook saves the following artifacts to the `outputs/` directory:

- `model_comparison.csv` — accuracy/F1 across all component models
- `consistency_reasoning_results.csv` — per-sample consistency scores
- `threshold_analysis.csv` — consistency threshold sweep
- `ablation_table.csv` — ablation study results
- `trust_aware_results.csv` — trust-calibrated decisions (Accept/Abstain/Reject)
- `multiseed_robustness.csv`, `baseline_comparison.csv`, `weight_sensitivity_full_descriptive.csv` — robustness & sensitivity analyses



## 📄 Author

Md. Sohag Hossain
Data science & ML Engineer

## 🙏 Acknowledgments

- [Fakeddit dataset](https://github.com/entitize/Fakeddit)
- Hugging Face `transformers` / `sentence-transformers`
- OpenAI CLIP
