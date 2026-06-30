# CDIME-AI Implementation

This repository contains a full, runnable implementation of the **CDIME-AI**
research proposal (*Causal Domain-Incremental Multimodal Explainable AI for
Trustworthy Chest X-ray Diagnosis and Therapeutic Decision Support*).

➡️ **Code:** [`cdime_ai/`](cdime_ai/) — see [`cdime_ai/README.md`](cdime_ai/README.md)
➡️ **Colab (free T4):** [`notebooks/CDIME_AI_Colab.ipynb`](notebooks/CDIME_AI_Colab.ipynb)

```bash
pip install -r cdime_ai/requirements.txt
python -m cdime_ai.main --fast                 # smoke test (<1 min)
python -m cdime_ai.main --ablation --seeds 42 43 44   # full study + stats
```

Implements: Swin + ClinicalBERT cross-attention fusion, causal IRM + V-REx,
EWC + replay + domain adapters continual learning across CheXpert → MIMIC-CXR →
VinDr-CXR, Grad-CAM / attention / counterfactual / occlusion explanations,
MC-Dropout + temperature-scaling uncertainty, therapeutic decision support,
ablations A1–A7, and full statistical significance testing.
