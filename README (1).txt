# DA5401 – 2025 Data Challenge
### Metric Learning for Conversational AI Evaluation

This repository contains my complete solution for the DA5401 Data Challenge 2025, focused on predicting the fitness score (0–10) of a conversational AI response relative to a given evaluation metric definition.

## Folder Structure
- final_kaggle.ipynb
- train_data.json
- test_data.json
- metric_name_embeddings.npy
- metric_names.json
- README.md

## How to Run
1. Install dependencies:
   pip install sentence-transformers scikit-learn pandas numpy torch

2. Ensure all dataset files are in the working directory.

3. Run final_kaggle.ipynb to generate submission.csv.

## Summary
- Used multilingual-mpnet-base-v2 embeddings
- Added interaction features (L1, product)
- 3072‑dim vector input
- Applied Cross‑Metric Negative Sampling
- Two‑Tower Siamese DNN
- Focal MSE Loss, AdamW, Cosine Annealing, 5‑Fold CV

## Results
Final RMSE: 3.001

## Conclusion
Careful feature engineering + imbalance correction + Siamese architecture significantly improved performance.
