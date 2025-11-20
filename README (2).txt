DA5401 – 2025 Data Challenge
Metric Learning for Conversational AI Evaluation
------------------------------------------------

This repository contains my complete solution for the DA5401 Data Challenge 2025, focused on predicting the fitness score (0–10) of a conversational AI response relative to a given evaluation metric definition. The task is framed as a metric learning regression problem involving two asymmetric text inputs.

------------------------------------------------
FOLDER STRUCTURE
------------------------------------------------
- final_kaggle.ipynb
- train_data.json
- test_data.json
- metric_name_embeddings.npy
- metric_names.json
- README.txt

------------------------------------------------
HOW TO RUN
------------------------------------------------

1. ENVIRONMENT SETUP
   Run:
   pip install sentence-transformers scikit-learn pandas numpy torch

2. DATA PLACEMENT
   Ensure all files are in the same directory or update DATA_DIR.

3. EXECUTION
   Open final_kaggle.ipynb in Jupyter, Colab, or Kaggle and run all cells.
   The notebook:
   - Preprocesses data
   - Generates embeddings
   - Trains with 5-fold CV
   - Produces submission.csv

------------------------------------------------
SOLUTION SUMMARY
------------------------------------------------

1. DATA PREPARATION & FEATURE ENGINEERING

- Combined system_prompt, user_prompt, response using tokens [P], [R], [S]
- Used paraphrase-multilingual-mpnet-base-v2 to generate 768-d embeddings
- Created interaction features:
    * L1 distance |E_m - E_t|
    * Element-wise product E_m ⊙ E_t
- Final vector = 3072 dimensions (E_m + E_t + Diff + Prod)
- Scaled using StandardScaler

------------------------------------------------
2. CLASS IMBALANCE FIX
------------------------------------------------

Problem:
- 91% of scores were 9 or 10 → Baseline predicted mean=9.1 → RMSE ~4.6

Solution: Cross-Metric Negative Sampling
- Select 50% high-score samples (>=9)
- Replace metric with random different metric
- Assign synthetic score = 1.0
- Forces model to learn “bad fits”

This step drastically improved performance.

------------------------------------------------
3. MODEL ARCHITECTURE – TWO-TOWER SIAMESE DNN
------------------------------------------------

- Tower A: Processes metric embedding
- Tower B: Processes text embedding
- Fusion layer merges:
    * Tower outputs
    * L1 distance features
    * Product features
- Regression head uses BatchNorm + Dropout

------------------------------------------------
4. TRAINING STRATEGY
------------------------------------------------

Loss Function:
- Focal MSE Loss (gamma = 3.0)
  Down-weights easy (score 9–10) samples and focuses on hard negatives.

Optimizer:
- AdamW

Scheduler:
- Cosine Annealing LR

Evaluation:
- 5-Fold Cross Validation for stability

------------------------------------------------
RESULTS
------------------------------------------------

Final Validation RMSE: 3.001

WHY IT WORKED:
- Cross-metric negative sampling solved class imbalance
- Distance features provided strong similarity signals
- Siamese architecture effectively modeled dual inputs
- 5-fold averaging improved robustness

------------------------------------------------
CONCLUSION
------------------------------------------------

This project shows that effective metric learning for conversational AI requires:
- Careful preprocessing
- Strong embeddings
- Smart synthetic negative generation
- Siamese architecture
- Feature engineering (L1 + product)
- Robust CV-based training

These methods significantly outperformed baseline models and provided stable predictions.

------------------------------------------------
