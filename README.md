# DA5401 – Kaggle Data Challenge

Name: Anan madhav T V 
Roll No: MM2B013


## 📂 Folder Structure
```
├── final_kaggle.ipynb          # Main notebook containing the complete solution
├── train_data.json             # Training dataset (features + scores)
├── test_data.json              # Test dataset (features only)
├── metric_name_embeddings.npy  # Pre-computed embeddings for metric definitions
├── metric_names.json           # Mapping of metric names
└── README.md                   # Project documentation
```

---

## ⚙️ How to Run

### 1️⃣ Environment Setup
Install dependencies:
```bash
pip install sentence-transformers scikit-learn pandas numpy torch
```

### 2️⃣ Data Placement
Ensure the following files are present:
- train_data.json
- test_data.json
- metric_name_embeddings.npy
- metric_names.json

Update `DATA_DIR` if using a custom directory.

### 3️⃣ Execution
Run `final_kaggle.ipynb` in Jupyter/Colab/Kaggle.
The notebook:
- Preprocesses and embeds text  
- Trains via 5-Fold CV  
- Outputs `submission.csv`

---

# 📝 Challenge Summary

Predicting semantic alignment between:
- A **metric definition**
- A **prompt–response pair**

The dataset was **heavily imbalanced**, requiring careful augmentation and modeling.

---

# 🚀 Solution Overview

## 1️⃣ Data Preparation & Feature Engineering

### Text Preprocessing
Merged texts using:
```
[P] user_prompt
[R] response
[S] system_prompt
```

### Embedding Model
Used:
```
sentence-transformers/paraphrase-multilingual-mpnet-base-v2
```
→ 768-d embeddings each for:
- Metric definition (E_m)
- Conversation text (E_t)

### Interaction Features
To help the model compare distances:
- **L1 Distance:** |E_m - E_t|
- **Element-wise Product:** E_m ⊙ E_t

### Final Input Vector
3072 dimensions total.

---

# 2️⃣ Fixing Class Imbalance (Critical)

### Issue:
91% of scores were **9 or 10**, causing models to predict only high values.

### Solution: Cross-Metric Negative Sampling
- Select 50% of high-score samples (≥9)
- Replace metric with a different randomly chosen metric
- Assign synthetic score = **1.0**

This forces the model to learn what *bad matches* look like.

---

# 3️⃣ Model Architecture – Two-Tower Siamese Network

### Towers
- **Tower A:** Metric embedding → Dense layers
- **Tower B:** Text embedding → Dense layers

### Fusion
Concatenates:
- Tower outputs
- L1 distance
- Product features

### Regression Head
BatchNorm + ReLU + Dropout → Score output

---

# Training

### Optimization
- AdamW optimizer  
- Cosine Annealing LR Scheduler  
- 5-Fold Cross Validation  

---

# Results

### **Final RMSE: 2.720**

### Why It Worked
- Imbalance solved via targeted augmentation  
- Explicit similarity features improved learning  
- Siamese architecture suited dual inputs  
- 5-fold averaging stabilized predictions  

---

# Conclusion

The project shows that high performance requires:
- Proper feature engineering  
- Handling skewed labels  
- Two-tower architectures  
- Carefully designed training loops  

The combination significantly outperformed baseline models.

