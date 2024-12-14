
# **Model Card: Random Forest (Tuned)**

**Run ID:** `20241213_210538`  
**Training Date:** `2024-12-13 22:42:35`

---

## **Model Details**

- **Version:** `v1.0`
- **Algorithm:** Random Forest
- **Hyperparameters:**
  - **General Settings:**
    - `bootstrap`: True  
    - `ccp_alpha`: 0.0  
    - `class_weight`: balanced  
    - `criterion`: gini  
    - `random_state`: 42  
    - `n_jobs`: -1  
    - `verbose`: 0  
    - `warm_start`: False  
  - **Tree Settings:**
    - `max_depth`: 19  
    - `max_features`: log2  
    - `max_leaf_nodes`: None  
    - `min_impurity_decrease`: 0.0  
    - `min_samples_leaf`: 3  
    - `min_samples_split`: 7  
    - `min_weight_fraction_leaf`: 0.0  
    - `monotonic_cst`: None  
  - **Ensemble Settings:**
    - `n_estimators`: 480  
    - `oob_score`: False  
    - `max_samples`: None  

---

## **Training Data**

- **Dataset:** *PhysioNet Sepsis Prediction Dataset*  
- **Samples:**
  - **Training:** 659,755  
  - **Validation:** 141,866  
  - **Test:** 138,629  
- **Features:** 23 (after preprocessing)  
- **Class Distribution (Training Set):**
  - Sepsis: 1.63%  
  - Non-Sepsis: 98.37%  
- **Preprocessing:**
  - Missing value imputation: Median  
  - Scaling: StandardScaler  
  - Resampling: SMOTEENN  

---

## **Evaluation**

### **Metrics (Test Set)**

| Metric       | Value  |
|--------------|--------|
| Specificity  | 0.9913 |
| AUROC        | 0.9760 |
| F1 Score     | 0.5594 |
| Precision    | 0.5280 |
| Recall       | 0.5948 |
