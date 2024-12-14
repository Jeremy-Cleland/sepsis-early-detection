# Model Card: Random Forest (Tuned)

**Run ID:** 20241213_210538

**Training Date:** 2024-12-13 22:42:35

## Model Details

- **Version:** v1.0
- **Hyperparameters:**
  - **Random_forest**:
    - bootstrap: True
    - ccp_alpha: 0.0
    - class_weight: balanced
    - criterion: gini
    - max_depth: 19
    - max_features: log2
    - max_leaf_nodes: None
    - max_samples: None
    - min_impurity_decrease: 0.0
    - min_samples_leaf: 3
    - min_samples_split: 7
    - min_weight_fraction_leaf: 0.0
    - monotonic_cst: None
    - n_estimators: 480
    - n_jobs: -1
    - oob_score: False
    - random_state: 42
    - verbose: 0
    - warm_start: False

## Training Data

- **Dataset:** PhysioNet Sepsis Prediction Dataset
- **Samples:**
  - Training: 659755
  - Validation: 141866
  - Test: 138629
- **Features:** 23 (after preprocessing)
- **Class Distribution (Training):**
  - Sepsis: 1.63%
  - Non-sepsis: 98.37%
- **Preprocessing:** Missing value imputation (median), scaling (StandardScaler), SMOTEENN resampling

## Evaluation

### Metrics (Test Set)

| Metric | Value |
|---|---|
| F1 Score | 0.5594 |
| Precision | 0.5280 |
| Recall | 0.5948 |
| Specificity | 0.9913 |
| AUROC | 0.9760 |
