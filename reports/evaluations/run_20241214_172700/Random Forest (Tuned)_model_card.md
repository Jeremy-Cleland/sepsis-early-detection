# Model Card: Random Forest (Tuned)

**Run ID:** 20241214_172700

**Training Date:** 2024-12-14 17:51:54

## Model Details

- **Version:** v1.0
- **Hyperparameters:**
  - **Random_forest**:
      - bootstrap: True
      - ccp_alpha: 0.0
      - class_weight: balanced
      - criterion: gini
      - max_depth: 29
      - max_features: sqrt
      - max_leaf_nodes: None
      - max_samples: None
      - min_impurity_decrease: 0.0
      - min_samples_leaf: 1
      - min_samples_split: 8
      - min_weight_fraction_leaf: 0.0
      - monotonic_cst: None
      - n_estimators: 205
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
- **Features:** 21 (after preprocessing)
- **Class Distribution (Training):**
    - Sepsis: 1.63%
    - Non-sepsis: 98.37%
- **Preprocessing:** Missing value imputation (median), scaling (StandardScaler), SMOTEENN resampling

## Evaluation

### Metrics (Test Set)

| Metric | Value |
|---|---|
| F1 Score | 0.5549 |
| Precision | 0.6289 |
| Recall | 0.4964 |
| Specificity | 0.9952 |
| AUROC | 0.9742 |
| AUPRC | 0.6006 |

### Confusion Matrix

![Confusion Matrix](Random Forest (Tuned)_confusion_matrix.png)

### Precision-Recall Curve

![Precision-Recall Curve](Random Forest (Tuned)_precision_recall_curve.png)

## Usage and Limitations

- **Intended Use:** Early warning system for sepsis in ICU patients.
- **Limitations:**
    - The model was trained on data from a specific population and may not generalize well to other populations.
    - The model's performance may be limited in cases with atypical presentations of sepsis.
    - The model is not fully interpretable.

## Ethical Considerations

- **Fairness:** The model's performance should be monitored across different demographic groups to ensure fairness.
- **Privacy:** Patient data was anonymized during model training.
- **Transparency:** This model card provides information about the model's development, performance, and limitations.
