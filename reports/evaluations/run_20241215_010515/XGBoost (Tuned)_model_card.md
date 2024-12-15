# Model Card: XGBoost (Tuned)

**Run ID:** 20241215_010515

**Training Date:** 2024-12-15 01:39:38

## Model Details

- **Model Type:** XGBoost
- **Architecture:** Gradient boosting with decision trees (learning_rate=0.17255776234553302, max_depth=9).
- **Version:** v1.0
- **Hyperparameters:**
  - **Xgb_classifier**:
      - objective: binary:logistic
      - base_score: None
      - booster: None
      - callbacks: None
      - colsample_bylevel: None
      - colsample_bynode: None
      - colsample_bytree: 0.5433439671072606
      - device: None
      - early_stopping_rounds: None
      - enable_categorical: False
      - eval_metric: logloss
      - feature_types: None
      - gamma: 0.012455929246615097
      - grow_policy: None
      - importance_type: None
      - interaction_constraints: None
      - learning_rate: 0.17255776234553302
      - max_bin: None
      - max_cat_threshold: None
      - max_cat_to_onehot: None
      - max_delta_step: None
      - max_depth: 9
      - max_leaves: None
      - min_child_weight: None
      - missing: nan
      - monotone_constraints: None
      - multi_strategy: None
      - n_estimators: 457
      - n_jobs: -1
      - num_parallel_tree: None
      - random_state: 42
      - reg_alpha: 0.07875095024685662
      - reg_lambda: 0.5035646682042978
      - sampling_method: None
      - scale_pos_weight: None
      - subsample: 0.7574329529424287
      - tree_method: None
      - validate_parameters: None
      - verbosity: None

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
| F1 Score | 0.3309 |
| Precision | 0.2056 |
| Recall | 0.8475 |
| Specificity | 0.9463 |
| AUROC | 0.9587 |
| AUPRC | 0.6273 |

### Confusion Matrix

![Confusion Matrix](XGBoost (Tuned)_confusion_matrix.png)

### Precision-Recall Curve

![Precision-Recall Curve](XGBoost (Tuned)_precision_recall_curve.png)

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
