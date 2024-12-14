# Sepsis Prediction Pipeline

## Overview

This project aims to predict sepsis in patients using advanced machine learning models. The workflow encompasses data preprocessing, feature engineering, class imbalance handling, hyperparameter optimization, model training, evaluation, model card generation, and model registry management for reproducibility and scalability.

**Team:** Jeremy Cleland, Anthony Lewis, and Salif Khan

---

## Project Structure

```bash
sepsis_prediction/
│
├── data/
│   ├── raw/
│   │   └── Dataset.csv            # Original dataset
│   ├── processed/
│   │   ├── train_data.csv         # Processed training data
│   │   ├── val_data.csv           # Processed validation data
│   │   └── test_data.csv          # Processed test data
│
├── notebooks/
│   └── EDA.ipynb                  # Exploratory Data Analysis notebooks
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py         # Data loading, splitting, validation
│   ├── feature_engineering.py     # Preprocessing and feature transformations
│   ├── evaluation.py              # Model evaluation & visualization
│   ├── logger_config.py           # Logging configuration
│   ├── logger_utils.py            # Logging utilities
│   ├── model_registry.py          # Model registry for versioning and retrieval
│   ├── utils.py                   # Utility functions (logging, saving metrics)
│   └── ... (other related files)
│
├── main.py                        # Main execution script of the pipeline
├── requirements.txt               # Python package dependencies
├── environment.yml                # Conda environment definition
├── README.md                      # This README file
├── registry.json                  # Model registry metadata
└── tests/
    └── test_data_processing.py    # Example tests
```

---

## Key Features

- **Data Loading and Preprocessing:**
  - Robust loading from CSV files with error handling.
  - Patient-level splitting into training, validation, and test sets to prevent data leakage.
  - Imputes missing values, encodes categorical features, applies transformations (log, scaling), and removes redundant or highly correlated features.

- **Feature Engineering:**
  - Drops redundant and null columns.
  - Imputes missing values using Iterative Imputer (MICE algorithm).
  - One-hot encodes categorical variables.
  - Applies log transformations to handle skewed features.
  - Scales numerical features using StandardScaler and RobustScaler.

- **Class Imbalance Handling:**
  - Utilizes SMOTEENN to address class imbalance, combining synthetic oversampling and undersampling techniques.

- **Model Training and Hyperparameter Optimization:**
  - Supports Random Forest, Logistic Regression, and XGBoost models.
  - Hyperparameter tuning with Optuna using cross-validation and MedianPruner for efficient pruning.
  - Automatically creates or loads Optuna studies from checkpoints for reproducible experiments.

- **Comprehensive Evaluation and Visualization:**
  - Computes multiple metrics: Accuracy, Precision, Recall, F1 Score, AUROC, Specificity, and more.
  - Generates extensive visualizations: Confusion matrices, ROC curves, Precision-Recall curves, Feature Importance, Missing Value Heatmaps, Temporal Progression plots, Error Analysis by patient groups, Calibration curves, and Feature Interactions.
  - Produces prediction timelines to understand how the model's probability predictions evolve over time for individual patients.

- **Model Cards:**
  - Automatically generates detailed model cards in Markdown format.
  - Includes training details, hyperparameters, performance metrics, feature importance, and ethical considerations.

- **Model Registry:**
  - Centralized registry for versioning, saving, and loading models along with their metrics, parameters, and artifacts.
  - Facilitates retrieval of the best model based on chosen metrics.
  - Supports model lifecycle management (saving, loading, deleting versions).

- **Logging and Checkpoints:**
  - Detailed logging (console and file) for all pipeline steps.
  - Utilizes checkpoints to store intermediate preprocessing steps, enabling faster re-runs without repeating heavy computations.

---

## Setup and Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Jeremy-Cleland/sepsis-early-detection.git
   cd sepsis-prediction-pipeline
   ```

2. **Set up a Virtual Environment:**

   ```bash
   conda env create -f environment.yml
   conda activate sepsis-prediction
   ```

3. **Data:**
   - Download the dataset and place it in `data/raw/Dataset.csv`.
   - Example dataset: [Sepsis Patient Data](https://www.kaggle.com/datasets/salikhussaini49/prediction-of-sepsis)

---

## Usage

1. **Run the Main Pipeline:**

   ```bash
   python main.py --help
   ```

   This command will display the available command-line arguments. For example:

   ```bash
   python main.py \
       --optuna-n-jobs 4 \
       --rf-trials 20 \
       --lr-trials 20 \
       --xgb-trials 20 \
       --data-path data/raw/Dataset.csv \
       --report-dir reports/evaluations \
       --model-dir models
   ```

   To create new Optuna studies (avoiding loading from old checkpoints):

   ```bash
   python main.py \
       --optuna-n-jobs 4 \
       --rf-trials 20 \
       --lr-trials 20 \
       --xgb-trials 20 \
       --data-path data/raw/Dataset.csv \
       --report-dir reports/evaluations \
       --model-dir models \
       --new-study
   ```

   **Note:** By default, the script runs with minimal trials (1 trial per model) and single-job optimization to quickly showcase the workflow. Increase these values to fully tune the models.

2. **Reviewing Results:**
   - Evaluation reports, metrics, and model cards are saved under `reports/evaluations/run_YYYYMMDD_HHMMSS`.
   - Each run contains metrics JSON files, plots, and a `*_model_card.md` file detailing model performance and training context.
   - Models are saved under the `models/` directory along with their metadata.

---

## Command-Line Arguments

**From `main.py`:**

- `--optuna-n-jobs`: Number of parallel jobs for Optuna hyperparameter tuning. **Default: 1**
- `--rf-trials`: Number of trials for Random Forest optimization. **Default: 1**
- `--lr-trials`: Number of trials for Logistic Regression optimization. **Default: 1**
- `--xgb-trials`: Number of trials for XGBoost optimization. **Default: 1**
- `--data-path`: Path to the raw dataset CSV file. **Default: `data/raw/Dataset.csv`**
- `--report-dir`: Directory to save evaluation reports and plots. **Default: `reports/evaluations`**
- `--model-dir`: Directory to save trained models. **Default: `models`**
- `--new-study`: Create new Optuna studies instead of loading from checkpoint.
- `--force`: Force re-training and hyperparameter tuning by ignoring existing checkpoints.

Increase the trial parameters for thorough hyperparameter optimization.

---

## Model Performance

### Random Forest

| CV Iteration | F1 Score | Accuracy | ROC AUC | Precision | Recall | F1 Std |
|--------------|----------|----------|---------|-----------|--------|--------|
| 1            | 0.8323   | 0.8737   | 0.9586  | 0.7641    | 0.9196 | 0.0445 |
| 2            | 0.9680   | 0.9780   | 0.9982  | 0.9502    | 0.9869 | 0.0146 |
| 3            | 0.8309   | 0.8725   | 0.9580  | 0.7627    | 0.9186 | 0.0458 |
| 4            | 0.9763   | 0.9838   | 0.9995  | 0.9588    | 0.9949 | 0.0119 |
| 5            | 0.7671   | 0.8220   | 0.9088  | 0.7057    | 0.8522 | 0.0554 |
| 6            | 0.9648   | 0.9755   | 0.9986  | 0.9398    | 0.9918 | 0.0180 |
| 7            | 0.9563   | 0.9695   | 0.9978  | 0.9271    | 0.9885 | 0.0217 |
| 8            | 0.9673   | 0.9774   | 0.9989  | 0.9436    | 0.9930 | 0.0171 |
| 9            | 0.9312   | 0.9510   | 0.9937  | 0.8917    | 0.9764 | 0.0303 |
| 10           | 0.9722   | 0.9808   | 0.9992  | 0.9515    | 0.9942 | 0.0145 |

**Best Parameters:**

```json
{
  "n_estimators": 480,
  "max_depth": 19,
  "min_samples_split": 7,
  "min_samples_leaf": 3,
  "max_features": "log2",
  "bootstrap": true,
  "criterion": "gini"
}
```

---

### XGBoost

| CV Iteration | F1 Score | Accuracy | ROC AUC | Precision | Recall | F1 Std |
|--------------|----------|----------|---------|-----------|--------|--------|
| 1            | 0.7563   | 0.8437   | 0.9066  | 0.8001    | 0.7210 | 0.0227 |
| 2            | 0.9863   | 0.9908   | 0.9995  | 0.9801    | 0.9926 | 0.0059 |
| 3            | 0.9825   | 0.9882   | 0.9994  | 0.9732    | 0.9921 | 0.0065 |
| 4            | 0.7922   | 0.8614   | 0.9256  | 0.8045    | 0.7845 | 0.0258 |
| 5            | 0.9515   | 0.9673   | 0.9944  | 0.9469    | 0.9565 | 0.0127 |
| 6            | 0.9902   | 0.9934   | 0.9997  | 0.9848    | 0.9957 | 0.0041 |
| 7            | 0.9921   | 0.9947   | 0.9998  | 0.9869    | 0.9974 | 0.0027 |
| 8            | 0.8749   | 0.9170   | 0.9719  | 0.8891    | 0.8631 | 0.0192 |
| 9            | 0.9702   | 0.9798   | 0.9980  | 0.9621    | 0.9785 | 0.0096 |
| 10           | 0.9895   | 0.9930   | 0.9998  | 0.9830    | 0.9962 | 0.0043 |

**Best Parameters:**

```json
{
  "n_estimators": 480,
  "max_depth": 19,
  "learning_rate": 0.1867,
  "subsample": 0.7843,
  "colsample_bytree": 0.6701,
  "gamma": 0.2101,
  "reg_alpha": 0.0692,
  "reg_lambda": 0.4269
}
```

---

### Logistic Regression

| CV Iteration | F1 Score | Accuracy | ROC AUC | Precision | Recall | F1 Std |
|--------------|----------|----------|---------|-----------|--------|--------|
| 1            | 0.7828   | 0.8223   | 0.8954  | 0.7162    | 0.8857 | 0.0995 |
| 2            | 0.7820   | 0.8216   | 0.8951  | 0.7152    | 0.8853 | 0.0996 |
| 3            | 0.7830   | 0.8225   | 0.8955  | 0.7164    | 0.8858 | 0.0993 |
| 4            | 0.7828   | 0.8224   | 0.8954  | 0.7162    | 0.8858 | 0.0995 |
| 5            | 0.7809   | 0.8206   | 0.8947  | 0.7139    | 0.8847 | 0.0997 |
| 6            | 0.7814   | 0.8211   | 0.8949  | 0.7145    | 0.8849 | 0.0997 |
| 7            | 0.7830   | 0.8225   | 0.8955  | 0.7164    | 0.8858 | 0.0994 |
| 8            | 0.7829   | 0.8225   | 0.8955  | 0.7163    | 0.8858 | 0.0994 |
| 9            | 0.7828   | 0.8223   | 0.8954  | 0.7162    | 0.8857 | 0.0995 |
| 10           | 0.7830   | 0.8225   | 0.8955  | 0.7164    | 0.8858 | 0.0994 |

**Best Parameters:**

```json
{
  "penalty": "elasticnet",
  "C": 1.1723,
  "max_iter": 1143,
  "tol": 0.0004264,
  "l1_ratio": 0.6383,
  "solver": "saga",
  "random_state": 42,
  "n_jobs": 10,
  "class_weight": "balanced"
}
```

---

## Evaluation

### Metrics (Test Set)

| Model                       | Specificity | AUROC  | F1 Score | Precision | Recall |
|-----------------------------|-------------|--------|----------|-----------|--------|
| **Random Forest (Tuned)**   | 0.9913      | 0.9760 | 0.5594   | 0.5280    | 0.5948 |
| **XGBoost (Tuned)**         | 0.9978      | 0.9998 | 0.9962   | 0.9830    | 0.9962 |
| **Logistic Regression (Tuned)** | 0.8955      | 0.8955 | 0.7830   | 0.7164    | 0.8858 |

---

## Model Cards

### Random Forest (Tuned)

**Run ID:** `20241213_210538`  
**Training Date:** `2024-12-13 22:42:35`

---

### Model Details

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

### Training Data

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

### Evaluation

#### Metrics (Test Set)

| Metric       | Value  |
|--------------|--------|
| Specificity  | 0.9913 |
| AUROC        | 0.9760 |
| F1 Score     | 0.5594 |
| Precision    | 0.5280 |
| Recall       | 0.5948 |

#### Confusion Matrix

![Confusion Matrix](random_forest_(tuned)_confusion_matrix.png)

#### ROC Curve

![ROC Curve](random_forest_(tuned)_roc_curve.png)

#### Precision-Recall Curve

![Precision-Recall Curve](random_forest_(tuned)_precision_recall_curve.png)

#### Feature Importance

![Feature Importance](random_forest_(tuned)_feature_importance.png)

---

### Usage and Limitations

- **Intended Use:** Early warning system for sepsis in ICU patients.
- **Limitations:**
  - The model was trained on data from a specific population and may not generalize well to other populations.
  - The model's performance may be limited in cases with atypical presentations of sepsis.
  - The model is not fully interpretable.

---

### Ethical Considerations

- **Fairness:** The model's performance should be monitored across different demographic groups to ensure fairness.
- **Privacy:** Patient data was anonymized during model training.
- **Transparency:** This model card provides information about the model's development, performance, and limitations.

---

## Code Overview

### `main.py`

- **Functionality:** Orchestrates the entire pipeline: data loading, splitting, preprocessing, resampling, model tuning, training, evaluation, model card generation, and final saving to the model registry.
- **Features:**
  - Integrates Optuna for hyperparameter tuning with checkpointing for reproducibility.
  - Handles model training and evaluation for Random Forest, Logistic Regression, and XGBoost.
  - Generates comprehensive evaluation reports and model cards.
  - Manages logging and memory usage monitoring.

### `src/data_processing.py`

- **Functionality:** Provides functions for loading, validating, and splitting the dataset at the patient level.
- **Features:**
  - Ensures no patient overlap between training, validation, and test splits.
  - Validates dataset integrity and structure.

### `src/feature_engineering.py`

- **Functionality:** Conducts extensive preprocessing including:
  - Dropping redundant and null columns.
  - Imputing missing values using Iterative Imputer (MICE algorithm).
  - One-hot encoding of categorical variables.
  - Log transformation of skewed features.
  - Scaling of numerical features using StandardScaler and RobustScaler.

### `src/evaluation.py`

- **Functionality:** Offers a comprehensive evaluation suite.
- **Features:**
  - Computes multiple evaluation metrics.
  - Generates various plots (Confusion Matrix, ROC Curve, Precision-Recall Curve, Feature Importance).
  - Saves evaluation metrics and visualizations in JSON and image formats.

### `src/model_registry.py`

- **Functionality:** Manages model lifecycle, enabling versioning and traceability of model artifacts.
- **Features:**
  - Saves and retrieves models along with their metadata.
  - Facilitates tracking of model performance over different versions.

### `src/logger_utils.py` and `src/logger_config.py`

- **Functionality:** Provides decorators and utilities for structured logging at every step of the pipeline.
- **Features:**
  - Ensures thorough traceability and debuggability.
  - Configures logging formats and handlers.

### `src/utils.py`

- **Functionality:** Contains helper functions for logging and saving metrics.
- **Features:**
  - Facilitates reusable utility operations across the pipeline.

---

## Model Cards

Each successful run generates a model card (e.g., `Random_Forest_(Tuned)_model_card.md`). The model card includes:

- Model version and training timestamp
- Training data statistics and preprocessing steps
- Hyperparameters and model architecture details
- Performance metrics and evaluation plots
- Ethical considerations and limitations of the model

---

## Contributing

Contributions are welcome! To contribute:

1. **Fork the Repository.**
2. **Create a New Branch for Your Changes.**
3. **Commit and Push Your Code.**
4. **Submit a Pull Request with a Clear Description.**

Please ensure that your contributions adhere to the project's coding standards and include appropriate tests and documentation.

---

## Contact

For any questions or suggestions, please contact:

- **Jeremy Cleland** - [jeremy.cleland@example.com](mailto:jdcl@umich.edu)
- **Anthony Lewis** - [anthony.lewis@example.com](mailto:alewi@umich.edu)
- **Salif Khan** - [salif.khan@example.com](mailto:khansaif@umich.edu)

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

- [PhysioNet Sepsis Prediction Dataset](https://www.kaggle.com/datasets/salikhussaini49/prediction-of-sepsis)
- [Optuna - Hyperparameter Optimization Framework](https://optuna.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [Imbalanced-learn](https://imbalanced-learn.org/)
