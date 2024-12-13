
# Sepsis Prediction Project

## Overview

This project aims to predict sepsis in patients using advanced machine learning models. The workflow includes data preprocessing, feature engineering, class imbalance handling, hyperparameter optimization, model training, evaluation, model card generation, and model registry management for reproducibility and scalability.

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

## Key Features

- **Data Loading and Preprocessing:**
  - Loads data from CSV files with robust error handling.
  - Splits into training, validation, and test sets at the patient level to avoid data leakage.
  - Imputes missing values, encodes categorical features, applies transformations (log, scaling), and drops redundant or highly correlated features.

- **Class Imbalance Handling:**
  - Utilizes SMOTEENN to address class imbalance, combining synthetic oversampling and undersampling techniques.

- **Model Training and Hyperparameter Optimization:**
  - Supports Random Forest, Logistic Regression, and XGBoost models.
  - Hyperparameter tuning with Optuna using cross-validation and the MedianPruner for efficient pruning.
  - The pipeline can automatically create or load Optuna studies from checkpoints, enabling reproducible experiments.

- **Comprehensive Evaluation and Visualization:**
  - Computes multiple metrics: Accuracy, Precision, Recall, F1 Score, AUC-ROC, Specificity, and more.
  - Generates extensive visualizations: Confusion matrices, ROC curves, Precision-Recall curves, Feature Importance, Missing Value Heatmaps, Temporal Progression plots, Error Analysis by patient groups, Calibration curves, and Feature Interactions.
  - Produces prediction timelines to understand how the model's probability predictions evolve over time for individual patients.

- **Model Cards:**
  - Automatically generates detailed model cards in Markdown format.
  - The model card includes training details, hyperparameters, performance metrics, feature importance, and ethical considerations.

- **Model Registry:**
  - A centralized registry to version, save, and load models along with their metrics, parameters, and artifacts.
  - Facilitates retrieval of the best model based on a chosen metric.
  - Allows model lifecycle management (saving, loading, deleting versions).

- **Logging and Checkpoints:**
  - Detailed logging (both console and file) for all steps of the pipeline.
  - Uses checkpoints to store intermediate preprocessing steps, enabling faster re-runs without repeating heavy computations.

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

## Usage

1. **Run the main pipeline:**

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
   - Models are saved under `models/` directory along with their metadata.

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

Increase the trials parameters for thorough hyperparameter optimization.

## Code Overview

### `main.py`

- Orchestrates the entire pipeline: data loading, splitting, preprocessing, resampling, model tuning, training, evaluation, model card generation, and final saving to the model registry.
- Integrates Optuna for hyperparameter tuning and automatically uses checkpoints to resume studies.
- On completion, saves the best model and artifacts.

### `src/data_processing.py`

- Provides functions for loading, validating, and splitting the dataset at a patient level.
- Ensures no patient overlap between train, validation, and test splits.

### `src/feature_engineering.py`

- Conducts extensive preprocessing including:
  - Dropping redundant columns
  - Missing value imputation using IterativeImputer
  - One-hot encoding of categorical variables
  - Log transformation of skewed features
  - Scaling of numerical features

### `src/evaluation.py`

- Offers a comprehensive evaluation suite.
- Generates multiple plots (confusion matrix, ROC, PRC, calibration, temporal analysis, error analysis).
- Logs evaluation metrics and saves them in JSON format.

### `src/model_registry.py`

- Manages model lifecycle, enabling versioning and traceability of model artifacts.
- Allows retrieval of the best model based on a chosen performance metric.

### `src/logger_utils.py` and `src/logger_config.py`

- Provides decorators and utilities for structured logging at every step of the pipeline.
- Ensures thorough traceability and debuggability.

### `src/utils.py`

- Helper functions for logging and saving metrics.

## Model Cards

Each successful run generates a model card (e.g., `Random_Forest_(Tuned)_model_card.md`). The model card includes:

- Model version and training timestamp
- Training data statistics and preprocessing steps
- Hyperparameters and model architecture details
- Performance metrics and evaluation plots
- Ethical considerations and limitations of the model

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository.
2. Create a new branch for your changes.
3. Commit and push your code.
4. Submit a pull request with a clear description.
