# Sepsis Prediction Project

## Overview

This project aims to predict sepsis in patients using machine learning models. The workflow includes data preprocessing, feature engineering, model training, evaluation, and testing on external data.

## Project Structure

```bash
sepsis_prediction/
│
├── data/
│   ├── raw/
│   │   └── Dataset.csv
│   ├── processed/
│   │   ├── train_data.csv
│   │   ├── test_data.csv
│   │   └── val_data.csv
│
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── logger_config.py
│   ├── logger_untils.py
│   ├── evaluation.py
│   ├── model_registry.py
│   └── utils.py
│
├── tests/
│   └── test_data_processing.py
│
├── main.py
├── requirements.txt
├── .gitignore
├── README.md
└── registry.json
```

## Key Features

- **Data Loading and Preprocessing:**
  - Handles loading data from CSV files with error handling and validation.
  - Performs data splitting into training, validation, and testing sets.
  - Applies extensive preprocessing including missing value imputation, feature scaling, log transformation, and one-hot encoding.
- **Class Imbalance Handling:**
  - Employs SMOTEENN for robust class imbalance handling, combining oversampling and undersampling techniques.
- **Model Training:**
  - Supports training for Random Forest, Logistic Regression, and XGBoost classifiers.
  - Includes hyperparameter optimization using Optuna with cross-validation.
  - Provides reusable model training and evaluation functions for consistent experimentation.
- **Hyperparameter Optimization:**
  - Uses Optuna for efficient hyperparameter tuning, optimizing for the F1 score.
  - Allows for parallel optimization using multiple jobs.
  - Supports loading and creating new Optuna studies with Median Pruner
- **Model Evaluation:**
  - Generates a wide array of evaluation metrics including Accuracy, Precision, Recall, F1 Score, AUC-ROC, etc.
  - Creates detailed visualizations such as confusion matrices, ROC curves, precision-recall curves, feature importance plots, and more.
  - Produces detailed error analysis plots broken down by patient groups.
  - Includes temporal progression plots and prediction timelines to understand model behavior over time.
  - Generates calibration plots to understand the agreement between predicted and observed probabilities
- **Model Registry:**
  - Provides a structured model registry to save, load, and manage trained models.
  - Allows for versioning of models with automatic timestamping and unique identifiers.
  - Supports saving model parameters, metrics, and associated artifacts for full reproducibility and tracking.
  - Offers functionalities for fetching the best performing model based on a specified metric.
- **Logging:**
  - Utilizes a comprehensive logging system for detailed tracking of pipeline execution.
  - Supports both console and file logging, including JSON formatting.
  - Provides logging utilities for memory tracking, step completion, and function execution duration.
- **Reproducibility:**
  - Utilizes checkpoints to save intermediate data, which allows for faster re-runs.
  - Includes saving of the hyperparameters of the models to facilitate reproduction of best models.
  - Ensures deterministic behavior using random seeds where applicable.
- **Modularity and Extensibility:**
  - Designed with modularity in mind with separation of concerns and reusable functions.
  - Facilitates easy addition of new models, evaluation metrics, or data processing steps.
- **Model Cards**:
  - Generates detailed markdown model cards, providing essential information about the model architecture, training dataset, evaluation metrics, and ethical considerations.

## Setup and Installation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/Jeremy-Cleland/sepsis-early-detection
    cd sepsis-prediction-pipeline
    ```

2. **Set up a Virtual Environment:**

    ```bash
    conda -env create -f environment.yml
    ```

3. **Activate the Virtual Environment:**

    ```bash
    conda activate sepsis-prediction
    ```

4. **Data:** Download the dataset and place it in the `data/raw` directory named `Dataset.csv`.
    - Dataset: [Sepsis Patient Data](https://www.kaggle.com/datasets/salikhussaini49/prediction-of-sepsis)

## Usage

1. **Run the main script:**

    ```bash
    python main.py --help
    ```

    This command will display the available command-line arguments.

    Example usage:

    ```bash
    python main.py --optuna-n-jobs 8 --rf-trials 10 --lr-trials 10 --xgb-trials 10  --data-path data/raw/Dataset.csv --report-dir reports/evaluations --model-dir models
    ```

    or create new optuna studies even if checkpoint exists

    ```bash
    python main.py --optuna-n-jobs 8 --rf-trials 10 --lr-trials 10 --xgb-trials 10  --data-path data/raw/Dataset.csv --report-dir reports/evaluations --model-dir models --new-study
    ```

    The `--new-study` flag will force the pipeline to create new Optuna studies instead of loading from checkpoint which is very useful to train the model and get a full optimization.

2. **Reviewing Results**
    - The results of the model training and hyperparameter optimization, including the metrics, plots, and model cards can be found inside the `/reports/evaluations` directory. Each run will have its own folder with the `run_YYYYMMDD_HHMMSS` naming format.
    - The model cards are in markdown format inside each run directory, along with the relevant plots of the metrics.

## Command-Line Arguments

The `main.py` script accepts several command-line arguments to customize the pipeline:

- `--optuna-n-jobs`: Number of parallel jobs for Optuna hyperparameter tuning (default: 10)
- `--rf-trials`: Number of trials for Random Forest optimization (default: 5)
- `--lr-trials`: Number of trials for Logistic Regression optimization (default: 5)
- `--xgb-trials`: Number of trials for XGBoost optimization (default: 5)
- `--data-path`: Path to the raw dataset CSV file (default: `data/raw/Dataset.csv`)
- `--report-dir`: Directory to save evaluation reports and plots (default: `reports/evaluations`)
- `--model-dir`: Directory to save trained models (default: `models`)
- `--new-study`: Create new Optuna studies instead of loading from checkpoint (default: False)

## Code Overview

### `main.py`

The main execution script orchestrates the entire machine learning pipeline:

- Parses command-line arguments for customization.
- Loads and preprocesses the dataset, splitting into training, validation, and test sets.
- Applies SMOTEENN to balance class distribution in the training data.
- Conducts hyperparameter tuning using Optuna for Random Forest, Logistic Regression, and XGBoost.
- Trains the models using the best hyperparameters.
- Evaluates the models on the test set with comprehensive metrics and visualizations.
- Saves the best model and all the associated artifacts using the `ModelRegistry`.
- Exports the optimization trials to CSV files to analyze in depth.

### `src/data_processing.py`

This module provides functions for data loading and preprocessing:

- `load_data()`: Loads the dataset from a CSV file.
- `split_data()`: Splits the dataset into training, validation, and test sets.
- `load_processed_data()`: Loads pre-processed data with validations.
- `validate_dataset()`: Validation of dataset format and content.

### `src/feature_engineering.py`

This module contains feature engineering and preprocessing logic:

- `drop_columns()`: Removes redundant columns.
- `fill_missing_values()`: Imputes missing values using iterative imputation.
- `drop_null_columns()`: Drops specified columns with missing values.
- `one_hot_encode_gender()`: One-hot encodes the Gender column.
- `log_transform()`: Applies a log transformation to specified columns.
- `standard_scale()`: Applies standard scaling to specified columns.
- `robust_scale()`: Applies robust scaling to specified columns.
- `preprocess_data()`: Orchestrates the complete data preprocessing pipeline.

### `src/evaluation.py`

This module handles model evaluation and visualization:

- `evaluate_model()`: Calculates evaluation metrics and generates plots.
- `generate_evaluation_plots()`: Generates various plots (confusion matrix, ROC curve, etc.)
- `plot_confusion_matrix()`: Plots the confusion matrix.
- `plot_class_distribution()`: Plots class distribution.
- `plot_feature_importance()`: Plots feature importances.
- `plot_roc_curve()`: Plots the Receiver Operating Characteristic (ROC) curve.
- `plot_precision_recall_curve()`: Plots the precision-recall curve.
- `plot_feature_correlation_heatmap()`: Plots a feature correlation heatmap.
- `plot_missing_values()`: Plots the missing values heatmap.
- `plot_temporal_progression()`: Plots the temporal progression of vital signs.
- `plot_error_analysis()`: Plots the error analysis broken down by patient groups.
- `plot_calibration()`: Plots the calibration curve
- `plot_prediction_timeline()`: Plots the prediction timeline
- `plot_feature_interactions()`: Plots feature interactions.
- `save_plot()`: Saves a given plot to the reports directory.
- `log_metrics()`: Logs evaluation metrics to the console/file.
- `save_metrics_to_json()`: Saves the metrics to a JSON file.

### `src/logger_config.py`

This module sets up logging configuration for the application:

- `setup_logger()`: Configures a logger with console and file handlers.
- `get_logger()`: Retrieves an existing logger, or creates a new one if it doesn't exist
- `disable_duplicate_logging()`: Prevents duplicate log messages.

### `src/logger_utils.py`

This module includes logging utility functions:

- `log_phase()`: Decorator for logging phase start and end information.
- `log_step()`: Context manager for logging steps within phases.
- `log_function()`: Decorator for logging function calls.
- `log_memory()`: Logs current memory usage.
- `log_dataframe_info()`: Logs information about a DataFrame (shape, missing values, etc).

### `src/model_registry.py`

This module implements a model registry for tracking and managing models:

- `ModelVersion` : A dataclass for storing information about a model.
- `ModelRegistry`: A class to manage the model lifecycle. It includes functionalities to:
  - `save_model()`: Save a trained model with its metadata.
  - `load_model()`: Load a specific model version.
  - `get_best_model()`: Load the best performing model based on a metric.
  - `list_models()`: List all the registered models.
  - `get_model_versions()`: Get all the versions of a specific model.
  - `delete_model()`: Deletes a specific model version.

### `src/resampling.py`

This module includes custom resampling functions that are not being used in the pipeline:

- `ModernSMOTEENN`: A modern implementation of SMOTEENN.

### `src/utils.py`

This module provides general utility functions:

- `log_message()`: Logs a message using a logger or prints to console.
- `log_metrics()`: Logs evaluation metrics.
- `save_metrics_to_json()`: Saves metrics to a JSON file.

## Contributing

Contributions to this project are welcome. Please follow these guidelines:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with descriptive messages.
4. Submit a pull request with a detailed description of your changes.
