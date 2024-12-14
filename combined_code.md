# Combined Python Code

## combine.py

```python
import os
from pathlib import Path


def combine_python_files(directory: str, output_file: str = "combined_code.md"):
    """
    Find all Python files in the given directory and its subdirectories,
    and combine their contents into a single Markdown file.

    Args:
        directory (str): Root directory to search for Python files
        output_file (str): Name of the output Markdown file
    """
    # Convert directory to Path object
    root_dir = Path(directory)

    # Find all Python files
    python_files = list(root_dir.rglob("*.py"))

    # Sort files for consistent output
    python_files.sort()

    # Create or overwrite the output file
    with open(output_file, "w", encoding="utf-8") as outfile:
        outfile.write("# Combined Python Code\n\n")

        # Process each Python file
        for file_path in python_files:
            # Get relative path from root directory
            try:
                relative_path = file_path.relative_to(root_dir)
            except ValueError:
                relative_path = file_path

            # Write file header
            outfile.write(f"## {relative_path}\n\n")
            outfile.write("```python\n")

            # Read and write file contents
            try:
                with open(file_path, "r", encoding="utf-8") as infile:
                    content = infile.read()
                    outfile.write(content)

                    # Ensure there's a newline at the end
                    if not content.endswith("\n"):
                        outfile.write("\n")
            except Exception as e:
                outfile.write(f"# Error reading file: {str(e)}\n")

            outfile.write("```\n\n")


if __name__ == "__main__":
    # Get the current working directory
    current_dir = os.getcwd()

    print(f"Searching for Python files in: {current_dir}")
    combine_python_files(current_dir)
    print("Done! Check combined_code.md for the output.")
```

## main.py

```python
import datetime
import gc
import json
import logging
import os
import argparse
import time
from typing import Any, Tuple, Dict

import joblib
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import optuna
import psutil  # For memory usage

from src.logger_config import get_logger, disable_duplicate_logging
from src.model_registry import ModelRegistry
from src.logger_utils import log_dataframe_info, log_step, log_phase
from src import (
    evaluate_model,
    load_data,
    preprocess_data,
    split_data,
)
from src.evaluation import plot_class_distribution, generate_evaluation_plots


def parse_arguments():
    """Parse command-line arguments for configurability."""
    parser = argparse.ArgumentParser(description="Sepsis Prediction Pipeline")
    parser.add_argument(
        "--optuna-n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for Optuna hyperparameter tuning (default: 10)",
    )
    parser.add_argument(
        "--rf-trials",
        type=int,
        default=10,
        help="Number of trials for Random Forest optimization (default: 20)",
    )
    parser.add_argument(
        "--lr-trials",
        type=int,
        default=10,
        help="Number of trials for Logistic Regression optimization (default: 20)",
    )
    parser.add_argument(
        "--xgb-trials",
        type=int,
        default=10,
        help="Number of trials for XGBoost optimization (default: 20)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/Dataset.csv",
        help="Path to the raw dataset CSV file",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="reports/evaluations",
        help="Directory to save evaluation reports and plots",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--new-study",
        action="store_true",
        help="Create new Optuna studies instead of loading from checkpoint",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-training and hyperparameter tuning by ignoring existing checkpoints.",
    )
    return parser.parse_args()


# Initialize logger and related components once at the top level
args = parse_arguments()
os.makedirs("logs", exist_ok=True)
disable_duplicate_logging()
logger = get_logger(
    name="sepsis_prediction",
    level="INFO",
    use_json=False,
)
logging.getLogger().setLevel(logging.WARNING)
model_registry = ModelRegistry(base_dir="./", logger=logger)


def create_or_load_studies(
    storage_url: str, new_study: bool = False
) -> Dict[str, optuna.Study]:
    """
    Create new or load existing Optuna studies.

    Args:
        storage_url: URL for the SQLite database
        new_study: If True, creates new studies even if checkpoint exists

    Returns:
        Dictionary of study names to Study objects
    """
    studies = {}
    study_names = {
        "Random_Forest_Optimization": "Random Forest Study",
        "Logistic_Regression_Optimization": "Logistic Regression Study",
        "XGBoost_Optimization": "XGBoost Study",
    }

    for study_id, study_name in study_names.items():
        if new_study:
            # Generate unique study name with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_study_name = f"{study_id}_{timestamp}"

            studies[study_id] = optuna.create_study(
                direction="maximize",
                study_name=unique_study_name,
                storage=storage_url,
                load_if_exists=False,  # Force new study creation
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=5, n_warmup_steps=10
                ),
            )
        else:
            # Try to load existing study or create new one
            studies[study_id] = optuna.create_study(
                direction="maximize",
                study_name=study_id,
                storage=storage_url,
                load_if_exists=True,
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=5, n_warmup_steps=10
                ),
            )

    return studies


def train_and_evaluate_model(
    model_name: str,
    pipeline: ImbPipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    df_val_original: pd.DataFrame,
    unique_report_dir: str,
    logger: logging.Logger,
) -> Tuple[Dict[str, float], ImbPipeline]:
    """
    Trains and evaluates a model. Returns the entire pipeline instead of just the estimator.

    Args:
        model_name: Name of the model (e.g., "Random Forest (Tuned)")
        pipeline: imbalanced-learn Pipeline object containing preprocessing and the estimator
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        df_val_original: Original validation DataFrame (including Patient_ID and SepsisLabel)
        unique_report_dir: Directory to save reports and plots for this run
        logger: Logger object for logging information and errors

    Returns:
        A tuple containing:
            - metrics: Dictionary of evaluation metrics
            - pipeline: The entire trained pipeline object
    """
    logger.info(f"Training {model_name}...")

    try:
        # Train the model
        pipeline.fit(X_train, y_train)
        logger.info(f"Training completed for {model_name}.")

        # Make predictions
        y_pred = pipeline.predict(X_val)
        y_pred_proba = (
            pipeline.predict_proba(X_val)[:, 1]
            if hasattr(pipeline, "predict_proba")
            else None
        )
        logger.info(f"Predictions made for {model_name}.")

        # Convert predictions to pandas.Series
        y_pred = pd.Series(y_pred, index=y_val.index, name="Predicted")
        y_pred_proba = (
            pd.Series(y_pred_proba, index=y_val.index, name="Predicted_Prob")
            if y_pred_proba is not None
            else None
        )

        # Merge Patient_ID back for evaluation
        df_val_processed = df_val_original.copy()
        df_val_processed[y_pred.name] = y_pred
        if y_pred_proba is not None:
            df_val_processed[y_pred_proba.name] = y_pred_proba

        # Check if Patient_ID exists in df_val_original before merging
        if "Patient_ID" in df_val_original.columns:
            logger.debug("Patient_ID found in df_val_original.")
        else:
            logger.debug("Patient_ID not found in df_val_original.")

        # Check if Patient_ID exists in df_val_processed after merging
        if "Patient_ID" in df_val_processed.columns:
            logger.debug("Patient_ID found in df_val_processed after merging.")
        else:
            logger.debug("Patient_ID not found in df_val_processed after merging.")

        # Map display names to pipeline step names
        step_name_mapping = {
            "Random Forest (Tuned)": "random_forest",
            "Logistic Regression (Tuned)": "logistic_regression",
            "XGBoost (Tuned)": "xgb_classifier",
        }

        # Get the correct pipeline step name
        estimator_step = step_name_mapping.get(model_name)
        if estimator_step is None:
            logger.warning(
                f"No mapping found for model name: {model_name}. Using the last step."
            )
            estimator_step = pipeline.steps[-1][0]

        # Extract the model from the pipeline
        model = pipeline.named_steps.get(estimator_step)
        if model is None:
            model = pipeline[-1]
            logger.warning(
                f"Estimator step '{estimator_step}' not found in the pipeline. Using the last step '{pipeline.steps[-1][0]}' instead."
            )

        # Log the extracted model
        logger.info(f"Extracted model for evaluation: {type(model)}")

        # Evaluate the model, handling potential errors by re-raising
        try:
            metrics = evaluate_model(
                y_true=y_val,
                y_pred=y_pred,
                data=df_val_processed,
                model_name=model_name,
                y_pred_proba=y_pred_proba,
                model=model,
                report_dir=unique_report_dir,
                logger=logger,
            )
            logger.info(f"Evaluation completed for {model_name}.")
        except Exception as eval_error:
            logger.error(
                f"Error evaluating {model_name}: {str(eval_error)}", exc_info=True
            )
            raise

        return metrics, pipeline  # Return the entire pipeline instead of just the model

    except Exception as e:
        logger.error(
            f"Error training or evaluating {model_name}: {str(e)}", exc_info=True
        )
        raise

    finally:
        del pipeline
        gc.collect()


def save_hyperparameters(
    filename: str,
    params: Dict[str, Any],
    logger: logging.Logger,
    report_dir: str,
) -> None:
    """Save hyperparameters to a JSON file.

    Args:
        filename: Name of the JSON file to save the hyperparameters.
        params: Dictionary of hyperparameters.
        logger: Logger object for logging information and errors.
        report_dir: Directory where the JSON file will be saved.
    """
    os.makedirs(report_dir, exist_ok=True)
    params_path = os.path.join(report_dir, filename)
    try:
        with open(params_path, "w") as f:
            json.dump(params, f, indent=4)
        logger.info(f"Saved hyperparameters to {params_path}")
    except Exception as e:
        logger.error(f"Failed to save hyperparameters to {params_path}: {e}")
        raise


def generate_model_card(
    model,
    model_name,
    metrics,
    train_data,
    val_data,
    test_data,
    report_dir,
    run_id,
    logger,
):
    """Generates a model card in Markdown format.

    Args:
        model: Trained model object.
        model_name: Name of the model.
        metrics: Dictionary of evaluation metrics.
        train_data: Training DataFrame.
        val_data: Validation DataFrame.
        test_data: Test DataFrame.
        report_dir: Directory to save the model card and plots.
        run_id: Unique identifier for this run.
        logger: Logger object for logging information and errors.
    """

    model_card_path = os.path.join(report_dir, f"{model_name}_model_card.md")

    with open(model_card_path, "w") as f:
        f.write(f"# Model Card: {model_name}\n\n")
        f.write(f"**Run ID:** {run_id}\n\n")
        f.write(
            f"**Training Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )

        # Model Details
        f.write("## Model Details\n\n")
        if "random_forest" in model_name.lower():
            rf_step = model.named_steps.get("random_forest")
            if rf_step:
                n_estimators = rf_step.n_estimators
                max_depth = rf_step.max_depth
                f.write(f"- **Model Type:** Random Forest\n")
                f.write(
                    f"- **Architecture:** Ensemble of {n_estimators} decision trees (max_depth={max_depth}) trained using bagging.\n"
                )
            else:
                logger.warning(
                    f"Random Forest step not found in the model pipeline for {model_name}."
                )
                f.write("- **Model Type:** Random Forest (Details not available)\n")
        elif "logistic_regression" in model_name.lower():
            lr_step = model.named_steps.get("logistic_regression")
            if lr_step:
                penalty = lr_step.penalty
                C = lr_step.C
                f.write(f"- **Model Type:** Logistic Regression\n")
                f.write(
                    f"- **Architecture:** Linear model with sigmoid function (regularization: {penalty}, C={C}).\n"
                )
            else:
                logger.warning(
                    f"Logistic Regression step not found in the model pipeline for {model_name}."
                )
                f.write(
                    "- **Model Type:** Logistic Regression (Details not available)\n"
                )
        elif "xgboost" in model_name.lower():
            xgb_step = model.named_steps.get("xgb_classifier")
            if xgb_step:
                learning_rate = xgb_step.learning_rate
                max_depth = xgb_step.max_depth
                f.write(f"- **Model Type:** XGBoost\n")
                f.write(
                    f"- **Architecture:** Gradient boosting with decision trees (learning_rate={learning_rate}, max_depth={max_depth}).\n"
                )
            else:
                logger.warning(
                    f"XGBoost step not found in the model pipeline for {model_name}."
                )
                f.write("- **Model Type:** XGBoost (Details not available)\n")

        # Add version if applicable
        f.write(f"- **Version:** v1.0\n")
        f.write(f"- **Hyperparameters:**\n")
        for step_name, step in model.named_steps.items():
            if step_name != "scaler":
                f.write(f"  - **{step_name.capitalize()}**:\n")
                for hyperparam, hyperval in step.get_params().items():
                    f.write(f"      - {hyperparam}: {hyperval}\n")

        # Training Data Summary
        f.write("\n## Training Data\n\n")
        f.write("- **Dataset:** PhysioNet Sepsis Prediction Dataset\n")
        f.write(f"- **Samples:**\n")
        f.write(f"    - Training: {len(train_data)}\n")
        f.write(f"    - Validation: {len(val_data)}\n")
        f.write(f"    - Test: {len(test_data)}\n")
        f.write(f"- **Features:** {train_data.shape[1]} (after preprocessing)\n")
        f.write(f"- **Class Distribution (Training):**\n")
        train_class_counts = train_data["SepsisLabel"].value_counts(normalize=True)
        f.write(f"    - Sepsis: {train_class_counts[1]:.2%}\n")
        f.write(f"    - Non-sepsis: {train_class_counts[0]:.2%}\n")
        f.write(
            "- **Preprocessing:** Missing value imputation (median), scaling (StandardScaler), SMOTEENN resampling\n"
        )

        # Evaluation
        f.write("\n## Evaluation\n\n")
        f.write(f"### Metrics (Test Set)\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|---|---|\n")
        f.write(f"| F1 Score | {metrics.get('F1 Score', 'N/A'):.4f} |\n")
        f.write(f"| Precision | {metrics.get('Precision', 'N/A'):.4f} |\n")
        f.write(f"| Recall | {metrics.get('Recall', 'N/A'):.4f} |\n")
        f.write(f"| Specificity | {metrics.get('Specificity', 'N/A'):.4f} |\n")
        f.write(f"| AUROC | {metrics.get('AUC-ROC', 'N/A'):.4f} |\n")
        f.write(f"| AUPRC | {metrics.get('AUPRC', 'N/A'):.4f} |\n")

        # Generate and save essential plots
        generate_evaluation_plots(
            y_true=test_data["SepsisLabel"],
            y_pred=pd.Series(
                model.predict(test_data.drop("SepsisLabel", axis=1)),
                index=test_data.index,
                name="Predicted",
            ),
            y_pred_proba=(
                pd.Series(
                    model.predict_proba(test_data.drop("SepsisLabel", axis=1))[:, 1],
                    index=test_data.index,
                    name="Predicted_Prob",
                )
                if hasattr(model, "predict_proba")
                else None
            ),
            data=test_data,  # Pass the entire test_data DataFrame
            model=model,
            model_name=model_name,
            report_dir=report_dir,
            logger=logger,
        )

        # Add confusion matrix image
        f.write("\n### Confusion Matrix\n\n")
        f.write(f"![Confusion Matrix]({model_name}_confusion_matrix.png)\n")

        # Add ROC curve image
        if metrics.get("AUC-ROC") is not None:
            f.write("\n### ROC Curve\n\n")
            f.write(f"![ROC Curve]({model_name}_roc_curve.png)\n")

        # Add Precision-Recall curve image
        if metrics.get("AUPRC") is not None:
            f.write("\n### Precision-Recall Curve\n\n")
            f.write(
                f"![Precision-Recall Curve]({model_name}_precision_recall_curve.png)\n"
            )

        # Add feature importance plot if available
        # Assuming 'plot_feature_importance' saves the plot correctly
        if os.path.exists(
            os.path.join(report_dir, f"{model_name}_feature_importance.png")
        ):
            f.write("\n### Feature Importance\n\n")
            f.write(f"![Feature Importance]({model_name}_feature_importance.png)\n")

        # Usage and Limitations
        f.write("\n## Usage and Limitations\n\n")
        f.write(
            "- **Intended Use:** Early warning system for sepsis in ICU patients.\n"
        )
        f.write("- **Limitations:**\n")
        f.write(
            "    - The model was trained on data from a specific population and may not generalize well to other populations.\n"
        )
        f.write(
            "    - The model's performance may be limited in cases with atypical presentations of sepsis.\n"
        )
        f.write("    - The model is not fully interpretable.\n")

        # Ethical Considerations
        f.write("\n## Ethical Considerations\n\n")
        f.write(
            "- **Fairness:** The model's performance should be monitored across different demographic groups to ensure fairness.\n"
        )
        f.write("- **Privacy:** Patient data was anonymized during model training.\n")
        f.write(
            "- **Transparency:** This model card provides information about the model's development, performance, and limitations.\n"
        )

    logger.info(f"Model card saved to {model_card_path}")


@log_phase(logger)  # Placeholder; actual logger is defined within main()
def main():
    """Main function to execute the Sepsis Prediction Pipeline."""

    logger.info("Starting Sepsis Prediction Pipeline")

    # Record start time and initial memory usage
    start_time = time.time()
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / (1024**2)  # Convert to MB
    logger.info(f"Initial Memory Usage: {initial_memory:.2f} MB")

    # Generate a unique identifier for this run
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_report_dir = os.path.join("reports", "evaluations", f"run_{run_id}")
    os.makedirs(unique_report_dir, exist_ok=True)
    logger.info(f"Reports and metrics will be saved to: {unique_report_dir}")

    # Define run_id-specific checkpoint paths
    checkpoints = {
        "preprocessed_data": f"checkpoints/preprocessed_data_{run_id}.pkl",
        "resampled_data": f"checkpoints/resampled_data_{run_id}.pkl",
        "trained_models": f"checkpoints/trained_models_{run_id}.pkl",
        "optuna_studies": f"checkpoints/optuna_studies_{run_id}.pkl",
    }

    # Create checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)

    try:
        # Initialize best score
        models = {}
        best_score = 0
        best_model_name = None

        # Determine whether to load existing models or run optimizations
        should_run_optimizations = True  # Default to running optimizations

        if not args.force and not args.new_study:
            # Check if all run_id-specific checkpoints exist
            checkpoints_exist = all(
                os.path.exists(path) for path in checkpoints.values()
            )
            if checkpoints_exist:
                should_run_optimizations = False
                logger.info("Loading trained models from checkpoint.")
                models = joblib.load(checkpoints["trained_models"])
                # Determine the best model
                for name, model_info in models.items():
                    if model_info["metrics"]["F1 Score"] > best_score:
                        best_score = model_info["metrics"]["F1 Score"]
                        best_model_name = name
                logger.info(
                    f"Best model so far: {best_model_name} with F1 Score: {best_score:.4f}"
                )

        if should_run_optimizations:
            # Step 1: Load and preprocess data
            if os.path.exists(checkpoints["preprocessed_data"]):
                logger.info("Loading preprocessed data from checkpoint.")
                (
                    df_train_processed,
                    df_val_processed,
                    df_test_processed,
                    df_val_original,
                    df_test_original,
                ) = joblib.load(checkpoints["preprocessed_data"])
            else:
                logger.info("Loading and preprocessing data.")
                combined_df = load_data(args.data_path)
                logger.info(
                    "Splitting data into training, validation, and testing sets"
                )
                df_train, df_val, df_test = split_data(
                    combined_df, train_size=0.7, val_size=0.15
                )

                with log_step(logger, "Preprocessing training data"):
                    df_train_processed = preprocess_data(df_train)

                with log_step(logger, "Preprocessing validation data"):
                    df_val_processed = preprocess_data(df_val)
                    df_val_original = df_val.copy()  # Preserve original df_val

                with log_step(logger, "Preprocessing testing data"):
                    df_test_processed = preprocess_data(df_test)
                    df_test_original = df_test.copy()  # Preserve original df_test

                # Save preprocessed data along with original df_val and df_test
                joblib.dump(
                    (
                        df_train_processed,
                        df_val_processed,
                        df_test_processed,
                        df_val_original,
                        df_test_original,
                    ),
                    checkpoints["preprocessed_data"],
                )
                logger.info(
                    f"Saved preprocessed data to {checkpoints['preprocessed_data']}"
                )

            # Step 2: Handle class imbalance with SMOTEENN
            if os.path.exists(checkpoints["resampled_data"]):
                logger.info("Loading resampled data from checkpoint.")
                X_train_resampled, y_train_resampled = joblib.load(
                    checkpoints["resampled_data"]
                )
            else:
                logger.info("Applying SMOTEENN to training data.")
                smote_enn = SMOTEENN(
                    smote=SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=6),
                    enn=EditedNearestNeighbours(
                        n_jobs=-1,
                        n_neighbors=3,
                    ),
                    random_state=42,
                )
                X_train = df_train_processed.drop("SepsisLabel", axis=1)
                y_train = df_train_processed["SepsisLabel"]
                X_train_resampled, y_train_resampled = smote_enn.fit_resample(
                    X_train, y_train
                )

                # Plot resampled class distribution
                plot_class_distribution(
                    y=y_train_resampled,
                    model_name="Resampled Training Data",
                    report_dir=unique_report_dir,
                    title_suffix="_after_resampling",
                    logger=logger,
                )

                # Save resampled data
                joblib.dump(
                    (X_train_resampled, y_train_resampled),
                    checkpoints["resampled_data"],
                )
                logger.info(f"Saved resampled data to {checkpoints['resampled_data']}")

            # Step 3: Hyperparameter Tuning and Model Training
            # Define Optuna studies with descriptive storage URL
            storage_url = (
                f"sqlite:///sepsis_prediction_optimization_{run_id}.db"
                if args.new_study
                else "sqlite:///sepsis_prediction_optimization.db"
            )

            if os.path.exists(checkpoints["optuna_studies"]) and not args.new_study:
                logger.info("Loading Optuna studies from checkpoint.")
                studies = joblib.load(checkpoints["optuna_studies"])
            else:
                logger.info("Initializing new Optuna studies.")
                studies = create_or_load_studies(storage_url, new_study=args.new_study)
                # Save initial studies
                joblib.dump(studies, checkpoints["optuna_studies"])
                logger.info(f"Saved Optuna studies to {checkpoints['optuna_studies']}")

            parallel_jobs = args.optuna_n_jobs
            logger.info(
                f"Starting Optuna hyperparameter tuning with {parallel_jobs} parallel jobs."
            )

            # Define Optuna objective functions with additional metrics storage

            def xgb_objective(trial: optuna.Trial) -> float:
                """Objective function for XGBoost optimization."""
                param = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 0.01, 0.3, log=True
                    ),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree", 0.5, 1.0
                    ),
                    "gamma": trial.suggest_float("gamma", 0, 5),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
                }

                xgb_pipeline = ImbPipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "xgb_classifier",
                            XGBClassifier(
                                random_state=42,
                                eval_metric="logloss",
                                n_jobs=-1,
                                **param,
                            ),
                        ),
                    ]
                )

                # Cross-validation setup with multiple metrics
                scoring = {
                    "f1": "f1",
                    "accuracy": "accuracy",
                    "roc_auc": "roc_auc",
                    "precision": "precision",
                    "recall": "recall",
                }
                cv_results = cross_validate(
                    xgb_pipeline,
                    X_train_resampled,
                    y_train_resampled,
                    cv=3,
                    scoring=scoring,
                    n_jobs=-1,
                )

                # Store all metrics
                trial.set_user_attr("accuracy", cv_results["test_accuracy"].mean())
                trial.set_user_attr("roc_auc", cv_results["test_roc_auc"].mean())
                trial.set_user_attr("precision", cv_results["test_precision"].mean())
                trial.set_user_attr("recall", cv_results["test_recall"].mean())
                trial.set_user_attr("f1_std", cv_results["test_f1"].std())
                trial.set_user_attr(
                    "cv_iteration", len(studies["XGBoost_Optimization"].trials)
                )

                # Return F1 score as the primary optimization metric
                mean_f1 = cv_results["test_f1"].mean()
                trial.set_user_attr(
                    "f1_score", mean_f1
                )  # Store the objective value explicitly

                return mean_f1

            def rf_objective(trial: optuna.Trial) -> float:
                """Objective function for Random Forest optimization."""
                param = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "max_depth": trial.suggest_int("max_depth", 5, 30),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 11),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                    "max_features": trial.suggest_categorical(
                        "max_features", ["sqrt", "log2", None]
                    ),
                }
                rf_pipeline = ImbPipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "random_forest",
                            RandomForestClassifier(
                                random_state=42,
                                class_weight="balanced",
                                n_jobs=-1,
                                **param,
                            ),
                        ),
                    ]
                )

                # Cross-validation setup with multiple metrics
                scoring = {
                    "f1": "f1",
                    "accuracy": "accuracy",
                    "roc_auc": "roc_auc",
                    "precision": "precision",
                    "recall": "recall",
                }
                cv_results = cross_validate(
                    rf_pipeline,
                    X_train_resampled,
                    y_train_resampled,
                    cv=3,  # Number of folds
                    scoring=scoring,
                    n_jobs=-1,
                )
                # Store all metrics
                trial.set_user_attr("accuracy", cv_results["test_accuracy"].mean())
                trial.set_user_attr("roc_auc", cv_results["test_roc_auc"].mean())
                trial.set_user_attr("precision", cv_results["test_precision"].mean())
                trial.set_user_attr("recall", cv_results["test_recall"].mean())
                trial.set_user_attr("f1_std", cv_results["test_f1"].std())
                trial.set_user_attr(
                    "cv_iteration", len(studies["Random_Forest_Optimization"].trials)
                )

                # Return F1 score as the primary optimization metric
                mean_f1 = cv_results["test_f1"].mean()
                trial.set_user_attr(
                    "f1_score", mean_f1
                )  # Store the objective value explicitly

                return mean_f1

            def lr_objective(trial: optuna.Trial) -> float:
                """Objective function for Logistic Regression optimization."""
                # Suggest penalty type
                penalty = trial.suggest_categorical(
                    "penalty", ["l1", "l2", "elasticnet"]
                )

                # Define base parameters
                param = {
                    "C": trial.suggest_float("C", 0.001, 100, log=True),
                    "penalty": penalty,
                    "solver": "saga",  # Only saga supports all penalties
                    "max_iter": trial.suggest_int("max_iter", 500, 1500),
                    "tol": trial.suggest_float("tol", 1e-5, 1e-3),
                    "random_state": 42,
                    "n_jobs": 10,
                    "class_weight": "balanced",  # Set directly
                }

                # Add 'l1_ratio' if penalty is 'elasticnet'
                if penalty == "elasticnet":
                    param["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)

                # Create the pipeline with the current trial's parameters
                lr_pipeline = ImbPipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("logistic_regression", LogisticRegression(**param)),
                    ]
                )

                # Perform cross-validation with multiple metrics
                scoring = {
                    "f1": "f1",
                    "accuracy": "accuracy",
                    "roc_auc": "roc_auc",
                    "precision": "precision",
                    "recall": "recall",
                }
                cv_results = cross_validate(
                    lr_pipeline,
                    X_train_resampled,
                    y_train_resampled,
                    cv=3,
                    scoring=scoring,
                    n_jobs=-1,
                )
                # Store all metrics
                trial.set_user_attr("accuracy", cv_results["test_accuracy"].mean())
                trial.set_user_attr("roc_auc", cv_results["test_roc_auc"].mean())
                trial.set_user_attr("precision", cv_results["test_precision"].mean())
                trial.set_user_attr("recall", cv_results["test_recall"].mean())
                trial.set_user_attr("f1_std", cv_results["test_f1"].std())
                trial.set_user_attr(
                    "cv_iteration",
                    len(studies["Logistic_Regression_Optimization"].trials),
                )

                # Return F1 score as the primary optimization metric
                mean_f1 = cv_results["test_f1"].mean()
                trial.set_user_attr(
                    "f1_score", mean_f1
                )  # Store the objective value explicitly

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

                return mean_f1

            # Optimize and train Random Forest within a try-except block
            try:
                logger.info("Optimizing Random Forest hyperparameters with Optuna.")
                studies["Random_Forest_Optimization"].set_user_attr(
                    "model_type", "RandomForest"
                )
                studies["Random_Forest_Optimization"].set_user_attr(
                    "optimization_metric", "F1 Score"
                )
                studies["Random_Forest_Optimization"].set_user_attr(
                    "dataset_size", len(X_train_resampled)
                )
                studies["Random_Forest_Optimization"].set_user_attr(
                    "class_distribution", str(dict(y_train_resampled.value_counts()))
                )
                studies["Random_Forest_Optimization"].set_user_attr("cv_folds", 3)
                studies["Random_Forest_Optimization"].set_user_attr(
                    "timestamp", datetime.datetime.now().isoformat()
                )

                studies["Random_Forest_Optimization"].optimize(
                    rf_objective, n_trials=args.rf_trials, n_jobs=parallel_jobs
                )
                best_rf_params = studies["Random_Forest_Optimization"].best_params
                logger.info(f"Best Random Forest parameters: {best_rf_params}")

                # Log the number of trials and their statuses
                study_rf = studies["Random_Forest_Optimization"]
                completed_trials = [
                    t
                    for t in study_rf.trials
                    if t.state == optuna.trial.TrialState.COMPLETE
                ]
                pruned_trials = [
                    t
                    for t in study_rf.trials
                    if t.state == optuna.trial.TrialState.PRUNED
                ]
                failed_trials = [
                    t
                    for t in study_rf.trials
                    if t.state == optuna.trial.TrialState.FAIL
                ]

                logger.info(f"Random Forest - Total trials: {len(study_rf.trials)}")
                logger.info(
                    f"Random Forest - Completed trials: {len(completed_trials)}"
                )
                logger.info(f"Random Forest - Pruned trials: {len(pruned_trials)}")
                logger.info(f"Random Forest - Failed trials: {len(failed_trials)}")

                # Initialize pipeline with best parameters
                rf_pipeline = ImbPipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "random_forest",
                            RandomForestClassifier(
                                random_state=42,
                                class_weight="balanced",
                                n_jobs=-1,
                                **best_rf_params,
                            ),
                        ),
                    ]
                )

                # Train and evaluate Random Forest
                metrics_rf, best_rf_model = train_and_evaluate_model(
                    model_name="Random Forest (Tuned)",
                    pipeline=rf_pipeline,
                    X_train=X_train_resampled,
                    y_train=y_train_resampled,
                    X_val=df_val_processed.drop("SepsisLabel", axis=1),
                    y_val=df_val_processed["SepsisLabel"],
                    df_val_original=df_val_original,
                    unique_report_dir=unique_report_dir,
                    logger=logger,
                )

                # Compare F1 scores to determine the best model
                if metrics_rf["F1 Score"] > best_score:
                    best_score = metrics_rf["F1 Score"]
                    best_model_name = "Random Forest (Tuned)"
                    models["Random Forest (Tuned)"] = {
                        "model": best_rf_model,
                        "metrics": metrics_rf,
                    }
                    logger.info(
                        f"New best model: {best_model_name} with F1 Score: {best_score:.4f}"
                    )

                # Save Random Forest hyperparameters
                save_hyperparameters(
                    "random_forest_tuned_params.json",
                    best_rf_params,
                    logger,
                    unique_report_dir,
                )
                del rf_pipeline
                gc.collect()

            except Exception as e:
                logger.error(
                    f"Error during Random Forest optimization or training: {str(e)}",
                    exc_info=True,
                )
                # Optionally, continue to the next model or halt the pipeline
                # For this example, we'll continue
                pass

            # Optimize and train Logistic Regression within a try-except block
            try:
                logger.info(
                    "Optimizing Logistic Regression hyperparameters with Optuna."
                )
                studies["Logistic_Regression_Optimization"].set_user_attr(
                    "model_type", "LogisticRegression"
                )
                studies["Logistic_Regression_Optimization"].set_user_attr(
                    "optimization_metric", "F1 Score"
                )
                studies["Logistic_Regression_Optimization"].set_user_attr(
                    "dataset_size", len(X_train_resampled)
                )
                studies["Logistic_Regression_Optimization"].set_user_attr(
                    "class_distribution", str(dict(y_train_resampled.value_counts()))
                )
                studies["Logistic_Regression_Optimization"].set_user_attr("cv_folds", 3)
                studies["Logistic_Regression_Optimization"].set_user_attr(
                    "timestamp", datetime.datetime.now().isoformat()
                )

                studies["Logistic_Regression_Optimization"].optimize(
                    lr_objective, n_trials=args.lr_trials, n_jobs=parallel_jobs
                )
                best_lr_params = studies["Logistic_Regression_Optimization"].best_params
                logger.info(f"Best Logistic Regression parameters: {best_lr_params}")

                # Log the number of trials and their statuses
                study_lr = studies["Logistic_Regression_Optimization"]
                completed_trials_lr = [
                    t
                    for t in study_lr.trials
                    if t.state == optuna.trial.TrialState.COMPLETE
                ]
                pruned_trials_lr = [
                    t
                    for t in study_lr.trials
                    if t.state == optuna.trial.TrialState.PRUNED
                ]
                failed_trials_lr = [
                    t
                    for t in study_lr.trials
                    if t.state == optuna.trial.TrialState.FAIL
                ]

                logger.info(
                    f"Logistic Regression - Total trials: {len(study_lr.trials)}"
                )
                logger.info(
                    f"Logistic Regression - Completed trials: {len(completed_trials_lr)}"
                )
                logger.info(
                    f"Logistic Regression - Pruned trials: {len(pruned_trials_lr)}"
                )
                logger.info(
                    f"Logistic Regression - Failed trials: {len(failed_trials_lr)}"
                )

                # Post-Processing to Ensure Solver Compatibility
                if best_lr_params.get("penalty") == "elasticnet":
                    best_lr_params["solver"] = "saga"
                else:
                    # You can choose to set another solver like 'lbfgs' or keep 'saga'
                    best_lr_params["solver"] = "lbfgs"  # or 'saga'

                # Initialize pipeline with best parameters
                lr_pipeline = ImbPipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "logistic_regression",
                            LogisticRegression(
                                random_state=42, n_jobs=-1, **best_lr_params
                            ),
                        ),
                    ]
                )

                # Train and evaluate Logistic Regression
                metrics_lr, best_lr_model = train_and_evaluate_model(
                    model_name="Logistic Regression (Tuned)",
                    pipeline=lr_pipeline,
                    X_train=X_train_resampled,
                    y_train=y_train_resampled,
                    X_val=df_val_processed.drop("SepsisLabel", axis=1),
                    y_val=df_val_processed["SepsisLabel"],
                    df_val_original=df_val_original,
                    unique_report_dir=unique_report_dir,
                    logger=logger,
                )

                # Compare F1 scores to determine the best model
                if metrics_lr["F1 Score"] > best_score:
                    best_score = metrics_lr["F1 Score"]
                    best_model_name = "Logistic Regression (Tuned)"
                    models["Logistic Regression (Tuned)"] = {
                        "model": best_lr_model,
                        "metrics": metrics_lr,
                    }
                    logger.info(
                        f"New best model: {best_model_name} with F1 Score: {best_score:.4f}"
                    )

                # Save Logistic Regression hyperparameters
                save_hyperparameters(
                    "logistic_regression_tuned_params.json",
                    best_lr_params,
                    logger,
                    unique_report_dir,
                )
                del lr_pipeline
                gc.collect()

            except Exception as e:
                logger.error(
                    f"Error during Logistic Regression optimization or training: {str(e)}",
                    exc_info=True,
                )
                # Optionally, continue to the next model or halt the pipeline
                # For this example, we'll continue
                pass

            # Optimize and train XGBoost within a try-except block
            try:
                logger.info("Optimizing XGBoost hyperparameters with Optuna.")
                studies["XGBoost_Optimization"].set_user_attr("model_type", "XGBoost")
                studies["XGBoost_Optimization"].set_user_attr(
                    "optimization_metric", "F1 Score"
                )
                studies["XGBoost_Optimization"].set_user_attr(
                    "dataset_size", len(X_train_resampled)
                )
                studies["XGBoost_Optimization"].set_user_attr(
                    "class_distribution", str(dict(y_train_resampled.value_counts()))
                )
                studies["XGBoost_Optimization"].set_user_attr("cv_folds", 3)
                studies["XGBoost_Optimization"].set_user_attr(
                    "timestamp", datetime.datetime.now().isoformat()
                )

                # Suggestion: Add early stopping
                studies["XGBoost_Optimization"].optimize(
                    xgb_objective,
                    n_trials=args.xgb_trials,
                    n_jobs=parallel_jobs,
                    callbacks=[
                        optuna.callbacks.EarlyStopping(patience=10, min_delta=0.001)
                    ],
                )
                best_xgb_params = studies["XGBoost_Optimization"].best_params
                logger.info(f"Best XGBoost parameters: {best_xgb_params}")

                # Log the number of trials and their statuses
                study_xgb = studies["XGBoost_Optimization"]
                completed_trials_xgb = [
                    t
                    for t in study_xgb.trials
                    if t.state == optuna.trial.TrialState.COMPLETE
                ]
                pruned_trials_xgb = [
                    t
                    for t in study_xgb.trials
                    if t.state == optuna.trial.TrialState.PRUNED
                ]
                failed_trials_xgb = [
                    t
                    for t in study_xgb.trials
                    if t.state == optuna.trial.TrialState.FAIL
                ]

                logger.info(f"XGBoost - Total trials: {len(study_xgb.trials)}")
                logger.info(f"XGBoost - Completed trials: {len(completed_trials_xgb)}")
                logger.info(f"XGBoost - Pruned trials: {len(pruned_trials_xgb)}")
                logger.info(f"XGBoost - Failed trials: {len(failed_trials_xgb)}")

                # Initialize pipeline with best parameters
                xgb_pipeline = ImbPipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "xgb_classifier",
                            XGBClassifier(
                                random_state=42,
                                use_label_encoder=False,
                                eval_metric="logloss",
                                n_jobs=-1,
                                **best_xgb_params,
                            ),
                        ),
                    ]
                )

                # Train and evaluate XGBoost
                metrics_xgb, best_xgb_model = train_and_evaluate_model(
                    model_name="XGBoost (Tuned)",
                    pipeline=xgb_pipeline,
                    X_train=X_train_resampled,
                    y_train=y_train_resampled,
                    X_val=df_val_processed.drop("SepsisLabel", axis=1),
                    y_val=df_val_processed["SepsisLabel"],
                    df_val_original=df_val_original,
                    unique_report_dir=unique_report_dir,
                    logger=logger,
                )

                # Compare F1 scores to determine the best model
                if metrics_xgb["F1 Score"] > best_score:
                    best_score = metrics_xgb["F1 Score"]
                    best_model_name = "XGBoost (Tuned)"
                    models["XGBoost (Tuned)"] = {
                        "model": best_xgb_model,
                        "metrics": metrics_xgb,
                    }
                    logger.info(
                        f"New best model: {best_model_name} with F1 Score: {best_score:.4f}"
                    )

                # Save XGBoost hyperparameters
                save_hyperparameters(
                    "xgboost_tuned_params.json",
                    best_xgb_params,
                    logger,
                    unique_report_dir,
                )
                del xgb_pipeline
                gc.collect()

            except Exception as e:
                logger.error(
                    f"Error during XGBoost optimization or training: {str(e)}",
                    exc_info=True,
                )
                # Optionally, continue to the next process or halt the pipeline
                # For this example, we'll continue
                pass

            # Save all trained models to checkpoint
            joblib.dump(models, checkpoints["trained_models"])
            logger.info(f"Saved trained models to {checkpoints['trained_models']}")

            # Save updated Optuna studies
            joblib.dump(studies, checkpoints["optuna_studies"])
            logger.info(
                f"Saved updated Optuna studies to {checkpoints['optuna_studies']}"
            )

            # Export trial data for all studies
            for study_name, study in studies.items():
                trial_data = []
                for trial in study.trials:
                    if trial.state == optuna.trial.TrialState.COMPLETE:
                        data = {
                            "number": trial.number,
                            "value": trial.value,  # This is the F1 score
                            "f1_score": trial.user_attrs.get("f1_score", None),
                            "accuracy": trial.user_attrs.get("accuracy", None),
                            "roc_auc": trial.user_attrs.get("roc_auc", None),
                            "precision": trial.user_attrs.get("precision", None),
                            "recall": trial.user_attrs.get("recall", None),
                            "f1_std": trial.user_attrs.get("f1_std", None),
                            "cv_iteration": trial.user_attrs.get("cv_iteration", None),
                            "params": trial.params,
                        }
                        
                        trial_data.append(data)

                # Convert to DataFrame for easy analysis
                trials_df = pd.DataFrame(trial_data)

                # Save comprehensive results
                trials_csv_path = os.path.join(
                    unique_report_dir, f"{study_name.lower()}_optimization_trials.csv"
                )
                trials_df.to_csv(trials_csv_path, index=False)
                logger.info(f"Saved trial data to {trials_csv_path}")

            # Step 4: Final Evaluation on Test Set
            if best_model_name:
                try:
                    logger.info(
                        f"\nPerforming final evaluation with best model: {best_model_name}"
                    )
                    best_model_pipeline = models[best_model_name]["model"]

                    # Use the pipeline directly for predictions
                    if hasattr(best_model_pipeline, "predict_proba"):
                        final_predictions_proba = best_model_pipeline.predict_proba(
                            df_test_processed.drop("SepsisLabel", axis=1)
                        )[:, 1]
                        final_predictions = (final_predictions_proba > 0.5).astype(int)
                        logger.info(
                            f"Predictions made with {best_model_name} on test set."
                        )
                    else:
                        final_predictions = best_model_pipeline.predict(
                            df_test_processed.drop("SepsisLabel", axis=1)
                        )
                        final_predictions_proba = None
                        logger.info(
                            f"Predictions made with {best_model_name} on test set."
                        )

                    # Convert predictions to pandas.Series
                    final_predictions = pd.Series(
                        final_predictions,
                        index=df_test_processed.index,
                        name="Predicted",
                    )
                    final_predictions_proba = (
                        pd.Series(
                            final_predictions_proba,
                            index=df_test_processed.index,
                            name="Predicted_Prob",
                        )
                        if final_predictions_proba is not None
                        else None
                    )

                    # Merge Patient_ID back for final evaluation
                    df_test_with_predictions = (
                        df_test_original.copy()
                    )  # Use original df_test
                    df_test_with_predictions[final_predictions.name] = final_predictions
                    if final_predictions_proba is not None:
                        df_test_with_predictions[final_predictions_proba.name] = (
                            final_predictions_proba
                        )

                    # Evaluate the best model on the test set
                    metrics = evaluate_model(
                        y_true=df_test_processed["SepsisLabel"],
                        y_pred=final_predictions,
                        data=df_test_with_predictions,  # Use df_test_with_predictions
                        model_name=f"Final_{best_model_name.replace(' ', '_').lower()}",
                        y_pred_proba=final_predictions_proba,
                        model=best_model_pipeline,  # Use the pipeline directly
                        report_dir=unique_report_dir,
                        logger=logger,
                    )
                    logger.info(f"Final evaluation completed for {best_model_name}.")

                    # Generate model card
                    generate_model_card(
                        model=best_model_pipeline,
                        model_name=best_model_name,
                        metrics=metrics,
                        train_data=df_train_processed,
                        val_data=df_val_processed,
                        test_data=df_test_processed,
                        report_dir=unique_report_dir,
                        run_id=run_id,
                        logger=logger,
                    )

                    # Save the best model using ModelRegistry
                    logger.info(f"Saving the best model ({best_model_name})")
                    model_registry.save_model(
                        model=best_model_pipeline,  # Save the entire pipeline
                        name=best_model_name,
                        params=best_model_pipeline.get_params(),
                        metrics=metrics,
                        artifacts={
                            "confusion_matrix": os.path.join(
                                unique_report_dir,
                                f"{best_model_name}_confusion_matrix.png",
                            ),
                            "roc_curve": os.path.join(
                                unique_report_dir, f"{best_model_name}_roc_curve.png"
                            ),
                            "precision_recall_curve": os.path.join(
                                unique_report_dir,
                                f"{best_model_name}_precision_recall_curve.png",
                            ),
                            "feature_importance": os.path.join(
                                unique_report_dir,
                                f"{best_model_name}_feature_importance.png",
                            ),
                            "missing_values_heatmap": os.path.join(
                                unique_report_dir,
                                f"{best_model_name}_missing_values_heatmap.png",
                            ),
                        },
                        tags=["tuned"],
                    )
                    logger.info(f"Model ({best_model_name}) saved successfully.")

                    # Clean up
                    del best_model_pipeline, final_predictions, final_predictions_proba
                    gc.collect()

                    logger.info("Sepsis Prediction Pipeline completed successfully.")

                except Exception as e:
                    logger.error(
                        f"An error occurred during final evaluation: {str(e)}",
                        exc_info=True,
                    )
                    raise

            else:
                logger.warning(
                    "No models were trained. Pipeline did not complete successfully."
                )

        else:
            logger.info("Skipping optimizations and training as checkpoints exist.")

        logger.info("Sepsis Prediction Pipeline completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

    finally:
        # Record end time and final memory usage
        end_time = time.time()
        duration = end_time - start_time
        final_memory = process.memory_info().rss / (1024**2)  # Convert to MB
        memory_change = final_memory - initial_memory

        logger.info("Phase Complete: Main")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Final Memory Usage: {final_memory:.2f} MB")
        logger.info(
            f"Memory Change: {'+' if memory_change >=0 else '-'}{abs(memory_change):.2f} MB"
        )


if __name__ == "__main__":
    main()
```

## src/__init__.py

```python
# src/__init__.py

from .data_processing import (
    load_data,
    load_processed_data,
    split_data,
)
from .evaluation import (
    evaluate_model,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_feature_correlation_heatmap,
    plot_feature_importance,
    plot_precision_recall_curve,
    plot_roc_curve,
    generate_evaluation_plots,
    plot_missing_values,
    plot_temporal_progression,
    plot_error_analysis,
    plot_calibration,
    plot_prediction_timeline,
    plot_feature_interactions,
)
from .feature_engineering import preprocess_data

from .utils import (
    log_message,
    log_metrics,
    save_metrics_to_json,
)

from .logger_utils import (
    log_phase,
    log_memory,
    log_dataframe_info,
    log_step,
    log_function,
)
from .logger_config import get_logger, disable_duplicate_logging

from .model_registry import ModelRegistry

__all__ = [
    # Data Processing
    "load_data",
    "load_processed_data",
    "split_data",
    "preprocess_data",
    # Evaluation and Plotting
    "evaluate_model",
    "generate_evaluation_plots",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_precision_recall_curve",
    "plot_feature_importance",
    "plot_class_distribution",
    "plot_feature_correlation_heatmap",
    "plot_missing_values",  # New
    "plot_temporal_progression",  # New
    "plot_error_analysis",  # New
    "plot_calibration",  # New
    "plot_prediction_timeline",  # New
    "plot_feature_interactions",  # New
    # Logging and Utilities
    "log_message",
    "log_metrics",
    "save_metrics_to_json",
    "log_phase",
    "log_memory",
    "log_dataframe_info",
    "log_step",
    "log_function",
    "get_logger",
    "disable_duplicate_logging",
    "ModelRegistry",
]
```

## src/data_processing.py

```python
import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file with error handling and validation.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at: {filepath}")

    # Load the data
    df = pd.read_csv(filepath)

    # Validate required columns
    required_columns = ["Patient_ID", "SepsisLabel"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Convert Patient_ID to string - using inplace operation
    df = df.assign(Patient_ID=df["Patient_ID"].astype(str))

    # Basic data validation
    if df["SepsisLabel"].nunique() > 2:
        raise ValueError("SepsisLabel contains more than two unique values")

    logging.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    return df


def split_data(
    combined_df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    random_state: int = 42,
    stratify_by_sepsis: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the combined data into training, validation, and testing datasets.
    """
    # Input validation
    if not 0 < train_size + val_size < 1:
        raise ValueError("train_size + val_size must be between 0 and 1")

    if not isinstance(combined_df, pd.DataFrame):
        raise ValueError("combined_df must be a pandas DataFrame")

    # Ensure Patient_ID is string type before splitting
    combined_df = combined_df.assign(Patient_ID=combined_df["Patient_ID"].astype(str))

    # Get unique Patient_IDs and their characteristics
    patient_stats = (
        combined_df.groupby("Patient_ID")
        .agg(
            {
                "SepsisLabel": "max",  # 1 if patient ever had sepsis
                "Hour": "count",  # number of measurements per patient
            }
        )
        .reset_index()
    )

    # Set random seed for reproducibility
    np.random.seed(random_state)

    if stratify_by_sepsis:
        # Split separately for sepsis and non-sepsis patients
        sepsis_patients = patient_stats[patient_stats["SepsisLabel"] == 1]["Patient_ID"]
        non_sepsis_patients = patient_stats[patient_stats["SepsisLabel"] == 0][
            "Patient_ID"
        ]

        def split_patient_ids(patient_ids):
            n_train = int(len(patient_ids) * train_size)
            n_val = int(len(patient_ids) * val_size)
            shuffled = np.random.permutation(patient_ids)
            return (
                shuffled[:n_train],
                shuffled[n_train : n_train + n_val],
                shuffled[n_train + n_val :],
            )

        # Split both groups
        train_sepsis, val_sepsis, test_sepsis = split_patient_ids(sepsis_patients)
        train_non_sepsis, val_non_sepsis, test_non_sepsis = split_patient_ids(
            non_sepsis_patients
        )

        # Combine splits
        train_patients = np.concatenate([train_sepsis, train_non_sepsis])
        val_patients = np.concatenate([val_sepsis, val_non_sepsis])
        test_patients = np.concatenate([test_sepsis, test_non_sepsis])

    else:
        # Simple random split without stratification
        shuffled_patients = np.random.permutation(patient_stats["Patient_ID"])
        n_train = int(len(shuffled_patients) * train_size)
        n_val = int(len(shuffled_patients) * val_size)

        train_patients = shuffled_patients[:n_train]
        val_patients = shuffled_patients[n_train : n_train + n_val]
        test_patients = shuffled_patients[n_train + n_val :]

    # Create the splits - using copy() to ensure we have independent DataFrames
    df_train = combined_df[combined_df["Patient_ID"].isin(train_patients)].copy()
    df_val = combined_df[combined_df["Patient_ID"].isin(val_patients)].copy()
    df_test = combined_df[combined_df["Patient_ID"].isin(test_patients)].copy()

    # Verify no patient overlap
    assert (
        len(set(train_patients) & set(val_patients)) == 0
    ), "Patient overlap between train and val"
    assert (
        len(set(train_patients) & set(test_patients)) == 0
    ), "Patient overlap between train and test"
    assert (
        len(set(val_patients) & set(test_patients)) == 0
    ), "Patient overlap between val and test"

    # Log split information
    logging.info("\nData Split Summary:")
    logging.info("-" * 50)
    logging.info(
        f"Training set:   {len(df_train)} rows, {len(train_patients)} unique patients"
    )
    logging.info(
        f"Validation set: {len(df_val)} rows, {len(val_patients)} unique patients"
    )
    logging.info(
        f"Testing set:    {len(df_test)} rows, {len(test_patients)} unique patients"
    )

    # Log sepsis distribution
    for name, df in [
        ("Training", df_train),
        ("Validation", df_val),
        ("Testing", df_test),
    ]:
        sepsis_rate = df.groupby("Patient_ID")["SepsisLabel"].max().mean()
        logging.info(f"{name} set sepsis rate: {sepsis_rate:.1%}")

    # Save the split datasets
    os.makedirs("data/processed", exist_ok=True)
    df_train.to_csv("data/processed/train_data.csv", index=False)
    df_val.to_csv("data/processed/val_data.csv", index=False)
    df_test.to_csv("data/processed/test_data.csv", index=False)

    return df_train, df_val, df_test


def load_processed_data(
    train_path: str, val_path: str, test_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load processed training, validation, and testing data with validation.
    """
    # Check file existence
    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found at: {path}")

    # Load datasets
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # Validate columns match across datasets
    if not (set(df_train.columns) == set(df_val.columns) == set(df_test.columns)):
        raise ValueError("Column mismatch between datasets")

    # Convert Patient_ID to string in all datasets using assign
    dfs = [df_train, df_val, df_test]
    for i in range(len(dfs)):
        dfs[i] = dfs[i].assign(Patient_ID=dfs[i]["Patient_ID"].astype(str))
    df_train, df_val, df_test = dfs

    # Verify no patient overlap
    train_patients = set(df_train["Patient_ID"])
    val_patients = set(df_val["Patient_ID"])
    test_patients = set(df_test["Patient_ID"])

    if train_patients & val_patients:
        raise ValueError("Patient overlap between train and validation sets")
    if train_patients & test_patients:
        raise ValueError("Patient overlap between train and test sets")
    if val_patients & test_patients:
        raise ValueError("Patient overlap between validation and test sets")

    return df_train, df_val, df_test


def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validate the dataset format and content.
    """
    # Check required columns
    required_columns = ["Patient_ID", "Hour", "SepsisLabel"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Validate data types
    if not pd.api.types.is_numeric_dtype(df["Hour"]):
        raise ValueError("Hour column must be numeric")
    if not pd.api.types.is_numeric_dtype(df["SepsisLabel"]):
        raise ValueError("SepsisLabel column must be numeric")

    # Validate value ranges
    if df["Hour"].min() < 0:
        raise ValueError("Hour column contains negative values")
    if not set(df["SepsisLabel"].unique()).issubset({0, 1}):
        raise ValueError("SepsisLabel must contain only 0 and 1")

    # Check for duplicates
    duplicates = df.groupby(["Patient_ID", "Hour"]).size()
    if (duplicates > 1).any():
        raise ValueError("Found duplicate time points for some patients")
```

## src/evaluation.py

```python
# evaluation.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from typing import List, Dict, Optional, Any
import logging
from itertools import combinations
import json
import os
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .logger_utils import log_function


@log_function(logging.getLogger(__name__))
def setup_plot_style() -> None:
    """Set up the plotting style for all visualizations."""
    try:
        sns.set_style("whitegrid")
        plt.rcParams.update(
            {
                "figure.figsize": (8, 6),
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
                "font.size": 12,
                "lines.linewidth": 2,
            }
        )
    except Exception as e:
        logging.error(f"Error setting up plot style: {str(e)}")
        raise


def evaluate_model(
    y_true: pd.Series,
    y_pred: pd.Series,
    data: pd.DataFrame,
    model_name: str,
    report_dir: str = "reports/evaluations",
    y_pred_proba: Optional[pd.Series] = None,
    model: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, float]:
    """
    Comprehensive model evaluation function that calculates metrics and generates essential visualization plots.

    Args:
        y_true (pd.Series): True labels.
        y_pred (pd.Series): Predicted labels.
        data (pd.DataFrame): DataFrame containing the data.
        model_name (str): Name of the model.
        report_dir (str): Directory to save evaluation reports and plots.
        y_pred_proba (Optional[pd.Series]): Predicted probabilities.
        model (Optional[Any]): Trained model object.
        logger (Optional[logging.Logger]): Logger object.

    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics.
    """
    try:
        # Calculate basic metrics
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1 Score": f1_score(y_true, y_pred, zero_division=0),
            "Mean Absolute Error": mean_absolute_error(y_true, y_pred),
            "Root Mean Squared Error": np.sqrt(mean_squared_error(y_true, y_pred)),
        }

        # Calculate probability-based metrics if probabilities are available
        if y_pred_proba is not None:
            try:
                metrics["AUC-ROC"] = roc_auc_score(y_true, y_pred_proba)
                metrics["AUPRC"] = average_precision_score(y_true, y_pred_proba)
            except ValueError as e:
                if logger:
                    logger.warning(f"Cannot calculate probability-based metrics: {e}")
                metrics["AUC-ROC"] = np.nan
                metrics["AUPRC"] = np.nan

        # Calculate Specificity using confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["Specificity"] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        else:
            metrics["Specificity"] = np.nan
            if logger:
                logger.warning(
                    f"{model_name} - Specificity calculation is not applicable for non-binary classifications."
                )

        # Log metrics
        log_metrics(logger, model_name, metrics)

        # Generate and save essential plots
        generate_evaluation_plots(
            y_true=y_true,
            y_pred=y_pred,
            data=data,
            y_pred_proba=y_pred_proba,
            model=model,
            model_name=model_name,
            report_dir=report_dir,
            logger=logger,
        )

        # Save metrics to JSON
        save_metrics_to_json(metrics, model_name, report_dir, logger)

        return metrics

    except Exception as e:
        if logger:
            logger.error(f"Error in model evaluation: {str(e)}", exc_info=True)
        raise
    finally:
        plt.close("all")


# def evaluate_model(
#     y_true: pd.Series,
#     y_pred: pd.Series,
#     data: pd.DataFrame,
#     model_name: str,
#     report_dir: str = "reports/evaluations",
#     y_pred_proba: Optional[pd.Series] = None,
#     model: Optional[Any] = None,
#     logger: Optional[logging.Logger] = None,
# ) -> Dict[str, float]:
#     try:
#         # Ensure essential columns exist in data
#         assert "Hour" in data.columns, "Hour column missing in data."
#         assert "SepsisLabel" in data.columns, "SepsisLabel column missing in data."

#         setup_plot_style()
#         os.makedirs(report_dir, exist_ok=True)

#         # Calculate basic metrics
#         metrics = {
#             "Accuracy": accuracy_score(y_true, y_pred),
#             "Precision": precision_score(y_true, y_pred, zero_division=0),
#             "Recall": recall_score(y_true, y_pred, zero_division=0),
#             "F1 Score": f1_score(y_true, y_pred, zero_division=0),
#             "Mean Absolute Error": mean_absolute_error(y_true, y_pred),
#             "Root Mean Squared Error": np.sqrt(mean_squared_error(y_true, y_pred)),
#         }

#         # Calculate AUC-ROC if probabilities are available
#         if y_pred_proba is not None:
#             try:
#                 metrics["AUC-ROC"] = roc_auc_score(y_true, y_pred_proba)
#             except ValueError as e:
#                 if logger:
#                     logger.warning(f"Cannot calculate AUC-ROC: {e}")
#                 metrics["AUC-ROC"] = None

#         # Calculate Specificity
#         cm = confusion_matrix(y_true, y_pred)
#         if cm.shape == (2, 2):
#             tn, fp, fn, tp = cm.ravel()
#             metrics["Specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
#         else:
#             metrics["Specificity"] = 0.0  # Handle non-binary classifications
#             if logger:
#                 logger.warning(
#                     f"{model_name} - Specificity calculation is not applicable for non-binary classifications."
#                 )

#         # Log metrics
#         log_metrics(logger, model_name, metrics)

#         # Generate and save essential plots
#         generate_evaluation_plots(
#             y_true=y_true,
#             y_pred=y_pred,
#             data=data,  # Pass the entire DataFrame
#             y_pred_proba=y_pred_proba,
#             model=model,
#             model_name=model_name,
#             report_dir=report_dir,
#             logger=logger,
#         )

#         # Save metrics to JSON
#         save_metrics_to_json(metrics, model_name, report_dir, logger)

#         return metrics

#     except AssertionError as ae:
#         if logger:
#             logger.error(f"Assertion Error: {ae}", exc_info=True)
#         raise
#     except Exception as e:
#         if logger:
#             logger.error(f"Error in model evaluation: {str(e)}", exc_info=True)
#         raise
#     finally:
#         plt.close("all")


def generate_evaluation_plots(
    y_true: pd.Series,
    y_pred: pd.Series,
    data: pd.DataFrame,  # Added data parameter
    y_pred_proba: Optional[pd.Series],
    model: Any,
    model_name: str,
    report_dir: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Generate comprehensive evaluation plots."""
    try:
        # 1. Base plots (existing)
        plot_confusion_matrix(
            y_true, y_pred, model_name, report_dir, normalize=False, logger=logger
        )
        plot_confusion_matrix(
            y_true, y_pred, model_name, report_dir, normalize=True, logger=logger
        )

        # 2. ROC and Precision-Recall curves (if probabilities available)
        if y_pred_proba is not None:
            plot_roc_curve(y_true, y_pred_proba, model_name, report_dir, logger=logger)
            plot_precision_recall_curve(
                y_true, y_pred_proba, model_name, report_dir, logger=logger
            )
            plot_calibration(
                y_true, y_pred_proba, report_dir, model_name, logger=logger
            )

        # 3. Feature importance (if model available)
        if model is not None:
            plot_feature_importance(model, model_name, report_dir, logger=logger)

        # 4. Data quality plots
        plot_missing_values(data, report_dir, model_name=model_name, logger=logger)

        # 5. Temporal analysis plots
        vital_features = ["HR", "O2Sat", "Temp", "MAP", "Resp"]

        # Extract Patient_IDs, assuming same index as y_true
        patient_ids = data["Patient_ID"]

        plot_temporal_progression(
            data,
            patient_ids,  # Pass patient_ids
            vital_features,
            report_dir,
            model_name=model_name,
            max_patients=5,
            logger=logger,
        )

        # 6. Error analysis
        # Define patient groups based on available features
        patient_groups = {}
        if "Age" in data.columns:
            patient_groups.update(
                {"Young": data["Age"] < 50, "Elderly": data["Age"] >= 50}
            )
        if "Gender_1" in data.columns:  # Assuming one-hot encoded
            patient_groups.update(
                {"Male": data["Gender_1"] == 1, "Female": data["Gender_1"] == 0}
            )

        if patient_groups:  # Only plot if we have groups defined
            plot_error_analysis(
                y_true,
                y_pred,
                patient_groups,
                report_dir,
                model_name=model_name,
                logger=logger,
            )

        # 7. Feature interactions
        numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
        # Remove target variable if present
        if "SepsisLabel" in numeric_features:
            numeric_features.remove("SepsisLabel")

        plot_feature_interactions(
            data,
            numeric_features,
            report_dir,
            model_name=model_name,
            top_n=5,
            logger=logger,
        )

        # 8. Prediction timeline
        plot_prediction_timeline(
            data,
            y_pred,
            report_dir,
            model_name=model_name,
            max_patients=5,
            logger=logger,
        )

    except Exception as e:
        if logger:
            logger.error(f"Error generating evaluation plots: {e}", exc_info=True)
        raise
    finally:
        plt.close("all")


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: pd.Series,
    model_name: str,
    report_dir: str,
    normalize: bool = False,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Plot and save confusion matrix."""
    try:
        cm = confusion_matrix(y_true, y_pred, normalize="true" if normalize else None)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(
            f"Confusion Matrix for {model_name} {'(Normalized)' if normalize else ''}"
        )
        plt.tight_layout()

        cm_type = "confusion_matrix_normalized" if normalize else "confusion_matrix"
        save_plot(plt, report_dir, f"{model_name}_{cm_type}.png", logger)

    except Exception as e:
        if logger:
            logger.error(f"Error plotting confusion matrix: {e}", exc_info=True)
        raise
    finally:
        plt.close()


def plot_class_distribution(
    y: pd.Series,
    model_name: str,
    report_dir: str,
    title_suffix: str = "",
    logger: Optional[logging.Logger] = None,
) -> None:
    """Plot and save the class distribution without redundant hue."""
    try:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=y, color="blue")
        plt.title(f"Class Distribution for {model_name} {title_suffix}")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.tight_layout()

        suffix = title_suffix.replace(" ", "_") if title_suffix else ""
        filename = f"{model_name.replace(' ', '_')}_class_distribution_{suffix}.png"
        save_plot(plt, report_dir, filename, logger)

    except Exception as e:
        if logger:
            logger.error(f"Error plotting class distribution: {e}", exc_info=True)
        raise
    finally:
        plt.close()


def plot_feature_importance(
    model: Any,
    model_name: str,
    report_dir: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Plot and save feature importance, or log model attributes if not available."""
    try:
        plt.figure(figsize=(10, 8))

        # Log model attributes for debugging
        if logger:
            logger.debug(f"Model attributes for {model_name}:")
            for attr in dir(model):
                if not attr.startswith("_"):  # Exclude private attributes
                    logger.debug(f"  - {attr}")

        if hasattr(model, "feature_importances_"):
            # Handle tree-based models (Random Forest, XGBoost)
            importances = model.feature_importances_
            feature_names = (
                model.feature_names_in_
                if hasattr(model, "feature_names_in_")
                else [f"Feature {i}" for i in range(len(importances))]
            )

            feature_importance_df = (
                pd.DataFrame({"feature": feature_names, "importance": importances})
                .sort_values(by="importance", ascending=False)
                .head(20)
            )

            sns.barplot(
                x="importance",
                y="feature",
                data=feature_importance_df,
                color="skyblue",
            )
            plt.title(f"Top 20 Feature Importances for {model_name}")

        elif hasattr(model, "coef_"):
            # Handle Logistic Regression
            importances = np.abs(model.coef_[0])  # Use absolute coefficients
            feature_names = (
                model.feature_names_in_
                if hasattr(model, "feature_names_in_")
                else [f"Feature {i}" for i in range(len(importances))]
            )

            feature_importance_df = (
                pd.DataFrame({"feature": feature_names, "importance": importances})
                .sort_values(by="importance", ascending=False)
                .head(20)
            )

            sns.barplot(
                x="importance",
                y="feature",
                data=feature_importance_df,
                color="skyblue",
            )
            plt.title(
                f"Top 20 Feature Importances for {model_name} (Absolute Coefficients)"
            )

        else:
            if logger:
                logger.warning(
                    f"Model type {type(model)} does not support feature importance plotting."
                )
            return

        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()

        filename = f"{model_name.replace(' ', '_')}_feature_importance.png"
        save_plot(plt, report_dir, filename, logger)

    except Exception as e:
        if logger:
            logger.error(f"Error plotting feature importance: {e}", exc_info=True)
        raise
    finally:
        plt.close()


def plot_roc_curve(
    y_true: pd.Series,
    y_pred_proba: pd.Series,
    model_name: str,
    report_dir: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Plot and save ROC curve."""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve for {model_name}")
        plt.legend(loc="lower right")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        filename = f"{model_name.replace(' ', '_')}_roc_curve.png"
        save_plot(plt, report_dir, filename, logger)

    except Exception as e:
        if logger:
            logger.error(f"Error plotting ROC curve: {e}", exc_info=True)
        raise
    finally:
        plt.close()


def plot_precision_recall_curve(
    y_true: pd.Series,
    y_pred_proba: pd.Series,
    model_name: str,
    report_dir: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Plot and save the Precision-Recall curve."""
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        average_precision = average_precision_score(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(
            recall,
            precision,
            color="purple",
            lw=2,
            label=f"PR curve (AP = {average_precision:.2f})",
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve for {model_name}")
        plt.legend(loc="lower left")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        filename = f"{model_name.replace(' ', '_')}_precision_recall_curve.png"
        save_plot(plt, report_dir, filename, logger)

    except Exception as e:
        if logger:
            logger.error(f"Error plotting precision-recall curve: {e}", exc_info=True)
        raise
    finally:
        plt.close()


def plot_feature_correlation_heatmap(
    df: pd.DataFrame,
    model_name: str,
    report_dir: str,
    top_n: int = 20,
    threshold: float = 0.6,  # Made threshold a parameter with default value
    logger: Optional[logging.Logger] = None,
) -> None:
    """Plot and save the feature correlation heatmap with dynamic threshold."""
    try:
        plt.figure(figsize=(12, 10))
        corr = df.corr().abs()
        # Select upper triangle
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        # Find features with correlation greater than the threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        # Compute correlation matrix of selected features
        corr_selected = corr.drop(columns=to_drop).drop(index=to_drop)

        sns.heatmap(
            corr_selected,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            linewidths=0.5,
            linecolor="grey",
        )
        plt.title(f"Feature Correlation Heatmap for {model_name}")
        plt.xlabel("Features")
        plt.ylabel("Features")
        plt.tight_layout()

        filename = f"{model_name.replace(' ', '_')}_feature_correlation_heatmap.png"
        save_plot(plt, report_dir, filename, logger)

    except Exception as e:
        if logger:
            logger.error(
                f"Error plotting feature correlation heatmap: {e}", exc_info=True
            )
        raise
    finally:
        plt.close()


def plot_missing_values(
    data: pd.DataFrame, report_dir: str, model_name: str, logger: logging.Logger
):
    """
    Plots a heatmap of missing values in the dataset with annotated missing percentages.

    Args:
        data (pd.DataFrame): The input DataFrame.
        report_dir (str): Directory to save the plot.
        model_name (str): Name of the model (used in the plot title and filename).
        logger (logging.Logger): Logger object for logging information and errors.
    """
    try:
        # Calculate missing percentage per column
        missing_percentage = data.isnull().mean() * 100

        # Set up the matplotlib figure
        plt.figure(figsize=(12, 8))

        # Create a boolean mask of missing values
        sns.heatmap(data.isnull(), cbar=False, yticklabels=False, cmap="viridis")
        plt.title(f"Missing Values Heatmap for {model_name}")

        # Annotate missing percentages on the heatmap
        # Adjust y-coordinate based on the number of rows in the heatmap
        # Assuming one row per data sample; for large datasets, adjust accordingly
        # Alternatively, skip annotation for large datasets to avoid clutter
        if len(data) < 1000:  # Example threshold
            for idx, column in enumerate(data.columns):
                # Extract scalar value using .item()
                perc = missing_percentage[column].item()
                plt.text(
                    idx + 0.5,  # x-coordinate (center of the column)
                    0.5,  # y-coordinate (top of the heatmap)
                    f"{perc:.1f}%",  # formatted string
                    ha="center",
                    va="center",
                    color="red",
                    fontsize=8,
                    fontweight="bold",
                )
        else:
            logger.warning(
                f"{model_name} - Dataset too large for missing values annotation on heatmap."
            )

        # Save the heatmap
        heatmap_path = os.path.join(
            report_dir, f"{model_name}_missing_values_heatmap.png"
        )
        plt.savefig(heatmap_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Missing values heatmap saved to {heatmap_path}")

    except Exception as e:
        logger.error(f"Error plotting missing values heatmap: {e}", exc_info=True)
        raise


def plot_temporal_progression(
    data: pd.DataFrame,
    patient_ids: pd.Series,  # Add patient_ids parameter
    features: List[str],
    report_dir: str,
    model_name: str = "model",
    max_patients: int = 5,
    window_size: int = 5,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Plot temporal progression of vital signs for selected patients with sorted data.
    """
    try:
        # Select random sepsis patients using provided patient_ids
        sepsis_patients = patient_ids[data["SepsisLabel"] == 1].unique()
        if len(sepsis_patients) == 0:
            logger.warning("No sepsis patients found for temporal progression plots.")
            return

        selected_patients = np.random.choice(
            sepsis_patients, size=min(max_patients, len(sepsis_patients)), replace=False
        )

        for patient_id in selected_patients:
            plt.figure(figsize=(15, 8))
            patient_mask = patient_ids == patient_id
            patient_data = data[patient_mask].copy()
            patient_data = patient_data.sort_values(by="Hour")

            # Apply rolling mean smoothing
            for feature in features:
                if feature not in patient_data.columns:
                    logger.warning(f"Feature '{feature}' not found in patient data.")
                    continue
                smoothed_values = (
                    patient_data[feature]
                    .rolling(window=window_size, center=True)
                    .mean()
                )
                plt.plot(patient_data["Hour"], smoothed_values, label=feature)

            # Mark sepsis onset
            sepsis_onset_series = patient_data[patient_data["SepsisLabel"] == 1]["Hour"]
            sepsis_onset = (
                sepsis_onset_series.iloc[0] if not sepsis_onset_series.empty else None
            )
            if sepsis_onset is not None:
                plt.axvline(
                    x=sepsis_onset, color="r", linestyle="--", label="Sepsis Onset"
                )

            plt.title(f"Temporal Progression - Patient {patient_id}")
            plt.xlabel("Hours")
            plt.ylabel("Normalized Values")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.grid(True, linestyle="--", alpha=0.5)

            plt.tight_layout()
            filename = (
                f"{model_name.replace(' ', '_')}_temporal_patient_{patient_id}.png"
            )
            save_plot(plt, report_dir, filename, logger)

        if logger:
            logger.info(
                f"Saved temporal progression plots for {len(selected_patients)} patients"
            )

    except Exception as e:
        if logger:
            logger.error(
                f"Error plotting temporal progression: {str(e)}", exc_info=True
            )
        raise
    finally:
        plt.close("all")


def plot_error_analysis(
    y_true: pd.Series,
    y_pred: pd.Series,
    patient_groups: Dict[str, pd.Series],
    report_dir: str,
    model_name: str = "model",
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Create error analysis plots broken down by patient groups with normalized metrics.
    """
    try:
        metrics_by_group = {}

        # Calculate metrics for each group
        for group_name, group_mask in patient_groups.items():
            if len(y_true[group_mask]) > 0:  # Check if group has samples
                cm = confusion_matrix(y_true[group_mask], y_pred[group_mask])
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    accuracy = (
                        (tp + tn) / (tp + tn + fp + fn)
                        if (tp + tn + fp + fn) > 0
                        else 0
                    )
                    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
                    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
                else:
                    accuracy = 0.0
                    false_positive_rate = 0.0
                    false_negative_rate = 0.0
                    if logger:
                        logger.warning(
                            f"{model_name} - Confusion matrix for group '{group_name}' is not binary."
                        )

                metrics_by_group[group_name] = {
                    "Accuracy (%)": accuracy * 100,
                    "False Positive Rate (%)": false_positive_rate * 100,
                    "False Negative Rate (%)": false_negative_rate * 100,
                    "Sample Size": np.sum(group_mask),
                }

        # Create visualization
        metrics_df = pd.DataFrame(metrics_by_group).T

        # Plot multiple metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Error Analysis by Patient Groups for {model_name}", fontsize=16)

        # Accuracy
        sns.barplot(
            x=metrics_df.index,
            y=metrics_df["Accuracy (%)"],
            ax=axes[0, 0],
            color="skyblue",  # Use a single color if hue is not used
        )
        axes[0, 0].set_title("Accuracy by Group")
        axes[0, 0].set_ylim(0, 100)
        axes[0, 0].set_ylabel("Accuracy (%)")
        axes[0, 0].set_xlabel("")
        axes[0, 0].set_xticks(range(len(metrics_df.index)))
        axes[0, 0].set_xticklabels(metrics_df.index, rotation=45)

        # False Positive Rate
        sns.barplot(
            x=metrics_df.index,
            y=metrics_df["False Positive Rate (%)"],
            ax=axes[0, 1],
            color="salmon",
        )
        axes[0, 1].set_title("False Positive Rate by Group")
        axes[0, 1].set_ylim(0, 100)
        axes[0, 1].set_ylabel("False Positive Rate (%)")
        axes[0, 1].set_xlabel("")
        axes[0, 1].set_xticks(range(len(metrics_df.index)))
        axes[0, 1].set_xticklabels(metrics_df.index, rotation=45)

        # False Negative Rate
        sns.barplot(
            x=metrics_df.index,
            y=metrics_df["False Negative Rate (%)"],
            ax=axes[1, 0],
            color="lightgreen",
        )
        axes[1, 0].set_title("False Negative Rate by Group")
        axes[1, 0].set_ylim(0, 100)
        axes[1, 0].set_ylabel("False Negative Rate (%)")
        axes[1, 0].set_xlabel("")
        axes[1, 0].set_xticks(range(len(metrics_df.index)))
        axes[1, 0].set_xticklabels(metrics_df.index, rotation=45)

        # Sample Size
        sns.barplot(
            x=metrics_df.index,
            y=metrics_df["Sample Size"],
            ax=axes[1, 1],
            color="plum",
        )
        axes[1, 1].set_title("Sample Size by Group")
        axes[1, 1].set_ylabel("Sample Size")
        axes[1, 1].set_xlabel("")
        axes[1, 1].set_xticks(range(len(metrics_df.index)))
        axes[1, 1].set_xticklabels(metrics_df.index, rotation=45)

        plt.tight_layout(
            rect=[0, 0.03, 1, 0.95]
        )  # Adjust layout to accommodate suptitle
        filename = f"{model_name.replace(' ', '_')}_error_analysis.png"
        save_plot(plt, report_dir, filename, logger)

        if logger:
            logger.info(f"Saved error analysis plots to {report_dir}")

    except Exception as e:
        if logger:
            logger.error(f"Error plotting error analysis: {e}", exc_info=True)
        raise
    finally:
        plt.close("all")


def plot_calibration(
    y_true: pd.Series,
    y_pred_proba: pd.Series,
    report_dir: str,
    model_name: str = "model",
    n_bins: int = 10,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Create calibration plot comparing predicted probabilities to observed frequencies with enhanced readability.
    """
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)

        plt.figure(figsize=(8, 8))

        # Plot perfectly calibrated line
        plt.plot(
            [0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated"
        )

        # Plot model calibration
        plt.plot(
            prob_pred, prob_true, marker="o", color="blue", label="Model Calibration"
        )

        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Observed Frequency")
        plt.title(f"Calibration Plot for {model_name}")
        plt.legend(loc="lower right")
        plt.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        filename = f"{model_name.replace(' ', '_')}_calibration.png"
        save_plot(plt, report_dir, filename, logger)

    except Exception as e:
        if logger:
            logger.error(f"Error plotting calibration curve: {e}", exc_info=True)
        raise
    finally:
        plt.close()


def plot_prediction_timeline(
    data: pd.DataFrame,
    y_pred: pd.Series,
    report_dir: str,
    model_name: str = "model",
    max_patients: int = 5,
    threshold: float = 0.5,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Create timeline comparing predicted vs actual sepsis onset with sorted data and threshold parameter.
    """
    try:
        # Ensure essential columns exist
        assert "Hour" in data.columns, "Hour column missing in data."
        assert "SepsisLabel" in data.columns, "SepsisLabel column missing in data."

        # Select sepsis patients
        sepsis_patients = data[data["SepsisLabel"] == 1]["Patient_ID"].unique()
        if len(sepsis_patients) == 0:
            logger.warning("No sepsis patients found for prediction timeline plots.")
            return

        # Select patients who have predictions
        sepsis_patients_with_preds = [
            pid for pid in sepsis_patients if pid in y_pred.index
        ]
        if not sepsis_patients_with_preds:
            logger.warning("No sepsis patients have predictions for timeline plots.")
            return

        # Ensure max_patients does not exceed available patients
        selected_patients = np.random.choice(
            sepsis_patients_with_preds,
            size=min(max_patients, len(sepsis_patients_with_preds)),
            replace=False,
        )

        for patient_id in selected_patients:
            plt.figure(figsize=(15, 5))

            patient_mask = data["Patient_ID"] == patient_id
            patient_data = data[patient_mask].copy()
            patient_data = patient_data.sort_values(by="Hour")  # Ensure sorted

            # Retrieve predictions for the patient
            if patient_id in y_pred.index:
                patient_preds = y_pred.loc[patient_id]
            else:
                patient_preds = pd.Series()

            # Log the start of plotting for the patient
            if patient_preds.empty:
                logger.warning(
                    f"No predictions found for patient {patient_id}. Skipping plot."
                )
                plt.close()
                continue
            else:
                logger.info(f"Plotting prediction timeline for patient {patient_id}.")

            # Ensure y_pred is aligned with patient_data
            # Assuming y_pred index aligns with data index
            patient_preds_aligned = (
                y_pred.loc[patient_data.index]
                if isinstance(y_pred, pd.Series)
                else y_pred
            )

            if len(patient_preds_aligned) != len(patient_data):
                logger.warning(
                    f"Mismatch in data and predictions lengths for patient {patient_id}. Skipping plot."
                )
                plt.close()
                continue

            # Plot actual sepsis label
            plt.plot(
                patient_data["Hour"],
                patient_data["SepsisLabel"],
                label="Actual",
                color="black",
                linewidth=2,
            )

            # Plot predictions
            plt.plot(
                patient_data["Hour"],
                patient_preds_aligned,
                label="Predicted",
                color="blue",
                alpha=0.7,
            )

            # Mark actual sepsis onset
            sepsis_onset_series = patient_data[patient_data["SepsisLabel"] == 1]["Hour"]
            sepsis_onset = (
                sepsis_onset_series.iloc[0] if not sepsis_onset_series.empty else None
            )
            if sepsis_onset is not None:
                plt.axvline(
                    x=sepsis_onset, color="red", linestyle="--", label="Actual Onset"
                )

            # Mark predicted onset (first prediction above threshold)
            pred_onset_indices = np.where(patient_preds_aligned > threshold)[0]
            if len(pred_onset_indices) > 0:
                pred_onset = patient_data["Hour"].iloc[pred_onset_indices[0]]
                plt.axvline(
                    x=pred_onset, color="green", linestyle="--", label="Predicted Onset"
                )

            plt.title(f"Prediction Timeline - Patient {patient_id}")
            plt.xlabel("Hours")
            plt.ylabel("Sepsis Probability")
            plt.legend(loc="upper left")
            plt.grid(True, linestyle="--", alpha=0.5)

            plt.tight_layout()
            filename = (
                f"{model_name.replace(' ', '_')}_timeline_patient_{patient_id}.png"
            )
            save_plot(plt, report_dir, filename, logger)
            logger.info(f"Saved prediction timeline plot for patient {patient_id}.")

        if logger:
            logger.info(
                f"Saved prediction timeline plots for {len(selected_patients)} patients"
            )

    except AssertionError as ae:
        if logger:
            logger.error(f"Assertion Error: {ae}", exc_info=True)
        raise
    except Exception as e:
        if logger:
            logger.error(f"Error plotting prediction timeline: {e}", exc_info=True)
        raise
    finally:
        plt.close("all")


def plot_feature_interactions(
    data: pd.DataFrame,
    features: List[str],
    report_dir: str,
    model_name: str = "model",
    top_n: int = 5,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Create feature interaction plots for top correlated feature pairs.
    """
    try:
        # Calculate correlation matrix
        corr_matrix = data[features].corr().abs()

        # Get top correlated pairs
        pairs = []
        for i, j in combinations(range(len(features)), 2):
            pairs.append((features[i], features[j], corr_matrix.iloc[i, j]))

        top_pairs = sorted(pairs, key=lambda x: x[2], reverse=True)[:top_n]

        # Create plots for top pairs
        for feat1, feat2, corr in top_pairs:
            plt.figure(figsize=(10, 8))

            # Create scatter plot with sepsis hue
            sns.scatterplot(data=data, x=feat1, y=feat2, hue="SepsisLabel", alpha=0.5)

            plt.title(
                f"Feature Interaction: {feat1} vs {feat2}\nCorrelation: {corr:.2f}"
            )

            # Add regression lines for each class
            sns.regplot(
                data=data[data["SepsisLabel"] == 0],
                x=feat1,
                y=feat2,
                scatter=False,
                label="Non-Sepsis Trend",
            )
            sns.regplot(
                data=data[data["SepsisLabel"] == 1],
                x=feat1,
                y=feat2,
                scatter=False,
                label="Sepsis Trend",
            )

            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{report_dir}/{model_name}_interaction_{feat1}_{feat2}.png")
            plt.close()

        if logger:
            logger.info(f"Saved feature interaction plots for top {top_n} pairs")

    except Exception as e:
        if logger:
            logger.error(f"Error plotting feature interactions: {e}", exc_info=True)
        raise


def save_plot(
    plt_obj: plt,
    report_dir: str,
    filename: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Save plot to file and close it."""
    try:
        plt_obj.savefig(os.path.join(report_dir, filename))
        if logger:
            logger.info(f"Saved plot: {filename}")
    except Exception as e:
        if logger:
            logger.error(f"Failed to save plot {filename}: {e}")
        raise
    finally:
        plt_obj.close()


def log_message(
    logger: Optional[logging.Logger], message: str, level: str = "info"
) -> None:
    """Log message using logger if available, otherwise print."""
    if logger:
        getattr(logger, level)(message)
    else:
        print(f"{level.upper()}: {message}")


def log_metrics(
    logger: Optional[logging.Logger], model_name: str, metrics: Dict[str, float]
) -> None:
    """Log evaluation metrics."""
    log_message(logger, f"\n{model_name} Evaluation:")
    for metric, value in metrics.items():
        if value is not None:
            log_message(logger, f"  {metric:<25} : {value:.4f}")
        else:
            log_message(logger, f"  {metric:<25} : Not Available")


def save_metrics_to_json(
    metrics: Dict[str, float],
    model_name: str,
    report_dir: str,
    logger: Optional[logging.Logger],
) -> None:
    """Save metrics to JSON file."""
    try:
        metrics_path = os.path.join(report_dir, f"{model_name}_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        log_message(logger, f"Saved evaluation metrics to {metrics_path}\n")
    except Exception as e:
        if logger:
            logger.error(f"Failed to save metrics to JSON: {e}")
        raise
```

## src/feature_engineering.py

```python
"""Feature engineering module for preprocessing medical data.

This module provides functions for preprocessing medical dataset features including:
- Dropping redundant and null columns
- Imputing missing values
- Encoding categorical variables
- Applying transformations (log, scaling)

The module is designed to work with patient medical data containing vital signs
and lab test results.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler, StandardScaler


def drop_columns(df):
    """Drop specified redundant columns from the dataset.

    Args:
        df (pd.DataFrame): Input dataframe containing medical data

    Returns:
        pd.DataFrame: Dataframe with redundant columns removed

    Note:
        Preserves Unit1 and Unit2 columns while removing other specified columns
    """
    columns_drop = {
        "Unnamed: 0",  # Index column
        # Vital signs that are derived or redundant
        "SBP",
        "DBP",
        "EtCO2",
        # Blood gas and chemistry redundancies
        "BaseExcess",
        "HCO3",
        "pH",
        "PaCO2",
        # Duplicated or highly correlated lab values
        "Alkalinephos",
        "Calcium",
        "Magnesium",
        "Phosphate",
        "Potassium",
        "PTT",
        "Fibrinogen",
    }
    df = df.drop(columns=columns_drop)
    return df


def fill_missing_values(df):
    """Impute missing values using iterative imputation (MICE algorithm).

    Args:
        df (pd.DataFrame): Input dataframe with missing values

    Returns:
        pd.DataFrame: Dataframe with imputed values

    Note:
        - Preserves categorical columns during imputation
        - Uses mean initialization and 20 maximum iterations
        - Maintains reproducibility with fixed random state
    """
    # Create a copy of the dataframe
    df_copy = df.copy()

    # Separate Patient_ID and categorical columns
    id_column = df_copy["Patient_ID"]
    categorical_columns = ["Gender", "Unit1", "Unit2"]
    categorical_data = df_copy[categorical_columns]

    # Get numerical columns for imputation
    numerical_columns = df_copy.select_dtypes(include=[np.number]).columns
    numerical_data = df_copy[numerical_columns]

    # Initialize and fit the IterativeImputer
    imputer = IterativeImputer(
        random_state=42,  # Ensures reproducibility.
        max_iter=100,  # Maximum number of iterations.
        initial_strategy="mean",  # Initial imputation strategy.
        skip_complete=True,  # Skips columns without missing values to save computation.
    )

    # Perform imputation on numerical columns
    imputed_numerical = pd.DataFrame(
        imputer.fit_transform(numerical_data),  # Perform imputation.
        columns=numerical_columns,  # Preserve original column names.
        index=df_copy.index,  # Maintain original index.
    )

    # Combine the imputed numerical data with categorical data
    df = pd.concat([id_column, categorical_data, imputed_numerical], axis=1)

    return df


def drop_null_columns(df):
    """Drop specified columns if they exist."""
    null_col = [
        "TroponinI",
        "Bilirubin_direct",
        "AST",
        "Bilirubin_total",
        "Lactate",
        "SaO2",
        "FiO2",
    ]

    # Identify columns that exist in the DataFrame
    existing_cols = [col for col in null_col if col in df.columns]

    if existing_cols:
        df = df.drop(columns=existing_cols)
        logging.info(f"Dropped columns: {existing_cols}")
    else:
        logging.warning(f"No columns to drop from drop_null_columns: {null_col}")

    return df


def one_hot_encode_gender(df):
    """One-hot encode the Gender column, ensuring no column name conflicts."""
    one_hot = pd.get_dummies(df["Gender"], prefix="Gender")

    # Ensure there are no overlapping columns
    overlapping_columns = set(one_hot.columns) & set(df.columns)
    if overlapping_columns:
        df = df.drop(columns=overlapping_columns)

    df = df.join(one_hot)
    df = df.drop("Gender", axis=1)
    return df


def log_transform(df, columns):
    """Apply log transformation to handle skewed numeric features.

    Args:
        df (pd.DataFrame): Input dataframe
        columns (list): List of column names to transform

    Returns:
        pd.DataFrame: Dataframe with log-transformed columns

    Note:
        Uses log(x + 1) transformation with minimum clipping at 1e-5
        to handle zeros and small values
    """
    for col in columns:
        # Clip values to prevent log(0) or log(negative)
        df[col] = np.log(df[col].clip(lower=1e-5) + 1)
    return df


def standard_scale(df, columns):
    """Standardize specified columns."""
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


def robust_scale(df, columns):
    """Apply Robust Scaling to specified columns."""
    scaler = RobustScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


def preprocess_data(df):
    """Execute complete preprocessing pipeline for medical data.

    Pipeline steps:
    1. Drop redundant columns
    2. Impute missing values using MICE
    3. Drop specified null columns
    4. One-hot encode gender
    5. Apply log transformation to skewed features
    6. Apply robust scaling to numeric features
    7. Handle remaining NaN values
    8. Standardize column names

    Args:
        df (pd.DataFrame): Raw input dataframe

    Returns:
        pd.DataFrame: Fully preprocessed dataframe ready for modeling

    Note:
        Logs progress at each major preprocessing step
    """
    logging.info("Starting preprocessing")

    df = drop_columns(df)
    logging.info(f"After dropping columns: {df.columns.tolist()}")

    df = fill_missing_values(df)
    logging.info(f"After imputing missing values: {df.columns.tolist()}")

    df = drop_null_columns(df)
    logging.info(f"After dropping null columns: {df.columns.tolist()}")

    df = one_hot_encode_gender(df)
    logging.info(f"After one-hot encoding gender: {df.columns.tolist()}")

    # Log transformations
    columns_log = ["MAP", "BUN", "Creatinine", "Glucose", "WBC", "Platelets"]
    df = log_transform(df, columns_log)
    logging.info(f"After log transformation: {df.columns.tolist()}")

    # Standard scaling
    columns_scale = [
        "HR",
        "O2Sat",
        "Temp",
        "MAP",
        "Resp",
        "BUN",
        "Chloride",
        "Creatinine",
        "Glucose",
        "Hct",
        "Hgb",
        "WBC",
        "Platelets",
    ]
    df = robust_scale(df, columns_scale)
    logging.info(f"After scaling: {df.columns.tolist()}")

    # Drop any remaining NaNs
    df = df.dropna()
    logging.info(f"After dropping remaining NaNs: {df.columns.tolist()}")

    # Convert all column names to strings
    df.columns = df.columns.astype(str)
    logging.info(f"Final preprocessed columns: {df.columns.tolist()}")

    return df
```

## src/logger_config.py

```python
# src/logger_config.py

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Union

import colorlog
from pythonjsonlogger import jsonlogger

# Global dict to track configured loggers
_CONFIGURED_LOGGERS = {}


def setup_logger(
    name: str = "sepsis_prediction",
    log_file: str = "logs/sepsis_prediction.log",
    level: Union[str, int] = "INFO",
    use_json: bool = False,
    formatter: str = "%(asctime)s - %(message)s",
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 5,
    console: bool = True,
) -> logging.Logger:
    """
    Set up and configure a logger with both console and file handling capabilities.
    """
    # Check if logger was already configured
    if name in _CONFIGURED_LOGGERS:
        return _CONFIGURED_LOGGERS[name]

    try:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # Handle logging level configuration
        if isinstance(level, str):
            level = logging.getLevelName(level.upper())
            if not isinstance(level, int):
                raise ValueError(f"Invalid log level: {level}")

        # Initialize logger
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False  # Prevent propagation to ancestor loggers

        # Remove any existing handlers
        logger.handlers.clear()

        # Configure console logging if enabled
        if console:
            console_handler = colorlog.StreamHandler()
            console_handler.setLevel(level)

            if use_json:
                console_formatter = jsonlogger.JsonFormatter(
                    "%(asctime)s %(name)s %(levelname)s %(message)s"
                )
            else:
                console_formatter = colorlog.ColoredFormatter(
                    "%(log_color)s%(asctime)s - %(message)s%(reset)s",
                    datefmt="%m-%d %H:%M",
                    log_colors={
                        "DEBUG": "cyan",
                        "INFO": "green",
                        "WARNING": "yellow",
                        "ERROR": "red",
                        "CRITICAL": "red,bg_white",
                    },
                    secondary_log_colors={},
                    style="%",
                )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        # Configure rotating file handler
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setLevel(level)

        if use_json:
            file_formatter = jsonlogger.JsonFormatter(
                "%(asctime)s %(name)s %(levelname)s %(message)s"
            )
        else:
            file_formatter = logging.Formatter(formatter, datefmt="%m-%d %H:%M")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Store the configured logger
        _CONFIGURED_LOGGERS[name] = logger

    except Exception as e:
        logging.basicConfig(level=logging.INFO)
        fallback_logger = logging.getLogger(name)
        fallback_logger.error(f"Failed to set up logger '{name}': {e}")
        return fallback_logger

    return logger


# Add function to get existing logger
def get_logger(
    name: str = "sepsis_prediction",
    level: Union[str, int] = "INFO",
    use_json: bool = False,
) -> logging.Logger:
    """Get an existing logger or create a new one with custom settings."""
    if name in _CONFIGURED_LOGGERS:
        logger = _CONFIGURED_LOGGERS[name]
        logger.setLevel(level)  # Update level even if logger exists
        return logger

    return setup_logger(
        name=name, level=level, use_json=use_json, log_file=f"logs/{name}.log"
    )


def disable_duplicate_logging():
    """Disable duplicate logging by removing handlers from the root logger."""
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
```

## src/logger_utils.py

```python
# src/logging_utils.py
import time
from typing import Optional, Callable, Union
import psutil

import logging
from contextlib import contextmanager
import pandas as pd
import functools
from functools import wraps


def log_phase(logger: logging.Logger) -> Callable:
    """
    Decorator to log the start, end, and duration of each processing phase.
    Compatible with the existing sepsis prediction logger.

    Usage:
    @log_phase(logger)
    def your_function():
        ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            phase_name = func.__name__.replace("_", " ").title()

            # Start phase logging
            logger.info("=" * 80)
            logger.info(f"Starting Phase: {phase_name}")
            logger.info("-" * 80)

            # Log initial memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            logger.info(f"Initial Memory Usage: {initial_memory:.2f} MB")

            # Execute phase
            start_time = time.time()
            try:
                result = func(*args, **kwargs)

                # Log successful completion
                duration = time.time() - start_time
                final_memory = process.memory_info().rss / 1024 / 1024
                memory_diff = final_memory - initial_memory

                logger.info("-" * 80)
                logger.info(f"Phase Complete: {phase_name}")
                logger.info(f"Duration: {duration:.2f} seconds")
                logger.info(f"Final Memory Usage: {final_memory:.2f} MB")
                logger.info(f"Memory Change: {memory_diff:+.2f} MB")
                logger.info("=" * 80)

                return result

            except Exception as e:
                # Log failure
                logger.error(f"Phase Failed: {phase_name}")
                logger.error(f"Error: {str(e)}", exc_info=True)
                logger.error("=" * 80)
                raise

        return wrapper

    return decorator


@contextmanager
def log_step(logger: logging.Logger, step_name: str):
    """
    Context manager for logging individual steps within a phase.

    Usage:
    with log_step(logger, "Data Preprocessing"):
        preprocess_data()
    """
    try:
        logger.info(f"Starting step: {step_name}")
        start_time = time.time()
        yield
        duration = time.time() - start_time
        logger.info(f"Completed step: {step_name} (Duration: {duration:.2f}s)")
    except Exception as e:
        logger.error(f"Error in step {step_name}: {str(e)}")
        raise


def log_function(logger: Optional[logging.Logger] = None) -> Callable:
    """Decorator for logging function entry and exit."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__.replace("_", " ").title()
            if logger:
                logger.info(f"Starting {func_name}")
            try:
                result = func(*args, **kwargs)
                if logger:
                    logger.info(f"Completed {func_name}")
                return result
            except Exception as e:
                if logger:
                    logger.error(f"Error in {func_name}: {str(e)}", exc_info=True)
                raise

        return wrapper

    return decorator


def log_memory(logger: logging.Logger, label: str = "Current"):
    """Log current memory usage."""
    process = psutil.Process()
    memory_usage = process.memory_info().rss / 1024 / 1024  # MB
    logger.info(f"{label} Memory Usage: {memory_usage:.2f} MB")


def log_dataframe_info(
    logger: logging.Logger, df: Union[pd.DataFrame, pd.Series], name: str
):
    logger.info(f"\n{name} Information:")
    logger.info(f"Shape: {df.shape}")  # Shape works for both Series and DataFrame
    if isinstance(df, pd.DataFrame):
        logger.info(f"Memory usage: {df.memory_usage().sum() / 1024 / 1024:.2f} MB")
    elif isinstance(df, pd.Series):
        logger.info(f"Memory usage: {df.memory_usage(deep=True) / 1024 / 1024:.2f} MB")
    else:
        logger.warning(
            f"log_dataframe_info called with an unsupported type: {type(df)}"
        )

    if isinstance(df, pd.DataFrame):
        logger.info(f"Columns: {list(df.columns)}")
        # Log missing value information
        missing_info = df.isnull().sum()
        if missing_info.any():
            missing_cols = missing_info[missing_info > 0]
            logger.info("\nColumns with missing values:")
            for col, count in missing_cols.items():
                percentage = (count / len(df)) * 100
                logger.info(f"{col}: {count} missing values ({percentage:.2f}%)")

        # Log class distribution if this is target data
        if "SepsisLabel" in df.columns:
            class_dist = df["SepsisLabel"].value_counts()
            logger.info(f"Class distribution: {dict(class_dist)}")

    elif isinstance(df, pd.Series):
        logger.info(f"Series name: {df.name}")
        # Log value counts for series (useful for target variables)
        if df.dtype in ["int64", "float64", "bool", "category"]:
            value_counts = df.value_counts()
            if len(value_counts) <= 10:
                logger.info(f"Value Counts:\n{value_counts}")
        # Log missing values for series
        missing_count = df.isnull().sum()
        if missing_count > 0:
            percentage = (missing_count / len(df)) * 100
            logger.info(f"Missing values: {missing_count} ({percentage:.2f}%)")

        # Log class distribution if this is target data
        if name.lower().startswith("y_") or (
            df.name and "sepsislabel" in str(df.name).lower()
        ):
            class_dist = df.value_counts()
            logger.info(f"Class distribution: {dict(class_dist)}")
```

## src/model_registry.py

```python
# src/model_registry.py

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import json
import logging
import shutil
import re
import uuid
import fcntl  # For file locking on Unix-based systems


@dataclass
class ModelVersion:
    """Data class to store model version information."""

    name: str
    version: str
    timestamp: str
    metrics: Dict[str, float]
    params: Dict[str, Any]
    tags: List[str]

    def to_dict(self) -> dict:
        """Convert ModelVersion to dictionary."""
        return asdict(self)


class ModelRegistry:
    """Centralized model registry for managing ML models and their metadata."""

    def __init__(self, base_dir: str, logger: Optional[logging.Logger] = None):
        """
        Initialize ModelRegistry.

        Args:
            base_dir: Base directory for model storage
            logger: Optional logger instance
        """
        if not isinstance(base_dir, (str, Path)):
            raise TypeError("base_dir must be a string or Path object")

        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "models"
        self.reports_dir = self.base_dir / "reports"
        self.logger = logger

        # Create necessary directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Initialize registry metadata
        self.registry_file = self.base_dir / "registry.json"
        self._initialize_registry()

    def _initialize_registry(self):
        """Initialize or load the registry metadata file with file locking."""
        # Ensure the registry file exists
        self.registry_file.touch(exist_ok=True)

        # Acquire a lock for thread-safe and process-safe operations
        with self.registry_file.open("r+") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    f.seek(0)
                    content = f.read()
                    if content:
                        self.registry = json.loads(content)
                    else:
                        self.registry = {"models": {}}
                        f.write(json.dumps(self.registry, indent=4))
                        f.flush()
                except json.JSONDecodeError:
                    # Handle corrupted JSON file
                    self.logger.error(
                        f"Registry file {self.registry_file} is corrupted. Reinitializing."
                    )
                    self.registry = {"models": {}}
                    f.seek(0)
                    f.truncate()
                    f.write(json.dumps(self.registry, indent=4))
                    f.flush()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def _save_registry(self):
        """Save the registry metadata to file with file locking."""
        with self.registry_file.open("w") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX)
                json.dump(self.registry, f, indent=4)
                f.flush()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def _sanitize_model_name(self, name: str) -> str:
        """
        Sanitize the model name to create a valid directory name.

        Args:
            name: Original model name

        Returns:
            Sanitized model name
        """
        # Remove or replace invalid characters
        sanitized_name = re.sub(r'[<>:"/\\|?*]', "_", name)
        return sanitized_name

    def _log(self, message: str, level: str = "info"):
        """Log message if logger is configured."""
        if self.logger:
            getattr(self.logger, level)(message)

    def save_model(
        self,
        model: Any,
        name: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        artifacts: Optional[Dict[str, str]] = None,
        tags: Optional[List[str]] = None,
    ) -> ModelVersion:
        """
        Save a model and its associated metadata.

        Args:
            model: The trained model object
            name: Name of the model
            params: Model parameters/hyperparameters
            metrics: Model performance metrics
            artifacts: Optional dictionary of associated artifacts (e.g., plots)
            tags: Optional list of tags for the model

        Returns:
            ModelVersion object containing model metadata
        """
        sanitized_name = self._sanitize_model_name(name)
        # Generate timestamp and ensure uniqueness with UUID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:6]  # 6-character unique identifier
        version = f"v{len(self.registry['models'].get(sanitized_name, [])) + 1}"

        # Create unique model directory
        model_dir_name = f"{sanitized_name}_{timestamp}_{unique_id}"
        model_dir = self.models_dir / model_dir_name
        try:
            model_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            # Extremely unlikely due to UUID, but handle just in case
            model_dir_name = (
                f"{sanitized_name}_{timestamp}_{unique_id}_{uuid.uuid4().hex[:4]}"
            )
            model_dir = self.models_dir / model_dir_name
            model_dir.mkdir(parents=True, exist_ok=False)

        try:
            # Save model
            model_path = model_dir / "model.pkl"
            joblib.dump(model, model_path)
        except Exception as e:
            # Cleanup the model directory in case of failure
            shutil.rmtree(model_dir, ignore_errors=True)
            self._log(
                f"Failed to save model '{name}' version '{version}': {e}", "error"
            )
            raise

        # Create model version object
        model_version = ModelVersion(
            name=name,
            version=version,
            timestamp=timestamp,
            metrics=metrics,
            params=params,
            tags=tags or [],
        )

        # Save metadata
        metadata = {
            **model_version.to_dict(),
            "model_path": str(model_path.resolve()),
            "artifacts": artifacts or {},
        }

        try:
            with (model_dir / "metadata.json").open("w") as f:
                json.dump(metadata, f, indent=4)
        except Exception as e:
            # Cleanup in case of failure
            shutil.rmtree(model_dir, ignore_errors=True)
            self._log(
                f"Failed to save metadata for model '{name}' version '{version}': {e}",
                "error",
            )
            raise

        # Update registry with file locking, ensuring that the model is stored under "models"
        self.registry["models"].setdefault(sanitized_name, []).append(metadata)
        self._save_registry()

        self._log(f"Saved model '{name}' version '{version}' to '{model_dir}'", "info")
        return model_version

    def load_model(
        self, name: str, version: Optional[str] = None
    ) -> Tuple[Any, ModelVersion]:
        """
        Load a model and its metadata.

        Args:
            name: Name of the model
            version: Optional version string (loads latest if not specified)

        Returns:
            Tuple of (model object, ModelVersion)
        """
        sanitized_name = self._sanitize_model_name(name)
        if sanitized_name not in self.registry["models"]:
            raise ValueError(f"Model '{name}' not found in registry")

        versions = self.registry["models"][sanitized_name]
        if not versions:
            raise ValueError(f"No versions found for model '{name}'")

        # Select version
        if version:
            model_metadata = next(
                (v for v in versions if v["version"] == version), None
            )
            if not model_metadata:
                raise ValueError(f"Version '{version}' not found for model '{name}'")
        else:
            model_metadata = versions[-1]  # Latest version

        model_path = Path(model_metadata["model_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        try:
            # Load model
            model = joblib.load(model_path)
        except Exception as e:
            self._log(
                f"Failed to load model '{name}' version '{model_metadata['version']}': {e}",
                "error",
            )
            raise

        # Create ModelVersion object
        model_version = ModelVersion(
            name=model_metadata["name"],
            version=model_metadata["version"],
            timestamp=model_metadata["timestamp"],
            metrics=model_metadata["metrics"],
            params=model_metadata["params"],
            tags=model_metadata["tags"],
        )

        self._log(f"Loaded model '{name}' version '{model_version.version}'", "info")
        return model, model_version

    def get_best_model(
        self, name: str, metric: str, higher_is_better: bool = True
    ) -> Tuple[Any, ModelVersion]:
        """
        Load the best performing model based on a specific metric.

        Args:
            name: Name of the model
            metric: Metric to use for comparison
            higher_is_better: Whether higher metric values are better

        Returns:
            Tuple of (model object, ModelVersion)
        """
        sanitized_name = self._sanitize_model_name(name)
        if sanitized_name not in self.registry["models"]:
            raise ValueError(f"Model '{name}' not found in registry")

        versions = self.registry["models"][sanitized_name]
        if not versions:
            raise ValueError(f"No versions found for model '{name}'")

        # Filter versions that have the specified metric
        valid_versions = [v for v in versions if metric in v["metrics"]]
        if not valid_versions:
            raise ValueError(
                f"No versions of model '{name}' have the metric '{metric}'"
            )

        # Determine the best version
        if higher_is_better:
            best_version_metadata = max(
                valid_versions, key=lambda v: v["metrics"][metric]
            )
        else:
            best_version_metadata = min(
                valid_versions, key=lambda v: v["metrics"][metric]
            )

        model_path = Path(best_version_metadata["model_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        try:
            # Load model
            model = joblib.load(model_path)
        except Exception as e:
            self._log(
                f"Failed to load best model '{name}' version '{best_version_metadata['version']}': {e}",
                "error",
            )
            raise

        # Create ModelVersion object
        model_version = ModelVersion(
            name=best_version_metadata["name"],
            version=best_version_metadata["version"],
            timestamp=best_version_metadata["timestamp"],
            metrics=best_version_metadata["metrics"],
            params=best_version_metadata["params"],
            tags=best_version_metadata["tags"],
        )

        self._log(
            f"Loaded best model '{name}' version '{model_version.version}' "
            f"with {metric}={best_version_metadata['metrics'][metric]}",
            "info",
        )
        return model, model_version

    def list_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """List all models in the registry."""
        return self.registry["models"]

    def get_model_versions(self, name: str) -> List[ModelVersion]:
        """Get all versions of a specific model."""
        sanitized_name = self._sanitize_model_name(name)
        if sanitized_name not in self.registry["models"]:
            raise ValueError(f"Model '{name}' not found in registry")

        versions = []
        for v in self.registry["models"][sanitized_name]:
            version = ModelVersion(
                name=v["name"],
                version=v["version"],
                timestamp=v["timestamp"],
                metrics=v["metrics"],
                params=v["params"],
                tags=v["tags"],
            )
            versions.append(version)

        return versions

    def delete_model(self, name: str, version: Optional[str] = None):
        """
        Delete a model or specific version from the registry.

        Args:
            name: Name of the model
            version: Optional version to delete (deletes all versions if not specified)
        """
        sanitized_name = self._sanitize_model_name(name)
        if sanitized_name not in self.registry["models"]:
            raise ValueError(f"Model '{name}' not found in registry")

        if version:
            # Delete specific version
            versions = self.registry["models"][sanitized_name]
            version_metadata = next(
                (v for v in versions if v["version"] == version), None
            )
            if not version_metadata:
                raise ValueError(f"Version '{version}' not found for model '{name}'")

            model_path = Path(version_metadata["model_path"])
            model_dir = model_path.parent

            # Remove model directory
            if model_dir.exists() and model_dir.is_dir():
                try:
                    shutil.rmtree(model_dir)
                    self._log(
                        f"Deleted model '{name}' version '{version}' from '{model_dir}'",
                        "info",
                    )
                except Exception as e:
                    self._log(
                        f"Failed to delete model directory '{model_dir}': {e}", "error"
                    )
                    raise
            else:
                self._log(f"Model directory '{model_dir}' does not exist.", "warning")

            # Remove version from registry
            self.registry["models"][sanitized_name] = [
                v for v in versions if v["version"] != version
            ]
            if not self.registry["models"][sanitized_name]:
                del self.registry["models"][sanitized_name]
        else:
            # Delete all versions
            versions = self.registry["models"][sanitized_name]
            for version_metadata in versions:
                model_path = Path(version_metadata["model_path"])
                model_dir = model_path.parent
                if model_dir.exists() and model_dir.is_dir():
                    try:
                        shutil.rmtree(model_dir)
                        self._log(f"Deleted model directory '{model_dir}'", "info")
                    except Exception as e:
                        self._log(
                            f"Failed to delete model directory '{model_dir}': {e}",
                            "error",
                        )
                        raise
                else:
                    self._log(
                        f"Model directory '{model_dir}' does not exist.", "warning"
                    )

            # Remove model entry from registry
            del self.registry["models"][sanitized_name]

        # Save updated registry
        self._save_registry()
        self._log(
            f"Deleted model '{name}'" f"{f' version {version}' if version else ''}",
            "info",
        )
```

## src/utils.py

```python
import json
import logging
import os
from typing import Dict, Optional


def log_message(
    logger: Optional[logging.Logger], message: str, level: str = "info"
) -> None:
    """Log message using logger if available, otherwise print."""
    if logger:
        getattr(logger, level)(message)
    else:
        # Only print if no logger is available
        print(f"{level.upper()}: {message}")


def log_metrics(
    logger: Optional[logging.Logger], model_name: str, metrics: Dict[str, float]
) -> None:
    """Log evaluation metrics."""
    log_message(logger, f"\n{model_name} Evaluation:")
    for metric, value in metrics.items():
        if value is not None:
            log_message(logger, f" {metric:<25} : {value:.4f}")
        else:
            log_message(logger, f" {metric:<25} : Not Available")


def save_metrics_to_json(
    metrics: Dict[str, float],
    model_name: str,
    report_dir: str,
    logger: Optional[logging.Logger],
) -> None:
    """Save metrics to JSON file."""
    metrics_path = os.path.join(report_dir, f"{model_name}_metrics.json")
    try:
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        log_message(logger, f"Saved evaluation metrics to {metrics_path}\n")
    except Exception as e:
        if logger:
            logger.error(f"Failed to save metrics to JSON: {e}")
        else:
            print(f"Failed to save metrics to JSON: {e}")
```
