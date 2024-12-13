import datetime
import gc
import json
import logging
import os
import argparse
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
        default=1,
        help="Number of trials for Random Forest optimization (default: 20)",
    )
    parser.add_argument(
        "--lr-trials",
        type=int,
        default=1,
        help="Number of trials for Logistic Regression optimization (default: 20)",
    )
    parser.add_argument(
        "--xgb-trials",
        type=int,
        default=1,
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
    return parser.parse_args()


# Initialize ModelRegistry early to use it in case of exceptions during setup
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
) -> Tuple[Dict[str, float], Any]:
    """
    Trains and evaluates a model. Handles potential errors in evaluation by re-raising exceptions.

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
            - model: Trained model object
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

        return metrics, model

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


@log_phase(logger)
def main():
    """Main function to execute the Sepsis Prediction Pipeline."""
    # Set up logger
    logger = get_logger(name="sepsis_prediction", level="INFO", use_json=False)
    logger.info("Starting Sepsis Prediction Pipeline")

    # Define checkpoint paths
    checkpoints = {
        "preprocessed_data": "checkpoints/preprocessed_data.pkl",
        "resampled_data": "checkpoints/resampled_data.pkl",
        "trained_models": "checkpoints/trained_models.pkl",
        "optuna_studies": "checkpoints/optuna_studies.pkl",
    }

    # Create checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)

    try:
        # Generate a unique identifier for this run
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_report_dir = os.path.join("reports", "evaluations", f"run_{run_id}")
        os.makedirs(unique_report_dir, exist_ok=True)
        logger.info(f"Reports and metrics will be saved to: {unique_report_dir}")

        # Step 1: Load and preprocess data
        if os.path.exists(checkpoints["preprocessed_data"]):
            logger.info("Loading preprocessed data from checkpoint.")
            df_train_processed, df_val_processed, df_test_processed, df_val_original = (
                joblib.load(checkpoints["preprocessed_data"])
            )
        else:
            logger.info("Loading and preprocessing data.")
            combined_df = load_data(args.data_path)
            logger.info("Splitting data into training, validation, and testing sets")
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

            # Save preprocessed data along with original df_val
            joblib.dump(
                (
                    df_train_processed,
                    df_val_processed,
                    df_test_processed,
                    df_val_original,
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

        # Initialize best score
        models = {}
        best_score = 0
        best_model_name = None

        # Check if trained models exist
        if os.path.exists(checkpoints["trained_models"]):
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
        else:
            # Step 3: Hyperparameter Tuning and Model Training
            # Define Optuna studies with descriptive storage URL
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            storage_url = (
                f"sqlite:///sepsis_prediction_optimization_{timestamp}.db"
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
                                use_label_encoder=False,
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

                studies["XGBoost_Optimization"].optimize(
                    xgb_objective, n_trials=args.xgb_trials, n_jobs=parallel_jobs
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
                best_model = models[best_model_name]["model"]

                # Make predictions on the test set
                if "xgb_classifier" in best_model.named_steps:
                    # Handle XGBoost predictions within the pipeline
                    final_predictions_proba = best_model.predict_proba(
                        df_test_processed.drop("SepsisLabel", axis=1)
                    )[:, 1]
                    final_predictions = (final_predictions_proba > 0.5).astype(int)
                    logger.info(f"Predictions made with {best_model_name} on test set.")
                else:
                    # Handle other models
                    final_predictions = best_model.predict(
                        df_test_processed.drop("SepsisLabel", axis=1)
                    )
                    final_predictions_proba = (
                        best_model.predict_proba(
                            df_test_processed.drop("SepsisLabel", axis=1)
                        )[:, 1]
                        if hasattr(best_model, "predict_proba")
                        else None
                    )
                    logger.info(f"Predictions made with {best_model_name} on test set.")

                # Convert predictions to pandas.Series
                final_predictions = pd.Series(
                    final_predictions, index=df_test_processed.index, name="Predicted"
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
                df_test_with_predictions = df_test.copy()  # Use original df_test
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
                    model=best_model,
                    report_dir=unique_report_dir,
                    logger=logger,
                )
                logger.info(f"Final evaluation completed for {best_model_name}.")

                # Generate model card
                generate_model_card(
                    model=best_model,
                    model_name=best_model_name,
                    metrics=metrics
                    if "metrics" in locals()
                    else metrics_xgb,  # Use appropriate metrics
                    train_data=df_train_processed,
                    val_data=df_val_processed,
                    test_data=df_test_processed,
                    report_dir=unique_report_dir,
                    run_id=run_id,
                    logger=logger,
                )

                # Step 5: Save the best model using ModelRegistry
                logger.info(f"Saving the best model ({best_model_name})")
                # Extract the estimator's parameters for saving
                estimator_step = best_model_name.lower().replace(" ", "_")
                best_model_params = best_model.named_steps.get(
                    estimator_step
                ).get_params()

                model_registry.save_model(
                    model=best_model,
                    name=best_model_name,
                    params=best_model_params,
                    metrics=metrics if "metrics" in locals() else metrics_xgb,
                    artifacts={
                        "confusion_matrix": os.path.join(
                            unique_report_dir, f"{best_model_name}_confusion_matrix.png"
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
                del best_model, final_predictions, final_predictions_proba
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

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
