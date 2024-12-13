# optuna_main.py

# optuna_main.py

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
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score
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
        default=10,
        help="Number of parallel jobs for Optuna hyperparameter tuning (default: 20)",
    )
    parser.add_argument(
        "--rf-trials",
        type=int,
        default=20,
        help="Number of trials for Random Forest optimization (default: 50)",
    )
    parser.add_argument(
        "--lr-trials",
        type=int,
        default=20,
        help="Number of trials for Logistic Regression optimization (default: 30)",
    )
    parser.add_argument(
        "--xgb-trials",
        type=int,
        default=20,
        help="Number of trials for XGBoost optimization (default: 50)",
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


def train_and_evaluate_model(
    model_name: str,
    pipeline: ImbPipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    logger: logging.Logger,
) -> Tuple[Dict[str, float], Any]:
    """
    Train and evaluate a model using the unified evaluation function.
    """
    logger.info(f"Training {model_name}...")

    try:
        # Train the model
        pipeline.fit(X_train, y_train)

        # Make predictions
        y_pred = pipeline.predict(X_val)
        y_pred_proba = (
            pipeline.predict_proba(X_val)[:, 1]
            if hasattr(pipeline, "predict_proba")
            else None
        )

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

        # Evaluate the model
        metrics = evaluate_model(
            y_true=y_val,
            y_pred=y_pred,
            model_name=model_name,
            y_pred_proba=y_pred_proba,
            model=model,
            logger=logger,
        )

        return metrics, model

    except Exception as e:
        logger.error(f"Error training {model_name}: {str(e)}", exc_info=True)
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
    """Save hyperparameters to a JSON file."""
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
    """Generates a model card in Markdown format."""

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
            n_estimators = model.named_steps["random_forest"].n_estimators
            max_depth = model.named_steps["random_forest"].max_depth
            f.write(f"- **Model Type:** Random Forest\n")
            f.write(
                f"- **Architecture:** Ensemble of {n_estimators} decision trees (max_depth={max_depth}) trained using bagging.\n"
            )
        elif "logistic_regression" in model_name.lower():
            penalty = model.named_steps["logistic_regression"].penalty
            C = model.named_steps["logistic_regression"].C
            f.write(f"- **Model Type:** Logistic Regression\n")
            f.write(
                f"- **Architecture:** Linear model with sigmoid function (regularization: {penalty}, C={C}).\n"
            )
        elif "xgboost" in model_name.lower():
            learning_rate = model.named_steps["xgb_classifier"].learning_rate
            max_depth = model.named_steps["xgb_classifier"].max_depth
            f.write(f"- **Model Type:** XGBoost\n")
            f.write(
                f"- **Architecture:** Gradient boosting with decision trees (learning_rate={learning_rate}, max_depth={max_depth}).\n"
            )

        # Add version if applicable
        f.write(f"- **Version:** v1.0\n")
        f.write(f"- **Hyperparameters:**\n")
        for param, value in model.named_steps.items():
            if param != "scaler":
                for hyperparam, hyperval in (
                    model.named_steps[param].get_params().items()
                ):
                    f.write(f"    - {hyperparam}: {hyperval}\n")

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
        f.write(f"| F1 Score | {metrics['F1 Score']:.4f} |\n")
        f.write(f"| Precision | {metrics['Precision']:.4f} |\n")
        f.write(f"| Recall | {metrics['Recall']:.4f} |\n")
        f.write(f"| Specificity | {metrics.get('Specificity', 'N/A'):.4f} |\n")
        f.write(f"| AUROC | {metrics.get('AUC-ROC', 'N/A'):.4f} |\n")
        f.write(f"| AUPRC | {metrics.get('AUPRC', 'N/A'):.4f} |\n")

        # Generate and save essential plots
        generate_evaluation_plots(
            y_true=test_data["SepsisLabel"],
            y_pred=model.predict(test_data.drop("SepsisLabel", axis=1)),
            y_pred_proba=(
                model.predict_proba(test_data.drop("SepsisLabel", axis=1))[:, 1]
                if hasattr(model, "predict_proba")
                else None
            ),
            model=model,
            model_name=model_name,
            report_dir=report_dir,
            logger=logger,
        )

        # Add confusion matrix image
        f.write("\n### Confusion Matrix\n\n")
        f.write(f"![Confusion Matrix]({model_name}_confusion_matrix.png)\n")

        # Add ROC curve image
        f.write("\n### ROC Curve\n\n")
        f.write(f"![ROC Curve]({model_name}_roc_curve.png)\n")

        # Add Precision-Recall curve image
        f.write("\n### Precision-Recall Curve\n\n")
        f.write(f"![Precision-Recall Curve]({model_name}_precision_recall_curve.png)\n")

        # Add feature importance plot if available
        if "feature_importance" in metrics:
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
            df_train_processed, df_val_processed, df_test_processed = joblib.load(
                checkpoints["preprocessed_data"]
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

            with log_step(logger, "Preprocessing testing data"):
                df_test_processed = preprocess_data(df_test)

            # Save preprocessed data
            joblib.dump(
                (df_train_processed, df_val_processed, df_test_processed),
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
                    n_jobs=10,
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
        else:
            # Step 3: Hyperparameter Tuning and Model Training
            # Define Optuna studies
            storage_url = "sqlite:///12_dec_optimization.db"
            if os.path.exists(checkpoints["optuna_studies"]):
                logger.info("Loading Optuna studies from checkpoint.")
                studies = joblib.load(checkpoints["optuna_studies"])
            else:
                logger.info("Initializing Optuna studies.")
                studies = {
                    "Random_Forest_Optimization": optuna.create_study(
                        direction="maximize",
                        study_name="Random_Forest_Optimization",
                        storage=storage_url,
                        load_if_exists=True,
                        pruner=optuna.pruners.MedianPruner(
                            n_startup_trials=5, n_warmup_steps=10
                        ),
                    ),
                    "Logistic_Regression_Optimization": optuna.create_study(
                        direction="maximize",
                        study_name="Logistic_Regression_Optimization",
                        storage=storage_url,
                        load_if_exists=True,
                        pruner=optuna.pruners.MedianPruner(
                            n_startup_trials=5, n_warmup_steps=10
                        ),
                    ),
                    "XGBoost_Optimization": optuna.create_study(
                        direction="maximize",
                        study_name="XGBoost_Optimization",
                        storage=storage_url,
                        load_if_exists=True,
                        pruner=optuna.pruners.MedianPruner(
                            n_startup_trials=5, n_warmup_steps=10
                        ),
                    ),
                }
                # Save initial studies
                joblib.dump(studies, checkpoints["optuna_studies"])
                logger.info(f"Saved Optuna studies to {checkpoints['optuna_studies']}")

            parallel_jobs = args.optuna_n_jobs
            logger.info(
                f"Starting Optuna hyperparameter tuning with {parallel_jobs} parallel jobs."
            )

            # Define Optuna objective functions
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
                                n_jobs=10,
                                **param,
                            ),
                        ),
                    ]
                )

                # Cross-validation setup
                f1_scorer = make_scorer(f1_score)
                cv_scores = cross_val_score(
                    xgb_pipeline,
                    X_train_resampled,
                    y_train_resampled,
                    cv=3,
                    scoring=f1_scorer,
                    n_jobs=10,
                )
                mean_cv_score = cv_scores.mean()
                trial.report(mean_cv_score, step=1)

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

                return mean_cv_score

            def rf_objective(trial: optuna.Trial) -> float:
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
                                n_jobs=10,
                                **param,
                            ),
                        ),
                    ]
                )

                # Cross-validation setup
                f1_scorer = make_scorer(f1_score)
                cv_scores = cross_val_score(
                    rf_pipeline,
                    X_train_resampled,
                    y_train_resampled,
                    cv=3,  # Number of folds
                    scoring=f1_scorer,
                    n_jobs=10,
                )
                mean_cv_score = cv_scores.mean()
                trial.report(mean_cv_score, step=1)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                return mean_cv_score

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

                # Perform cross-validation
                f1_scorer = make_scorer(f1_score)
                cv_scores = cross_val_score(
                    lr_pipeline,
                    X_train_resampled,
                    y_train_resampled,
                    cv=3,
                    scoring=f1_scorer,
                    n_jobs=10,
                )
                mean_cv_score = cv_scores.mean()
                trial.report(mean_cv_score, step=1)

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

                return mean_cv_score

            # Optimize with Optuna
            # Random Forest
            logger.info("Starting Optuna hyperparameter tuning for Random Forest")
            studies["Random_Forest_Optimization"].optimize(
                rf_objective, n_trials=args.rf_trials, n_jobs=parallel_jobs
            )
            best_rf_params = studies["Random_Forest_Optimization"].best_params
            # Log the number of trials and their statuses
            study_rf = studies["Random_Forest_Optimization"]
            completed_trials = [
                t
                for t in study_rf.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ]
            pruned_trials = [
                t for t in study_rf.trials if t.state == optuna.trial.TrialState.PRUNED
            ]
            failed_trials = [
                t for t in study_rf.trials if t.state == optuna.trial.TrialState.FAIL
            ]

            logger.info(f"Total trials: {len(study_rf.trials)}")
            logger.info(f"Completed trials: {len(completed_trials)}")
            logger.info(f"Pruned trials: {len(pruned_trials)}")
            logger.info(f"Failed trials: {len(failed_trials)}")

            rf_pipeline = ImbPipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "random_forest",
                        RandomForestClassifier(
                            random_state=42,
                            class_weight="balanced",
                            n_jobs=10,
                            **best_rf_params,
                        ),
                    ),
                ]
            )

            metrics_rf, best_rf_model = train_and_evaluate_model(
                model_name="Random Forest (Tuned)",
                pipeline=rf_pipeline,
                X_train=X_train_resampled,
                y_train=y_train_resampled,
                X_val=df_val_processed.drop("SepsisLabel", axis=1),
                y_val=df_val_processed["SepsisLabel"],
                logger=logger,
            )

            # Compare F1 scores
            if metrics_rf["F1 Score"] > best_score:
                best_score = metrics_rf["F1 Score"]
                best_model_name = "Random Forest (Tuned)"
                models["Random Forest (Tuned)"] = {
                    "model": best_rf_model,
                    "metrics": metrics_rf,
                }

            #  Save RF parameters and cleanup
            save_hyperparameters(
                "random_forest_tuned_params.json",
                best_rf_params,
                logger,
                unique_report_dir,
            )
            del rf_pipeline
            gc.collect()

            # Logistic Regression
            logger.info("Starting Optuna hyperparameter tuning for Logistic Regression")
            studies["Logistic_Regression_Optimization"].optimize(
                lr_objective, n_trials=args.lr_trials, n_jobs=parallel_jobs
            )
            best_lr_params = studies["Logistic_Regression_Optimization"].best_params
            logger.info(f"Best LR parameters: {best_lr_params}")
            logger.info(
                f"Best LR F1 score: {studies['Logistic_Regression_Optimization'].best_value:.4f}"
            )

            # Logistic Regression
            lr_pipeline = ImbPipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "logistic_regression",
                        LogisticRegression(
                            random_state=42, n_jobs=10, **best_lr_params
                        ),
                    ),
                ]
            )

            metrics_lr, best_lr_model = train_and_evaluate_model(
                model_name="Logistic Regression (Tuned)",
                pipeline=lr_pipeline,
                X_train=X_train_resampled,
                y_train=y_train_resampled,
                X_val=df_val_processed.drop("SepsisLabel", axis=1),
                y_val=df_val_processed["SepsisLabel"],
                logger=logger,
            )

            if metrics_lr["F1 Score"] > best_score:
                best_score = metrics_lr["F1 Score"]
                best_model_name = "Logistic Regression (Tuned)"
                models["Logistic Regression (Tuned)"] = {
                    "model": best_lr_model,
                    "metrics": metrics_lr,
                }

            # Save LR parameters and cleanup
            save_hyperparameters(
                "logistic_regression_tuned_params.json",
                best_lr_params,
                logger,
                unique_report_dir,
            )
            del lr_pipeline
            gc.collect()

            # XGBoost
            logger.info("Starting Optuna hyperparameter tuning for XGBoost")
            studies["XGBoost_Optimization"].optimize(
                xgb_objective, n_trials=args.xgb_trials, n_jobs=parallel_jobs
            )
            best_xgb_params = studies["XGBoost_Optimization"].best_params
            logger.info(f"Best XGBoost parameters: {best_xgb_params}")
            logger.info(
                f"Best XGBoost F1 score: {studies['XGBoost_Optimization'].best_value:.4f}"
            )

            xgb_pipeline = ImbPipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "xgb_classifier",
                        XGBClassifier(
                            random_state=42,
                            use_label_encoder=False,
                            eval_metric="logloss",
                            n_jobs=10,
                            **best_xgb_params,
                        ),
                    ),
                ]
            )

            metrics_xgb, best_xgb_model = train_and_evaluate_model(
                model_name="XGBoost (Tuned)",
                pipeline=xgb_pipeline,
                X_train=X_train_resampled,
                y_train=y_train_resampled,
                X_val=df_val_processed.drop("SepsisLabel", axis=1),
                y_val=df_val_processed["SepsisLabel"],
                logger=logger,
            )

            if metrics_xgb["F1 Score"] > best_score:
                best_score = metrics_xgb["F1 Score"]
                best_model_name = "XGBoost (Tuned)"
                models["XGBoost (Tuned)"] = {
                    "model": best_xgb_model,
                    "metrics": metrics_xgb,
                }

            # Save XGBoost parameters and cleanup
            save_hyperparameters(
                "xgboost_tuned_params.json",
                best_xgb_params,
                logger,
                unique_report_dir,
            )
            del xgb_pipeline
            gc.collect()

            # Save all trained models to checkpoint
            joblib.dump(models, checkpoints["trained_models"])
            logger.info(f"Saved trained models to {checkpoints['trained_models']}")

            # Save updated Optuna studies
            joblib.dump(studies, checkpoints["optuna_studies"])
            logger.info(
                f"Saved updated Optuna studies to {checkpoints['optuna_studies']}"
            )

        # Step 4: Final Evaluation on Test Set
        if best_model_name:
            logger.info(
                f"\nPerforming final evaluation with best model: {best_model_name}"
            )
            best_model = models[best_model_name]["model"]

            # Determine if the best model is XGBoost or another model
            if "xgb_classifier" in best_model.named_steps:
                # Handle XGBoost predictions within the pipeline
                final_predictions_proba = best_model.predict_proba(
                    df_test_processed.drop("SepsisLabel", axis=1)
                )[:, 1]
                final_predictions = (final_predictions_proba > 0.5).astype(int)
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

            # Evaluate
            metrics = evaluate_model(
                y_true=df_test_processed["SepsisLabel"],
                y_pred=final_predictions,
                model_name=f"Final_{best_model_name.replace(' ', '_').lower()}",
                y_pred_proba=final_predictions_proba,
                model=best_model,
                logger=logger,
            )

            # Generate model card
            generate_model_card(
                model=best_model,
                model_name=best_model_name,
                metrics=metrics,
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
            best_model_params = best_model.named_steps.get(estimator_step).get_params()

            model_registry.save_model(
                model=best_model,
                name=best_model_name,
                params=best_model_params,
                metrics=metrics,
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
                        unique_report_dir, f"{best_model_name}_feature_importance.png"
                    ),
                },
                tags=["tuned"],
            )
            logger.info(f"Model ({best_model_name}) saved successfully.")

            # Clean up
            del best_model
            gc.collect()

            logger.info("Sepsis Prediction Pipeline completed successfully.")

        else:
            logger.warning(
                "No models were trained. Pipeline did not complete successfully."
            )

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
