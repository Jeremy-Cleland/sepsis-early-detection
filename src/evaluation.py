import json
import logging
import os
from typing import Any, Dict, Optional


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    report_dir: str = "reports/evaluations",
    y_pred_proba: Optional[np.ndarray] = None,
    model: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, float]:
    """
    Comprehensive model evaluation function that calculates metrics and generates essential visualization plots.
    """
    try:
        setup_plot_style()
        os.makedirs(report_dir, exist_ok=True)

        # Calculate basic metrics
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1 Score": f1_score(y_true, y_pred, zero_division=0),
            "Mean Absolute Error": mean_absolute_error(y_true, y_pred),
            "Root Mean Squared Error": np.sqrt(mean_squared_error(y_true, y_pred)),
        }

        # Calculate AUC-ROC if probabilities are available
        if y_pred_proba is not None:
            try:
                metrics["AUC-ROC"] = roc_auc_score(y_true, y_pred_proba)
            except ValueError as e:
                logger.warning(f"Cannot calculate AUC-ROC: {e}")
                metrics["AUC-ROC"] = None

        # Log metrics
        log_metrics(logger, model_name, metrics)

        # Generate and save essential plots
        generate_evaluation_plots(
            y_true=y_true,
            y_pred=y_pred,
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


def generate_evaluation_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray],
    model: Any,
    model_name: str,
    report_dir: str,
    logger: Optional[logging.Logger],
) -> None:
    """Generate essential evaluation plots for the model."""
    try:
        # Confusion Matrix (Raw and Normalized)
        plot_confusion_matrix(
            y_true, y_pred, model_name, report_dir, normalize=False, logger=logger
        )
        plot_confusion_matrix(
            y_true, y_pred, model_name, report_dir, normalize=True, logger=logger
        )

        # ROC and Precision-Recall Curves
        if y_pred_proba is not None:
            plot_roc_curve(y_true, y_pred_proba, model_name, report_dir, logger=logger)
            plot_precision_recall_curve(
                y_true, y_pred_proba, model_name, report_dir, logger=logger
            )

        # Feature Importance
        if model is not None:
            plot_feature_importance(model, model_name, report_dir, logger=logger)

    except Exception as e:
        if logger:
            logger.error(f"Error generating plots: {str(e)}", exc_info=True)
        raise
    finally:
        plt.close("all")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
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
            logger.error(f"Error plotting confusion matrix: {str(e)}", exc_info=True)
        raise
    finally:
        plt.close()


def plot_class_distribution(
    y: np.ndarray,
    model_name: str,
    report_dir: str,
    title_suffix: str = "",
    logger: Optional[logging.Logger] = None,
) -> None:
    """Plot and save the class distribution."""
    try:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=y, hue=y, palette="viridis", legend=False)
        plt.title(f"Class Distribution for {model_name} {title_suffix}")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.tight_layout()

        suffix = title_suffix.replace(" ", "_") if title_suffix else ""
        save_plot(
            plt, report_dir, f"{model_name}_class_distribution_{suffix}.png", logger
        )

    except Exception as e:
        if logger:
            logger.error(f"Error plotting class distribution: {str(e)}", exc_info=True)
        raise
    finally:
        plt.close()


def plot_feature_importance(
    model: Any,
    model_name: str,
    report_dir: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Plot and save feature importance."""
    try:
        plt.figure(figsize=(10, 8))

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_names = (
                model.get_booster().feature_names
                if hasattr(model, "get_booster")
                else [f"Feature {i}" for i in range(len(importances))]
            )

            feature_importance_df = pd.DataFrame(
                {"feature": feature_names, "importance": importances}
            ).sort_values(by="importance", ascending=False)

            # Modified barplot code to use hue instead of palette
            sns.barplot(
                x="importance",
                y="feature",
                data=feature_importance_df.head(20),
                hue="feature",  # Add hue parameter
                legend=False,  # Hide the legend since we don't need it
                dodge=False,
            )

        elif isinstance(model, xgb.Booster):
            importances = model.get_score(importance_type="weight")
            importance_df = pd.DataFrame(
                {
                    "feature": list(importances.keys()),
                    "importance": list(importances.values()),
                }
            ).sort_values("importance", ascending=False)

            # Modified barplot code for XGBoost case
            sns.barplot(
                x="importance",
                y="feature",
                data=importance_df.head(20),
                hue="feature",  # Add hue parameter
                legend=False,  # Hide the legend
                dodge=False,
            )
        else:
            raise AttributeError("Model does not have feature_importances_ attribute.")

        plt.title(f"Top 20 Feature Importances for {model_name}")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()

        save_plot(plt, report_dir, f"{model_name}_feature_importance.png", logger)

    except Exception as e:
        if logger:
            logger.error(f"Error plotting feature importance: {str(e)}", exc_info=True)
    finally:
        plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
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
        plt.tight_layout()

        save_plot(plt, report_dir, f"{model_name}_roc_curve.png", logger)

    except Exception as e:
        if logger:
            logger.error(f"Error plotting ROC curve: {str(e)}", exc_info=True)
        raise
    finally:
        plt.close()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
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
        plt.tight_layout()

        save_plot(plt, report_dir, f"{model_name}_precision_recall_curve.png", logger)

    except Exception as e:
        if logger:
            logger.error(
                f"Error plotting precision-recall curve: {str(e)}", exc_info=True
            )
        raise
    finally:
        plt.close()


def plot_feature_correlation_heatmap(
    df: pd.DataFrame,
    model_name: str,
    report_dir: str,
    top_n: int = 20,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Plot and save the feature correlation heatmap."""
    plt.figure(figsize=(12, 10))
    corr = df.corr().abs()
    # Select upper triangle
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    # Find features with correlation greater than a threshold
    threshold = 0.6
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Compute correlation matrix of selected features
    corr_selected = corr.drop(columns=to_drop).drop(index=to_drop)

    sns.heatmap(corr_selected, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(f"Feature Correlation Heatmap for {model_name}")
    plt.tight_layout()

    filename = f"{model_name.replace(' ', '_').lower()}_feature_correlation_heatmap.png"
    save_plot(plt, report_dir, filename, logger=logger)


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
