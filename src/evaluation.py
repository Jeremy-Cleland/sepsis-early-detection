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
        # Log current columns
        if logger:
            logger.debug(
                f"Data columns before plotting interactions: {data.columns.tolist()}"
            )
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
        # vital_features = ["HR", "O2Sat", "Temp", "MAP", "Resp"]

        # # Extract Patient_IDs, assuming same index as y_true
        # patient_ids = data["Patient_ID"]

        plot_temporal_progression(
            data=data,  # Contains 'SepsisLabel'
            patient_ids=data["Patient_ID"],  # Assuming 'Patient_ID' exists
            features=["HR", "O2Sat", "Temp", "MAP", "Resp"],  # Example features
            report_dir=report_dir,
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
        if "SepsisLabel" not in data.columns:
            raise KeyError("'SepsisLabel' column is missing from the data.")

        sepsis_patients = patient_ids[data["SepsisLabel"] == 1].unique()

        if len(sepsis_patients) == 0:
            logger.warning("No sepsis patients found for temporal progression plots.")
            return

        selected_patients = np.random.choice(
            sepsis_patients,
            size=min(max_patients, len(sepsis_patients)),
            replace=False,
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
        # Verify that all specified features are in the data
        missing_features = set(features) - set(data.columns)
        if missing_features:
            if logger:
                logger.error(
                    f"The following features are missing in the data: {missing_features}"
                )
            raise KeyError(f"Missing features: {missing_features}")

        # Calculate correlation matrix
        corr_matrix = data[features].corr().abs()

        # Get top correlated pairs
        pairs = []
        for i, j in combinations(range(len(features)), 2):
            pairs.append((features[i], features[j], corr_matrix.iloc[i, j]))

        top_pairs = sorted(pairs, key=lambda x: x[2], reverse=True)[:top_n]

        # Ensure columns are unique
        for feat1, feat2, corr in top_pairs:
            if feat1 == feat2:
                logger.warning(f"Skipping identical features: {feat1}")
                continue

        # Create plots for top pairs
        for feat1, feat2, corr in top_pairs:
            # Skip identical features
            if feat1 == feat2:
                continue

            plt.figure(figsize=(10, 8))

            # Subset data with required columns
            subset = data[[feat1, feat2, "SepsisLabel"]].dropna()

            # Debugging: Log the subset shape and data types
            if logger:
                logger.debug(
                    f"Plotting feature interaction for: {feat1} vs {feat2}. Subset shape: {subset.shape}"
                )
                for col in subset.columns:
                    logger.debug(f"Column '{col}' dtype: {subset.dtypes[col]}")
                    # Sample data for object dtype columns
                    if subset[col].dtype == "object":
                        logger.debug(f"Sample data from '{col}': {subset[col].head()}")

            # Check for non-scalar data
            non_scalar = subset.apply(
                lambda col: col.map(lambda x: isinstance(x, (list, np.ndarray))).any()
            )
            if non_scalar.any():
                problematic_cols = non_scalar[non_scalar].index.tolist()
                logger.warning(
                    f"Skipping plot for {feat1} vs {feat2} due to non-scalar data in columns: {problematic_cols}"
                )
                plt.close()
                continue  # Skip plotting this pair

            # Plot scatterplot
            sns.scatterplot(
                data=subset,
                x=feat1,
                y=feat2,
                hue="SepsisLabel",
                alpha=0.5,
            )

            plt.title(
                f"Feature Interaction: {feat1} vs {feat2}\nCorrelation: {corr:.2f}"
            )

            # Add regression lines for each class
            sns.regplot(
                data=subset[subset["SepsisLabel"] == 0],
                x=feat1,
                y=feat2,
                scatter=False,
                label="Non-Sepsis Trend",
            )
            sns.regplot(
                data=subset[subset["SepsisLabel"] == 1],
                x=feat1,
                y=feat2,
                scatter=False,
                label="Sepsis Trend",
            )

            plt.legend()
            plt.tight_layout()
            filename = f"{model_name}_interaction_{feat1}_{feat2}.png"
            save_plot(plt, report_dir, filename, logger)

        if logger:
            logger.info(f"Saved feature interaction plots for top {top_n} pairs")

    except Exception as e:
        if logger:
            logger.error(f"Error plotting feature interactions: {e}", exc_info=True)
        raise
    finally:
        plt.close("all")


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
