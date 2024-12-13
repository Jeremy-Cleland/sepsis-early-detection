import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from typing import List, Dict, Optional
import logging
from itertools import combinations


def plot_missing_values(
    data: pd.DataFrame,
    report_dir: str,
    figsize: tuple = (12, 8),
    model_name: str = "model",
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Create a heatmap visualization of missing values in the dataset.
    """
    try:
        plt.figure(figsize=figsize)

        # Calculate percentage of missing values
        missing_percentage = (data.isnull().mean() * 100).sort_values(ascending=False)

        # Create mask for values that are not missing
        mask = data.notna().values

        # Create heatmap
        sns.heatmap(
            data.isnull(), yticklabels=False, cbar=True, cmap="viridis", mask=mask
        )

        plt.title("Missing Values Heatmap")
        plt.xlabel("Features")
        plt.ylabel("Samples")

        # Add text annotation of percentage missing
        ax2 = plt.twinx()
        ax2.set_ylim(plt.ylim())
        ax2.set_ylabel("Percentage Missing")

        plt.tight_layout()
        plt.savefig(f"{report_dir}/{model_name}_missing_values_heatmap.png")
        plt.close()

        if logger:
            logger.info(f"Saved missing values heatmap to {report_dir}")

    except Exception as e:
        if logger:
            logger.error(f"Error plotting missing values heatmap: {str(e)}")
        raise


def plot_temporal_progression(
    data: pd.DataFrame,
    features: List[str],
    report_dir: str,
    model_name: str = "model",
    max_patients: int = 5,
    window_size: int = 5,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Plot temporal progression of vital signs for selected patients.
    """
    try:
        # Select random sepsis patients
        sepsis_patients = data[data["SepsisLabel"] == 1]["Patient_ID"].unique()
        selected_patients = np.random.choice(
            sepsis_patients, size=min(max_patients, len(sepsis_patients)), replace=False
        )

        for patient_id in selected_patients:
            plt.figure(figsize=(15, 8))
            patient_data = data[data["Patient_ID"] == patient_id].copy()

            # Apply rolling mean smoothing
            for feature in features:
                smoothed_values = (
                    patient_data[feature]
                    .rolling(window=window_size, center=True)
                    .mean()
                )
                plt.plot(patient_data["Hour"], smoothed_values, label=feature)

            # Mark sepsis onset
            sepsis_onset = patient_data[patient_data["SepsisLabel"] == 1]["Hour"].iloc[
                0
            ]
            plt.axvline(x=sepsis_onset, color="r", linestyle="--", label="Sepsis Onset")

            plt.title(f"Temporal Progression - Patient {patient_id}")
            plt.xlabel("Hours")
            plt.ylabel("Normalized Values")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(f"{report_dir}/{model_name}_temporal_patient_{patient_id}.png")
            plt.close()

        if logger:
            logger.info(
                f"Saved temporal progression plots for {len(selected_patients)} patients"
            )

    except Exception as e:
        if logger:
            logger.error(f"Error plotting temporal progression: {str(e)}")
        raise


def plot_error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    patient_groups: Dict[str, np.ndarray],
    report_dir: str,
    model_name: str = "model",
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Create error analysis plots broken down by patient groups.
    """
    try:
        metrics_by_group = {}

        # Calculate metrics for each group
        for group_name, group_mask in patient_groups.items():
            if len(y_true[group_mask]) > 0:  # Check if group has samples
                metrics_by_group[group_name] = {
                    "Accuracy": np.mean(y_true[group_mask] == y_pred[group_mask]),
                    "False Positives": np.sum(
                        (y_pred[group_mask] == 1) & (y_true[group_mask] == 0)
                    ),
                    "False Negatives": np.sum(
                        (y_pred[group_mask] == 0) & (y_true[group_mask] == 1)
                    ),
                    "Sample Size": np.sum(group_mask),
                }

        # Create visualization
        metrics_df = pd.DataFrame(metrics_by_group).T

        # Plot multiple metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Error Analysis by Patient Groups")

        # Accuracy
        sns.barplot(x=metrics_df.index, y=metrics_df["Accuracy"], ax=axes[0, 0])
        axes[0, 0].set_title("Accuracy by Group")
        axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45)

        # False Positives
        sns.barplot(x=metrics_df.index, y=metrics_df["False Positives"], ax=axes[0, 1])
        axes[0, 1].set_title("False Positives by Group")
        axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45)

        # False Negatives
        sns.barplot(x=metrics_df.index, y=metrics_df["False Negatives"], ax=axes[1, 0])
        axes[1, 0].set_title("False Negatives by Group")
        axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45)

        # Sample Size
        sns.barplot(x=metrics_df.index, y=metrics_df["Sample Size"], ax=axes[1, 1])
        axes[1, 1].set_title("Sample Size by Group")
        axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45)

        plt.tight_layout()
        plt.savefig(f"{report_dir}/{model_name}_error_analysis.png")
        plt.close()

        if logger:
            logger.info(f"Saved error analysis plots to {report_dir}")

    except Exception as e:
        if logger:
            logger.error(f"Error plotting error analysis: {str(e)}")
        raise


def plot_calibration(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    report_dir: str,
    model_name: str = "model",
    n_bins: int = 10,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Create calibration plot comparing predicted probabilities to observed frequencies.
    """
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)

        plt.figure(figsize=(8, 8))

        # Plot perfectly calibrated line
        plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly Calibrated")

        # Plot model calibration
        plt.plot(prob_pred, prob_true, marker="o", label="Model Calibration")

        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Observed Frequency")
        plt.title("Calibration Plot")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"{report_dir}/{model_name}_calibration.png")
        plt.close()

        if logger:
            logger.info(f"Saved calibration plot to {report_dir}")

    except Exception as e:
        if logger:
            logger.error(f"Error plotting calibration curve: {str(e)}")
        raise


def plot_prediction_timeline(
    data: pd.DataFrame,
    y_pred: np.ndarray,
    report_dir: str,
    model_name: str = "model",
    max_patients: int = 5,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Create timeline comparing predicted vs actual sepsis onset.
    """
    try:
        # Select random sepsis patients
        sepsis_patients = data[data["SepsisLabel"] == 1]["Patient_ID"].unique()
        selected_patients = np.random.choice(
            sepsis_patients, size=min(max_patients, len(sepsis_patients)), replace=False
        )

        for patient_id in selected_patients:
            plt.figure(figsize=(15, 5))

            patient_mask = data["Patient_ID"] == patient_id
            patient_data = data[patient_mask].copy()
            patient_preds = y_pred[patient_mask]

            # Plot actual sepsis label
            plt.plot(
                patient_data["Hour"],
                patient_data["SepsisLabel"],
                label="Actual",
                linewidth=2,
            )

            # Plot predictions
            plt.plot(patient_data["Hour"], patient_preds, label="Predicted", alpha=0.7)

            # Mark actual sepsis onset
            sepsis_onset = patient_data[patient_data["SepsisLabel"] == 1]["Hour"].iloc[
                0
            ]
            plt.axvline(x=sepsis_onset, color="r", linestyle="--", label="Actual Onset")

            # Mark predicted onset (first prediction above threshold)
            if np.any(patient_preds > 0.5):
                pred_onset = patient_data["Hour"].iloc[
                    np.where(patient_preds > 0.5)[0][0]
                ]
                plt.axvline(
                    x=pred_onset, color="g", linestyle="--", label="Predicted Onset"
                )

            plt.title(f"Prediction Timeline - Patient {patient_id}")
            plt.xlabel("Hours")
            plt.ylabel("Sepsis Probability")
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(f"{report_dir}/{model_name}_timeline_patient_{patient_id}.png")
            plt.close()

        if logger:
            logger.info(
                f"Saved prediction timeline plots for {len(selected_patients)} patients"
            )

    except Exception as e:
        if logger:
            logger.error(f"Error plotting prediction timeline: {str(e)}")
        raise


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
            logger.error(f"Error plotting feature interactions: {str(e)}")
        raise
