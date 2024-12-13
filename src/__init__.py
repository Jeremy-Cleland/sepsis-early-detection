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
from .models import (
    predict_xgboost,
    train_knn,
    train_logistic_regression,
    train_naive_bayes,
    train_random_forest,
    train_xgboost,
)
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
    # Models
    "train_random_forest",
    "train_naive_bayes",
    "train_knn",
    "train_logistic_regression",
    "train_xgboost",
    "predict_xgboost",
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
    # Remove unused functions
    # "corr_matrix",
    # "diagnostic_plots",
    # "try_gaussian",
    # "setup_logger" (duplicate),
    # "save_model",
]
