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
