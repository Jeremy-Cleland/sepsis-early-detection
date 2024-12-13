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

        # Update registry with file locking
        sanitized_name = self._sanitize_model_name(name)
        self.registry.setdefault(sanitized_name, []).append(metadata)
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
