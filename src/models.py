# src/models.py

import logging

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def train_random_forest(
    X_train, y_train, n_estimators=300, max_depth=20, random_state=42
):
    """
    Train a Random Forest classifier with improved parameters.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,  # Use all available cores
        max_features="sqrt",  # Standard RF practice
        min_samples_split=5,  # Prevent overfitting
        min_samples_leaf=2,  # Prevent overfitting
    )
    model.fit(X_train, y_train)
    return model


def train_naive_bayes(X_train, y_train):
    """
    Train a Gaussian Naive Bayes classifier with default parameters.
    """
    model = GaussianNB(
        var_smoothing=1e-9  # Default parameter
    )
    model.fit(X_train, y_train)
    return model


def train_knn(X_train, y_train, n_neighbors=5):
    """
    Train a K-Nearest Neighbors classifier with improved parameters.
    """
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights="distance",  # Weight points by distance
        algorithm="auto",  # Automatically choose best algorithm
        leaf_size=30,  # Default leaf size
        p=2,  # Euclidean distance
        n_jobs=-1,  # Use all available cores
        metric="minkowski",  # Standard distance metric
    )
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression classifier with improved parameters.
    """
    model = LogisticRegression(
        max_iter=1000,
        C=1.0,  # Inverse of regularization strength
        penalty="l2",  # Ridge regularization
        solver="lbfgs",  # Efficient solver
        tol=1e-4,  # Tolerance for stopping criteria
        random_state=42,
        n_jobs=-1,  # Use all available cores
        class_weight="balanced",  # Handle class imbalance
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, params=None, num_round=100, eval_set=None):
    """
    Train XGBoost model with early stopping if evaluation set is provided.

    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    params : dict, optional
        XGBoost parameters
    num_round : int, default=100
        Number of boosting rounds
    eval_set : list of (X, y) tuples, optional
        Validation set for early stopping

    Returns:
    --------
    xgb.Booster
        Trained XGBoost model
    """
    try:
        # Convert pandas DataFrame/Series to numpy arrays if necessary
        if hasattr(X_train, "values"):
            X_train = X_train.values
        if hasattr(y_train, "values"):
            y_train = y_train.values

        if params is None:
            params = {
                "max_depth": 6,
                "min_child_weight": 1,
                "eta": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "binary:logistic",
                "eval_metric": ["auc", "error"],
                "alpha": 1,
                "lambda": 1,
                "tree_method": "hist",
                "random_state": 42,
            }

        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)

        # Prepare evaluation set if provided
        evals = []
        if eval_set is not None:
            for i, (X_eval, y_eval) in enumerate(eval_set):
                if hasattr(X_eval, "values"):
                    X_eval = X_eval.values
                if hasattr(y_eval, "values"):
                    y_eval = y_eval.values
                deval = xgb.DMatrix(X_eval, label=y_eval)
                evals.append((deval, f"eval_{i}"))

        # Training parameters
        training_params = {
            "params": params,
            "dtrain": dtrain,
            "num_boost_round": num_round,
            "verbose_eval": 10,
        }

        # Add early stopping if evaluation set is provided
        if evals:
            training_params.update(
                {
                    "evals": evals,
                    "early_stopping_rounds": 10,
                }
            )

        # Train the model
        bst = xgb.train(**training_params)
        return bst

    except Exception as e:
        logging.error(f"Error in XGBoost training: {str(e)}")
        raise


def predict_xgboost(bst, X_test):
    """
    Make predictions using trained XGBoost model.

    Parameters:
    -----------
    bst : xgb.Booster
        Trained XGBoost model
    X_test : array-like
        Test features

    Returns:
    --------
    numpy.ndarray
        Binary predictions (0 or 1)
    """
    try:
        # Convert to numpy array if necessary
        if hasattr(X_test, "values"):
            X_test = X_test.values

        # Create DMatrix
        dtest = xgb.DMatrix(X_test)

        # Make predictions
        preds = bst.predict(dtest)
        return (preds >= 0.5).astype(int)

    except Exception as e:
        logging.error(f"Error in XGBoost prediction: {str(e)}")
        raise
