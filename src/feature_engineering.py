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
        max_iter=30,  # Maximum number of iterations.
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
