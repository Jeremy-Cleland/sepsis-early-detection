import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file with error handling and validation.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at: {filepath}")

    # Load the data
    df = pd.read_csv(filepath)

    # Validate required columns
    required_columns = ["Patient_ID", "SepsisLabel"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Convert Patient_ID to string - using inplace operation
    df = df.assign(Patient_ID=df["Patient_ID"].astype(str))

    # Basic data validation
    if df["SepsisLabel"].nunique() > 2:
        raise ValueError("SepsisLabel contains more than two unique values")

    logging.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    return df


def split_data(
    combined_df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    random_state: int = 42,
    stratify_by_sepsis: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the combined data into training, validation, and testing datasets.
    """
    # Input validation
    if not 0 < train_size + val_size < 1:
        raise ValueError("train_size + val_size must be between 0 and 1")

    if not isinstance(combined_df, pd.DataFrame):
        raise ValueError("combined_df must be a pandas DataFrame")

    # Ensure Patient_ID is string type before splitting
    combined_df = combined_df.assign(Patient_ID=combined_df["Patient_ID"].astype(str))

    # Get unique Patient_IDs and their characteristics
    patient_stats = (
        combined_df.groupby("Patient_ID")
        .agg(
            {
                "SepsisLabel": "max",  # 1 if patient ever had sepsis
                "Hour": "count",  # number of measurements per patient
            }
        )
        .reset_index()
    )

    # Set random seed for reproducibility
    np.random.seed(random_state)

    if stratify_by_sepsis:
        # Split separately for sepsis and non-sepsis patients
        sepsis_patients = patient_stats[patient_stats["SepsisLabel"] == 1]["Patient_ID"]
        non_sepsis_patients = patient_stats[patient_stats["SepsisLabel"] == 0][
            "Patient_ID"
        ]

        def split_patient_ids(patient_ids):
            n_train = int(len(patient_ids) * train_size)
            n_val = int(len(patient_ids) * val_size)
            shuffled = np.random.permutation(patient_ids)
            return (
                shuffled[:n_train],
                shuffled[n_train : n_train + n_val],
                shuffled[n_train + n_val :],
            )

        # Split both groups
        train_sepsis, val_sepsis, test_sepsis = split_patient_ids(sepsis_patients)
        train_non_sepsis, val_non_sepsis, test_non_sepsis = split_patient_ids(
            non_sepsis_patients
        )

        # Combine splits
        train_patients = np.concatenate([train_sepsis, train_non_sepsis])
        val_patients = np.concatenate([val_sepsis, val_non_sepsis])
        test_patients = np.concatenate([test_sepsis, test_non_sepsis])

    else:
        # Simple random split without stratification
        shuffled_patients = np.random.permutation(patient_stats["Patient_ID"])
        n_train = int(len(shuffled_patients) * train_size)
        n_val = int(len(shuffled_patients) * val_size)

        train_patients = shuffled_patients[:n_train]
        val_patients = shuffled_patients[n_train : n_train + n_val]
        test_patients = shuffled_patients[n_train + n_val :]

    # Create the splits - using copy() to ensure we have independent DataFrames
    df_train = combined_df[combined_df["Patient_ID"].isin(train_patients)].copy()
    df_val = combined_df[combined_df["Patient_ID"].isin(val_patients)].copy()
    df_test = combined_df[combined_df["Patient_ID"].isin(test_patients)].copy()

    # Verify no patient overlap
    assert (
        len(set(train_patients) & set(val_patients)) == 0
    ), "Patient overlap between train and val"
    assert (
        len(set(train_patients) & set(test_patients)) == 0
    ), "Patient overlap between train and test"
    assert (
        len(set(val_patients) & set(test_patients)) == 0
    ), "Patient overlap between val and test"

    # Log split information
    logging.info("\nData Split Summary:")
    logging.info("-" * 50)
    logging.info(
        f"Training set:   {len(df_train)} rows, {len(train_patients)} unique patients"
    )
    logging.info(
        f"Validation set: {len(df_val)} rows, {len(val_patients)} unique patients"
    )
    logging.info(
        f"Testing set:    {len(df_test)} rows, {len(test_patients)} unique patients"
    )

    # Log sepsis distribution
    for name, df in [
        ("Training", df_train),
        ("Validation", df_val),
        ("Testing", df_test),
    ]:
        sepsis_rate = df.groupby("Patient_ID")["SepsisLabel"].max().mean()
        logging.info(f"{name} set sepsis rate: {sepsis_rate:.1%}")

    # Save the split datasets
    os.makedirs("data/processed", exist_ok=True)
    df_train.to_csv("data/processed/train_data.csv", index=False)
    df_val.to_csv("data/processed/val_data.csv", index=False)
    df_test.to_csv("data/processed/test_data.csv", index=False)

    return df_train, df_val, df_test


def load_processed_data(
    train_path: str, val_path: str, test_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load processed training, validation, and testing data with validation.
    """
    # Check file existence
    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found at: {path}")

    # Load datasets
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # Validate columns match across datasets
    if not (set(df_train.columns) == set(df_val.columns) == set(df_test.columns)):
        raise ValueError("Column mismatch between datasets")

    # Convert Patient_ID to string in all datasets using assign
    dfs = [df_train, df_val, df_test]
    for i in range(len(dfs)):
        dfs[i] = dfs[i].assign(Patient_ID=dfs[i]["Patient_ID"].astype(str))
    df_train, df_val, df_test = dfs

    # Verify no patient overlap
    train_patients = set(df_train["Patient_ID"])
    val_patients = set(df_val["Patient_ID"])
    test_patients = set(df_test["Patient_ID"])

    if train_patients & val_patients:
        raise ValueError("Patient overlap between train and validation sets")
    if train_patients & test_patients:
        raise ValueError("Patient overlap between train and test sets")
    if val_patients & test_patients:
        raise ValueError("Patient overlap between validation and test sets")

    return df_train, df_val, df_test


def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validate the dataset format and content.
    """
    # Check required columns
    required_columns = ["Patient_ID", "Hour", "SepsisLabel"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Validate data types
    if not pd.api.types.is_numeric_dtype(df["Hour"]):
        raise ValueError("Hour column must be numeric")
    if not pd.api.types.is_numeric_dtype(df["SepsisLabel"]):
        raise ValueError("SepsisLabel column must be numeric")

    # Validate value ranges
    if df["Hour"].min() < 0:
        raise ValueError("Hour column contains negative values")
    if not set(df["SepsisLabel"].unique()).issubset({0, 1}):
        raise ValueError("SepsisLabel must contain only 0 and 1")

    # Check for duplicates
    duplicates = df.groupby(["Patient_ID", "Hour"]).size()
    if (duplicates > 1).any():
        raise ValueError("Found duplicate time points for some patients")
