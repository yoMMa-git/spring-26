from typing import Optional
import time

from .load_data import load_data
from .process_data import prepare_features, introduce_missing_values, train_val_test_split


def run_data_pipeline(
    missing_rate: float = 0.05,
    random_seed: Optional[int] = None,
    return_split: bool = True,
    train_size: float = 0.6,
    val_size: float = 0.2,
    save_path: Optional[str] = None
):
    """
    Run all data preprocessing pipeline:
        1. loading data
        2. Preprocessing (encoding, scaling)
        3. Introducing missing values
        4. Splitting into train/val/test (if return_split=True)

    :param missing_rate: Proportion of missing values to introduce per column (default 0.05)
    :param random_seed: Random seed for reproducibility
    :param return_split: If True, return split arrays; else return full DataFrame
    :param train_size: Proportion for training set
    :param val_size: Proportion for validation set
    :param save_path: Optional path to save processed CSV

    :return: If return_split: X_train, X_val, X_test, y_train, y_val, y_test
            Else: processed DataFrame
    """
    print("Running data pipeline...")
    start_time = time.time()

    # Load raw data
    df = load_data()

    # Prepare features (encoding, scaling)
    df = prepare_features(df)

    # Introduce missing values
    df = introduce_missing_values(df, missing_rate=missing_rate, random_seed=random_seed)

    if save_path is not None:
        df.to_csv(save_path, index=False)

    if return_split:
        result = train_val_test_split(
            df, train_size=train_size, val_size=val_size, random_seed=random_seed
        )
    else:
        result = df

    print(f"Data pipeline finished in {time.time() - start_time:.2f} sec")
    return result
