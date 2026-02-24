import numpy as np
import pandas as pd
from typing import Optional, Tuple


class StandardScaler:
    def __init__(self):
        self._mean = None
        self._std = None

    def fit(self, X: np.ndarray):
        self._mean = X.mean(axis=0, keepdims=True)
        self._std = X.std(axis=0, keepdims=True)

    def transform(self, X: np.ndarray):
        if self._mean is None or self._std is None:
            raise ValueError("StandardScaler wasn't fitted")
        return (X - self._mean) / self._std

    def fit_transform(self, X: np.ndarray):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray):
        if self._mean is None or self._std is None:
            raise ValueError("StandardScaler wasn't fitted")
        return X * self._std + self._mean


def introduce_missing_values(
        df: pd.DataFrame, missing_rate: float = 0.05, random_seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Randomly introduce missing values (NaN) in each column except the target.
    Args:
        df: Input DataFrame
        missing_rate: Proportion of values to set to NaN in each column (0-1)
        random_seed: Random seed for reproducibility
    Returns:
        DataFrame with missing values introduced
    """
    if missing_rate <= 0:
        return df

    rng = np.random.default_rng(random_seed)
    df_missing = df.copy()

    # For each column except 'target' and 'ID' (if present), randomly set missing_rate to NaN
    for col in df_missing.columns:
        if col.lower() in ['target', 'smoking', 'id']:
            continue

        # Skip if column is already all NaN
        if df_missing[col].isna().all():
            continue

        # Get indices of non-missing values
        non_missing_idx = df_missing.index[df_missing[col].notna()].tolist()
        n_to_missing = int(len(non_missing_idx) * missing_rate)

        if n_to_missing > 0:
            # Randomly select indices
            chosen_idx = rng.choice(non_missing_idx, size=n_to_missing, replace=False)
            df_missing.loc[chosen_idx, col] = np.nan

    return df_missing


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for the smoking dataset.
    - Convert target to -1/1
    - One-hot encode categorical variables
    - Scale numerical variables
    - Handle remaining object columns

    Args:
        df: Raw DataFrame

    Returns:
        Processed DataFrame with all numeric features
    """
    df = df.copy()

    # Identify target column (smoking)
    target_col = 'smoking' if 'smoking' in df.columns else ' Smoking'

    if target_col not in df.columns:
        raise ValueError(f"Target column 'smoking' not found. Available columns: {list(df.columns)}")

    # Clean column names (strip spaces)
    df.columns = df.columns.str.strip()

    # Convert target: assuming 0/1 -> -1/1
    target_original = df[target_col].copy()
    if set(np.unique(target_original)).issubset({0, 1}):
        df[target_col] = df[target_col].replace({0: -1, 1: 1})
    else:
        # If already other values, convert: 0->-1, 1->1, others keep but flag
        unique_vals = np.unique(target_original)
        print(f"Warning: Target values are {unique_vals}, expected 0/1")
        df[target_col] = df[target_col].apply(lambda x: -1 if x == 0 else 1)

    # Rename to 'target' for consistency
    df = df.rename(columns={target_col: 'target'})

    # Separate target from features
    y = df['target']
    X = df.drop(columns=['target'])

    # Identify categorical and numerical columns
    categorical_cols = []
    numerical_cols = []

    for col in X.columns:
        # Skip ID-like columns
        if col.lower() in ['id']:
            continue

        dtype = X[col].dtype

        if dtype == 'object' or dtype == 'str' or dtype.name == 'category':
            categorical_cols.append(col)
        elif dtype in ['int64', 'float64']:
            # Check cardinality for low-cardinality integers (might be categorical)
            unique_count = X[col].nunique()
            if unique_count <= 10 and dtype in ['int64']:
                # Treat as categorical
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        else:
            # Default to numerical if unclear
            numerical_cols.append(col)

    print(f"Detected {len(categorical_cols)} categorical columns: {categorical_cols}")
    print(f"Detected {len(numerical_cols)} numerical columns: {numerical_cols}")

    # One-hot encode categorical variables
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=False, dtype=float)

    # Scale numerical features using StandardScaler
    if numerical_cols:
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols].to_numpy())

    # Convert boolean to int
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)

    # Combine with target
    X['target'] = y

    return X


def train_val_test_split(
        df: pd.DataFrame, train_size: float = 0.6,
        val_size: float = 0.2, random_seed: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train, validation, and test sets with stratification.

    Args:
        df: DataFrame with 'target' column
        train_size: Proportion of data for training
        val_size: Proportion of data for validation
        random_seed: Random seed

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test as numpy arrays
    """
    rng = np.random.default_rng(random_seed)

    # Stratify by target
    targets_probs = df['target'].value_counts(normalize=True)
    probs = df['target'].map(targets_probs)
    probs /= probs.sum()

    rnd_indexes = rng.choice(df.shape[0], df.shape[0], replace=False, p=probs.to_numpy())

    # Compute splits
    n_total = df.shape[0]
    n_train = round(n_total * train_size)
    n_val = round(n_total * val_size)

    features_arr = df.drop('target', axis=1).to_numpy()
    target_arr = df['target'].to_numpy()

    X_train = features_arr[rnd_indexes[:n_train]]
    y_train = target_arr[rnd_indexes[:n_train]]

    X_val = features_arr[rnd_indexes[n_train:n_train + n_val]]
    y_val = target_arr[rnd_indexes[n_train:n_train + n_val]]

    X_test = features_arr[rnd_indexes[n_train + n_val:]]
    y_test = target_arr[rnd_indexes[n_train + n_val:]]

    columns = [col for col in df.columns if col != 'target']
    X_train = pd.DataFrame(data=X_train, columns=columns)
    X_test = pd.DataFrame(data=X_test, columns=columns)
    X_val = pd.DataFrame(data=X_val, columns=columns)

    return X_train, X_val, X_test, y_train, y_val, y_test
