import numpy as np
import pandas as pd


def inject_missing_values(
    df: pd.DataFrame,
    missing_ratio: float = 0.1,
    random_state: int | None = None
) -> pd.DataFrame:

    rng = np.random.default_rng(random_state)

    df_missing = df.copy()

    n_rows, n_cols = df_missing.shape
    total_values = n_rows * n_cols

    n_missing = int(total_values * missing_ratio)

    rows = rng.integers(0, n_rows, n_missing)
    cols = rng.integers(0, n_cols, n_missing)

    df_missing.values[rows, cols] = np.nan

    return df_missing
