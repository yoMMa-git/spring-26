import numpy as np
import pandas as pd


def probabilistic_impute(df: pd.DataFrame, random_state: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    df_imputed = df.copy()

    for col in df_imputed.columns:
        mask_nan = df_imputed[col].isna()
        n_missing = mask_nan.sum()
        if n_missing == 0:
            continue

        values, counts = np.unique(df_imputed[col].dropna(), return_counts=True)
        probs = counts / counts.sum()

        sampled_values = rng.choice(values, size=n_missing, p=probs)
        df_imputed.loc[mask_nan, col] = sampled_values

    return df_imputed
