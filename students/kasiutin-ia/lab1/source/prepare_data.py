import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


NUMERIC_FEATURES = [
    "Pressure9am", "Pressure3pm", "MaxTemp", "MinTemp", "Temp9am", "Temp3pm",
    "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Rainfall",
    "Humidity9am", "Humidity3pm", "Evaporation", "Sunshine", "Cloud9am", "Cloud3pm",
]
CATEGORICAL_FEATURES = ["Location", "WindGustDir", "WindDir9am", "WindDir3pm"]
BINARY_FEATURES = ["RainToday"]
TARGET = "RainTomorrow"
DROP_COLUMNS = ["Date"]


def load_and_prepare(
    csv_path: str = None,
    test_size: float = 0.2,
    random_state: int = 42,
    normalize: bool = True,
):
    """
    Класс для подготовки данных для обучения и оценки модели.
    Отдельно предобрабатываются числовые и категориальные признаки и делается train/test split.
    """
    df = pd.read_csv(csv_path)
    df = df[~df[TARGET].isna()].copy()

    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0}).astype(int)
    for col in BINARY_FEATURES:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})

    for col in DROP_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])

    use_numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
    use_cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    use_binary = [c for c in BINARY_FEATURES if c in df.columns]
    feature_order = use_numeric + use_cat + use_binary

    X = df[feature_order].copy()
    y = df[TARGET].values

    # Label encoding для категориальных (сохраняем NaN как отдельный класс)
    label_encoders = {}
    for col in use_cat:
        le = LabelEncoder()
        vals = X[col].astype(str).fillna("__NA__")
        le.fit(vals)
        X[col] = le.transform(vals)
        # Вернём NaN обратно для обработки пропусков в дереве
        X.loc[df[col].isna(), col] = np.nan
        label_encoders[col] = le

    feature_types = {}
    for c in use_numeric:
        feature_types[c] = "numeric"
    for c in use_cat:
        feature_types[c] = "categorical"
    for c in use_binary:
        feature_types[c] = "numeric"  # работаем как с числовыми признаками

    scaler = None
    if normalize:
        scaler = MinMaxScaler()
        num_cols = use_numeric + use_binary
        X_fill = X[num_cols].fillna(X[num_cols].median())
        scaler.fit(X_fill)
        T = scaler.transform(X_fill)
        for i, c in enumerate(num_cols):
            not_na = X[c].notna()
            X.loc[not_na, c] = T[not_na.values, i]

    X = X[feature_order]
    feature_metadata = {
        "feature_order": feature_order,
        "feature_types": feature_types,
        "numeric_cols": use_numeric + use_binary,
        "categorical_cols": use_cat,
        "label_encoders": label_encoders,
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, feature_metadata, label_encoders
