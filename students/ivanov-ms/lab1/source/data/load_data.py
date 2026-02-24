import kagglehub
import pandas as pd
import os

DATASET_NAME = 'kukuroo3/body-signal-of-smoking'
DATA_FILENAME = "smoking.csv"


def load_data_from_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df


def load_data() -> pd.DataFrame:
    """
    Download and load the Body Signal of Smoking dataset from Kaggle.

    Returns:
        pandas DataFrame with the dataset
    """
    path = kagglehub.dataset_download(DATASET_NAME)
    file_path = os.path.join(path, DATA_FILENAME)
    return load_data_from_csv(file_path)
