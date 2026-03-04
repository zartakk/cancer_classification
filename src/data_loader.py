import os
import pandas as pd
from sklearn.datasets import load_breast_cancer

# Get absolute path to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "breast_cancer.csv")


def export_dataset():
    """
    Load dataset from sklearn and save it to data/raw as CSV.
    Runs only if CSV does not already exist.
    """

    data = load_breast_cancer()

   

    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    df = pd.concat([X, y], axis=1)

    # df = data.frame

    os.makedirs("data/raw", exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)

    print("Dataset saved to data/raw/breast_cancer.csv")


def load_data():
    """
    Load dataset from CSV.
    If CSV does not exist, export it first.
    Returns:
        df (pd.DataFrame)
    """

    if not os.path.exists(RAW_DATA_PATH):
        export_dataset()

    df = pd.read_csv(RAW_DATA_PATH)

    print(f"Dataset loaded. Shape: {df.shape}")

    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head())