from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")

COLUMNS = (
    ["unit", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"s_{i}" for i in range(1, 22)]
)


def load_split(name, subset):
    path = RAW_DIR / f"{name}_{subset}.txt"
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)


def load_rul(subset):
    path = RAW_DIR / f"RUL_{subset}.txt"
    return pd.read_csv(path, header=None, names=["rul"])


if __name__ == "__main__":
    train = load_split("train", "FD001")
    test = load_split("test", "FD001")
    rul = load_rul("FD001")

    print("train:", train.shape, "test:", test.shape, "rul:", rul.shape)
    print("Engines in train:", train["unit"].nunique())
    print(train.head())