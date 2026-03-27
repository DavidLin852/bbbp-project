from pathlib import Path
import pandas as pd

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")

def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    df.to_csv(path, index=False)
