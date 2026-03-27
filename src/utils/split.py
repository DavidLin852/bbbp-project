from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

@dataclass(frozen=True)
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

def stratified_train_val_test(
    df: pd.DataFrame,
    label_col: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> SplitResult:
    # sanity
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1. Got {total}")

    y = df[label_col].to_numpy()
    idx = np.arange(len(df))

    # 1) split train vs temp
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(1 - train_ratio), random_state=seed)
    train_idx, temp_idx = next(sss1.split(idx, y))

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_temp = df.iloc[temp_idx].reset_index(drop=True)

    # 2) split temp into val/test
    # temp size = val+test; val fraction within temp:
    val_frac = val_ratio / (val_ratio + test_ratio)
    y_temp = df_temp[label_col].to_numpy()
    idx_temp = np.arange(len(df_temp))
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=(1 - val_frac), random_state=seed)
    val_idx, test_idx = next(sss2.split(idx_temp, y_temp))

    df_val = df_temp.iloc[val_idx].reset_index(drop=True)
    df_test = df_temp.iloc[test_idx].reset_index(drop=True)
    return SplitResult(train=df_train, val=df_val, test=df_test)
