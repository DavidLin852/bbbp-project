from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support, accuracy_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
)


@dataclass(frozen=True)
class ClsMetrics:
    auc: float
    auprc: float
    accuracy: float
    precision_pos: float
    recall_pos: float
    f1_pos: float
    tn: int
    fp: int
    fn: int
    tp: int


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> ClsMetrics:
    y_true = y_true.astype(int)
    y_pred = (y_prob >= threshold).astype(int)

    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    auprc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0,1])
    # p/r/f1 arrays: for class 0 and class 1
    precision_pos, recall_pos, f1_pos = float(p[1]), float(r[1]), float(f1[1])
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return ClsMetrics(
        auc=float(auc), auprc=float(auprc), accuracy=float(acc),
        precision_pos=precision_pos, recall_pos=recall_pos, f1_pos=f1_pos,
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)
    )


@dataclass(frozen=True)
class RegMetrics:
    """Metrics for regression tasks."""
    mse: float
    rmse: float
    mae: float
    r2: float


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegMetrics:
    """Compute regression metrics."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return RegMetrics(
        mse=float(mse),
        rmse=float(rmse),
        mae=float(mae),
        r2=float(r2),
    )
