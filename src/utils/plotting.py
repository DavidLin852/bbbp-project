from __future__ import annotations
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def set_times_new_roman():
    # Try to enforce Times New Roman; if not available, matplotlib will fallback.
    matplotlib.rcParams["font.family"] = "Times New Roman"
    matplotlib.rcParams["font.size"] = 14

def plot_roc_curves(items, out_path: Path, title: str = "ROC"):
    """
    items: [{"name": str, "y_true": np.ndarray, "y_prob": np.ndarray}, ...]
    Enforce Times New Roman font.
    """
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14

    fig = plt.figure(figsize=(6.2, 5.2))
    ax = fig.add_subplot(111)

    for it in items:
        y_true = it["y_true"]
        y_prob = it["y_prob"]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{it["name"]} (AUC={roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
