import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =========================
# CONFIG
# =========================
SEED = 0
SPLIT = "test"
TOP_POS = 8
TOP_NEG = 10
MIN_FREQ = 20

BASE = Path("artifacts/metrics/global_explain")
OUT = Path("artifacts/figures/fig3_global_smarts_importance.png")

# =========================
# LOAD
# =========================
pos = pd.read_csv(BASE / f"global_smarts_positive_{SPLIT}_seed{SEED}.csv")
neg = pd.read_csv(BASE / f"global_smarts_negative_{SPLIT}_seed{SEED}.csv")

# safety filter (should already be filtered in step-10)
pos = pos[pos["freq_mols"] >= MIN_FREQ].head(TOP_POS)
neg = neg[neg["freq_mols"] >= MIN_FREQ].head(TOP_NEG)

df = pd.concat([pos, neg], ignore_index=True)

# sort: positive → negative (top to bottom)
df = df.sort_values("mean_delta", ascending=False).reset_index(drop=True)

df_plot = df.set_index("smarts_name")
df_heat = df_plot[["mean_delta"]]
freq = df_plot["freq_mols"]

# =========================
# PLOT
# =========================
sns.set(style="white")
fig, (ax_hm, ax_bar) = plt.subplots(
    ncols=2,
    figsize=(10, 1 * len(df_plot)),
    gridspec_kw={"width_ratios": [3, 1]},
    sharey=True
)

# ---- Heatmap ----
sns.heatmap(
    df_heat,
    ax=ax_hm,
    cmap="RdBu_r",
    center=0.0,
    linewidths=0.5,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 11},
    cbar_kws={"label": "Mean Δ P(BBB+)  (Occlusion)"}
)

ax_hm.set_xlabel("")
ax_hm.set_ylabel("SMARTS Substructure", fontsize=12, fontname="Times New Roman")
ax_hm.set_title(
    "Global Substructure Contributions to BBB Permeability",
    fontsize=14,
    fontname="Times New Roman",
    pad=12
)

# ---- Frequency bar ----
y_pos = range(len(freq))
ax_bar.barh(
    y=y_pos,
    width=freq.values,
    color="gray",
    alpha=0.8
)

ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(freq.index)
ax_bar.invert_xaxis()
ax_bar.set_xlabel("Frequency\n(# Molecules)", fontsize=11, fontname="Times New Roman")
ax_bar.grid(axis="x", linestyle="--", alpha=0.4)

# ---- Font control ----
for ax in (ax_hm, ax_bar):
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontname("Times New Roman")

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=300)
plt.close()

print("Saved:", OUT)
