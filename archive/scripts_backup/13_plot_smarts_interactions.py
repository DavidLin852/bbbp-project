import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =========================
# CONFIG
# =========================
SEED = 0
SPLIT = "test"
TOPK = 10

BASE = Path("artifacts/metrics/global_explain")
INP = BASE / f"global_smarts_interactions_{SPLIT}_seed{SEED}.csv"

OUT_HM = Path("artifacts/figures/fig4_smarts_interaction_heatmap.png")
OUT_BAR = Path("artifacts/figures/fig5_top_smarts_interactions.png")

# =========================
# LOAD
# =========================
df = pd.read_csv(INP)

# keep top by magnitude
df_top = df.sort_values("mean_abs_interaction", ascending=False).head(TOPK)

# =========================
# 1. Interaction Heatmap
# =========================
# build symmetric matrix
labels = sorted(set(df_top["A"]).union(set(df_top["B"])))
mat = pd.DataFrame(index=labels, columns=labels, dtype=float)


for _, r in df_top.iterrows():
    a, b, v = r["A"], r["B"], r["mean_interaction"]
    mat.loc[a, b] = v
    mat.loc[b, a] = v

plt.figure(figsize=(0.7 * len(labels), 0.7 * len(labels)))
sns.heatmap(
    mat,
    cmap="RdBu_r",
    center=0.0,
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    mask=mat.isna(),   # <<< 关键
    cbar_kws={"label": "Interaction  (Synergy > 0, Red = stronger suppression of BBB)"}
)


plt.title(
    "Substructure Interaction Heatmap (Synergistic vs Redundant Effects)",
    fontsize=14,
    fontname="Times New Roman",
    pad=12
)

plt.xticks(rotation=45, ha="right", fontname="Times New Roman")
plt.yticks(fontname="Times New Roman")
plt.tight_layout()
OUT_HM.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_HM, dpi=300)
plt.close()
print("Saved:", OUT_HM)

# =========================
# 2. Top-K Interaction Barplot
# =========================
df_bar = df_top.copy()
df_bar["pair"] = df_bar["A"] + " + " + df_bar["B"]
df_bar = df_bar.sort_values("mean_interaction", ascending=True)

plt.figure(figsize=(8, 0.5 * len(df_bar)))
sns.barplot(
    x="mean_interaction",
    y="pair",
    data=df_bar,
    hue="pair",        # 关键
    palette="RdBu_r",
    legend=False
)


plt.axvline(0, color="black", linewidth=1)
plt.xlabel("Mean Interaction  ( >0 = Synergistic Suppression of BBB )", fontsize=12, fontname="Times New Roman")
plt.ylabel("Substructure Pair", fontsize=12, fontname="Times New Roman")
plt.title(
    "Top Synergistic and Redundant Substructure Interactions",
    fontsize=14,
    fontname="Times New Roman",
    pad=12
)

plt.xticks(fontname="Times New Roman")
plt.yticks(fontname="Times New Roman")
plt.tight_layout()
plt.savefig(OUT_BAR, dpi=300)
plt.close()

print("Saved:", OUT_BAR)
