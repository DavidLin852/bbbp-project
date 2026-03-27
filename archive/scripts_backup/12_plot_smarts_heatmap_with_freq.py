import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ================= CONFIG =================
CSV_POS = Path("artifacts/metrics/global_explain/global_smarts_positive_test_seed0.csv")
CSV_NEG = Path("artifacts/metrics/global_explain/global_smarts_negative_test_seed0.csv")
OUT = Path("artifacts/figures/smarts_global_heatmap_with_freq.png")

TOP_NEG = 10
TOP_POS = 8

# ================= LOAD =================
pos = pd.read_csv(CSV_POS).head(TOP_POS)
neg = pd.read_csv(CSV_NEG).head(TOP_NEG)

MIN_FREQ = 20

# 合并正负结构
df = pd.concat([pos, neg], ignore_index=True)

# 频率过滤（关键）
df = df[df["freq_mols"] >= MIN_FREQ]

# positive → negative 排序
df = df.sort_values("mean_delta", ascending=False).reset_index(drop=True)

# 统一 index
df_plot = df.set_index("smarts_name")
df_heat = df_plot[["mean_delta"]]
freq = df_plot["freq_mols"]



# ================= PLOT =================
sns.set(style="white")
fig, (ax_hm, ax_bar) = plt.subplots(
    ncols=2,
    figsize=(10, 0.45 * len(df)),
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
    annot_kws={"size": 10},
    cbar_kws={"label": "Mean ΔBBB Probability"}
)

ax_hm.set_xlabel("")
ax_hm.set_ylabel("SMARTS Substructure", fontsize=12, fontname="Times New Roman")
ax_hm.set_title(
    "Global Contribution of Chemical Substructures to BBB Permeability",
    fontsize=14,
    fontname="Times New Roman",
    pad=12
)

# ---- Frequency bar ----
ax_bar.barh(
    y=range(len(freq)),       # 用位置索引，而不是 label
    width=freq.values,
    color="gray",
    alpha=0.8
)

ax_bar.set_yticks(range(len(freq)))
ax_bar.set_yticklabels(freq.index)


ax_bar.set_xlabel(
    "Frequency",
    fontsize=11,
    fontname="Times New Roman"
)

ax_bar.invert_xaxis()  # bars point toward heatmap
ax_bar.grid(axis="x", linestyle="--", alpha=0.4)

# ---- Font control (Times New Roman everywhere) ----
for ax in (ax_hm, ax_bar):
    for item in (
        ax.get_xticklabels()
        + ax.get_yticklabels()
        + ax.get_figure().texts
    ):
        item.set_fontname("Times New Roman")

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=300)
plt.close()

print("Saved:", OUT)
