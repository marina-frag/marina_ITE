import shutil
import os
import pandas as pd
import numpy as np
from scipy.stats import sem
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import plot_utils as pu

df = pd.read_csv("statistics/lookback25341.csv")
df['lookback'] = df['run'].str.extract(r'lb(\d+)_').astype(int)

metrics = ["Accuracy", "Recall", "Specificity", "F1", "Pearson Correlation"]

real_splits = ["Train", "Val", "Test"]


agg = (
    df
    .groupby(['lookback','split'])[metrics]
    .agg(['mean','sem'])
    .reset_index()
)

# ── 3) Fresh output directory
out_dir = "results/sensitivity"
if os.path.isdir(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir)
print("→ writing plots into:", os.path.abspath(out_dir))

# ── 4) Define colors for the three splits
COLORS = {"Train": "blue", "Val": "green", "Test": "red"}

# ── 5) Plot each metric
for m in metrics:
    if m == "Recall":
        m = "Sensitivity"
    fig, ax = pu.configure_plot(
        title=f"{m} vs Lookback",
        xlabel="Lookback",
        ylabel=m,
        fontsize=24,
        figsize=(8,6)
    )
    plotted = False
    if m == "Sensitivity":
        m = "Recall"
    for split in real_splits:
        sub = agg[agg.split == split]
        if sub.empty:
            continue
        plotted = True

        x    = sub.lookback.values
        y    = sub[(m,'mean')].values
        yerr = sub[(m,'sem')].values

        ax.errorbar(
            x, y, yerr=yerr,
            label=split,
            color=COLORS[split],
            marker='o', markersize=6,
            linewidth=2,
            **pu.ERROR_KW(elinewidth=2, capsize=5)
        )

    if not plotted:
        print(f"⚠️  skipping {m}: no Train/Val/Test data")
        plt.close(fig)
        continue

    # ── Final formatting
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    pu.configure_legend(
        ax,
        colors=[COLORS[s] for s in real_splits if not agg[agg.split==s].empty],
        loc='upper left',
        bbox_to_anchor=(1.02,1),
        font_size=14,
        legend_line=('vertical',1)
    )
    fig.tight_layout()
    fig.subplots_adjust(right=0.75)

    out_path = os.path.join(out_dir, f"{m.lower().replace(' ','_')}_vs_lookback.png")
    print("→ saving", out_path)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
