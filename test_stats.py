import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plot_utils as pu
import os

csv_path = "statistics/all_runs_summaryhl10lk10ep8_threshold.csv"
df = pd.read_csv(csv_path)

output_path = "results"
os.makedirs(output_path, exist_ok=True)  # ensure output directory exists

def plot_metric(metric_real, metric_null, metric_name, split, output_path):
    error_bar_style = {'ecolor': 'black', 'elinewidth': 2, 'capsize': 4, 'capthick': 2}
    fig, ax = pu.configure_plot(
        # title=(
        #     f'{metric_name} Distribution (Observed vs Null) - {split}'
        #     if metric_null is not None
        #     else f'{metric_name} Distribution (Real) - {split}'
        # ),
        title = (f'{split}'),
        xlabel=metric_name,
        ylabel='Probability'
    )

    all_vals = pd.concat([metric_real, metric_null]) if metric_null is not None else metric_real
    min_val, max_val = all_vals.min(), all_vals.max()
    margin = 0.05 * (max_val - min_val)

    # Draw Observed *first* (so it will appear above Null in the legend and layering)
    pu.across_mice_hist(
        across_mice_data=[metric_real.values],
        ax=ax,
        x_left=max(0.0, min_val - margin),
        x_right=min(1.0, max_val + margin),
        bins=20,
        color='blue',
        label='Observed',
        alpha=0.7,
        across_mice=False,
        zorder=2,
        error_kw=error_bar_style
    )

    if metric_null is not None:
        pu.across_mice_hist(
            across_mice_data=[metric_null.values],
            ax=ax,
            x_left=max(0.0, min_val - margin),
            x_right=min(1.0, max_val + margin),
            bins=20,
            color='gray',
            label='Null',
            alpha=0.3,
            across_mice=False,
            zorder=1,
            error_kw=error_bar_style
        )

    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    legend_colors = ['blue', 'gray'] if metric_null is not None else ['blue']
    pu.configure_legend(ax, colors=legend_colors, font_size=30, pos=(0.5, 1.025))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


metrics = ["Accuracy","Precision","Recall","Specificity","F1","AP","ROC AUC"]
pairs   = [("Train","Null Train"), ("Val","Null Val"), ("Test","Null Test")]


for m in metrics:
    for real, null in pairs:
        dr = df.loc[df.split == real, m]
        dn_series = df.loc[df.split == null, m]
        dn = dn_series if not dn_series.empty else None

        fname = f"{m.lower().replace(' ','_')}_{real.lower().replace(' ','_')}"
        if dn is not None:
            fname += f"_vs_{null.lower().replace(' ','_')}"
        fname += ".png"
        out_file = os.path.join(output_path, fname)

        plot_metric(dr, dn, m, real, out_file)
        print(f"Saved plot for {m} ({real}" + (f" vs {null})" if dn is not None else ")") )