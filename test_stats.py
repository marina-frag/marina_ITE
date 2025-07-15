import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plot_utils as pu
import os

csv_path = r"C:/Users/marin/Documents/ITE/PROJECT/statistics/all_runs_summary.csv"
df = pd.read_csv(csv_path)

output_path = r"C:/Users/marin/Documents/ITE/PROJECT/results"
os.makedirs(output_path, exist_ok=True)  # ensure output directory exists

def plot_metric(metric_real, metric_null, metric_name, output_path):
    error_bar_style = {'ecolor': 'black', 'elinewidth': 2, 'capsize': 4, 'capthick': 2}
    fig, ax = pu.configure_plot(
        title=f'{metric_name} Distribution (Real vs Null)' if metric_null is not None else f'{metric_name} Distribution (Real)',
        xlabel=metric_name,
        ylabel='Probability'
    )
    if metric_null is not None:
        all_vals = pd.concat([metric_real, metric_null])
    else:
        all_vals = metric_real
    min_val = all_vals.min()
    max_val = all_vals.max()
    margin = 0.05 * (max_val - min_val)
    if metric_null is not None:
        pu.across_mice_hist(
            across_mice_data=[metric_null.values],
            ax=ax,
            x_left=max(0.0, min_val - margin),
            x_right=min(1.0, max_val + margin),
            bins=20,
            color='black',
            label='Null',
            alpha=0.3,
            across_mice=False,
            zorder=1,
            error_kw=error_bar_style
        )
    pu.across_mice_hist(
        across_mice_data=[metric_real.values],
        ax=ax,
        x_left=max(0.0, min_val - margin),
        x_right=min(1.0, max_val + margin),
        bins=20,
        color='blue',
        label='Real',
        alpha=0.7,
        across_mice=False,
        zorder=2,
        error_kw=error_bar_style
    )
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    legend_colors = ['orange', 'blue'] if metric_null is not None else ['blue']
    pu.configure_legend(ax, colors=legend_colors, font_size=30)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


metrics = ["Accuracy","Precision","Recall","Specificity","F1","AP","ROC AUC"]
pairs   = [("Train","Null Train"), ("Val","Null Val"), ("Test","Null Test")]


for m in metrics:
    for real, null in pairs:
        # παίρνουμε pandas Series αντί για .values
        dr = df.loc[df.split == real, m]
        # αν δεν υπάρχουν null τιμές, βάζουμε None
        
        dn_series = df.loc[df.split == null, m]
        dn = dn_series if not dn_series.empty else None

        # φτιάχνουμε πλήρες όνομα αρχείου
        fname = f"{m.lower().replace(' ','_')}_{real.lower().replace(' ','_')}"
        if dn is not None:
            fname += f"_vs_{null.lower().replace(' ','_')}"
        fname += ".png"
        out_file = os.path.join(output_path, fname)

        plot_metric(dr, dn, m, out_file)
        print(f"Saved plot for {m} ({real}" + (f" vs {null})" if dn is not None else ")") )
