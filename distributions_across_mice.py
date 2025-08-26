import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import plot_utils as pu

# -------------------
# CONFIG
# -------------------
CSV_FILES = {
    "25387": "statistics/destribution25387.csv",
    "25341": "statistics/destribution25341.csv",
    "24617": "statistics/destribution24617.csv",
}
SPLIT_PAIRS = [("Train", "Null Train"), ("Val", "Null Val"), ("Test", "Null Test")]
METRICS_ALL = ["Accuracy", "Recall", "Specificity", "F1", "Pearson Correlation"]
OUTPUT_DIR = "results/across_mice"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------
# Utils
# -------------------
def _norm_col(c: str) -> str:
    return ' '.join(
        c.replace('\ufeff', '').replace('\xa0', ' ').strip().split()
    )

ALIASES = {
    'sensitivity': 'Recall',
    'pearson': 'Pearson Correlation',
    'pearson correlation': 'Pearson Correlation',
    'pearson_corr': 'Pearson Correlation',
}

def load_all_mice():
    dfs = []
    for mouse_id, path in CSV_FILES.items():
        df = pd.read_csv(path)
        # normalize headers
        df.columns = [_norm_col(c) for c in df.columns]
        df = df.rename(columns={c: ALIASES.get(c.lower(), c) for c in df.columns})
        df["mouse"] = mouse_id
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    # keep only available metrics
    metrics = [m for m in METRICS_ALL if m in all_df.columns]
    return all_df, metrics

def ttest_across_mice(real_series_by_mouse, null_series_by_mouse):
    """
    real_series_by_mouse / null_series_by_mouse: dict mouse_id -> 1D np.array (values)
    Do paired t-test across mice on per-mouse means if possible.
    """
    mice_real = set(real_series_by_mouse.keys())
    mice_null = set(null_series_by_mouse.keys())
    common = sorted(mice_real & mice_null)
    if len(common) >= 2:
        a = np.array([np.nanmean(real_series_by_mouse[m]) for m in common])
        b = np.array([np.nanmean(null_series_by_mouse[m]) for m in common])
        # if any NaNs remain, drop paired NaNs
        mask = np.isfinite(a) & np.isfinite(b)
        a, b = a[mask], b[mask]
        if len(a) >= 2:
            t, p = stats.ttest_rel(a, b)  # paired test across mice
            return t, p, "paired"
    # Fallback: Welch on pooled per-mouse means (or pooled values if you prefer)
    a = np.array([np.nanmean(v) for v in real_series_by_mouse.values() if len(v) > 0])
    b = np.array([np.nanmean(v) for v in null_series_by_mouse.values() if len(v) > 0])
    if len(a) >= 2 and len(b) >= 2:
        t, p = stats.ttest_ind(a, b, equal_var=False)
        return t, p, "welch"
    return np.nan, np.nan, "insufficient"

def plot_metric_across_mice(df, metric, split, split_null, outpath):
    # For legend label: rename Recall -> Sensitivity (only label)
    label_metric = "Sensitivity" if metric == "Recall" else metric

    # Collect per-mouse arrays
    real_by_mouse = {}
    null_by_mouse = {}
    for m in sorted(df["mouse"].unique()):
        real_by_mouse[m] = df.loc[(df["mouse"] == m) & (df["split"] == split), metric].dropna().values
        if split_null is not None:
            null_by_mouse[m] = df.loc[(df["mouse"] == m) & (df["split"] == split_null), metric].dropna().values

    # Build list-of-arrays for pu.across_mice_hist
    real_arrays = [v for v in real_by_mouse.values() if len(v) > 0]
    null_arrays = [v for v in null_by_mouse.values() if len(v) > 0] if split_null else []

    if len(real_arrays) == 0:
        print(f"[skip] {metric} {split}: no data")
        return

    # Determine common x-range from pooled values (clip to [0,1] for bounded metrics)
    all_vals = np.concatenate(real_arrays + null_arrays) if null_arrays else np.concatenate(real_arrays)
    xmin, xmax = np.nanmin(all_vals), np.nanmax(all_vals)
    margin = 0.05 * (xmax - xmin if xmax > xmin else 1.0)
    x_left = max(0.0, xmin - margin)
    x_right = min(1.0, xmax + margin) if xmax <= 1.0 else xmax + margin

    fig, ax = pu.configure_plot(
        title=f"{split}",
        xlabel=label_metric,
        ylabel="Probability"
    )

    # Observed (draw second so it sits on top)
    # First Null (if present)
    if split_null and len(null_arrays) > 0:
        pu.across_mice_hist(
            across_mice_data=null_arrays,
            ax=ax,
            x_left=x_left,
            x_right=x_right,
            bins=20,
            color="gray",
            label="Null",
            alpha=0.3,
            across_mice=True,   # mean+SEM across mice
            zorder=1,
            error_kw={'ecolor': 'black', 'elinewidth': 2, 'capsize': 4, 'capthick': 2}
        )

    pu.across_mice_hist(
        across_mice_data=real_arrays,
        ax=ax,
        x_left=x_left,
        x_right=x_right,
        bins=20,
        color="blue",
        label="Observed",
        alpha=0.7,
        across_mice=True,       # mean+SEM across mice
        zorder=2,
        error_kw={'ecolor': 'black', 'elinewidth': 2, 'capsize': 4, 'capthick': 2}
    )

    # Stats (paired across mice on per-mouse means when possible)
    if split_null and len(null_arrays) > 0:
        t, p, mode = ttest_across_mice(real_by_mouse, null_by_mouse)
        print(f"{label_metric} ({split}): t = {t:.3f}, p = {p:.3e} [{mode}]")
        if np.isfinite(t) and np.isfinite(p):
            ax.text(
                0.03, 0.95, f"t = {t:.2f}, p = {p:.2e} ({mode})",
                transform=ax.transAxes, fontsize=16, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.5)
            )

    # Cosmetics
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    legend_colors = ["blue", "gray"] if split_null else ["blue"]
    pu.configure_legend(ax, colors=legend_colors, font_size=14, pos=(0.5, 1.02))

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

# -------------------
# Run
# -------------------
df_all, METRICS = load_all_mice()

for metric in METRICS:
    for split, split_null in SPLIT_PAIRS:
        fname = f"{metric.lower().replace(' ','_')}_{split.lower().replace(' ','_')}"
        if split_null:
            fname += f"_vs_{split_null.lower().replace(' ','_')}"
        out_file = os.path.join(OUTPUT_DIR, fname + ".png")
        plot_metric_across_mice(df_all, metric, split, split_null, out_file)
        print(f"saved → {out_file}")
