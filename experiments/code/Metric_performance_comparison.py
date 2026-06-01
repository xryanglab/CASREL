#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C = {
    "casrel":   "#00A087",
    "baseline": "#BBBBBB",
}

def to_bool(s):
    def _c(x):
        if isinstance(x, (bool, np.bool_)):
            return bool(x)
        if isinstance(x, str):
            return x.strip().lower() == "true"
        return bool(x)
    return s.apply(_c)

def _significance_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"

def _violin_and_median(ax, data_pairs, labels, colors, positions=None):
    n = len(data_pairs)
    if n == 0:
        return
    if positions is None:
        positions = np.arange(n)
    w = 0.4
    cas_color, bas_color = colors
    for i, (cas_arr, bas_arr) in enumerate(data_pairs):
        x_center = positions[i]

        vc = ax.violinplot(cas_arr, positions=[x_center - w/2],
                           showextrema=False, widths=w*0.9)
        vc["bodies"][0].set_facecolor(cas_color)
        vc["bodies"][0].set_alpha(0.45)

        vc2 = ax.violinplot(bas_arr, positions=[x_center + w/2],
                            showextrema=False, widths=w*0.9)
        vc2["bodies"][0].set_facecolor(bas_color)
        vc2["bodies"][0].set_alpha(0.45)

        bp = ax.boxplot([cas_arr, bas_arr],
                        positions=[x_center - w/2, x_center + w/2],
                        widths=w*0.5, patch_artist=True,
                        showfliers=False,
                        medianprops=dict(color="black", lw=1.5))
        bp["boxes"][0].set_facecolor(cas_color)
        bp["boxes"][1].set_facecolor(bas_color)

        med_c, med_b = np.median(cas_arr), np.median(bas_arr)
        ax.text(x_center - w/2, min(med_c + 0.04, 1.06), f"{med_c:.3f}",
                ha="center", fontsize=8, fontweight="bold")
        ax.text(x_center + w/2, min(med_b + 0.04, 1.06), f"{med_b:.3f}",
                ha="center", fontsize=8, fontweight="bold")

        improved = (cas_arr > bas_arr).mean()
        y_top = max(np.percentile(cas_arr, 95), np.percentile(bas_arr, 95)) + 0.07
        try:
            stat, p = wilcoxon(cas_arr, bas_arr, zero_method='wilcox', alternative='two-sided')
        except Exception:
            p = np.nan
        p_str = _significance_stars(p) if not np.isnan(p) else "?"
        ax.text(x_center, min(y_top, 1.15), f"{improved:.0%}\n{p_str}",
                ha="center", fontsize=8, fontweight="bold", color="#333")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Score", fontsize=10)

def plot_casrel_vs_baseline(qual, dataset_name, output_dir):
    metric_pairs = [
        ("balanced_accuracy", "bacc_maj", "Balanced\nAccuracy"),
        ("macro_f1", "mf1_maj", "Macro-F1"),
    ]
    data_pairs, labels = [], []
    for col, maj_col, lab in metric_pairs:
        if col not in qual.columns or maj_col not in qual.columns:
            continue
        valid = qual[[col, maj_col]].dropna()
        if len(valid) < 2:
            continue
        data_pairs.append((valid[col].values, valid[maj_col].values))
        labels.append(lab)

    if not data_pairs:
        print("No valid data for selected metrics.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    _violin_and_median(ax, data_pairs, labels, [C["casrel"], C["baseline"]])
    ax.set_title(f"{dataset_name}: CASREL vs Baseline", fontweight="bold", fontsize=11)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()

    for ext in ["pdf", "png"]:
        out_path = os.path.join(output_dir, f"{dataset_name}_casrel_vs_baseline.{ext}")
        fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved figure to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Plot CASREL vs Baseline (Balanced Accuracy & Macro-F1 only)")
    parser.add_argument("--metrics_csv", required=True, help="CSV with metrics")
    parser.add_argument("--dataset_name", required=True, help="Dataset name for output")
    parser.add_argument("--output_dir", default="reviewer_figures", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.metrics_csv)
    print(f"Loaded {len(df)} rows from {args.metrics_csv}")

    df["qualifies"] = to_bool(df["qualifies"])
    qual = df[df["qualifies"]].copy()

    plot_casrel_vs_baseline(qual, args.dataset_name, args.output_dir)
    print("Done.")

if __name__ == "__main__":
    main()