#!/usr/bin/env python3

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, gaussian_kde

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Colors ──
FOLD_COLORS = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F"]
C_CASREL = "#00A087"
C_BASELINE = "#BBBBBB"


def to_bool(s):
    """Convert column to boolean safely."""
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


def _save(fig, out, ds, name):
    for ext in ["pdf", "png"]:
        path = os.path.join(out, f"{ds}_{name}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {ds}_{name}.pdf/.png")


def compute_unfiltered_weighted_f1(df):
    """
    Compute unfiltered weighted-F1 for ALL sites from per-class F1 and support
    already stored in the CSV. No sample/class filtering applied.
    
    weighted_f1 = sum(f1_class_c * support_class_c) / sum(support_class_c)
    """
    f1_cols = [f"f1_class_{c}" for c in range(4)]
    sup_cols = [f"support_class_{c}" for c in range(4)]

    for col in f1_cols + sup_cols:
        if col not in df.columns:
            print(f"  [WARN] Column {col} not found; cannot compute unfiltered wf1.")
            return pd.Series(np.nan, index=df.index)

    f1_arr = df[f1_cols].values 
    sup_arr = df[sup_cols].values  

    numerator = (f1_arr * sup_arr).sum(axis=1)
    denominator = sup_arr.sum(axis=1)
    wf1 = np.where(denominator > 0, numerator / denominator, np.nan)
    return pd.Series(wf1, index=df.index)


def fig_per_fold_metric(df, cas_col, bas_col, metric_title, ds, out, filename_suffix):
    """
    Per-fold paired violin+box plot for one metric.
    CASREL vs Majority Baseline, qualifying sites only.
    """
    qual = df[to_bool(df["qualifies"])].copy()
    folds = sorted(qual["fold"].unique())
    n_folds = len(folds)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    positions_cas = []
    positions_bas = []
    data_cas = []
    data_bas = []
    fold_labels = []
    fold_n = []

    for i, fold in enumerate(folds):
        fold_data = qual[qual["fold"] == fold]
        valid = fold_data[[cas_col, bas_col]].dropna()
        if len(valid) == 0:
            continue

        x_center = i * 2.0
        positions_cas.append(x_center - 0.35)
        positions_bas.append(x_center + 0.35)
        data_cas.append(valid[cas_col].values)
        data_bas.append(valid[bas_col].values)
        fold_labels.append(fold.upper())
        fold_n.append(len(valid))

    if not data_cas:
        ax.text(0.5, 0.5, "No qualifying data", transform=ax.transAxes,
                ha="center", fontsize=14)
        _save(fig, out, ds, filename_suffix)
        return

    vp_cas = ax.violinplot(data_cas, positions=positions_cas,
                           showextrema=False, widths=0.6)
    for body in vp_cas["bodies"]:
        body.set_facecolor(C_CASREL)
        body.set_alpha(0.35)


    vp_bas = ax.violinplot(data_bas, positions=positions_bas,
                           showextrema=False, widths=0.6)
    for body in vp_bas["bodies"]:
        body.set_facecolor(C_BASELINE)
        body.set_alpha(0.35)

    bp_cas = ax.boxplot(data_cas, positions=positions_cas,
                        widths=0.28, patch_artist=True, showfliers=False,
                        medianprops=dict(color="white", lw=1.8),
                        whiskerprops=dict(color=C_CASREL, lw=1.2),
                        capprops=dict(color=C_CASREL, lw=1.2))
    for patch in bp_cas["boxes"]:
        patch.set_facecolor(C_CASREL)
        patch.set_alpha(0.75)

    bp_bas = ax.boxplot(data_bas, positions=positions_bas,
                        widths=0.28, patch_artist=True, showfliers=False,
                        medianprops=dict(color="white", lw=1.8),
                        whiskerprops=dict(color=C_BASELINE, lw=1.2),
                        capprops=dict(color=C_BASELINE, lw=1.2))
    for patch in bp_bas["boxes"]:
        patch.set_facecolor(C_BASELINE)
        patch.set_alpha(0.75)

    fold_medians_cas = []
    fold_medians_bas = []
    for i in range(len(data_cas)):
        med_c = np.median(data_cas[i])
        med_b = np.median(data_bas[i])
        fold_medians_cas.append(med_c)
        fold_medians_bas.append(med_b)

        ax.text(positions_cas[i], med_c + 0.025, f"{med_c:.3f}",
                ha="center", fontsize=8, fontweight="bold", color="#1a6b5c")
        ax.text(positions_bas[i], med_b + 0.025, f"{med_b:.3f}",
                ha="center", fontsize=8, fontweight="bold", color="#666")

        try:
            _, p = wilcoxon(data_cas[i], data_bas[i], alternative='two-sided')
            stars = _significance_stars(p)
        except Exception:
            stars = "?"

        x_center = (positions_cas[i] + positions_bas[i]) / 2
        y_top = max(np.percentile(data_cas[i], 97),
                    np.percentile(data_bas[i], 97)) + 0.06
        ax.text(x_center, min(y_top, 1.12), stars,
                ha="center", fontsize=10, fontweight="bold", color="#333")

        ax.text(x_center, -0.07, f"n={fold_n[i]:,}",
                ha="center", fontsize=8, color="#555")

    x_centers = [(positions_cas[i] + positions_bas[i]) / 2
                 for i in range(len(positions_cas))]
    ax.set_xticks(x_centers)
    ax.set_xticklabels(fold_labels, fontsize=11)
    ax.set_xlabel("Cross-Validation Fold", fontsize=11)
    ax.set_ylabel(metric_title, fontsize=11)
    ax.set_ylim(0, 1.20)

    mean_cas = np.mean(fold_medians_cas)
    std_cas = np.std(fold_medians_cas)
    mean_bas = np.mean(fold_medians_bas)
    std_bas = np.std(fold_medians_bas)
    cv_cas = (std_cas / mean_cas * 100) if mean_cas > 0 else 0
    improvement = ((mean_cas - mean_bas) / mean_bas * 100) if mean_bas > 0 else 0

    summary = (
        f"CASREL median across folds: {mean_cas:.3f} ± {std_cas:.3f} (CV={cv_cas:.1f}%)\n"
        f"Baseline median across folds: {mean_bas:.3f} ± {std_bas:.3f}\n"
        f"Relative improvement: +{improvement:.1f}%"
    )
    ax.text(0.98, 0.98, summary, transform=ax.transAxes,
            fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow",
                      alpha=0.9, ec="#ccc"))


    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C_CASREL, alpha=0.75, edgecolor="none", label="CASREL"),
        Patch(facecolor=C_BASELINE, alpha=0.75, edgecolor="none", label="Majority Baseline"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10, frameon=False)

    ax.set_title(
        f"{ds}: Per-Fold {metric_title}\n(5-Fold CV, Qualifying Sites)",
        fontweight="bold", fontsize=12, pad=12
    )

    fig.tight_layout()
    _save(fig, out, ds, filename_suffix)



def fig_per_fold_wf1_all_sites(df, ds, out):
    """
    Per-fold Weighted-F1 distribution for ALL sites (no class/sample filtering).
    Panel A: Overlaid KDE curves (shows distribution overlap = consistency).
    Panel B: Side-by-side violin+box (shows fold-level summary).
    """
    folds = sorted(df["fold"].unique())

    df = df.copy()
    df["wf1_unfiltered"] = compute_unfiltered_weighted_f1(df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    fold_medians = []
    fold_data_list = []

    for i, fold in enumerate(folds):
        fold_data = df[df["fold"] == fold]
        wf1 = fold_data["wf1_unfiltered"].dropna().values
        if len(wf1) == 0:
            continue

        fold_data_list.append(wf1)
        fold_medians.append(np.median(wf1))

        try:
            kde = gaussian_kde(wf1, bw_method=0.12)
            x_range = np.linspace(0, 1, 500)
            density = kde(x_range)
            ax.plot(x_range, density, color=FOLD_COLORS[i], lw=2.2,
                    label=f"{fold.upper()} (median={np.median(wf1):.3f}, n={len(wf1):,})")
            ax.fill_between(x_range, density, alpha=0.08, color=FOLD_COLORS[i])
        except Exception:
            pass


    if fold_medians:
        cv = (np.std(fold_medians) / np.mean(fold_medians) * 100) if np.mean(fold_medians) > 0 else 0
        ax.text(0.02, 0.98,
                f"Mean of fold medians: {np.mean(fold_medians):.3f} ± {np.std(fold_medians):.3f}\n"
                f"CV = {cv:.1f}% (low = stable)",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="#ccc"))

    ax.set_xlabel("Weighted F1-Score", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("(A) Overlaid KDE per Fold", fontweight="bold", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_xlim(0, 1.05)

    ax = axes[1]
    data_per_fold = []
    labels_per_fold = []

    for i, fold in enumerate(folds):
        fold_data = df[df["fold"] == fold]
        wf1 = fold_data["wf1_unfiltered"].dropna().values
        if len(wf1) > 0:
            data_per_fold.append(wf1)
            labels_per_fold.append(fold.upper())

    if data_per_fold:
        positions = np.arange(len(data_per_fold))

        vp = ax.violinplot(data_per_fold, positions=positions,
                           showextrema=False, widths=0.75)
        for i, body in enumerate(vp["bodies"]):
            body.set_facecolor(FOLD_COLORS[i % len(FOLD_COLORS)])
            body.set_alpha(0.4)

        bp = ax.boxplot(data_per_fold, positions=positions,
                        widths=0.3, patch_artist=True,
                        showfliers=False,
                        medianprops=dict(color="white", lw=2))
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(FOLD_COLORS[i % len(FOLD_COLORS)])
            patch.set_alpha(0.75)

        for i, d in enumerate(data_per_fold):
            med = np.median(d)
            ax.text(i, med + 0.03, f"{med:.3f}",
                    ha="center", fontsize=9, fontweight="bold")
            ax.text(i, -0.06, f"n={len(d):,}",
                    ha="center", fontsize=8, color="#555")

        overall_med = np.median(np.concatenate(data_per_fold))
        ax.axhline(overall_med, color="#333", ls="--", lw=1.2, alpha=0.6,
                   label=f"Overall median = {overall_med:.3f}")
        ax.legend(frameon=False, fontsize=9, loc="lower right")

        ax.set_xticks(positions)
        ax.set_xticklabels(labels_per_fold, fontsize=11)

    ax.set_xlabel("Cross-Validation Fold", fontsize=11)
    ax.set_ylabel("Weighted F1-Score", fontsize=11)
    ax.set_title("(B) Per-Fold Violin Plots", fontweight="bold", fontsize=11)
    ax.set_ylim(0, 1.15)

    fig.suptitle(
        f"{ds}: Cross-Validation Stability — Weighted F1-Score (All Sites, No Filtering)",
        fontsize=12, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    _save(fig, out, ds, "per_fold_wf1_all_sites")


def main():
    ap = argparse.ArgumentParser(
        description="Per-fold CV stability visualization for reviewer response.")
    ap.add_argument("--metrics_csv", required=True,
                    help="Recomputed metrics CSV (output of recompute_metrics_v4.py)")
    ap.add_argument("--dataset_name", required=True)
    ap.add_argument("--output_dir", default="reviewer_figures")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ds = args.dataset_name

    df = pd.read_csv(args.metrics_csv)
    print(f"Loaded {len(df):,} rows from {args.metrics_csv}")
    print(f"Folds found: {sorted(df['fold'].unique())}")

    df["qualifies"] = to_bool(df["qualifies"])

    print("\n[1/3] Per-fold Balanced Accuracy (CASREL vs Baseline) ...")
    fig_per_fold_metric(df, "balanced_accuracy", "bacc_maj",
                        "Balanced Accuracy", ds, args.output_dir,
                        "per_fold_balanced_accuracy")

    print("\n[2/3] Per-fold Macro-F1 (CASREL vs Baseline) ...")
    fig_per_fold_metric(df, "macro_f1", "mf1_maj",
                        "Macro F1-Score", ds, args.output_dir,
                        "per_fold_macro_f1")

    print("\n[3/3] Per-fold Weighted-F1 distribution (all sites, no filtering) ...")
    fig_per_fold_wf1_all_sites(df, ds, args.output_dir)

    print(f"\n{'=' * 60}")
    print(f"  CROSS-FOLD STABILITY SUMMARY — {ds}")
    print(f"{'=' * 60}")

    qual = df[to_bool(df["qualifies"])].copy()
    folds = sorted(qual["fold"].unique())

    for metric, label in [("balanced_accuracy", "Balanced Accuracy"),
                          ("macro_f1", "Macro-F1")]:
        if metric not in qual.columns:
            continue
        fold_meds = []
        for fold in folds:
            v = qual.loc[qual["fold"] == fold, metric].dropna()
            if len(v) > 0:
                fold_meds.append(v.median())
        if fold_meds:
            mean_m = np.mean(fold_meds)
            std_m = np.std(fold_meds)
            cv = (std_m / mean_m * 100) if mean_m > 0 else 0
            print(f"  {label:25s}  fold medians = {[f'{m:.4f}' for m in fold_meds]}")
            print(f"  {'':25s}  mean ± std = {mean_m:.4f} ± {std_m:.4f}  (CV = {cv:.1f}%)")

    df_temp = df.copy()
    df_temp["wf1_unfiltered"] = compute_unfiltered_weighted_f1(df_temp)
    fold_meds_wf1 = []
    for fold in sorted(df_temp["fold"].unique()):
        v = df_temp.loc[df_temp["fold"] == fold, "wf1_unfiltered"].dropna()
        if len(v) > 0:
            fold_meds_wf1.append(v.median())
    if fold_meds_wf1:
        mean_w = np.mean(fold_meds_wf1)
        std_w = np.std(fold_meds_wf1)
        cv_w = (std_w / mean_w * 100) if mean_w > 0 else 0
        print(f"  {'Weighted-F1':25s}  fold medians = {[f'{m:.4f}' for m in fold_meds_wf1]}")
        print(f"  {'':25s}  mean ± std = {mean_w:.4f} ± {std_w:.4f}  (CV = {cv_w:.1f}%)")

    print(f"\n{'=' * 60}")
    print(f"All per-fold outputs saved to: {args.output_dir}")
    print("Done.\n")


if __name__ == "__main__":
    main()
