#!/usr/bin/env python


import argparse
import os
import sys
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from src.preprocess import norm_only, save_kfold_splits

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "figure.dpi": 300,
})

PALETTE = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
           "#8491B4", "#91D1C2", "#DC0000"]


def make_categorizer(low_thresh, high_thresh):
    def categorize(prob):
        if prob < low_thresh:
            return 1
        elif prob < high_thresh:
            return 2
        elif prob <= 1:
            return 3
    return categorize


def load_and_transform_custom(splice_file, gene_file, low_thresh, high_thresh):
    splice_start_df = norm_only(splice_file, "start")
    splice_end_df = norm_only(splice_file, "end")
    gene_exp_df = pd.read_csv(gene_file, index_col=0)

    common_columns = list(
        set(splice_start_df.columns) & set(splice_end_df.columns) & set(gene_exp_df.columns)
    )
    splice_start_df = splice_start_df[common_columns]
    splice_end_df = splice_end_df[common_columns]
    gene_exp_df = gene_exp_df[common_columns]

    categorizer = make_categorizer(low_thresh, high_thresh)
    splice_start = splice_start_df.applymap(categorizer, na_action="ignore")
    splice_end = splice_end_df.applymap(categorizer, na_action="ignore")

    splice_start.index = [str(idx) + "_s" for idx in splice_start.index]
    splice_end.index = [str(idx) + "_e" for idx in splice_end.index]

    splice_df = pd.concat([splice_start, splice_end], axis=0).fillna(0).astype(int)
    splice_df = splice_df.T
    gene_exp_df = gene_exp_df.T

    return {
        "splice_df": splice_df,
        "gene_exp_df": gene_exp_df,
    }


def run_pipeline_for_threshold(splice_file, gene_file, low_thresh, high_thresh,
                               output_dir, k_arg, train_script="src/train_review.py",
                               summary_script="src/reg_summary.py"):
    tag = f"t{low_thresh}_{high_thresh}"
    prep_dir = os.path.join(output_dir, "post_process", tag)
    train_dir = os.path.join(output_dir, "train_output", tag)

    print(f"\n{'=' * 60}")
    print(f"Threshold: ({low_thresh}, {high_thresh})")
    print(f"{'=' * 60}")

    result = load_and_transform_custom(splice_file, gene_file, low_thresh, high_thresh)
    os.makedirs(prep_dir, exist_ok=True)
    result["splice_df"].to_csv(os.path.join(prep_dir, "splice_df.csv"), index=False)
    result["gene_exp_df"].to_csv(os.path.join(prep_dir, "gene_exp_df.csv"), index=False)
    save_kfold_splits(result["gene_exp_df"], result["splice_df"], prep_dir)
    print(f"Data prepared at {prep_dir}")

    folders = "k1,k2,k3,k4,k5"
    cmd_train = [
        sys.executable, train_script,
        "-i", prep_dir, "-k", k_arg, "-o", train_dir, "-f", folders,
    ]
    print(f"Training: {' '.join(cmd_train)}")
    subprocess.run(cmd_train, check=True)

    cmd_summary = [
        sys.executable, summary_script,
        "-b", train_dir, "--top-per-site",
    ]
    print(f"Summarizing: {' '.join(cmd_summary)}")
    subprocess.run(cmd_summary, check=True)

    return tag, train_dir


def collect_metrics(train_dir, tag):
    all_dfs = []
    numeric_cols = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
    for k in ["k1", "k2", "k3", "k4", "k5"]:
        csv_path = os.path.join(train_dir, k, "metric_logs", "lgbm", "detailed_metrics.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=numeric_cols, how="any")
            if df.empty:
                continue
            df = df[(df["accuracy"] >= 0) & (df["accuracy"] <= 1)].copy()
            df["fold"] = k
            df["threshold"] = tag
            all_dfs.append(df)
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return None


def collect_reg_summary(train_dir, tag):
    path = os.path.join(train_dir, "Regulator_contribution_summary.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["threshold"] = tag
        return df
    return None


def find_contribution_col(df):
    candidates = [
        "contribution", "shap_value", "mean_shap", "importance",
        "mean_importance", "shap", "score", "mean_shap_value",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    meta_cols = {"RBP", "AS_Site", "Direction", "threshold", "fold"}
    numeric_cols = [
        c for c in df.columns
        if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    return numeric_cols[0] if numeric_cols else None


def get_pair_contrib_series(all_regs, thresholds, contrib_col):
    pair_data = {}
    for t in thresholds:
        sub = all_regs[all_regs["threshold"] == t].copy()
        if "Direction" in sub.columns:
            sub["pair_id"] = (
                sub["RBP"].astype(str) + "|" +
                sub["AS_Site"].astype(str) + "|" +
                sub["Direction"].astype(str)
            )
        else:
            sub["pair_id"] = sub["RBP"].astype(str) + "|" + sub["AS_Site"].astype(str)
        pair_data[t] = sub.groupby("pair_id")[contrib_col].mean()
    return pair_data


def get_pair_sets(all_regs, thresholds):
    pair_sets = {}
    for t in thresholds:
        sub = all_regs[all_regs["threshold"] == t]
        if "RBP" in sub.columns and "AS_Site" in sub.columns and "Direction" in sub.columns:
            pairs = set(
                sub["RBP"].astype(str) + "|" +
                sub["AS_Site"].astype(str) + "|" +
                sub["Direction"].astype(str)
            )
        elif "RBP" in sub.columns and "AS_Site" in sub.columns:
            pairs = set(sub["RBP"].astype(str) + "|" + sub["AS_Site"].astype(str))
        else:
            pairs = set()
        pair_sets[t] = pairs
    return pair_sets


def plot_pearson_heatmap(all_regs, output_dir):
    if all_regs is None or all_regs.empty:
        print("[WARN] No regulatory data for Pearson correlation heatmap.")
        return
    os.makedirs(output_dir, exist_ok=True)

    thresholds = sorted(all_regs["threshold"].unique())
    n = len(thresholds)

    contrib_col = find_contribution_col(all_regs)
    if contrib_col is None:
        print("[WARN] No contribution column found for Pearson correlation heatmap.")
        return

    pair_data = get_pair_contrib_series(all_regs, thresholds, contrib_col)

    pearson_mat = np.full((n, n), np.nan)
    for i, t1 in enumerate(thresholds):
        for j, t2 in enumerate(thresholds):
            if i == j:
                pearson_mat[i, j] = 1.0
                continue
            common = pair_data[t1].index.intersection(pair_data[t2].index)
            if len(common) >= 2:
                v1 = pair_data[t1].loc[common].values.astype(float)
                v2 = pair_data[t2].loc[common].values.astype(float)
                mask = ~(np.isnan(v1) | np.isnan(v2))
                if mask.sum() >= 2:
                    r, _ = pearsonr(v1[mask], v2[mask])
                    pearson_mat[i, j] = r

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pearson_mat, cmap="RdBu_r", vmin=-1, vmax=1)
    tick_labels = [t.replace("t", "").replace("_", " / ") for t in thresholds]
    ax.set_xticks(range(n))
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(tick_labels, fontsize=9)
    for i in range(n):
        for j in range(n):
            val = pearson_mat[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)
    plt.colorbar(im, ax=ax, label="Pearson r", shrink=0.85)
    ax.set_title(
        "Pearson Correlation of RBP-AS Gene Contribution\nAcross Discretization Thresholds",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "threshold_pearson_heatmap.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Saved threshold_pearson_heatmap.pdf")


def plot_shap_pairwise_scatter(all_regs, output_dir):
    if all_regs is None or all_regs.empty:
        print("[WARN] No regulatory data for pairwise SHAP scatter plot.")
        return
    os.makedirs(output_dir, exist_ok=True)

    thresholds = sorted(all_regs["threshold"].unique())
    n = len(thresholds)

    contrib_col = find_contribution_col(all_regs)
    if contrib_col is None:
        print("[WARN] No contribution column found for pairwise SHAP scatter plot.")
        return

    pair_data = get_pair_contrib_series(all_regs, thresholds, contrib_col)
    tick_labels = {t: t.replace("t", "").replace("_", " / ") for t in thresholds}

    fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))
    if n == 1:
        axes = np.array([[axes]])

    for i in range(n):
        for j in range(n):
            ax = axes[i][j]
            t1, t2 = thresholds[i], thresholds[j]

            if i == j:
                vals = pair_data[t1].dropna().values.astype(float)
                ax.hist(vals, bins=20, color=PALETTE[i % len(PALETTE)],
                        alpha=0.75, edgecolor="white")
                ax.set_title(tick_labels[t1], fontsize=8, fontweight="bold")
                ax.set_xlabel(contrib_col, fontsize=7)
                ax.set_ylabel("Count", fontsize=7)
            elif i < j:
                common = pair_data[t1].index.intersection(pair_data[t2].index)
                if len(common) > 0:
                    v1 = pair_data[t1].loc[common].values.astype(float)
                    v2 = pair_data[t2].loc[common].values.astype(float)
                    mask = ~(np.isnan(v1) | np.isnan(v2))
                    v1c, v2c = v1[mask], v2[mask]
                    ax.scatter(v1c, v2c, alpha=0.6, s=18,
                               color=PALETTE[j % len(PALETTE)], edgecolors="none")
                    if len(v1c) > 1:
                        vmin = min(v1c.min(), v2c.min())
                        vmax = max(v1c.max(), v2c.max())
                        ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=0.8, alpha=0.5)
                        r, _ = pearsonr(v1c, v2c)
                        ax.text(
                            0.05, 0.93,
                            f"r = {r:.2f}\nn = {int(mask.sum())}",
                            transform=ax.transAxes, fontsize=7, va="top",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
                        )
                    ax.set_xlabel(tick_labels[t1], fontsize=7)
                    ax.set_ylabel(tick_labels[t2], fontsize=7)
                else:
                    ax.text(0.5, 0.5, "No shared pairs", ha="center", va="center",
                            transform=ax.transAxes, fontsize=8, color="gray")
                    ax.set_xlabel(tick_labels[t1], fontsize=7)
                    ax.set_ylabel(tick_labels[t2], fontsize=7)
            else:
                ax.set_visible(False)

    fig.suptitle(
        "Pairwise Comparison of SHAP Contribution Values\n"
        "for Shared Regulatory RBP-AS Circuits Across Threshold Settings",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(
        os.path.join(output_dir, "threshold_shap_pairwise_scatter.pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)
    print("Saved threshold_shap_pairwise_scatter.pdf")


def plot_overlap_coefficient_heatmap(all_regs, output_dir):
    if all_regs is None or all_regs.empty:
        print("[WARN] No regulatory data for overlap coefficient heatmap.")
        return
    os.makedirs(output_dir, exist_ok=True)

    thresholds = sorted(all_regs["threshold"].unique())
    n = len(thresholds)
    pair_sets = get_pair_sets(all_regs, thresholds)

    overlap_mat = np.zeros((n, n))
    for i, t1 in enumerate(thresholds):
        for j, t2 in enumerate(thresholds):
            inter = len(pair_sets[t1] & pair_sets[t2])
            min_size = min(len(pair_sets[t1]), len(pair_sets[t2]))
            overlap_mat[i, j] = inter / min_size if min_size > 0 else 0.0

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(overlap_mat, cmap="YlOrRd", vmin=0, vmax=1)
    tick_labels = [t.replace("t", "").replace("_", " / ") for t in thresholds]
    ax.set_xticks(range(n))
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(tick_labels, fontsize=9)
    for i in range(n):
        for j in range(n):
            color = "white" if overlap_mat[i, j] > 0.5 else "black"
            ax.text(j, i, f"{overlap_mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)
    plt.colorbar(im, ax=ax, label="Overlap Coefficient", shrink=0.85)
    ax.set_title(
        "RBP-AS Regulatory Circuit Overlap Coefficient\nAcross Discretization Thresholds",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "threshold_overlap_coefficient_heatmap.pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)
    print("Saved threshold_overlap_coefficient_heatmap.pdf")


def plot_overlap_count_heatmap(all_regs, output_dir):
    if all_regs is None or all_regs.empty:
        print("[WARN] No regulatory data for overlap count heatmap.")
        return
    os.makedirs(output_dir, exist_ok=True)

    thresholds = sorted(all_regs["threshold"].unique())
    n = len(thresholds)
    pair_sets = get_pair_sets(all_regs, thresholds)

    count_mat = np.zeros((n, n), dtype=int)
    for i, t1 in enumerate(thresholds):
        for j, t2 in enumerate(thresholds):
            count_mat[i, j] = len(pair_sets[t1] & pair_sets[t2])

    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = int(count_mat.max()) if count_mat.max() > 0 else 1
    im = ax.imshow(count_mat, cmap="Blues", vmin=0, vmax=vmax)
    tick_labels = [t.replace("t", "").replace("_", " / ") for t in thresholds]
    ax.set_xticks(range(n))
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(tick_labels, fontsize=9)
    for i in range(n):
        for j in range(n):
            color = "white" if count_mat[i, j] > vmax * 0.6 else "black"
            ax.text(j, i, str(count_mat[i, j]), ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)
    plt.colorbar(im, ax=ax, label="Number of Shared RBP-AS Pairs", shrink=0.85)
    ax.set_title(
        "RBP-AS Regulatory Circuit Overlap Count\nAcross Discretization Thresholds",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "threshold_overlap_count_heatmap.pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)
    print("Saved threshold_overlap_count_heatmap.pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splice_file", type=str, required=True)
    parser.add_argument("--gene_file", type=str, required=True)
    parser.add_argument("--base_output_dir", type=str, default="sensitivity_results")
    parser.add_argument("--k_arg", type=str, default="0-99")
    parser.add_argument("--thresholds", type=str,
                        default="0.2,0.8;0.3,0.7;0.35,0.65;0.4,0.6;0.45,0.55")
    parser.add_argument("--train_script", type=str, default="src/train_review.py")
    parser.add_argument("--summary_script", type=str, default="src/reg_summary.py")
    args = parser.parse_args()

    threshold_pairs = []
    for pair_str in args.thresholds.split(";"):
        low, high = pair_str.strip().split(",")
        threshold_pairs.append((float(low), float(high)))

    all_metrics_list = []
    all_regs_list = []

    for low, high in threshold_pairs:
        tag, train_dir = run_pipeline_for_threshold(
            args.splice_file, args.gene_file, low, high,
            args.base_output_dir, args.k_arg,
            args.train_script, args.summary_script,
        )
        metrics = collect_metrics(train_dir, tag)
        if metrics is not None:
            all_metrics_list.append(metrics)
        regs = collect_reg_summary(train_dir, tag)
        if regs is not None:
            all_regs_list.append(regs)

    all_regs = pd.concat(all_regs_list, ignore_index=True) if all_regs_list else None

    if all_regs is not None:
        fig_dir = os.path.join(args.base_output_dir, "figures")
        plot_pearson_heatmap(all_regs, fig_dir)
        plot_shap_pairwise_scatter(all_regs, fig_dir)
        plot_overlap_coefficient_heatmap(all_regs, fig_dir)
        plot_overlap_count_heatmap(all_regs, fig_dir)
        all_regs.to_csv(
            os.path.join(args.base_output_dir, "all_threshold_regs.csv"), index=False
        )

    if all_metrics_list:
        all_metrics = pd.concat(all_metrics_list, ignore_index=True)
        all_metrics.to_csv(
            os.path.join(args.base_output_dir, "all_threshold_metrics.csv"), index=False
        )

    print("\nThreshold sensitivity analysis complete.")


if __name__ == "__main__":
    main()