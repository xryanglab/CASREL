#!/usr/bin/env python

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "figure.dpi": 300,
})

PALETTE = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488"]


def fdr_correction(pvals, alpha=0.05):
    """
    Benjamini-Hochberg FDR correction.
    Returns adjusted q-values.
    """
    pvals = np.asarray(pvals)
    n = len(pvals)
    if n == 0:
        return pvals
    sorted_idx = np.argsort(pvals)
    sorted_p = pvals[sorted_idx]
    qvals = np.zeros(n)
    for i, p in enumerate(sorted_p):
        q = p * n / (i + 1)
        qvals[sorted_idx[i]] = min(q, 1.0)
    for i in range(n-2, -1, -1):
        qvals[sorted_idx[i]] = min(qvals[sorted_idx[i]], qvals[sorted_idx[i+1]])
    return qvals


def compute_correlation_baseline_fdr(gene_exp_df, splice_df, fdr_threshold=0.05, rho_threshold=0.1):
    """
    For each AS event, compute Spearman correlations with all RBPs,
    perform FDR correction on the p-values, and retain only pairs with:
        q < fdr_threshold AND |rho| > rho_threshold.
    """
    results = []
    rbp_names = gene_exp_df.columns.tolist()
    n_events = splice_df.shape[1]

    for col_idx in range(n_events):
        y = splice_df.iloc[:, col_idx].values.astype(float)
        site_name = splice_df.columns[col_idx]

        valid_y = y[~np.isnan(y)]
        if len(np.unique(valid_y)) < 2:
            continue

        corrs = []
        pvals = []
        for rbp in rbp_names:
            x = gene_exp_df[rbp].values.astype(float)
            try:
                rho, pval = spearmanr(x, y)
                if not np.isnan(rho):
                    corrs.append({
                        "RBP": rbp,
                        "AS_Site": site_name,
                        "correlation": rho,
                        "abs_correlation": abs(rho),
                        "pvalue": pval,
                        "Direction": "+" if rho > 0 else "-",
                    })
                    pvals.append(pval)
                else:
                    corrs.append({
                        "RBP": rbp,
                        "AS_Site": site_name,
                        "correlation": np.nan,
                        "abs_correlation": np.nan,
                        "pvalue": np.nan,
                        "Direction": "?",
                    })
                    pvals.append(np.nan)
            except Exception:
                corrs.append({
                    "RBP": rbp,
                    "AS_Site": site_name,
                    "correlation": np.nan,
                    "abs_correlation": np.nan,
                    "pvalue": np.nan,
                    "Direction": "?",
                })
                pvals.append(np.nan)

        if not corrs:
            continue

        corr_df = pd.DataFrame(corrs)

        valid_mask = ~np.isnan(pvals)
        if not valid_mask.any():
            continue

        valid_p = np.array(pvals)[valid_mask]
        valid_q = fdr_correction(valid_p, alpha=fdr_threshold)
        qvals = np.full(len(pvals), np.nan)
        qvals[valid_mask] = valid_q

        corr_df["qvalue"] = qvals
        keep = (corr_df["qvalue"] < fdr_threshold) & (corr_df["abs_correlation"] > rho_threshold)
        corr_df = corr_df[keep].copy()

        if not corr_df.empty:
            results.append(corr_df)

    if results:
        return pd.concat(results, ignore_index=True)
    return None


def plot_correlation_distribution(corr_results, output_dir, rho_threshold=0.1):
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(
        corr_results["abs_correlation"], bins=50,
        color=PALETTE[1], alpha=0.7, edgecolor="white",
    )
    ax.set_xlabel("|Spearman Correlation|")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of RBP-AS |Spearman rho|\n(FDR < 0.05, |rho| > {rho_threshold})")
    median_val = np.median(corr_results["abs_correlation"])
    ax.axvline(
        x=median_val, color="red", linestyle="--",
        label=f"median = {median_val:.3f}",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "correlation_distribution.pdf"))
    plt.close(fig)
    print(f"Saved: {os.path.join(output_dir, 'correlation_distribution.pdf')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing gene_exp_df.csv and splice_df.csv")
    parser.add_argument("--output_dir", type=str, default="correlation_baseline_results")
    parser.add_argument("--fdr_threshold", type=float, default=0.05,
                        help="FDR threshold for selecting significant pairs (default 0.05)")
    parser.add_argument("--rho_threshold", type=float, default=0.1,
                        help="Minimum absolute Spearman correlation to keep (default 0.1)")
    args = parser.parse_args()

    gene_exp_path = os.path.join(args.data_dir, "gene_exp_df.csv")
    splice_path = os.path.join(args.data_dir, "splice_df.csv")
    for p in [gene_exp_path, splice_path]:
        if not os.path.exists(p):
            print(f"ERROR: File not found: {p}")
            sys.exit(1)

    gene_exp_df = pd.read_csv(gene_exp_path)
    splice_df = pd.read_csv(splice_path)

    corr_results = compute_correlation_baseline_fdr(
        gene_exp_df, splice_df,
        fdr_threshold=args.fdr_threshold,
        rho_threshold=args.rho_threshold
    )

    if corr_results is not None:
        plot_correlation_distribution(corr_results, args.output_dir, args.rho_threshold)
        out_csv = os.path.join(args.output_dir, "correlation_baseline_fdr.csv")
        corr_results.to_csv(out_csv, index=False)
        print(f"Correlation baseline saved: {out_csv} (n={len(corr_results)} pairs)")
    else:
        print("No significant correlation pairs found.")

    print("Done.")


if __name__ == "__main__":
    main()