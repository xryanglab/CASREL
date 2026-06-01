#!/usr/bin/env python

import argparse
import os
import sys
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from src.preprocess import norm_only, save_kfold_splits

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "figure.dpi": 300,
})

PALETTE = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F"]


def prepare_regression_data(splice_file, gene_file, output_dir):
    splice_start_df = norm_only(splice_file, "start")
    splice_end_df = norm_only(splice_file, "end")
    gene_exp_df = pd.read_csv(gene_file, index_col=0)

    common_columns = list(
        set(splice_start_df.columns) & set(splice_end_df.columns) & set(gene_exp_df.columns)
    )
    splice_start_df = splice_start_df[common_columns]
    splice_end_df = splice_end_df[common_columns]
    gene_exp_df = gene_exp_df[common_columns]

    splice_df = pd.concat([splice_start_df, splice_end_df], axis=0).fillna(0)
    splice_df = splice_df.T
    gene_exp_df = gene_exp_df.T

    os.makedirs(output_dir, exist_ok=True)
    splice_df.to_csv(os.path.join(output_dir, "splice_df.csv"), index=False)
    gene_exp_df.to_csv(os.path.join(output_dir, "gene_exp_df.csv"), index=False)
    save_kfold_splits(gene_exp_df, splice_df, output_dir)

    return gene_exp_df, splice_df


def train_regression_models(prep_dir, output_dir, k_arg):
    import lightgbm as lgb
    import json

    results = []

    for fold in ["k1", "k2", "k3", "k4", "k5"]:
        fold_dir = os.path.join(prep_dir, fold)
        X_train = pd.read_csv(os.path.join(fold_dir, "train_gene_exp_df.csv")).values
        y_train = pd.read_csv(os.path.join(fold_dir, "train_splice_df.csv")).values
        X_test = pd.read_csv(os.path.join(fold_dir, "test_gene_exp_df.csv")).values
        y_test = pd.read_csv(os.path.join(fold_dir, "test_splice_df.csv")).values
        splice_df = pd.read_csv(os.path.join(fold_dir, "splice_df.csv"))
        gene_exp_df = pd.read_csv(os.path.join(fold_dir, "gene_exp_df.csv"))

        n_cols = y_train.shape[1]

        if k_arg == "-1":
            col_list = list(range(n_cols))
        elif "-" in k_arg:
            start, end = k_arg.split("-")
            col_list = list(range(int(start), min(int(end) + 1, n_cols)))
        else:
            col_list = [int(k_arg)]

        for col_idx in col_list:
            try:
                y_tr = y_train[:, col_idx].astype(float)
                y_te = y_test[:, col_idx].astype(float)

                if len(np.unique(y_tr)) < 2:
                    continue

                model = lgb.LGBMRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=6,
                    num_leaves=31,
                    verbose=-1,
                    n_jobs=1,
                )
                model.fit(
                    X_train, y_tr,
                    eval_set=[(X_test, y_te)],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )

                y_pred = model.predict(X_test)

                r_pearson, _ = pearsonr(y_te, y_pred)
                r_spearman, _ = spearmanr(y_te, y_pred)
                mse = np.mean((y_te - y_pred) ** 2)
                mae = np.mean(np.abs(y_te - y_pred))

                results.append({
                    "fold": fold,
                    "site_index": col_idx,
                    "site": splice_df.columns[col_idx],
                    "pearson_r": r_pearson,
                    "spearman_r": r_spearman,
                    "mse": mse,
                    "mae": mae,
                })
            except Exception:
                traceback.print_exc()
                continue

    if results:
        results_df = pd.DataFrame(results)
        out_path = os.path.join(output_dir, "regression_results.csv")
        os.makedirs(output_dir, exist_ok=True)
        results_df.to_csv(out_path, index=False)
        return results_df
    return None


def collect_classification_metrics(casrel_dir):
    all_dfs = []
    for fold in ["k1", "k2", "k3", "k4", "k5"]:
        path = os.path.join(casrel_dir, fold, "metric_logs", "lgbm", "detailed_metrics.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["fold"] = fold
            all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else None


def plot_comparison(reg_df, cls_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax_idx, (metric, label) in enumerate(zip(["pearson_r", "spearman_r"],
                                                   ["Pearson r", "Spearman ρ"])):
        vals = reg_df[metric].dropna().values
        axes[ax_idx].hist(vals, bins=40, color=PALETTE[1], alpha=0.7, edgecolor="white")
        axes[ax_idx].axvline(np.median(vals), color="red", linestyle="--",
                             label=f"median={np.median(vals):.3f}")
        axes[ax_idx].set_xlabel(label)
        axes[ax_idx].set_ylabel("Frequency")
        axes[ax_idx].legend()
    fig.suptitle("Regression Model: Prediction Quality", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(output_dir, "regression_quality.pdf"))
    plt.close(fig)

    if cls_df is not None:
        reg_by_site = reg_df.groupby("site_index").agg({"pearson_r": "mean"}).reset_index()
        cls_by_site = cls_df.groupby("site_index").agg({"accuracy": "mean"}).reset_index()
        merged = pd.merge(reg_by_site, cls_by_site, on="site_index", how="inner")

        if len(merged) > 0:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(merged["accuracy"], merged["pearson_r"],
                       alpha=0.4, s=12, c=PALETTE[3], edgecolors="none")
            ax.set_xlabel("Classification Accuracy")
            ax.set_ylabel("Regression Pearson r")
            ax.set_title("Classification vs Regression\nper AS Event")
            if len(merged) > 3:
                r, p = spearmanr(merged["accuracy"], merged["pearson_r"])
                ax.text(0.05, 0.95, f"Spearman ρ = {r:.3f}\np = {p:.2e}",
                        transform=ax.transAxes, fontsize=9, va="top")
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "classification_vs_regression.pdf"))
            plt.close(fig)

    print(f"Regression comparison figures saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splice_file", type=str, required=True)
    parser.add_argument("--gene_file", type=str, required=True)
    parser.add_argument("--casrel_output_dir", type=str, required=True)
    parser.add_argument("--base_output_dir", type=str, default="regression_comparison")
    parser.add_argument("--k_arg", type=str, default="0-99")
    args = parser.parse_args()

    reg_prep_dir = os.path.join(args.base_output_dir, "post_process")
    prepare_regression_data(args.splice_file, args.gene_file, reg_prep_dir)

    reg_output = os.path.join(args.base_output_dir, "regression_output")
    print("Training regression models...")
    reg_df = train_regression_models(reg_prep_dir, reg_output, args.k_arg)

    cls_df = collect_classification_metrics(args.casrel_output_dir)

    if reg_df is not None:
        fig_dir = os.path.join(args.base_output_dir, "figures")
        plot_comparison(reg_df, cls_df, fig_dir)
    else:
        print("[WARN] No regression results generated.")

    print("Regression comparison complete.")


if __name__ == "__main__":
    main()