#!/usr/bin/env python
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ranksums

plt.rcParams.update({
    "font.family": "Arial",
    "font.size":   10,
    "figure.dpi":  300,
})

PALETTE = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488"]


def find_knee_point(importance_sorted_desc):
    arr = np.asarray(importance_sorted_desc, dtype=float)
    n = len(arr)
    if n <= 1:
        return 0
    if n == 2:
        return 0 if arr[0] >= arr[1] else 1
    total = arr.sum()
    if total <= 0:
        return 0
    cum_frac = np.cumsum(arr) / total
    x_norm   = np.linspace(0, 1, n)
    return int(np.argmax(cum_frac - x_norm))


def categorize_splice_prob(prob):
    if pd.isna(prob):
        return 0
    if prob < 0.4:
        return 1
    elif prob < 0.6:
        return 2
    else:
        return 3


def classify_genes(genes, rbp_genes_set, ctrl_prefix="ctrl_"):
    result = []
    for g in genes:
        if g.startswith(ctrl_prefix):
            result.append(False)
        elif g in rbp_genes_set:
            result.append(True)
        else:
            result.append(False)
    return np.array(result, dtype=bool)


def apply_adaptive_threshold(mergedf, method="knee", cumulative_pct=0.80):
    filtered_dfs   = []
    threshold_info = {}
    for site, group in mergedf.groupby("site"):
        group = group.copy()
        group["importance"] = group[["shap1", "shap2"]].max(axis=1)
        group = group.sort_values("importance", ascending=False).reset_index(drop=True)
        total_importance = group["importance"].sum()
        if total_importance <= 0:
            continue
        imp_arr = group["importance"].values
        if method == "knee":
            cutoff_pos = find_knee_point(imp_arr)
        elif method == "cumulative":
            cum_frac = np.cumsum(imp_arr) / total_importance
            above    = cum_frac >= cumulative_pct
            cutoff_pos = int(np.argmax(above)) if above.any() else len(group) - 1
        else:
            raise ValueError(f"Unknown adaptive method: {method}")
        cutoff_pos = max(0, cutoff_pos)
        selected   = group.iloc[: cutoff_pos + 1].copy()
        threshold_info[site] = {
            "n_total_genes":           len(group),
            "n_selected":              len(selected),
            "adaptive_threshold":      float(selected["importance"].iloc[-1]),
            "cumulative_contribution": float(np.sum(imp_arr[: cutoff_pos + 1]) / total_importance),
        }
        filtered_dfs.append(selected.drop(columns=["importance"]))
    if filtered_dfs:
        return pd.concat(filtered_dfs, ignore_index=True), threshold_info
    return pd.DataFrame(), {}


def apply_contribution_filter(regsum, contrib_filter, verbose=True):
    if contrib_filter == "none" or len(regsum) == 0:
        return regsum
    contributions = regsum["Contribution"].values
    min_contrib   = 0.0
    if contrib_filter == "knee":
        sorted_c   = np.sort(contributions)[::-1]
        knee_idx   = find_knee_point(sorted_c)
        min_contrib = float(sorted_c[knee_idx]) if knee_idx < len(sorted_c) else 0.0
        if verbose:
            print(f"  [contribution-filter=knee] threshold = {min_contrib:.6f}")
    elif contrib_filter.startswith("quantile:"):
        try:
            q = float(contrib_filter.split(":", 1)[1])
            if not 0.0 <= q <= 1.0:
                raise ValueError("quantile must be in [0, 1]")
            min_contrib = float(np.quantile(contributions, q))
            if verbose:
                print(f"  [contribution-filter=quantile:{q}] threshold = {min_contrib:.6f}")
        except Exception as exc:
            print(f"[WARN] Invalid quantile spec '{contrib_filter}': {exc}. Skipping filter.")
            return regsum
    else:
        print(f"[WARN] Unknown contribution-filter '{contrib_filter}'. Skipping.")
        return regsum
    if min_contrib > 0.0:
        before = len(regsum)
        regsum = regsum[regsum["Contribution"] >= min_contrib].copy()
        if verbose:
            print(f"  Contribution filter: {before} -> {len(regsum)} pairs kept (>= {min_contrib:.6f})")
    return regsum


def compute_comparison_stats(regsum_with_isrbp):
    regsum     = regsum_with_isrbp
    rbp_rows   = regsum[regsum["is_rbp"]]
    nonrbp_rows= regsum[~regsum["is_rbp"]]
    contrib_rbp    = rbp_rows["Contribution"].values
    contrib_nonrbp = nonrbp_rows["Contribution"].values
    if len(contrib_rbp) == 0:
        print("[WARN] No RBP entries found.")
    if len(contrib_nonrbp) == 0:
        print("[WARN] No non-RBP entries found.")
    top_n    = max(1, int(len(regsum) * 0.1))
    top_rows = regsum.nlargest(top_n, "Contribution")
    top_counts = {
        "RBP":     int(top_rows["is_rbp"].sum()),
        "Non-RBP": int((~top_rows["is_rbp"]).sum()),
    }
    stat, pval = (
        ranksums(contrib_rbp, contrib_nonrbp)
        if len(contrib_rbp) > 0 and len(contrib_nonrbp) > 0
        else (np.nan, np.nan)
    )
    n_total = max(1, len(regsum))
    summary_stats = {
        "n_total_pairs":              int(len(regsum)),
        "n_rbp_pairs":                int(len(rbp_rows)),
        "n_nonrbp_pairs":             int(len(nonrbp_rows)),
        "pct_rbp_pairs":              float(len(rbp_rows) / n_total),
        "n_unique_rbp_genes":         int(rbp_rows["RBP"].nunique()),
        "n_unique_nonrbp_genes":      int(nonrbp_rows["RBP"].nunique()),
        "mean_contribution_rbp":      float(np.mean(contrib_rbp)) if len(contrib_rbp) > 0 else np.nan,
        "median_contribution_rbp":    float(np.median(contrib_rbp)) if len(contrib_rbp) > 0 else np.nan,
        "mean_contribution_nonrbp":   float(np.mean(contrib_nonrbp)) if len(contrib_nonrbp) > 0 else np.nan,
        "median_contribution_nonrbp": float(np.median(contrib_nonrbp)) if len(contrib_nonrbp) > 0 else np.nan,
        "wilcoxon_stat":              float(stat),
        "wilcoxon_p":                 float(pval),
        "top10pct_n":                 top_n,
        "top10pct_rbp":               top_counts["RBP"],
        "top10pct_nonrbp":            top_counts["Non-RBP"],
    }
    return contrib_rbp, contrib_nonrbp, top_counts, summary_stats


def plot_two_barplots(regsum, summary_stats, output_dir, title_suffix=""):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    ax1, ax2 = axes

    labels = ["RBP", "Non-RBP"]

    counts = [summary_stats["n_rbp_pairs"], summary_stats["n_nonrbp_pairs"]]
    total = max(1, sum(counts))
    bars1 = ax1.bar(labels, counts, color=PALETTE[:2], edgecolor="white")
    ax1.set_ylabel("Gene-Site Pairs")
    ax1.set_title("Filtered Features")
    for bar, count in zip(bars1, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + total * 0.012,
                 f"{count}\n({count / total:.1%})", ha="center", fontsize=9)

    top_counts = {"RBP": summary_stats["top10pct_rbp"], "Non-RBP": summary_stats["top10pct_nonrbp"]}
    total_top = max(1, sum(top_counts.values()))
    bars2 = ax2.bar(list(top_counts.keys()), list(top_counts.values()), color=PALETTE[:2], edgecolor="white")
    ax2.set_ylabel("Count in Top-10% Regulators")
    ax2.set_title("Top-10% Regulator Composition")
    for bar, v in zip(bars2, top_counts.values()):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + total_top * 0.012,
                 f"{v} ({v / total_top:.1%})", ha="center", fontsize=9)

    pval = summary_stats.get("wilcoxon_p", np.nan)
    sig_text = ("Wilcoxon: N/A" if np.isnan(pval)
                else "p < 1e-300" if pval < 1e-300
                else f"Wilcoxon p = {pval:.2e}")
    fig.suptitle(f"Non-RBP Control Analysis{title_suffix}\n{sig_text}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_pdf = os.path.join(output_dir, "nonrbp_control_filtered_top10.pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved nonrbp_control_filtered_top10.pdf -> {output_dir}")


def run_summary_mode(regulator_file, gene_file, output_dir, ctrl_prefix="ctrl_", contribution_filter="none"):
    os.makedirs(output_dir, exist_ok=True)
    regsum = pd.read_csv(regulator_file)
    required = {"RBP", "AS_Site", "Contribution", "Direction"}
    missing = required - set(regsum.columns)
    if missing:
        raise ValueError(f"regulator_file missing columns: {missing}")
    rbp_df = pd.read_csv(gene_file, index_col=0)
    rbp_genes_set = set(rbp_df.index)
    regsum = regsum.copy()
    regsum["is_rbp"] = classify_genes(regsum["RBP"].values, rbp_genes_set, ctrl_prefix)
    if contribution_filter != "none":
        print(f"Applying additional contribution filter ({contribution_filter}) on summary-mode input...")
        regsum = apply_contribution_filter(regsum, contribution_filter)
    return regsum


def load_data_standalone(splice_file, gene_file, full_gene_file=None):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    try:
        from src.preprocess import norm_only
    except ImportError as exc:
        raise ImportError("Cannot import 'prepare.norm_only'. Make sure src/prepare.py is accessible.") from exc
    splice_start_df = norm_only(splice_file, "start")
    splice_end_df   = norm_only(splice_file, "end")
    rbp_exp_df      = pd.read_csv(gene_file, index_col=0)
    common_columns = list(set(splice_start_df.columns) & set(splice_end_df.columns) & set(rbp_exp_df.columns))
    splice_start_df = splice_start_df[common_columns]
    splice_end_df   = splice_end_df[common_columns]
    rbp_exp_df      = rbp_exp_df[common_columns]
    _cat = np.vectorize(categorize_splice_prob)
    splice_start = pd.DataFrame(_cat(splice_start_df.values), index=splice_start_df.index, columns=splice_start_df.columns)
    splice_end   = pd.DataFrame(_cat(splice_end_df.values),   index=splice_end_df.index,   columns=splice_end_df.columns)
    splice_start.index = [str(i) + "_s" for i in splice_start.index]
    splice_end.index   = [str(i) + "_e" for i in splice_end.index]
    splice_df = pd.concat([splice_start, splice_end], axis=0).fillna(0).astype(int).T
    rbp_genes = list(rbp_exp_df.index)
    n_rbp = len(rbp_genes)
    if full_gene_file and os.path.exists(full_gene_file):
        full_exp_df = pd.read_csv(full_gene_file, index_col=0)
        common_columns = list(set(common_columns) & set(full_exp_df.columns))
        splice_start_df = splice_start_df[common_columns]
        splice_end_df   = splice_end_df[common_columns]
        rbp_exp_df      = rbp_exp_df[common_columns]
        full_exp_df     = full_exp_df[common_columns]
        non_rbp_candidates = [g for g in full_exp_df.index if g not in set(rbp_genes)]
        np.random.seed(42)
        selected_non_rbp = np.random.choice(non_rbp_candidates, size=min(n_rbp, len(non_rbp_candidates)), replace=False)
        non_rbp_exp_df  = full_exp_df.loc[selected_non_rbp]
        combined_exp_df = pd.concat([rbp_exp_df, non_rbp_exp_df], axis=0)
        is_rbp_labels   = [True] * n_rbp + [False] * len(selected_non_rbp)
    else:
        print("[INFO] No full_gene_file provided. Using shuffled RBP expression as control.")
        np.random.seed(42)
        shuffled = rbp_exp_df.copy()
        for col in shuffled.columns:
            shuffled[col] = np.random.permutation(shuffled[col].values)
        shuffled.index  = ["ctrl_" + g for g in shuffled.index]
        combined_exp_df = pd.concat([rbp_exp_df, shuffled], axis=0)
        is_rbp_labels   = [True] * n_rbp + [False] * n_rbp
    combined_exp_df = combined_exp_df.T
    gene_names = list(combined_exp_df.columns)
    return combined_exp_df, splice_df, gene_names, is_rbp_labels


def _extract_class_shap_sum(shap_output, class_idx, n_samples, n_genes):
    try:
        if isinstance(shap_output, list):
            if class_idx < len(shap_output):
                sv = np.asarray(shap_output[class_idx])
                if sv.ndim == 2 and sv.shape[0] == n_samples and sv.shape[1] == n_genes:
                    return np.abs(sv).sum(axis=0)
        elif isinstance(shap_output, np.ndarray) and shap_output.ndim == 3:
            if (shap_output.shape[0] == n_samples and shap_output.shape[1] == n_genes and class_idx < shap_output.shape[2]):
                return np.abs(shap_output[:, :, class_idx]).sum(axis=0)
            if (class_idx < shap_output.shape[0] and shap_output.shape[1] == n_samples and shap_output.shape[2] == n_genes):
                return np.abs(shap_output[class_idx]).sum(axis=0)
        elif isinstance(shap_output, np.ndarray) and shap_output.ndim == 2:
            if class_idx == 0 and shap_output.shape == (n_samples, n_genes):
                return np.abs(shap_output).sum(axis=0)
    except Exception:
        pass
    return None


def _make_lgb_callbacks():
    try:
        cb = [__import__("lightgbm").log_evaluation(period=-1)]
        return cb
    except Exception:
        pass
    try:
        cb = [__import__("lightgbm").callback.log_evaluation(period=-1)]
        return cb
    except Exception:
        pass
    return []


def run_standalone_mode(splice_file, gene_file, full_gene_file, k_arg, output_dir, ctrl_prefix,
                        threshold_mode="adaptive-knee", cumulative_pct=0.80, fixed_shap_threshold=10.0,
                        contribution_filter="knee", acc_threshold=0.0):
    import lightgbm as lgb
    import shap as shap_lib
    os.makedirs(output_dir, exist_ok=True)
    combined_exp_df, splice_df, gene_names, is_rbp_labels = load_data_standalone(splice_file, gene_file, full_gene_file)
    n_rbp_feat    = sum(is_rbp_labels)
    n_nonrbp_feat = len(is_rbp_labels) - n_rbp_feat
    print(f"Features : {n_rbp_feat} RBPs + {n_nonrbp_feat} non-RBPs")
    print(f"Cells    : {combined_exp_df.shape[0]}, AS events: {splice_df.shape[1]}")
    rbp_df        = pd.read_csv(gene_file, index_col=0)
    rbp_genes_set = set(rbp_df.index)
    n_genes       = len(gene_names)
    num_cols = splice_df.shape[1]
    if k_arg == "-1":
        col_indices = list(range(num_cols))
    elif "-" in k_arg:
        s, e = k_arg.split("-")
        col_indices = list(range(int(s), min(int(e) + 1, num_cols)))
    else:
        col_indices = [int(k_arg)]
    num_samples = combined_exp_df.shape[0]
    np.random.seed(42)
    perm  = np.arange(num_samples)
    np.random.shuffle(perm)
    split = int(0.8 * num_samples)
    train_idx, test_idx = perm[:split], perm[split:]
    X_train = combined_exp_df.iloc[train_idx].values
    X_test  = combined_exp_df.iloc[test_idx].values
    X_all = combined_exp_df.values
    n_all = num_samples
    lgb_params = {
        "max_bin":           255,
        "learning_rate":     0.1,
        "boosting_type":     "gbdt",
        "objective":         "multiclass",
        "num_class":         4,
        "metric":            "multi_logloss",
        "verbose":           -1,
        "boost_from_average": True,
        "max_depth":         4,
        "num_leaves":        15,
        "reg_alpha":         0.01,
        "reg_lambda":        0.01,
        "n_jobs":            -1,
    }
    lgb_callbacks = _make_lgb_callbacks()
    print(f"\nRunning per-site SHAP analysis (threshold_mode={threshold_mode}, contribution_filter={contribution_filter}, acc_threshold={acc_threshold})...")
    all_records       = []
    sites_processed   = 0
    sites_skipped_var = 0
    sites_skipped_acc = 0
    for col_i in col_indices:
        y_all    = splice_df.iloc[:, col_i].values.astype(int)
        y_train_ = y_all[train_idx]
        y_test_  = y_all[test_idx]
        if len(np.unique(y_train_)) < 2:
            sites_skipped_var += 1
            continue
        train_data_lgb = lgb.Dataset(X_train, label=y_train_)
        test_data_lgb  = lgb.Dataset(X_test,  label=y_test_, reference=train_data_lgb)
        try:
            model = lgb.train(lgb_params, train_data_lgb, num_boost_round=100, valid_sets=[test_data_lgb], callbacks=lgb_callbacks)
        except Exception as exc:
            print(f"  [WARN] site {col_i}: training failed – {exc}")
            continue
        if acc_threshold > 0.0:
            y_pred_prob  = model.predict(X_test)
            y_pred_label = np.argmax(y_pred_prob, axis=1)
            acc = float(np.mean(y_pred_label == y_test_))
            if acc < acc_threshold:
                sites_skipped_acc += 1
                continue
        try:
            explainer   = shap_lib.TreeExplainer(model)
            shap_output = explainer.shap_values(X_all)
            shap1_vec = _extract_class_shap_sum(shap_output, 1, n_all, n_genes)
            shap2_vec = _extract_class_shap_sum(shap_output, 3, n_all, n_genes)
            if shap1_vec is None or shap2_vec is None:
                if isinstance(shap_output, list) and len(shap_output) > 0:
                    fallback = np.mean([np.abs(np.asarray(sv)).sum(axis=0) for sv in shap_output if np.asarray(sv).ndim == 2], axis=0)
                elif isinstance(shap_output, np.ndarray) and shap_output.ndim == 3:
                    fallback = np.abs(shap_output).sum(axis=(0, 2)) if shap_output.shape[1] == n_genes else np.abs(shap_output).sum(axis=(1, 0))
                else:
                    print(f"  [WARN] Cannot extract class SHAP for site {col_i}; skipping.")
                    continue
                if len(fallback) != n_genes:
                    print(f"  [WARN] SHAP length mismatch at site {col_i}; skipping.")
                    continue
                shap1_vec = fallback * 0.5
                shap2_vec = fallback * 0.5
            if len(shap1_vec) != n_genes or len(shap2_vec) != n_genes:
                print(f"  [WARN] SHAP length mismatch at site {col_i}; skipping.")
                continue
        except Exception as exc:
            print(f"  [WARN] site {col_i}: {exc}")
            continue
        sites_processed += 1
        site_name = f"column_{col_i}"
        for j in range(n_genes):
            s1, s2 = float(shap1_vec[j]), float(shap2_vec[j])
            if s1 > 0.0 or s2 > 0.0:
                all_records.append({"gene": gene_names[j], "site": site_name, "shap1": s1, "shap2": s2})
        if sites_processed % 50 == 0:
            print(f"  [{sites_processed} sites done] records so far: {len(all_records)}")
    print(f"\n  Sites: processed={sites_processed}, skipped(no-var)={sites_skipped_var}, skipped(acc<{acc_threshold})={sites_skipped_acc}")
    if not all_records:
        print("[ERROR] No gene-site records collected. Aborting.")
        return pd.DataFrame(columns=["RBP", "AS_Site", "Contribution", "Direction", "is_rbp"])
    records_df = pd.DataFrame(all_records)
    print(f"  Raw gene-site records (nonzero SHAP): {len(records_df)}")
    is_adaptive = threshold_mode.startswith("adaptive")
    if is_adaptive:
        adaptive_method = "knee" if threshold_mode == "adaptive-knee" else "cumulative"
        print(f"\nApplying per-site adaptive threshold ({adaptive_method})...")
        records_df, tinfo = apply_adaptive_threshold(records_df, method=adaptive_method, cumulative_pct=cumulative_pct)
        if records_df.empty:
            print("[ERROR] No data after adaptive thresholding.")
            return pd.DataFrame(columns=["RBP", "AS_Site", "Contribution", "Direction", "is_rbp"])
        all_thresh = [v["adaptive_threshold"] for v in tinfo.values()]
        all_nsel   = [v["n_selected"] for v in tinfo.values()]
        all_cum    = [v["cumulative_contribution"] for v in tinfo.values()]
        print(f"\n{'=' * 68}")
        print(f"  PER-SITE ADAPTIVE THRESHOLD (method = {adaptive_method})")
        print(f"{'=' * 68}")
        print(f"  Sites processed          : {len(tinfo)}")
        print(f"  Avg  SHAP threshold      : {np.mean(all_thresh):.4f}")
        print(f"  Med  SHAP threshold      : {np.median(all_thresh):.4f}")
        print(f"  Range                    : [{np.min(all_thresh):.4f}, {np.max(all_thresh):.4f}]")
        print(f"  Avg  genes selected/site : {np.mean(all_nsel):.1f}")
        print(f"  Avg  cumulative contrib  : {np.mean(all_cum):.2%}")
        print(f"  Gene-site pairs retained : {len(records_df)}")
        print(f"{'=' * 68}\n")
        thresh_path = os.path.join(output_dir, "adaptive_threshold_per_site.csv")
        pd.DataFrame([{"site": s, **v} for s, v in tinfo.items()]).to_csv(thresh_path, index=False)
        print(f"Saved per-site thresholds -> {thresh_path}")
    else:
        before = len(records_df)
        mask = ((records_df["shap1"] > fixed_shap_threshold) | (records_df["shap2"] > fixed_shap_threshold))
        records_df = records_df[mask].copy()
        print(f"Fixed SHAP threshold (>{fixed_shap_threshold}): {before} -> {len(records_df)} pairs")
        if records_df.empty:
            print("[ERROR] No data after fixed threshold.")
            return pd.DataFrame(columns=["RBP", "AS_Site", "Contribution", "Direction", "is_rbp"])
    records_df = records_df[records_df["shap1"] != records_df["shap2"]].copy()
    records_df["Direction"] = np.where(records_df["shap1"] > records_df["shap2"], "-", "+")
    records_df["Contribution"] = np.abs(records_df["shap1"] - records_df["shap2"])
    regsum = pd.DataFrame({
        "RBP": records_df["gene"].values,
        "AS_Site": records_df["site"].values,
        "Contribution": records_df["Contribution"].values,
        "Direction": records_df["Direction"].values,
    })
    regsum = regsum.sort_values("Contribution", ascending=False).reset_index(drop=True)
    print(f"\nApplying global contribution filter ({contribution_filter})...")
    regsum = apply_contribution_filter(regsum, contribution_filter)
    if regsum.empty:
        print("[ERROR] No data after contribution filter.")
        return pd.DataFrame(columns=["RBP", "AS_Site", "Contribution", "Direction", "is_rbp"])
    print(f"  Final gene-site pairs after all filters: {len(regsum)}")
    regsum["is_rbp"] = classify_genes(regsum["RBP"].values, rbp_genes_set, ctrl_prefix)
    return regsum


def main():
    parser = argparse.ArgumentParser(description="Non-RBP control analysis (Filtered Features & Top-10 only)")
    parser.add_argument("--mode", type=str, default="summary", choices=["summary", "standalone"])
    parser.add_argument("--regulator_file", type=str, default=None)
    parser.add_argument("--ctrl_prefix", type=str, default="ctrl_")
    parser.add_argument("--splice_file", type=str, default=None)
    parser.add_argument("--full_gene_file", type=str, default=None)
    parser.add_argument("--k_arg", type=str, default="0-99")
    parser.add_argument("--threshold-mode", type=str, default="adaptive-knee", dest="threshold_mode",
                        choices=["adaptive-knee", "adaptive-cumulative", "fixed"])
    parser.add_argument("--fixed-shap-threshold", type=float, default=10.0, dest="fixed_shap_threshold")
    parser.add_argument("--acc-threshold", type=float, default=0.0, dest="acc_threshold")
    parser.add_argument("--no_adaptive", action="store_true")
    parser.add_argument("--contribution-filter", type=str, default="knee", dest="contribution_filter")
    parser.add_argument("--cumulative-pct", type=float, default=0.80, dest="cumulative_pct")
    parser.add_argument("--gene_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="nonrbp_control")
    args = parser.parse_args()
    if args.no_adaptive:
        print("[INFO] --no_adaptive is deprecated; using --threshold-mode fixed.")
        args.threshold_mode = "fixed"
    os.makedirs(args.output_dir, exist_ok=True)
    fig_dir = os.path.join(args.output_dir, "figures")
    if args.mode == "summary":
        if not args.regulator_file:
            parser.error("--regulator_file is required in summary mode.")
        if not os.path.exists(args.regulator_file):
            parser.error(f"--regulator_file not found: {args.regulator_file}")
        print(f"\n[Mode: summary] Reading {args.regulator_file}")
        regsum = run_summary_mode(args.regulator_file, args.gene_file, args.output_dir,
                                  args.ctrl_prefix, args.contribution_filter)
        if regsum.empty:
            print("[ERROR] No data after filtering. Exiting.")
            return
        _, _, _, stats = compute_comparison_stats(regsum)
        plot_two_barplots(regsum, stats, fig_dir, title_suffix="  [reg_summary_v2 output]")
        regsum.to_csv(os.path.join(args.output_dir, "nonrbp_control_regsum_filtered.csv"), index=False)
        pd.DataFrame([stats]).to_csv(os.path.join(args.output_dir, "nonrbp_control_summary.csv"), index=False)
    else:
        if not args.splice_file:
            parser.error("--splice_file is required in standalone mode.")
        print(f"\n[Mode: standalone] threshold_mode={args.threshold_mode}, contribution_filter={args.contribution_filter}, acc_threshold={args.acc_threshold}")
        regsum = run_standalone_mode(args.splice_file, args.gene_file, args.full_gene_file, args.k_arg,
                                     args.output_dir, args.ctrl_prefix, args.threshold_mode,
                                     args.cumulative_pct, args.fixed_shap_threshold,
                                     args.contribution_filter, args.acc_threshold)
        if regsum.empty:
            print("[ERROR] No data after full filtering pipeline. Exiting.")
            return
        _, _, _, stats = compute_comparison_stats(regsum)
        plot_two_barplots(regsum, stats, fig_dir, title_suffix="  [standalone mode]")
        regsum.to_csv(os.path.join(args.output_dir, "nonrbp_control_regsum_filtered.csv"), index=False)
        pd.DataFrame([stats]).to_csv(os.path.join(args.output_dir, "nonrbp_control_summary.csv"), index=False)
    print("\nNon-RBP control analysis complete.")


if __name__ == "__main__":
    main()