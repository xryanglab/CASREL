#!/usr/bin/env python

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import KNNImputer
import lightgbm as lgb

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from src.preprocess import norm_only

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 10,
        "figure.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

PALETTE = ["#E64B35", "#4DBBD5", "#00A087"]

STRATEGY_ORDER = ["S3_keep_na", "S5_knn_impute", "S1_exclude_na"]
STRATEGY_LABELS = {
    "S1_exclude_na": "Exclude NA\n(3-class)",
    "S3_keep_na": "Keep NA\n(4-class, Original)",
    "S5_knn_impute": "KNN Impute\n(3-class)",
}
STRATEGY_SHORT = {
    "S1_exclude_na": "Exclude NA",
    "S3_keep_na": "Keep NA (Original)",
    "S5_knn_impute": "KNN Impute",
}

LGB_BASE = {
    "objective": "multiclass",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_samples": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbose": -1,
    "n_jobs": 4,
}

N_SPLITS = 5
N_TOP_RBPS = 20
MIN_SAMPLES = 30
CHANCE_MARGIN = 0.05
EARLY_STOPPING = 30
NUM_BOOST_ROUND = 300


def _categorize(prob):
    prob = float(prob)
    if prob < 0.4:
        return 1
    elif prob < 0.7:
        return 2
    else:
        return 3


def load_base_data(splice_file, gene_file):
    prob_start = norm_only(splice_file, "start")
    prob_end = norm_only(splice_file, "end")
    g_df = pd.read_csv(gene_file, index_col=0)

    common_cols = list(
        set(prob_start.columns) & set(prob_end.columns) & set(g_df.columns)
    )
    prob_start = prob_start[common_cols]
    prob_end = prob_end[common_cols]
    g_df = g_df[common_cols]

    cat_start = prob_start.applymap(_categorize, na_action="ignore")
    cat_end = prob_end.applymap(_categorize, na_action="ignore")
    cat_start.index = [f"{i}_s" for i in cat_start.index]
    cat_end.index = [f"{i}_e" for i in cat_end.index]

    splice_cat = (
        pd.concat([cat_start, cat_end], axis=0).fillna(0).astype(int).T
    )

    prob_start.index = [f"{i}_s" for i in prob_start.index]
    prob_end.index = [f"{i}_e" for i in prob_end.index]
    splice_prob = pd.concat([prob_start, prob_end], axis=0).T

    g_df = g_df.T
    shared = splice_cat.index.intersection(g_df.index)
    splice_cat = splice_cat.loc[shared]
    splice_prob = splice_prob.loc[shared]
    g_df = g_df.loc[shared]

    return g_df, splice_cat, splice_prob


def apply_strategy(splice_cat_raw, strategy, seed=42, splice_prob_raw=None):
    df = splice_cat_raw.copy().astype(float)

    if strategy == "S3_keep_na":
        return df, 4

    elif strategy == "S1_exclude_na":
        df[df == 0] = np.nan
        return df, 3

    elif strategy == "S5_knn_impute":
        if splice_prob_raw is None:
            raise ValueError("S5_knn_impute requires splice_prob_raw")
        prob_df = splice_prob_raw.copy()
        all_nan_cols = prob_df.columns[prob_df.isna().all(axis=0)].tolist()
        valid_cols = prob_df.columns.difference(all_nan_cols).tolist()

        if valid_cols:
            imp = KNNImputer(n_neighbors=5, weights="uniform")
            prob_imputed_valid = pd.DataFrame(
                imp.fit_transform(prob_df[valid_cols]),
                index=prob_df.index,
                columns=valid_cols,
            )
        else:
            prob_imputed_valid = pd.DataFrame(index=prob_df.index)

        for col in all_nan_cols:
            prob_imputed_valid[col] = 0.5

        prob_imputed = prob_imputed_valid[splice_prob_raw.columns]
        cat_df = prob_imputed.applymap(_categorize).astype(float)
        return cat_df, 3

    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")


def assess_trainability(splice_df, min_samples=MIN_SAMPLES, min_per_class=N_SPLITS):
    results = []
    for site in splice_df.columns:
        y_raw = splice_df[site].values.astype(float)
        valid = ~np.isnan(y_raw)
        y = y_raw[valid].astype(int)
        n_valid = len(y)

        if n_valid < min_samples:
            results.append({"site": site, "trainable": False, "n_samples": n_valid, "reason": "insufficient_samples"})
            continue

        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2:
            results.append({"site": site, "trainable": False, "n_samples": n_valid, "reason": "single_class"})
            continue

        if np.any(counts < min_per_class):
            results.append({"site": site, "trainable": False, "n_samples": n_valid, "reason": "class_too_small"})
            continue

        results.append({"site": site, "trainable": True, "n_samples": n_valid, "reason": "ok"})
    return pd.DataFrame(results)


def train_one_site(X_full, y_raw, feature_names, use_shap=True):
    valid = ~np.isnan(y_raw)
    X = X_full[valid]
    y_orig = y_raw[valid].astype(int)

    unique_labels = sorted(np.unique(y_orig))
    if len(unique_labels) < 2:
        return None
    label_map = {v: i for i, v in enumerate(unique_labels)}
    y = np.array([label_map[v] for v in y_orig], dtype=int)
    K = len(unique_labels)

    if len(y) < MIN_SAMPLES:
        return None
    if np.any(np.bincount(y, minlength=K) < N_SPLITS):
        return None

    params = {**LGB_BASE, "num_class": K}
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    fold_acc, fold_wf1 = [], []
    fold_importances = []
    fold_tops = []
    fold_shap1_list, fold_shap2_list = [], []

    try:
        for tr_idx, te_idx in skf.split(X, y):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            if len(np.unique(y_tr)) < 2:
                continue

            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dval = lgb.Dataset(X_te, label=y_te, reference=dtrain)

            model = lgb.train(
                params,
                dtrain,
                num_boost_round=NUM_BOOST_ROUND,
                valid_sets=[dval],
                callbacks=[
                    lgb.early_stopping(EARLY_STOPPING, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
            )

            y_pred = model.predict(X_te).argmax(axis=1)
            fold_acc.append(accuracy_score(y_te, y_pred))
            fold_wf1.append(f1_score(y_te, y_pred, average="weighted", zero_division=0))

            if use_shap and HAS_SHAP:
                exp = shap.TreeExplainer(model)
                svals = exp.shap_values(X_te)
                if isinstance(svals, list):
                    imp = np.mean([np.abs(sv).mean(axis=0) for sv in svals], axis=0)
                    idx1 = label_map.get(1)
                    idx3 = label_map.get(3)
                    shap1_fold = np.abs(svals[idx1]).mean(axis=0) if idx1 is not None else None
                    shap2_fold = np.abs(svals[idx3]).mean(axis=0) if idx3 is not None else None
                    if shap1_fold is not None and shap2_fold is not None:
                        fold_shap1_list.append(shap1_fold)
                        fold_shap2_list.append(shap2_fold)
                else:
                    imp = np.abs(svals).mean(axis=0)
            else:
                imp = model.feature_importance(importance_type="gain").astype(float)

            fold_importances.append(imp)
            fold_tops.append(set(np.argsort(imp)[::-1][:N_TOP_RBPS]))

    except Exception:
        return None

    if not fold_acc:
        return None

    mean_imp = np.mean(fold_importances, axis=0) if fold_importances else None
    consensus_top = set.intersection(*fold_tops) if fold_tops else set()
    chance = 1.0 / K

    shap_df = None
    if fold_shap1_list and fold_shap2_list:
        avg_shap1 = np.mean(fold_shap1_list, axis=0)
        avg_shap2 = np.mean(fold_shap2_list, axis=0)
        shap_df = pd.DataFrame({"gene": feature_names, "shap1": avg_shap1, "shap2": avg_shap2})

    return {
        "n_samples": int(len(y)),
        "num_class_actual": K,
        "mean_accuracy": float(np.mean(fold_acc)),
        "std_accuracy": float(np.std(fold_acc)),
        "mean_weighted_f1": float(np.mean(fold_wf1)),
        "chance_level": float(chance),
        "above_chance": float(np.mean(fold_acc)) > chance + CHANCE_MARGIN,
        "feature_importance": mean_imp,
        "consensus_top_rbps": consensus_top,
        "shap_df": shap_df,
    }


def build_reg_summary(shap_records, output_path):
    rows = []
    for site, sdf in shap_records:
        for _, g_row in sdf.iterrows():
            g_name = g_row["gene"]
            s1 = g_row["shap1"]
            s2 = g_row["shap2"]
            if not np.isnan(s1) and not np.isnan(s2):
                rows.append({"RBP": g_name, "AS_Site": site, "shap1": s1, "shap2": s2})

    if not rows:
        return None

    df_full = pd.DataFrame(rows)
    df_full["Contribution"] = np.abs(df_full["shap1"] - df_full["shap2"])
    df_full["Direction"] = np.where(df_full["shap1"] > df_full["shap2"], "-", "+")

    top_per_site = (
        df_full.sort_values("Contribution", ascending=False)
        .groupby("AS_Site")
        .head(N_TOP_RBPS)
    )

    summary = top_per_site[["RBP", "AS_Site", "Contribution", "Direction"]].reset_index(drop=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary.to_csv(output_path, index=False)
    return summary


def run_strategy(gene_exp_df, splice_cat_raw, splice_prob_raw, strategy, output_dir, max_sites=None, use_shap=True):
    print(f"  Applying transformation ...")
    splice_df, _ = apply_strategy(splice_cat_raw, strategy, splice_prob_raw=splice_prob_raw)

    sites = list(splice_df.columns)
    if max_sites is not None:
        sites = sites[:max_sites]

    X_full = gene_exp_df.reindex(splice_df.index).values.astype(float)
    feature_names = list(gene_exp_df.columns)
    na_raw = splice_cat_raw == 0

    results = []
    shap_records = []
    n_total = len(sites)

    for idx, site in enumerate(sites):
        if idx % 20 == 0 or idx == n_total - 1:
            print(f"    [{idx+1:>4}/{n_total}] {site}")

        y_raw = splice_df[site].values.astype(float)
        res = train_one_site(X_full, y_raw, feature_names, use_shap=use_shap)
        if res is not None:
            res["site"] = site
            res["strategy"] = strategy
            res["na_count"] = int(na_raw[site].sum())
            res["na_fraction"] = float(na_raw[site].mean())
            results.append(res)

            if res["above_chance"] and res["shap_df"] is not None:
                shap_records.append((site, res["shap_df"]))

    pct = len(results) / n_total * 100 if n_total else 0
    print(f"    -> Successfully trained: {len(results)}/{n_total} ({pct:.1f}%)")

    reg_sum_path = os.path.join(output_dir, strategy, "Regulator_contribution_summary.csv")
    if shap_records:
        reg_summary = build_reg_summary(shap_records, reg_sum_path)
        if reg_summary is not None:
            print(f"    -> Regulator summary saved: {reg_sum_path}")
    else:
        print(f"    -> No regulator summary (no above-chance sites or no SHAP)")

    return results


def validation_rate(results):
    if not results:
        return 0.0
    return sum(r["above_chance"] for r in results) / len(results)


def _violin_box(ax, data_list, positions, colors, ylim=None):
    non_empty = [d for d in data_list if len(d) > 0]
    if not non_empty:
        return
    safe = [d if len(d) > 1 else list(d) + [d[0]] for d in data_list]
    parts = ax.violinplot(safe, positions=positions, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_alpha(0.38)
        pc.set_edgecolor(colors[i % len(colors)])
    bp = ax.boxplot(
        data_list,
        positions=positions,
        widths=0.22,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=2.2),
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.78)
    for i, d in enumerate(data_list):
        if len(d) > 0:
            m = float(np.median(d))
            ax.text(
                positions[i], m + 0.04, f"{m:.3f}",
                ha="center", va="bottom", fontsize=8,
                fontweight="bold", color=colors[i % len(colors)],
            )
    if ylim:
        ax.set_ylim(*ylim)
    ax.yaxis.grid(True, linestyle=":", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)


def _style_ax(ax, xtick_labels, fontsize=8):
    ax.set_xticks(range(len(xtick_labels)))
    ax.set_xticklabels(xtick_labels, fontsize=fontsize)


def plot_accuracy_violin(all_results, strategies, colors, tick_labs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    pos = list(range(len(strategies)))

    data = [
        [r["mean_accuracy"] for r in all_results[s] if "mean_accuracy" in r]
        for s in strategies
    ]
    data = [d if d else [0] for d in data]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    _violin_box(ax, data, pos, colors, ylim=(0.0, 1.0))
    _style_ax(ax, tick_labs)
    ax.set_ylabel("Accuracy Score", fontsize=9)
    ax.set_title("Accuracy Score Comparison", fontsize=11, fontweight="bold")
    ax.axhline(0.333, color="dimgray", linestyle=":", linewidth=1, alpha=0.5, label="3-class chance")
    ax.axhline(0.250, color="dimgray", linestyle="--", linewidth=1, alpha=0.5, label="4-class chance")
    ax.legend(fontsize=7, loc="upper right")
    n_sites = ", ".join(str(len(all_results[s])) for s in strategies)
    ax.set_xlabel(f"Trainable sites per strategy: {n_sites}", fontsize=8)
    fig.tight_layout()
    out = os.path.join(output_dir, "fig1_accuracy_violin.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_trainable_events(all_trainability, strategies, colors, tick_labs, total_sites_tested, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    pos = list(range(len(strategies)))

    trainable_counts = []
    trainable_fracs = []
    for s in strategies:
        df_t = all_trainability[s]
        n_train = int(df_t["trainable"].sum())
        n_total = len(df_t)
        trainable_counts.append(n_train)
        trainable_fracs.append(n_train / n_total if n_total > 0 else 0)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    bars = ax.bar(pos, trainable_counts, color=colors, alpha=0.82, edgecolor="black", linewidth=0.8, zorder=3)
    for bar, cnt, frac in zip(bars, trainable_counts, trainable_fracs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            cnt + max(trainable_counts) * 0.03,
            f"{cnt}\n({frac:.0%})",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
    _style_ax(ax, tick_labs)
    ax.set_ylabel("Number of Trainable Sites", fontsize=9)
    ax.set_title(f"Trainable AS Events\n(of {total_sites_tested} tested)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(trainable_counts) * 1.28)
    ax.yaxis.grid(True, linestyle=":", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = os.path.join(output_dir, "fig2_trainable_events.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_sample_size_bar(all_results, strategies, colors, tick_labs, total_cells, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    pos = list(range(len(strategies)))

    median_ns = []
    for s in strategies:
        rs = all_results[s]
        median_ns.append(int(np.median([r["n_samples"] for r in rs])) if rs else 0)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    bars = ax.bar(pos, median_ns, color=colors, alpha=0.82, edgecolor="black", linewidth=0.8, zorder=3)
    for bar, v in zip(bars, median_ns):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + max(median_ns) * 0.02,
            str(v),
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
    ax.axhline(
        total_cells, color="dimgray", linestyle="--", linewidth=1.3,
        alpha=0.7, label=f"All cells ({total_cells})", zorder=2,
    )
    _style_ax(ax, tick_labs)
    ax.set_ylabel("Median Samples per Site", fontsize=9)
    ax.set_title("Trainable Sample Size\n(Median per Strategy)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(total_cells, max(median_ns)) * 1.2)
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, linestyle=":", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = os.path.join(output_dir, "fig3_sample_size_bar.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def print_summary(all_trainability, all_results, total_cells):
    print("\n" + "=" * 68)
    print("  NA ABLATION STUDY - RESULTS SUMMARY")
    print("=" * 68)
    for s in STRATEGY_ORDER:
        if s not in all_results:
            continue
        rs = all_results.get(s, [])
        df_t = all_trainability.get(s)
        label = STRATEGY_SHORT.get(s, s)
        n_trainable = int(df_t["trainable"].sum()) if df_t is not None else 0
        n_total = len(df_t) if df_t is not None else 0
        print(f"\n  {label}")
        if n_total > 0:
            print(f"    Trainable sites   : {n_trainable} / {n_total} ({n_trainable/n_total*100:.1f}%)")
        if rs:
            accs = [r["mean_accuracy"] for r in rs]
            wf1s = [r["mean_weighted_f1"] for r in rs]
            ns = [r["n_samples"] for r in rs]
            print(f"    Sites trained     : {len(rs)}")
            print(f"    Accuracy          : {np.median(accs):.3f} (median)")
            print(f"    Weighted-F1       : {np.median(wf1s):.3f} (median)")
            print(f"    Validation rate   : {validation_rate(rs):.1%}")
            print(f"    Samples/site      : {int(np.median(ns))} (median)")
        else:
            print(f"    No models trained.")
    print("=" * 68 + "\n")


def main():
    parser = argparse.ArgumentParser(description="NA Category Ablation - 3-figure output")
    parser.add_argument("--splice_file", required=True)
    parser.add_argument("--gene_file", required=True)
    parser.add_argument("--output_dir", default="na_ablation_results")
    parser.add_argument("--max_sites", type=int, default=None)
    parser.add_argument("--strategies", default="S1_exclude_na,S3_keep_na,S5_knn_impute")
    parser.add_argument("--no_shap", action="store_true")
    args = parser.parse_args()

    strategies_to_run = [s.strip() for s in args.strategies.split(",")]
    use_shap = not args.no_shap

    print("Loading data ...")
    gene_exp_df, splice_cat_raw, splice_prob_raw = load_base_data(args.splice_file, args.gene_file)
    feature_names = list(gene_exp_df.columns)
    total_cells = len(splice_cat_raw)
    total_sites = splice_cat_raw.shape[1]

    na_total = (splice_cat_raw == 0).sum().sum()
    na_global = na_total / splice_cat_raw.size * 100
    print(f"  Cells = {total_cells} | Sites = {total_sites} | RBPs = {len(feature_names)}")
    print(f"  Global NA fraction: {na_global:.2f}%\n")

    print("Phase 1: Trainability assessment (all sites) ...")
    all_trainability = {}
    for strategy in strategies_to_run:
        print(f"  {STRATEGY_SHORT.get(strategy, strategy)} ...", end=" ")
        splice_df, _ = apply_strategy(splice_cat_raw, strategy, splice_prob_raw=splice_prob_raw)
        df_t = assess_trainability(splice_df)
        all_trainability[strategy] = df_t
        n_ok = df_t["trainable"].sum()
        print(f"{n_ok} / {len(df_t)} trainable ({n_ok/len(df_t)*100:.1f}%)")

    total_sites_tested = len(all_trainability[strategies_to_run[0]])

    print(f"\nPhase 2: Model training (max_sites={args.max_sites}) ...")
    all_results = {}
    for strategy in strategies_to_run:
        print(f"\n{'='*58}")
        print(f"  Strategy: {STRATEGY_SHORT.get(strategy, strategy)}")
        print(f"{'='*58}")
        all_results[strategy] = run_strategy(
            gene_exp_df, splice_cat_raw, splice_prob_raw, strategy,
            args.output_dir, max_sites=args.max_sites, use_shap=use_shap,
        )

    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    for s, rs in all_results.items():
        for r in rs:
            keep = {k: v for k, v in r.items()
                    if k not in ("consensus_top_rbps", "feature_importance", "shap_df")}
            rows.append(keep)
    csv_path = os.path.join(args.output_dir, "na_ablation_metrics.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nMetrics CSV saved -> {csv_path}")

    train_rows = []
    for s, df_t in all_trainability.items():
        df_copy = df_t.copy()
        df_copy["strategy"] = s
        train_rows.append(df_copy)
    train_csv = os.path.join(args.output_dir, "na_ablation_trainability.csv")
    pd.concat(train_rows).to_csv(train_csv, index=False)
    print(f"Trainability CSV saved -> {train_csv}")

    print_summary(all_trainability, all_results, total_cells)

    fig_dir = os.path.join(args.output_dir, "figures")
    strategies = [s for s in STRATEGY_ORDER if s in all_results]
    colors = PALETTE[:len(strategies)]
    tick_labs = [STRATEGY_LABELS.get(s, s) for s in strategies]

    print("\nGenerating figures ...")
    plot_accuracy_violin(all_results, strategies, colors, tick_labs, fig_dir)
    plot_trainable_events(all_trainability, strategies, colors, tick_labs, total_sites_tested, fig_dir)
    plot_sample_size_bar(all_results, strategies, colors, tick_labs, total_cells, fig_dir)

    print(f"\nAll outputs saved to: {args.output_dir}")
    print("NA ablation complete.\n")


if __name__ == "__main__":
    main()