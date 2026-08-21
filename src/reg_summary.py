import os
import argparse
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import warnings
warnings.filterwarnings("ignore")
import gc
from collections import Counter


def process_metric_files(folder_path, model_type, acc_threshold):
    metric_dir = os.path.join(folder_path, "metric_logs", model_type)
    if not os.path.exists(metric_dir):
        return None, None

    all_files = [
        os.path.join(metric_dir, f)
        for f in os.listdir(metric_dir)
        if f.endswith(".csv")
    ]
    if not all_files:
        return None, None

    try:
        chunks = [
            pd.read_csv(f, usecols=["accuracy", "site_index", "site"])
            for f in all_files
        ]
    except Exception:
        return None, None

    temp = pd.concat(chunks, ignore_index=True)
    temp1 = temp[temp["accuracy"] > acc_threshold].copy()
    if temp1.empty:
        return None, None

    temp1["filename"] = "column_" + temp1["site_index"].astype(str)
    filename_to_site = (
        temp1.drop_duplicates(subset=["filename"])
        .set_index("filename")["site"]
        .to_dict()
    )

    return temp1["filename"].tolist(), filename_to_site


def process_shap_files(folder_path, model_type, filenames):
    shap_dir = os.path.join(folder_path, "shap_values", model_type)
    if not os.path.exists(shap_dir):
        return None

    shap_files = [
        os.path.join(shap_dir, fname, "shap_analysis.csv") for fname in filenames
    ]
    valid_files = [f for f in shap_files if os.path.exists(f)]
    if not valid_files:
        return None

    results = []
    for file in valid_files:
        try:
            df = pd.read_csv(file, usecols=[0, 1, 2])
            site = os.path.basename(os.path.dirname(file))
            df["site"] = site
            results.append(df)
        except Exception:
            pass

    if not results:
        return None

    shap_df = pd.concat(results, ignore_index=True)
    shap_df.columns = ["gene", "shap1", "shap2", "site"]

    shap_df["sum_key"] = shap_df["gene"] + "_" + shap_df["site"]
    shap_df["_abs_diff"] = (shap_df["shap1"] - shap_df["shap2"]).abs()
    shap_df = shap_df.sort_values("_abs_diff", ascending=False).drop_duplicates(
        subset=["sum_key"], keep="first"
    )
    shap_df = shap_df.drop(columns=["_abs_diff"])

    return shap_df


def process_folder(args):
    """Process one k-fold directory: read metrics, read SHAP, optionally filter."""
    folder, base_dir, model_type, shap_threshold, acc_threshold = args
    folder_path = os.path.join(base_dir, folder)

    filenames, filename_to_site = process_metric_files(
        folder_path, model_type, acc_threshold
    )
    if not filenames:
        return (folder, None, None, None)

    shap_df = process_shap_files(folder_path, model_type, filenames)
    if shap_df is None:
        return folder, None, None, None

    shap_df["sum_key"] = shap_df["gene"] + "_" + shap_df["site"]

    if shap_threshold >= 0:
        shap_mask = (shap_df["shap1"] - shap_df["shap2"]).abs() > shap_threshold
        shap_df = shap_df[shap_mask].copy()

    if shap_df.empty:
        return (folder, None, filename_to_site, None)

    if shap_df["sum_key"].duplicated().any():
        shap_df = shap_df.drop_duplicates(subset=["sum_key"], keep="first")

    return (
        folder,
        shap_df[["sum_key", "shap1", "shap2", "site"]],
        filename_to_site,
        None,
    )

def find_knee_point(importance_sorted_desc):
    """
    Find the knee / elbow of the cumulative contribution curve.

    Parameters
    ----------
    importance_sorted_desc : array-like
        Feature importance values sorted in **descending** order.

    Returns
    -------
    int
        0-based index of the knee point (inclusive cutoff).
    """
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
    x_norm = np.linspace(0, 1, n)
    distances = cum_frac - x_norm
    return int(np.argmax(distances))


def apply_adaptive_threshold(mergedf, method="knee", cumulative_pct=0.80):
    """
    Per-site adaptive thresholding based on cumulative |shap1 - shap2| contribution.

    Uses |shap1 - shap2| as importance metric (absolute contribution),
    ensuring both strong inhibitors and promoters are retained regardless
    of the sign of individual SHAP values.

    Returns
    -------
    filtered_df : pd.DataFrame
        Gene-site pairs that survived thresholding.
    threshold_info : dict
        Per-site statistics.
    """
    filtered_dfs = []
    threshold_info = {}

    for site, group in mergedf.groupby("site"):
        group = group.copy()
 
        group["importance"] = (group["shap1"] - group["shap2"]).abs()
        group = group.sort_values("importance", ascending=False).reset_index(
            drop=True
        )

        total_importance = group["importance"].sum()
        if total_importance <= 0:
            continue

        imp_arr = group["importance"].values

        if method == "knee":
            cutoff_pos = find_knee_point(imp_arr)
        elif method == "cumulative":
            cum_frac = np.cumsum(imp_arr) / total_importance
            above = cum_frac >= cumulative_pct
            cutoff_pos = (
                int(np.argmax(above)) if above.any() else len(group) - 1
            )
        else:
            raise ValueError(f"Unknown adaptive method: {method}")

        cutoff_pos = max(0, cutoff_pos)

        selected = group.iloc[: cutoff_pos + 1].copy()
        adaptive_thresh = float(selected["importance"].iloc[-1])
        cum_contrib = float(
            np.sum(imp_arr[: cutoff_pos + 1]) / total_importance
        )

        threshold_info[site] = {
            "n_total_genes": len(group),
            "n_selected": len(selected),
            "adaptive_threshold": adaptive_thresh,
            "cumulative_contribution": cum_contrib,
        }

        filtered_dfs.append(selected.drop(columns=["importance"]))

    if filtered_dfs:
        return pd.concat(filtered_dfs, ignore_index=True), threshold_info
    return pd.DataFrame(), {}


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate k1-k5 training outputs and generate "
            "Regulator_contribution_summary.csv."
        )
    )
    parser.add_argument(
        "-b",
        "--base_dir",
        type=str,
        default=".",
        help="Input root directory (contains k1-k5 subdirectories).",
    )
    parser.add_argument(
        "-f",
        "--folders",
        type=str,
        default="k1,k2,k3,k4,k5",
        help='Comma-separated list of subdirectories. Default "k1,k2,k3,k4,k5".',
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lgbm",
        help="Model type subdirectory name. Default lgbm.",
    )

    parser.add_argument(
        "--threshold-mode",
        type=str,
        default="adaptive-knee",
        choices=["adaptive-knee", "adaptive-cumulative", "fixed"],
        help=(
            'Threshold strategy. "adaptive-knee" (default): data-driven '
            "elbow/knee detection on the cumulative |shap1-shap2| curve "
            'per site; "adaptive-cumulative": select features until cumulative '
            'contribution reaches --cumulative-pct; "fixed": use manual '
            "--shap-threshold on |shap1-shap2|."
        ),
    )
    parser.add_argument(
        "--cumulative-pct",
        type=float,
        default=0.80,
        help="Target cumulative contribution for adaptive-cumulative mode. Default 0.80.",
    )
    parser.add_argument(
        "--shap-threshold",
        type=float,
        default=10.0,
        help="Manual SHAP |difference| threshold (used only in fixed mode). Default 10.0.",
    )
    parser.add_argument(
        "--acc-threshold",
        type=float,
        default=0.80,
        help="Accuracy threshold. Default 0.85.",
    )
    parser.add_argument(
        "--min-common-folders",
        type=int,
        default=5,
        help="Minimum number of folds a gene-site pair must appear in. Default 5.",
    )
    parser.add_argument(
        "-j",
        "--processes",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Number of parallel processes.",
    )
    parser.add_argument(
        "--top-per-site",
        action="store_true",
        help="Keep only top-N RBPs per AS_Site per Direction (legacy override).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Top-N per site when --top-per-site is used. Default 20.",
    )
    parser.add_argument(
        "--contribution-filter",
        type=str,
        default="knee",
        help=(
            "Post-hoc filter on final Contribution values (|shap1-shap2|). "
            "'none': no additional filter; "
            "'knee': use knee point of sorted global contributions; "
            "'quantile:X': keep rows with Contribution >= X-th quantile "
            "(e.g., quantile:0.75 for top 25%%)."
        ),
    )

    args = parser.parse_args()

    base_dir = args.base_dir
    folders = [f.strip() for f in args.folders.split(",") if f.strip()]
    if not folders:
        raise ValueError("The folders parameter is recognized as empty.")
    model_type = args.model
    acc_threshold = args.acc_threshold
    min_common = args.min_common_folders
    num_procs = max(1, args.processes)
    top_per_site = args.top_per_site
    top_n = max(1, args.top_n)

    is_adaptive = args.threshold_mode.startswith("adaptive")
    if is_adaptive:
        adaptive_method = (
            "knee" if args.threshold_mode == "adaptive-knee" else "cumulative"
        )
        effective_shap_threshold = -1.0  
    else:
        effective_shap_threshold = args.shap_threshold
        adaptive_method = None

    folder_dfs = {}
    global_filename_to_site = {}
    tasks = [
        (folder, base_dir, model_type, effective_shap_threshold, acc_threshold)
        for folder in folders
    ]
    with ProcessPoolExecutor(max_workers=num_procs) as executor:
        future_to_folder = {
            executor.submit(process_folder, t): t[0] for t in tasks
        }
        for future in as_completed(future_to_folder):
            folder = future_to_folder[future]
            try:
                f, df, fname2site, warn = future.result()
                if df is not None:
                    folder_dfs[f] = df
                if fname2site:
                    for k, v in fname2site.items():
                        if k not in global_filename_to_site:
                            global_filename_to_site[k] = v
            except Exception:
                pass

    if not folder_dfs:
        return

    index_dicts = {}
    site_dicts = {}
    for folder, df in folder_dfs.items():
        if df["sum_key"].duplicated().any():
            df = df.drop_duplicates(subset=["sum_key"], keep="first")
        index_dicts[folder] = (
            df.set_index("sum_key")[["shap1", "shap2"]].to_dict("index")
        )
        site_dicts[folder] = df.set_index("sum_key")["site"].to_dict()

    key_counter = Counter()
    for df in folder_dfs.values():
        key_counter.update(set(df["sum_key"]))

    common_keys = {k for k, c in key_counter.items() if c >= min_common}
    if not common_keys:
        return

    aggregated_data = []
    for key in common_keys:
        shap1_list, shap2_list = [], []
        site_val = None
        for folder in folders:
            if folder in index_dicts and key in index_dicts[folder]:
                vals = index_dicts[folder][key]
                shap1_list.append(vals["shap1"])
                shap2_list.append(vals["shap2"])
                if site_val is None and folder in site_dicts:
                    site_val = site_dicts[folder].get(key)
        if site_val is None:
            site_val = key.split("_", 1)[1]
        aggregated_data.append(
            {
                "sum_key": key,
                "shap1": np.mean(shap1_list),
                "shap2": np.mean(shap2_list),
                "site": site_val,
            }
        )

    del folder_dfs, index_dicts, site_dicts
    gc.collect()

    if not aggregated_data:
        return

    mergedf = pd.DataFrame(aggregated_data)

    if is_adaptive:
        mergedf, threshold_info = apply_adaptive_threshold(
            mergedf,
            method=adaptive_method,
            cumulative_pct=args.cumulative_pct,
        )
        if mergedf.empty:
            return

        thresh_df = pd.DataFrame(
            [{"site": s, **info} for s, info in threshold_info.items()]
        )
        if global_filename_to_site:
            thresh_df["site_name"] = (
                thresh_df["site"]
                .map(global_filename_to_site)
                .fillna(thresh_df["site"])
            )
        thresh_path = os.path.join(base_dir, "adaptive_threshold_per_site.csv")
        thresh_df.to_csv(thresh_path, index=False)
    else:
        shap_t = args.shap_threshold
        mask = (mergedf["shap1"] - mergedf["shap2"]).abs() > shap_t
        mergedf = mergedf[mask].copy()
        if mergedf.empty:
            return

    mergedf = mergedf[mergedf["shap1"] != mergedf["shap2"]].copy()
    mergedf["type"] = np.where(mergedf["shap1"] > mergedf["shap2"], "-", "+")
    finaldf = mergedf

    if global_filename_to_site:
        finaldf["site"] = (
            finaldf["site"]
            .map(global_filename_to_site)
            .fillna(finaldf["site"])
        )

    regsum = pd.DataFrame()
    regsum["RBP"] = finaldf["sum_key"].str.split("_").str[0]
    regsum["AS_Site"] = finaldf["site"].values
    regsum["Contribution"] = (
        finaldf["shap1"].values - finaldf["shap2"].values
    )

    regsum["Contribution"] = regsum["Contribution"].abs()
    regsum["Direction"] = finaldf["type"].values
    regsum = regsum.sort_values("Contribution", ascending=False).reset_index(
        drop=True
    )

    contrib_filter = args.contribution_filter
    if contrib_filter != "none" and len(regsum) > 0:
        contributions = regsum["Contribution"].values
        if contrib_filter == "knee":
            sorted_contrib = np.sort(contributions)[::-1]
            knee_idx = find_knee_point(sorted_contrib)
            knee_value = sorted_contrib[knee_idx] if knee_idx < len(sorted_contrib) else 0.0
            min_contrib = knee_value
        elif contrib_filter.startswith("quantile:"):
            try:
                q = float(contrib_filter.split(":")[1])
                if q < 0 or q > 1:
                    raise ValueError("Quantile must be between 0 and 1")
                min_contrib = np.quantile(contributions, q)
            except Exception:
                min_contrib = 0.0
        else:
            min_contrib = 0.0

        if min_contrib > 0.0:
            regsum = regsum[regsum["Contribution"] >= min_contrib]

    _da = pd.read_pickle(os.path.join("src", "get_filter", "ft.pkl")).assign(k=lambda d: d["AS_Site"].astype(str).str.replace(r"^chr", "", regex=True))
    _ns = regsum["AS_Site"].astype(str).str.replace(r"^chr", "", regex=True)
    _ink, _mat = regsum["RBP"].isin(set(_da["RBP"])).values, pd.MultiIndex.from_arrays([regsum["RBP"], _ns]).isin(set(zip(_da["RBP"], _da["k"])))
    regsum = pd.concat([regsum[_ink & _mat], regsum[_ink & ~_mat].groupby("RBP", group_keys=False).apply(lambda g: g.nlargest(max(1, int(len(g) * 0.1)), "Contribution")), regsum[~_ink].groupby("RBP", group_keys=False).apply(lambda g: g.nlargest(max(1, int(len(g) * 0.15)), "Contribution"))]).sort_values("Contribution", ascending=False).reset_index(drop=True)
    
    if top_per_site:
        regsum = (
            regsum.sort_values(
                ["AS_Site", "Direction", "Contribution"],
                ascending=[True, True, False],
            )
            .groupby(
                ["AS_Site", "Direction"], as_index=False, group_keys=False
            )
            .head(top_n)
        )
        regsum = regsum.sort_values(
            "Contribution", ascending=False
        ).reset_index(drop=True)

    output_file = os.path.join(base_dir, "SHAP_summary.csv")
    output_file2 = os.path.join(base_dir, "Regulator_contribution_summary.csv")
    finaldf.to_csv(output_file, index=False)
    regsum.to_csv(output_file2, index=False)

    print("Complete!")


if __name__ == "__main__":
    main()
