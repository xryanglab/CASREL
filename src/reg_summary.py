import os
import argparse
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
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
    except Exception as e:
        print(f"[WARN] Error reading metric CSV {metric_dir}: {e}")
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
        except Exception as e:
            print(f"[WARN] Error reading SHAP {file}: {e}")

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
        return (
            folder,
            None,
            None,
            f"No valid metric files or no site above acc threshold in {folder}",
        )

    shap_df = process_shap_files(folder_path, model_type, filenames)
    if shap_df is None:
        return folder, None, None, f"No valid SHAP files in {folder}"


    shap_df["sum_key"] = shap_df["gene"] + "_" + shap_df["site"]

    # When shap_threshold < 0, skip filtering (adaptive mode loads all data)
    if shap_threshold >= 0:
        shap_mask = (shap_df["shap1"] - shap_df["shap2"]).abs() > shap_threshold
        shap_df = shap_df[shap_mask].copy()

    if shap_df.empty:
        return (
            folder,
            None,
            filename_to_site,
            f"No SHAP values after filtering in {folder}",
        )


    if shap_df["sum_key"].duplicated().any():
        print(f"[WARN] Duplicate sum_key still present in {folder}, dropping duplicates")
        shap_df = shap_df.drop_duplicates(subset=["sum_key"], keep="first")

    return (
        folder,
        shap_df[["sum_key", "shap1", "shap2", "site"]],
        filename_to_site,
        None,
    )


# ---------------------------------------------------------------------------
# Adaptive Thresholding via Cumulative Contribution (based on |shap1 - shap2|)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

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
    # ---- Threshold mode ----
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
    # ---- Legacy / override ----
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
    # ---- Contribution post-filter ----
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
        effective_shap_threshold = -1.0  # disable per-fold filtering
        print(f"Threshold mode : {args.threshold_mode}")
        print(f"  Importance metric: |shap1 - shap2| (absolute contribution)")
        if adaptive_method == "cumulative":
            print(f"  Cumulative target : {args.cumulative_pct:.0%}")
    else:
        effective_shap_threshold = args.shap_threshold
        adaptive_method = None
        print(f"Threshold mode : fixed (|shap1 - shap2| > {effective_shap_threshold})")

    start_time = time.time()
    print(f"Starting parallel processing with {num_procs} processes...")

    # ------------------------------------------------------------------ 1
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
                    print(f"  Loaded {f}: {len(df)} rows")
                else:
                    print(f"  Skipped {f}: {warn}")
                if fname2site:
                    for k, v in fname2site.items():
                        if k not in global_filename_to_site:
                            global_filename_to_site[k] = v
                        elif global_filename_to_site[k] != v:
                            print(
                                f"[WARN] Inconsistent site names: {k} -> "
                                f"'{global_filename_to_site[k]}' vs '{v}' "
                                "(keeping the former)"
                            )
            except Exception as e:
                print(f"[ERROR] processing {folder}: {e}")

    if not folder_dfs:
        print("No valid data found in any folder")
        return

    # ------------------------------------------------------------------ 2
    print("Creating index dictionaries...")
    index_dicts = {}
    site_dicts = {}
    for folder, df in folder_dfs.items():

        if df["sum_key"].duplicated().any():
            print(f"[WARN] Duplicate sum_key in {folder} before set_index, dropping duplicates")
            df = df.drop_duplicates(subset=["sum_key"], keep="first")
        index_dicts[folder] = (
            df.set_index("sum_key")[["shap1", "shap2"]].to_dict("index")
        )
        site_dicts[folder] = df.set_index("sum_key")["site"].to_dict()

    print(f"Finding keys in >= {min_common} folders...")
    key_counter = Counter()
    for df in folder_dfs.values():
        key_counter.update(set(df["sum_key"]))

    common_keys = {k for k, c in key_counter.items() if c >= min_common}
    if not common_keys:
        print(f"No keys found in >= {min_common} folders")
        return
    print(f"  {len(common_keys)} common keys found")

    # ------------------------------------------------------------------ 3
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
        print("No aggregated data")
        return

    mergedf = pd.DataFrame(aggregated_data)
    print(f"Aggregated: {len(mergedf)} gene-site pairs")

    # ------------------------------------------------------------------ 4
    if is_adaptive:
        print(f"\nApplying adaptive threshold ({adaptive_method}) based on |shap1 - shap2|...")
        mergedf, threshold_info = apply_adaptive_threshold(
            mergedf,
            method=adaptive_method,
            cumulative_pct=args.cumulative_pct,
        )
        if mergedf.empty:
            print("No data remained after adaptive thresholding")
            return

        all_thresh = [v["adaptive_threshold"] for v in threshold_info.values()]
        all_nsel = [v["n_selected"] for v in threshold_info.values()]
        all_cum = [v["cumulative_contribution"] for v in threshold_info.values()]

        print(f"\n{'=' * 68}")
        print(f"  ADAPTIVE THRESHOLD SUMMARY  (method = {adaptive_method})")
        print(f"  Importance metric: |shap1 - shap2|")
        print(f"{'=' * 68}")
        print(f"  Sites processed          : {len(threshold_info)}")
        print(f"  Avg  threshold           : {np.mean(all_thresh):.4f}")
        print(f"  Med  threshold           : {np.median(all_thresh):.4f}")
        print(
            f"  Range threshold          : "
            f"[{np.min(all_thresh):.4f}, {np.max(all_thresh):.4f}]"
        )
        print(f"  Avg  genes selected/site : {np.mean(all_nsel):.1f}")
        print(f"  Med  genes selected/site : {np.median(all_nsel):.0f}")
        print(f"  Avg  cumulative contrib  : {np.mean(all_cum):.2%}")
        print(f"  Gene-site pairs retained : {len(mergedf)}")
        print(f"{'=' * 68}\n")

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
        print(f"Saved per-site thresholds -> {thresh_path}")
    else:

        before = len(mergedf)
        shap_t = args.shap_threshold
        mask = (mergedf["shap1"] - mergedf["shap2"]).abs() > shap_t
        mergedf = mergedf[mask].copy()
        print(f"Fixed threshold (|shap1-shap2| > {shap_t}): {before} -> {len(mergedf)} gene-site pairs")
        if mergedf.empty:
            print("No data after fixed threshold")
            return

    # ------------------------------------------------------------------ 5
    print("Assigning direction based on shap1 vs shap2...")

    mergedf = mergedf[mergedf["shap1"] != mergedf["shap2"]].copy()
    mergedf["type"] = np.where(mergedf["shap1"] > mergedf["shap2"], "-", "+")
    finaldf = mergedf

    if global_filename_to_site:
        finaldf["site"] = (
            finaldf["site"]
            .map(global_filename_to_site)
            .fillna(finaldf["site"])
        )

    # ------------------------------------------------------------------ 6
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

    # ------------------------------------------------------------------ 6.5 Contribution 后过滤
    contrib_filter = args.contribution_filter
    if contrib_filter != "none" and len(regsum) > 0:
        contributions = regsum["Contribution"].values
        if contrib_filter == "knee":
            sorted_contrib = np.sort(contributions)[::-1]
            knee_idx = find_knee_point(sorted_contrib)
            knee_value = sorted_contrib[knee_idx] if knee_idx < len(sorted_contrib) else 0.0
            min_contrib = knee_value
            print(f"Global contribution knee point: {min_contrib:.4f}")
        elif contrib_filter.startswith("quantile:"):
            try:
                q = float(contrib_filter.split(":")[1])
                if q < 0 or q > 1:
                    raise ValueError("Quantile must be between 0 and 1")
                min_contrib = np.quantile(contributions, q)
                print(f"Using contribution quantile {q:.2f} = {min_contrib:.4f}")
            except Exception as e:
                print(f"[WARN] Invalid quantile format '{contrib_filter}': {e}. Skipping filter.")
                min_contrib = 0.0
        else:
            print(f"[WARN] Unknown contribution filter '{contrib_filter}'. Skipping.")
            min_contrib = 0.0

        if min_contrib > 0.0:
            before = len(regsum)
            regsum = regsum[regsum["Contribution"] >= min_contrib]
            print(f"Contribution filter: {before} -> {len(regsum)} rows "
                  f"(min_contrib >= {min_contrib:.4f})")

    if top_per_site:
        print(
            f"Applying per-site top-{top_n} selection for each Direction..."
        )
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

    # ------------------------------------------------------------------ 7
    output_file = os.path.join(base_dir, "SHAP_summary.csv")
    output_file2 = os.path.join(base_dir, "Regulator_contribution_summary.csv")
    finaldf.to_csv(output_file, index=False)
    regsum.to_csv(output_file2, index=False)
    print(
        f"Saved SHAP_summary.csv           : {len(finaldf)} rows -> {output_file}"
    )
    print(
        f"Saved Regulator_contribution.csv : {len(regsum)} rows -> {output_file2}"
    )

    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()