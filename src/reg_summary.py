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
    # If the same filename appears multiple times, the first occurrence or the first occurrence after deduplication is selected.
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
            site = os.path.basename(os.path.dirname(file))  # e.g., "column_3"
            df["site"] = site
            results.append(df)
        except Exception as e:
            print(f"[WARN] Error reading SHAP {file}: {e}")

    if not results:
        return None

    return pd.concat(results, ignore_index=True)


def process_folder(args):

    folder, base_dir, model_type, shap_threshold, acc_threshold = args
    folder_path = os.path.join(base_dir, folder)

    filenames, filename_to_site = process_metric_files(folder_path, model_type, acc_threshold)
    if not filenames:
        return folder, None, None, f"No valid metric files or no site above acc threshold in {folder}"

    shap_df = process_shap_files(folder_path, model_type, filenames)
    if shap_df is None:
        return folder, None, None, f"No valid SHAP files in {folder}"


    shap_df.columns = ["gene", "shap1", "shap2", "site"] 
    shap_df["sum_key"] = shap_df["gene"] + "_" + shap_df["site"]


    shap_mask = (shap_df["shap1"] > shap_threshold) | (shap_df["shap2"] > shap_threshold)
    shap_df = shap_df[shap_mask].copy()
    if shap_df.empty:
        return folder, None, filename_to_site, f"No SHAP values above threshold in {folder}"

    return folder, shap_df[["sum_key", "shap1", "shap2", "site"]], filename_to_site, None


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate k1-k5 training outputs and generate Regulator_contribution_summary.csv."
    )
    parser.add_argument(
        "-b",
        "--base_dir",
        type=str,
        default=".",
        help="Input root directory (contains k1-k5 subdirectories). Defaults to the current directory.",
    )
    parser.add_argument(
        "-f",
        "--folders",
        type=str,
        default="k1,k2,k3,k4,k5",
        help='List of subdirectories to be summarized, comma-separated, default "k1,k2,k3,k4,k5".',
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lgbm",
        help="Model type subdirectory name, default lgbm (keep consistent with training output).",
    )
    parser.add_argument(
        "--shap-threshold",
        type=float,
        default=10.0,
        help="SHAP value threshold, default 10.0.",
    )
    parser.add_argument(
        "--acc-threshold",
        type=float,
        default=0.85,
        help="Accuracy threshold, default 0.85.",
    )
    parser.add_argument(
        "--min-common-folders",
        type=int,
        default=5,
        help="The minimum number of folders that sum_key appears in, default 5.",
    )
    parser.add_argument(
        "-j",
        "--processes",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Number of parallel processes, default min(4, number of CPU cores).",
    )
    args = parser.parse_args()

    base_dir = args.base_dir
    folders = [f.strip() for f in args.folders.split(",") if f.strip()]
    model_type = args.model
    shap_threshold = args.shap_threshold
    acc_threshold = args.acc_threshold
    min_common = args.min_common_folders
    num_procs = max(1, args.processes)

    start_time = time.time()
    print(f"Starting parallel processing with {num_procs} processes...")


    folder_dfs = {}
    global_filename_to_site = {}
    tasks = [(folder, base_dir, model_type, shap_threshold, acc_threshold) for folder in folders]
    with ProcessPoolExecutor(max_workers=num_procs) as executor:
        future_to_folder = {executor.submit(process_folder, t): t[0] for t in tasks}
        for future in as_completed(future_to_folder):
            folder = future_to_folder[future]
            try:
                f, df, fname2site, warn = future.result()
                if df is not None:
                    folder_dfs[f] = df
                    print(f"Processed {f} with {len(df)} rows (after threshold filtering)")
                else:
                    print(f"Skipped {f}: {warn}")

                if fname2site:
                    for k, v in fname2site.items():
                        if k not in global_filename_to_site:
                            global_filename_to_site[k] = v
                        elif global_filename_to_site[k] != v:

                            print(f"[WARN] Inconsistent site names: {k} -> '{global_filename_to_site[k]}' vs '{v}' (Keep the former)")
            except Exception as e:
                print(f"[ERROR] processing {folder}: {e}")

    if not folder_dfs:
        print("No valid data found in any folder")
        return


    print("Creating index dictionaries...")
    index_dicts = {}
    for folder, df in folder_dfs.items():
        index_dicts[folder] = df.set_index("sum_key")[["shap1", "shap2"]].to_dict("index")


    print(f"Finding keys appearing in at least {min_common} folders...")
    key_counter = Counter()
    for df in folder_dfs.values():
        key_counter.update(set(df["sum_key"]))

    common_keys = {key for key, count in key_counter.items() if count >= min_common}
    if not common_keys:
        print(f"No keys found in at least {min_common} folders")
        return
    print(f"Found {len(common_keys)} keys appearing in at least {min_common} folders")


    aggregated_data = []
    site_info = {}  
    for key in common_keys:
        shap1_list, shap2_list = [], []
        for folder in folders:
            if folder in index_dicts and key in index_dicts[folder]:
                vals = index_dicts[folder][key]
                shap1_list.append(vals["shap1"])
                shap2_list.append(vals["shap2"])
                if key not in site_info:

                    site_info[key] = key.split("_", 1)[1]
        avg_shap1 = np.mean(shap1_list) if shap1_list else 0.0
        avg_shap2 = np.mean(shap2_list) if shap2_list else 0.0
        aggregated_data.append(
            {"sum_key": key, "shap1": avg_shap1, "shap2": avg_shap2, "site": site_info[key]}
        )

    del folder_dfs, index_dicts
    gc.collect()

    if not aggregated_data:
        print("No aggregated data found")
        return

    mergedf = pd.DataFrame(aggregated_data)


    print("Processing sites...")
    shap1_mask = mergedf["shap1"] > shap_threshold
    shap2_mask = mergedf["shap2"] > shap_threshold

    dfi = mergedf[shap1_mask].copy()
    dfi["type"] = "-"

    dfe = mergedf[shap2_mask].copy()
    dfe["type"] = "+"

    finaldf = pd.concat([dfi, dfe], ignore_index=True)


    if global_filename_to_site:

        finaldf["site"] = finaldf["site"].map(global_filename_to_site).fillna(finaldf["site"])
    else:
        print("[WARN] No 'column_*' -> real site mapping was obtained, retaining the original site index.")


    before_rows = len(finaldf)
    finaldf = finaldf[
        ((finaldf["type"] == "+") & (finaldf["shap1"] < finaldf["shap2"])) |
        ((finaldf["type"] == "-") & (finaldf["shap1"] > finaldf["shap2"]))
    ].copy()
    after_rows = len(finaldf)
    dropped = before_rows - after_rows
    if dropped > 0:
        print(f"Filtered {dropped} rows that did not satisfy shap inequality constraints")

    regsum = pd.DataFrame()
    regsum['RBP'] = finaldf['sum_key'].str.split('_').str[0]
    regsum['AS_Site'] = finaldf['site']
    regsum['Contribution'] = abs(finaldf['shap1'] - finaldf['shap2'])
    regsum['Direction'] = finaldf['type']

    regsum = regsum.sort_values(by='Contribution', ascending=False).reset_index(drop=True)


    output_file = os.path.join(base_dir, "SHAP_summary.csv")
    output_file2 = os.path.join(base_dir, "Regulator_contribution_summary.csv")
    finaldf.to_csv(output_file, index=False)
    regsum.to_csv(output_file2, index=False)
    print(f"Saved final result with {len(finaldf)} rows to {output_file}")

    elapsed = time.time() - start_time
    print(f"Total processing time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()