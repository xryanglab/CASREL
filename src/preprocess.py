import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def fill_na_within_group(df, groupby):

    df_filled = df.copy()
    sample_cols = [col for col in df.columns if col not in ['start', 'end']]

    for col in sample_cols:
        df_filled[col] = df.groupby(groupby)[col].transform(
            lambda x: x.fillna(0) if not x.isna().all() else x
        )
    return df_filled


def to_prob(df, groupby):
    sums = df.groupby(groupby).agg(sum)
    sums = pd.merge(df, sums, how="left", on=groupby)
    sums = sums.drop(columns=["start", "end"])
    df_copy = df.copy()
    df_copy = df_copy.drop(columns=["start", "end"])
    num_samples = df_copy.shape[1]
    sums = sums.iloc[:, num_samples:]
    probs = df_copy.values / sums.values
    df_result = df.copy()
    df_result.iloc[:, :-2] = probs
    df_result = df_result.sort_values(by=groupby)
    return df_result


def norm_only(df_path, groupby):
    print(f"Reading data from {df_path}...")
    df = pd.read_csv(df_path + "_" + groupby + ".csv", index_col=False)
    df = df.set_index("Site")

    df_filled = fill_na_within_group(df, groupby)
    df_prob = to_prob(df_filled, groupby=groupby)
    df_prob = df_prob.drop(["start", "end"], axis=1)
    print("Done.")
    return df_prob


def categorize_splice_prob(prob, low_threshold=0.4, high_threshold=0.6):

    if pd.isna(prob):
        return 0
    if prob < low_threshold:
        return 1
    elif prob < high_threshold:
        return 2
    else:
        return 3


def load_and_transform(splice_file, gene_file, low_threshold=0.4, high_threshold=0.6):
    splice_start_df = norm_only(splice_file, "start")
    splice_end_df   = norm_only(splice_file, "end")
    gene_exp_df     = pd.read_csv(gene_file, index_col=0)

    common_columns = list(
        set(splice_start_df.columns) & set(splice_end_df.columns) & set(gene_exp_df.columns)
    )

    splice_start_df = splice_start_df[common_columns]
    splice_end_df   = splice_end_df[common_columns]
    gene_exp_df     = gene_exp_df[common_columns]

    # 应用分类函数（NaN 映射为 0）
    splice_start = splice_start_df.applymap(
        lambda p: categorize_splice_prob(p, low_threshold, high_threshold)
    )
    splice_end = splice_end_df.applymap(
        lambda p: categorize_splice_prob(p, low_threshold, high_threshold)
    )

    splice_df = pd.concat([splice_start, splice_end], axis=0)
    splice_df = splice_df.fillna(0).astype(int)
    splice_df = splice_df.T
    gene_exp_df = gene_exp_df.T

    train_index, test_index = train_test_split(
        np.arange(len(splice_df)), test_size=0.2, random_state=42
    )
    train_splice_df    = splice_df.iloc[train_index, :]
    test_splice_df     = splice_df.iloc[test_index, :]
    train_gene_exp_df  = gene_exp_df.iloc[train_index, :]
    test_gene_exp_df   = gene_exp_df.iloc[test_index, :]

    return {
        "splice_df":          splice_df,
        "gene_exp_df":        gene_exp_df,
        "train_splice_df":    train_splice_df,
        "test_splice_df":     test_splice_df,
        "train_gene_exp_df":  train_gene_exp_df,
        "test_gene_exp_df":   test_gene_exp_df,
    }


def save_main_outputs(result_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    result_dict["splice_df"].to_csv(os.path.join(output_dir, "splice_df.csv"),                  index=False)
    result_dict["gene_exp_df"].to_csv(os.path.join(output_dir, "gene_exp_df.csv"),              index=False)
    result_dict["train_splice_df"].to_csv(os.path.join(output_dir, "train_splice_df.csv"),      index=False)
    result_dict["test_splice_df"].to_csv(os.path.join(output_dir, "test_splice_df.csv"),        index=False)
    result_dict["train_gene_exp_df"].to_csv(os.path.join(output_dir, "train_gene_exp_df.csv"),  index=False)
    result_dict["test_gene_exp_df"].to_csv(os.path.join(output_dir, "test_gene_exp_df.csv"),    index=False)


def save_kfold_splits(gene_exp_df, splice_df, output_dir):
    k = 5
    num_samples = gene_exp_df.shape[0]
    fold_size   = num_samples // k

    for i in range(k):
        start_idx = i * fold_size
        end_idx   = (i + 1) * fold_size if i < k - 1 else num_samples

        test_indices  = np.arange(start_idx, end_idx)
        train_indices = np.concatenate([np.arange(0, start_idx), np.arange(end_idx, num_samples)])

        k_test_gene_exp_df  = gene_exp_df.iloc[test_indices]
        k_train_gene_exp_df = gene_exp_df.iloc[train_indices]
        k_test_splice_df    = splice_df.iloc[test_indices]
        k_train_splice_df   = splice_df.iloc[train_indices]

        k_dir = os.path.join(output_dir, f"k{i + 1}")
        os.makedirs(k_dir, exist_ok=True)

        k_test_gene_exp_df.to_csv(os.path.join(k_dir, "test_gene_exp_df.csv"),   index=False)
        k_train_gene_exp_df.to_csv(os.path.join(k_dir, "train_gene_exp_df.csv"), index=False)
        k_test_splice_df.to_csv(os.path.join(k_dir, "test_splice_df.csv"),       index=False)
        k_train_splice_df.to_csv(os.path.join(k_dir, "train_splice_df.csv"),     index=False)
        gene_exp_df.to_csv(os.path.join(k_dir, "gene_exp_df.csv"),               index=False)
        splice_df.to_csv(os.path.join(k_dir, "splice_df.csv"),                   index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Integrated preprocessing and fixed 5-fold preparation script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--splice_file",
        type=str,
        default="../data/epall_filter",
        help=(
            "Path prefix for splice CSV files. "
            "Given prefix 'xxx', the script expects 'xxx_start.csv' and 'xxx_end.csv' to exist."
        ),
    )
    parser.add_argument(
        "--gene_file",
        type=str,
        default="../data/epall_rbp.csv",
        help="Path to the CSV file containing RBP expression data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../post_process_data/epall/",
        help="Root directory for all output files.",
    )
    parser.add_argument(
        "--low_threshold",
        type=float,
        default=0.4,
        help=(
            "Lower boundary for splice probability categorization. "
            "Probabilities strictly below this value are assigned category 1 (low). "
            "Must satisfy 0 <= low_threshold < high_threshold <= 1."
        ),
    )
    parser.add_argument(
        "--high_threshold",
        type=float,
        default=0.6,
        help=(
            "Upper boundary for splice probability categorization. "
            "Probabilities greater than or equal to this value are assigned category 3 (high). "
            "Probabilities in [low_threshold, high_threshold) are assigned category 2 (medium). "
            "Must satisfy 0 <= low_threshold < high_threshold <= 1."
        ),
    )
    args = parser.parse_args()


    if not (0.0 <= args.low_threshold < args.high_threshold <= 1.0):
        parser.error(
            f"Thresholds must satisfy 0 <= low_threshold < high_threshold <= 1. "
            f"Got: low_threshold={args.low_threshold}, high_threshold={args.high_threshold}."
        )

    print("=" * 50)
    print("Splice probability categorization thresholds:")
    print(f"  NaN                                   -> 0 (unknown)")
    print(f"  prob < {args.low_threshold:<5}                       -> 1 (low)")
    print(f"  {args.low_threshold:<5} <= prob < {args.high_threshold:<5}          -> 2 (medium)")
    print(f"  prob >= {args.high_threshold:<5}                      -> 3 (high)")
    print("=" * 50)

    result = load_and_transform(
        args.splice_file,
        args.gene_file,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
    )
    save_main_outputs(result, args.output_dir)
    save_kfold_splits(
        gene_exp_df=result["gene_exp_df"],
        splice_df=result["splice_df"],
        output_dir=args.output_dir,
    )
    print("Done processing data and preparing fixed 5-fold splits.")


if __name__ == "__main__":
    main()