import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


def categorize_splice_prob(prob):
    if pd.isna(prob):
        return 0
    if prob < 0.4:
        return 1
    elif prob < 0.7:
        return 2
    else:
        return 3


def load_and_transform(splice_file, gene_file):
    splice_start_prob = norm_only(splice_file, "start")
    splice_end_prob = norm_only(splice_file, "end")
    gene_exp_df = pd.read_csv(gene_file, index_col=0)

    common_columns = list(
        set(splice_start_prob.columns) & set(splice_end_prob.columns) & set(gene_exp_df.columns)
    )

    prob_start = splice_start_prob[common_columns]
    prob_end = splice_end_prob[common_columns]
    gene_exp_df = gene_exp_df[common_columns]

    splice_start_class = prob_start.applymap(categorize_splice_prob)
    splice_end_class = prob_end.applymap(categorize_splice_prob)

    splice_df = pd.concat([splice_start_class, splice_end_class], axis=0)
    splice_df = splice_df.fillna(0).astype(int)
    splice_df = splice_df.T
    gene_exp_df = gene_exp_df.T

    return prob_start, prob_end, splice_df, gene_exp_df


def plot_prob_distribution_non_na(prob_start, prob_end, output_path):
    prob_combined = pd.concat([prob_start, prob_end], axis=0)
    all_vals = prob_combined.values.flatten()
    all_vals = all_vals[~np.isnan(all_vals)]

    plt.figure(figsize=(8, 5))
    sns.histplot(all_vals, bins=50, stat='probability', kde=True,
                 color='steelblue', edgecolor='black', alpha=0.7)
    plt.title("Distribution of AS probabilities across cells (non‑NA)")
    plt.xlabel("Probability")
    plt.ylabel("Proportion")
    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    plt.close()
    print(f"Probability distribution (non‑NA) saved to {output_path}")


def plot_class_distribution(splice_df, output_path):
    class_counts = splice_df.stack().value_counts().sort_index()
    total = class_counts.sum()
    proportions = class_counts / total

    class_labels = {0: "0 (NA)", 1: "1 (Low)", 2: "2 (Medium)", 3: "3 (High)"}
    labels = [class_labels.get(i, str(i)) for i in proportions.index]

    plt.figure(figsize=(6, 5))
    bars = plt.bar(labels, proportions.values,
                   color=['gray', 'skyblue', 'orange', 'red'], edgecolor='black')
    plt.title("Distribution of AS class across cells")
    plt.xlabel("Class")
    plt.ylabel("Proportion")
    plt.ylim(0, 1)
    for bar, prop in zip(bars, proportions.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{prop:.2%}", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    plt.close()
    print(f"Class distribution saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot AS probability and class distributions (PDF output, proportions on Y-axis)."
    )
    parser.add_argument("--splice_file", type=str, default="../data/epall_filter",
                        help="Path prefix for splice site CSV files (e.g., xxx_start.csv, xxx_end.csv).")
    parser.add_argument("--gene_file", type=str, default="../data/epall_rbp.csv",
                        help="Path to the RBP expression CSV file.")
    parser.add_argument("--output_dir", type=str, default="../post_process_data/epall/",
                        help="Output directory for PDF figures.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    prob_start, prob_end, splice_df, _ = load_and_transform(args.splice_file, args.gene_file)

    plot_prob_distribution_non_na(prob_start, prob_end,
                                  os.path.join(args.output_dir, "as_prob_distribution_non_na.pdf"))
    plot_class_distribution(splice_df,
                            os.path.join(args.output_dir, "as_class_distribution.pdf"))

    print("All plots saved as PDF.")


if __name__ == "__main__":
    main()