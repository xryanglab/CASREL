import argparse
import os
import traceback
import pandas as pd
from .lgb import LightGBM
from data_utils import save_shap, MetricLogger


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def parse_column_index(k_str, num_columns):
    if k_str == "-1":
        return list(range(num_columns))

    if "-" in k_str:
        try:
            start, end = k_str.split("-")
            start_i, end_i = int(start), int(end)
        except Exception:
            raise ValueError(
                f"Unable to parse column index range: {k_str}. Valid format is 0-99 or -1"
            )
        if start_i < 0:
            start_i = 0
        end_i = min(num_columns - 1, end_i)
        if start_i > end_i:
            raise ValueError(f"Invalid column index range: {k_str}")
        return list(range(start_i, end_i + 1))


    try:
        k_int = int(k_str)
    except Exception:
        raise ValueError(f"Unable to analyze column index: {k_str}")
    if not (0 <= k_int < num_columns):
        raise ValueError(f"Column index out of bounds: {k_int}, available range [0, {num_columns-1}]")
    return [k_int]


def train_one_site(
    k,
    site_name,
    train_gene_exp_df,
    X_train,
    y_train,
    X_test,
    y_test,
    output_dir,
    y_true,
    accuracy_logger,
):


    model = LightGBM(input_dim=X_train.shape[1], num_classes=4)


    accuracy = model.train(X_train, y_train[:, k], X_test, y_test[:, k])
    accuracy_logger.log_metric((k, site_name), accuracy)


    model_output_path = os.path.join(
        output_dir, "model_weights", "lgbm", f"column_{k}_model.pkl"
    )
    ensure_dir(os.path.dirname(model_output_path))
    model.save_model(model_output_path)


    shap_dict = model.calc_shap(train_gene_exp_df, y_true)
    shap_output_dir = os.path.join(output_dir, "shap_values", "lgbm", f"column_{k}")
    ensure_dir(shap_output_dir)
    save_shap(shap_dict, shap_output_dir)


def run_for_folder(input_dir, output_dir, k_arg):
    """
    Run the complete process (load data, train specified columns, and print results) for a single k* subdirectory.
    """
    gene_exp_df = pd.read_csv(os.path.join(input_dir, "gene_exp_df.csv"))
    splice_df = pd.read_csv(os.path.join(input_dir, "splice_df.csv"))
    train_gene_exp_df = pd.read_csv(os.path.join(input_dir, "train_gene_exp_df.csv"))
    test_gene_exp_df = pd.read_csv(os.path.join(input_dir, "test_gene_exp_df.csv"))
    train_splice_df = pd.read_csv(os.path.join(input_dir, "train_splice_df.csv"))
    test_splice_df = pd.read_csv(os.path.join(input_dir, "test_splice_df.csv"))


    X_train = train_gene_exp_df.values
    X_test = test_gene_exp_df.values
    y_train = train_splice_df.values
    y_test = test_splice_df.values


    k_list = parse_column_index(k_arg, splice_df.shape[1])


    acc_log_dir = os.path.join(output_dir, "metric_logs", "lgbm")
    ensure_dir(acc_log_dir)
    accuracy_logger = MetricLogger(
        ("site_index", "site"),
        "accuracy",
        os.path.join(
            acc_log_dir,
            f"accuracy_log_[{k_list[0]}-{k_list[-1]}].csv",
        ),
    )


    for i in k_list:
        y_true = splice_df.values[:, i].astype(int)
        try:
            train_one_site(
                i,
                splice_df.columns[i],
                gene_exp_df,
                X_train,
                y_train,
                X_test,
                y_test,
                output_dir,
                y_true,
                accuracy_logger,
            )
            print(f"[{os.path.basename(output_dir)}] Trained site index: {i}")
        except Exception:
            print(
                f"[{os.path.basename(output_dir)}] 训练出错，site index: {i}"
            )
            print("stack trace:")
            print(traceback.format_exc())


def main():
    parser = argparse.ArgumentParser(
        description="Use LightGBM to train batches in the k1-k5 subdirectories and write the results to the corresponding output subdirectories."
    )
    parser.add_argument(
        "-k",
        type=str,
        default="-1",
        help='Training column index. "-1" means all columns; can also be a single value such as "3" or a range such as "0-99".',
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="../post_process_data/epall",
        help="Enter the root directory (which contains the k1-k5 subdirectories).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="../output/epall",
        help="Output the root directory (subdirectories k1-k5 will be created and written to the result).",
    )
    parser.add_argument(
        "-f",
        "--folders",
        type=str,
        default="k1,k2,k3,k4,k5",
        help='A comma-separated list of subdirectories to be processed. The default value is "k1,k2,k3,k4,k5".',
    )
    args = parser.parse_args()

    folders = [f.strip() for f in args.folders.split(",") if f.strip()]
    if not folders:
        raise ValueError("The folders parameter is recognized as empty.")

    for folder in folders:
        in_dir = os.path.join(args.input, folder)
        out_dir = os.path.join(args.output, folder)
        if not os.path.isdir(in_dir):
            print(f"Skipping {folder}: Input directory does not exist -> {in_dir}")
            continue
        os.makedirs(out_dir, exist_ok=True)
        print(f"Start processing: {folder}")
        run_for_folder(in_dir, out_dir, args.k)
        print(f"Complete processing: {folder}")


if __name__ == "__main__":
    main()
