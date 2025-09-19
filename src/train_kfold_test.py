import argparse
import os
import traceback

import pandas as pd
import numpy as np
from models import FCNet, LightGBM, LReg
from data_utils import save_shap, MetricLogger



def train_model(
    model_type,
    k,
    X_train,
    y_train,
    X_test,
    y_test,
    prev_model=None
):
    # Build the model
    if model_type == "fcn":
        model = FCNet(num_classes=4)
    elif model_type == "lgbm":
        model = LightGBM(num_classes=4)
    elif model_type == "lreg":
        model = LReg(num_classes=4)

    # Train the model
    accuracy = model.train(X_train, y_train[:, k], X_test, y_test[:, k], init_model=prev_model)
    return model, accuracy


# An utility function to parse column indices from the command line argument.
def parse_column_index(k_str, num_columns):
    # If -1 is provided, return all columns.
    if k_str == "-1":
        return list(range(num_columns))

    # If a range is provided, return a list of indices.
    if "-" in k_str:
        try:
            start, end = k_str.split("-")
        except Exception as e:
            raise ValueError(
                "Unable to parse the column index range from the command line argument:"
                + k_str
            )
        return list(range(int(start), min(num_columns, int(end) + 1)))

    # Else, parse the index as an integer.
    try:
        k_int = int(k_str)
    except Exception as e:
        raise ValueError(
            "Unable to parse the column index from the command line argument:" + k_str
        )

    return k_int


# The main entry point for training the model.
def run(k, model_type, input_dir, output_dir):
    # Load the CSV files
    gene_exp_df = pd.read_csv(os.path.join(input_dir, "gene_exp_df.csv"))
    splice_df = pd.read_csv(os.path.join(input_dir, "splice_df.csv"))
    # parse the column index
    k_list = parse_column_index(k, splice_df.shape[1])

    # Train model for all the specified column indices (sites).
    for i in k_list:
        prev_model = None
        accuracies = []
        for j in range(5):
            train_gene_exp_df = pd.read_csv(os.path.join(input_dir, f'k{j+1}_train_gene_exp_df.csv'))
            test_gene_exp_df = pd.read_csv(os.path.join(input_dir, f'k{j+1}_test_gene_exp_df.csv'))
            train_splice_df = pd.read_csv(os.path.join(input_dir, f'k{j+1}_train_splice_df.csv'))
            test_splice_df = pd.read_csv(os.path.join(input_dir, f'k{j+1}_test_splice_df.csv'))

            # Build the training and test arrays.
            X_train = train_gene_exp_df.values
            X_test = test_gene_exp_df.values
            y_train = train_splice_df.values
            y_test = test_splice_df.values

            # The ground truth for shap analysis.
            y_true = splice_df.values[:, i].astype(int)

            try:
                train_model(
                    model_type,
                    i,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    prev_model=prev_model
                    )
                prev_model = model
                accuracies.append(accuracy)
            except Exception as e:
                print(
                    "Run into an error while training the model with site index:",
                    i)
                traceback_str = traceback.format_exc()
                print("stack trace:")
                print(traceback_str)
        final_accuracy = np.mean(accuracies)
        site_name = splice_df.columns[i]
        accuracy_logger.log_metric((k, site_name), final_accuracy)

        # Save the model
        model_output_path = os.path.join(
            output_dir, "model_weights", model_type, "column_" + str(k) + "_model.pkl"
        )
        model.save_model(model_output_path)

        # Calculate and save the SHAP values
        shap_dict = model.calc_shap(train_gene_exp_df, y_true)
        shap_output_dir = os.path.join(
            output_dir, "shap_values", model_type, "column_" + str(k)
        )
        save_shap(shap_dict, shap_output_dir)

        accuracy_logger = MetricLogger(
        ("site_index", "site"),
        "accuracy",
        os.path.join(output_dir, "metric_logs", model_type, "accuracy_log_[%d-%d].csv" % (k_list[0], k_list[-1])),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-k",
        type=str,
        default=0,
        help="""column (site) index to train the model with. -1 means all columns.
        You can either specify a single column like 0, 1, or 2, or a range like 0-2.
        """,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="../post_process_data/epall/",
        help="input directory path.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="fcn",
        help="type of model to train. Default is fcn",
        choices=["fcn", "lgbm", "lreg"],
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="../output/epall",
        help="output directory path.",
    )
    args = parser.parse_args()
    run(args.k, args.model, args.input, args.output)
                                                        
            

            
                
                
    
    









