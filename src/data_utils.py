import os
import pickle as pkl

import pandas as pd


def save_shap(shap_dict, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if "shap" in shap_dict:
        with open(os.path.join(output_dir, "shap.pkl"), "wb") as f:
            pkl.dump(shap_dict["shap"], f)

    if "aggregated_shap" in shap_dict:
        with open(os.path.join(output_dir, "aggregated_shap.pkl"), "wb") as f:
            pkl.dump(shap_dict["aggregated_shap"], f)

    if "shap_analysis" in shap_dict:
        shap_dict["shap_analysis"].to_csv(os.path.join(output_dir, "shap_analysis.csv"))


def load_shap(shap_dir):
    shap_dict = {}

    if os.path.exists(os.path.join(shap_dir, "shap.pkl")):
        with open(os.path.join(shap_dir, "shap.pkl"), "rb") as f:
            shap_dict["shap"] = pkl.load(f)

    if os.path.exists(os.path.join(shap_dir, "aggregated_shap.pkl")):
        with open(os.path.join(shap_dir, "aggregated_shap.pkl"), "rb") as f:
            shap_dict["aggregated_shap"] = pkl.load(f)

    if os.path.exists(os.path.join(shap_dir, "shap_analysis.csv")):
        shap_dict["shap_analysis"] = pd.read_csv(
            os.path.join(shap_dir, "shap_analysis.csv")
        )

    return shap_dict


class MetricLogger:
    def __init__(self, index_names, metric_name, output_path):
        self.metric_name = metric_name
        self.index_names = index_names
        self.output_path = output_path
        self.metrics = {}
        if not os.path.exists(os.path.dirname(self.output_path)):
            os.makedirs(os.path.dirname(self.output_path))
        with open(self.output_path, "w") as f:
            f.write(",".join(map(str, index_names)) + "," + metric_name + "\n")

    def log_metric(self, indices, metric_value):
        index_str = ",".join(map(str, indices))
        self.metrics[index_str] = metric_value
        with open(self.output_path, "a") as f:
            f.write(index_str + "," + str(metric_value) + "\n")