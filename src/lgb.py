import os

import lightgbm as lgb
import numpy as np
import shap
from sklearn.metrics import accuracy_score

from .analyze_shap import generate_shap_analysis


class LightGBM:
    def __init__(self, num_classes, input_dim):
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.params = {
            "max_bin": 255,
            "learning_rate": 0.1,
            "boosting_type": "gbdt",
            "objective": "multiclass",
            "num_class": num_classes,
            "metric": "multi_logloss",
            "verbose": -1,
            "boost_from_average": True,
        }

    def train(self, X_train, y_train, X_test, y_test):
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test)
        self.model = lgb.train(self.params, train_data, 100, valid_sets=[test_data])

        # eval
        y_pred = self.model.predict(X_test)

        # Calculate accuracy on the test set
        y_pred_labels = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_test, y_pred_labels)
        print(f"Test Accuracy: {(accuracy * 100):.6f}%")
        return accuracy

    def predict(self, input):
        y_pred = self.model.predict(input)
        return np.argmax(y_pred, axis=1)

    def save_model(self, filename):
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        self.model.save_model(filename)

    def load_model(self, filename):
        self.model = lgb.Booster(model_file=filename)

    def calc_shap(self, explain_df, y_true):
        # Run the SHAP explainer.
        print("Running TreeExplainer...")
        tree_explainer = shap.TreeExplainer(self.model)
        tree_shap = tree_explainer.shap_values(explain_df.values)
        print("Done.")

        # Calculate the SHAP values.
        tree_shap = np.array(tree_shap)  
        _, num_samples, num_genes = tree_shap.shape
        aggregated_shap = np.zeros((num_samples, num_genes))
        y_pred = self.predict(explain_df.values)
        for i in range(num_samples):
            for j in range(num_genes):
                aggregated_shap[i, j] = tree_shap[y_pred[i], i, j]

        # Run SHAP analysis.
        print("Running SHAP analysis...")
        shap_analysis = generate_shap_analysis(explain_df, tree_shap, y_true, y_pred)
        print("Done.")

        return {
            "shap": tree_shap,
            "aggregated_shap": aggregated_shap,
            "shap_analysis": shap_analysis,
        }
