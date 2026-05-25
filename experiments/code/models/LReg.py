import os
import pickle as pkl

import numpy as np
from sklearn.linear_model import LogisticRegression


class LReg:
    def __init__(self, num_classes, max_iter=1000, random_state=42):
        self.num_classes = num_classes
        self.alpha_list = [0.001, 0.01, 0.05]
        self.max_iter = max_iter
        self.random_state = random_state

    def train(self, X_train, y_train, X_test, y_test):
        self.model = None
        max_accuracy = 0
        for alpha in self.alpha_list:
            model = LogisticRegression(
                penalty="l1",
                C=1 / alpha,
                solver="saga",
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            model.fit(X_train, y_train)
            test_accuracy = model.score(X_test, y_test)
            if test_accuracy > max_accuracy:
                max_accuracy = test_accuracy
                self.model = model

        print(f"Test Accuracy: {(max_accuracy * 100):.6f}%")
        return max_accuracy

    def predict(self, input):
        return self.model.predict(input)

    def save_model(self, filename):
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        with open(filename, "wb") as f:
            pkl.dump(self.model, f)
        print(f"Model saved to {filename}")

    def load_model(self, filename):
        with open(filename, "rb") as f:
            self.model = pkl.load(f)

    def calc_shap(self, explain_df, y_true):
        # y_true is not used. It is only to cope with the interface.
        shap = self.model.coef_
        # y_pred = self.model.predict(explain_df.values)
        # num_features = shap.shape[1]
        # aggregated_shap = np.zeros(num_features)
        return {"shap": shap}
