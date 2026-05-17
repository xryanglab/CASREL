import os

import numpy as np
import shap
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from tqdm import trange

from .analyze_shap import generate_shap_analysis


class GeneSpliceModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GeneSpliceModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FCNet:
    def __init__(self, num_classes, hidden_dim_list=None, num_epochs=100, lr=0.001):
        self.num_classes = num_classes
        self.hidden_dim_list = hidden_dim_list if hidden_dim_list is not None else [256, 512, 1024, 2048]
        self.num_epochs = num_epochs
        self.lr = lr

    def _train_single(self, X_train, y_train, X_test, y_test, hidden_dim):
        input_dim = X_train.shape[1]
        model = GeneSpliceModel(input_dim, hidden_dim, self.num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_tensor = torch.LongTensor(y_train).to(device)
        X_test_tensor = torch.FloatTensor(X_test).to(device)

        # Training loop
        model.train()
        bar = trange(self.num_epochs, desc=f"hidden_dim={hidden_dim}")
        for epoch in bar:
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            bar.set_description(f"hidden_dim={hidden_dim} Epoch {epoch + 1:02d} Loss: {loss.item():.8f}")
            optimizer.step()

        # Eval
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor).cpu().numpy()

        y_pred_labels = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_test, y_pred_labels)
        return model, accuracy

    def train(self, X_train, y_train, X_test, y_test):
        self.model = None
        max_accuracy = 0
        best_hidden_dim = None

        for hidden_dim in self.hidden_dim_list:
            model, accuracy = self._train_single(X_train, y_train, X_test, y_test, hidden_dim)
            print(f"  hidden_dim={hidden_dim}, Test Accuracy: {(accuracy * 100):.6f}%")
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                self.model = model
                best_hidden_dim = hidden_dim

        print(f"Best hidden_dim={best_hidden_dim}, Test Accuracy: {(max_accuracy * 100):.6f}%")
        return max_accuracy

    def predict(self, input):
        self.model.eval()
        X_tensor = torch.FloatTensor(input).to(device)
        with torch.no_grad():
            y_pred = self.model(X_tensor).cpu().numpy()
        y_pred_labels = np.argmax(y_pred, axis=1)
        return y_pred_labels

    def save_model(self, filename):
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        torch.save(self.model.state_dict(), filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename):
        state_dict = torch.load(filename)
        self.model.load_state_dict(state_dict)
        print(f"Model loaded from {filename}")

    def calc_shap(self, explain_df, y_true):
        # Prepare the data
        self.model.eval()
        y_pred = self.predict(explain_df.values)
        explain_tensor = torch.FloatTensor(explain_df.values).to(device)

        # Run the explainer
        print("Running DeepExplainer...")
        deep_explainer = shap.DeepExplainer(self.model, explain_tensor)
        print("Done.")

        # Calculate the SHAP values.
        deep_shap = deep_explainer.shap_values(explain_tensor)
        deep_shap = np.array(deep_shap)
        _, num_samples, num_genes = deep_shap.shape
        aggregated_shap = np.zeros((num_samples, num_genes))
        for i in range(num_samples):
            for j in range(num_genes):
                aggregated_shap[i, j] = deep_shap[y_pred[i], i, j]

        # Run SHAP analysis.
        print("Running SHAP analysis...")
        shap_analysis = generate_shap_analysis(explain_df, deep_shap, y_true, y_pred)
        print("Done.")

        return {
            "shap": deep_shap,
            "aggregated_shap": aggregated_shap,
            "shap_analysis": shap_analysis,
        }