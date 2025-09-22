import numpy as np
import pandas as pd


def generate_shap_analysis(
    gene_exp_df: pd.DataFrame, shap: np.ndarray, truth: np.ndarray, pred: np.ndarray
) -> pd.DataFrame:
    """
    Generates a SHAP anlaysis table for a given shap values.

    The input shap should be a numpy array of shape
    (num_classes, num_samples, num_genes) where num_classes == 5. Started from index
    0, we consider class 1 to be inhibiting and class 3 to be exhibiting.

    We will record 3 sets of SHAP values for each gene:

    1. The sum of inhibiting and exhibiting SHAP values aggregated across all samples.
       We call this the "aggregated SHAP".
    2. The sum of inhibiting SHAP values aggregated over samples where the gene is
       inhibiting the AS site, and the sum of exhibiting SHAP values aggregated over
       samples where the gene is exhibiting the AS site. We call this the
       "ground truth SHAP"
    3. The sum of inhibiting SHAP values aggregated over samples where the model
       predicts that the gene is inhibiting the AS site, and the sum of exhibiting
       SHAP values aggregated over samples where the model predicts that the gene is
       exhibiting the AS site. We call this the "predicted SHAP".
    """
    # Initialize the result DataFrame. We will populate this later.
    _, num_samples, num_genes = shap.shape

    result = pd.DataFrame(
        data=np.zeros((num_genes, 6)),
        columns=[
            "aggregated_shap_inhibit",
            "aggregated_shap_exhibit",
            "ground_truth_shap_inhibit",
            "ground_truth_shap_exhibit",
            "predicted_shap_inhibit",
            "predicted_shap_exhibit",
        ],
        index=gene_exp_df.columns,
    )

    # Populate the aggregated SHAP values for each gene.
    result["aggregated_shap_inhibit"] = np.abs(shap[1, :, :]).sum(axis=0)
    result["aggregated_shap_exhibit"] = np.abs(shap[3, :, :]).sum(axis=0)

    # Populate the ground truth SHAP values for each gene.
    for i in range(num_samples):
        if truth[i] == 1:
            result["ground_truth_shap_inhibit"] += np.abs(shap[1, i, :])
        elif truth[i] == 3:
            result["ground_truth_shap_exhibit"] += np.abs(shap[3, i, :])

    # Populate the predicted SHAP values for each gene.
    for i in range(num_samples):
        if pred[i] == 1:
            result["predicted_shap_inhibit"] += np.abs(shap[1, i, :])
        elif pred[i] == 3:
            result["predicted_shap_exhibit"] += np.abs(shap[3, i, :])

    return result
