"""Compute summary statistics about the output"""

import anndata
import numpy as np

from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from typing import Dict


def compute_basal_state_classifier_stats(
    adata: anndata.AnnData,
    perturbation_key: str,
    basal_key: str,
) -> Dict[str, np.ndarray]:
    """For the learned basal state, compute TPR and FPR for a ROC curve, as well as AUC

    Args:
        adata: AnnData object
        perturbation_key: Key of adata.obs that specifies perturbation information
        basal_key: Key of adata.obsm that contains the learned basal state

    Returns:
        Dict with [FPR, TPR, AUC, unique_perturbation_conditions] where
        unique_perturbation_conditions labels the other arrays

    """

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    unique_perturbation_conditions = adata.obs[perturbation_key].unique()

    for c in unique_perturbation_conditions:
        X = adata.obsm[basal_key]
        y = adata.obs[perturbation_key] == c

        random_state = np.random.RandomState(0)

        # shuffle and split training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=random_state
        )

        # Learn to predict each class against the others
        classifier = LogisticRegression(random_state=random_state)
        y_score = classifier.fit(X_train, y_train).decision_function(X_test)

        fpr[c], tpr[c], _ = roc_curve(y_test, y_score, pos_label=classifier.classes_[1])
        roc_auc[c] = auc(fpr[c], tpr[c])

    return {
        "fpr": fpr,
        "tpr": tpr,
        "unique_perturbation_conditions": unique_perturbation_conditions,
        "auc": roc_auc,
    }
