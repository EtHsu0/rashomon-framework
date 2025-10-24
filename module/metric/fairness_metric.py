"""fairness_metric.py: Metrics for evaluating fairness across sensitive groups.

This module provides metrics for assessing fairness constraints including
demographic parity, equal opportunity, and equalized odds across different
sensitive attribute groups.
"""
from module.metric.base_metrics import BaseMetric, register_metric

from typing import Dict
import numpy as np
from module.model.core.post_base import PostBase


@register_metric() 
class FairnessMetric(BaseMetric):
    """Fairness metric evaluating demographic parity and equalized odds.
    
    Computes fairness violations by comparing positive prediction rates and
    true/false positive rates across sensitive groups.
    """

    NAME = "fairness"
    NEED = [""]  
    
    def setup(self, _model, _hparams, split_data) -> None:
        """Validate that sensitive attributes are present in the dataset.
        
        Args:
            _model: Trained model (unused).
            _hparams: Hyperparameters (unused).
            split_data (Split): Data split that must contain sens_train and sens_test.
            
        Raises:
            AssertionError: If sensitive attributes are missing.
        """
        assert split_data.sens_train is not None and split_data.sens_test is not None, "FairnessMetric requires sensitive attributes in the dataset."
        pass

    def compute(self, predictions, split_data, **params) -> Dict[str, float]:
        """
        Required. Perform evaluation and return a dict of named scores.
        Must raise informative errors on unmet capabilities or invalid params.
        """
        criterion = params.get("criterion", "all")  # demographic parity
        pred_train_y = predictions["train"]
        pred_test_y = predictions["test"]

        assert np.array_equal(np.unique(split_data.sens_train), np.unique(split_data.sens_test)), "Sensitive features in train and test must have the same unique values."
        uniq = np.unique(split_data.sens_train)
        probs_train = np.zeros(len(uniq))
        probs_test = np.zeros(len(uniq))
        tprs_train = np.zeros(len(uniq))
        tprs_test = np.zeros(len(uniq))
        fprs_train = np.zeros(len(uniq))
        fprs_test = np.zeros(len(uniq))

        for i, u in enumerate(uniq):
            idx_train = np.where(split_data.sens_train == u)[0]
            idx_test = np.where(split_data.sens_test == u)[0]
            probs_train[i] = np.mean(pred_train_y[idx_train])
            probs_test[i] = np.mean(pred_test_y[idx_test])

            ## TPR/FPR
            idx_tpr_train = np.where((split_data.sens_train == u) & (split_data.y_train == 1))[0]
            idx_fpr_train = np.where((split_data.sens_train == u) & (split_data.y_train == 0))[0]
            idx_tpr_test = np.where((split_data.sens_test == u) & (split_data.y_test == 1))[0]
            idx_fpr_test = np.where((split_data.sens_test == u) & (split_data.y_test == 0))[0]
            tprs_train[i] = np.mean(pred_train_y[idx_tpr_train])
            fprs_train[i] = np.mean(pred_train_y[idx_fpr_train])
            tprs_test[i] = np.mean(pred_test_y[idx_tpr_test])
            fprs_test[i] = np.mean(pred_test_y[idx_fpr_test])
        

        train_dp = np.max(probs_train) - np.min(probs_train)
        test_dp = np.max(probs_test) - np.min(probs_test)
        if criterion == "sp":
            return {
                "train_demographic_parity_difference": train_dp,
                "test_demographic_parity_difference": test_dp,
            }

        train_tpr_gap = np.max(tprs_train) - np.min(tprs_train)
        test_tpr_gap = np.max(tprs_test) - np.min(tprs_test)
        train_fpr_gap = np.max(fprs_train) - np.min(fprs_train)
        test_fpr_gap = np.max(fprs_test) - np.min(fprs_test)

        train_eopp = train_tpr_gap
        test_eopp = test_tpr_gap
        
        if criterion == "eopp":
            return {
                "train_equal_opportunity_difference": train_eopp,
                "test_equal_opportunity_difference": test_eopp,
            }

        train_eodds = max(train_tpr_gap, train_fpr_gap)
        test_eodds = max(test_tpr_gap, test_fpr_gap)

        if criterion == "eo":
            return {
                "train_equalized_odds_difference": train_eodds,
                "test_equalized_odds_difference": test_eodds,
            }

        if criterion == "all":
            return {
                "train_demographic_parity_difference": train_dp,
                "test_demographic_parity_difference": test_dp,
                "train_equal_opportunity_difference": train_eopp,
                "test_equal_opportunity_difference": test_eopp,
                "train_equalized_odds_difference": train_eodds,
                "test_equalized_odds_difference": test_eodds,
            }

        raise ValueError(f"Unknown criterion {criterion}. Supported criteria are 'dp', 'eopp', and 'eo'.")

def statistical_parity_difference(y_true, y_pred, sens_attr) -> float:
    """Compute Statistical Parity Difference."""
    idx0 = np.where(sens_attr == 0)[0]
    idx1 = np.where(sens_attr == 1)[0]
    p0 = np.mean(y_pred[idx0])
    p1 = np.mean(y_pred[idx1])
    return abs(p0 - p1)

def equal_opportunity_difference(y_true, y_pred, sens_attr) -> float:
    """Compute Equal Opportunity Difference."""
    idx0 = np.where((sens_attr == 0) & (y_true == 1))[0]
    idx1 = np.where((sens_attr == 1) & (y_true == 1))[0]
    tpr0 = np.mean(y_pred[idx0])
    tpr1 = np.mean(y_pred[idx1])
    return abs(tpr0 - tpr1)

def equalized_odds_difference(y_true, y_pred, sens_attr) -> float:
    """Compute Equalized Odds Difference."""
    idx0_pos = np.where((sens_attr == 0) & (y_true == 1))[0]
    idx1_pos = np.where((sens_attr == 1) & (y_true == 1))[0]
    idx0_neg = np.where((sens_attr == 0) & (y_true == 0))[0]
    idx1_neg = np.where((sens_attr == 1) & (y_true == 0))[0]
    tpr0 = np.mean(y_pred[idx0_pos])
    tpr1 = np.mean(y_pred[idx1_pos])
    fpr0 = np.mean(y_pred[idx0_neg])
    fpr1 = np.mean(y_pred[idx1_neg])
    return max(abs(tpr0 - tpr1), abs(fpr0 - fpr1))
