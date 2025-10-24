""" standard_metric.py:  Standard metrics for model evaluation. """
from module.metric.base_metrics import BaseMetric, register_metric
from sklearn.metrics import accuracy_score

from typing import Dict

@register_metric()
class StandardMetric(BaseMetric):
    """
    A standard metric that computes a single score.
    """

    NAME = "accuracy"

    def compute(self, predictions, split_data, **params) -> Dict[str, float]:
        """
        Required. Perform evaluation and return a dict of named scores.
        Must raise informative errors on unmet capabilities or invalid params.
        """
        pred_train_y = predictions["train"]
        pred_test_y = predictions["test"]
        train_score = accuracy_score(split_data.y_train, pred_train_y)
        test_score = accuracy_score(split_data.y_test, pred_test_y)

        
        return {"train_accuracy": train_score, "test_accuracy": test_score}