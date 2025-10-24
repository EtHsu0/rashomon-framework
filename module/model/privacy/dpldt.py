"""dpldt.py: Differentially Private CART decision tree wrapper.

This module provides a wrapper around IBM's diffprivlib DecisionTreeClassifier,
implementing differential privacy guarantees during tree construction using the
exponential mechanism for split selection.
"""
import logging
from typing import Any, Dict, Iterable, Tuple
import numpy as np

from diffprivlib.models import DecisionTreeClassifier
from module.model.core.base import BaseEstimator, register_model
from module.hparams import register_hparams


@register_model("dpldt")
class Dpldt(BaseEstimator):
    """Differentially private CART decision tree model.
    
    Wraps diffprivlib's DecisionTreeClassifier to provide DP guarantees during
    training. Privacy budget (epsilon) controls the privacy-accuracy tradeoff.
    """
    
    def __init__(self, **params):
        """Initialize differentially private decision tree.
        
        Args:
            **params: Parameters passed to diffprivlib DecisionTreeClassifier,
                     including epsilon (privacy budget).
        """
        self.params = params
        self.model = DecisionTreeClassifier(**params)

    def fit(self, X, y):
        self.model.fit(X, y)
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)

    def score(self, X, y):
        return self.model.score(X,y)
    
    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def tune(self, nested_cv: Iterable) -> Tuple[Dict[str, Any], Dict[int, Iterable[float]]]:
        """
        Tune hyperparameters for differentially private DecisionTreeClassifier using nested CV.

        Grid search over max_depth; epsilon kept fixed.

        Args:
            nested_cv: Iterable of Split objects for inner CV

        Returns:
            (best_config, all_scores) where best_config is a dict and all_scores maps
            max_depth -> list of fold scores.
        """
        logger = logging.getLogger("Dpldt.tune")
        max_depths = list(range(2, 7))  # modest depths for stability (2-6)

        nested_cv = list(nested_cv)
        best_score = -float("inf")
        best_config: Dict[str, Any] | None = None
        all_scores: Dict[int, list[float]] = {}

        # Base params from current instance
        base_params = {k: v for k, v in self.params.items() if k != "max_depth"}

        for max_depth in max_depths:
            logger.debug(f"Max Depth: {max_depth}")
            config = {**base_params, "max_depth": max_depth}

            scores: list[float] = []
            for split in nested_cv:
                model = DecisionTreeClassifier(**config)
                model.fit(split.X_train, split.y_train)
                score = float(model.score(split.X_test, split.y_test))
                scores.append(score)

            avg_score = float(np.mean(scores)) if scores else -float("inf")
            logger.info(f"Max Depth: {max_depth}, Avg Score: {avg_score:.4f}")
            all_scores[max_depth] = scores

            if avg_score > best_score:
                best_score = avg_score
                best_config = config

        if best_config is None:
            best_config = {**base_params, "max_depth": max_depths[0]}
        logger.info(f"Best Config: {best_config} with score: {best_score:.4f}")
        return best_config, all_scores
    
@register_hparams("dpldt")
def update_hparams(hparams, args, dataset):
    hparams.model_params = {
        "max_depth": int(args.max_depth) if hasattr(args, "max_depth") else 4,
        "epsilon": float(args.epsilon) if hasattr(args, "epsilon") else 0.1,
        "random_state": hparams.rs,
        "bounds": (0, 1),  # assuming normalized data
        "classes": [0, 1],  # assuming binary classification
    }