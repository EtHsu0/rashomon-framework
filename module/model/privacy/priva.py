"""priva.py: PrivaTree differentially private decision tree wrapper.

This module wraps the PrivaTree algorithm which provides differential privacy
guarantees for decision tree learning through careful noise injection and tree
construction procedures.
"""
from module.model.lib.privatree.privatree import PrivaTreeClassifier
import logging
from typing import Any, Dict, Iterable, Tuple, Optional, List
import numpy as np
from module.model.core.base import BaseEstimator, register_model
from module.hparams import register_hparams


@register_model("priva")
class PrivaTree(BaseEstimator):
    """PrivaTree differentially private decision tree model.
    
    Implements differential privacy through the PrivaTree algorithm, providing
    formal privacy guarantees while building decision trees.
    """
    
    def __init__(self, **params):
        """Initialize PrivaTree model.
        
        Args:
            **params: Parameters for PrivaTreeClassifier including epsilon (privacy budget).
        """
        self.params = params
        self.n_features_ = None
        self.classes_ = None

    def fit(self, X, y):
        feature_range = np.tile(np.array([0.0, 1.0]), (X.shape[1], 1))
        params = {**self.params, "feature_range": feature_range}
        self.model = PrivaTreeClassifier(**params)
        self.model.fit(X, y)
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)

    def tune(self, nested_cv: Iterable) -> Tuple[Dict[str, Any], Dict[Tuple[int, int, int], Iterable[float]]]:
        """
        Tune hyperparameters for PrivaTree using nested cross-validation.

        We grid search over max_depth, min_samples_split, and min_samples_leaf,
        keeping epsilon fixed.

        Args:
            nested_cv: Iterable of Split objects for inner CV

        Returns:
            (best_config, all_scores) where best_config is a dict and all_scores maps
            (max_depth, min_samples_split, min_samples_leaf) -> list of fold scores.
        """
        logger = logging.getLogger("PrivaTree.tune")

        max_depths = [2, 3, 4, 5, 6]
        min_samples_splits = [10, 20, 40]
        min_samples_leafs = [5, 10, 20]
        epsilon = float(self.params.get("epsilon", 0.1))  # fixed during tuning

        nested_cv = list(nested_cv)
        best_score = -float("inf")
        best_config: Optional[Dict[str, Any]] = None
        all_scores: Dict[Tuple[int, int, int], List[float]] = {}

        for max_depth in max_depths:
            for min_samples_split in min_samples_splits:
                for min_samples_leaf in min_samples_leafs:
                    if min_samples_split < 2 * min_samples_leaf:
                        continue
                    config = {
                        "max_depth": max_depth,
                        "min_samples_split": min_samples_split,
                        "min_samples_leaf": min_samples_leaf,
                        "epsilon": epsilon,
                    }

                    scores: List[float] = []
                    for split in nested_cv:
                        pt_config = {**self.params, **config}
                        feature_range = np.tile(np.array([0.0, 1.0]), (split.X_train.shape[1], 1))
                        pt_config["feature_range"] = feature_range
                        model = PrivaTreeClassifier(**pt_config)
                        model.fit(split.X_train, split.y_train)
                        score = float(model.score(split.X_test, split.y_test))
                        scores.append(score)

                    avg_score = float(np.mean(scores)) if scores else -float("inf")
                    key = (max_depth, min_samples_split, min_samples_leaf)
                    all_scores[key] = scores
                    logger.debug(f"Config {key}: Avg Score = {avg_score:.4f}")

                    if avg_score > best_score:
                        best_score = avg_score
                        best_config = config

        if best_config is None:
            best_config = {
                "max_depth": max_depths[0],
                "min_samples_split": min_samples_splits[0],
                "min_samples_leaf": min_samples_leafs[0],
                "epsilon": epsilon,
            }
        logger.info(f"Best Config: {best_config} with score: {best_score:.4f}")
        return best_config, all_scores

@register_hparams("priva")
def update_hparams(hparams, args, dataset):
    hparams.model_params = {
        "max_depth": int(args.max_depth) if hasattr(args, "max_depth") else 4,
        "min_samples_split": int(args.min_samples_split) if hasattr(args, "min_samples_split") else 10,
        "min_samples_leaf": int(args.min_samples_leaf) if hasattr(args, "min_samples_leaf") else 5,
        "epsilon": float(args.epsilon) if hasattr(args, "epsilon") else 0.1,
    }