from module.model.lib.privatree.bdpt import BDPTClassifier
from sklearn.metrics import accuracy_score
import logging
from typing import Any, Dict, Iterable, Tuple, Optional, List
import numpy as np
from module.model.core.base import BaseEstimator, register_model
from module.hparams import register_hparams


@register_model("bdpt")
class BDPTWrapper(BaseEstimator):
    """
    Wrapper for BDPTClassifier (Privacy-preserving decision tree).
    Includes a tune() method for simple grid search using nested CV.
    """
    params: Dict[str, Any]
    model: BDPTClassifier

    def __init__(self, **params: Any):
        super().__init__(**params)
        self.params = params

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BDPTWrapper":
        # Range [0, 1] for all features by default. 
        feature_range = np.tile(np.array([0.0, 1.0]), (X.shape[1], 1))
        params = {**self.params, "feature_range": feature_range}
        self.model = BDPTClassifier(**params)
        self.model.fit(X, y)
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        return self

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(accuracy_score(y, self.model.predict(X)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def tune(self, nested_cv: Iterable) -> Tuple[Dict[str, Any], Dict[Tuple[int, int], Iterable[float]]]:
        """
        Tune hyperparameters for BDPT using nested cross-validation.

        We grid search over max_depth and min_samples_split, keeping epsilon fixed.

        Args:
            nested_cv: Iterable of Split objects for inner CV

        Returns:
            (best_config, all_scores) where best_config is a dict and all_scores maps
            (max_depth, min_samples_split) -> list of fold scores.
        """
        logger = logging.getLogger("BDPT.tune")

        max_depths = [2, 3, 4, 5, 6]
        min_samples_splits = [10, 20, 40]
        epsilon = float(self.params.get("epsilon", 0.1))  # fixed during tuning

        nested_cv = list(nested_cv)
        best_score = -float("inf")
        best_config: Optional[Dict[str, Any]] = None
        all_scores: Dict[Tuple[int, int], List[float]] = {}

        for max_depth in max_depths:
            for min_samples_split in min_samples_splits:
                config = {
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                    "epsilon": epsilon,
                }

                scores: List[float] = []
                for split in nested_cv:
                    # Build feature_range based on train set
                    feature_range = np.tile(np.array([0.0, 1.0]), (split.X_train.shape[1], 1))
                    bdpt_config = {**self.params, **config, "feature_range": feature_range}
                    model = BDPTClassifier(**bdpt_config)
                    model.fit(split.X_train, split.y_train)
                    score = float(accuracy_score(split.y_test, model.predict(split.X_test)))
                    scores.append(score)

                avg_score = float(np.mean(scores)) if scores else -float("inf")
                key = (max_depth, min_samples_split)
                all_scores[key] = scores
                logger.debug(f"Config {key}: Avg Score = {avg_score:.4f}")

                if avg_score > best_score:
                    best_score = avg_score
                    best_config = config

        if best_config is None:
            best_config = {"max_depth": max_depths[0], "min_samples_split": min_samples_splits[0], "epsilon": epsilon}
        logger.info(f"Best Config: {best_config} with score: {best_score:.4f}")
        return best_config, all_scores

@register_hparams("bdpt")
def update_hparams(hparams, args, dataset):
    hparams.model_params = {
        "max_depth": int(args.max_depth) if hasattr(args, "max_depth") else 4,
        "min_samples_split": int(args.min_samples_split) if hasattr(args, "min_samples_split") else 10,
        "epsilon": float(args.epsilon) if hasattr(args, "epsilon") else 0.1,
    }