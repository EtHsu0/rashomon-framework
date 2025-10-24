"""fprdt.py: Fast Provably Robust Decision Tree wrapper.

This module implements FPRDT, an adversarially robust decision tree thatn provide certified robustness guarantees against L-infinity
norm bounded adversarial perturbations.
"""
import logging
from typing import Any, Optional
import numpy as np
from module.model.core.base import BaseEstimator, register_model
from module.hparams import Hparams, register_hparams
from module.datasets import DatasetLoader
from module.model.lib.fprdt import FPRDecisionTree


@register_model("fprdt")
class FprdtWrapper(BaseEstimator):
    """Fast Provably Robust Decision Tree model.
    
    Provides certified adversarial robustness by ensuring tree structure
    is robust to bounded perturbations within epsilon.
    """
    params: dict
    model: Any

    def __init__(self, **params: Any):
        """Initialize FPRDT wrapper.
        
        Args:
            **params: Model hyperparameters passed to FPRDecisionTree.
        """
        """
        Initialize FprdtWrapper.
        Args:
            **params: Model hyperparameters.
        """
        super().__init__(**params)
        self.params = params
        self.model = FPRDecisionTree(**params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FprdtWrapper':
        """
        Fit the FPRDecisionTree model.
        Args:
            X: Features
            y: Labels
        Returns:
            self
        """
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for X.
        """
        return self.model.predict(X)

    def tune(self, nested_cv):
        """
        Tune hyperparameters for FPRDT using nested cross-validation.
        
        FPRDT is a robust decision tree that uses fairness and provably robust training.
        We tune max_depth, min_samples_leaf, and min_samples_split. 
        Epsilon is preserved from initialization (dataset-specific or default).
        
        Args:
            nested_cv: Iterable of Split objects for inner CV
        
        Returns:
            tuple: (best_config dict, all_scores dict)
        """
        logger = logging.getLogger("Fprdt.tune")
        
        # Hyperparameter grid - FPRDT specific ranges
        max_depths = [2, 3, 4, 5, 6]  # Shallow to medium depth
        min_samples_leafs = [5, 10, 20]   # Prevent overfitting
        min_samples_splits = [10, 20, 40]  # Control tree growth
        # Use epsilon from model initialization (dataset-specific)
        epsilon = self.params.get('epsilon', 0.1)
        
        nested_cv = list(nested_cv)
        best_score = -float('inf')
        best_config = None
        all_scores = {}
        
        # Get current params to use as base
        random_seed = self.params.get('random_seed', 42)
        
        logger.info(f"Tuning with epsilon={epsilon} (from model initialization)")
        
        # Grid search over hyperparameters
        for max_depth in max_depths:
            for min_samples_leaf in min_samples_leafs:
                for min_samples_split in min_samples_splits:
                    # Skip invalid combinations
                    if min_samples_split < 2 * min_samples_leaf:
                        continue
                    
                    config = {
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'epsilon': epsilon,
                        'random_seed': random_seed,
                    }
                    
                    scores = []
                    for split in nested_cv:
                        # Use binary data if available (when binarize_mode is enabled)
                        if hasattr(split, 'binarizer') and split.binarizer is not None:
                            binary_data = split.get_binary_data()
                            X_train, X_test = binary_data['X_train'], binary_data['X_test']
                        else:
                            X_train, X_test = split.X_train, split.X_test
                        
                        fprdt_model = FPRDecisionTree(**config)
                        fprdt_model.fit(X_train, split.y_train)
                        
                        score = fprdt_model.score(X_test, split.y_test)
                        scores.append(score)
                    
                    avg_score = np.mean(scores)
                    config_key = (max_depth, min_samples_leaf, min_samples_split)
                    all_scores[config_key] = scores
                    
                    logger.debug(f"Config {config_key}: Avg Score = {avg_score:.4f}")
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_config = config
        
        logger.info(f"Best Config: {best_config} with score: {best_score:.4f}")
        logger.info(f"Evaluated {len(all_scores)} configurations")
        return best_config, all_scores

    
@register_hparams("fprdt")
def update_hparams(hparams, args, dataset=None):
    hparams.model_params = {
        "max_depth": args.max_depth if hasattr(args, "max_depth") else 4,
        "min_samples_split": args.min_samples_split if hasattr(args, "min_samples_split") else 10,
        "min_samples_leaf": args.min_samples_leaf if hasattr(args, "min_samples_leaf") else 5,
        "epsilon": float(args.epsilon) if hasattr(args, "epsilon") else 0.1,
        "random_seed": hparams.rs,
    }