"""
CART (Classification and Regression Trees) model wrapper.
Example usage for BaseEstimator subclassing, not needed when model has sklearn-like API.
"""
import argparse
import logging

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from module.model.core.base import BaseEstimator, register_model
from module.hparams import Hparams, register_hparams
from module.datasets import DatasetLoader

@register_model("cart")
class CartWrapper(BaseEstimator):
    """
    Wrapper for CART (Classification and Regression Trees) model.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = DecisionTreeClassifier(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        return self.model.set_params(**params)

    def tune(self, nested_cv):
        """
        Tune hyperparameters for CART using nested cross-validation.
        
        Tunes max_depth, min_samples_leaf, and min_samples_split via grid search.
        For trustworthiness research, we use conservative ranges to avoid overfitting
        while maintaining interpretability.
        
        Args:
            nested_cv: Iterable of tuples (fold_id, (X_train, y_train), (X_val, y_val))
        
        Returns:
            tuple: (best_config dict, all_scores dict)
        """
        logger = logging.getLogger("CartWrapper.tune")
        
        # Hyperparameter grid - conservative ranges for interpretable, trustworthy trees
        max_depths = [2, 3, 4, 5, 6]  # Shallow to medium depth for interpretability
        min_samples_leafs = [5, 10, 20]   # Prevent overfitting
        min_samples_splits = [10, 20, 40]  # Control tree growth
        
        nested_cv = list(nested_cv)
        best_score = -float('inf')
        best_config = None
        all_scores = {}
        
        # Get current params to use as base
        current_params = self.get_params()
        random_state = current_params.get('random_state', 42)
        
        # Grid search over hyperparameters
        for max_depth in max_depths:
            for min_samples_leaf in min_samples_leafs:
                for min_samples_split in min_samples_splits:
                    # Skip invalid combinations (min_samples_split must be >= 2 * min_samples_leaf)
                    if min_samples_split < 2 * min_samples_leaf:
                        continue
                    
                    config = {
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'random_state': random_state,
                    }
                    
                    scores = []
                    for split in nested_cv:
                        # Use binary data if available (when binarize_mode is enabled)
                        if hasattr(split, 'binarizer') and split.binarizer is not None:
                            binary_data = split.get_binary_data()
                            X_train, X_test = binary_data['X_train'], binary_data['X_test']
                        else:
                            X_train, X_test = split.X_train, split.X_test
                        
                        cart_model = DecisionTreeClassifier(**config)
                        cart_model.fit(X_train, split.y_train)
                        
                        score = cart_model.score(X_test, split.y_test)
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


@register_hparams('cart')
def update_hparams(hparams: Hparams, args: argparse.Namespace, _dataset: DatasetLoader=None):
    """ Update hparams with CART-specific parameters. """
    if args is None:
        return
    hparams.model_params = {
        'max_depth': int(args.max_depth) if hasattr(args, 'max_depth') else 4,
        'min_samples_split': int(args.min_samples_split) if hasattr(args, 'min_samples_split') else 10,
        'min_samples_leaf': int(args.min_samples_leaf) if hasattr(args, 'min_samples_leaf') else 5,
        'random_state': int(args.random_state) if hasattr(args, 'random_state') else 42,
    }
