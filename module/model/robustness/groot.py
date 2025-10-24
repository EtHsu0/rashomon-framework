

import logging
from typing import Any, Optional
import numpy as np
from module.model.core.base import BaseEstimator, register_model
from module.hparams import Hparams, register_hparams
from module.datasets import DatasetLoader
try:
    from groot.model import GrootTreeClassifier
except:
    print("Error importing groot.model. Make sure the groot package is installed.")

@register_model("groot")
class Groot(BaseEstimator):
    """
    Wrapper for GROOT robust decision tree model.
    """
    params: dict
    model: Any

    def __init__(self, **params: Any):
        """
        Initialize Groot model.
        Args:
            **params: Model hyperparameters.
        """
        super().__init__(**params)
        self.params = params
        self.model = None # Initialized in fit method
        self.epsilon = params.pop("epsilon", 0.1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Groot':
        """
        Fit the GROOT model. Sets up attack_model and fits GrootTreeClassifier.
        Args:
            X: Features
            y: Labels
        Returns:
            self
        """
        self.params["attack_model"] = [self.epsilon] * X.shape[1]
        self.model = GrootTreeClassifier(**self.params)
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for X.
        """
        return self.model.predict(X)

    def tune(self, nested_cv):
        """
        Tune hyperparameters for GROOT using nested cross-validation.
        
        GROOT is a robust decision tree that considers adversarial perturbations.
        We tune max_depth, min_samples_leaf, and min_samples_split. 
        Epsilon is preserved from initialization (dataset-specific or default).
        
        Args:
            nested_cv: Iterable of Split objects for inner CV
        
        Returns:
            tuple: (best_config dict, all_scores dict)
        """
        logger = logging.getLogger("Groot.tune")
        
        # Hyperparameter grid - GROOT specific ranges
        max_depths = [2, 3, 4, 5, 6]  # Shallow to medium depth
        min_samples_leafs = [5, 10, 20]   # Prevent overfitting
        min_samples_splits = [10, 20, 40]  # Control tree growth
        # Use epsilon from model initialization (dataset-specific)
        epsilon = self.epsilon
        
        nested_cv = list(nested_cv)
        best_score = -float('inf')
        best_config = None
        all_scores = {}
        
        # Get current params to use as base
        random_state = self.params.get('random_state', 42)
        
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
                        
                        # Create attack model based on number of features
                        attack_model = [epsilon] * X_train.shape[1]
                        groot_config = config.copy()
                        groot_config['attack_model'] = attack_model
                        del groot_config['epsilon']
                        
                        groot_model = GrootTreeClassifier(**groot_config)
                        groot_model.fit(X_train, split.y_train)
                        
                        score = groot_model.score(X_test, split.y_test)
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

    
@register_hparams("groot")
def update_hparams(hparams, args, dataset):
    hparams.model_params = {
        "max_depth": args.max_depth if hasattr(args, "max_depth") else 4,
        "min_samples_split": args.min_samples_split if hasattr(args, "min_samples_split") else 10,
        "min_samples_leaf": args.min_samples_leaf if hasattr(args, "min_samples_leaf") else 5,
        "epsilon": float(args.epsilon) if hasattr(args, "epsilon") else 0.1,
        "random_state": hparams.rs,
    }