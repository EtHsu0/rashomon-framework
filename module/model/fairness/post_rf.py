"""post_rf.py: Post-processing fairness wrapper for Random Forest models.

This module implements fairness-aware post-processing on top of Random Forest models,
using the same approach as PostCart but with ensemble methods for improved accuracy
while maintaining fairness guarantees.
"""
import logging
from sklearn.ensemble import RandomForestClassifier
from module.model.lib.linear_post import LinearPostProcessor
import numpy as np
from module.model.core.post_base import PostBase
from module.model.core.base import register_model
from module.hparams import Hparams, register_hparams
from collections import defaultdict


@register_model("post_rf")
class PostRFWrapper(PostBase):
    """Post-processing fairness model using Random Forest.
    
    Trains three Random Forest models (for labels, sensitive attributes, and their
    joint distribution) and applies linear post-processing to satisfy fairness
    criteria while leveraging ensemble accuracy.
    """
    
    def __init__(self, **params):
        """Initialize PostRFWrapper with fairness parameters.
        
        Args:
            **params: Must include 'alpha' (fairness weight) and 'sweep' (whether to
                     sweep over alpha values). Remaining params passed to RandomForest.
        """
        super().__init__(**params)
        self.model = None
        self.alpha = params["alpha"]; del params["alpha"]
        self.sweep = params["sweep"]; del params["sweep"]
        self.model_y = RandomForestClassifier(**params)
        self.model_a = RandomForestClassifier(**params)
        self.model_ay = RandomForestClassifier(**params)
        self.pareto_models = defaultdict(list) # key: criterion, value: list of (alpha, model)
        self.models = {} # key: criterion, value: model

    def fit(self, X, y, sensitive):
        self.model_y.fit(X, y)
        self.model_a.fit(X, sensitive)
        self.model_ay.fit(X, sensitive * len(np.unique(y)) + y)

    def predict(self, X):
        return self.model_y.predict(X)

    def post_predict(self, X, criterion):
        return self.models[criterion].predict(X)

    # Consistent helper fns with post_cart
    def _pred_y(self, X):
        return self.model_y.predict_proba(X)

    def _pred_a(self, X):
        return self.model_a.predict_proba(X)


    def _pred_ay(self, X):
        return self.model_ay.predict_proba(X)

    def tune(self, nested_cv):
        """
        Tune hyperparameters for the base Random Forest classifier using nested cross-validation.
        
        Tunes n_estimators, max_depth, min_samples_leaf, and min_samples_split for all three 
        classifiers (model_y, model_a, model_ay) using grid search.
        
        Args:
            nested_cv: Iterable of Split objects for inner CV
        
        Returns:
            tuple: (best_config dict, all_scores dict)
        """
        logger = logging.getLogger("PostRF.tune")
        
        # Hyperparameter grid for Random Forest
        n_estimators_list = [50, 100, 200]
        max_depths = [3, 4, 5, 6]
        min_samples_leafs = [1, 5, 10]
        min_samples_splits = [2, 10, 20]
        
        nested_cv = list(nested_cv)
        best_score = -float('inf')
        best_config = None
        all_scores = {}
        
        # Get random_state from current params
        current_params = self.model_y.get_params()
        random_state = current_params.get('random_state', 42)
        
        # Grid search over hyperparameters
        for n_estimators in n_estimators_list:
            for max_depth in max_depths:
                for min_samples_leaf in min_samples_leafs:
                    for min_samples_split in min_samples_splits:
                        # Skip invalid combinations
                        if min_samples_split < 2 * min_samples_leaf:
                            continue
                        
                        config = {
                            'n_estimators': n_estimators,
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split,
                            'min_samples_leaf': min_samples_leaf,
                            'random_state': random_state,
                        }
                        
                        scores = []
                        for split in nested_cv:
                            # Create temporary classifier with this config
                            temp_model_y = RandomForestClassifier(**config)
                            temp_model_y.fit(split.X_train, split.y_train)
                            
                            score = temp_model_y.score(split.X_test, split.y_test)
                            scores.append(score)
                        
                        avg_score = np.mean(scores)
                        config_key = (n_estimators, max_depth, min_samples_leaf, min_samples_split)
                        all_scores[config_key] = scores
                        
                        logger.debug(f"Config {config_key}: Avg Score = {avg_score:.4f}")
                        
                        if avg_score > best_score:
                            best_score = avg_score
                            best_config = config
        
        logger.info(f"Best Config: {best_config} with score: {best_score:.4f}")
        logger.info(f"Evaluated {len(all_scores)} configurations")
        return best_config, all_scores

    def post_process(self, X, y, sensitive, criterion=['sp', 'eo', 'eopp']):
        """Post-process the model using a linear post-processor."""
        assert X is not None and y is not None and sensitive is not None, "Post-processing requires X, y, and sensitive attributes."
        predict_y = self._pred_y
        predict_a = self._pred_a
        predict_ay = self._pred_ay

        for crit in criterion:
            if self.sweep:
                alphas = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 1000]
                for alpha in alphas:
                    model = LinearPostProcessor(
                        len(np.unique(y)),
                        len(np.unique(sensitive)),
                        pred_y_fn=predict_y,
                        pred_a_fn=predict_a,
                        pred_ay_fn=predict_ay,
                        criterion=crit,
                        alpha=alpha,
                    )
                    model.fit(X, solver=None)
                    self.pareto_models[crit].append((alpha, model))
                    if self.alpha == alpha:
                        self.model = model
                if self.alpha not in alphas:
                    ref_model = LinearPostProcessor(
                        len(np.unique(y)),
                        len(np.unique(sensitive)),
                        pred_y_fn=predict_y,
                        pred_a_fn=predict_a,
                        pred_ay_fn=predict_ay,
                        criterion=crit,
                        alpha=self.alpha,
                    )
                    ref_model.fit(X, solver=None)
                    self.pareto_models[crit].append((self.alpha, ref_model))
                    self.models[crit] = ref_model
            else:
                self.models[crit] = LinearPostProcessor(
                    len(np.unique(y)),
                    len(np.unique(sensitive)),
                    pred_y_fn=predict_y,
                    pred_a_fn=predict_a,
                    criterion=crit,
                    alpha=self.alpha,
                )
                self.models[crit].fit(X, solver=None)
        return


@register_hparams('post_rf')
def update_hparams(hparams: Hparams, args, _dataset=None):
    """ Update hparams with PostRF-specific parameters. """
    if args is None:
        return
    hparams.model_params = {
        "n_estimators": int(args.n_estimators) if hasattr(args, 'n_estimators') else 100,
        "max_depth": int(args.max_depth) if hasattr(args, 'max_depth') else 4,
        "min_samples_split": int(args.min_samples_split) if hasattr(args, 'min_samples_split') else 10,
        "min_samples_leaf": int(args.min_samples_leaf) if hasattr(args, 'min_samples_leaf') else 5,
        "alpha": float(args.alpha) if hasattr(args, 'alpha') else 0.03,
        "sweep": getattr(args, 'sweep', False),  # added for consistency with post_cart/xgboost
    }
