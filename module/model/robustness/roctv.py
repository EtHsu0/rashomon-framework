"""roctv.py: Robust Optimal Classification Tree.

This module implements ROCT-V (Vos variant), which learns adversarially robust
optimal decision trees using a verification-based approach with MILP or MaxSAT
solvers to ensure certified robustness.
"""
from module.model.core.base import BaseEstimator, register_model
from module.hparams import Hparams, register_hparams
from module.datasets import DatasetLoader

from typing import Any, Optional
import numpy as np
try:
    from roct.milp import OptimalRobustTree
except ImportError as e:
    print("Error importing roct.milp. Make sure the roct package is installed.")


@register_model("roctv")
class RoctV(BaseEstimator):
    """Robust Optimal Classification Tree with verification (ROCT-V).
    
    Uses MILP or MaxSAT solvers to learn provably robust decision trees with
    certified guarantees against adversarial perturbations within epsilon.
    """
    params: dict
    model: Any

    def __init__(self, **params: Any):
        """Initialize ROCT-V model.
        
        Args:
            **params: Model hyperparameters including epsilon (robustness radius).
        """
        super().__init__(**params)
        self.params = params
        self.model = None
        self.epsilon = params.pop("epsilon", 0.1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RoctV':
        """
        Fit the ROCT-V model. Sets up attack_model and fits SATOptimalRobustTree.
        Args:
            X: Features
            y: Labels
        Returns:
            self
        """
        self.params["attack_model"] = [self.epsilon] * X.shape[1]
        self.model = OptimalRobustTree(**self.params)   
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for X.
        """
        return self.model.predict(X)
@register_hparams("roctv")
def update_hparams(hparams, args, dataset=None):
    hparams.model_params = {
        "max_depth": int(args.max_depth) if hasattr(args, "max_depth") else 4,
        "epsilon": float(args.epsilon) if hasattr(args, "epsilon") else 0.1,
        "time_limit": int(args.time_limit) if hasattr(args, "time_limit") else 1800,
    }